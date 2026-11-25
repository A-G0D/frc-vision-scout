"""Weighted Box Fusion (WBF) of multiple detection sources.

The refine layer takes the *same frame* seen through several detectors — the NN,
the classical flight/blob detector, the SAHI-style tiled pass, and any TTA
(flip/scale) variants — and merges their boxes into one consensus set. Unlike
NMS, which simply *discards* all but the highest-confidence box in a cluster,
Weighted Box Fusion *blends* the overlapping boxes' coordinates weighted by their
confidence, so two sources that each clip a different edge of the same ball pull
the fused box toward the true extent. This is the technique from Solovyev,
Wang & Gribov, "Weighted boxes fusion" (Image and Vision Computing, 2021).

Why this matters for the 60fps scout: the cheap 60Hz robot NN and the high-res
5Hz tiled ball pass disagree at the margins; a robot bumper detected by both the
NN and a flipped-TTA pass should fuse to a tighter, higher-confidence box rather
than leaving two near-duplicates for the tracker to fight over.

Contract
--------
``fuse(list_of_detlists, iou_thr=0.5)`` -> ``list[dict]`` with keys
``{"xyxy", "conf", "alliance"}``:
    * ``xyxy``      — ``(x1, y1, x2, y2)`` float tuple, confidence-weighted blend
    * ``conf``      — fused confidence (mean of cluster, down-weighted when only a
                      minority of sources contributed — standard WBF behaviour)
    * ``alliance``  — majority vote over the cluster's alliances ("red"/"blue"/None)

Input detections may be plain dicts (``{"xyxy":..., "conf":..., "alliance":...}``)
or any object exposing ``.xyxy`` / ``.conf`` (e.g. the frozen ``Detection``
dataclass, which has no alliance — treated as ``None``). Mixed types are fine.

Class-agnostic
--------------
Fusion is class-agnostic: clustering is by box overlap only, not by ``cls``.
That is deliberate — the sources here detect compatible object kinds and we want
spatial consensus. A class-agnostic NMS fallback is provided as :func:`nms` for
callers that want pure suppression instead of blending; ``fuse`` itself already
degrades to "keep the box" for any detection with no overlapping partner.

Pure numpy. The module is import-guarded so it *parses* even when numpy is
absent; the fusion math itself needs numpy at call time.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

try:  # guarded so the file parses in a bare env (pure-logic still unit-testable)
    import numpy as np
except Exception:  # pragma: no cover - exercised only when numpy is missing
    np = None  # type: ignore[assignment]


# Public, JSON-friendly detection record.
FusedDet = dict  # {"xyxy": (x1,y1,x2,y2), "conf": float, "alliance": Optional[str]}

_VALID_ALLIANCES = ("red", "blue")


# ---------------------------------------------------------------------------
# Normalisation: accept dicts OR objects with .xyxy/.conf/.alliance
# ---------------------------------------------------------------------------

def _get(d: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict or as an attribute of an object."""
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


def _norm_alliance(value: Any) -> Optional[str]:
    """Coerce an alliance value to "red"/"blue"/None. Unknown -> None."""
    if value is None:
        return None
    s = str(value).strip().lower()
    return s if s in _VALID_ALLIANCES else None


def _normalize(det: Any) -> Optional[dict]:
    """Turn one input detection into ``{"xyxy", "conf", "alliance"}`` or None if invalid."""
    xyxy = _get(det, "xyxy")
    if xyxy is None:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in xyxy)
    except (TypeError, ValueError):
        return None
    # Canonicalise corner order so a flipped-TTA box [x2,x1] still overlaps correctly.
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    conf = _get(det, "conf", 0.0)
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.0
    return {
        "xyxy": (x1, y1, x2, y2),
        "conf": conf,
        "alliance": _norm_alliance(_get(det, "alliance")),
    }


# ---------------------------------------------------------------------------
# IoU — pure numpy (mirrors detect/ball_net._iou so behaviour matches the NMS
# the tiled pass already uses, but lives here independently to avoid a cross-
# layer import).
# ---------------------------------------------------------------------------

def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-Union of two ``[x1,y1,x2,y2]`` boxes. Degenerate boxes -> 0."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Class-agnostic NMS fallback (greedy, highest-conf-wins)
# ---------------------------------------------------------------------------

def nms(detlists: Sequence[Sequence[Any]] | Sequence[Any], iou_thr: float = 0.5) -> list[FusedDet]:
    """Class-agnostic greedy NMS over one or many detection lists.

    Accepts either a flat list of detections or a list-of-lists (same shape as
    :func:`fuse`); the lists are concatenated and treated as one pool. Returns the
    surviving boxes unchanged (no coordinate blending) as ``{"xyxy","conf","alliance"}``.

    This is the *fallback* the contract asks for: callers that want pure
    suppression rather than fusion can call this directly.
    """
    pool = _flatten_and_normalize(detlists)
    if not pool:
        return []
    pool.sort(key=lambda d: d["conf"], reverse=True)
    kept: list[FusedDet] = []
    while pool:
        best = pool.pop(0)
        kept.append(best)
        pool = [d for d in pool if iou_xyxy(best["xyxy"], d["xyxy"]) < iou_thr]
    return kept


# ---------------------------------------------------------------------------
# Helpers shared by fuse() and nms()
# ---------------------------------------------------------------------------

def _looks_like_detection(obj: Any) -> bool:
    """True if obj is a single detection (dict with xyxy, or object with .xyxy)."""
    if isinstance(obj, dict):
        return "xyxy" in obj
    return hasattr(obj, "xyxy")


def _flatten_and_normalize(detlists: Any) -> list[dict]:
    """Accept a flat list of detections OR a list of detection-lists; return a flat,
    normalised, valid pool. Empty / None sources are skipped gracefully."""
    if not detlists:
        return []
    pool: list[dict] = []
    # Distinguish "flat list of detections" from "list of lists of detections".
    first = next((x for x in detlists if x is not None), None)
    is_flat = _looks_like_detection(first) if first is not None else False
    sources: Sequence[Any] = [detlists] if is_flat else detlists
    for src in sources:
        if not src:
            continue
        for det in src:
            nd = _normalize(det)
            if nd is not None:
                pool.append(nd)
    return pool


def _majority_alliance(alliances: Sequence[Optional[str]],
                       weights: Sequence[float]) -> Optional[str]:
    """Confidence-weighted majority vote over a cluster's alliances.

    ``None`` votes (e.g. ball detections) do not count toward either alliance.
    Ties between red and blue resolve to None (genuinely ambiguous — let the
    tracker's bumper-colour history decide downstream rather than guessing here).
    If every vote is None, the result is None.
    """
    tally: dict[str, float] = {"red": 0.0, "blue": 0.0}
    for al, w in zip(alliances, weights):
        if al in tally:
            tally[al] += max(0.0, float(w))
    red, blue = tally["red"], tally["blue"]
    if red == 0.0 and blue == 0.0:
        return None
    if red > blue:
        return "red"
    if blue > red:
        return "blue"
    return None  # exact tie -> ambiguous


# ---------------------------------------------------------------------------
# Weighted Box Fusion
# ---------------------------------------------------------------------------

def fuse(
    detlists: Sequence[Sequence[Any]] | Sequence[Any],
    iou_thr: float = 0.5,
    conf_thr: float = 0.0,
    weights: Optional[Sequence[float]] = None,
) -> list[FusedDet]:
    """Weighted Box Fusion across multiple detection sources.

    Args:
        detlists:  list of per-source detection lists, e.g.
                   ``[nn_dets, classical_dets, tiled_dets, tta_dets]``. Each
                   detection is a dict ``{"xyxy", "conf", "alliance"?}`` or an
                   object exposing ``.xyxy`` / ``.conf`` (alliance optional).
                   A single flat list of detections is also accepted.
        iou_thr:   boxes whose IoU with a cluster's *running fused box* is
                   ``>= iou_thr`` join that cluster; otherwise they seed a new one.
        conf_thr:  drop input detections below this confidence before fusing.
        weights:   optional per-source multipliers (len == number of sources).
                   A source weight scales its detections' contribution to both
                   the coordinate blend and the fused confidence. Defaults to all 1.

    Returns:
        list of fused ``{"xyxy", "conf", "alliance"}`` dicts, sorted by
        descending fused confidence.

    Algorithm (Solovyev et al. 2021):
        1. Normalise every detection and tag it with its source index.
        2. Sort all detections by (weighted) confidence, descending.
        3. Greedily assign each detection to the first existing cluster whose
           running fused box has IoU >= iou_thr with it; else start a new cluster.
        4. The cluster's fused box is the confidence-weighted average of its
           members' corners, recomputed as members are added.
        5. Fused confidence = mean member confidence, scaled by
           ``min(distinct_sources_in_cluster, num_sources) / num_sources`` so a
           box only one source saw is down-weighted relative to a consensus box.
        6. Alliance = confidence-weighted majority vote over the cluster.

    Disjoint boxes survive untouched (a one-member cluster fuses to itself, modulo
    the source down-weighting). Overlapping boxes from different sources blend.
    """
    if np is None:  # pragma: no cover
        raise RuntimeError("detect_fuse.fuse requires numpy")

    pool = _flatten_and_normalize(detlists)

    # Determine the source count for the WBF confidence normalisation. When a flat
    # list was passed there is exactly one logical source.
    num_sources = _count_sources(detlists)

    # Tag detections with their originating source index (for distinct-source counts
    # and per-source weights). Recompute the flat/nested decision once, consistently.
    tagged = _tag_with_source(detlists, conf_thr=conf_thr)
    if not tagged:
        return []

    if weights is not None:
        w = [float(x) for x in weights]
    else:
        w = [1.0] * num_sources

    def src_weight(si: int) -> float:
        return w[si] if 0 <= si < len(w) else 1.0

    # Sort by weighted confidence, highest first.
    tagged.sort(key=lambda t: t["conf"] * src_weight(t["src"]), reverse=True)

    clusters: list[_Cluster] = []
    for det in tagged:
        box = det["xyxy"]
        placed = False
        best_i = -1
        best_iou = iou_thr
        for i, cl in enumerate(clusters):
            ov = iou_xyxy(cl.fused_box, box)
            if ov >= best_iou:
                best_iou = ov
                best_i = i
        if best_i >= 0:
            clusters[best_i].add(det, src_weight(det["src"]))
            placed = True
        if not placed:
            clusters.append(_Cluster.seed(det, src_weight(det["src"])))

    out: list[FusedDet] = []
    for cl in clusters:
        out.append(cl.finalize(num_sources))
    out.sort(key=lambda d: d["conf"], reverse=True)
    return out


def _count_sources(detlists: Any) -> int:
    if not detlists:
        return 1
    first = next((x for x in detlists if x is not None), None)
    if first is not None and _looks_like_detection(first):
        return 1  # flat list -> single source
    return max(1, len(detlists))


def _tag_with_source(detlists: Any, conf_thr: float) -> list[dict]:
    """Flatten to normalised detections each tagged with ``src`` (source index)."""
    if not detlists:
        return []
    first = next((x for x in detlists if x is not None), None)
    is_flat = _looks_like_detection(first) if first is not None else False
    sources: Sequence[Any] = [detlists] if is_flat else detlists
    tagged: list[dict] = []
    for si, src in enumerate(sources):
        if not src:
            continue
        for det in src:
            nd = _normalize(det)
            if nd is None or nd["conf"] < conf_thr:
                continue
            nd["src"] = si
            tagged.append(nd)
    return tagged


# ---------------------------------------------------------------------------
# Cluster accumulator
# ---------------------------------------------------------------------------

class _Cluster:
    """Accumulates the members of one fused box and maintains the running blend."""

    __slots__ = ("boxes", "confs", "alliances", "src_weights", "srcs", "fused_box")

    def __init__(self) -> None:
        self.boxes: list[tuple[float, float, float, float]] = []
        self.confs: list[float] = []
        self.alliances: list[Optional[str]] = []
        self.src_weights: list[float] = []
        self.srcs: set[int] = set()
        self.fused_box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @classmethod
    def seed(cls, det: dict, src_w: float) -> "_Cluster":
        c = cls()
        c.add(det, src_w)
        return c

    def add(self, det: dict, src_w: float) -> None:
        self.boxes.append(det["xyxy"])
        self.confs.append(det["conf"])
        self.alliances.append(det["alliance"])
        self.src_weights.append(src_w)
        self.srcs.add(det["src"])
        self._recompute()

    def _recompute(self) -> None:
        # Weight each member by conf * source_weight for the coordinate blend.
        boxes = np.asarray(self.boxes, dtype=float)          # (n, 4)
        confs = np.asarray(self.confs, dtype=float)          # (n,)
        sw = np.asarray(self.src_weights, dtype=float)       # (n,)
        wts = np.clip(confs * sw, 0.0, None)
        if wts.sum() <= 0.0:
            # All-zero confidence: fall back to an unweighted mean so we still
            # produce a sane box rather than dividing by zero.
            wts = np.ones_like(wts)
        fused = (boxes * wts[:, None]).sum(axis=0) / wts.sum()
        self.fused_box = (float(fused[0]), float(fused[1]),
                          float(fused[2]), float(fused[3]))

    def finalize(self, num_sources: int) -> FusedDet:
        confs = np.asarray(self.confs, dtype=float)
        # Mean member confidence, down-weighted by the fraction of sources that
        # actually contributed (standard WBF). A box seen by every source keeps
        # its mean conf; a box seen by one of N sources is scaled by 1/N.
        distinct = min(len(self.srcs), num_sources)
        scale = distinct / num_sources if num_sources > 0 else 1.0
        fused_conf = float(confs.mean() * scale)
        # Confidence-weighted alliance vote (source weight folded in).
        vote_w = [c * sw for c, sw in zip(self.confs, self.src_weights)]
        alliance = _majority_alliance(self.alliances, vote_w)
        return {
            "xyxy": self.fused_box,
            "conf": fused_conf,
            "alliance": alliance,
        }
