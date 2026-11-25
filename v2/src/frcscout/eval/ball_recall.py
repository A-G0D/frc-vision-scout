"""Fuel-ball recall harness — honest, per-regime ground truth for the ball detector.

A single blended "ball recall" number is a LIE on this footage: a detector can ace the
dense centre pile and miss every airborne ball, or vice versa, and report the same scalar.
So this harness NEVER prints one blended number. It reports recall *per regime*:

    pile      — balls in a dense static pile (heavy occlusion; many balls per blob)
    spread    — balls resting/rolling on the open carpet (the easy case)
    airborne  — balls in flight (few, fast, above the carpet — the parallax case)

Two truth sources, both honest about what they can and cannot establish:

  (A) CONSERVATION CHECK (free, dense, no hand-labeling). The broadcast overlay carries a
      running scored-fuel counter (OverlayState.blue_fuel / red_fuel, read by
      scripts/hud_read.py). Fuel is conserved: a ball that leaves the floor inventory by
      being scored increments the counter. Over a window the change in scored fuel bounds
      how much floor inventory should have drained. We don't claim per-ball recall from
      this — we report whether the detector's floor count is *consistent* with the scored
      delta (i.e. the detector isn't hallucinating balls that were never on the floor, and
      isn't blind to a floor that is clearly draining). Returned as a {value, available,
      needs} envelope so a missing overlay degrades to "not available", never a fabricated
      pass.

  (B) HAND-LABELED CENTROIDS (precise, sparse). A tiny JSON of human-clicked ball centres,
      each tagged with its regime. We match detections to GT centroids (a detection covers
      a GT point if the point falls inside the detection box, or within a px tolerance of
      its centre) and compute recall *within each regime*. A 4-6 point sample is enough to
      demonstrate the table; add more points to tighten the estimate.

The harness imports nothing heavy: pure python + (optionally) the Detection dataclass. cv2
is NOT required.

GT JSON schema (see :func:`load_centroid_gt`)::

    {
      "video": "qual59.mp4",
      "points": [
        {"frame": 1000, "x": 980.0, "y": 470.0, "regime": "pile"},
        {"frame": 1000, "x": 760.0, "y": 590.0, "regime": "spread"},
        {"frame": 1000, "x": 610.0, "y": 300.0, "regime": "airborne"}
      ]
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

REGIMES = ("pile", "spread", "airborne")


# ---------------------------------------------------------------------------
# Honest metric envelope (mirrors the {value, available, needs} convention used
# across the eval layer for metrics that can't always be derived).
# ---------------------------------------------------------------------------

def envelope(value: Any, available: bool, needs: str = "") -> dict:
    """A gateable metric that is honest about whether it could be computed.

    available=False means *do not trust* `value`; `needs` says what was missing.
    """
    return {"value": value, "available": bool(available), "needs": needs}


# ---------------------------------------------------------------------------
# Detection access (works on Detection dataclass OR plain dicts/tuples)
# ---------------------------------------------------------------------------

def _box_of(det: Any) -> Optional[tuple[float, float, float, float]]:
    xyxy = det.get("xyxy") if isinstance(det, dict) else getattr(det, "xyxy", None)
    if xyxy is None:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in xyxy)
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _covers(box: tuple[float, float, float, float], px: float, py: float,
            tol_px: float) -> bool:
    """True if GT point (px,py) is inside the box, or within tol_px of its centre."""
    x1, y1, x2, y2 = box
    if x1 - tol_px <= px <= x2 + tol_px and y1 - tol_px <= py <= y2 + tol_px:
        return True
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5 <= tol_px


# ---------------------------------------------------------------------------
# (B) Hand-labeled centroid GT  ->  per-regime recall
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GtPoint:
    frame: int
    x: float
    y: float
    regime: str


def load_centroid_gt(path: str | Path) -> list[GtPoint]:
    """Load hand-clicked ball centroids from a JSON file (schema in the module docstring).

    Unknown regimes are coerced to "spread" with no error (so a typo never silently drops a
    point from every bucket). Missing file -> empty list (the caller reports "no GT").
    """
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    pts: list[GtPoint] = []
    for d in data.get("points", []):
        regime = str(d.get("regime", "spread")).strip().lower()
        if regime not in REGIMES:
            regime = "spread"
        pts.append(GtPoint(
            frame=int(d.get("frame", 0)),
            x=float(d["x"]),
            y=float(d["y"]),
            regime=regime,
        ))
    return pts


@dataclass
class RegimeRecall:
    regime: str
    n_gt: int = 0
    n_hit: int = 0

    @property
    def recall(self) -> Optional[float]:
        """Hit fraction, or None when there is no GT in this regime (NOT 0.0 — absence of
        truth is not a failure; reporting 0 would be a lie)."""
        return (self.n_hit / self.n_gt) if self.n_gt else None


def per_regime_recall(
    detections_by_frame: dict[int, Sequence[Any]],
    gt: Iterable[GtPoint],
    tol_px: float = 40.0,
) -> dict[str, RegimeRecall]:
    """Match GT centroids to detections per frame; tally hits per regime.

    A GT point is HIT if any detection on its frame covers it (inside the box or within
    tol_px of the box centre). ``tol_px`` defaults to ~40 px, about 1.5 fuel-ball diameters
    at this broadcast's camera distance — it absorbs the human centroid-click error and the
    way a dense-pile blob's centroid lands on a neighbouring ball; tighten it for sparse,
    well-isolated GT. Returns a dict regime -> RegimeRecall for ALL regimes (n_gt=0 buckets
    included, so the printed table always shows every regime).
    """
    out = {r: RegimeRecall(regime=r) for r in REGIMES}
    for g in gt:
        bucket = out.setdefault(g.regime, RegimeRecall(regime=g.regime))
        bucket.n_gt += 1
        dets = detections_by_frame.get(g.frame, [])
        hit = False
        for det in dets:
            box = _box_of(det)
            if box is not None and _covers(box, g.x, g.y, tol_px):
                hit = True
                break
        if hit:
            bucket.n_hit += 1
    return out


def format_recall_table(buckets: dict[str, RegimeRecall]) -> str:
    """Render the per-regime recall table. NEVER collapses to one blended number."""
    lines = ["per-regime ball recall (hand-labeled centroids)",
             "  regime    n_gt  n_hit  recall",
             "  --------  ----  -----  ------"]
    for r in REGIMES:
        b = buckets.get(r, RegimeRecall(regime=r))
        rec = "  n/a " if b.recall is None else f"{b.recall:6.1%}"
        lines.append(f"  {r:<8}  {b.n_gt:>4}  {b.n_hit:>5}  {rec}")
    # Any extra (non-standard) regimes a GT file introduced.
    for r, b in buckets.items():
        if r not in REGIMES:
            rec = "  n/a " if b.recall is None else f"{b.recall:6.1%}"
            lines.append(f"  {r:<8}  {b.n_gt:>4}  {b.n_hit:>5}  {rec}  (custom)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# (A) Conservation check against the overlay scored-fuel counter
# ---------------------------------------------------------------------------

def _scored_total(overlay: Any) -> Optional[int]:
    """Sum blue_fuel + red_fuel from an OverlayState (or dict). None if neither present."""
    blue = overlay.get("blue_fuel") if isinstance(overlay, dict) else getattr(overlay, "blue_fuel", None)
    red = overlay.get("red_fuel") if isinstance(overlay, dict) else getattr(overlay, "red_fuel", None)
    if blue is None and red is None:
        return None
    return int(blue or 0) + int(red or 0)


def conservation_check(
    floor_counts: Sequence[int],
    overlays: Sequence[Any],
    tolerance: float = 0.5,
) -> dict:
    """Are the detector's per-frame FLOOR ball counts consistent with scored-fuel flow?

    Physical invariant over a window: fuel is conserved. When the overlay scored counter
    rises by `scored_delta`, that many balls left the floor inventory (were shot/scored).
    So the floor count should not be *rising* while a lot is being scored, and a draining
    floor should be matched by a rising counter.

    We don't claim per-ball recall here (the floor is replenished by human/robot intake we
    can't see) — we report two cheap CONSISTENCY signals:
        * mean_floor      : average detected floor balls over the window (sanity magnitude).
        * sign_consistent : fraction of windows where the floor-count change and the scored
                            change don't BOTH point "up" by a lot (a floor that grows while
                            scoring surges is physically suspect -> likely FP inflation).
    Returned as an envelope: available=False (with `needs`) when the overlay carries no
    fuel counter, so a missing HUD degrades honestly instead of fabricating a pass.
    """
    n = min(len(floor_counts), len(overlays))
    scored = [_scored_total(overlays[i]) for i in range(n)]
    have = [s for s in scored if s is not None]
    if len(have) < 2:
        return envelope(
            {"mean_floor": (sum(floor_counts[:n]) / n) if n else None,
             "sign_consistent": None},
            available=False,
            needs="overlay scored-fuel counter (blue_fuel/red_fuel) on >=2 frames "
                  "in the window — read via scripts/hud_read.py",
        )

    mean_floor = sum(floor_counts[:n]) / n if n else 0.0

    # Compare consecutive frames where BOTH a floor count and a scored total exist.
    pairs = 0
    consistent = 0
    prev_floor = floor_counts[0]
    prev_scored = scored[0]
    for i in range(1, n):
        s = scored[i]
        if s is None or prev_scored is None:
            prev_floor, prev_scored = floor_counts[i], (s if s is not None else prev_scored)
            continue
        d_floor = floor_counts[i] - prev_floor
        d_scored = s - prev_scored
        pairs += 1
        # Suspect ONLY when scoring surged AND the floor also surged up (FP inflation):
        # a ball can't be both newly-scored and newly-on-the-floor. A draining floor
        # (d_floor <= 0) is always consistent; a flat floor is consistent.
        surge = max(1.0, tolerance * max(prev_floor, 1))
        suspect = (d_scored > 0) and (d_floor > surge)
        if not suspect:
            consistent += 1
        prev_floor, prev_scored = floor_counts[i], s

    sign_consistent = (consistent / pairs) if pairs else None
    return envelope(
        {"mean_floor": mean_floor, "sign_consistent": sign_consistent, "n_pairs": pairs},
        available=sign_consistent is not None,
        needs="" if sign_consistent is not None
              else "consecutive frames with both a floor count and an overlay scored total",
    )


# ---------------------------------------------------------------------------
# One-call report (prints the table + the conservation envelope)
# ---------------------------------------------------------------------------

@dataclass
class RecallReport:
    regime: dict[str, RegimeRecall] = field(default_factory=dict)
    conservation: dict = field(default_factory=dict)

    def text(self) -> str:
        parts = [format_recall_table(self.regime), "", "conservation check (overlay fuel)"]
        c = self.conservation
        if not c.get("available"):
            parts.append(f"  NOT AVAILABLE — needs: {c.get('needs', 'overlay fuel counter')}")
        else:
            v = c["value"]
            sc = v.get("sign_consistent")
            parts.append(f"  mean floor balls : {v.get('mean_floor'):.1f}"
                         if v.get("mean_floor") is not None else "  mean floor balls : n/a")
            parts.append(f"  flow-consistent  : {sc:6.1%} of {v.get('n_pairs')} frame-pairs"
                         if sc is not None else "  flow-consistent  : n/a")
        return "\n".join(parts)


def evaluate(
    detections_by_frame: dict[int, Sequence[Any]],
    gt_path: Optional[str | Path] = None,
    floor_counts: Optional[Sequence[int]] = None,
    overlays: Optional[Sequence[Any]] = None,
    tol_px: float = 40.0,
) -> RecallReport:
    """End-to-end: load GT (if any), compute per-regime recall + the conservation envelope.

    Pass `floor_counts`+`overlays` (same length, frame-aligned) to populate the
    conservation check; omit them and it degrades to a not-available envelope.
    """
    gt = load_centroid_gt(gt_path) if gt_path else []
    regime = per_regime_recall(detections_by_frame, gt, tol_px=tol_px)
    if floor_counts is not None and overlays is not None:
        cons = conservation_check(floor_counts, overlays)
    else:
        cons = envelope({"mean_floor": None, "sign_consistent": None},
                        available=False,
                        needs="floor_counts + overlays (per-frame detected ball count and "
                              "OverlayState with fuel) to run the conservation check")
    return RecallReport(regime=regime, conservation=cons)
