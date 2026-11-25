"""Test-time augmentation (TTA) for the detectors.

Given an injected ``detect(frame) -> list[Detection]`` callable, we run it on a
set of cheap, *invertible* views of the same image — the original, a horizontal
flip, and one or more rescaled copies — then map every box back into ORIGINAL
image-pixel coordinates and fuse the union into a single detection list.

Why this layer exists
----------------------
A single forward pass at one scale misses balls that are tiny (helped by an
up-scale) or partly occluded from one side (helped by the mirror). TTA trades a
few extra detector calls for recall, off the 60Hz hot path (it is only worth
running on the LOW-cadence iGPU ball net, never per-frame). All the geometry
here is pure logic + numpy: the detector is injected, so the inversion math is
unit-testable with a mock that returns hand-placed boxes.

Design contract
---------------
* The detector callable takes a single ``frame`` (an HxWxC ``np.ndarray``) and
  returns ``list[Detection]`` whose ``xyxy`` is in the pixel coordinates *of the
  frame it was handed* (NOT the original). Each view records the exact transform
  used to build its frame, and we invert that transform on the returned boxes —
  so a flip inverts to the pixel for a flip, and a rescale inverts by the exact
  resized-vs-original size ratio (no accumulated rounding error).
* Frame resizing uses cv2 when present, but the *inversion* never depends on
  cv2: we invert by the realized ``(out_w, out_h)`` of the resized frame, so the
  mock detector path (which doesn't need cv2 at all) inverts exactly.
* Fusion is class-aware greedy NMS over the union, keeping the highest-conf box
  in each cluster. This module owns its own tiny IoU/NMS so it has no hard
  dependency on the detect layer.

Everything parses without cv2/numpy installed; only the functions that actually
build resized frames touch those libraries, and they are import-guarded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from frcscout.io.types import Detection

# --- guarded heavy imports -------------------------------------------------
# The module must PARSE (and its pure-logic tests must run) even if numpy/cv2
# are absent. We import lazily/guarded and fall back to a pure-python IoU so the
# inversion + fusion logic stays testable on a bare interpreter.
try:  # pragma: no cover - exercised both ways across environments
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None  # type: ignore

try:  # pragma: no cover
    import cv2 as _cv2  # type: ignore
except Exception:  # pragma: no cover
    _cv2 = None  # type: ignore


Frame = "Any"  # an HxWxC ndarray at runtime; kept loose so we don't need numpy to parse
DetectFn = Callable[[object], List[Detection]]
XYXY = Tuple[float, float, float, float]


# ===========================================================================
# Geometry: a single invertible view of the source frame
# ===========================================================================

@dataclass(frozen=True)
class _View:
    """One augmented view of the source image.

    Attributes
    ----------
    name:
        Human-readable tag (``"orig"``, ``"hflip"``, ``"scale1.50"`` …), copied
        into each fused Detection's debugging info is intentionally avoided —
        Detection is frozen with a fixed schema — but the name is used in tests.
    flip:
        Whether this view horizontally mirrors the source.
    out_w, out_h:
        The realized pixel size of the view's frame. Box inversion divides by
        ``out_w / src_w`` (x) and ``out_h / src_h`` (y), so inversion is exact
        with respect to the size the detector actually saw, regardless of how the
        resize rounded.
    src_w, src_h:
        Original frame size, the target coordinate system for inverted boxes.
    """
    name: str
    flip: bool
    out_w: int
    out_h: int
    src_w: int
    src_h: int

    # -- forward: build the frame the detector will see --------------------
    def make_frame(self, frame):
        """Return the transformed frame for this view.

        Identity view returns the input untouched. Flip/scale use cv2/numpy and
        are import-guarded; callers that only test the math never reach here.
        """
        out = frame
        if self.out_w != self.src_w or self.out_h != self.src_h:
            if _cv2 is not None:
                out = _cv2.resize(frame, (self.out_w, self.out_h))
            elif _np is not None:
                out = _resize_nn_numpy(frame, self.out_w, self.out_h)
            else:  # pragma: no cover - no array backend at all
                raise RuntimeError("resizing a view requires cv2 or numpy")
        if self.flip:
            if _np is not None:
                out = out[:, ::-1]  # mirror columns; view, copy not required
            elif _cv2 is not None:  # pragma: no cover - numpy-less + cv2 present is rare
                out = _cv2.flip(out, 1)
            else:  # pragma: no cover
                raise RuntimeError("flipping a view requires numpy or cv2")
        return out

    # -- inverse: map a detector box back to ORIGINAL coordinates ----------
    def invert_box(self, xyxy: XYXY) -> XYXY:
        """Invert this view's transform on a single ``(x1,y1,x2,y2)`` box.

        Order matters: the forward transform is *resize then flip*, so the
        inverse is *un-flip then un-resize*. We un-flip in the VIEW's own pixel
        space (width ``out_w``) first, then scale back to source pixels.
        """
        x1, y1, x2, y2 = xyxy

        # 1) Undo the horizontal flip, in the view's own (resized) width.
        if self.flip:
            x1, x2 = (self.out_w - x2), (self.out_w - x1)

        # 2) Undo the resize. Scale by the exact realized ratio so a box the mock
        #    placed in resized coords lands back on the source pixel exactly.
        if self.out_w != self.src_w:
            sx = self.src_w / self.out_w
            x1 *= sx
            x2 *= sx
        if self.out_h != self.src_h:
            sy = self.src_h / self.out_h
            y1 *= sy
            y2 *= sy

        # Re-order so x1<=x2, y1<=y2 (flip already keeps them ordered, but guard).
        lo_x, hi_x = (x1, x2) if x1 <= x2 else (x2, x1)
        lo_y, hi_y = (y1, y2) if y1 <= y2 else (y2, y1)
        return (lo_x, lo_y, hi_x, hi_y)


def _resize_nn_numpy(frame, out_w: int, out_h: int):
    """Nearest-neighbour resize using only numpy (cv2 fallback).

    Good enough for a TTA view; the detector tolerates NN artifacts and the
    inversion math is independent of the interpolation anyway.
    """
    src_h, src_w = frame.shape[0], frame.shape[1]
    ys = (_np.arange(out_h) * (src_h / out_h)).astype(int).clip(0, src_h - 1)
    xs = (_np.arange(out_w) * (src_w / out_w)).astype(int).clip(0, src_w - 1)
    return frame[ys][:, xs]


# ===========================================================================
# Building the view set from a scale/flip plan
# ===========================================================================

def build_views(
    src_w: int,
    src_h: int,
    scales: Sequence[float] = (1.0,),
    flips: Sequence[bool] = (False, True),
    include_identity: bool = True,
) -> List[_View]:
    """Enumerate the (scale x flip) views for a frame of size ``src_w*src_h``.

    Args:
        src_w, src_h: original frame size in pixels.
        scales: multiplicative scale factors applied to BOTH axes. ``1.0`` is the
            native scale. Duplicate/near-duplicate scales are de-duplicated by the
            realized integer output size.
        flips: which mirror states to include (``False`` = upright, ``True`` =
            horizontal flip).
        include_identity: always ensure the plain upright native-scale view is
            present, even if ``scales``/``flips`` wouldn't have produced it. This
            is the anchor view and should essentially always run.

    Returns a de-duplicated list of :class:`_View`. The first element is the
    identity view when ``include_identity`` is True.
    """
    seen: set[Tuple[int, int, bool]] = set()
    views: List[_View] = []

    def _add(name: str, flip: bool, out_w: int, out_h: int) -> None:
        out_w = max(1, int(out_w))
        out_h = max(1, int(out_h))
        key = (out_w, out_h, flip)
        if key in seen:
            return
        seen.add(key)
        views.append(_View(name, flip, out_w, out_h, src_w, src_h))

    if include_identity:
        _add("orig", False, src_w, src_h)

    for s in scales:
        ow = round(src_w * s)
        oh = round(src_h * s)
        for fl in flips:
            tag = "hflip" if fl else "up"
            _add(f"{tag}_s{s:.2f}", fl, ow, oh)

    return views


# ===========================================================================
# IoU + class-aware greedy NMS (self-contained, pure logic)
# ===========================================================================

def _iou(a: XYXY, b: XYXY) -> float:
    """IoU of two ``(x1,y1,x2,y2)`` boxes. Pure python (no numpy needed)."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = x2 - x1
    ih = y2 - y1
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _greedy_nms(
    dets: List[Detection],
    iou_threshold: float,
    class_aware: bool,
) -> List[Detection]:
    """Greedy NMS: keep the highest-conf box, drop overlaps above threshold.

    Stable w.r.t. ties: sorts by ``(-conf, original_index)`` so equal-confidence
    boxes resolve deterministically by input order.
    """
    if not dets:
        return []
    order = sorted(range(len(dets)), key=lambda i: (-dets[i].conf, i))
    kept: List[Detection] = []
    suppressed = [False] * len(dets)
    for oi, i in enumerate(order):
        if suppressed[i]:
            continue
        best = dets[i]
        kept.append(best)
        for j in order[oi + 1:]:
            if suppressed[j]:
                continue
            other = dets[j]
            if class_aware and other.cls != best.cls:
                continue
            if _iou(best.xyxy, other.xyxy) >= iou_threshold:
                suppressed[j] = True
    return kept


# ===========================================================================
# Public API
# ===========================================================================

def tta_detect(
    frame,
    detect: DetectFn,
    scales: Sequence[float] = (1.0,),
    flips: Sequence[bool] = (False, True),
    include_identity: bool = True,
) -> List[Detection]:
    """Run ``detect`` over augmented views and return the UNION in source coords.

    No fusion/NMS is applied here — this is the raw augmented superset, so callers
    (and tests) can inspect every inverted box. Use :func:`detect_fuse` for the
    NMS-merged result.

    Args:
        frame: source image (HxWxC ndarray). Only ``frame.shape`` is read here;
            building each view's frame is delegated to :meth:`_View.make_frame`.
        detect: injected ``detect(frame) -> list[Detection]`` callable. It is
            called once per view with that view's transformed frame, and must
            return boxes in the coordinates of the frame it received.
        scales, flips, include_identity: forwarded to :func:`build_views`.

    Returns:
        Flat list of Detections, all in ORIGINAL frame coordinates.
    """
    src_h, src_w = _frame_hw(frame)
    views = build_views(src_w, src_h, scales, flips, include_identity)

    out: List[Detection] = []
    for v in views:
        view_frame = v.make_frame(frame)
        for d in detect(view_frame):
            out.append(
                Detection(
                    cls=d.cls,
                    xyxy=v.invert_box(d.xyxy),
                    conf=d.conf,
                    frame_idx=d.frame_idx,
                )
            )
    return out


def detect_fuse(
    frame,
    detect: DetectFn,
    scales: Sequence[float] = (1.0,),
    flips: Sequence[bool] = (False, True),
    include_identity: bool = True,
    iou_threshold: float = 0.55,
    class_aware: bool = True,
) -> List[Detection]:
    """TTA + fusion: the recall-boosted detection list a caller actually wants.

    Runs :func:`tta_detect` to get the inverted union across all views, then
    merges duplicates (the same object found in multiple views) with greedy NMS,
    keeping the highest-confidence box of each cluster.

    Args:
        frame, detect, scales, flips, include_identity: see :func:`tta_detect`.
        iou_threshold: boxes overlapping a kept box by >= this IoU are merged
            away. Slightly higher than single-pass NMS (0.45) because TTA copies
            of the same object land at nearly identical coords and we don't want
            to merge two genuinely adjacent balls.
        class_aware: only suppress boxes of the same ``cls`` (a ball and a robot
            that overlap are both kept).

    Returns:
        Fused Detections in ORIGINAL frame coordinates, highest-conf-first within
        each cluster (overall order is the NMS keep order).
    """
    union = tta_detect(frame, detect, scales, flips, include_identity)
    return _greedy_nms(union, iou_threshold, class_aware)


# ===========================================================================
# Small helpers
# ===========================================================================

def _frame_hw(frame) -> Tuple[int, int]:
    """Return (height, width) from an array-like frame, tolerating mocks.

    Accepts anything with a ``.shape`` (ndarray) or a plain ``(h, w[, c])`` tuple
    so tests can pass a lightweight stand-in instead of a real image.
    """
    shape = getattr(frame, "shape", None)
    if shape is None:
        if isinstance(frame, (tuple, list)) and len(frame) >= 2:
            return int(frame[0]), int(frame[1])
        raise TypeError("frame must be an ndarray-like with .shape or an (h,w) tuple")
    return int(shape[0]), int(shape[1])


__all__ = [
    "tta_detect",
    "detect_fuse",
    "build_views",
]
