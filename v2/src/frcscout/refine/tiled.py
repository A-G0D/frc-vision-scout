"""Tiled-inference helper for small-object recall (SAHI-style slicing).

Small game pieces (the ~5Hz "ball inventory" pass) shrink to a handful of pixels in a
1080p wide shot and get missed at native imgsz. The fix is to slice the frame — or just
the field region — into overlapping tiles, run the detector per tile (each tile is then
a larger fraction of imgsz, so the object is effectively up-sampled), then stitch the
per-tile detections back into one frame-coordinate list and drop the duplicates that
fall in the overlap seams.

This module is the PURE-LOGIC core of that pipeline — it owns the geometry and the
merge, and runs no model. The caller (detect/ball_net.py) supplies the actual crops to
the detector; here we only:

    build_tiles(h, w, tile, overlap, field_bbox)  -> [(x0, y0, x1, y1), ...]
    remap(tile_dets, x0, y0)                      -> dets in frame coordinates
    merge_tiles(per_tile_dets, offsets)           -> remap-all + dedup
    nms(dets, iou_thresh)                         -> class-aware greedy NMS

Everything here is plain Python (no numpy/cv2 needed) so the merge logic is unit-testable
in a bare interpreter. numpy is imported lazily and ONLY for the optional vectorised
`pairwise_iou` convenience; the file parses and every core function works without it.

Coordinate convention: boxes are (x1, y1, x2, y2) in PIXELS with x2 >= x1, y2 >= y1.
Tiles are (x0, y0, x1, y1) half-open on the high edge: a tile covers columns [x0, x1)
and rows [y0, y1), so adjacent non-overlapping tiles share no pixel.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, TypeVar, Union

# A box is any (x1, y1, x2, y2). A "detection" is either such a tuple or any object
# exposing an `.xyxy` 4-tuple (e.g. io.types.Detection). We stay duck-typed so the
# helper is usable both before Detection objects are built (raw model boxes) and after.
Box = tuple[float, float, float, float]
Tile = tuple[int, int, int, int]
DetT = TypeVar("DetT")


# ---------------------------------------------------------------------------
# 1. Tile geometry
# ---------------------------------------------------------------------------

def build_tiles(
    h: int,
    w: int,
    tile: int,
    overlap: float,
    field_bbox: Optional[tuple[int, int, int, int]] = None,
) -> list[Tile]:
    """Cover a frame (or a sub-region of it) with overlapping square-ish tiles.

    Args:
        h, w: full frame height / width in pixels.
        tile: target tile edge length in pixels (square).
        overlap: fraction of `tile` shared between adjacent tiles, in [0, 1).
                 0.0 → abutting tiles; 0.2 → 20% seam; clamped to [0, 0.95].
        field_bbox: (x0, y0, x1, y1) region to tile instead of the whole frame
                    (typically the homography field trapezoid's bounding box, so
                    the stands never enter a tile). None → tile the full frame.

    Returns:
        A list of (x0, y0, x1, y1) integer tiles, in row-major order, each clamped
        to the region and to the frame, each with strictly positive area. The union
        of the tiles fully covers the region: the last tile in every row/column is
        aligned flush to the region's right/bottom edge (so a partial remainder
        strip is still covered, never dropped) rather than left clamped-and-short.

    Guarantees / edge cases:
        * Region smaller than `tile` in a dimension → a single tile spanning it.
        * No duplicate identical tiles (the flush-alignment of the final tile is
          de-duplicated against the stepped one).
        * Degenerate / inverted / empty region → [].
    """
    h = int(h)
    w = int(w)
    tile = max(1, int(tile))
    # Clamp overlap into a sane, non-degenerate range so `step >= 1` always holds.
    if overlap != overlap:        # NaN guard
        overlap = 0.0
    overlap = min(max(float(overlap), 0.0), 0.95)
    step = max(1, int(round(tile * (1.0 - overlap))))

    if field_bbox is not None:
        rx0, ry0, rx1, ry1 = field_bbox
    else:
        rx0, ry0, rx1, ry1 = 0, 0, w, h

    # Clip the region to the frame.
    rx0 = max(0, int(rx0))
    ry0 = max(0, int(ry0))
    rx1 = min(w, int(rx1))
    ry1 = min(h, int(ry1))
    if rx1 <= rx0 or ry1 <= ry0:
        return []

    xs = _axis_starts(rx0, rx1, tile, step)
    ys = _axis_starts(ry0, ry1, tile, step)

    tiles: list[Tile] = []
    for y0 in ys:
        y1 = min(y0 + tile, ry1)
        for x0 in xs:
            x1 = min(x0 + tile, rx1)
            if x1 > x0 and y1 > y0:
                tiles.append((x0, y0, x1, y1))
    return tiles


def _axis_starts(lo: int, hi: int, tile: int, step: int) -> list[int]:
    """Tile start coordinates along one axis covering [lo, hi).

    Steps from `lo` by `step`; appends a final start flush to the high edge
    (hi - tile, but never below lo) so the trailing remainder is covered. The
    flush start is de-duplicated against the stepped sequence.
    """
    span = hi - lo
    if span <= tile:
        return [lo]
    starts: list[int] = []
    pos = lo
    last_valid_start = hi - tile        # a full tile from here ends exactly at hi
    while pos <= last_valid_start:
        starts.append(pos)
        pos += step
    # Ensure the region's tail is covered by a flush-right/bottom tile.
    if not starts or starts[-1] != last_valid_start:
        starts.append(last_valid_start)
    return starts


# ---------------------------------------------------------------------------
# 2. Coordinate remap (tile-local -> frame)
# ---------------------------------------------------------------------------

def _get_xyxy(det: object) -> Box:
    """Extract an (x1, y1, x2, y2) box from a detection-like value.

    Accepts a bare 4-sequence or any object with an `.xyxy` 4-tuple attribute
    (e.g. io.types.Detection). Raises TypeError otherwise.
    """
    xyxy = getattr(det, "xyxy", det)
    try:
        x1, y1, x2, y2 = xyxy  # type: ignore[misc]
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"cannot read an (x1,y1,x2,y2) box from {det!r}"
        ) from exc
    return float(x1), float(y1), float(x2), float(y2)


def _with_xyxy(det: DetT, box: Box) -> DetT:
    """Return a copy of `det` with its box replaced by `box`.

    * Bare tuple/list in → same-type box out.
    * frozen dataclass (Detection) → dataclasses.replace.
    * other object with a writable `.xyxy` → shallow copy with the box swapped.
    """
    if isinstance(det, (tuple, list)):
        return type(det)(box)  # type: ignore[return-value,call-arg]

    # Try dataclasses.replace first (handles frozen Detection cleanly).
    try:
        import dataclasses
        if dataclasses.is_dataclass(det) and not isinstance(det, type):
            return dataclasses.replace(det, xyxy=box)  # type: ignore[misc]
    except Exception:
        pass

    # Fallback: shallow-copy and set the attribute (mutable detection objects).
    import copy
    new = copy.copy(det)
    try:
        setattr(new, "xyxy", box)
    except Exception as exc:  # pragma: no cover - exotic immutable types
        raise TypeError(f"cannot set .xyxy on a copy of {det!r}") from exc
    return new


def remap(tile_dets: Sequence[DetT], x0: float, y0: float) -> list[DetT]:
    """Translate detections from tile-local to full-frame pixel coordinates.

    Adds the tile origin (x0, y0) to both corners of every box. Works on bare
    (x1,y1,x2,y2) tuples and on Detection-like objects alike, returning values of
    the same type (Detections are rebuilt via dataclasses.replace, so the frozen
    originals are left untouched).
    """
    out: list[DetT] = []
    for det in tile_dets:
        x1, y1, x2, y2 = _get_xyxy(det)
        out.append(_with_xyxy(det, (x1 + x0, y1 + y0, x2 + x0, y2 + y0)))
    return out


# ---------------------------------------------------------------------------
# 3. Merge + dedup across tiles
# ---------------------------------------------------------------------------

def iou(a: Box, b: Box) -> float:
    """Intersection-over-Union of two (x1,y1,x2,y2) boxes. Pure Python."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _conf_of(det: object) -> float:
    """Confidence used to order NMS. Falls back to 0.0 if absent (bare boxes)."""
    c = getattr(det, "conf", None)
    try:
        return float(c) if c is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _cls_of(det: object) -> object:
    """Class key for class-aware NMS. None for bare boxes (→ class-agnostic)."""
    return getattr(det, "cls", None)


def nms(
    dets: Sequence[DetT],
    iou_thresh: float = 0.45,
    class_aware: bool = True,
) -> list[DetT]:
    """Greedy non-max suppression to collapse duplicate boxes from tile seams.

    Sorts by descending confidence, keeps the top box, and suppresses any later
    box whose IoU against a kept box of the SAME class exceeds `iou_thresh`.
    Stable for equal confidences (insertion order is the tie-break, so the result
    is deterministic). Bare-box inputs have conf 0.0 and class None, giving plain
    class-agnostic NMS in original order.

    Returns the kept detections (original objects, not copies), highest-confidence
    first within each suppression decision.
    """
    if not dets:
        return []
    # Decorate with original index for a stable, deterministic sort.
    order = sorted(
        range(len(dets)),
        key=lambda i: (-_conf_of(dets[i]), i),
    )
    kept_idx: list[int] = []
    kept_boxes: list[Box] = []
    kept_cls: list[object] = []
    for i in order:
        box = _get_xyxy(dets[i])
        cls = _cls_of(dets[i]) if class_aware else None
        suppressed = False
        for kb, kc in zip(kept_boxes, kept_cls):
            if class_aware and kc != cls:
                continue
            if iou(box, kb) > iou_thresh:
                suppressed = True
                break
        if not suppressed:
            kept_idx.append(i)
            kept_boxes.append(box)
            kept_cls.append(cls)
    return [dets[i] for i in kept_idx]


# Alias: callers think of the cross-tile step as "dedup".
dedup = nms


def merge_tiles(
    per_tile_dets: Sequence[Sequence[DetT]],
    tile_offsets: Sequence[tuple[float, float]],
    iou_thresh: float = 0.45,
    class_aware: bool = True,
) -> list[DetT]:
    """Remap every tile's detections to frame coordinates, then dedup the seams.

    Args:
        per_tile_dets: one detection list per tile (same order as `tile_offsets`).
        tile_offsets:  the (x0, y0) origin of each tile — i.e. the first two
                       elements of each tile from build_tiles().
        iou_thresh:    overlap above which two same-class boxes are duplicates.
        class_aware:   suppress only within a class (True) or globally (False).

    Returns:
        One flat, frame-coordinate, deduplicated detection list.

    Raises:
        ValueError if the two sequences differ in length.
    """
    if len(per_tile_dets) != len(tile_offsets):
        raise ValueError(
            f"per_tile_dets ({len(per_tile_dets)}) and tile_offsets "
            f"({len(tile_offsets)}) length mismatch"
        )
    merged: list[DetT] = []
    for dets, (x0, y0) in zip(per_tile_dets, tile_offsets):
        merged.extend(remap(dets, x0, y0))
    return nms(merged, iou_thresh=iou_thresh, class_aware=class_aware)


# ---------------------------------------------------------------------------
# 4. Optional numpy convenience (guarded — not needed by the core)
# ---------------------------------------------------------------------------

def pairwise_iou(boxes: "Iterable[Box]"):
    """N×N IoU matrix as a numpy array. Requires numpy; raises if it's absent.

    Purely a convenience for callers that already work in numpy (e.g. diagnostics);
    the core merge path above uses the pure-Python `iou`/`nms` and needs no numpy.
    """
    import numpy as np  # guarded: only this optional helper needs numpy

    arr = np.asarray(list(boxes), dtype=np.float64).reshape(-1, 4)
    n = arr.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    x1 = np.maximum(arr[:, None, 0], arr[None, :, 0])
    y1 = np.maximum(arr[:, None, 1], arr[None, :, 1])
    x2 = np.minimum(arr[:, None, 2], arr[None, :, 2])
    y2 = np.minimum(arr[:, None, 3], arr[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = np.clip(arr[:, 2] - arr[:, 0], 0, None) * np.clip(arr[:, 3] - arr[:, 1], 0, None)
    union = area[:, None] + area[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out
