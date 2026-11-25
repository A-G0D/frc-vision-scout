"""Ball inventory detector — YOLO11s INT8 on the iGPU, LOW cadence (~5Hz), tiled.

This is the expensive one. It runs OFF the hot path (async, iGPU) at 5Hz because the
static carpet litter changes slowly. The few balls that move fast are caught by the
60Hz classical FlightDetector instead. That split is the core of the 60fps budget.

Tiling: only over the field trapezoid (from the homography), not the full frame —
fewer pixels, and crowd/stands never enter the tiles, killing the v1 false-positive
problem without a hardcoded pixel band.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from frcscout.io.types import Detection, Track


# ---------------------------------------------------------------------------
# NMS — pure numpy, no ultralytics dependency
# ---------------------------------------------------------------------------

def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Intersection-over-Union of two [x1,y1,x2,y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_detections(
    dets: list[Detection],
    iou_threshold: float = 0.45,
) -> list[Detection]:
    """
    Class-agnostic NMS on a list of Detection objects.

    Sorts by confidence descending, greedily keeps the highest-confidence box,
    and suppresses any box with IoU >= iou_threshold against it.

    Pure numpy — no ultralytics/torch required. Used in tests too.
    """
    if not dets:
        return []

    # Sort highest conf first.
    dets = sorted(dets, key=lambda d: d.conf, reverse=True)
    kept: list[Detection] = []

    while dets:
        best = dets.pop(0)
        kept.append(best)
        ba = np.array(best.xyxy)
        remaining: list[Detection] = []
        for d in dets:
            if _iou(ba, np.array(d.xyxy)) < iou_threshold:
                remaining.append(d)
        dets = remaining

    return kept


# ---------------------------------------------------------------------------
# Tile geometry helpers (pure logic, testable without a model)
# ---------------------------------------------------------------------------

def build_tiles(
    frame_h: int,
    frame_w: int,
    tile_size: int,
    overlap_frac: float,
    field_bbox_xyxy: Optional[tuple[int, int, int, int]] = None,
) -> list[tuple[int, int, int, int]]:
    """
    Return a list of (x0, y0, x1, y1) pixel tiles covering either the whole frame
    or the field_bbox sub-region, with the requested overlap.

    Args:
        frame_h, frame_w: full frame dimensions.
        tile_size: square tile edge length in pixels.
        overlap_frac: fraction of tile_size shared between adjacent tiles (e.g. 0.2).
        field_bbox_xyxy: restrict tiling to this sub-region; None → full frame.
    """
    step = max(1, int(tile_size * (1.0 - overlap_frac)))

    if field_bbox_xyxy is not None:
        fx0, fy0, fx1, fy1 = field_bbox_xyxy
    else:
        fx0, fy0, fx1, fy1 = 0, 0, frame_w, frame_h

    # Clip to frame bounds.
    fx0 = max(0, fx0)
    fy0 = max(0, fy0)
    fx1 = min(frame_w, fx1)
    fy1 = min(frame_h, fy1)

    tiles: list[tuple[int, int, int, int]] = []
    y = fy0
    while y < fy1:
        x = fx0
        while x < fx1:
            x1 = min(x + tile_size, fx1)
            y1 = min(y + tile_size, fy1)
            # Only include tiles with positive area.
            if x1 > x and y1 > y:
                tiles.append((x, y, x1, y1))
            x += step
            if x >= fx1:
                break
        y += step
        if y >= fy1:
            break

    return tiles


def remap_detections_to_frame(
    tile_dets: list[Detection],
    tile_x0: int,
    tile_y0: int,
) -> list[Detection]:
    """Translate box coordinates from tile-local to full-frame coordinates."""
    out: list[Detection] = []
    for d in tile_dets:
        x1, y1, x2, y2 = d.xyxy
        out.append(Detection(
            cls=d.cls,
            xyxy=(x1 + tile_x0, y1 + tile_y0, x2 + tile_x0, y2 + tile_y0),
            conf=d.conf,
            frame_idx=d.frame_idx,
        ))
    return out


# ---------------------------------------------------------------------------
# Field bounding box from mask
# ---------------------------------------------------------------------------

def _field_bbox_from_mask(mask: Optional[np.ndarray]) -> Optional[tuple[int, int, int, int]]:
    """
    Return the axis-aligned bounding box of all non-zero pixels in `mask`.
    Returns None if mask is None or all-zero.
    """
    if mask is None:
        return None
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BallInventoryNet:
    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self.imgsz = self.cfg.get("imgsz", 960)
        self.conf = self.cfg.get("conf", 0.15)
        self.device = self.cfg.get("device", "intel:gpu")
        self.tiled = self.cfg.get("tiled", True)
        self.tile = self.cfg.get("tile", {"size": 512, "overlap": 0.2})
        self._model = None
        self._next_id = 0
        # Optional field object injected after construction (set by the engine).
        self.field = None

    def _ensure(self):
        if self._model is None:
            from ultralytics import YOLO  # guarded — ultralytics may be absent
            self._model = YOLO(self.cfg["model"], task="detect")

    # ------------------------------------------------------------------
    # Single-tile inference helper
    # ------------------------------------------------------------------

    def _infer_crop(
        self,
        crop: np.ndarray,
        frame_idx: int,
    ) -> list[Detection]:
        """Run the model on a single image crop; return tile-local detections."""
        r = self._model.predict(
            crop,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )[0]
        out: list[Detection] = []
        if r.boxes is not None:
            for b, cf in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
            ):
                out.append(Detection("ball", tuple(map(float, b)), float(cf), frame_idx))
        return out

    # ------------------------------------------------------------------
    # Public infer entry points
    # ------------------------------------------------------------------

    def infer(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        self._ensure()
        if self.tiled:
            return self._infer_tiled(frame, frame_idx)
        return self._infer_full(frame, frame_idx)

    def _infer_full(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        r = self._model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )[0]
        out: list[Detection] = []
        if r.boxes is not None:
            for b, cf in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
            ):
                out.append(Detection("ball", tuple(map(float, b)), float(cf), frame_idx))
        return out

    def _infer_tiled(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        """
        SAHI-style tiled inference over the field region.

        Algorithm:
            1. Get field bounding box from the homography mask (if available).
            2. Slice the frame into overlapping tiles covering that region.
            3. Run the model per tile.
            4. Remap each tile's detections back to full-frame coordinates.
            5. Merge all detections with class-agnostic NMS to remove duplicates
               from overlapping tile regions.
        """
        h, w = frame.shape[:2]
        tile_size = int(self.tile.get("size", 512))
        overlap   = float(self.tile.get("overlap", 0.2))

        # Try to get the field bounding box from the field object.
        field_bbox: Optional[tuple[int, int, int, int]] = None
        if self.field is not None:
            try:
                mask = self.field.field_mask((h, w))
                field_bbox = _field_bbox_from_mask(mask)
            except Exception:
                pass  # field not ready yet — tile the whole frame

        tiles = build_tiles(h, w, tile_size, overlap, field_bbox)

        all_dets: list[Detection] = []
        for x0, y0, x1, y1 in tiles:
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            tile_dets = self._infer_crop(crop, frame_idx)
            all_dets.extend(remap_detections_to_frame(tile_dets, x0, y0))

        # Merge duplicates introduced by overlapping tiles.
        return nms_detections(all_dets, iou_threshold=0.45)

    # ------------------------------------------------------------------
    # Convenience wrapper
    # ------------------------------------------------------------------

    def as_tracks(self, dets: list[Detection]) -> list[Track]:
        """Inventory balls aren't ID-tracked frame-to-frame (they're static); expose as
        lightweight Tracks so the geometry layer can field-project them."""
        out: list[Track] = []
        for d in dets:
            x1, y1, x2, y2 = d.xyxy
            out.append(Track(
                tid=-1,
                cls="ball",
                xyxy=d.xyxy,
                centroid_px=((x1 + x2) / 2, (y1 + y2) / 2),
            ))
        return out
