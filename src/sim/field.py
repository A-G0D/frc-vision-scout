"""
field.py — field geometry calibration for the REBUILT (2026) playoff match
shot from the single static wide camera (Upper Bracket R4 M11, 640x360).

This module hard-codes the carpet trapezoid and the two HUB structure regions
for THIS camera so every downstream module (detect / track / events) can mask
to the playable carpet and reason about which alliance's HUB a shot went into.

Scene layout (measured from real frames, fractions of W x H):
  * Top ~0.13  : scoreboard bar (score / timer / fuel counters) — excluded.
  * ~0.13-0.27 : crowd / stands behind the field — excluded.
  * ~0.27-0.55 : grey carpet trapezoid, covered in yellow FUEL, robots driving
                 on it, two dark-navy HUB boxes standing mid-field. INCLUDED.
  * ~0.55+     : field barrier wall, scoring table, refs, and a lower /
                 reflected crowd region — ALL excluded.

The carpet is a trapezoid: narrower at the far (top) edge, wider at the near
(bottom) edge, per normal wide-camera perspective.

Interface (exact):
  CARPET_POLY_NORM : list[(x,y)] fractions, clockwise from top-left.
  carpet_mask(w,h) -> uint8 mask (255 inside carpet, else 0)
  in_carpet(cx,cy,w,h) -> bool
  hub_regions(w,h) -> [{'bbox':(x,y,w,h), 'alliance':'blue'|'red'}]
"""

from __future__ import annotations

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Carpet trapezoid (normalised, clockwise: TL, TR, BR, BL)
# Far (top) edge is slightly inset; near (bottom) edge is wider. Top sits just
# below the crowd band; bottom sits just above the barrier wall.
# ---------------------------------------------------------------------------
CARPET_POLY_NORM = [
    (0.055, 0.300),   # top-left  (far)
    (0.945, 0.300),   # top-right (far)
    (0.995, 0.545),   # bottom-right (near)
    (0.005, 0.545),   # bottom-left  (near)
]

# ---------------------------------------------------------------------------
# HUB structures (normalised bboxes). Two dark-navy box goals stand mid-field:
# the blue-trim HUB on the left, the red-trim HUB on the right. Robots shoot
# FUEL into these. Boxes calibrated to the dark body + white face panel.
# ---------------------------------------------------------------------------
_HUB_NORM = [
    {"bbox_norm": (0.183, 0.343, 0.132, 0.128), "alliance": "blue"},
    {"bbox_norm": (0.658, 0.298, 0.150, 0.138), "alliance": "red"},
]


def _poly_px(w: int, h: int) -> np.ndarray:
    return np.array([(int(x * w), int(y * h)) for x, y in CARPET_POLY_NORM],
                    dtype=np.int32)


def carpet_mask(w: int, h: int) -> np.ndarray:
    """Binary mask (uint8, 255 inside the carpet trapezoid) at size hxw."""
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(m, _poly_px(w, h), 255)
    return m


_MASK_CACHE: dict[tuple[int, int], np.ndarray] = {}


def in_carpet(cx: float, cy: float, w: int, h: int) -> bool:
    """True if pixel (cx,cy) falls inside the carpet trapezoid at size wxh."""
    key = (w, h)
    m = _MASK_CACHE.get(key)
    if m is None:
        m = carpet_mask(w, h)
        _MASK_CACHE[key] = m
    xi, yi = int(round(cx)), int(round(cy))
    if xi < 0 or yi < 0 or xi >= w or yi >= h:
        return False
    return bool(m[yi, xi])


def hub_regions(w: int, h: int) -> list:
    """Pixel bboxes + alliance for the two HUB structures at size wxh."""
    out = []
    for hub in _HUB_NORM:
        bx, by, bw, bh = hub["bbox_norm"]
        out.append({
            "bbox": (int(bx * w), int(by * h), int(bw * w), int(bh * h)),
            "alliance": hub["alliance"],
        })
    return out


# ---------------------------------------------------------------------------
# Visual self-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, glob, os

    paths = sys.argv[1:] or sorted(glob.glob("scratch_calib/t*.png"))
    out_dir = "scratch_calib/field_annot"
    os.makedirs(out_dir, exist_ok=True)
    for p in paths:
        fr = cv2.imread(p)
        if fr is None:
            continue
        h, w = fr.shape[:2]
        ann = fr.copy()
        # carpet outline + translucent fill
        poly = _poly_px(w, h)
        overlay = ann.copy()
        cv2.fillConvexPoly(overlay, poly, (0, 255, 0))
        ann = cv2.addWeighted(overlay, 0.22, ann, 0.78, 0)
        cv2.polylines(ann, [poly], True, (0, 255, 0), 2)
        # hubs
        for hub in hub_regions(w, h):
            x, y, bw, bh = hub["bbox"]
            col = (255, 0, 0) if hub["alliance"] == "blue" else (0, 0, 255)
            cv2.rectangle(ann, (x, y), (x + bw, y + bh), col, 2)
            cv2.putText(ann, "HUB " + hub["alliance"], (x, max(10, y - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        big = cv2.resize(ann, None, fx=2.0, fy=2.0,
                         interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(p)), big)
        print("annotated", p)
