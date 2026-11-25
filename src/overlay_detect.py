"""
overlay_detect.py — segment "active match" frames from "filler" using the broadcast
scoreboard overlay, with no OCR required.

Grounded in REAL frames from the 2026 NE Waterbury stream (ngvsHrmoW8U), observed layout
of the persistent top overlay bar during a match (fractions of frame W x H, 16:9):

    +---------------------------------------------------------------+  y=0
    | [BLUE fuel  | BLUE team panel |  TIMER  | RED team panel | RED|  <- overlay bar
    |   x/100 ]   | (3 team numbers)| 0:20    | (3 team numbers)|x/100]  ~ y in [0.00, 0.10]
    +---------------------------------------------------------------+
    |                       (live field view)                       |
    +---------------------------------------------------------------+

Key signature: during a match the LEFT third of the bar is blue-dominant and the RIGHT
third is red-dominant. In filler (crowd, ceiling, walking on field, sponsor slides) that
two-color signature is absent. This works at any resolution because it's a color-mass
test, not a text test.

ROIs (normalized) for the downstream OCR stage are exported in OVERLAY_ROIS so the same
geometry drives both segmentation and field extraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Normalized regions of interest (x0, y0, x1, y1) as fractions of (W, H).
# Tuned to the observed Waterbury overlay; treat as defaults to be refined per broadcast.
OVERLAY_ROIS = {
    "bar":          (0.00, 0.00, 1.00, 0.10),  # whole overlay bar
    "blue_panel":   (0.18, 0.00, 0.42, 0.10),  # blue alliance team numbers
    "red_panel":    (0.58, 0.00, 0.82, 0.10),  # red alliance team numbers
    "timer":        (0.45, 0.00, 0.55, 0.10),  # center match timer
    "blue_fuel":    (0.00, 0.00, 0.10, 0.10),  # blue "x/100" fuel/score box
    "red_fuel":     (0.90, 0.00, 1.00, 0.10),  # red "x/100" fuel/score box
    "match_label":  (0.30, 0.00, 0.70, 0.04),  # "Qualification N of M" strip (very top)
}

# Left third / right third of the bar used purely for the blue-vs-red signature.
_LEFT_SIG = (0.02, 0.01, 0.33, 0.09)
_RIGHT_SIG = (0.67, 0.01, 0.98, 0.09)


def crop_norm(img: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = roi
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def _color_fraction(bgr: np.ndarray, which: str) -> float:
    """Fraction of pixels that are strongly blue or strongly red (HSV gating)."""
    if bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat = (s > 90) & (v > 60)
    if which == "blue":
        mask = sat & (h >= 100) & (h <= 130)
    else:  # red wraps around 0/180
        mask = sat & (((h <= 10) | (h >= 170)))
    return float(mask.mean())


@dataclass
class OverlayVerdict:
    is_match: bool
    blue_left: float
    red_right: float
    score: float


def detect_overlay(img: np.ndarray, thresh: float = 0.12) -> OverlayVerdict:
    """
    Decide whether a frame shows the active-match overlay bar.
    is_match := left-third is blue-dominant AND right-third is red-dominant.
    """
    blue_left = _color_fraction(crop_norm(img, _LEFT_SIG), "blue")
    red_right = _color_fraction(crop_norm(img, _RIGHT_SIG), "red")
    score = min(blue_left, red_right)
    return OverlayVerdict(
        is_match=(blue_left >= thresh and red_right >= thresh),
        blue_left=round(blue_left, 3), red_right=round(red_right, 3),
        score=round(score, 3),
    )


def detect_file(path: str, thresh: float = 0.12) -> OverlayVerdict:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return detect_overlay(img, thresh)


if __name__ == "__main__":
    import sys, glob, os
    paths = sys.argv[1:] or sorted(glob.glob("frames/spread/s_*.jpg"))
    for p in paths:
        v = detect_file(p)
        flag = "MATCH " if v.is_match else "filler"
        print(f"{flag} blue_l={v.blue_left:.3f} red_r={v.red_right:.3f}  {os.path.basename(p)}")
