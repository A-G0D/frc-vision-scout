"""tests/test_ball_blob.py — fuel-ball colour-prior detector (detect/ball_blob.py).

Two groups:
  A. Pure-logic helpers (circularity, ROI geometry, field-radius projection) — NO cv2.
  B. End-to-end detect() on a SYNTHETIC frame (yellow discs on a gray field + a magenta
     "crowd" block + a big yellow non-round smear) — exercises the cv2 path when cv2 is
     present, and asserts the precision gates (crowd rejected, smear rejected, discs kept).
     Group B self-skips if cv2 is unavailable so the file still passes in a bare env.

Synthetic, no GPU, no model, no video.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from frcscout.detect.ball_blob import (
    BallBlobConfig,
    BallBlobDetector,
    circularity,
    detect,
    point_in_roi,
    projected_radius_m,
    roi_rect_px,
)
from frcscout.io.types import CameraPose

try:
    import cv2  # noqa: F401
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False

requires_cv2 = pytest.mark.skipif(not _HAVE_CV2, reason="cv2 not available")


# ===========================================================================
# A. Pure-logic helpers (no cv2)
# ===========================================================================

class TestCircularity:
    def test_perfect_circle_is_one(self):
        # A circle of radius r: area=pi r^2, peri=2 pi r -> 4 pi A / P^2 == 1.
        r = 10.0
        area = math.pi * r * r
        peri = 2 * math.pi * r
        assert circularity(area, peri) == pytest.approx(1.0, abs=1e-9)

    def test_square_below_one(self):
        # Square side s: area=s^2, peri=4s -> 4 pi s^2 / 16 s^2 = pi/4 ~ 0.785.
        assert circularity(100.0, 40.0) == pytest.approx(math.pi / 4, abs=1e-9)

    def test_degenerate_is_zero(self):
        assert circularity(0.0, 10.0) == 0.0
        assert circularity(10.0, 0.0) == 0.0


class TestRoiRect:
    def test_none_is_full_frame(self):
        assert roi_rect_px((1080, 1920), None) == (0, 0, 1920, 1080)

    def test_fraction_maps_and_clamps(self):
        x0, y0, x1, y1 = roi_rect_px((1000, 2000), (0.1, 0.2, 0.9, 0.8))
        assert (x0, y0) == (200, 200)
        assert (x1, y1) == (1800, 800)

    def test_reordered_fracs_are_sorted(self):
        # x1<x0 / y1<y0 should be swapped, never produce an empty rect.
        x0, y0, x1, y1 = roi_rect_px((100, 100), (0.9, 0.9, 0.1, 0.1))
        assert x1 > x0 and y1 > y0

    def test_point_in_roi(self):
        rect = (10, 20, 90, 80)
        assert point_in_roi(50, 50, rect)
        assert not point_in_roi(5, 50, rect)
        assert not point_in_roi(50, 95, rect)


class TestProjectedRadius:
    def test_identity_homography_radius_in_px_units(self):
        # H = identity maps px->"field" 1:1, so a 5px radius -> 5.0 "m".
        pose = CameraPose(H=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], valid=True)
        assert projected_radius_m(100, 100, 5.0, pose) == pytest.approx(5.0, abs=1e-6)

    def test_scaled_homography(self):
        # H scales by 0.01 (px -> m), so a 13px radius projects to 0.13 m (a fuel ball).
        pose = CameraPose(H=[[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1]], valid=True)
        assert projected_radius_m(500, 500, 13.0, pose) == pytest.approx(0.13, abs=1e-6)

    def test_invalid_pose_returns_none(self):
        assert projected_radius_m(0, 0, 5.0, None) is None
        assert projected_radius_m(0, 0, 5.0, CameraPose(H=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                                        valid=False)) is None


class TestConfig:
    def test_from_dict_overrides_and_defaults(self):
        c = BallBlobConfig.from_dict({"lab_b_min": 170, "min_circularity": 0.7})
        assert c.lab_b_min == 170
        assert c.min_circularity == 0.7
        # Untouched key keeps the calibrated default.
        assert c.hsv_h_lo == BallBlobConfig().hsv_h_lo

    def test_from_none_is_defaults(self):
        assert BallBlobConfig.from_dict(None) == BallBlobConfig()


# ===========================================================================
# B. End-to-end detect() on a synthetic frame (cv2 path)
# ===========================================================================

def _synthetic_frame() -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Gray 'carpet' frame with vivid-yellow discs, a magenta 'crowd' block, and a big
    yellow non-round smear. Returns (frame, [(cx,cy,r) of the real discs])."""
    h, w = 480, 640
    frame = np.full((h, w, 3), 90, dtype=np.uint8)  # neutral gray

    discs = [(160, 240, 10), (320, 240, 12), (480, 240, 9), (200, 360, 8)]
    yellow_bgr = (40, 220, 235)  # vivid yellow in BGR (high G+R, low B)
    for cx, cy, r in discs:
        cv2.circle(frame, (cx, cy), r, yellow_bgr, -1)

    # A magenta/blue "crowd" block (NOT yellow) — must be rejected by colour.
    cv2.rectangle(frame, (10, 10), (140, 90), (200, 30, 200), -1)

    # A big YELLOW but NON-ROUND smear (banner / jacket) — must be rejected by circularity.
    cv2.rectangle(frame, (500, 380), (620, 410), yellow_bgr, -1)

    return frame, discs


@requires_cv2
class TestDetectSynthetic:
    def test_finds_the_discs(self):
        frame, discs = _synthetic_frame()
        det = BallBlobDetector()
        # roi_frac=None to search the whole synthetic frame (no broadcast overlay here).
        out = det.detect(frame, frame_idx=7, roi_frac=None)
        assert all(d.cls == "ball" for d in out)
        assert all(d.frame_idx == 7 for d in out)
        # Every real disc should have a detection whose centre is within ~1 radius.
        cents = [((d.xyxy[0] + d.xyxy[2]) / 2, (d.xyxy[1] + d.xyxy[3]) / 2) for d in out]
        for cx, cy, r in discs:
            assert any(math.hypot(dx - cx, dy - cy) <= r + 3 for dx, dy in cents), \
                f"disc at ({cx},{cy}) not detected"

    def test_rejects_crowd_block(self):
        frame, _ = _synthetic_frame()
        out = BallBlobDetector().detect(frame, 0, roi_frac=None)
        # No detection centre inside the magenta crowd rectangle [10,10..140,90].
        for d in out:
            cx, cy = (d.xyxy[0] + d.xyxy[2]) / 2, (d.xyxy[1] + d.xyxy[3]) / 2
            assert not (10 <= cx <= 140 and 10 <= cy <= 90), "crowd block was detected"

    def test_rejects_nonround_smear(self):
        frame, _ = _synthetic_frame()
        out = BallBlobDetector().detect(frame, 0, roi_frac=None)
        # The yellow smear [500,380..620,410] is wide+thin -> low circularity -> rejected.
        for d in out:
            cx, cy = (d.xyxy[0] + d.xyxy[2]) / 2, (d.xyxy[1] + d.xyxy[3]) / 2
            assert not (500 <= cx <= 620 and 380 <= cy <= 410), "non-round smear detected"

    def test_module_level_detect_matches_class(self):
        frame, _ = _synthetic_frame()
        a = detect(frame, 0, roi_frac=None)
        b = BallBlobDetector().detect(frame, 0, roi_frac=None)
        assert len(a) == len(b)

    def test_pose_size_gate_rejects_oversize_blob(self):
        """With a pose whose H maps px->m at 0.02 m/px, a 12px-radius disc projects to
        ~0.24 m (< 0.35 cap, kept) while a 40px disc projects to ~0.80 m (> cap, rejected)."""
        h, w = 400, 400
        frame = np.full((h, w, 3), 90, dtype=np.uint8)
        yellow = (40, 220, 235)
        cv2.circle(frame, (120, 200), 12, yellow, -1)   # small -> kept
        cv2.circle(frame, (300, 200), 40, yellow, -1)   # big   -> rejected by field size
        pose = CameraPose(H=[[0.02, 0, 0], [0, 0.02, 0], [0, 0, 1]], valid=True)
        out = BallBlobDetector().detect(frame, 0, pose=pose, roi_frac=None)
        cents = [((d.xyxy[0] + d.xyxy[2]) / 2, (d.xyxy[1] + d.xyxy[3]) / 2) for d in out]
        assert any(math.hypot(cx - 120, cy - 200) <= 15 for cx, cy in cents), "small ball lost"
        assert not any(math.hypot(cx - 300, cy - 200) <= 45 for cx, cy in cents), \
            "oversize blob not rejected by field-radius gate"
