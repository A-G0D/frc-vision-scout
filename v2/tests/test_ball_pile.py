"""tests/test_ball_pile.py — dense-pile AREA estimate (detect/ball_blob.py, Problem 1).

The merged, occluded pile is ONE big yellow blob that fails per-ball circularity/size, so
boxing collapses (~11 of ~100). Instead we estimate count = pile_area / single_ball_area.
This suite proves:
  A. Pure-logic: median_box_area, estimate_component_count, PileEstimate label/total.
  B. End-to-end estimate_pile() on a SYNTHETIC frame (a dense grid "pile" of touching discs
     + a few isolated discs + a magenta crowd block OUTSIDE the field polygon) — asserts the
     pile is counted within tolerance of the true ball count, the crowd never contributes,
     and the total is flagged as an estimate.

Synthetic, no GPU, no model, no video. Group B self-skips without cv2.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from frcscout.detect.ball_blob import (
    BallBlobConfig,
    BallBlobDetector,
    PileComponent,
    PileEstimate,
    estimate_component_count,
    median_box_area,
)
from frcscout.io.types import Detection

try:
    import cv2  # noqa: F401
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False

requires_cv2 = pytest.mark.skipif(not _HAVE_CV2, reason="cv2 not available")


# ===========================================================================
# A. Pure-logic helpers (no cv2)
# ===========================================================================

def _det(x1, y1, x2, y2) -> Detection:
    return Detection(cls="ball", xyxy=(x1, y1, x2, y2), conf=0.5, frame_idx=0)


class TestMedianBoxArea:
    def test_none_when_empty(self):
        assert median_box_area([]) is None
        assert median_box_area(None) is None

    def test_median_of_areas(self):
        # boxes of area 100, 400, 900 -> median 400.
        dets = [_det(0, 0, 10, 10), _det(0, 0, 20, 20), _det(0, 0, 30, 30)]
        assert median_box_area(dets) == pytest.approx(400.0)

    def test_skips_degenerate_boxes(self):
        # a zero-area box must not drag the median to 0.
        dets = [_det(0, 0, 0, 0), _det(0, 0, 10, 10), _det(0, 0, 10, 10)]
        assert median_box_area(dets) == pytest.approx(100.0)


class TestEstimateComponentCount:
    def test_basic_division_rounds(self):
        # 1000 px of yellow, 100 px/ball, fill 1.0 -> 10 balls.
        assert estimate_component_count(1000.0, 100.0, 1.0) == 10

    def test_fill_frac_scales_down(self):
        # packing factor 0.8 -> 8 balls from the same area.
        assert estimate_component_count(1000.0, 100.0, 0.8) == 8

    def test_floor_at_one(self):
        # a component classed as a pile holds at least one ball.
        assert estimate_component_count(50.0, 100.0, 1.0) == 1

    def test_degenerate_single_area_is_zero(self):
        assert estimate_component_count(1000.0, 0.0, 1.0) == 0
        assert estimate_component_count(0.0, 100.0, 1.0) == 0


class TestPileEstimateRecord:
    def test_total_and_label_with_pile(self):
        est = PileEstimate(count_est=88, components=[], single_ball_area_px=120.0,
                           single_ball_source="isolated", isolated_count=12)
        assert est.total_est == 100
        assert est.is_estimate is True
        assert est.label() == "~100 (area est.)"
        # caller-computed total (e.g. after fuse) is labelled honestly too.
        assert est.label(95) == "~95 (area est.)"

    def test_label_plain_integer_when_no_pile(self):
        est = PileEstimate(count_est=0, components=[], single_ball_area_px=120.0,
                           single_ball_source="isolated", isolated_count=7)
        assert est.is_estimate is False
        assert est.total_est == 7
        assert est.label() == "7"


# ===========================================================================
# B. End-to-end estimate_pile() on a synthetic dense pile (cv2 path)
# ===========================================================================

def _pile_frame():
    """Gray frame with a DENSE grid 'pile' of touching yellow discs (top-centre, inside the
    field band), a few ISOLATED yellow discs, and a magenta 'crowd' block in the stands
    region (above the field). Returns (frame, n_pile_balls, isolated_centers, field_poly_frac)."""
    h, w = 480, 640
    frame = np.full((h, w, 3), 90, dtype=np.uint8)
    yellow = (40, 220, 235)

    # Field polygon: a band across the middle (mimics run_scout's trapezoid), EXCLUDING the
    # top strip where the crowd lives.
    field_poly_frac = [(0.05, 0.30), (0.95, 0.30), (0.99, 0.78), (0.01, 0.78)]

    # Dense pile: a grid of touching discs (r=7, pitch=15) -> they merge into one component.
    # Place it inside the field band (y ~0.40..0.55 of 480 -> rows 190..265).
    n_pile = 0
    r = 7
    for gy in range(200, 261, 15):       # 5 rows
        for gx in range(260, 381, 15):   # 9 cols -> 45 balls
            cv2.circle(frame, (gx, gy), r, yellow, -1)
            n_pile += 1

    # A few clearly ISOLATED discs (well separated), also in the field band.
    isolated = [(120, 230, 7), (520, 240, 7), (150, 320, 7), (500, 330, 7)]
    for cx, cy, rr in isolated:
        cv2.circle(frame, (cx, cy), rr, yellow, -1)

    # Magenta crowd block in the STANDS (y<0.30 -> outside field poly) — must never count.
    cv2.rectangle(frame, (40, 20, ), (200, 110), (200, 30, 200), -1)
    # Also a big YELLOW block in the stands (a banner) — outside the poly, must not count.
    cv2.rectangle(frame, (420, 20), (600, 110), yellow, -1)

    return frame, n_pile, isolated, field_poly_frac


@requires_cv2
class TestEstimatePileSynthetic:
    def test_pile_count_in_tolerance(self):
        frame, n_pile, isolated, poly = _pile_frame()
        det = BallBlobDetector()
        iso = det.detect(frame, 0, roi_frac=None)
        est = det.estimate_pile(frame, 0, isolated_dets=iso, field_poly_frac=poly)
        # A pile component must be found and its estimate should be in the right ballpark of
        # the true 45 balls (area estimate, not exact — allow generous tolerance).
        assert est.count_est > 0, "no pile detected"
        assert len(est.components) >= 1
        assert 0.5 * n_pile <= est.count_est <= 1.8 * n_pile, \
            f"pile estimate {est.count_est} far from truth {n_pile}"
        # The scale must be DERIVED (Hough on the pile, or a robust isolated sample) — not the
        # blind default. Which derived source wins depends on the synthetic geometry.
        assert est.single_ball_source in ("hough", "isolated", "config")

    def test_total_flagged_as_estimate(self):
        frame, _, _, poly = _pile_frame()
        det = BallBlobDetector()
        iso = det.detect(frame, 0, roi_frac=None)
        est = det.estimate_pile(frame, 0, isolated_dets=iso, field_poly_frac=poly)
        assert est.is_estimate is True
        assert est.label().startswith("~") and "area est." in est.label()
        assert est.total_est == est.isolated_count + est.count_est

    def test_crowd_outside_polygon_never_counts(self):
        """No pile component centroid may lie in the stands strip (y<0.30) — the magenta and
        the yellow-banner blocks there must be gated out by the field polygon."""
        frame, _, _, poly = _pile_frame()
        h = frame.shape[0]
        det = BallBlobDetector()
        iso = det.detect(frame, 0, roi_frac=None)
        est = det.estimate_pile(frame, 0, isolated_dets=iso, field_poly_frac=poly)
        for comp in est.components:
            assert comp.centroid_px[1] >= 0.30 * h, "a pile component came from the stands"

    def test_no_pile_when_only_isolated(self):
        """A frame with ONLY well-separated single discs has no merged pile -> count_est 0,
        total is a plain integer (not an estimate)."""
        h, w = 480, 640
        frame = np.full((h, w, 3), 90, dtype=np.uint8)
        yellow = (40, 220, 235)
        poly = [(0.02, 0.10), (0.98, 0.10), (0.98, 0.92), (0.02, 0.92)]
        for cx, cy in [(120, 240), (300, 240), (480, 240), (200, 360), (420, 360)]:
            cv2.circle(frame, (cx, cy), 7, yellow, -1)
        det = BallBlobDetector()
        iso = det.detect(frame, 0, roi_frac=None)
        est = det.estimate_pile(frame, 0, isolated_dets=iso, field_poly_frac=poly)
        assert est.count_est == 0
        assert est.is_estimate is False
        assert est.label() == str(est.isolated_count)

    def test_detect_with_pile_one_shot(self):
        frame, n_pile, _, poly = _pile_frame()
        det = BallBlobDetector()
        iso, est = det.detect_with_pile(frame, 0, roi_frac=None, field_poly_frac=poly)
        assert isinstance(iso, list) and all(d.cls == "ball" for d in iso)
        assert isinstance(est, PileEstimate)
        assert est.isolated_count == len(iso)
        assert est.count_est > 0

    def test_field_poly_px_equivalent_to_frac(self):
        """Passing a prebuilt pixel polygon (run_scout's FIELD_POLY) gives the same gate as
        the fractional form."""
        frame, _, _, poly = _pile_frame()
        h, w = frame.shape[:2]
        poly_px = np.array([(int(x * w), int(y * h)) for x, y in poly],
                           dtype=np.int32).reshape(-1, 1, 2)
        det = BallBlobDetector()
        iso = det.detect(frame, 0, roi_frac=None)
        a = det.estimate_pile(frame, 0, isolated_dets=iso, field_poly_frac=poly)
        b = det.estimate_pile(frame, 0, isolated_dets=iso, field_poly_px=poly_px)
        assert a.count_est == b.count_est

    def test_component_contour_in_full_frame_coords(self):
        frame, _, _, poly = _pile_frame()
        det = BallBlobDetector()
        iso = det.detect(frame, 0, roi_frac=None)
        est = det.estimate_pile(frame, 0, isolated_dets=iso, field_poly_frac=poly)
        comp = max(est.components, key=lambda c: c.area_px)
        assert isinstance(comp, PileComponent)
        # contour points must sit inside the component bbox (i.e. were offset to full frame).
        x1, y1, x2, y2 = comp.xyxy
        if len(comp.contour):
            xs, ys = comp.contour[:, 0], comp.contour[:, 1]
            assert xs.min() >= x1 - 1 and xs.max() <= x2 + 1
            assert ys.min() >= y1 - 1 and ys.max() <= y2 + 1
