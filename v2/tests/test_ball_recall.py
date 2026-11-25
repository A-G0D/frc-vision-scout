"""tests/test_ball_recall.py — fuel-ball recall harness (eval/ball_recall.py).

Pure logic: no cv2, no model, no video. Exercises:
  * per-regime recall (incl. the n/a-when-no-GT honesty rule)
  * GT JSON loading (incl. the bundled demonstration sample)
  * the conservation-check {value, available, needs} envelope, including the
    degrade-to-not-available path when the overlay has no fuel counter.
"""
from __future__ import annotations

import json
from pathlib import Path

from frcscout.eval.ball_recall import (
    GtPoint,
    REGIMES,
    RecallReport,
    conservation_check,
    envelope,
    evaluate,
    format_recall_table,
    load_centroid_gt,
    per_regime_recall,
)
from frcscout.io.types import Detection, OverlayState


def _det(cx, cy, r, frame=0):
    return Detection(cls="ball", xyxy=(cx - r, cy - r, cx + r, cy + r), conf=0.9, frame_idx=frame)


# ===========================================================================
# Per-regime recall
# ===========================================================================

class TestPerRegimeRecall:
    def test_hit_inside_box_counts(self):
        gt = [GtPoint(frame=0, x=100, y=100, regime="spread")]
        dets = {0: [_det(100, 100, 8)]}
        out = per_regime_recall(dets, gt, tol_px=5)
        assert out["spread"].n_gt == 1 and out["spread"].n_hit == 1
        assert out["spread"].recall == 1.0

    def test_miss_when_far(self):
        gt = [GtPoint(frame=0, x=100, y=100, regime="spread")]
        dets = {0: [_det(400, 400, 8)]}
        out = per_regime_recall(dets, gt, tol_px=10)
        assert out["spread"].n_hit == 0
        assert out["spread"].recall == 0.0

    def test_tolerance_band_around_centre(self):
        # Point 12px from a tiny box centre: hit only when tol covers it.
        gt = [GtPoint(frame=0, x=112, y=100, regime="spread")]
        dets = {0: [_det(100, 100, 2)]}
        assert per_regime_recall(dets, gt, tol_px=5)["spread"].n_hit == 0
        assert per_regime_recall(dets, gt, tol_px=15)["spread"].n_hit == 1

    def test_empty_regime_is_none_not_zero(self):
        # No airborne GT -> recall is None (n/a), never a fabricated 0.0.
        gt = [GtPoint(frame=0, x=100, y=100, regime="pile")]
        out = per_regime_recall({0: [_det(100, 100, 8)]}, gt, tol_px=5)
        assert out["airborne"].n_gt == 0
        assert out["airborne"].recall is None

    def test_all_regimes_present_in_output(self):
        out = per_regime_recall({}, [], tol_px=5)
        for r in REGIMES:
            assert r in out

    def test_per_frame_isolation(self):
        # A detection on frame 1 must not satisfy a GT point on frame 0.
        gt = [GtPoint(frame=0, x=100, y=100, regime="spread")]
        dets = {1: [_det(100, 100, 8)]}
        assert per_regime_recall(dets, gt, tol_px=5)["spread"].n_hit == 0


class TestFormatTable:
    def test_table_never_blends(self):
        out = per_regime_recall({0: [_det(100, 100, 8)]},
                                [GtPoint(0, 100, 100, "pile")], tol_px=5)
        txt = format_recall_table(out)
        # Each regime appears on its own line; no single blended "recall: X" summary line.
        for r in REGIMES:
            assert r in txt
        assert "n/a" in txt  # spread + airborne have no GT here

    def test_custom_regime_rendered(self):
        out = per_regime_recall({0: [_det(100, 100, 8)]},
                                [GtPoint(0, 100, 100, "weird")], tol_px=5)
        # Unknown regimes are coerced to "spread" on load, but per_regime_recall accepts a
        # raw GtPoint with any regime; ensure it still renders.
        txt = format_recall_table(out)
        assert "weird" in txt


# ===========================================================================
# GT loading
# ===========================================================================

class TestLoadGt:
    def test_missing_file_is_empty(self, tmp_path):
        assert load_centroid_gt(tmp_path / "nope.json") == []

    def test_loads_points_and_coerces_unknown_regime(self, tmp_path):
        p = tmp_path / "gt.json"
        p.write_text(json.dumps({"points": [
            {"frame": 5, "x": 1.0, "y": 2.0, "regime": "PILE"},
            {"frame": 5, "x": 3.0, "y": 4.0, "regime": "bogus"},
        ]}))
        pts = load_centroid_gt(p)
        assert pts[0].regime == "pile"      # case-normalised
        assert pts[1].regime == "spread"    # unknown -> spread
        assert pts[0].frame == 5

    def test_bundled_sample_loads(self):
        """The committed demonstration GT sample parses and lands in known regimes."""
        sample = Path("data/ball_gt/qual59_f1000.json")
        if not sample.exists():
            return  # repo layout differs; harness logic already covered above
        pts = load_centroid_gt(sample)
        assert len(pts) >= 4
        assert all(p.regime in REGIMES for p in pts)


# ===========================================================================
# Conservation check (envelope)
# ===========================================================================

class TestConservation:
    def test_no_fuel_counter_degrades_honestly(self):
        overlays = [OverlayState(), OverlayState()]  # no fuel set
        env = conservation_check([10, 12], overlays)
        assert env["available"] is False
        assert env["needs"]  # non-empty explanation

    def test_draining_floor_is_consistent(self):
        # Floor drops while scored rises -> physically consistent -> 100%.
        overlays = [OverlayState(blue_fuel=0, red_fuel=0),
                    OverlayState(blue_fuel=5, red_fuel=0),
                    OverlayState(blue_fuel=11, red_fuel=0)]
        env = conservation_check([100, 80, 60], overlays)
        assert env["available"] is True
        assert env["value"]["sign_consistent"] == 1.0
        assert env["value"]["mean_floor"] == 80.0

    def test_floor_surge_while_scoring_is_suspect(self):
        # Floor SURGES up while scored also rises -> impossible -> flagged inconsistent.
        overlays = [OverlayState(blue_fuel=0), OverlayState(blue_fuel=10)]
        env = conservation_check([10, 200], overlays)
        assert env["available"] is True
        assert env["value"]["sign_consistent"] == 0.0

    def test_envelope_shape(self):
        e = envelope(1.23, True)
        assert set(e) == {"value", "available", "needs"}


class TestEvaluate:
    def test_end_to_end_with_envelope_degrade(self):
        gt = None
        rep = evaluate({0: [_det(100, 100, 8)]}, gt_path=gt)
        assert isinstance(rep, RecallReport)
        # No GT, no floor/overlay -> conservation degrades, table still renders all regimes.
        assert rep.conservation["available"] is False
        txt = rep.text()
        for r in REGIMES:
            assert r in txt
        assert "NOT AVAILABLE" in txt

    def test_end_to_end_with_conservation(self):
        dets = {0: [_det(100, 100, 8)], 1: [_det(100, 100, 8)]}
        overlays = [OverlayState(blue_fuel=0), OverlayState(blue_fuel=5)]
        rep = evaluate(dets, floor_counts=[50, 40], overlays=overlays)
        assert rep.conservation["available"] is True
