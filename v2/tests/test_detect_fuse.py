"""tests/test_detect_fuse.py — pure-logic unit tests for refine/detect_fuse.py.

NO model, NO GPU, NO easyocr/ultralytics. numpy IS available in CI (matches the
detect-layer test assumptions). Everything here is exercisable in a bare env that
has only numpy installed.

Coverage:
  * WBF: two sources' overlapping boxes fuse to ONE confidence-weighted box
  * disjoint boxes are kept separate, untouched
  * alliance is majority-voted (incl. None votes, ties, confidence weighting)
  * flat-list input is accepted (single logical source)
  * class-agnostic NMS fallback (suppress, don't blend)
  * robustness: empty / None sources, degenerate boxes, flipped corners,
    Detection-dataclass inputs (no alliance), conf threshold, per-source weights
"""
from __future__ import annotations

import math

import pytest

from frcscout.refine.detect_fuse import fuse, nms, iou_xyxy


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _d(x1, y1, x2, y2, conf=0.9, alliance=None):
    return {"xyxy": (x1, y1, x2, y2), "conf": conf, "alliance": alliance}


def _approx_box(b, expected, tol=1e-6):
    assert len(b) == 4
    for got, exp in zip(b, expected):
        assert math.isclose(got, exp, abs_tol=tol), f"{b} != {expected}"


# ===========================================================================
# IoU sanity (the clustering primitive)
# ===========================================================================

class TestIoU:
    def test_identical_is_one(self):
        assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_disjoint_is_zero(self):
        assert iou_xyxy((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0

    def test_touching_edge_is_zero(self):
        # Share only an edge -> zero area intersection.
        assert iou_xyxy((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0

    def test_half_overlap(self):
        # A=[0,0,10,10] area100, B=[5,0,15,10] area100, inter=[5,0,10,10]=50,
        # union=150 -> 1/3.
        assert math.isclose(iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)

    def test_degenerate_box_zero(self):
        assert iou_xyxy((0, 0, 0, 0), (0, 0, 10, 10)) == 0.0


# ===========================================================================
# Core WBF behaviour (the contract)
# ===========================================================================

class TestFuseOverlap:
    def test_two_sources_overlap_fuse_to_one(self):
        """The headline requirement: one object seen by two sources -> one box."""
        src_a = [_d(10, 10, 50, 50, conf=0.9)]
        src_b = [_d(12, 12, 52, 52, conf=0.8)]  # IoU ~0.85 with A
        out = fuse([src_a, src_b], iou_thr=0.5)
        assert len(out) == 1

    def test_fused_box_is_confidence_weighted_blend(self):
        """Blend must be weighted by confidence, NOT a plain midpoint.

        A at x1=0 conf 0.9, B at x1=10 conf 0.1 (heavily overlapping in y).
        Weighted x1 = (0*0.9 + 10*0.1)/(0.9+0.1) = 1.0, not the midpoint 5.0.
        """
        src_a = [_d(0, 0, 100, 100, conf=0.9)]
        src_b = [_d(10, 0, 110, 100, conf=0.1)]
        out = fuse([src_a, src_b], iou_thr=0.5)
        assert len(out) == 1
        x1 = out[0]["xyxy"][0]
        assert math.isclose(x1, 1.0, abs_tol=1e-6)
        # x2 similarly: (100*0.9 + 110*0.1)/1.0 = 101.0
        assert math.isclose(out[0]["xyxy"][2], 101.0, abs_tol=1e-6)

    def test_three_identical_boxes_collapse(self):
        out = fuse([[_d(5, 5, 25, 25, 0.7)],
                    [_d(5, 5, 25, 25, 0.6)],
                    [_d(5, 5, 25, 25, 0.5)]], iou_thr=0.5)
        assert len(out) == 1
        _approx_box(out[0]["xyxy"], (5, 5, 25, 25))

    def test_full_consensus_keeps_mean_confidence(self):
        """A box seen by ALL sources is scaled by sources/sources = 1 -> mean conf."""
        out = fuse([[_d(0, 0, 10, 10, 0.8)],
                    [_d(0, 0, 10, 10, 0.6)]], iou_thr=0.5)
        assert len(out) == 1
        assert math.isclose(out[0]["conf"], 0.7, abs_tol=1e-9)  # mean(0.8,0.6)*1

    def test_single_source_box_is_downweighted(self):
        """With 2 sources, a box only one source saw is scaled by 1/2."""
        out = fuse([[_d(0, 0, 10, 10, 0.8)],
                    [_d(500, 500, 510, 510, 0.8)]], iou_thr=0.5)
        # Two disjoint boxes; each seen by exactly one of the two sources.
        assert len(out) == 2
        for o in out:
            assert math.isclose(o["conf"], 0.4, abs_tol=1e-9)  # 0.8 * (1/2)


class TestFuseDisjoint:
    def test_disjoint_boxes_kept_separate(self):
        src_a = [_d(0, 0, 30, 30, 0.9)]
        src_b = [_d(200, 200, 230, 230, 0.9)]
        out = fuse([src_a, src_b], iou_thr=0.5)
        assert len(out) == 2

    def test_disjoint_within_one_source_all_kept(self):
        src = [_d(0, 0, 20, 20, 0.9),
               _d(100, 0, 120, 20, 0.9),
               _d(200, 0, 220, 20, 0.9)]
        out = fuse([src], iou_thr=0.5)
        assert len(out) == 3

    def test_partial_overlap_below_thr_not_fused(self):
        """IoU below the threshold -> two boxes survive."""
        # A=[0,0,40,40], B=[30,30,70,70]; IoU ~0.032 < 0.5.
        out = fuse([[_d(0, 0, 40, 40, 0.9)], [_d(30, 30, 70, 70, 0.9)]],
                   iou_thr=0.5)
        assert len(out) == 2


# ===========================================================================
# Alliance majority voting
# ===========================================================================

class TestAllianceVote:
    def test_majority_red_wins(self):
        out = fuse([[_d(0, 0, 10, 10, 0.9, "red")],
                    [_d(0, 0, 10, 10, 0.9, "red")],
                    [_d(0, 0, 10, 10, 0.9, "blue")]], iou_thr=0.5)
        assert len(out) == 1
        assert out[0]["alliance"] == "red"

    def test_none_votes_ignored(self):
        """A ball detection (alliance None) must not dilute a clear red vote."""
        out = fuse([[_d(0, 0, 10, 10, 0.9, "red")],
                    [_d(0, 0, 10, 10, 0.9, None)]], iou_thr=0.5)
        assert out[0]["alliance"] == "red"

    def test_all_none_stays_none(self):
        out = fuse([[_d(0, 0, 10, 10, 0.9, None)],
                    [_d(0, 0, 10, 10, 0.9, None)]], iou_thr=0.5)
        assert out[0]["alliance"] is None

    def test_confidence_weighted_vote(self):
        """One high-conf blue beats two low-conf reds by confidence weight."""
        out = fuse([[_d(0, 0, 10, 10, 0.05, "red")],
                    [_d(0, 0, 10, 10, 0.05, "red")],
                    [_d(0, 0, 10, 10, 0.95, "blue")]], iou_thr=0.5)
        assert out[0]["alliance"] == "blue"

    def test_exact_tie_resolves_to_none(self):
        out = fuse([[_d(0, 0, 10, 10, 0.5, "red")],
                    [_d(0, 0, 10, 10, 0.5, "blue")]], iou_thr=0.5)
        assert out[0]["alliance"] is None

    def test_invalid_alliance_string_treated_as_none(self):
        out = fuse([[_d(0, 0, 10, 10, 0.9, "purple")],
                    [_d(0, 0, 10, 10, 0.9, "red")]], iou_thr=0.5)
        assert out[0]["alliance"] == "red"

    def test_alliance_case_insensitive(self):
        out = fuse([[_d(0, 0, 10, 10, 0.9, "RED")]], iou_thr=0.5)
        assert out[0]["alliance"] == "red"


# ===========================================================================
# Input flexibility / robustness
# ===========================================================================

class _FakeDet:
    """Mimics the frozen Detection dataclass: has .xyxy/.conf, no alliance attr."""
    def __init__(self, xyxy, conf):
        self.xyxy = xyxy
        self.conf = conf


class TestInputForms:
    def test_flat_list_accepted_as_single_source(self):
        out = fuse([_d(0, 0, 10, 10, 0.9), _d(100, 0, 110, 10, 0.9)], iou_thr=0.5)
        assert len(out) == 2  # disjoint, single source

    def test_object_detections_without_alliance(self):
        a = [_FakeDet((0, 0, 10, 10), 0.9)]
        b = [_FakeDet((1, 1, 11, 11), 0.8)]
        out = fuse([a, b], iou_thr=0.5)
        assert len(out) == 1
        assert out[0]["alliance"] is None

    def test_mixed_dict_and_object(self):
        a = [_FakeDet((0, 0, 10, 10), 0.9)]
        b = [_d(1, 1, 11, 11, 0.8, "blue")]
        out = fuse([a, b], iou_thr=0.5)
        assert len(out) == 1
        assert out[0]["alliance"] == "blue"

    def test_empty_overall_returns_empty(self):
        assert fuse([]) == []
        assert fuse([[], []]) == []

    def test_none_sources_skipped(self):
        out = fuse([None, [_d(0, 0, 10, 10, 0.9)], None], iou_thr=0.5)
        assert len(out) == 1

    def test_flipped_corners_canonicalized_and_fused(self):
        """A box given as (x2,y2,x1,y1) must still overlap its twin."""
        a = [_d(0, 0, 10, 10, 0.9)]
        b = [{"xyxy": (10, 10, 0, 0), "conf": 0.8, "alliance": None}]  # reversed
        out = fuse([a, b], iou_thr=0.5)
        assert len(out) == 1
        _approx_box(out[0]["xyxy"], (0, 0, 10, 10), tol=1e-6)

    def test_invalid_detection_skipped(self):
        bad = {"conf": 0.9}  # no xyxy
        out = fuse([[bad, _d(0, 0, 10, 10, 0.9)]], iou_thr=0.5)
        assert len(out) == 1

    def test_output_keys_exact(self):
        out = fuse([[_d(0, 0, 10, 10, 0.9, "red")]])
        assert set(out[0].keys()) == {"xyxy", "conf", "alliance"}
        assert isinstance(out[0]["xyxy"], tuple)
        assert len(out[0]["xyxy"]) == 4


class TestThresholdsAndWeights:
    def test_conf_threshold_filters_inputs(self):
        out = fuse([[_d(0, 0, 10, 10, 0.05), _d(100, 0, 110, 10, 0.9)]],
                   iou_thr=0.5, conf_thr=0.1)
        assert len(out) == 1
        _approx_box(out[0]["xyxy"], (100, 0, 110, 10))

    def test_per_source_weight_shifts_blend(self):
        """Up-weighting source B drags the fused box toward B's coordinates."""
        a = [_d(0, 0, 100, 100, conf=0.5)]
        b = [_d(20, 0, 120, 100, conf=0.5)]
        # Equal source weights: x1 = (0+20)/2 = 10.
        eq = fuse([a, b], iou_thr=0.5)
        assert math.isclose(eq[0]["xyxy"][0], 10.0, abs_tol=1e-6)
        # Weight B 3x: x1 = (0*0.5*1 + 20*0.5*3)/(0.5+1.5) = 30/2 = 15.
        wt = fuse([a, b], iou_thr=0.5, weights=[1.0, 3.0])
        assert math.isclose(wt[0]["xyxy"][0], 15.0, abs_tol=1e-6)

    def test_zero_confidence_does_not_crash(self):
        """All-zero confidence falls back to an unweighted mean box.

        Boxes overlap enough to cluster (IoU(0,0,10,10 vs 1,1,11,11)=81/119≈0.68
        > 0.5); with both weights zero the blend degrades to the plain mean.
        """
        out = fuse([[_d(0, 0, 10, 10, 0.0)], [_d(1, 1, 11, 11, 0.0)]],
                   iou_thr=0.5)
        assert len(out) == 1
        _approx_box(out[0]["xyxy"], (0.5, 0.5, 10.5, 10.5))

    def test_output_sorted_by_confidence_desc(self):
        out = fuse([[_d(0, 0, 10, 10, 0.3, "red")],
                    [_d(0, 0, 10, 10, 0.3, "red")],   # consensus -> conf 0.3
                    [_d(500, 500, 510, 510, 0.9)]],    # solo -> conf 0.9*1/3=0.3
                   iou_thr=0.5)
        confs = [o["conf"] for o in out]
        assert confs == sorted(confs, reverse=True)


# ===========================================================================
# Class-agnostic NMS fallback
# ===========================================================================

class TestNMSFallback:
    def test_nms_suppresses_overlap_keeps_highest(self):
        out = nms([[_d(10, 10, 50, 50, 0.85)], [_d(12, 12, 52, 52, 0.75)]],
                  iou_thr=0.5)
        assert len(out) == 1
        assert out[0]["conf"] == 0.85
        # NMS does NOT blend -> the surviving box is the original highest-conf one.
        _approx_box(out[0]["xyxy"], (10, 10, 50, 50))

    def test_nms_keeps_disjoint(self):
        out = nms([_d(0, 0, 30, 30, 0.9), _d(200, 200, 230, 230, 0.9)],
                  iou_thr=0.5)
        assert len(out) == 2

    def test_nms_empty(self):
        assert nms([]) == []

    def test_nms_flat_list(self):
        out = nms([_d(0, 0, 10, 10, 0.9), _d(1, 1, 11, 11, 0.8)], iou_thr=0.5)
        assert len(out) == 1
        assert out[0]["conf"] == 0.9


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
