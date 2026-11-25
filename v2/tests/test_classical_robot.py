"""tests/test_classical_robot.py — unit tests for the classical bumper-blob robot
proposer (refine/classical_robot.py).

Two groups:
  A. Pure gating/geometry logic — NO cv2 required, always runs.
  B. End-to-end propose() on synthetic colored-rectangle frames — requires cv2 and
     numpy; skipped (not failed) if cv2 is unavailable, per the task contract.

No model, no GPU, no network.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from frcscout.refine.classical_robot import (
    ClassicalRobotProposer,
    _area_frac,
    _aspect,
    _box_score,
    _clip_box,
    _dedup_boxes,
    _iou,
    _passes_gates,
)

_HAS_CV2 = importlib.util.find_spec("cv2") is not None
cv2_required = pytest.mark.skipif(not _HAS_CV2, reason="cv2 not installed")


# =========================================================================== #
# A. Pure-logic tests (no cv2)
# =========================================================================== #

class TestAreaFrac:
    def test_half_frame(self):
        # box 50x100 in a 100x100 frame = 5000 / 10000 = 0.5
        assert _area_frac((0, 0, 50, 100), (100, 100)) == pytest.approx(0.5)

    def test_full_frame(self):
        assert _area_frac((0, 0, 200, 100), (100, 200)) == pytest.approx(1.0)

    def test_zero_area_box(self):
        assert _area_frac((10, 10, 10, 10), (100, 100)) == 0.0

    def test_degenerate_frame_is_safe(self):
        assert _area_frac((0, 0, 10, 10), (0, 0)) == 0.0

    def test_resolution_invariance(self):
        """The SAME physical fraction at 720p and 1080p yields the same value —
        this is the whole point of fraction-based gating."""
        # A box covering a quarter of the width and half the height.
        f720 = _area_frac((0, 0, 1280 / 4, 720 / 2), (720, 1280))
        f1080 = _area_frac((0, 0, 1920 / 4, 1080 / 2), (1080, 1920))
        assert f720 == pytest.approx(f1080)
        assert f720 == pytest.approx(0.125)


class TestAspect:
    def test_wide_box(self):
        assert _aspect((0, 0, 40, 20)) == pytest.approx(2.0)

    def test_tall_box(self):
        assert _aspect((0, 0, 20, 40)) == pytest.approx(0.5)

    def test_zero_height_is_inf(self):
        assert _aspect((0, 0, 40, 0)) == float("inf")


class TestPassesGates:
    FRAME = (1080, 1920)

    def test_typical_robot_box_passes(self):
        # ~ 160x120 px @ 1080p: area frac ~0.0093, aspect 1.33 — a believable robot.
        assert _passes_gates((100, 100, 260, 220), self.FRAME) is True

    def test_too_small_rejected(self):
        # 10x10 px = way below min_area_frac
        assert _passes_gates((0, 0, 10, 10), self.FRAME) is False

    def test_too_large_rejected(self):
        # nearly the whole frame -> a banner/wall, not one robot
        assert _passes_gates((0, 0, 1900, 1000), self.FRAME) is False

    def test_too_wide_rejected(self):
        # a long thin sponsor stripe: 1200x60 -> aspect 20, but also size; force a
        # box that passes area yet fails aspect.
        # 600x40 @1080p: area frac 0.0116 (ok); aspect 15 (> max 6) -> reject.
        assert _passes_gates((0, 0, 600, 40), self.FRAME) is False

    def test_custom_gates_respected(self):
        box = (0, 0, 100, 100)  # square, area frac ~0.0048
        # Default would pass; a stricter min_area_frac rejects it.
        assert _passes_gates(box, self.FRAME, min_area_frac=0.01) is False
        # And a relaxed aspect window still passes the square.
        assert _passes_gates(box, self.FRAME, min_aspect=0.9, max_aspect=1.1) is True


class TestBoxScore:
    def test_monotonic_in_fill(self):
        assert _box_score(0.9, 0.9) > _box_score(0.4, 0.9)

    def test_monotonic_in_color(self):
        assert _box_score(0.9, 0.9) > _box_score(0.9, 0.4)

    def test_capped_below_one(self):
        # Even a perfect blob is capped so the NN wins ties downstream.
        assert _box_score(1.0, 1.0) <= 0.85

    def test_in_unit_interval_and_clamps(self):
        # Out-of-range inputs are clamped, never NaN / >1.
        s = _box_score(5.0, -1.0)
        assert 0.0 <= s <= 0.85

    def test_low_color_frac_penalised(self):
        """A solid blob in a box that is mostly background scores low."""
        assert _box_score(1.0, 0.05) < 0.3


class TestClipBox:
    def test_clamps_to_frame(self):
        assert _clip_box((-10, -5, 150, 120), (100, 100)) == (0.0, 0.0, 100.0, 100.0)

    def test_reorders_inverted(self):
        # x2<x1 / y2<y1 get swapped after clamping
        x1, y1, x2, y2 = _clip_box((80, 80, 20, 20), (100, 100))
        assert x1 <= x2 and y1 <= y2

    def test_inside_box_unchanged(self):
        assert _clip_box((10, 10, 40, 40), (100, 100)) == (10.0, 10.0, 40.0, 40.0)


class TestIoU:
    def test_identical_is_one(self):
        assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_disjoint_is_zero(self):
        assert _iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0

    def test_half_overlap(self):
        # A=[0,0,10,10] (100), B=[5,0,15,10] (100), inter=[5,0,10,10]=50,
        # union=150 -> IoU = 1/3
        assert _iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1.0 / 3.0)


class TestDedupBoxes:
    def test_empty(self):
        assert _dedup_boxes([]) == []

    def test_overlapping_same_alliance_collapses_keep_higher_score(self):
        props = [
            {"xyxy": (0, 0, 100, 100), "alliance": "red", "score": 0.4},
            {"xyxy": (5, 5, 105, 105), "alliance": "red", "score": 0.8},
        ]
        out = _dedup_boxes(props)
        assert len(out) == 1
        assert out[0]["score"] == 0.8

    def test_overlapping_different_alliance_both_kept(self):
        """A red robot beside a blue robot can overlap in image space — both must
        survive because alliance differs."""
        props = [
            {"xyxy": (0, 0, 100, 100), "alliance": "red", "score": 0.7},
            {"xyxy": (5, 5, 105, 105), "alliance": "blue", "score": 0.6},
        ]
        out = _dedup_boxes(props)
        assert len(out) == 2

    def test_far_apart_same_alliance_both_kept(self):
        props = [
            {"xyxy": (0, 0, 50, 50), "alliance": "blue", "score": 0.7},
            {"xyxy": (500, 500, 560, 560), "alliance": "blue", "score": 0.6},
        ]
        out = _dedup_boxes(props)
        assert len(out) == 2


# =========================================================================== #
# B. Synthetic-frame end-to-end tests (require cv2)
# =========================================================================== #

# Saturated BGR colors that land inside the HSV bumper bands.
_BGR_RED = (0, 0, 255)     # HSV H≈0,   S=255, V=255  -> red band
_BGR_BLUE = (255, 0, 0)    # HSV H≈120, S=255, V=255  -> blue band
_BGR_GRAY = (128, 128, 128)  # S≈0 -> rejected by saturation floor


def _blank(h: int = 720, w: int = 1280) -> "np.ndarray":
    """A desaturated gray 'carpet' background (passes nothing on its own)."""
    return np.full((h, w, 3), _BGR_GRAY, dtype=np.uint8)


def _paint(frame, xyxy, bgr) -> None:
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    frame[y1:y2, x1:x2] = bgr


@cv2_required
class TestProposeSynthetic:
    def test_finds_single_red_robot(self):
        frame = _blank()
        _paint(frame, (300, 200, 460, 360), _BGR_RED)  # 160x160 solid red bumper
        props = ClassicalRobotProposer().propose(frame)
        assert len(props) >= 1
        reds = [p for p in props if p["alliance"] == "red"]
        assert len(reds) == 1
        x1, y1, x2, y2 = reds[0]["xyxy"]
        # Recovered box should hug the painted rectangle (within a few px of morph).
        assert x1 == pytest.approx(300, abs=10)
        assert y1 == pytest.approx(200, abs=10)
        assert x2 == pytest.approx(460, abs=10)
        assert y2 == pytest.approx(360, abs=10)
        assert 0.0 < reds[0]["score"] <= 0.85

    def test_finds_single_blue_robot(self):
        frame = _blank()
        _paint(frame, (700, 300, 880, 460), _BGR_BLUE)
        props = ClassicalRobotProposer().propose(frame)
        blues = [p for p in props if p["alliance"] == "blue"]
        assert len(blues) == 1

    def test_finds_one_red_and_one_blue(self):
        frame = _blank()
        _paint(frame, (100, 100, 260, 260), _BGR_RED)
        _paint(frame, (900, 400, 1060, 560), _BGR_BLUE)
        props = ClassicalRobotProposer().propose(frame)
        assert sum(p["alliance"] == "red" for p in props) == 1
        assert sum(p["alliance"] == "blue" for p in props) == 1

    def test_blank_frame_yields_nothing(self):
        """A desaturated frame has no saturated bumper color -> no proposals."""
        assert ClassicalRobotProposer().propose(_blank()) == []

    def test_gray_rectangle_not_proposed(self):
        """A large but UNsaturated rectangle (e.g. a robot's metal frame) is not a
        bumper and must not be proposed."""
        frame = _blank()
        _paint(frame, (300, 200, 600, 500), _BGR_GRAY)
        assert ClassicalRobotProposer().propose(frame) == []

    def test_tiny_speck_rejected_by_size_gate(self):
        """A 6x6 saturated speck (noise / a far-away light) is below min_area_frac."""
        frame = _blank()
        _paint(frame, (640, 360, 646, 366), _BGR_RED)
        assert ClassicalRobotProposer().propose(frame) == []

    def test_full_frame_color_rejected_by_max_size(self):
        """An all-red frame (camera glitch / flag covering lens) exceeds max_area_frac
        and is not a single robot."""
        frame = np.full((720, 1280, 3), _BGR_RED, dtype=np.uint8)
        props = ClassicalRobotProposer().propose(frame)
        assert props == []

    def test_field_mask_suppresses_out_of_field_color(self):
        """Red 'crowd' color outside the field mask is suppressed; in-field red is
        kept. This is the key precision lever for broadcast footage."""
        frame = _blank()
        _paint(frame, (100, 50, 260, 150), _BGR_RED)     # 'stands' (top of frame)
        _paint(frame, (500, 400, 660, 560), _BGR_RED)    # on the 'carpet'
        # Field mask covers only the lower-center carpet region.
        mask = np.zeros((720, 1280), dtype=np.uint8)
        mask[350:600, 400:900] = 255
        props = ClassicalRobotProposer().propose(frame, field_mask=mask)
        reds = [p for p in props if p["alliance"] == "red"]
        assert len(reds) == 1
        # The surviving box is the carpet one (y around 400+).
        assert reds[0]["xyxy"][1] >= 300

    def test_field_mask_none_searches_whole_frame(self):
        """Without a mask both red regions are candidates (cold-start behavior)."""
        frame = _blank()
        _paint(frame, (100, 50, 260, 210), _BGR_RED)
        _paint(frame, (500, 400, 660, 560), _BGR_RED)
        props = ClassicalRobotProposer().propose(frame, field_mask=None)
        assert sum(p["alliance"] == "red" for p in props) == 2

    def test_adjacent_red_and_blue_both_recovered(self):
        """Two robots of different alliance touching each other -> two proposals,
        one per color (dedup must not merge across alliance)."""
        frame = _blank()
        _paint(frame, (300, 300, 460, 460), _BGR_RED)
        _paint(frame, (460, 300, 620, 460), _BGR_BLUE)  # shares the x=460 seam
        props = ClassicalRobotProposer().propose(frame)
        assert sum(p["alliance"] == "red" for p in props) == 1
        assert sum(p["alliance"] == "blue" for p in props) == 1

    def test_mismatched_mask_shape_ignored_not_crashed(self):
        """A wrong-shaped field_mask is defensively ignored, not fatal."""
        frame = _blank()
        _paint(frame, (300, 200, 460, 360), _BGR_RED)
        bad_mask = np.ones((100, 100), dtype=np.uint8) * 255
        props = ClassicalRobotProposer().propose(frame, field_mask=bad_mask)
        assert sum(p["alliance"] == "red" for p in props) == 1

    def test_empty_frame_returns_empty(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        assert ClassicalRobotProposer().propose(empty) == []

    def test_score_in_unit_band(self):
        frame = _blank()
        _paint(frame, (300, 200, 500, 400), _BGR_BLUE)
        for p in ClassicalRobotProposer().propose(frame):
            assert 0.0 < p["score"] <= 0.85


@cv2_required
class TestHSVBandSanity:
    """Confirm the chosen synthetic BGR colors actually fall in the HSV bands the
    proposer thresholds on — guards the test fixtures themselves."""

    def test_red_and_blue_bgr_map_into_bands(self):
        import cv2

        px = np.array([[_BGR_RED, _BGR_BLUE, _BGR_GRAY]], dtype=np.uint8)
        hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0]
        (hr, sr, vr), (hb, sb, vb), (hg, sg, vg) = hsv
        # red: H<=10 or H>=170, saturated
        assert (hr <= 10 or hr >= 170) and sr > 90 and vr > 60
        # blue: 100<=H<=130, saturated
        assert 100 <= hb <= 130 and sb > 90 and vb > 60
        # gray: desaturated -> fails the saturation floor
        assert sg <= 90
