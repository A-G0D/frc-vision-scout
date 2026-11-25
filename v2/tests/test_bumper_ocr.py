"""tests/test_bumper_ocr.py — pure-logic unit tests for refine/bumper_ocr.py.

NO easyocr, NO GPU. The whole BumperOCR.read() path is exercised with a STUB ocr_fn
that receives the (preprocessed) crop and returns canned (text, conf) items, so the
crop-geometry + digit-plausibility + OCR-parsing logic is covered without a model.

Three groups:
  A. bumper_roi_xyxy   — crop-region geometry
  B. plausible_team / parse_team_from_ocr — digit plausibility + result parsing
  C. BumperOCR.read    — end-to-end with a stub OCR fn (fake frame)
"""
from __future__ import annotations

import sys
import types

# ---------------------------------------------------------------------------
# Guard: stub easyocr if it's not importable, so the module imports in a bare env.
# (cv2 / numpy are present in CI; the module guards them internally regardless.)
# ---------------------------------------------------------------------------

def _maybe_stub(name: str) -> None:
    try:
        __import__(name)
    except ModuleNotFoundError:
        sys.modules[name] = types.ModuleType(name)

_maybe_stub("easyocr")

from frcscout.refine.bumper_ocr import (  # noqa: E402
    MAX_TEAM_DIGITS,
    TEAM_MAX,
    TEAM_MIN,
    BumperOCR,
    bumper_roi_xyxy,
    parse_team_from_ocr,
    plausible_team,
)


# ===========================================================================
# A. Crop-region geometry — bumper_roi_xyxy
# ===========================================================================

class TestBumperRoi:
    def test_lower_band_default_is_lower_40pct(self):
        """Default band_top_frac=0.60 → ROI starts 60% down a 100px-tall box."""
        roi = bumper_roi_xyxy((0, 0, 100, 100))
        assert roi is not None
        x1, y1, x2, y2 = roi
        assert y1 == 60.0          # 0 + 0.60*100
        assert y2 == 100.0         # 0 + 1.00*100

    def test_horizontal_inset_applied_each_side(self):
        """x_inset_frac=0.08 trims 8px off each side of a 100px-wide box."""
        roi = bumper_roi_xyxy((0, 0, 100, 100), x_inset_frac=0.08)
        assert roi is not None
        x1, _y1, x2, _y2 = roi
        assert x1 == 8.0
        assert x2 == 92.0

    def test_offset_box_translates_roi(self):
        """ROI is in absolute image px, so an offset box shifts the band."""
        roi = bumper_roi_xyxy((200, 300, 300, 400), x_inset_frac=0.0)
        assert roi == (200.0, 360.0, 300.0, 400.0)

    def test_degenerate_zero_area_box_returns_none(self):
        assert bumper_roi_xyxy((50, 50, 50, 50)) is None
        assert bumper_roi_xyxy((50, 50, 40, 80)) is None   # x2<x1

    def test_inverted_band_fracs_returns_none(self):
        """band_bottom <= band_top is invalid."""
        assert bumper_roi_xyxy((0, 0, 100, 100),
                               band_top_frac=0.8, band_bottom_frac=0.5) is None

    def test_clip_to_frame_bounds(self):
        """A box whose bumper band straddles the right/bottom edge is clipped to frame.

        Box (60,10)->(140,90): w=80,h=80. Band (lower 40%) = y[58,90], x-inset 0 -> x[60,140].
        With a 100x100 frame the right edge (x2=140) clips to 100; y stays in-frame.
        """
        roi = bumper_roi_xyxy((60, 10, 140, 90), x_inset_frac=0.0, clip_wh=(100, 100))
        assert roi is not None
        x1, y1, x2, y2 = roi
        assert x1 == 60.0
        assert x2 == 100.0    # 140 clipped to frame width
        assert y1 == 58.0     # 10 + 0.60*80
        assert y2 == 90.0     # in-frame, unchanged

    def test_box_fully_below_frame_returns_none(self):
        """If the bumper band falls entirely outside the frame, return None."""
        # 60% down a box at y=[200,300] is y=260, frame is only 100 tall.
        roi = bumper_roi_xyxy((0, 200, 100, 300), clip_wh=(100, 100))
        assert roi is None

    def test_inset_clamped_below_half(self):
        """An absurd inset (>=0.5) is clamped so the ROI never collapses from x-inset."""
        roi = bumper_roi_xyxy((0, 0, 100, 100), x_inset_frac=0.9)
        assert roi is not None
        x1, _, x2, _ = roi
        assert x2 > x1


# ===========================================================================
# B. Digit plausibility + OCR-result parsing
# ===========================================================================

class TestPlausibleTeam:
    def test_typical_team_numbers_ok(self):
        for n in (1, 33, 254, 1114, 9999, 11000):
            assert plausible_team(n), n

    def test_zero_and_negative_rejected(self):
        assert not plausible_team(0)
        assert not plausible_team(-5)

    def test_above_ceiling_rejected(self):
        assert not plausible_team(TEAM_MAX + 1)
        assert not plausible_team(99999)

    def test_digit_cap_matches_ceiling(self):
        """The digit cap is derived from TEAM_MAX so the two guards never contradict."""
        assert MAX_TEAM_DIGITS == len(str(TEAM_MAX)) == 5

    def test_value_above_ceiling_rejected_even_within_digit_cap(self):
        """12345 fits the 5-digit cap but exceeds the 12000 ceiling → rejected by value."""
        assert not plausible_team(12345)

    def test_six_digit_fused_read_rejected_by_digit_cap(self):
        """A 6-digit fused read (e.g. two numbers stuck together) is rejected."""
        assert not plausible_team(254971)

    def test_boundaries(self):
        assert plausible_team(TEAM_MIN)
        assert plausible_team(TEAM_MAX)


class TestParseTeamFromOcr:
    def test_empty_input(self):
        assert parse_team_from_ocr([]) == (None, 0.0)

    def test_single_clean_read(self):
        team, conf = parse_team_from_ocr([("254", 0.91)])
        assert team == 254
        assert conf == 0.91

    def test_strips_non_digits(self):
        """OCR junk around the number is stripped before parsing."""
        team, conf = parse_team_from_ocr([(" #1114 ", 0.8)])
        assert team == 1114

    def test_below_min_conf_rejected(self):
        assert parse_team_from_ocr([("254", 0.10)], min_conf=0.30) == (None, 0.0)

    def test_highest_confidence_wins(self):
        items = [("33", 0.55), ("254", 0.92), ("971", 0.40)]
        team, conf = parse_team_from_ocr(items)
        assert team == 254
        assert conf == 0.92

    def test_conf_tie_prefers_more_digits(self):
        """Equal confidence → the longer (harder-to-hallucinate) number wins."""
        items = [("7", 0.80), ("1678", 0.80)]
        team, _conf = parse_team_from_ocr(items)
        assert team == 1678

    def test_implausible_values_skipped(self):
        """A 5-digit fused read is dropped; the valid one is returned."""
        items = [("123456", 0.99), ("48", 0.70)]
        team, conf = parse_team_from_ocr(items)
        assert team == 48
        assert conf == 0.70

    def test_all_implausible_returns_none(self):
        assert parse_team_from_ocr([("0", 0.99), ("88888", 0.95)]) == (None, 0.0)

    def test_no_digits_returns_none(self):
        assert parse_team_from_ocr([("ABC", 0.99)]) == (None, 0.0)


# ===========================================================================
# C. BumperOCR.read — full path with a STUB ocr_fn (no easyocr)
# ===========================================================================

def _fake_frame(w: int = 200, h: int = 200):
    """A blank BGR frame as a numpy array if numpy is available, else a duck-typed stub
    that supports .shape and 2D slicing (so the geometry path still runs)."""
    try:
        import numpy as np
        return np.zeros((h, w, 3), dtype=np.uint8)
    except Exception:  # pragma: no cover
        class _F:
            shape = (h, w, 3)
            size = w * h * 3
            def __getitem__(self, _k):
                class _C:
                    ndim = 2
                    size = 1
                    shape = (1, 1)
                    def __getitem__(self, _k2):
                        return 0
                return _C()
        return _F()


class TestBumperOCRRead:
    def test_read_returns_team_from_stub(self):
        """End-to-end: a stub OCR fn returns a clean read; read() yields that team."""
        ocr = BumperOCR()
        seen = {}

        def stub(crop):
            seen["called"] = True
            return [("254", 0.88)]

        team, conf = ocr.read(_fake_frame(), (40, 40, 160, 180), ocr_fn=stub)
        assert seen.get("called") is True   # the crop pipeline actually invoked OCR
        assert team == 254
        assert conf == 0.88

    def test_read_rejects_tiny_box_without_calling_ocr(self):
        """A box below min_box_px must short-circuit (OCR never runs)."""
        ocr = BumperOCR({"min_box_px": 50})
        calls = []

        def stub(crop):
            calls.append(1)
            return [("33", 0.99)]

        # 20px-wide box → bumper band well under 50px → rejected.
        team, conf = ocr.read(_fake_frame(), (0, 0, 20, 20), ocr_fn=stub)
        assert team is None
        assert conf == 0.0
        assert calls == []   # never reached OCR

    def test_read_degenerate_box_returns_none(self):
        ocr = BumperOCR()
        team, conf = ocr.read(_fake_frame(), (10, 10, 10, 10), ocr_fn=lambda c: [("1", 0.9)])
        assert (team, conf) == (None, 0.0)

    def test_read_filters_implausible_stub_output(self):
        """Stub returns only garbage → read() returns (None, 0.0)."""
        ocr = BumperOCR()
        team, conf = ocr.read(
            _fake_frame(), (40, 40, 160, 180),
            ocr_fn=lambda c: [("999999", 0.95), ("xx", 0.9)],
        )
        assert (team, conf) == (None, 0.0)

    def test_read_respects_min_conf_from_config(self):
        """A read below the configured confidence floor is dropped."""
        ocr = BumperOCR({"min_conf": 0.80})
        team, conf = ocr.read(
            _fake_frame(), (40, 40, 160, 180),
            ocr_fn=lambda c: [("1678", 0.50)],
        )
        assert (team, conf) == (None, 0.0)

    def test_stub_receives_a_crop_object(self):
        """The stub is handed the preprocessed crop (not None) for a valid box."""
        ocr = BumperOCR()
        got = {}

        def stub(crop):
            got["crop"] = crop
            return [("48", 0.9)]

        ocr.read(_fake_frame(), (40, 40, 160, 180), ocr_fn=stub)
        assert "crop" in got
        assert got["crop"] is not None

    def test_read_picks_best_among_multiple_stub_reads(self):
        ocr = BumperOCR()
        team, conf = ocr.read(
            _fake_frame(), (40, 40, 160, 180),
            ocr_fn=lambda c: [("33", 0.40), ("1114", 0.93), ("0", 0.99)],
        )
        # "0" is implausible (rejected); 1114 has higher conf than 33.
        assert team == 1114
        assert conf == 0.93
