"""scout_render tests — headless, synthetic. cv2/numpy via importorskip.

We build a small ``np.zeros`` frame plus a hand-made FrameState / six-track list /
phase output and assert ``draw_scout_overlay`` returns an annotated ndarray of the SAME
height/width and does not raise. We also check the two contract guarantees that matter:
the module IMPORTS without cv2/numpy (only the draw needs them), and an UNAVAILABLE
metric is never rendered as a 0 (it must read as the NA token).
"""
from __future__ import annotations

import pytest

from frcscout.io.types import Event, FrameState, OverlayState, Track

# The module must import on a bare interpreter; the DRAW needs cv2 + numpy.
from frcscout.viz import scout_render
from frcscout.viz.scout_render import draw_scout_overlay

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")


# ───────────────────────────── synthetic inputs ──────────────────────────────
def _six():
    """Three robots: a known-red, a known-blue, and an unknown-alliance (gray) one, plus
    one coasting (est=True) to exercise the dimmed-box path."""
    return [
        {"id": 3, "alliance": "red", "xyxy": (100, 100, 160, 200),
         "centroid": (130, 150), "est": False, "hits": 30, "age": 40},
        {"id": 1, "alliance": "blue", "xyxy": (300, 120, 360, 220),
         "centroid": (330, 170), "est": False, "hits": 25, "age": 40},
        {"id": 5, "alliance": "red", "xyxy": (500, 140, 560, 240),
         "centroid": (530, 190), "est": True, "hits": 5, "age": 40},
        {"id": 9, "alliance": None, "xyxy": (600, 140, 660, 240),
         "centroid": (630, 190), "est": False, "hits": 3, "age": 10},
    ]


def _phase_out():
    """Per-robot phase records mirroring PhaseTracker.update() output, including the
    {value, available} envelope on gated metrics (one available, one NOT)."""
    return {
        3: {"robot_id": 3, "alliance": "red", "phase": "SHOOTING", "held": 0,
            "climb_level": 0, "label": "R3 SHOOTING", "team": 254,
            "metrics": _metrics(acc=1.0, acc_avail=True, max_held=2, climb=0)},
        1: {"robot_id": 1, "alliance": "blue", "phase": "HOLDING", "held": 2,
            "climb_level": 0, "label": "B1 HOLDING(2)", "team": 1678,
            "metrics": _metrics(acc=None, acc_avail=False, max_held=2, climb=0)},
        5: {"robot_id": 5, "alliance": "red", "phase": "CLIMBING", "held": 0,
            "climb_level": 2, "label": "R5 CLIMB L2",
            "metrics": _metrics(acc=None, acc_avail=False, max_held=1, climb=2)},
        # id 9 deliberately absent from phase_out (renderer must tolerate a six track
        # with no phase record yet).
    }


def _metrics(acc, acc_avail, max_held, climb):
    return {
        "shot_accuracy": {"value": acc, "available": acc_avail,
                          "needs": None if acc_avail else "a HUB pose or overlay pulse"},
        "max_held": max_held,
        "climb_level": climb,
        "defense_time_s": {"value": 0.0, "available": False, "needs": "a field pose"},
        "jitter_index": {"value": None, "available": False, "units": None,
                         "needs": ">=2 frames"},
    }


def _frame_state():
    ov = OverlayState(match_label="QM 12", blue_score=42, red_score=37,
                      blue_fuel=5, red_fuel=8, phase="teleop")
    flights = [Track(tid=90, cls="ball", xyxy=(400, 50, 420, 70),
                     centroid_px=(410, 60), vel_mps=(2.0, -1.0), is_flight=True)]
    inventory = [Track(tid=20, cls="ball", xyxy=(120, 180, 140, 200),
                       centroid_px=(130, 190))]
    events = [
        Event(t_s=1.0, frame_idx=60, type="shot", alliance="red", robot_id=3),
        Event(t_s=1.1, frame_idx=66, type="score", alliance="blue", robot_id=1),
    ]
    return FrameState(frame_idx=66, t_s=1.1, robots=[], balls_flight=flights,
                      balls_inventory=inventory, overlay=ov, events=events)


# ──────────────────────────────── the tests ──────────────────────────────────
def test_draw_returns_ndarray_same_hw():
    h, w = 480, 854
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    out = draw_scout_overlay(frame, _frame_state(), _six(), _phase_out(),
                             {3: _metrics(1.0, True, 2, 0)}, fps=58.7)
    assert isinstance(out, np.ndarray)
    assert out.shape == (h, w, 3)
    assert out.dtype == np.uint8


def test_draw_does_not_mutate_input():
    h, w = 360, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    before = frame.copy()
    draw_scout_overlay(frame, _frame_state(), _six(), _phase_out(),
                       {3: _metrics(1.0, True, 2, 0)}, fps=60.0)
    assert np.array_equal(frame, before), "draw_scout_overlay must not mutate the input frame"


def test_draw_actually_annotates():
    """Something is drawn — the output differs from the all-zero input."""
    h, w = 360, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    out = draw_scout_overlay(frame, _frame_state(), _six(), _phase_out(),
                             {3: _metrics(1.0, True, 2, 0)}, fps=60.0)
    assert int(out.sum()) > 0, "expected the overlay to draw SOMETHING on the blank frame"


def test_draw_handles_empty_inputs():
    """Empty six / phase / metrics and a bare FrameState must not raise."""
    h, w = 240, 320
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    st = FrameState(frame_idx=0, t_s=0.0)
    out = draw_scout_overlay(frame, st, [], {}, {})
    assert out.shape == (h, w, 3)


def test_draw_handles_unknown_alliance_and_missing_phase():
    """A six track with unknown alliance (gray) and no phase record must render fine."""
    h, w = 300, 500
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    six = [{"id": 9, "alliance": None, "xyxy": (10, 10, 50, 80),
            "centroid": (30, 45), "est": False, "hits": 1, "age": 1}]
    out = draw_scout_overlay(frame, _frame_state(), six, {}, {})
    assert out.shape == (h, w, 3)


def test_unavailable_metric_renders_na_not_zero():
    """A metric flagged available:False is formatted as the NA token, NEVER '0'."""
    # available metric -> its value; unavailable -> NA token, not 0.
    assert scout_render._fmt_metric({"value": 0.75, "available": True}) == "0.75"
    assert scout_render._fmt_metric({"value": None, "available": False}) == scout_render._NA
    assert scout_render._fmt_metric({"value": 0.0, "available": False}) == scout_render._NA
    assert scout_render._fmt_metric(None) == scout_render._NA
    # a plain available scalar still renders
    assert scout_render._fmt_metric(3, ndigits=0) == "3"


def test_alliance_color_strictly_by_alliance():
    """Colour is strictly red/blue/gray-by-alliance (never by track id)."""
    assert scout_render._alliance_color("red") == scout_render._RED_BGR
    assert scout_render._alliance_color("blue") == scout_render._BLUE_BGR
    assert scout_render._alliance_color(None) == scout_render._GRAY_BGR
    assert scout_render._alliance_color("purple") == scout_render._GRAY_BGR


def test_phase_label_splices_team_number():
    """The on-box label splices the resolved team number after the id token."""
    rec = {"alliance": "red", "label": "R3 SHOOTING", "team": 254, "phase": "SHOOTING"}
    assert scout_render._phase_label(3, rec, "red") == "R3 #254 SHOOTING"
    # no team -> id + phase only
    rec2 = {"alliance": "blue", "label": "B1 HOLDING(2)", "phase": "HOLDING"}
    assert scout_render._phase_label(1, rec2, "blue") == "B1 HOLDING(2)"
