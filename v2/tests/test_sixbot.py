"""SixRobotTracker tests — pure synthetic detections, no model / no GPU.

The structural prior under test: every FRC match has EXACTLY six robots (3 red, 3
blue). The tracker must therefore
  * converge to six stable tracks (<=3 per alliance) with monotonic IDs,
  * NEVER drop a track when a robot is occluded / leaves frame — instead coast it on
    the Kalman prediction and flag est=True, re-acquiring the SAME id on reappearance,
  * gate association by alliance (a red detection can never bind a blue track),
  * cap births at three per alliance (a phantom 4th detection cannot mint a 7th track),
  * stay importable and correct even if numpy / scipy are absent (greedy fallback).
Detection objects are constructed directly; nothing reads a frame buffer.
"""
from __future__ import annotations

import math

import pytest

from frcscout.io.types import Detection
from frcscout.refine.sixbot import (
    SixRobotTracker,
    HAS_SCIPY,
    _CVTrack,
    _iou,
)


FPS = 60.0
W = H = 80.0  # robot box size in px


def _box(cx, cy, conf=0.9, frame=0):
    return Detection("robot", (cx - W / 2, cy - H / 2, cx + W / 2, cy + H / 2), conf, frame)


def _six_layout(f: int):
    """Six well-separated robots (3 red top row, 3 blue bottom row) drifting right.
    Returns (detections, alliances) for frame f."""
    dets, allys = [], []
    for k in range(3):
        dets.append(_box(150 + 200 * k + 2 * f, 200, frame=f))
        allys.append("red")
    for k in range(3):
        dets.append(_box(150 + 200 * k + 2 * f, 500, frame=f))
        allys.append("blue")
    return dets, allys


def _ids_by_slot(out, alliance):
    """Map left->right slot index to track id for one alliance, by centroid x."""
    rows = sorted((d for d in out if d["alliance"] == alliance),
                  key=lambda d: d["centroid"][0])
    return [d["id"] for d in rows]


# ────────────────────────────── basic structure ──────────────────────────────
def test_six_robots_six_stable_tracks():
    """3 red + 3 blue, all visible every frame -> exactly six tracks with six
    distinct, perfectly stable IDs and no est flags."""
    trk = SixRobotTracker({}, fps=FPS)
    red_slots = blue_slots = None
    for f in range(60):
        dets, allys = _six_layout(f)
        out = trk.update(f, dets, allys)
        assert len(out) == 6, f"frame {f}: expected 6 tracks, got {len(out)}"
        assert sum(d["alliance"] == "red" for d in out) == 3
        assert sum(d["alliance"] == "blue" for d in out) == 3
        assert all(d["est"] is False for d in out), f"frame {f}: unexpected coast"

        r, b = _ids_by_slot(out, "red"), _ids_by_slot(out, "blue")
        if red_slots is None:
            red_slots, blue_slots = r, b
        else:
            assert r == red_slots, f"frame {f}: red ID switch {red_slots} -> {r}"
            assert b == blue_slots, f"frame {f}: blue ID switch {blue_slots} -> {b}"

    all_ids = red_slots + blue_slots
    assert len(set(all_ids)) == 6, f"IDs not distinct: {all_ids}"


def test_output_schema():
    """Each emitted dict carries exactly the documented keys with sane types."""
    trk = SixRobotTracker({}, fps=FPS)
    dets, allys = _six_layout(0)
    out = trk.update(0, dets, allys)
    assert out, "expected output"
    for d in out:
        assert set(d) == {"id", "alliance", "xyxy", "centroid", "est",
                          "confident", "hits", "age"}
        assert isinstance(d["id"], int)
        assert d["alliance"] in ("red", "blue")
        assert len(d["xyxy"]) == 4 and len(d["centroid"]) == 2
        assert isinstance(d["est"], bool)
        assert isinstance(d["confident"], bool)
        assert isinstance(d["hits"], int) and isinstance(d["age"], int)


# ──────────────────────── coasting through occlusion ─────────────────────────
def test_occlusion_coasts_then_reacquires_same_id():
    """One robot disappears for several frames. The track is NOT dropped: it stays
    in the output flagged est=True, and re-acquires the SAME id when it returns."""
    trk = SixRobotTracker({}, fps=FPS)
    target_id = None
    coasted = False
    GAP = range(20, 28)  # 8-frame disappearance of the first red robot

    for f in range(50):
        dets, allys = _six_layout(f)
        # drop the first red detection (index 0) during the gap
        if f in GAP:
            dets = dets[1:]
            allys = allys[1:]
        out = trk.update(f, dets, allys)

        # always still six tracks — the prior forbids dropping a robot
        assert len(out) == 6, f"frame {f}: expected 6 tracks, got {len(out)}"

        # identify the leftmost red track (the one that vanishes)
        leftmost = min((d for d in out if d["alliance"] == "red"),
                       key=lambda d: d["centroid"][0])
        if target_id is None:
            target_id = leftmost["id"]
        else:
            assert leftmost["id"] == target_id, (
                f"frame {f}: leftmost-red id changed {target_id} -> {leftmost['id']}")

        if f in GAP:
            assert leftmost["est"] is True, f"frame {f}: coasting track must be est=True"
            coasted = True
        elif f > max(GAP):
            assert leftmost["est"] is False, f"frame {f}: should be measured again"

    assert coasted and target_id is not None


def test_coasting_track_keeps_moving_forward():
    """A coasting track keeps its velocity (constant-velocity prediction), so its
    estimated centroid advances rather than freezing."""
    trk = SixRobotTracker({}, fps=FPS)
    # warm up so the Kalman learns a rightward velocity
    for f in range(10):
        dets, allys = _six_layout(f)
        trk.update(f, dets, allys)

    # find the leftmost-red id and its x just before the gap
    out = trk.update(10, *_six_layout(10))
    leftmost = min((d for d in out if d["alliance"] == "red"),
                   key=lambda d: d["centroid"][0])
    tid, x_before = leftmost["id"], leftmost["centroid"][0]

    # now hide that robot for a few frames; track should coast forward in x
    last_x = x_before
    for f in range(11, 16):
        dets, allys = _six_layout(f)
        dets, allys = dets[1:], allys[1:]   # drop leftmost red
        out = trk.update(f, dets, allys)
        d = next(d for d in out if d["id"] == tid)
        assert d["est"] is True
        assert d["centroid"][0] >= last_x - 1e-6, "coasting centroid moved backwards"
        last_x = d["centroid"][0]
    assert last_x > x_before, "coasting track did not advance with its velocity"


# ─────────────────────────── alliance gating ─────────────────────────────────
def test_alliance_gate_no_cross_binding():
    """A red detection must never bind to (or steal the id of) a blue track, even when
    a red robot sits spatially right where a blue one was."""
    trk = SixRobotTracker({}, fps=FPS)
    # establish all six
    for f in range(5):
        trk.update(f, *_six_layout(f))
    red_ids = set(d["id"] for d in trk.update(5, *_six_layout(5)) if d["alliance"] == "red")
    blue_ids = set(d["id"] for d in trk.update(6, *_six_layout(6)) if d["alliance"] == "blue")
    assert red_ids.isdisjoint(blue_ids)

    # Now feed a RED detection exactly at a blue track's location. It must NOT match the
    # blue track (different alliance); blue coasts (est) and a new/own red handles it.
    out = trk.update(7, *_six_layout(7))
    blue_track = min((d for d in out if d["alliance"] == "blue"),
                     key=lambda d: d["centroid"][0])
    bx, by = blue_track["centroid"]

    # one frame: only a single red detection, placed on top of that blue track
    out = trk.update(8, [_box(bx, by, frame=8)], ["red"])
    # every track id stays within its own alliance set; no red id appears among blues
    for d in out:
        if d["alliance"] == "blue":
            assert d["id"] in blue_ids, "a non-blue id leaked into the blue alliance"
        if d["alliance"] == "red":
            assert d["id"] not in blue_ids, "a red track took a blue id (cross-binding!)"
    # the blue tracks are unmatched this frame -> all coasting
    assert all(d["est"] for d in out if d["alliance"] == "blue")


# ───────────────────────────── 3-per-alliance cap ────────────────────────────
def test_cap_three_per_alliance():
    """A phantom 4th red detection must not create a 7th track; the cap is 3/alliance."""
    trk = SixRobotTracker({}, fps=FPS)
    for f in range(5):
        trk.update(f, *_six_layout(f))

    # frame with a spurious 4th red robot far from the others
    dets, allys = _six_layout(5)
    dets.append(_box(1400, 200, conf=0.95, frame=5))
    allys.append("red")
    out = trk.update(5, dets, allys)

    assert len(out) <= 6, f"cap violated: {len(out)} tracks"
    assert sum(d["alliance"] == "red" for d in out) == 3
    assert sum(d["alliance"] == "blue" for d in out) == 3


def test_unknown_alliance_ignored():
    """Detections with no alliance label are ignored (prior is per-alliance)."""
    trk = SixRobotTracker({}, fps=FPS)
    out = trk.update(0, [_box(300, 300, frame=0)], [None])
    assert out == []
    # and with alliances omitted entirely, still nothing is tracked
    out = trk.update(1, [_box(300, 300, frame=1)])
    assert out == []


def test_births_capped_from_empty():
    """Feeding four simultaneous red detections from cold start yields only three
    red tracks (birth cap), never four."""
    trk = SixRobotTracker({}, fps=FPS)
    dets = [_box(150, 200), _box(450, 200), _box(750, 200), _box(1050, 200)]
    allys = ["red"] * 4
    out = trk.update(0, dets, allys)
    assert sum(d["alliance"] == "red" for d in out) == 3
    assert len(out) == 3


# ───────────────────────── eviction conserves the count ──────────────────────
def test_long_gone_track_evicted_for_new_robot():
    """If a robot leaves for longer than the coast-grace window and a genuinely new
    robot appears in a 4th slot, the stale track is evicted so the count stays 3."""
    cfg = {"coast_grace_s": 0.1}              # 6 frames @60fps
    trk = SixRobotTracker(cfg, fps=FPS)
    assert trk.coast_grace == 6

    # three red robots present for a while
    def three_red(f, drop_first=False):
        dets = [_box(150 + 2 * f, 200), _box(450 + 2 * f, 200), _box(750 + 2 * f, 200)]
        allys = ["red", "red", "red"]
        if drop_first:
            dets, allys = dets[1:], allys[1:]
        return dets, allys

    for f in range(10):
        trk.update(f, *three_red(f))
    stale_id = min((d for d in trk.update(10, *three_red(10)) if d["alliance"] == "red"),
                   key=lambda d: d["centroid"][0])["id"]

    # leftmost red leaves for well beyond the 6-frame grace window
    for f in range(11, 25):
        trk.update(f, *three_red(f, drop_first=True))

    # now a brand-new red robot appears in a distant 4th location
    dets, allys = three_red(25, drop_first=True)
    dets.append(_box(1600, 200, frame=25))
    allys.append("red")
    out = trk.update(25, dets, allys)

    red = [d for d in out if d["alliance"] == "red"]
    assert len(red) == 3, "alliance must still hold exactly three"
    assert stale_id not in {d["id"] for d in red}, "stale, long-gone track should be evicted"


# ─────────────────────────── kalman / helpers units ──────────────────────────
def test_iou_helper():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert 0.0 < _iou((0, 0, 10, 10), (5, 5, 15, 15)) < 1.0


def test_cvtrack_predict_advances_centroid():
    """A track given a rightward velocity predicts forward on coast."""
    t = _CVTrack(0, "red", _box(100, 100, frame=0), 0)
    # feed a couple of measurements moving right to build velocity
    t.predict(); t.update(_box(110, 100, frame=1), 1)
    t.predict(); t.update(_box(120, 100, frame=2), 2)
    x_before = t.centroid[0]
    t.predict()  # coast one frame, no update
    assert t.centroid[0] > x_before
    assert t.est is True
    assert t.time_since_update == 1


def test_timing_uses_real_fps():
    """coast_grace must be derived from seconds via the real fps, not hardcoded."""
    t30 = SixRobotTracker({"coast_grace_s": 1.0}, fps=30.0)
    t60 = SixRobotTracker({"coast_grace_s": 1.0}, fps=60.0)
    assert t30.coast_grace == 30
    assert t60.coast_grace == 60        # same 1.0s window, twice the frames


# ───────────────────────── greedy-fallback equivalence ───────────────────────
def test_greedy_matches_scipy_on_simple_problem():
    """The greedy fallback must agree with scipy on the small, well-separated 3x3
    assignment this tracker actually poses (guards the no-scipy path)."""
    trk = SixRobotTracker({}, fps=FPS)
    # three tracks at x=100,300,500 ; three dets near them but shuffled
    tracks = [_CVTrack(i, "red", _box(100 + 200 * i, 200, frame=0), 0) for i in range(3)]
    dets = [_box(305, 200, frame=1), _box(98, 200, frame=1), _box(502, 200, frame=1)]
    cost = [[trk._cost(t, d) for d in dets] for t in tracks]

    rg, cg = SixRobotTracker._greedy(cost, 3, 3)
    greedy_pairs = set(zip(rg, cg))
    expected = {(0, 1), (1, 0), (2, 2)}      # nearest assignment
    assert greedy_pairs == expected

    if HAS_SCIPY:
        rs, cs = SixRobotTracker._hungarian_scipy(cost, 3, 3)
        assert set(zip(rs, cs)) == expected
