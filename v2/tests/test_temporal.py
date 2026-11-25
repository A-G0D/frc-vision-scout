"""Temporal recovery + smoothing tests — pure synthetic Tracks, no model/GPU/numpy.

Covers the two behaviours the layer must guarantee:
  * gap fill: an object missing for a short run is re-emitted at a
    velocity-extrapolated position with the SAME id, and stops/gets culled once
    the gap exceeds the recovery window;
  * EMA smoothing: a jittery static box comes out with materially lower variance,
    and the smoother degrades to a pass-through at alpha == 1 / on first sample.

All inputs are constructed directly as `Track` objects; nothing reads a frame
buffer, a detector, or numpy.
"""
from __future__ import annotations

import math
import statistics

from frcscout.io.types import Track
from frcscout.refine.temporal import (
    BoxSmoother,
    RecoveryResult,
    TemporalRecovery,
    extrapolate_box,
)


FPS = 60.0


# ─────────────────────────────── helpers ─────────────────────────────────────
def _track(tid, cx, cy, w=80.0, h=80.0, cls="robot", age=0, hits=1,
           is_flight=False, alliance=None):
    box = (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
    return Track(
        tid=tid, cls=cls, xyxy=box,
        centroid_px=(cx, cy), age=age, hits=hits,
        is_flight=is_flight, alliance=alliance,
    )


def _centroid(xyxy):
    return ((xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0)


# ════════════════════════════ extrapolate_box ════════════════════════════════
def test_extrapolate_box_translates_preserving_size():
    box = (10.0, 20.0, 30.0, 60.0)          # w=20, h=40
    out = extrapolate_box(box, (5.0, -2.0), 3.0)
    assert out == (25.0, 14.0, 45.0, 54.0)
    # size preserved
    assert (out[2] - out[0], out[3] - out[1]) == (20.0, 40.0)


def test_extrapolate_box_fractional_and_zero():
    box = (0.0, 0.0, 10.0, 10.0)
    assert extrapolate_box(box, (4.0, 8.0), 0.5) == (2.0, 4.0, 12.0, 14.0)
    assert extrapolate_box(box, (4.0, 8.0), 0.0) == box        # no motion


# ════════════════════════════ BoxSmoother ════════════════════════════════════
def test_smoother_first_sample_passthrough():
    s = BoxSmoother(alpha=0.3)
    first = (1.0, 2.0, 3.0, 4.0)
    assert s.smooth("k", first) == first          # nothing to blend with yet
    assert s.peek("k") == first


def test_smoother_alpha_one_is_passthrough():
    s = BoxSmoother(alpha=1.0)
    s.smooth("k", (0.0, 0.0, 0.0, 0.0))
    out = s.smooth("k", (100.0, 100.0, 100.0, 100.0))
    assert out == (100.0, 100.0, 100.0, 100.0)    # fully follows measurement


def test_smoother_blend_math():
    s = BoxSmoother(alpha=0.5)
    s.smooth("k", (0.0, 0.0, 0.0, 0.0))
    out = s.smooth("k", (10.0, 20.0, 30.0, 40.0))
    assert out == (5.0, 10.0, 15.0, 20.0)         # midpoint at alpha=0.5


def test_smoother_invalid_alpha_rejected():
    for bad in (0.0, -0.1, 1.5):
        try:
            BoxSmoother(alpha=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"alpha={bad} should have raised")


def test_smoother_reduces_variance_on_jitter():
    """A static box plus deterministic zero-mean jitter -> the smoothed x1 signal
    has materially lower variance than the raw input."""
    s = BoxSmoother(alpha=0.3)
    base = 200.0
    # deterministic pseudo-jitter, zero-ish mean, +-6px
    jit = [6, -5, 4, -6, 5, -4, 6, -5, 3, -3, 5, -6, 4, -4, 6, -5, 2, -2, 5, -6]
    raw, sm = [], []
    for i, j in enumerate(jit):
        x1 = base + j
        raw.append(x1)
        out = s.smooth("o", (x1, x1, x1 + 80.0, x1 + 80.0))
        sm.append(out[0])
    # discard the warm-up sample (verbatim first value) for a fair comparison
    var_raw = statistics.pvariance(raw[1:])
    var_sm = statistics.pvariance(sm[1:])
    assert var_sm < 0.5 * var_raw, f"smoothing weak: raw={var_raw:.2f} sm={var_sm:.2f}"


def test_smoother_reset():
    s = BoxSmoother(alpha=0.5)
    s.smooth("a", (0.0, 0.0, 1.0, 1.0))
    s.smooth("b", (0.0, 0.0, 1.0, 1.0))
    s.reset("a")
    assert s.peek("a") is None and s.peek("b") is not None
    s.reset()                                       # clear all
    assert s.peek("b") is None


# ════════════════════════ TemporalRecovery: config ═══════════════════════════
def test_max_gap_uses_real_fps():
    """Legacy frame count is authored @30fps and rescales with the real fps;
    an explicit seconds key wins and is fps-correct. Mirrors the tracker test."""
    g30 = TemporalRecovery({"max_gap_frames": 6}, fps=30.0)
    g60 = TemporalRecovery({"max_gap_frames": 6}, fps=60.0)
    assert g30.max_gap == 6
    assert g60.max_gap == 12                        # same 0.2s window, twice the frames
    gs = TemporalRecovery({"max_gap_s": 0.5}, fps=60.0)
    assert gs.max_gap == 30


def test_default_config_smoothing_on():
    r = TemporalRecovery(fps=FPS)                   # no cfg at all
    assert r.smooth_enabled is True
    assert r.max_gap >= 1


# ════════════════════════ TemporalRecovery: gap fill ═════════════════════════
def test_gap_fill_extrapolates_linear_dropout():
    """A robot moving at a constant 5 px/frame, dropped out for frames 8-11,
    is re-emitted with the SAME id at the correct extrapolated position. Smoothing
    is off so we can assert exact pixel positions on the real frames."""
    r = TemporalRecovery({"max_gap_frames": 30, "smooth": False}, fps=FPS)
    vx, vy = 5.0, 0.0
    first_seen_id = 7
    N = 20
    for f in range(N):
        cx = 100.0 + vx * f
        cy = 300.0
        gap = 8 <= f <= 11                          # 4-frame dropout, within window
        tracks = [] if gap else [_track(first_seen_id, cx, cy)]
        res = r.step(f, tracks)

        # find our id in the emitted set
        mine = [t for t in res.tracks if t.tid == first_seen_id]
        assert len(mine) == 1, f"frame {f}: expected exactly one emit for the id"
        got_cx = mine[0].centroid_px[0]

        if gap:
            assert first_seen_id in res.filled_tids
            assert first_seen_id not in res.present_tids
            # extrapolation should land within 1px of the true linear position
            assert abs(got_cx - cx) < 1.0, (
                f"frame {f}: filled cx={got_cx:.2f} vs true {cx:.2f}")
        else:
            assert first_seen_id in res.present_tids
            assert first_seen_id not in res.filled_tids
            assert abs(got_cx - cx) < 1e-6


def test_gap_fill_keeps_id_and_does_not_inflate_hits():
    """During a fill, age advances but hits is held (a fill is not a measurement),
    matching how a coasting track is marked."""
    r = TemporalRecovery({"max_gap_frames": 30, "smooth": False}, fps=FPS)
    # warm up two real frames so velocity is known and hits=2
    r.step(0, [_track(3, 100.0, 200.0, age=0, hits=1)])
    r.step(1, [_track(3, 110.0, 200.0, age=1, hits=2)])
    # frame 2: missing -> filled
    res = r.step(2, [])
    assert res.filled_tids == (3,)
    filled = res.tracks[0]
    assert filled.tid == 3
    assert filled.hits == 2, "fill must not increment hits"
    # last observed age was 1 (frame 1); gap of 1 advances it to 2
    assert filled.age == 2, "fill should advance age by the gap"
    # velocity re-emitted in px/s (10 px/frame * 60 fps)
    assert math.isclose(filled.vel_mps[0], 10.0 * FPS, rel_tol=1e-6)


def test_gap_fill_stops_after_window_and_culls():
    """Once the gap exceeds max_gap, the id is no longer filled and is dropped."""
    # explicit 3-frame window via seconds, fps-correct
    r = TemporalRecovery({"max_gap_s": 3.0 / FPS, "smooth": False}, fps=FPS)
    assert r.max_gap == 3
    r.step(0, [_track(1, 0.0, 0.0)])
    r.step(1, [_track(1, 10.0, 0.0)])               # vel known
    # gaps 1,2,3 -> filled; gap 4 -> gone + culled
    for g in (1, 2, 3):
        res = r.step(1 + g, [])
        assert res.filled_tids == (1,), f"gap {g} should be filled"
    res4 = r.step(1 + 4, [])
    assert res4.filled_tids == (), "gap beyond window must not be filled"
    assert 1 not in r.tracked_ids(), "id should be culled past the window"


def test_reacquire_after_gap_resets_fill_and_continues():
    """After a fill run, a real detection re-anchors the track and zeroes the gap;
    subsequent fills extrapolate from the NEW observation."""
    r = TemporalRecovery({"max_gap_frames": 60, "smooth": False}, fps=FPS)
    r.step(0, [_track(2, 0.0, 0.0)])
    r.step(1, [_track(2, 10.0, 0.0)])               # vel = 10 px/frame
    r.step(2, [])                                   # filled at ~20
    # reappear at frame 3 at a fresh spot moving slower (vel updates via EMA)
    res3 = r.step(3, [_track(2, 25.0, 0.0)])
    assert 2 in res3.present_tids and res3.filled_tids == ()
    # now miss frame 5 -> fill should be anchored near the last real obs (~ >=25)
    res = r.step(5, [])
    assert res.filled_tids == (2,)
    assert res.tracks[0].centroid_px[0] >= 25.0


def test_two_ids_one_present_one_filled():
    """With two ids, a frame where only one is seen yields one present + one filled,
    preserving both ids distinctly."""
    r = TemporalRecovery({"max_gap_frames": 30, "smooth": False}, fps=FPS)
    r.step(0, [_track(10, 100.0, 100.0), _track(20, 500.0, 400.0)])
    r.step(1, [_track(10, 110.0, 100.0), _track(20, 490.0, 400.0)])
    # frame 2: only id 10 present
    res = r.step(2, [_track(10, 120.0, 100.0)])
    assert set(res.present_tids) == {10}
    assert set(res.filled_tids) == {20}
    ids = {t.tid for t in res.tracks}
    assert ids == {10, 20}
    # filled id 20 extrapolated leftward from 490 with vel -10 -> ~480
    f20 = next(t for t in res.tracks if t.tid == 20)
    assert abs(f20.centroid_px[0] - 480.0) < 1.0


def test_update_matches_step_tracks():
    """update() is the tracker-compatible facade over step().tracks."""
    r1 = TemporalRecovery({"smooth": False}, fps=FPS)
    r2 = TemporalRecovery({"smooth": False}, fps=FPS)
    t = [_track(1, 50.0, 50.0)]
    out = r1.update(0, t)
    res = r2.step(0, t)
    assert [x.tid for x in out] == [x.tid for x in res.tracks]
    assert isinstance(res, RecoveryResult)


def test_template_fields_preserved_on_fill():
    """cls / alliance / is_flight carry over onto a synthesized fill."""
    r = TemporalRecovery({"max_gap_frames": 30, "smooth": False}, fps=FPS)
    r.step(0, [_track(5, 0.0, 0.0, cls="ball", is_flight=True, alliance="red")])
    r.step(1, [_track(5, 12.0, 0.0, cls="ball", is_flight=True, alliance="red")])
    res = r.step(2, [])
    fill = res.tracks[0]
    assert fill.cls == "ball"
    assert fill.is_flight is True
    assert fill.alliance == "red"


def test_velocity_of_and_tracked_ids():
    r = TemporalRecovery({"smooth": False}, fps=FPS)
    assert r.velocity_of(99) is None
    r.step(0, [_track(1, 0.0, 0.0)])
    r.step(1, [_track(1, 6.0, 9.0)])
    vx, vy = r.velocity_of(1)
    assert math.isclose(vx, 6.0) and math.isclose(vy, 9.0)
    assert r.tracked_ids() == (1,)


# ════════════════════ smoothing inside the buffer ════════════════════════════
def test_buffer_smooths_emitted_boxes_but_extrapolates_from_raw():
    """With smoothing ON, emitted observed boxes are EMA-filtered, yet gap fill
    rides the true (un-smoothed) measurement, so a constant-velocity fill is still
    accurate."""
    r = TemporalRecovery(
        {"max_gap_frames": 30, "smooth": True, "smooth_alpha": 0.4}, fps=FPS)
    vx = 10.0
    last_emit_cx = None
    for f in range(6):
        cx = 100.0 + vx * f
        res = r.step(f, [_track(1, cx, 200.0)])
        last_emit_cx = res.tracks[0].centroid_px[0]
    # smoothed emit lags the true position (true=150 at f=5)
    assert last_emit_cx < 100.0 + vx * 5
    # now a gap: fill must extrapolate from the RAW last obs (f=5 -> cx=150),
    # not the lagged smoothed value, so gap-1 fill ~ 160
    res = r.step(6, [])
    assert res.filled_tids == (1,)
    assert abs(res.tracks[0].centroid_px[0] - 160.0) < 1.5


def test_smoothing_reduces_centroid_variance_through_buffer():
    """End-to-end: a static, jittery detection fed through the buffer comes out
    with lower centroid variance than it went in."""
    r = TemporalRecovery({"smooth": True, "smooth_alpha": 0.3}, fps=FPS)
    jit = [6, -5, 4, -6, 5, -4, 6, -5, 3, -3, 5, -6, 4, -4, 6, -5]
    raw, out = [], []
    for f, j in enumerate(jit):
        cx = 300.0 + j
        raw.append(cx)
        res = r.step(f, [_track(1, cx, 300.0)])
        out.append(res.tracks[0].centroid_px[0])
    assert statistics.pvariance(out[1:]) < 0.6 * statistics.pvariance(raw[1:])
