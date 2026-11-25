"""Tracking-layer tests — pure synthetic detections, no model / no GPU.

Covers the three behaviours the layer must guarantee:
  * a robot moving linearly  -> exactly one stable ID, zero ID switches;
  * two balls crossing paths  -> both IDs preserved (no swap);
  * an occlusion gap          -> track coasts then re-acquires the SAME ID.
Detection objects are constructed directly; nothing reads a frame buffer.
"""
from __future__ import annotations

from frcscout.io.types import Detection
from frcscout.track.bytetrack import RobotTracker
from frcscout.track.ball_kalman import FlightTracker


FPS = 60.0


def _box(cx, cy, w, h, conf=0.9, frame=0, cls="robot"):
    return Detection(cls, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), conf, frame)


# ────────────────────────────── robot tracker ────────────────────────────────
def test_robot_linear_one_stable_id():
    """A single robot drifting across the frame -> exactly one track, one ID, no
    switches over the whole sequence."""
    trk = RobotTracker({"iou_match": 0.3, "max_age_frames": 30}, fps=FPS)
    seen_ids = set()
    last_id = None
    N = 60
    for f in range(N):
        cx = 100 + 5 * f
        cy = 300
        dets = [_box(cx, cy, 80, 80, conf=0.9, frame=f)]
        out = trk.update(f, dets)
        assert len(out) == 1, f"frame {f}: expected 1 track, got {len(out)}"
        tid = out[0].tid
        seen_ids.add(tid)
        if last_id is not None:
            assert tid == last_id, f"frame {f}: ID switch {last_id} -> {tid}"
        last_id = tid
    assert len(seen_ids) == 1, f"expected a single stable ID, saw {seen_ids}"


def test_robot_max_age_uses_real_fps():
    """max_age must be derived from seconds via the real fps, not a hardcoded frame
    count. The legacy `max_age_frames` (authored @30fps) should double at 60fps."""
    t30 = RobotTracker({"max_age_frames": 30}, fps=30.0)
    t60 = RobotTracker({"max_age_frames": 30}, fps=60.0)
    assert t30.max_age == 30
    assert t60.max_age == 60          # same 1.0s window, twice the frames
    # explicit seconds key wins and is fps-correct
    ts = RobotTracker({"max_age_s": 0.5}, fps=60.0)
    assert ts.max_age == 30


def test_robot_two_robots_distinct_ids():
    """Two well-separated robots -> two persistent, distinct IDs."""
    trk = RobotTracker({"iou_match": 0.3}, fps=FPS)
    ids_a, ids_b = set(), set()
    for f in range(40):
        a = _box(150 + 3 * f, 200, 80, 80, frame=f)
        b = _box(600 - 3 * f, 450, 80, 80, frame=f)
        out = trk.update(f, [a, b])
        assert len(out) == 2
        # the left/upper track vs right/lower track, by centroid
        out_sorted = sorted(out, key=lambda t: t.centroid_px[0])
        ids_a.add(out_sorted[0].tid)
        ids_b.add(out_sorted[1].tid)
    assert len(ids_a) == 1 and len(ids_b) == 1
    assert ids_a != ids_b


# ────────────────────────────── flight tracker ───────────────────────────────
def test_two_balls_crossing_preserve_ids():
    """Two balls on crossing linear paths keep their own IDs through the crossover."""
    trk = FlightTracker({"max_coast_frames": 6}, fps=FPS)
    # ball A goes left->right, ball B goes right->left; they meet near the middle
    id_a = id_b = None
    N = 30
    for f in range(N):
        ax, ay = 100 + 12 * f, 200 + 4 * f
        bx, by = 100 + 12 * (N - 1 - f), 200 + 4 * (N - 1 - f) + 0  # mirror in x
        # give them slightly different y tracks so NN can disambiguate at crossover
        a = _box(ax, ay, 16, 16, conf=0.8, frame=f, cls="ball")
        b = _box(bx, by + 30, 16, 16, conf=0.8, frame=f, cls="ball")
        out = trk.update(f, [a, b])
        assert len(out) == 2, f"frame {f}: expected 2 ball tracks, got {len(out)}"
        # identify each output by proximity to the true A / B positions
        by_pos = {}
        for t in out:
            da = abs(t.centroid_px[0] - ax) + abs(t.centroid_px[1] - ay)
            db = abs(t.centroid_px[0] - bx) + abs(t.centroid_px[1] - (by + 30))
            by_pos["A" if da < db else "B"] = t.tid
        if id_a is None:
            id_a, id_b = by_pos.get("A"), by_pos.get("B")
        else:
            # after warm-up, both IDs must persist with no swap
            if "A" in by_pos:
                assert by_pos["A"] == id_a, f"frame {f}: ball A id changed"
            if "B" in by_pos:
                assert by_pos["B"] == id_b, f"frame {f}: ball B id changed"
    assert id_a is not None and id_b is not None and id_a != id_b


def test_ball_occlusion_coast_then_reacquire():
    """A detection gap (occlusion) shorter than max_coast: the track coasts on its
    Kalman prediction, then re-acquires the SAME ID when the ball reappears."""
    trk = FlightTracker({"max_coast_frames": 6, "gate_px": 60.0}, fps=FPS)
    first_id = None
    vx = 10.0
    for f in range(20):
        cx = 100 + vx * f
        cy = 250
        gap = 8 <= f <= 11           # 4-frame occlusion, within max_coast
        dets = [] if gap else [_box(cx, cy, 16, 16, conf=0.8, frame=f, cls="ball")]
        out = trk.update(f, dets)
        if not gap:
            assert len(out) >= 1
            this = next(t for t in out
                        if abs(t.centroid_px[0] - cx) < 40 and abs(t.centroid_px[1] - cy) < 40)
            if first_id is None:
                first_id = this.tid
            else:
                assert this.tid == first_id, (
                    f"frame {f}: expected reacquire of id {first_id}, got {this.tid}")
    assert first_id is not None


def test_ball_flight_flag_px_fallback():
    """With no field_xy_m, a fast ball over enough px travel is flagged is_flight."""
    trk = FlightTracker(
        {"min_speed_px": 6.0, "min_travel_px": 60.0, "flight_min_frames": 2},
        fps=FPS)
    flagged = False
    for f in range(30):
        cx = 50 + 15 * f          # 15 px/frame >> 6 px/frame gate
        d = _box(cx, 300, 16, 16, conf=0.8, frame=f, cls="ball")
        out = trk.update(f, [d])
        if out and out[0].is_flight:
            flagged = True
    assert flagged, "fast, far-travelling ball should be marked is_flight"


def test_ball_slow_not_flight():
    """A slow ball below the speed gate never gets the flight flag."""
    trk = FlightTracker({"min_speed_px": 6.0, "min_travel_px": 60.0}, fps=FPS)
    any_flight = False
    for f in range(30):
        cx = 50 + 1 * f           # 1 px/frame, well under the gate
        d = _box(cx, 300, 16, 16, conf=0.8, frame=f, cls="ball")
        out = trk.update(f, [d])
        any_flight = any_flight or (out and out[0].is_flight)
    assert not any_flight
