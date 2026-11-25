"""Possession state-machine tests — pure synthetic, no model / cv2 / GPU.

We feed hand-built FrameState objects with field-projected robot/ball tracks and
assert the per-ball lifecycle + per-robot scouting counters:
  * a free ball converging on a robot then vanishing in its zone
        -> intake_attempt + intake_success, robot held_count++;
  * a possessed ball launching toward a hub it enters
        -> shot_attempt + shot_success;
  * a shot that never enters the hub and times out -> shot_attempt + FAIL;
  * the success-rate counters compute correctly.

A tiny stub Field supplies only ``in_hub`` (the lone geometry call the tracker
makes), so no YAML / FieldModel is needed.
"""
from __future__ import annotations

import json
import math

from frcscout.io.types import CameraPose, FrameState, Track
from frcscout.fuse.possession import PossessionTracker, robot_size


FPS = 60.0


class StubField:
    """Minimal field: a single circular HUB per alliance in meters."""
    def __init__(self, hubs=None):
        self.hubs = hubs or {
            "blue": {"center_m": [3.2, 4.1], "radius_m": 0.8},
            "red": {"center_m": [13.26, 4.1], "radius_m": 0.8},
        }

    def in_hub(self, xy, alliance, tol_m=0.0):
        hub = self.hubs.get(alliance)
        if not hub:
            return False
        cx, cy = hub["center_m"]
        r = hub["radius_m"] + tol_m
        return (xy[0] - cx) ** 2 + (xy[1] - cy) ** 2 <= r * r


def _robot(tid, xy, alliance="blue"):
    return Track(tid=tid, cls="robot", xyxy=(0, 0, 10, 10),
                 centroid_px=(0, 0), field_xy_m=xy, alliance=alliance)


def _ball(tid, xy, flight=False):
    return Track(tid=tid, cls="ball", xyxy=(0, 0, 5, 5),
                 centroid_px=(0, 0), field_xy_m=xy, is_flight=flight)


_VALID_POSE = CameraPose(H=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], valid=True)


def _state(frame_idx, robots, inventory=None, flights=None, pose=None):
    return FrameState(
        frame_idx=frame_idx, t_s=frame_idx / FPS,
        robots=robots,
        balls_inventory=inventory or [],
        balls_flight=flights or [],
        pose=pose)


# ─────────────────────────────── intake ──────────────────────────────────────
def test_intake_attempt_and_success():
    """A free ball closes on a stationary robot's intake zone, then vanishes
    inside it -> intake_attempt then intake_success, held_count -> 1."""
    pt = PossessionTracker(StubField(), {"intake_zone_m": 0.6,
                                          "converge_min_s": 0.0,
                                          "vanish_grace_s": 0.1}, fps=FPS)
    robot_xy = (5.0, 4.0)
    saw_attempt = saw_success = False
    f = 0
    # ball approaches from 3m out, closing 0.3m/frame, robot fixed
    for dist in [3.0, 2.0, 1.0, 0.5, 0.3]:
        bx = robot_xy[0] - dist
        evs = pt.step(_state(f, [_robot(1, robot_xy)], inventory=[_ball(50, (bx, 4.0))]))
        for e in evs:
            if e.type == "intake_attempt" and e.robot_id == 1:
                saw_attempt = True
        f += 1
    # now the ball disappears (consumed by the robot); step a few empty-inventory frames
    for _ in range(int(0.1 * FPS) + 2):
        evs = pt.step(_state(f, [_robot(1, robot_xy)], inventory=[]))
        for e in evs:
            if e.type == "intake_success" and e.robot_id == 1:
                saw_success = True
        f += 1

    assert saw_attempt, "expected an intake_attempt"
    assert saw_success, "expected an intake_success after the ball vanished in-zone"
    assert pt.robots[1].intake_attempts == 1
    assert pt.robots[1].intake_successes == 1
    assert pt.robots[1].held_count == 1
    assert pt.robots[1].intake_success_rate == 1.0


def _possess(pt, robot_xy, robot_id=1, alliance="blue", ball_id=50, start_f=0):
    """Drive a ball into possession; returns the next free frame index."""
    f = start_f
    for dist in [3.0, 2.0, 1.0, 0.5, 0.3]:
        bx = robot_xy[0] - dist
        pt.step(_state(f, [_robot(robot_id, robot_xy, alliance)],
                       inventory=[_ball(ball_id, (bx, robot_xy[1]))]))
        f += 1
    for _ in range(int(pt.vanish_grace_frames) + 2):
        pt.step(_state(f, [_robot(robot_id, robot_xy, alliance)], inventory=[]))
        f += 1
    return f


# ─────────────────────────────── shots ───────────────────────────────────────
def test_shot_attempt_and_success_into_hub():
    """A possessed ball launches as a flight track at the robot, and the arc
    enters the blue HUB -> shot_attempt then shot_success."""
    field = StubField()
    pt = PossessionTracker(field, {"intake_zone_m": 0.6, "converge_min_s": 0.0,
                                   "vanish_grace_s": 0.1, "shot_origin_m": 1.0,
                                   "hub_tol_m": 0.3, "shot_lookahead_s": 2.0}, fps=FPS)
    robot_xy = (5.0, 4.0)
    f = _possess(pt, robot_xy)
    assert pt.robots[1].held_count == 1

    # a flight track is born AT the robot (origin within shot_origin_m)
    saw_attempt = False
    evs = pt.step(_state(f, [_robot(1, robot_xy)],
                         flights=[_ball(90, (5.2, 4.0), flight=True)]))
    saw_attempt = any(e.type == "shot_attempt" and e.robot_id == 1 for e in evs)
    assert saw_attempt, "expected a shot_attempt at the possessing robot"
    assert pt.robots[1].shot_attempts == 1
    assert pt.robots[1].held_count == 0, "shot consumes a held ball"
    f += 1

    # the flight arc travels into the blue hub center -> success on a later frame
    saw_success = False
    hub_center = tuple(field.hubs["blue"]["center_m"])
    for step in range(8):
        # interpolate the flight toward the hub center
        fx = 5.2 + (hub_center[0] - 5.2) * (step + 1) / 8.0
        fy = 4.0 + (hub_center[1] - 4.0) * (step + 1) / 8.0
        evs = pt.step(_state(f, [_robot(1, robot_xy)],
                             flights=[_ball(90, (fx, fy), flight=True)]))
        if any(e.type == "shot_success" and e.robot_id == 1 for e in evs):
            saw_success = True
        f += 1

    assert saw_success, "arc entering the HUB should yield shot_success"
    assert pt.robots[1].shot_successes == 1
    assert pt.robots[1].shot_success_rate == 1.0


def test_shot_miss_times_out():
    """A possessed ball launches but the arc never enters the hub and there is no
    overlay confirmation -> shot_attempt counted, NO shot_success (a miss)."""
    field = StubField()
    pt = PossessionTracker(field, {"intake_zone_m": 0.6, "converge_min_s": 0.0,
                                   "vanish_grace_s": 0.1, "shot_origin_m": 1.0,
                                   "hub_tol_m": 0.3, "shot_lookahead_s": 0.2}, fps=FPS)
    robot_xy = (5.0, 4.0)
    f = _possess(pt, robot_xy)

    # launch a flight that flies AWAY from the hub (toward +y, never near (3.2,4.1))
    pt.step(_state(f, [_robot(1, robot_xy)],
                   flights=[_ball(90, (5.2, 4.0), flight=True)]))
    f += 1
    assert pt.robots[1].shot_attempts == 1

    saw_success = False
    # fly far away well past the lookahead window
    for step in range(int(0.2 * FPS) + 4):
        evs = pt.step(_state(f, [_robot(1, robot_xy)],
                             flights=[_ball(90, (8.0 + step, 8.0), flight=True)]))
        if any(e.type == "shot_success" for e in evs):
            saw_success = True
        f += 1

    assert not saw_success, "a shot that never enters the hub must not succeed"
    assert pt.robots[1].shot_attempts == 1
    assert pt.robots[1].shot_successes == 0
    assert pt.robots[1].shot_success_rate == 0.0


def test_shot_success_via_overlay_when_arc_inconclusive():
    """When the arc test is inconclusive, an overlay score increment in the
    window confirms the shot (authoritative count path)."""
    field = StubField()
    pt = PossessionTracker(field, {"intake_zone_m": 0.6, "converge_min_s": 0.0,
                                   "vanish_grace_s": 0.1, "shot_origin_m": 1.0,
                                   "shot_lookahead_s": 2.0, "score_window_s": 1.0},
                           fps=FPS)
    robot_xy = (5.0, 4.0)
    f = _possess(pt, robot_xy, alliance="blue")

    # launch the shot
    pt.step(_state(f, [_robot(1, robot_xy, "blue")],
                   flights=[_ball(90, (5.2, 4.0), flight=True)]))
    shot_frame = f
    f += 1
    # overlay score increments for blue shortly after the shot
    pt.note_overlay_score(shot_frame + 5, "blue")

    saw_success = False
    for step in range(10):
        # flight stays away from the hub; only the overlay can confirm
        evs = pt.step(_state(f, [_robot(1, robot_xy, "blue")],
                             flights=[_ball(90, (9.0, 7.0), flight=True)]))
        if any(e.type == "shot_success" and e.robot_id == 1 for e in evs):
            saw_success = True
        f += 1
    assert saw_success, "overlay score increment should confirm the shot"
    assert pt.robots[1].shot_successes == 1


def test_rates_with_mixed_outcomes():
    """Two intake attempts (one success), counters + rates compute correctly."""
    pt = PossessionTracker(StubField(), {"intake_zone_m": 0.6, "converge_min_s": 0.0,
                                         "vanish_grace_s": 0.1,
                                         "intake_dedup_s": 0.0}, fps=FPS)
    robot_xy = (5.0, 4.0)
    # success #1: full possess
    f = _possess(pt, robot_xy, ball_id=50, start_f=0)
    assert pt.robots[1].intake_successes == 1

    # attempt #2 that FAILS: ball enters zone (attempt) then drifts back out
    for dist in [3.0, 2.0, 1.0, 0.4]:
        bx = robot_xy[0] - dist
        pt.step(_state(f, [_robot(1, robot_xy)], inventory=[_ball(60, (bx, 4.0))]))
        f += 1
    # now move it back OUT of the zone (fails) — stays detected, no vanish
    for dist in [1.5, 2.5, 3.5, 4.5]:
        bx = robot_xy[0] - dist
        pt.step(_state(f, [_robot(1, robot_xy)], inventory=[_ball(60, (bx, 4.0))]))
        f += 1

    rs = pt.robots[1]
    assert rs.intake_attempts == 2, f"expected 2 attempts, got {rs.intake_attempts}"
    assert rs.intake_successes == 1
    assert math.isclose(rs.intake_success_rate, 0.5)


def test_stats_dict_shape():
    pt = PossessionTracker(StubField(), {"converge_min_s": 0.0,
                                         "vanish_grace_s": 0.1}, fps=FPS)
    _possess(pt, (5.0, 4.0))
    d = pt.stats_dict()
    assert 1 in d
    row = d[1]
    for key in ("intake_attempts", "intake_successes", "intake_success_rate",
                "shot_attempts", "shot_successes", "shot_success_rate",
                "held_count", "possession_time_s"):
        assert key in row


# ──────────────────── NEW: radius scaling + opportunity ───────────────────────
def _robot_m(tid, xy, size_m, alliance="blue"):
    """A robot whose bbox has been projected to a field-meter extent (size_m), via
    the geometry layer's ``field_wh_m`` attribute the tracker reads."""
    r = Track(tid=tid, cls="robot", xyxy=(0, 0, 10, 10),
              centroid_px=(0, 0), field_xy_m=xy, alliance=alliance)
    r.field_wh_m = (size_m, size_m)   # dynamic attr the tracker honours (meters)
    return r


def test_intake_radius_scales_with_robot_size():
    """intake_radius = intake_k * robot_size_m (when size is in meters), so a
    bigger robot has a proportionally bigger catchment."""
    pt = PossessionTracker(StubField(), {"intake_k": 1.2, "intake_zone_m": 0.3}, fps=FPS)
    small = _robot_m(1, (5.0, 4.0), size_m=0.5)
    big = _robot_m(2, (5.0, 4.0), size_m=1.0)
    r_small = pt.intake_radius(small)
    r_big = pt.intake_radius(big)
    assert math.isclose(r_small, 1.2 * 0.5)
    assert math.isclose(r_big, 1.2 * 1.0)
    assert r_big > r_small, "bigger robot -> bigger intake radius"
    # robot_size reports the field-meter extent and its units tag
    size, units = robot_size(big)
    assert math.isclose(size, 1.0) and units == "m"


def test_intake_radius_floor_when_no_field_size():
    """With no projected field size (only a px bbox), scaling px against meter ball
    distances is nonsense, so the radius falls back to the meter floor."""
    pt = PossessionTracker(StubField(), {"intake_k": 1.2, "intake_zone_m": 0.6}, fps=FPS)
    px_only = _robot(1, (5.0, 4.0))   # has xyxy px bbox, no field_wh_m
    assert math.isclose(pt.intake_radius(px_only), 0.6)
    size, units = robot_size(px_only)
    assert units == "px"


def test_balls_in_radius_counts_only_inside_scaled_radius():
    """balls_in_radius counts only balls whose centre lies within the robot's scaled
    intake_radius; balls just outside it are excluded; peak is tracked in stats."""
    pt = PossessionTracker(StubField(), {"intake_k": 1.0, "intake_zone_m": 0.1,
                                         "converge_min_s": 0.0, "intake_dedup_s": 5.0},
                           fps=FPS)
    robot = _robot_m(1, (5.0, 4.0), size_m=1.0)   # radius = 1.0 m
    inside = [
        _ball(50, (5.0, 4.5)),    # 0.5 m  -> in
        _ball(51, (5.9, 4.0)),    # 0.9 m  -> in
    ]
    outside = [
        _ball(52, (5.0, 5.2)),    # 1.2 m  -> out
        _ball(53, (7.0, 4.0)),    # 2.0 m  -> out
    ]
    n = pt.balls_in_radius(robot, inside + outside)
    assert n == 2, f"expected 2 balls inside the 1.0m radius, got {n}"

    # peak is recorded across frames via step()
    pt.step(_state(0, [robot], inventory=inside + outside))
    assert pt.robots[1].balls_in_radius_peak == 2


def test_time_to_intake_measured_from_radius_entry():
    """A ball entering the radius then vanishing inside it yields a time_to_intake
    equal to (possession time - radius-entry time)."""
    pt = PossessionTracker(StubField(), {"intake_k": 1.0, "intake_zone_m": 0.1,
                                         "converge_min_s": 0.0, "vanish_grace_s": 0.05,
                                         "intake_dedup_s": 0.0}, fps=FPS)
    robot_xy = (5.0, 4.0)
    f = 0
    # ball approaches; radius = 1.0m. It crosses inside at dist 0.9 (frame 2 below).
    # frames: dist 3.0(out), 2.0(out), 0.9(IN -> clock starts), 0.5(in), 0.3(in)
    dists = [3.0, 2.0, 0.9, 0.5, 0.3]
    enter_frame = None
    for i, dist in enumerate(dists):
        bx = robot_xy[0] - dist
        pt.step(_state(f, [_robot_m(1, robot_xy, size_m=1.0)],
                       inventory=[_ball(50, (bx, 4.0))]))
        if dist <= 1.0 and enter_frame is None:
            enter_frame = f
        f += 1
    last_seen = f - 1
    # ball vanishes -> success after the grace
    for _ in range(int(pt.vanish_grace_frames) + 2):
        pt.step(_state(f, [_robot_m(1, robot_xy, size_m=1.0)], inventory=[]))
        f += 1

    rs = pt.robots[1]
    assert rs.intake_successes == 1
    assert len(rs.time_to_intake_s) == 1
    expected = (last_seen - enter_frame) / FPS
    assert math.isclose(rs.time_to_intake_s[0], expected, abs_tol=1e-6), \
        f"time_to_intake {rs.time_to_intake_s[0]} != {expected}"
    assert math.isclose(rs.mean_time_to_intake_s, expected, abs_tol=1e-6)


def test_balls_near_but_never_taken_attempts_without_success():
    """Balls that approach (attempt) but drift away without vanishing -> attempts
    accrue with no successes, so intake_success_rate < 1."""
    pt = PossessionTracker(StubField(), {"intake_k": 1.0, "intake_zone_m": 0.1,
                                         "converge_min_s": 0.0, "vanish_grace_s": 0.1,
                                         "intake_dedup_s": 0.0}, fps=FPS)
    robot_xy = (5.0, 4.0)
    f = 0
    # approach into the 1.0m radius (attempt), then retreat out — never vanishes
    for dist in [3.0, 2.0, 0.8, 0.5]:        # closing into radius -> attempt
        bx = robot_xy[0] - dist
        pt.step(_state(f, [_robot_m(1, robot_xy, size_m=1.0)],
                       inventory=[_ball(60, (bx, 4.0))]))
        f += 1
    for dist in [1.5, 2.5, 3.5, 4.5]:        # retreat well past radius
        bx = robot_xy[0] - dist
        pt.step(_state(f, [_robot_m(1, robot_xy, size_m=1.0)],
                       inventory=[_ball(60, (bx, 4.0))]))
        f += 1
    rs = pt.robots[1]
    assert rs.intake_attempts >= 1
    assert rs.intake_successes == 0
    assert rs.intake_success_rate < 1.0


# ────────────────────────── NEW: shot accuracy / made ─────────────────────────
def test_shot_accuracy_with_one_made_one_missed():
    """One shot enters the HUB (made), one times out away from it (missed) -> both
    are resolvable, shot_accuracy = 1/2."""
    field = StubField()
    pt = PossessionTracker(field, {"intake_k": 1.0, "intake_zone_m": 0.6,
                                   "converge_min_s": 0.0, "vanish_grace_s": 0.1,
                                   "shot_origin_m": 1.0, "hub_tol_m": 0.3,
                                   "shot_lookahead_s": 0.3, "intake_dedup_s": 0.0},
                          fps=FPS)
    robot_xy = (5.0, 4.0)
    hub_center = tuple(field.hubs["blue"]["center_m"])

    # ── made shot (valid pose so the arc is trustworthy/judgeable) ──
    f = _possess(pt, robot_xy, ball_id=50, start_f=0)
    pt.step(_state(f, [_robot(1, robot_xy)],
                   flights=[_ball(90, (5.2, 4.0), flight=True)], pose=_VALID_POSE))
    f += 1
    for step in range(8):
        fx = 5.2 + (hub_center[0] - 5.2) * (step + 1) / 8.0
        fy = 4.0 + (hub_center[1] - 4.0) * (step + 1) / 8.0
        pt.step(_state(f, [_robot(1, robot_xy)],
                       flights=[_ball(90, (fx, fy), flight=True)], pose=_VALID_POSE))
        f += 1
    assert pt.robots[1].shots_made == 1

    # ── missed shot (times out away from hub; pose valid -> judgeable miss) ──
    f = _possess(pt, robot_xy, ball_id=51, start_f=f)
    pt.step(_state(f, [_robot(1, robot_xy)],
                   flights=[_ball(91, (5.2, 4.0), flight=True)], pose=_VALID_POSE))
    f += 1
    for step in range(int(0.3 * FPS) + 4):
        pt.step(_state(f, [_robot(1, robot_xy)],
                       flights=[_ball(91, (8.0 + step, 8.0), flight=True)], pose=_VALID_POSE))
        f += 1

    rs = pt.robots[1]
    assert rs.shot_attempts == 2
    assert rs.shots_made == 1
    assert rs.shots_resolvable == 2, "both shots judgeable via geometry"
    assert math.isclose(rs.shot_accuracy, 0.5)
    assert math.isclose(rs.miss_rate, 0.5)


def test_shot_accuracy_none_without_localization():
    """With no field pose AND no overlay pulse, a shot's outcome is UNKNOWN. Here the
    flight track has no field_xy_m so the arc can't be tested -> shot_accuracy None,
    never a fabricated 0."""
    class NoHubField:
        # in_hub never callable on a None position; geometry untestable
        def in_hub(self, xy, alliance, tol_m=0.0):
            return False

    pt = PossessionTracker(NoHubField(), {"intake_k": 1.0, "intake_zone_m": 0.6,
                                          "converge_min_s": 0.0, "vanish_grace_s": 0.1,
                                          "shot_origin_m": 1.0, "shot_lookahead_s": 0.2,
                                          "intake_dedup_s": 0.0}, fps=FPS)
    robot_xy = (5.0, 4.0)
    f = _possess(pt, robot_xy)
    # flight with NO field_xy_m on the resolution frames -> geom untestable, no overlay
    fl = Track(tid=92, cls="ball", xyxy=(0, 0, 5, 5), centroid_px=(0, 0),
               field_xy_m=(5.2, 4.0), is_flight=True)
    pt.step(_state(f, [_robot(1, robot_xy)], flights=[fl]))
    f += 1
    # subsequent frames: flight track gone (field_xy_m None path) -> lost, untestable
    nofix = Track(tid=92, cls="ball", xyxy=(0, 0, 5, 5), centroid_px=(0, 0),
                  field_xy_m=None, is_flight=True)
    for _ in range(int(0.2 * FPS) + 4):
        pt.step(_state(f, [_robot(1, robot_xy)], flights=[nofix]))
        f += 1

    rs = pt.robots[1]
    assert rs.shot_attempts == 1
    assert rs.shots_made == 0
    assert rs.shots_resolvable == 0, "no geometry / overlay -> not judgeable"
    assert rs.shot_accuracy is None, "accuracy unknown, must not be fabricated"
    assert rs.miss_rate is None


def test_stats_dict_json_serializable():
    """The full stats_dict round-trips through JSON (all values primitive/None)."""
    field = StubField()
    pt = PossessionTracker(field, {"intake_k": 1.0, "intake_zone_m": 0.6,
                                   "converge_min_s": 0.0, "vanish_grace_s": 0.1,
                                   "shot_origin_m": 1.0, "hub_tol_m": 0.3,
                                   "shot_lookahead_s": 0.5, "intake_dedup_s": 0.0},
                          fps=FPS)
    robot_xy = (5.0, 4.0)
    f = _possess(pt, robot_xy)
    # land one shot so made/accuracy populate
    pt.step(_state(f, [_robot(1, robot_xy)],
                   flights=[_ball(90, (5.2, 4.0), flight=True)]))
    f += 1
    hub_center = tuple(field.hubs["blue"]["center_m"])
    for step in range(8):
        fx = 5.2 + (hub_center[0] - 5.2) * (step + 1) / 8.0
        fy = 4.0 + (hub_center[1] - 4.0) * (step + 1) / 8.0
        pt.step(_state(f, [_robot(1, robot_xy)],
                       flights=[_ball(90, (fx, fy), flight=True)]))
        f += 1

    d = pt.stats_dict()
    s = json.dumps(d)            # raises if anything is not JSON-serializable
    back = json.loads(s)
    assert "1" in back           # int keys serialize to str keys
    row = back["1"]
    for key in ("intake_attempts", "intake_successes", "intake_success_rate",
                "mean_time_to_intake_s", "shot_attempts", "shots_made",
                "shot_accuracy", "miss_rate", "shot_successes", "shot_success_rate",
                "held_count", "possession_time_s", "balls_in_radius_peak"):
        assert key in row, f"missing {key} in stats row"
