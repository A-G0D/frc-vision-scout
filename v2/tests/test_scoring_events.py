"""Intake/Outtake/SCORED state-machine tests — pure synthetic, no model/cv2/GPU.

We feed hand-built FrameState objects with field-projected robot/ball tracks and
assert the per-robot Intake/Outtake/Scored tallies + the canonical Events:
  * a free ball converging on a robot then vanishing in its scaled intake radius
        -> a `pickup` event (kind="intake"), intakes++ / held_count++;
  * a possessed ball launching as a fast outbound flight
        -> a `shot` event (kind="shot"/"outtake"), outtakes++/shots++, held--;
  * a shot whose arc enters the HUB (mock in_hub True) with a valid pose
        -> a `score` event (kind="shot_made"), made++;
  * a shot that times out away from the HUB (pose valid) -> missed++;
  * with NO localization the outcome is UNKNOWN -> shot_accuracy None (never 0).

A tiny stub localizer supplies only ``in_hub`` (the lone geometry call the tracker
makes), so no YAML / FieldModel is needed.
"""
from __future__ import annotations

import json
import math

from frcscout.io.types import CameraPose, FrameState, Track
from frcscout.refine.scoring_events import ScoringEventTracker, RobotScoring


FPS = 60.0
_VALID_POSE = CameraPose(H=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], valid=True)


# ───────────────────────────── stub localizers ───────────────────────────────
class StubHub:
    """One circular HUB per alliance, in field meters."""
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


class AlwaysInHub:
    """Localizer whose in_hub is always True (mock a confirmed make)."""
    def in_hub(self, xy, alliance, tol_m=0.0):
        return True


class NeverInHub:
    def in_hub(self, xy, alliance, tol_m=0.0):
        return False


class ConfidentHub(AlwaysInHub):
    """Exposes its own localization confidence — the tracker must carry it."""
    def localization_confidence(self, alliance):
        return 0.95


# ───────────────────────────── track builders ────────────────────────────────
def _robot(tid, xy, alliance="blue", size_m=None):
    r = Track(tid=tid, cls="robot", xyxy=(0, 0, 10, 10), centroid_px=(0, 0),
              field_xy_m=xy, alliance=alliance)
    if size_m is not None:
        r.field_wh_m = (size_m, size_m)   # projected field-meter extent (dynamic attr)
    return r


def _ball(tid, xy):
    return Track(tid=tid, cls="ball", xyxy=(0, 0, 5, 5), centroid_px=(0, 0),
                 field_xy_m=xy)


def _flight(tid, xy, vel=(0.0, 0.0)):
    return Track(tid=tid, cls="ball", xyxy=(0, 0, 5, 5), centroid_px=(0, 0),
                 field_xy_m=xy, vel_mps=vel, is_flight=True)


def _state(frame_idx, robots, inventory=None, flights=None, pose=None):
    return FrameState(frame_idx=frame_idx, t_s=frame_idx / FPS, robots=robots,
                      balls_inventory=inventory or [], balls_flight=flights or [],
                      pose=pose)


# ─────────────────── drive a ball into possession (helper) ────────────────────
def _possess(pt, robot, robot_id=1, alliance="blue", ball_id=50, start_f=0,
             pose=None):
    """Closes a free ball onto the robot then vanishes it -> POSSESSED.
    Returns the next free frame index."""
    f = start_f
    rxy = robot.field_xy_m
    for dist in [3.0, 2.0, 1.0, 0.5, 0.3]:
        bx = rxy[0] - dist
        pt.step(_state(f, [robot], inventory=[_ball(ball_id, (bx, rxy[1]))], pose=pose))
        f += 1
    for _ in range(int(pt.vanish_grace_frames) + 2):
        pt.step(_state(f, [robot], inventory=[], pose=pose))
        f += 1
    return f


# ═══════════════════════════════ INTAKE ══════════════════════════════════════
def test_intake_attempt_and_success():
    """A free ball closes on a stationary robot's intake radius, then vanishes
    inside it -> a `pickup` (kind=intake) event, intakes=1, held_count=1."""
    pt = ScoringEventTracker(StubHub(), {"intake_floor_m": 0.6, "converge_min_s": 0.0,
                                         "vanish_grace_s": 0.1}, fps=FPS)
    robot = _robot(1, (5.0, 4.0))
    saw_intake = False
    f = 0
    for dist in [3.0, 2.0, 1.0, 0.5, 0.3]:
        bx = 5.0 - dist
        pt.step(_state(f, [robot], inventory=[_ball(50, (bx, 4.0))]))
        f += 1
    for _ in range(int(0.1 * FPS) + 2):
        evs = pt.step(_state(f, [robot], inventory=[]))
        for e in evs:
            if e.type == "pickup" and e.detail.get("kind") == "intake" and e.robot_id == 1:
                saw_intake = True
        f += 1

    assert saw_intake, "expected a pickup/intake event after the ball vanished in-radius"
    rs = pt.robots[1]
    assert rs.intakes == 1
    assert rs.held_count == 1


def test_intake_radius_scales_with_robot_size():
    """intake_radius = intake_k * robot_size_m (meters); a bigger robot reaches
    further; with only a px bbox it falls back to the meter floor."""
    pt = ScoringEventTracker(StubHub(), {"intake_k": 1.2, "intake_floor_m": 0.3}, fps=FPS)
    small = _robot(1, (5.0, 4.0), size_m=0.5)
    big = _robot(2, (5.0, 4.0), size_m=1.0)
    assert math.isclose(pt.intake_radius(small), 1.2 * 0.5)
    assert math.isclose(pt.intake_radius(big), 1.2 * 1.0)
    assert pt.intake_radius(big) > pt.intake_radius(small)
    # no field_wh_m -> floor (px can't be compared to meter distances)
    px_only = _robot(3, (5.0, 4.0))
    assert math.isclose(pt.intake_radius(px_only), 0.3)


def test_intake_floor_protects_degenerate_bbox():
    pt = ScoringEventTracker(StubHub(), {"intake_k": 1.2, "intake_floor_m": 0.6}, fps=FPS)
    r = _robot(1, (5.0, 4.0), size_m=0.0)   # zero-area projected bbox
    assert math.isclose(pt.intake_radius(r), 0.6)


# ═══════════════════════════════ OUTTAKE ═════════════════════════════════════
def test_outtake_emitted_when_possessed_ball_ejects_fast():
    """A possessed ball launching as a fast outbound flight at the robot ->
    outtakes++ and (fast enough) shots++, held_count back to 0."""
    pt = ScoringEventTracker(StubHub(), {"intake_floor_m": 0.6, "converge_min_s": 0.0,
                                         "vanish_grace_s": 0.1, "outtake_origin_m": 1.0,
                                         "outtake_min_speed_mps": 1.5,
                                         "shot_min_speed_mps": 2.5,
                                         "shot_lookahead_s": 2.0}, fps=FPS)
    robot = _robot(1, (5.0, 4.0))
    f = _possess(pt, robot)
    assert pt.robots[1].held_count == 1

    # flight born AT the robot, moving outbound fast (8 m/s toward +x)
    evs = pt.step(_state(f, [robot], flights=[_flight(90, (5.2, 4.0), vel=(8.0, 0.0))]))
    kinds = [e.detail.get("kind") for e in evs if e.type == "shot"]
    assert "shot" in kinds, "a fast outbound eject should be a SHOT"
    rs = pt.robots[1]
    assert rs.outtakes == 1
    assert rs.shots == 1
    assert rs.held_count == 0, "an outtake consumes a held ball"


def test_slow_eject_is_outtake_but_not_shot():
    """An eject above the outtake floor but below the shot speed counts as an
    outtake (kind=outtake) only — no shot, no SCORED resolution."""
    pt = ScoringEventTracker(StubHub(), {"intake_floor_m": 0.6, "converge_min_s": 0.0,
                                         "vanish_grace_s": 0.1, "outtake_origin_m": 1.0,
                                         "outtake_min_speed_mps": 1.5,
                                         "shot_min_speed_mps": 2.5}, fps=FPS)
    robot = _robot(1, (5.0, 4.0))
    f = _possess(pt, robot)
    # slow outbound eject: 2.0 m/s (>= outtake 1.5, < shot 2.5)
    evs = pt.step(_state(f, [robot], flights=[_flight(90, (5.2, 4.0), vel=(2.0, 0.0))]))
    kinds = [e.detail.get("kind") for e in evs if e.type == "shot"]
    assert kinds == ["outtake"], f"expected a lone outtake, got {kinds}"
    rs = pt.robots[1]
    assert rs.outtakes == 1
    assert rs.shots == 0


def test_inbound_or_too_slow_flight_is_not_an_outtake():
    """A flight that is too slow (below the outtake floor) does not credit an
    outtake even if it is born at a possessing robot."""
    pt = ScoringEventTracker(StubHub(), {"intake_floor_m": 0.6, "converge_min_s": 0.0,
                                         "vanish_grace_s": 0.1, "outtake_origin_m": 1.0,
                                         "outtake_min_speed_mps": 1.5}, fps=FPS)
    robot = _robot(1, (5.0, 4.0))
    f = _possess(pt, robot)
    pt.step(_state(f, [robot], flights=[_flight(90, (5.2, 4.0), vel=(0.5, 0.0))]))
    rs = pt.robots[1]
    assert rs.outtakes == 0
    assert rs.held_count == 1, "a non-eject must not consume the held ball"


# ═══════════════════════════════ SCORED (MADE) ═══════════════════════════════
def test_shot_made_when_arc_enters_hub():
    """A possessed ball is shot and its arc enters the HUB (mock in_hub True) with a
    valid pose -> a `score` event (kind=shot_made), made=1."""
    pt = ScoringEventTracker(AlwaysInHub(), {"intake_floor_m": 0.6, "converge_min_s": 0.0,
                                             "vanish_grace_s": 0.1, "outtake_origin_m": 1.0,
                                             "shot_min_speed_mps": 2.5,
                                             "shot_lookahead_s": 2.0}, fps=FPS)
    robot = _robot(1, (5.0, 4.0))
    f = _possess(pt, robot, pose=_VALID_POSE)

    # launch the shot (fast outbound) with a valid pose. The SCORED resolution can
    # fire on the launch frame itself (the HUB test is positional), so we scan the
    # launch step's events too.
    saw_made = False
    all_evs = pt.step(_state(f, [robot], flights=[_flight(90, (5.2, 4.0), vel=(8.0, 0.0))],
                             pose=_VALID_POSE))
    f += 1
    assert pt.robots[1].shots == 1

    for _ in range(8):
        all_evs += pt.step(_state(f, [robot],
                                  flights=[_flight(90, (6.0, 4.0), vel=(8.0, 0.0))],
                                  pose=_VALID_POSE))
        f += 1
    for e in all_evs:
        if e.type == "score" and e.detail.get("kind") == "shot_made" and e.robot_id == 1:
            saw_made = True

    assert saw_made, "arc entering the HUB (in_hub True) with a pose should SCORE"
    rs = pt.robots[1]
    assert rs.made == 1
    assert rs.missed == 0
    assert rs.shot_accuracy == 1.0


def test_made_confidence_carries_localizer_value_not_fabricated():
    """When the localizer reports its own confidence, the SCORED event carries it
    (not a hardcoded 1.0)."""
    pt = ScoringEventTracker(ConfidentHub(), {"intake_floor_m": 0.6, "converge_min_s": 0.0,
                                              "vanish_grace_s": 0.1, "outtake_origin_m": 1.0,
                                              "shot_min_speed_mps": 2.5,
                                              "shot_lookahead_s": 2.0}, fps=FPS)
    robot = _robot(1, (5.0, 4.0))
    f = _possess(pt, robot, pose=_VALID_POSE)
    conf_seen = None
    all_evs = pt.step(_state(f, [robot], flights=[_flight(90, (5.2, 4.0), vel=(8.0, 0.0))],
                             pose=_VALID_POSE))
    f += 1
    for _ in range(6):
        all_evs += pt.step(_state(f, [robot],
                                  flights=[_flight(90, (6.0, 4.0), vel=(8.0, 0.0))],
                                  pose=_VALID_POSE))
        f += 1
    for e in all_evs:
        if e.type == "score":
            conf_seen = e.confidence
    assert conf_seen is not None
    assert math.isclose(conf_seen, 0.95), f"expected localizer confidence, got {conf_seen}"


# ═══════════════════════════════ SCORED (MISS) ═══════════════════════════════
def test_shot_miss_times_out_away_from_hub():
    """A shot whose arc never enters the HUB and times out, with a valid pose so it
    is judgeable -> missed=1, shot_accuracy 0."""
    pt = ScoringEventTracker(NeverInHub(), {"intake_floor_m": 0.6, "converge_min_s": 0.0,
                                            "vanish_grace_s": 0.1, "outtake_origin_m": 1.0,
                                            "shot_min_speed_mps": 2.5,
                                            "shot_lookahead_s": 0.2}, fps=FPS)
    robot = _robot(1, (5.0, 4.0))
    f = _possess(pt, robot, pose=_VALID_POSE)
    pt.step(_state(f, [robot], flights=[_flight(90, (5.2, 4.0), vel=(8.0, 0.0))],
                   pose=_VALID_POSE))
    f += 1
    assert pt.robots[1].shots == 1

    saw_made = False
    for step in range(int(0.2 * FPS) + 4):
        evs = pt.step(_state(f, [robot],
                             flights=[_flight(90, (8.0 + step, 8.0), vel=(8.0, 0.0))],
                             pose=_VALID_POSE))
        if any(e.type == "score" for e in evs):
            saw_made = True
        f += 1

    assert not saw_made, "a shot that never enters the HUB must not score"
    rs = pt.robots[1]
    assert rs.shots == 1
    assert rs.made == 0
    assert rs.missed == 1, "a judgeable (pose-valid) miss is counted"
    assert math.isclose(rs.shot_accuracy, 0.0)
    assert math.isclose(rs.miss_rate, 1.0)


def test_shot_made_via_overlay_when_arc_inconclusive():
    """With no pose (arc untrustworthy) an overlay score increment in the window
    confirms the shot (authoritative count path) -> made=1."""
    pt = ScoringEventTracker(NeverInHub(), {"intake_floor_m": 0.6, "converge_min_s": 0.0,
                                            "vanish_grace_s": 0.1, "outtake_origin_m": 1.0,
                                            "shot_min_speed_mps": 2.5,
                                            "shot_lookahead_s": 2.0, "score_window_s": 1.0},
                             fps=FPS)
    robot = _robot(1, (5.0, 4.0), alliance="blue")
    f = _possess(pt, robot, alliance="blue")   # no pose
    pt.step(_state(f, [robot], flights=[_flight(90, (5.2, 4.0), vel=(8.0, 0.0))]))
    shot_frame = f
    f += 1
    pt.note_overlay_score(shot_frame + 5, "blue")

    saw_made = False
    for _ in range(10):
        evs = pt.step(_state(f, [robot], flights=[_flight(90, (9.0, 7.0), vel=(8.0, 0.0))]))
        for e in evs:
            if e.type == "score" and e.robot_id == 1:
                saw_made = True
                assert e.detail.get("via") == "overlay"
        f += 1
    assert saw_made, "overlay score increment should confirm the shot"
    assert pt.robots[1].made == 1


def test_shot_accuracy_none_without_localization():
    """With no pose AND no overlay pulse the outcome is UNKNOWN: missed stays 0,
    shot_accuracy is None (never a fabricated 0), and no SCORED/MISS event fires."""
    pt = ScoringEventTracker(NeverInHub(), {"intake_floor_m": 0.6, "converge_min_s": 0.0,
                                            "vanish_grace_s": 0.1, "outtake_origin_m": 1.0,
                                            "shot_min_speed_mps": 2.5,
                                            "shot_lookahead_s": 0.2}, fps=FPS)
    robot = _robot(1, (5.0, 4.0))
    f = _possess(pt, robot)   # NO pose anywhere
    pt.step(_state(f, [robot], flights=[_flight(90, (5.2, 4.0), vel=(8.0, 0.0))]))
    f += 1
    saw_outcome = False
    for step in range(int(0.2 * FPS) + 6):
        evs = pt.step(_state(f, [robot],
                             flights=[_flight(90, (8.0 + step, 8.0), vel=(8.0, 0.0))]))
        if any(e.type == "score" or e.detail.get("kind") == "shot_miss" for e in evs):
            saw_outcome = True
        f += 1

    rs = pt.robots[1]
    assert rs.shots == 1
    assert rs.made == 0
    assert rs.missed == 0, "no localization -> not judgeable, not a charged miss"
    assert rs.shot_accuracy is None, "accuracy unknown, must NOT be fabricated"
    assert rs.miss_rate is None
    assert not saw_outcome, "no SCORED/MISS event when the outcome is unknowable"


def test_shot_accuracy_one_made_one_missed():
    """One made (HUB+pose), one missed (timeout+pose) -> accuracy = 1/2."""
    field = StubHub()
    pt = ScoringEventTracker(field, {"intake_floor_m": 0.6, "converge_min_s": 0.0,
                                     "vanish_grace_s": 0.1, "outtake_origin_m": 1.0,
                                     "hub_tol_m": 0.3, "shot_min_speed_mps": 2.5,
                                     "shot_lookahead_s": 0.3, "intake_dedup_s": 0.0},
                             fps=FPS)
    robot = _robot(1, (5.0, 4.0))
    hub_center = tuple(field.hubs["blue"]["center_m"])

    # made: arc walks into the blue HUB center (toward -x), pose valid. Spawn the
    # flight on the -x side of the robot so the -x velocity reads as OUTBOUND.
    f = _possess(pt, robot, ball_id=50, start_f=0, pose=_VALID_POSE)
    origin_x = 4.8
    pt.step(_state(f, [robot], flights=[_flight(90, (origin_x, 4.0), vel=(-8.0, 0.0))],
                   pose=_VALID_POSE))
    f += 1
    for step in range(8):
        fx = origin_x + (hub_center[0] - origin_x) * (step + 1) / 8.0
        fy = 4.0 + (hub_center[1] - 4.0) * (step + 1) / 8.0
        pt.step(_state(f, [robot], flights=[_flight(90, (fx, fy), vel=(-8.0, 0.0))],
                       pose=_VALID_POSE))
        f += 1
    assert pt.robots[1].made == 1

    # missed: flies away from the hub, times out, pose valid -> judgeable miss
    f = _possess(pt, robot, ball_id=51, start_f=f, pose=_VALID_POSE)
    pt.step(_state(f, [robot], flights=[_flight(91, (5.2, 4.0), vel=(8.0, 0.0))],
                   pose=_VALID_POSE))
    f += 1
    for step in range(int(0.3 * FPS) + 4):
        pt.step(_state(f, [robot],
                       flights=[_flight(91, (8.0 + step, 8.0), vel=(8.0, 0.0))],
                       pose=_VALID_POSE))
        f += 1

    rs = pt.robots[1]
    assert rs.shots == 2
    assert rs.made == 1
    assert rs.missed == 1
    assert rs.resolvable_shots == 2
    assert math.isclose(rs.shot_accuracy, 0.5)
    assert math.isclose(rs.miss_rate, 0.5)


# ═══════════════════════════════ reporting ═══════════════════════════════════
def test_tallies_shape_and_json_serializable():
    """tallies() carries the named per-robot counters and round-trips through JSON."""
    field = StubHub()
    pt = ScoringEventTracker(field, {"intake_floor_m": 0.6, "converge_min_s": 0.0,
                                     "vanish_grace_s": 0.1, "outtake_origin_m": 1.0,
                                     "shot_min_speed_mps": 2.5, "shot_lookahead_s": 0.5,
                                     "intake_dedup_s": 0.0}, fps=FPS)
    robot = _robot(1, (5.0, 4.0))
    _possess(pt, robot, pose=_VALID_POSE)
    d = pt.tallies()
    assert 1 in d
    row = d[1]
    for key in ("intakes", "outtakes", "shots", "made", "missed",
                "shot_accuracy", "miss_rate", "held_count", "resolvable_shots"):
        assert key in row, f"missing {key}"

    s = json.dumps(d)            # raises if anything is not JSON-serializable
    back = json.loads(s)
    assert "1" in back           # int keys serialize to str keys
    assert back["1"]["intakes"] == 1


def test_robotscoring_accuracy_property_directly():
    """RobotScoring.shot_accuracy: None with nothing resolvable; defined once
    made/missed populate."""
    rs = RobotScoring(robot_id=7)
    assert rs.shot_accuracy is None
    assert rs.miss_rate is None
    rs.made, rs.missed = 3, 1
    assert math.isclose(rs.shot_accuracy, 0.75)
    assert math.isclose(rs.miss_rate, 0.25)
