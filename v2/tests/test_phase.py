"""Per-robot PHASE state-machine tests — pure synthetic, no model / cv2 / GPU.

We feed hand-built FrameState sequences (field-projected Track objects + an overlay
phase) into ``PhaseTracker`` and assert the chosen PHASE, the inferred held-count, the
climb latch, and the scouting metrics. The embedded ScoringEventTracker is driven by
the same frames, so a tiny ``in_hub`` stub is all the geometry the shot/score path
needs; a tiny tower/hub stub supplies climb + defense geometry.

Coverage (mirrors the task's required cases):
  * a robot over vanishing balls -> INTAKING then HOLDING(n) with the right n;
  * a fast outbound ball -> SHOOTING then SCORED (mock in_hub True) or MISS;
  * endgame + rising bumper at the tower -> CLIMBING with a latched level;
  * jitter_index higher for a wobbling track than a smooth one;
  * phase_timeline + metrics correct;
  * the whole output is JSON-serializable.
"""
from __future__ import annotations

import json
import math

from frcscout.io.types import CameraPose, FrameState, OverlayState, Track
from frcscout.refine.phase import (
    PhaseTracker,
    TRAVERSING, INTAKING, HOLDING, SHOOTING, SCORED, CLIMBING, DEFENSE,
)


FPS = 60.0
_VALID_POSE = CameraPose(H=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], valid=True)


# ───────────────────────────── stub geometry ─────────────────────────────────
class StubField:
    """Duck-typed field model: hubs + towers in field meters, + length_m.

    Layout matches configs/field_2026_rebuilt.yaml: blue HUB/TOWER at low-x, red at
    high-x; field ~16.46 m long. Satisfies what PhaseTracker reads (towers, hubs/
    hub_for, length_m) AND the ScoringEventTracker's in_hub.
    """
    length_m = 16.46

    def __init__(self):
        self.hubs = {
            "blue": {"center_m": [3.20, 4.10], "radius_m": 0.80},
            "red": {"center_m": [13.26, 4.10], "radius_m": 0.80},
        }
        self.towers = {
            "blue": {"center_m": [1.50, 4.10], "climb_height_m": 1.20},
            "red": {"center_m": [14.96, 4.10], "climb_height_m": 1.20},
        }

    def in_hub(self, xy, alliance, tol_m=0.0):
        hub = self.hubs.get(alliance)
        if not hub:
            return False
        cx, cy = hub["center_m"]
        r = hub["radius_m"] + tol_m
        return (xy[0] - cx) ** 2 + (xy[1] - cy) ** 2 <= r * r


class AlwaysInHub:
    def in_hub(self, xy, alliance, tol_m=0.0):
        return True


class NeverInHub:
    def in_hub(self, xy, alliance, tol_m=0.0):
        return False


# ───────────────────────────── track builders ────────────────────────────────
def _robot(tid, xy, alliance="blue", size_m=0.8, vel=(0.0, 0.0), box=None,
           z_m=None, heading_rad=None):
    x1, y1, x2, y2 = box if box is not None else (0.0, 0.0, 60.0, 60.0)
    r = Track(tid=tid, cls="robot", xyxy=(x1, y1, x2, y2),
              centroid_px=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
              field_xy_m=xy, alliance=alliance, vel_mps=vel)
    if size_m is not None:
        r.field_wh_m = (size_m, size_m)
    if z_m is not None:
        r.z_m = z_m
    if heading_rad is not None:
        r.heading_rad = heading_rad
    return r


def _ball(tid, xy):
    return Track(tid=tid, cls="ball", xyxy=(0, 0, 5, 5), centroid_px=(0, 0),
                 field_xy_m=xy)


def _flight(tid, xy, vel=(0.0, 0.0)):
    return Track(tid=tid, cls="ball", xyxy=(0, 0, 5, 5), centroid_px=(0, 0),
                 field_xy_m=xy, vel_mps=vel, is_flight=True)


def _state(frame_idx, robots, inventory=None, flights=None, pose=None, phase=None,
           timer_s=None):
    ov = OverlayState(phase=phase, timer_s=timer_s)
    return FrameState(frame_idx=frame_idx, t_s=frame_idx / FPS, robots=robots,
                      balls_inventory=inventory or [], balls_flight=flights or [],
                      overlay=ov, pose=pose)


_FAST_CFG = {
    "intake_floor_m": 0.6, "converge_min_s": 0.0, "vanish_grace_s": 0.1,
    "outtake_origin_m": 1.0, "outtake_min_speed_mps": 1.5, "shot_min_speed_mps": 2.5,
    "shot_lookahead_s": 2.0, "intake_dedup_s": 0.0,
}


def _mk(field=None, hub=None, cfg=None, **kw):
    """PhaseTracker with the fast scoring config merged in."""
    c = dict(_FAST_CFG)
    c.update(cfg or {})
    return PhaseTracker(field_model=field, hub_localizer=hub if hub is not None else field,
                        cfg=c, fps=FPS, **kw)


def _drive_in(pt, robot, ball_id=50, start_f=0, pose=None, phase="teleop"):
    """Close a free ball onto the robot, then vanish it -> POSSESSED. Returns the next
    free frame index. Mirrors the scoring-events possession helper."""
    f = start_f
    rxy = robot.field_xy_m
    last_out = None
    for dist in [3.0, 2.0, 1.0, 0.5, 0.3]:
        bx = rxy[0] - dist
        last_out = pt.update(_state(f, [robot], inventory=[_ball(ball_id, (bx, rxy[1]))],
                                    pose=pose, phase=phase))
        f += 1
    for _ in range(int(pt.scoring.vanish_grace_frames) + 2):
        last_out = pt.update(_state(f, [robot], inventory=[], pose=pose, phase=phase))
        f += 1
    return f, last_out


# ═══════════════════════════════ INTAKING -> HOLDING ═════════════════════════
def test_intaking_then_holding_with_correct_n():
    """A robot sits over balls that converge and then VANISH inside it: it reads INTAKING
    at the CONSUMPTION (the ScoringEventTracker confirms the intake on the vanish), then
    HOLDING(1) once swallowed; held-count == intaken − shot.

    NOTE (the dense-field fix): INTAKING is now gated on real CONSUMPTION evidence — a ball
    that converges AND disappears INTO the robot — NOT mere proximity-in-heading during the
    approach. So we assert INTAKING appears across the converge->vanish->confirm sequence
    (it fires on the confirm frame + brief latch), then HOLDING."""
    field = StubField()
    pt = _mk(field)
    robot = _robot(1, (5.0, 4.0), alliance="blue")

    saw_intaking = False
    f = 0
    # approach frames: a ball converges inside the intake radius (tracked approach).
    for dist in [3.0, 2.0, 1.0, 0.5, 0.3]:
        out = pt.update(_state(f, [robot], inventory=[_ball(50, (5.0 - dist, 4.0))],
                               pose=_VALID_POSE, phase="teleop"))
        if out[1]["phase"] == INTAKING:
            saw_intaking = True
        f += 1

    # ball vanishes -> CONSUMPTION confirmed -> INTAKING (on the confirm frame) then HOLDING(1)
    out = None
    for _ in range(int(pt.scoring.vanish_grace_frames) + 3):
        out = pt.update(_state(f, [robot], inventory=[], pose=_VALID_POSE, phase="teleop"))
        if out[1]["phase"] == INTAKING:
            saw_intaking = True
        f += 1
    assert saw_intaking, "expected INTAKING on the confirmed consumption (converge -> vanish)"
    assert out[1]["held"] == 1

    # step past the brief INTAKING latch -> settles to HOLDING(1)
    for _ in range(int(pt.intake_hold_frames) + 2):
        out = pt.update(_state(f, [robot], inventory=[], pose=_VALID_POSE, phase="teleop"))
        f += 1
    assert out[1]["phase"] == HOLDING
    assert out[1]["label"] == "B1 HOLDING(1)"
    assert pt.robots[1].max_held == 1


def test_holding_two_after_two_intakes():
    """Two successive intakes -> HOLDING(2); n tracks the inferred held-count."""
    field = StubField()
    pt = _mk(field)
    robot = _robot(1, (5.0, 4.0), alliance="red")

    f, _ = _drive_in(pt, robot, ball_id=50, start_f=0, pose=_VALID_POSE)
    f, _ = _drive_in(pt, robot, ball_id=51, start_f=f, pose=_VALID_POSE)

    # step past the brief INTAKING latch from the second confirmed intake -> HOLDING(2)
    out = pt.update(_state(f, [robot], pose=_VALID_POSE, phase="teleop"))
    f += 1
    for _ in range(int(pt.intake_hold_frames) + 2):
        out = pt.update(_state(f, [robot], pose=_VALID_POSE, phase="teleop"))
        f += 1
    assert out[1]["held"] == 2
    assert out[1]["phase"] == HOLDING
    assert out[1]["label"] == "R1 HOLDING(2)"
    assert out[1]["metrics"]["max_held"] == 2
    assert out[1]["metrics"]["held_inferred"] is True


# ═══════════════════════════════ SHOOTING -> SCORED ══════════════════════════
def test_shooting_then_scored_when_arc_enters_hub():
    """A possessed ball launches fast+outbound -> SHOOTING; its arc then walks INTO the
    blue HUB (real geometry) with a valid pose -> SCORED, shots_made=1.

    We use the real StubField HUB (not an always-true stub) and spawn the flight on the
    robot's -x side so the -x velocity reads as OUTBOUND and the arc only reaches the
    HUB after a few frames — so SHOOTING is observable on the launch frame before SCORED
    latches. (An always-in-hub stub would resolve the make on the launch frame itself,
    by design of scoring_events — covered separately via the metrics assertions.)"""
    field = StubField()
    pt = _mk(field, hub=field)
    robot = _robot(1, (5.0, 4.0), alliance="blue")
    hub_center = tuple(field.hubs["blue"]["center_m"])
    f, _ = _drive_in(pt, robot, pose=_VALID_POSE)

    # launch frame: fast flight born just -x of the robot, heading toward the HUB
    origin_x = 4.8
    out = pt.update(_state(f, [robot], flights=[_flight(90, (origin_x, 4.0), vel=(-8.0, 0.0))],
                           pose=_VALID_POSE, phase="teleop"))
    f += 1
    assert out[1]["phase"] == SHOOTING, "a fresh fast outbound eject should read SHOOTING"
    assert out[1]["metrics"]["shot_attempts"] == 1

    saw_scored = False
    for step in range(8):
        fx = origin_x + (hub_center[0] - origin_x) * (step + 1) / 8.0
        fy = 4.0 + (hub_center[1] - 4.0) * (step + 1) / 8.0
        out = pt.update(_state(f, [robot], flights=[_flight(90, (fx, fy), vel=(-8.0, 0.0))],
                               pose=_VALID_POSE, phase="teleop"))
        if out[1]["phase"] == SCORED:
            saw_scored = True
        f += 1
    assert saw_scored, "arc entering the HUB with a pose should latch SCORED"
    m = pt.metrics(1)
    assert m["shots_made"] == 1
    assert m["shot_accuracy"]["value"] == 1.0
    assert m["shot_accuracy"]["available"] is True


def test_shooting_then_miss_times_out():
    """A shot that never enters the HUB and times out (pose valid) -> no SCORED; the
    embedded scorer charges a resolvable miss so accuracy == 0."""
    field = StubField()
    pt = _mk(field, hub=NeverInHub(), cfg={"shot_lookahead_s": 0.2})
    robot = _robot(1, (5.0, 4.0), alliance="blue")
    f, _ = _drive_in(pt, robot, pose=_VALID_POSE)

    out = pt.update(_state(f, [robot], flights=[_flight(90, (5.2, 4.0), vel=(8.0, 0.0))],
                           pose=_VALID_POSE, phase="teleop"))
    f += 1
    assert out[1]["phase"] == SHOOTING

    saw_scored = False
    for step in range(int(0.2 * FPS) + 6):
        out = pt.update(_state(f, [robot],
                               flights=[_flight(90, (9.0 + step, 8.0), vel=(8.0, 0.0))],
                               pose=_VALID_POSE, phase="teleop"))
        if out[1]["phase"] == SCORED:
            saw_scored = True
        f += 1
    assert not saw_scored, "a shot that never enters the HUB must not read SCORED"
    m = pt.metrics(1)
    assert m["shots_made"] == 0
    assert m["shot_accuracy"]["value"] == 0.0, "a judgeable (pose) miss -> accuracy 0"


def test_shot_accuracy_unknown_without_localization():
    """No pose AND no overlay pulse -> outcome unknowable: shot_accuracy stays None
    (never a fabricated 0) and is flagged unavailable with a needs note."""
    pt = _mk(field=None, hub=NeverInHub(), cfg={"shot_lookahead_s": 0.2})
    robot = _robot(1, (5.0, 4.0), alliance="blue")
    f, _ = _drive_in(pt, robot, pose=None, phase="teleop")   # NO pose anywhere
    pt.update(_state(f, [robot], flights=[_flight(90, (5.2, 4.0), vel=(8.0, 0.0))],
                     phase="teleop"))
    f += 1
    for step in range(int(0.2 * FPS) + 6):
        pt.update(_state(f, [robot],
                         flights=[_flight(90, (9.0 + step, 8.0), vel=(8.0, 0.0))],
                         phase="teleop"))
        f += 1
    m = pt.metrics(1)
    assert m["shots_made"] == 0
    assert m["shot_accuracy"]["value"] is None
    assert m["shot_accuracy"]["available"] is False
    assert m["shot_accuracy"]["needs"]


def test_balls_shot_at_once_burst():
    """Two shots inside one burst window -> balls_shot_at_once == 2."""
    field = StubField()
    pt = _mk(field, hub=NeverInHub(), cfg={"shot_burst_window_s": 1.0,
                                           "shot_lookahead_s": 2.0})
    robot = _robot(1, (5.0, 4.0), alliance="blue")
    # stock two balls
    f, _ = _drive_in(pt, robot, ball_id=50, start_f=0, pose=_VALID_POSE)
    f, _ = _drive_in(pt, robot, ball_id=51, start_f=f, pose=_VALID_POSE)
    assert pt._held(1) == 2

    # two fast outbound flights a few frames apart (within 1.0s) -> a salvo
    pt.update(_state(f, [robot], flights=[_flight(90, (5.2, 4.0), vel=(8.0, 0.0))],
                     pose=_VALID_POSE, phase="teleop"))
    f += 3
    pt.update(_state(f, [robot], flights=[_flight(91, (5.2, 4.0), vel=(8.0, 0.0))],
                     pose=_VALID_POSE, phase="teleop"))
    f += 1
    assert pt.metrics(1)["balls_shot_at_once"] == 2


# ═══════════════════════════════ CLIMBING ════════════════════════════════════
def test_climbing_latches_level_via_z_when_present():
    """Endgame + robot at its TOWER + a real z above the MID rung -> CLIMBING with a
    latched LEVEL 3 (z path is exact; method flagged 'z')."""
    field = StubField()
    pt = _mk(field, cfg={"climb_confirm_s": 0.0})
    tower_xy = tuple(field.towers["blue"]["center_m"])

    # sit on the tower, height climbs 0 -> above MID rung (1.143 m) -> L3
    out = None
    for i, z in enumerate([0.0, 0.3, 0.7, 1.2, 1.3]):
        out = pt.update(_state(i, [_robot(1, tower_xy, alliance="blue", z_m=z)],
                               pose=_VALID_POSE, phase="endgame"))
    assert out[1]["phase"] == CLIMBING
    assert out[1]["climb_level"] == 3
    assert out[1]["label"] == "B1 CLIMB L3"
    m = pt.metrics(1)
    assert m["climb_method"] == "z"
    assert m["climb_level"] == 3
    assert m["climb_available"] is True


def test_climbing_via_bumper_pixel_heuristic():
    """With no z, a sustained bumper-pixel-height GROWTH while endgame+tower-adjacent
    latches a climb level (the monocular heuristic). Method flagged as such."""
    field = StubField()
    pt = _mk(field, cfg={"climb_confirm_s": 0.05, "climb_baseline_s": 0.2,
                         "climb_px_l1_ratio": 1.06, "climb_px_l2_ratio": 1.14,
                         "climb_px_l3_ratio": 1.24})
    tower_xy = tuple(field.towers["blue"]["center_m"])

    def robo(h):  # bbox of pixel-height h, sitting at the tower
        return _robot(1, tower_xy, alliance="blue", size_m=0.8, box=(0, 0, 60, h))

    out = None
    # baseline on-ground samples (h≈60), then a clear, sustained growth to ~76 (1.27x)
    for i in range(int(0.2 * FPS) + 1):
        out = pt.update(_state(i, [robo(60)], pose=_VALID_POSE, phase="endgame"))
    f = int(0.2 * FPS) + 1
    for i in range(8):
        out = pt.update(_state(f + i, [robo(78)], pose=_VALID_POSE, phase="endgame"))
    assert out[1]["phase"] == CLIMBING
    assert out[1]["climb_level"] >= 1
    assert pt.metrics(1)["climb_method"] == "bumper_px_heuristic"


def test_no_climb_outside_endgame():
    """The same rise OUTSIDE endgame is NOT a climb (climb is endgame-gated)."""
    field = StubField()
    pt = _mk(field, cfg={"climb_confirm_s": 0.0})
    tower_xy = tuple(field.towers["blue"]["center_m"])
    out = None
    for i, z in enumerate([0.0, 0.7, 1.3]):
        out = pt.update(_state(i, [_robot(1, tower_xy, alliance="blue", z_m=z)],
                               pose=_VALID_POSE, phase="teleop"))
    assert out[1]["phase"] != CLIMBING
    assert out[1]["climb_level"] == 0


def test_climb_level_monotonic_no_unlatch():
    """A latched level never decreases even if the height proxy dips (a climb does not
    reverse on the scoreboard)."""
    field = StubField()
    pt = _mk(field, cfg={"climb_confirm_s": 0.0})
    tower_xy = tuple(field.towers["blue"]["center_m"])
    pt.update(_state(0, [_robot(1, tower_xy, alliance="blue", z_m=1.3)],
                     pose=_VALID_POSE, phase="endgame"))
    assert pt.robots[1].climb_level == 3
    # height drops back near ground -> level must NOT fall
    out = pt.update(_state(1, [_robot(1, tower_xy, alliance="blue", z_m=0.2)],
                           pose=_VALID_POSE, phase="endgame"))
    assert out[1]["climb_level"] == 3


# ═══════════════════════════════ DEFENSE ═════════════════════════════════════
def test_defense_in_opponent_zone_near_opponent():
    """A blue robot parked in the RED half next to a red robot, not scoring -> DEFENSE."""
    field = StubField()
    pt = _mk(field)
    # blue robot deep in red territory (+x half), a red opponent right beside it
    blue = _robot(1, (13.0, 4.0), alliance="blue")
    red = _robot(2, (13.5, 4.0), alliance="red")
    out = pt.update(_state(0, [blue, red], pose=_VALID_POSE, phase="teleop"))
    assert out[1]["phase"] == DEFENSE
    assert pt.metrics(1)["defense_time_s"]["available"] is True


def test_no_defense_in_own_zone():
    """The same blue robot in its OWN (blue/low-x) half is not on defense."""
    field = StubField()
    pt = _mk(field)
    blue = _robot(1, (3.0, 4.0), alliance="blue")
    red = _robot(2, (3.5, 4.0), alliance="red")
    out = pt.update(_state(0, [blue, red], pose=_VALID_POSE, phase="teleop"))
    assert out[1]["phase"] != DEFENSE


def test_defense_unavailable_without_pose():
    """Without a field pose the opponent-zone geofence is impossible -> the defense
    metric is flagged unavailable (NOT a fabricated 0)."""
    pt = _mk(field=StubField())
    # robots with NO field_xy_m
    r = Track(tid=1, cls="robot", xyxy=(0, 0, 60, 60), centroid_px=(30, 30),
              alliance="blue", field_xy_m=None)
    out = pt.update(_state(0, [r], phase="teleop"))
    assert out[1]["phase"] == TRAVERSING
    assert out[1]["metrics"]["defense_time_s"]["available"] is False
    assert out[1]["metrics"]["defense_time_s"]["needs"]


# ═══════════════════════════════ JITTER ══════════════════════════════════════
def test_jitter_higher_for_wobbling_than_smooth():
    """jitter_index is larger for a robot whose field position + heading oscillate than
    for one driving in a smooth straight line over the same window."""
    field = StubField()
    pt = _mk(field)

    # smooth robot (id 1): glides +x in a straight line, steady heading
    # wobbly robot (id 2): oscillates in y and swings heading back and forth
    for f in range(int(1.0 * FPS)):
        t = f / FPS
        smooth = _robot(1, (3.0 + 0.5 * t, 4.0), alliance="blue",
                        vel=(0.5, 0.0), heading_rad=0.0)
        wob_y = 4.0 + 0.4 * math.sin(2 * math.pi * 3.0 * t)   # 3 Hz, ±0.4 m wobble
        wob_h = 0.8 * math.sin(2 * math.pi * 3.0 * t)         # heading swings
        wobbly = _robot(2, (3.0 + 0.5 * t, wob_y), alliance="red",
                        vel=(0.5, 0.0), heading_rad=wob_h)
        pt.update(_state(f, [smooth, wobbly], pose=_VALID_POSE, phase="teleop"))

    j_smooth = pt.metrics(1)["jitter_index"]
    j_wobbly = pt.metrics(2)["jitter_index"]
    assert j_smooth["available"] and j_wobbly["available"]
    assert j_smooth["units"] == "m" and j_wobbly["units"] == "m"
    assert j_wobbly["value"] > j_smooth["value"], (
        f"wobbly jitter {j_wobbly['value']} should exceed smooth {j_smooth['value']}")


def test_jitter_flagged_px_without_pose():
    """With no field pose, jitter is still reported (from px centroids) but flagged
    px-scaled with a needs note — never silently passed off as meters."""
    pt = _mk(field=None)
    for f in range(10):
        r = Track(tid=1, cls="robot", xyxy=(0, 0, 60, 60),
                  centroid_px=(30.0 + (f % 2) * 5.0, 30.0), alliance="blue",
                  field_xy_m=None)
        pt.update(_state(f, [r], phase="teleop"))
    j = pt.metrics(1)["jitter_index"]
    assert j["units"] == "px"
    assert j["needs"] and "px-scaled" in j["needs"]


# ═══════════════════════════ TIMELINE + METRICS ══════════════════════════════
def test_phase_timeline_segments_and_metrics():
    """The timeline records contiguous {phase, t_start, t_end} segments in order, and
    a full intake->hold->shoot cycle populates the headline metrics."""
    field = StubField()
    pt = _mk(field, hub=AlwaysInHub())
    robot = _robot(1, (5.0, 4.0), alliance="blue")

    # start idle (TRAVERSING), intake a ball, hold, then shoot+score
    pt.update(_state(0, [robot], pose=_VALID_POSE, phase="teleop"))
    f, _ = _drive_in(pt, robot, start_f=1, pose=_VALID_POSE)
    pt.update(_state(f, [robot], pose=_VALID_POSE, phase="teleop"))   # HOLDING
    f += 1
    pt.update(_state(f, [robot], flights=[_flight(90, (5.2, 4.0), vel=(8.0, 0.0))],
                     pose=_VALID_POSE, phase="teleop"))
    f += 1
    for _ in range(6):
        pt.update(_state(f, [robot], flights=[_flight(90, (6.0, 4.0), vel=(8.0, 0.0))],
                         pose=_VALID_POSE, phase="teleop"))
        f += 1

    m = pt.metrics(1)
    tl = m["phase_timeline"]
    assert len(tl) >= 3, f"expected several phase segments, got {tl}"
    # segments are time-ordered and non-overlapping
    for a, b in zip(tl, tl[1:]):
        assert a["t_end"] <= b["t_start"] + 1e-9
        assert a["t_start"] <= a["t_end"]
    phases_seen = {seg["phase"] for seg in tl}
    assert TRAVERSING in phases_seen
    assert {INTAKING, HOLDING} & phases_seen
    assert {SHOOTING, SCORED} & phases_seen

    # headline metrics populated
    assert m["intakes"] == 1
    assert m["intake_duration_s"] > 0.0
    assert m["time_to_intake_s"] is not None
    assert m["max_held"] >= 1
    assert m["possession_time_s"] > 0.0
    assert m["shot_attempts"] == 1
    assert m["shots_made"] == 1


def test_label_for_renderer_strings():
    """label_for renders the exact overlay strings the renderer draws."""
    field = StubField()
    # NeverInHub so the launch frame stays SHOOTING (no instant make to latch SCORED).
    pt = _mk(field, hub=NeverInHub())
    robot = _robot(3, (5.0, 4.0), alliance="red")

    # SHOOTING
    f, _ = _drive_in(pt, robot, pose=_VALID_POSE)
    pt.update(_state(f, [robot], flights=[_flight(90, (5.2, 4.0), vel=(8.0, 0.0))],
                     pose=_VALID_POSE, phase="teleop"))
    assert pt.label_for(3) == "R3 SHOOTING"
    # unknown robot -> empty string
    assert pt.label_for(999) == ""


def test_traversing_is_default():
    """A lone moving robot with no ball, not at hub/tower, not near an opponent ->
    TRAVERSING."""
    field = StubField()
    pt = _mk(field)
    r = _robot(1, (8.0, 4.0), alliance="blue", vel=(1.0, 0.0))
    out = pt.update(_state(0, [r], pose=_VALID_POSE, phase="teleop"))
    assert out[1]["phase"] == TRAVERSING
    assert out[1]["label"] == "B1 TRAVERSING"
    assert out[1]["held"] == 0


def test_output_json_serializable():
    """update() output AND the full snapshot/metrics round-trip through JSON (every
    value is a primitive / None / list / nested dict)."""
    field = StubField()
    pt = _mk(field, hub=AlwaysInHub())
    robot = _robot(1, (5.0, 4.0), alliance="blue")
    f, last = _drive_in(pt, robot, pose=_VALID_POSE)
    last = pt.update(_state(f, [robot], flights=[_flight(90, (5.2, 4.0), vel=(8.0, 0.0))],
                            pose=_VALID_POSE, phase="teleop"))

    # the per-frame update() dict
    s = json.dumps(last)
    back = json.loads(s)
    assert "1" in back                      # int keys serialize to str keys
    rec = back["1"]
    for key in ("robot_id", "alliance", "phase", "held", "climb_level", "label", "metrics"):
        assert key in rec, f"missing {key}"

    # the full snapshot + metrics dumps too
    json.dumps(pt.snapshot())
    json.dumps(pt.metrics())
    json.dumps(pt.metrics(1))


# ═══════════════════ collector-bin (bbox/velocity) integration ═══════════════
def test_bin_scorer_bbox_cues_or_into_phase():
    """With an embedded BinScorer (enable_bin_scorer), the bbox cues OR-in: a ball inside
    a robot's box reads HOLDING even with no field-meter intake, and a CONSUMED ball (one
    that crosses the robot's tracked approach cone INTO its box) reads INTAKING via the
    BinScorer — independent of scoring_events. Mere proximity-in-heading does NOT (the
    dense-field consumption gate, on by default)."""
    from frcscout.refine.collector_bin import CollectorBin

    # a generous bin far away so it doesn't interfere; pixel frame 1920x1080.
    binloc = CollectorBin(regions={"blue": [(0.80, 0.30), (0.90, 0.30),
                                            (0.90, 0.40), (0.80, 0.40)]}, frame_wh=(1920, 1080))
    pt = PhaseTracker(field_model=StubField(), hub_localizer=NeverInHub(),
                      cfg={"enable_bin_scorer": True, "intake_confirm_s": 0.0,
                           "held_confirm_s": 0.0}, fps=FPS, collector_bin=binloc)
    pt.bin_scorer.set_frame_wh(1920, 1080)

    # robot with a PIXEL box; a ball whose centre is INSIDE the box -> HELD (bbox cue),
    # even though no field-meter intake ever ran (the ball has no field_xy_m).
    r = Track(tid=1, cls="robot", xyxy=(400, 400, 520, 520), centroid_px=(460, 460),
              field_xy_m=(5.0, 4.0), alliance="blue", vel_mps=(0.0, 0.0))
    ball_in_box = Track(tid=50, cls="ball", xyxy=(455, 455, 465, 465), centroid_px=(460, 460))
    out = pt.update(_state(0, [r], inventory=[ball_in_box], pose=_VALID_POSE, phase="teleop"))
    assert out[1]["phase"] == HOLDING, "a ball inside the robot box must read HOLDING (bbox cue)"
    assert out[1]["held"] >= 1

    # a moving robot with a ball AHEAD of it, that then crosses INTO its box -> INTAKING via
    # the BinScorer CONSUMPTION gate. (Proximity-in-heading alone does not — verified below.)
    pt2 = PhaseTracker(field_model=StubField(), hub_localizer=NeverInHub(),
                       cfg={"enable_bin_scorer": True, "intake_confirm_s": 0.0,
                            "held_confirm_s": 0.0}, fps=FPS, collector_bin=binloc)
    pt2.bin_scorer.set_frame_wh(1920, 1080)
    r2 = Track(tid=1, cls="robot", xyxy=(440, 440, 560, 560), centroid_px=(500, 500),
               field_xy_m=(8.0, 4.0), alliance="red", vel_mps=(2.0, 0.0))
    r2.vel_px = (40.0, 0.0)                       # moving +x in image
    # frame 0: ball ahead in the cone (outside the box) -> tracked approach, NOT yet intaking
    ball_ahead = Track(tid=51, cls="ball", xyxy=(595, 495, 605, 505), centroid_px=(600, 500))
    out2a = pt2.update(_state(0, [r2], inventory=[ball_ahead], pose=_VALID_POSE, phase="teleop"))
    assert out2a[1]["phase"] != INTAKING, "proximity-in-heading alone is NOT intaking (gate)"
    # frame 1: that ball crosses INTO the box (consumed) -> INTAKING fires
    ball_in = Track(tid=51, cls="ball", xyxy=(515, 495, 525, 505), centroid_px=(520, 500))
    out2b = pt2.update(_state(1, [r2], inventory=[ball_in], pose=_VALID_POSE, phase="teleop"))
    assert out2b[1]["phase"] == INTAKING, "a ball consumed into the box must read INTAKING"


def test_hold_from_bin_false_litter_overlap_is_not_holding():
    """The dense-field HOLDING fix: with hold_from_bin=False (deploy default), a robot PARKED
    OVER loose balls (bbox HELD cue overlapping litter) but with NO tracked field-meter intake
    reads TRAVERSING, not HOLDING. Only TRUE tracked possession (held_count>0) lights HOLDING."""
    from frcscout.refine.collector_bin import CollectorBin

    binloc = CollectorBin(regions={"blue": [(0.80, 0.30), (0.90, 0.30),
                                            (0.90, 0.40), (0.80, 0.40)]}, frame_wh=(1920, 1080))
    pt = PhaseTracker(field_model=StubField(), hub_localizer=NeverInHub(),
                      cfg={"enable_bin_scorer": True, "held_confirm_s": 0.0,
                           "hold_from_bin": False}, fps=FPS, collector_bin=binloc)
    pt.bin_scorer.set_frame_wh(1920, 1080)

    # robot sitting still over loose balls (centres inside its bbox) — pure litter overlap,
    # no field-meter intake ever ran (these balls have no field_xy_m / never converged+vanished).
    r = _robot(1, (5.0, 4.0), alliance="blue", vel=(0.0, 0.0), box=(400, 400, 520, 520))
    litter = [Track(tid=50, cls="ball", xyxy=(455, 455, 465, 465), centroid_px=(460, 460)),
              Track(tid=51, cls="ball", xyxy=(470, 470, 480, 480), centroid_px=(475, 475))]
    out = None
    for f in range(5):
        out = pt.update(_state(f, [r], inventory=litter, pose=_VALID_POSE, phase="teleop"))
    assert out[1]["phase"] == TRAVERSING, \
        "a robot parked over litter (no tracked intake) must read TRAVERSING, not HOLDING"
    assert out[1]["held"] == 0, "displayed held is litter-free (tracked possession only)"

    # sanity: the SAME scene with hold_from_bin=True (clean/synthetic default) DOES read HOLDING
    pt2 = PhaseTracker(field_model=StubField(), hub_localizer=NeverInHub(),
                       cfg={"enable_bin_scorer": True, "held_confirm_s": 0.0,
                            "hold_from_bin": True}, fps=FPS, collector_bin=binloc)
    pt2.bin_scorer.set_frame_wh(1920, 1080)
    out2 = pt2.update(_state(0, [r], inventory=litter, pose=_VALID_POSE, phase="teleop"))
    assert out2[1]["phase"] == HOLDING, "with hold_from_bin=True the bbox cue lights HOLDING"


def test_hold_from_bin_false_still_holds_on_tracked_possession():
    """With hold_from_bin=False, a REAL tracked intake (field-meter converge->vanish) still
    reads HOLDING — the conservative gate suppresses only litter, never true possession."""
    field = StubField()
    pt = _mk(field, cfg={"hold_from_bin": False})
    robot = _robot(1, (5.0, 4.0), alliance="blue")
    f, _ = _drive_in(pt, robot, pose=_VALID_POSE)        # confirmed field-meter intake
    out = None
    for _ in range(int(pt.intake_hold_frames) + 2):       # step past the INTAKING latch
        out = pt.update(_state(f, [robot], pose=_VALID_POSE, phase="teleop"))
        f += 1
    assert out[1]["phase"] == HOLDING
    assert out[1]["held"] == 1


def test_bin_scorer_exposes_score_candidates_and_tallies():
    """The PhaseTracker surfaces bin-entry SCORE candidates + per-alliance bin tallies from
    the embedded BinScorer (never the authoritative count — candidates with confidence)."""
    from frcscout.refine.collector_bin import CollectorBin

    binloc = CollectorBin(regions={"blue": [(0.10, 0.30), (0.20, 0.30),
                                            (0.20, 0.40), (0.10, 0.40)]}, frame_wh=(1920, 1080))
    pt = PhaseTracker(field_model=StubField(), hub_localizer=NeverInHub(),
                      cfg={"enable_bin_scorer": True, "held_confirm_s": 0.0}, fps=FPS,
                      collector_bin=binloc)
    pt.bin_scorer.set_frame_wh(1920, 1080)
    pt.bin_scorer.set_hub_active("blue", True)

    # robot away from the bin; ball held, shot fast, then enters the bin clear of the box.
    r = Track(tid=1, cls="robot", xyxy=(550, 430, 650, 530), centroid_px=(600, 480),
              field_xy_m=(5.0, 4.0), alliance="blue", vel_mps=(0.0, 0.0))
    pt.update(_state(0, [r], inventory=[Track(tid=10, cls="ball", xyxy=(595, 475, 605, 485),
                                              centroid_px=(600, 480))], pose=_VALID_POSE, phase="teleop"))
    pt.update(_state(1, [r], flights=[Track(tid=10, cls="ball", xyxy=(495, 435, 505, 445),
                                            centroid_px=(500, 440), is_flight=True)],
                     pose=_VALID_POSE, phase="teleop"))
    pt.update(_state(2, [r], flights=[Track(tid=10, cls="ball", xyxy=(283, 373, 293, 383),
                                            centroid_px=(288, 378), is_flight=True)],
                     pose=_VALID_POSE, phase="teleop"))
    cands = pt.bin_score_events()
    assert any(c["alliance"] == "blue" and c["points"] == 1 for c in cands)
    assert pt.bin_tallies()["blue"]["bin_scores"] >= 1
    assert pt.bin_tallies()["blue"]["points"] >= 1
    json.dumps(cands)            # candidates are JSON-serializable
    json.dumps(pt.bin_tallies())


def test_multi_robot_independent_phases():
    """Six robots tracked at once keep independent phases/metrics in one update()."""
    field = StubField()
    pt = _mk(field)
    robots = [
        _robot(1, (3.0, 4.0), alliance="blue", vel=(1.0, 0.0)),   # traversing
        _robot(2, (13.0, 2.0), alliance="blue"),                  # in red zone...
        _robot(3, (13.3, 2.0), alliance="red"),                   # ...beside this red -> #2 DEFENSE
        _robot(4, (5.0, 6.0), alliance="red", vel=(0.5, 0.5)),    # traversing
        _robot(5, (8.0, 4.0), alliance="red"),                    # traversing
        _robot(6, (2.0, 7.0), alliance="blue"),                   # traversing
    ]
    out = pt.update(_state(0, robots, pose=_VALID_POSE, phase="teleop"))
    assert len(out) == 6
    assert out[2]["phase"] == DEFENSE
    assert out[1]["phase"] == TRAVERSING
    # all ids present and distinct
    assert set(out.keys()) == {1, 2, 3, 4, 5, 6}
