"""COLLECTOR BIN (HUB) localization + ball<->robot<->bin interaction tests.

Pure synthetic — no model / cv2 / GPU. We feed hand-built FrameState sequences (Track
objects with pixel boxes + centroids + velocities) into ``BinScorer`` and assert the
user's EXACT spec:

  * INTAKING  — ball within radius R AND in the robot's DIRECTION OF TRAVEL (velocity
                dot-product gate). A ball off to the side of a moving robot is NOT intake.
  * HELD      — a ball centre inside the robot's bounding box.
  * SHOOTING  — a HELD ball leaving the box at HIGH exit speed (a real shot), discriminated
                from a slow drop / loss of grip (LOST) by exit speed.
  * SCORING   — a post-shot ball entering the bin bounds awards the FUEL point value, and
                is NEVER counted while the ball is still inside a robot bounding box.

Plus: bin localization (image-polygon point-in-polygon + which_bin), HUB active/inactive
point gating (manual Table 6-4: 1 pt active, 0 inactive), and JSON-serializability.
"""
from __future__ import annotations

import json
import math

from frcscout.io.types import FrameState, OverlayState, Track
from frcscout.refine.collector_bin import (
    BinScorer, CollectorBin, FUEL_POINTS_ACTIVE, FUEL_POINTS_INACTIVE,
)

FPS = 60.0
WH = (1920, 1080)

# A tiny, easy-to-reason-about bin: blue bin is a square near the LEFT of the frame in
# fraction space; red bin near the RIGHT. (px: blue ~[192..384]x[324..432], red ~[1536..1728].)
_BINS = {
    "blue": [(0.10, 0.30), (0.20, 0.30), (0.20, 0.40), (0.10, 0.40)],
    "red":  [(0.80, 0.30), (0.90, 0.30), (0.90, 0.40), (0.80, 0.40)],
}


# ───────────────────────────── builders ──────────────────────────────────────
def _robot(tid, cx, cy, alliance="blue", w=120.0, h=120.0, vel_px=None, vel_mps=None):
    box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    r = Track(tid=tid, cls="robot", xyxy=box, centroid_px=(cx, cy), alliance=alliance)
    if vel_px is not None:
        r.vel_px = vel_px
    if vel_mps is not None:
        r.vel_mps = vel_mps
    return r


def _ball(tid, cx, cy, flight=False):
    return Track(tid=tid, cls="ball", xyxy=(cx - 5, cy - 5, cx + 5, cy + 5),
                 centroid_px=(cx, cy), is_flight=flight)


def _state(fidx, robots, inv=None, flights=None, phase="teleop"):
    return FrameState(frame_idx=fidx, t_s=fidx / FPS, robots=robots,
                      balls_inventory=inv or [], balls_flight=flights or [],
                      overlay=OverlayState(phase=phase))


def _mk(cfg=None, hub_active=None):
    binloc = CollectorBin(regions=_BINS, frame_wh=WH)
    # Default these unit tests to LEGACY cone-membership intake (the dot-gate discriminator
    # tested in isolation here): require_consumption_for_intake=False so the per-frame
    # ``intaking`` signal reflects pure cone-membership. The dense-field CONSUMPTION gate
    # (the new default, on in deploy.yaml + PhaseTracker) is covered by its own tests below
    # and the phase-layer tests.
    fast = {"intake_confirm_s": 0.0, "held_confirm_s": 0.0,
            "require_consumption_for_intake": False}
    if cfg:
        fast.update(cfg)
    bs = BinScorer(bin_loc=binloc, cfg=fast, fps=FPS)
    bs.set_frame_wh(*WH)
    if hub_active:
        for al, v in hub_active.items():
            bs.set_hub_active(al, v)
    return bs


# ═══════════════════════════ bin localization ════════════════════════════════
def test_which_bin_image_polygon():
    """which_bin returns the right alliance for a point inside each opening, None outside."""
    b = CollectorBin(regions=_BINS, frame_wh=WH)
    # centre of blue bin ~ (0.15,0.35) -> (288, 378)
    assert b.which_bin((288, 378), frame_wh=WH) == "blue"
    # centre of red bin ~ (0.85,0.35) -> (1632, 378)
    assert b.which_bin((1632, 378), frame_wh=WH) == "red"
    # mid-field -> neither
    assert b.which_bin((960, 540), frame_wh=WH) is None
    assert b.in_bin((288, 378), "blue", frame_wh=WH)
    assert not b.in_bin((288, 378), "red", frame_wh=WH)


def test_default_regions_are_measured_qual59():
    """The default CollectorBin carries the measured qual59 HUB regions (two alliances)."""
    b = CollectorBin(frame_wh=WH)
    assert set(b.alliances()) == {"blue", "red"}
    # blue opening is on the LEFT third, red on the RIGHT third (sanity on the measured frac)
    bx = sum(p[0] for p in b._frac["blue"]) / 4
    rx = sum(p[0] for p in b._frac["red"]) / 4
    assert bx < 0.4 < 0.6 < rx


# ═══════════════════════════ INTAKING (dot gate) ═════════════════════════════
def test_intaking_requires_ball_in_direction_of_travel():
    """A ball within radius AND ahead of a robot moving toward it -> INTAKING. The SAME
    ball geometry but with the robot moving the OTHER way (ball behind) -> NOT intaking."""
    # robot at (500,500) moving +x; ball just ahead at (560,500), within radius.
    bs = _mk()
    r = _robot(1, 500, 500, vel_px=(40.0, 0.0))
    ball_ahead = _ball(10, 580, 500)   # ahead (+x), outside the 60px-half box, within radius
    bs.step(_state(0, [r], inv=[ball_ahead]))
    sig = bs.robot_signals()
    assert sig.get(1, {}).get("intaking") is True, "ball ahead of a forward-moving robot is intake"

    # now the ball is BEHIND the +x-moving robot (same distance) -> dot gate rejects it
    bs2 = _mk()
    r2 = _robot(1, 500, 500, vel_px=(40.0, 0.0))
    ball_behind = _ball(11, 420, 500)  # behind (-x)
    bs2.step(_state(0, [r2], inv=[ball_behind]))
    assert not bs2.robot_signals().get(1, {}).get("intaking", False), \
        "a ball behind a forward-moving robot must NOT read as intaking (dot gate)"


def test_intaking_stationary_robot_uses_proximity_only():
    """A (near-)stationary robot has no travel direction, so a ball within radius on ANY
    side reads INTAKING (proximity gate only — the dot gate is skipped). LEGACY mode."""
    bs = _mk()
    r = _robot(1, 500, 500, vel_px=(0.0, 0.0))   # not moving
    ball_behind = _ball(10, 430, 500)            # behind, but robot isn't moving
    bs.step(_state(0, [r], inv=[ball_behind]))
    assert bs.robot_signals().get(1, {}).get("intaking") is True


# ═══════════════ CONSUMPTION-gated INTAKING (the dense-field default) ══════════
def test_consumption_gate_proximity_alone_is_not_intaking():
    """With require_consumption_for_intake (the DEFAULT + deploy), a ball merely IN the
    cone of a robot driving past does NOT raise the intaking signal — that is exactly the
    dense-field over-fire we kill (every robot always has a ball in its heading)."""
    binloc = CollectorBin(regions=_BINS, frame_wh=WH)
    bs = BinScorer(bin_loc=binloc, cfg={"intake_confirm_s": 0.0, "held_confirm_s": 0.0},
                   fps=FPS)                      # consumption gate ON (default)
    bs.set_frame_wh(*WH)
    r = _robot(1, 500, 500, vel_px=(40.0, 0.0))
    ball_ahead = _ball(10, 580, 500)             # in the cone, but never taken in
    bs.step(_state(0, [r], inv=[ball_ahead]))
    assert not bs.robot_signals().get(1, {}).get("intaking", False), \
        "proximity-in-heading alone must NOT read intaking under the consumption gate"


def test_consumption_gate_fires_on_cone_then_box_crossing():
    """The consumption gate DOES fire when a ball crosses from the robot's tracked approach
    cone INTO its box (a real pickup), and latches briefly so the 1-frame event is visible."""
    binloc = CollectorBin(regions=_BINS, frame_wh=WH)
    bs = BinScorer(bin_loc=binloc, cfg={"intake_confirm_s": 0.0, "held_confirm_s": 0.0,
                                        "intake_signal_hold_s": 0.2}, fps=FPS)
    bs.set_frame_wh(*WH)
    r = _robot(1, 500, 500, vel_px=(40.0, 0.0), w=120, h=120)   # box [440..560]
    # frame 0: ball ahead in the cone (outside the box) -> tracked approach, no signal yet
    bs.step(_state(0, [r], inv=[_ball(10, 600, 500)]))
    assert not bs.robot_signals().get(1, {}).get("intaking", False)
    # frame 1: that SAME ball is now inside the box (consumed) -> intaking signal fires
    evs_sig = bs.step(_state(1, [r], inv=[_ball(10, 520, 500)]))   # 520 is inside [440..560]
    assert bs.robot_signals().get(1, {}).get("intaking") is True, \
        "a ball crossing the cone INTO the box is a confirmed consumption -> intaking"
    assert bs.tallies()["blue"]["intakes"] == 1
    # latch keeps it visible the next frame even though the ball is now just held
    bs.step(_state(2, [r], inv=[_ball(10, 520, 500)]))
    assert bs.robot_signals().get(1, {}).get("intaking") is True, "intaking latch holds briefly"


# ═══════════════════════════ HELD (inside bbox) ══════════════════════════════
def test_held_when_ball_inside_box():
    """A ball whose centre is inside the robot box reads HELD (the robot carries it)."""
    bs = _mk()
    r = _robot(1, 500, 500, w=120, h=120)        # box [440..560]x[440..560]
    ball_inside = _ball(10, 505, 495)            # inside the box
    bs.step(_state(0, [r], inv=[ball_inside]))
    sig = bs.robot_signals()
    assert sig.get(1, {}).get("held_balls") == [10]
    assert bs.ball_states()[10] == "HELD"
    assert bs.tallies()["blue"]["intakes"] == 1   # a fresh possession counts an intake


def test_held_persists_when_swallowed_ball_vanishes():
    """Once HELD, a ball that the detector loses (swallowed) stays HELD (invisible), so
    HOLDING persists — same honesty as the field-meter scorer."""
    bs = _mk()
    r = _robot(1, 500, 500)
    bs.step(_state(0, [r], inv=[_ball(10, 505, 500)]))   # HELD
    # ball disappears, robot still present
    bs.step(_state(1, [r], inv=[]))
    assert bs.ball_states().get(10) == "HELD"
    assert bs.robot_signals().get(1, {}).get("held_balls") == [10]


# ═══════════════════════ SHOOTING vs LOSS (exit speed) ════════════════════════
def test_shooting_on_fast_exit_then_score_into_bin():
    """A HELD ball that leaves the box FAST -> SHOT; when it then enters the bin (and is
    OUTSIDE every robot box) it SCORES 1 pt (active HUB) to the shooter's alliance."""
    bs = _mk(hub_active={"blue": True})
    # blue robot a bit AWAY from the blue bin (bin centre 288,378) so the fast exit frame
    # is NOT already over the bin -> we observe SHOT, then SCORE on the bin-entry frame.
    r = _robot(1, 600, 480, alliance="blue", w=100, h=100)   # box ~[550..650]x[430..530]
    # frame 0: ball held inside the box
    bs.step(_state(0, [r], inv=[_ball(10, 600, 480)]))
    assert bs.ball_states()[10] == "HELD"
    # frame 1: ball leaves the box FAST, heading toward the bin but not yet over it.
    bs.step(_state(1, [r], flights=[_ball(10, 500, 440, flight=True)]))
    assert bs.ball_states()[10] == "SHOT", f"a fast exit (not yet at bin) should be SHOT"
    assert bs.tallies()["blue"]["shots"] == 1
    # frame 2: ball squarely inside the blue bin (288,378), clear of the robot box -> SCORE
    evs = bs.step(_state(2, [r], flights=[_ball(10, 288, 378, flight=True)]))
    assert bs.ball_states()[10] == "SCORED"
    assert bs.tallies()["blue"]["bin_scores"] == 1
    assert bs.tallies()["blue"]["points"] == FUEL_POINTS_ACTIVE == 1
    # an event was emitted, crediting blue, with hub_active known True
    assert any(e.alliance == "blue" and e.points == 1 and e.hub_active and e.hub_active_known
               for e in evs)


def test_slow_exit_below_threshold_is_loss():
    """Unambiguous slow-exit case: exit speed below the shot threshold -> LOST, no score."""
    # box [440..560] (+6px held pad -> padded right edge 566). Place the ball at the edge,
    # then step it CLEARLY past the padded edge but SLOWLY (below the shot threshold).
    bs = _mk(cfg={"shot_min_exit_speed_px_s": 1000.0})  # very high bar so our small step is 'slow'
    r = _robot(1, 500, 500, alliance="blue", w=120, h=120)
    bs.step(_state(0, [r], inv=[_ball(10, 560, 500)]))   # at the right edge -> HELD
    assert bs.ball_states()[10] == "HELD"
    # step 1: ball at 572 (clear of the padded 566 edge), moved 12px/frame = 720 px/s < 1000 -> LOSS
    bs.step(_state(1, [r], inv=[_ball(10, 572, 500)]))
    assert bs.ball_states()[10] == "LOST"
    assert bs.tallies()["blue"]["shots"] == 0
    assert bs.tallies()["blue"]["losses"] == 1


# ════════════════════ the HARD rule: never score inside a box ═════════════════
def test_never_scores_while_ball_inside_robot_box():
    """Even if a (shot) ball's centre lies inside the bin polygon, it must NOT score while
    that ball is also inside a robot's bounding box (the user's hard rule)."""
    bs = _mk(hub_active={"blue": True})
    # robot positioned so its box COVERS the blue bin centre (288,378).
    r = _robot(1, 288, 378, alliance="blue", w=160, h=160)   # box [208..368]x[298..458] covers bin
    # ball held, then 'shot' but still within the (huge) box that overlaps the bin
    bs.step(_state(0, [r], inv=[_ball(10, 288, 378)]))        # HELD
    # fast exit but staying inside the box (move within box): from (288,378) to (340,420)
    bs.step(_state(1, [r], flights=[_ball(10, 340, 420, flight=True)]))
    # ball at the bin centre BUT still inside the robot box -> NO score
    evs = bs.step(_state(2, [r], flights=[_ball(10, 288, 378, flight=True)]))
    assert bs.tallies()["blue"]["bin_scores"] == 0, "scoring while inside a robot box is forbidden"
    assert not evs


# ═══════════════════════ HUB active/inactive point gating ════════════════════
def test_inactive_hub_scores_zero_points():
    """A ball into an INACTIVE hub still counts as a bin_score (tracked) but awards 0
    points (manual §6.4.1 / Table 6-4)."""
    bs = _mk(hub_active={"blue": False})
    r = _robot(1, 360, 420, alliance="blue", w=100, h=100)
    bs.step(_state(0, [r], inv=[_ball(10, 360, 420)]))
    bs.step(_state(1, [r], flights=[_ball(10, 300, 390, flight=True)]))
    bs.step(_state(2, [r], flights=[_ball(10, 288, 378, flight=True)]))
    t = bs.tallies()["blue"]
    assert t["bin_scores"] == 1
    assert t["points"] == FUEL_POINTS_INACTIVE == 0


def test_unknown_hub_assumed_active_but_flagged():
    """With HUB-active unknown, a score is awarded 1 pt (assume active) but the event is
    flagged hub_active_known=False so the assumption is never hidden."""
    bs = _mk()  # no set_hub_active
    r = _robot(1, 600, 480, alliance="blue", w=100, h=100)
    bs.step(_state(0, [r], inv=[_ball(10, 600, 480)]))
    bs.step(_state(1, [r], flights=[_ball(10, 500, 440, flight=True)]))   # SHOT (not at bin yet)
    evs = bs.step(_state(2, [r], flights=[_ball(10, 288, 378, flight=True)]))  # enters bin
    assert bs.tallies()["blue"]["points"] == 1
    assert evs and evs[0].hub_active is True and evs[0].hub_active_known is False


# ═══════════════════════════ serialization ═══════════════════════════════════
def test_outputs_json_serializable():
    """tallies + ball_states + robot_signals + the score events round-trip through JSON."""
    bs = _mk(hub_active={"red": True})
    r = _robot(1, 1632, 378, alliance="red", w=100, h=100)   # at the red bin
    bs.step(_state(0, [r], inv=[_ball(10, 1632, 378)]))
    evs = bs.step(_state(1, [r], flights=[_ball(10, 1600, 378, flight=True)]))
    json.dumps(bs.tallies())
    json.dumps(bs.ball_states())
    json.dumps(bs.robot_signals())
    json.dumps([e.to_event().detail for e in evs])
