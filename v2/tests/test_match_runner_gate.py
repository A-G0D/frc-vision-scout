"""MatchRunner field-gate wiring + note_overlay_score passthrough + sixbot honesty.

Synthetic FrameStates only (no model / cv2 / GPU). We use the REAL FieldModel as the gate
(it is pure-python for back-projection) with a known affine image->field homography, prove
the crowd box never reaches the 6-robot prior, prove note_overlay_score reaches the embedded
scorer, and prove a slot that loses in-field evidence is no longer asserted as confident /
is eventually dropped (R9 honesty).
"""
from __future__ import annotations

from frcscout.geometry.field import FieldModel
from frcscout.io.types import CameraPose, FrameState, OverlayState, Track
from frcscout.pipeline.match_runner import MatchRunner
from frcscout.refine.sixbot import SixRobotTracker
from frcscout.io.types import Detection

FPS = 60.0
FIELD_CFG = "configs/field_2026_rebuilt.yaml"


def _affine_pose(a=0.02, b=0.012, tx=-4.0, ty=-2.4, valid=True):
    H = [[a, 0.0, tx], [0.0, b, ty], [0.0, 0.0, 1.0]]
    return CameraPose(H=H, valid=valid, n_tags=4, reproj_err_px=0.5)


def _robot(tid, cx, y2, alliance, w=60.0, h=80.0, vel=(0.0, 0.0)):
    x1, x2 = cx - w / 2.0, cx + w / 2.0
    y1 = y2 - h
    cy = (y1 + y2) / 2.0
    t = Track(tid=tid, cls="robot", xyxy=(x1, y1, x2, y2), centroid_px=(cx, cy),
              alliance=alliance, vel_mps=vel)
    t.conf = 1.0
    return t


def _runner(gate_cfg=None):
    fm = FieldModel(FIELD_CFG)
    cfg = {"gate": {"main_pane_frac": 0.52, "frame_h": 1080.0, "tol_px": 24.0}}
    if gate_cfg:
        cfg["gate"].update(gate_cfg)
    mr = MatchRunner(field_model=fm, hub_localizer=fm, cfg=cfg, fps=FPS)
    return mr


def _state(f, robots, pose):
    return FrameState(frame_idx=f, t_s=f / FPS, robots=robots,
                      overlay=OverlayState(phase="teleop"), pose=pose)


# ───────────────────── (2) MatchRunner gates before the prior ─────────────────
def test_crowd_box_never_reaches_prior():
    """Six on-field robots + a crowd 'robot' (a person far right of the field). The crowd
    box back-projects off-field and is dropped BEFORE sixbot, so the prior emits exactly
    six and the crowd box is not among them."""
    mr = _runner()
    pose = _affine_pose()
    # 3 red + 3 blue on the field (floor-contact y2 ~ 450, in the main pane)
    base = [(300, "red"), (600, "red"), (900, "red"),
            (350, "blue"), (650, "blue"), (950, "blue")]
    robots = [_robot(10 + i, cx, 450.0, ally) for i, (cx, ally) in enumerate(base)]
    # a CROWD box masquerading as a red robot, far right (image x 1600 -> field ~28 m)
    crowd = _robot(999, 1600.0, 450.0, "red")
    res = mr.process(_state(0, robots + [crowd], pose))
    st = res["state"]
    assert len(st.robots) == 6, f"crowd box inflated the prior to {len(st.robots)}"
    assert sum(r.alliance == "red" for r in st.robots) == 3
    assert sum(r.alliance == "blue" for r in st.robots) == 3
    # no stable track sits where the crowd box was (image x ~1600)
    assert all(r.centroid_px[0] < 1400 for r in st.robots)


def test_person_box_rejected_red_slot_not_minted():
    """A single frame with only TWO real blue robots + a 'red' detection that is actually a
    person off the field: the person must NOT mint a red track (the gate drops it first)."""
    mr = _runner()
    pose = _affine_pose()
    real = [_robot(1, 400.0, 450.0, "blue"), _robot(2, 700.0, 450.0, "blue")]
    person = _robot(3, 60.0, 450.0, "red")  # image x 60 -> field ~ -2.8 m (off field, left)
    res = mr.process(_state(0, real + [person], pose))
    st = res["state"]
    assert all(r.alliance == "blue" for r in st.robots), "a person minted a red robot slot"
    assert len(st.robots) == 2


def test_gate_noop_when_pose_absent():
    """With no pose on the state, the gate is a no-op and the prior sees every detection
    (we never blank the prior just because a pose is missing)."""
    mr = _runner()
    base = [(300, "red"), (600, "red"), (900, "red"),
            (350, "blue"), (650, "blue"), (950, "blue")]
    robots = [_robot(10 + i, cx, 450.0, ally) for i, (cx, ally) in enumerate(base)]
    res = mr.process(_state(0, robots, pose=None))
    # all six still tracked (nothing dropped for a missing pose)
    assert len(res["state"].robots) == 6


# ─────────────────── (2) note_overlay_score passthrough ───────────────────────
def test_note_overlay_score_reaches_scorer():
    mr = _runner()
    assert mr.phase.scoring._score_pulses == []
    mr.note_overlay_score(120, "blue")
    mr.note_overlay_score(121, "red")
    assert (120, "blue") in mr.phase.scoring._score_pulses
    assert (121, "red") in mr.phase.scoring._score_pulses


# ─────────────────── (3) sixbot honesty: confident + drop ─────────────────────
def _det(cx, cy, frame, w=60.0, h=60.0):
    return Detection("robot", (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), 1.0, frame)


def test_coasting_track_loses_confidence_then_drops():
    """A robot seen briefly then gone: within confident_grace it is confident; past it the
    emitted slot is confident=False (we don't CLAIM a measured robot without evidence); past
    drop_after the slot is dropped entirely (we don't invent six forever)."""
    cfg = {"confident_grace_s": 0.05, "drop_after_s": 0.2}  # 3 frames / 12 frames @60
    trk = SixRobotTracker(cfg, fps=FPS)
    assert trk.confident_grace == 3 and trk.drop_after == 12

    # establish one red track over a few frames
    for f in range(5):
        out = trk.update(f, [_det(300, 200, f)], ["red"])
    assert len(out) == 1 and out[0]["confident"] is True

    # now it vanishes; track coasts. Just past confident_grace -> confident drops to False.
    for f in range(5, 10):  # 5 coasting frames (> 3 grace, < 12 drop)
        out = trk.update(f, [])
    assert len(out) == 1, "track dropped too early (within drop_after it must persist)"
    assert out[0]["est"] is True
    assert out[0]["confident"] is False, "a long-coasting slot must not be claimed confident"

    # keep coasting past drop_after -> the phantom slot is dropped (honest: no evidence).
    for f in range(10, 25):
        out = trk.update(f, [])
    assert len(out) == 0, "a slot with no evidence past drop_after must be dropped, not invented"


def test_confident_true_while_measured():
    """A continuously-measured track is always confident=True and never dropped."""
    trk = SixRobotTracker({}, fps=FPS)
    for f in range(40):
        out = trk.update(f, [_det(300 + f, 200, f)], ["red"])
    assert len(out) == 1
    assert out[0]["confident"] is True and out[0]["est"] is False
