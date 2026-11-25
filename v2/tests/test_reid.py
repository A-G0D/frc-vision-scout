"""Re-ID / persistence tests — pure synthetic, no model / cv2 / easyocr / GPU.

Covers the three guarantees of the persistence layer:
  * a robot lost for LONGER than max_age, reappearing with the same appearance +
    a plausible field/pixel position -> the SAME id is rebound (not a fresh one);
  * a robot of a DIFFERENT color reappearing -> a NEW id (hard alliance gate);
  * a bumper-number anchor binds id<->team and survives an arbitrary gap.

Appearance histograms are injected directly (frame=None path), and the OCR hook
is a pure-python stub, so nothing imports cv2 or easyocr.
"""
from __future__ import annotations

import numpy as np

from frcscout.io.types import Detection
from frcscout.track.bytetrack import RobotTracker
from frcscout.track.reid import (Appearance, ReIDManager, appearance_similarity)


FPS = 60.0


def _box(cx, cy, w=80, h=80, conf=0.9, frame=0, cls="robot"):
    return Detection(cls, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), conf, frame)


def _hist(peak_bin, bins=12):
    """A normalized 1-hot-ish hue histogram peaked at `peak_bin`."""
    h = np.zeros(bins, dtype=np.float32)
    h[peak_bin % bins] = 1.0
    return h


# ───────────────────────── manager-level unit tests ──────────────────────────
def test_appearance_similarity_alliance_gate():
    red = Appearance(alliance="red", hue_hist=_hist(0), size=(80, 80))
    blue = Appearance(alliance="blue", hue_hist=_hist(0), size=(80, 80))
    # identical histogram but different alliance -> hard 0
    assert appearance_similarity(red, blue) == 0.0
    # same alliance, same hist -> high
    red2 = Appearance(alliance="red", hue_hist=_hist(0), size=(80, 80))
    assert appearance_similarity(red, red2) > 0.9


def test_rebind_same_appearance_same_position():
    """A lost track reappearing nearby with a matching histogram is rebound."""
    mgr = ReIDManager({"lost_ttl_s": 4.0}, fps=FPS)
    app = Appearance(alliance="red", hue_hist=_hist(3), size=(80, 80))
    mgr.note_lost(tid=7, appearance=app, last_centroid_px=(500, 300),
                  vel_px_per_frame=(0.0, 0.0), last_field_xy=None, last_frame=100)
    # reappears 30 frames later (0.5s) a little away, same appearance
    tid = mgr.rebind(alliance="red", xyxy=(470, 270, 550, 350),
                     centroid_px=(510, 305), frame_idx=130,
                     appearance=Appearance(alliance="red", hue_hist=_hist(3), size=(80, 80)))
    assert tid == 7
    # the lost entry is consumed
    assert mgr.lost_ids() == []


def test_rebind_rejects_different_color():
    """A blue detection never rebinds a red lost track -> None (new id upstream)."""
    mgr = ReIDManager({}, fps=FPS)
    app = Appearance(alliance="red", hue_hist=_hist(3), size=(80, 80))
    mgr.note_lost(tid=7, appearance=app, last_centroid_px=(500, 300),
                  vel_px_per_frame=(0.0, 0.0), last_field_xy=None, last_frame=100)
    tid = mgr.rebind(alliance="blue", xyxy=(470, 270, 550, 350),
                     centroid_px=(510, 305), frame_idx=130,
                     appearance=Appearance(alliance="blue", hue_hist=_hist(3), size=(80, 80)))
    assert tid is None


def test_rebind_rejects_implausible_position():
    """Same color but teleported across the frame -> not rebound."""
    mgr = ReIDManager({"base_gate_px": 40.0, "per_frame_gate_px": 9.0,
                       "max_gate_px": 400.0}, fps=FPS)
    app = Appearance(alliance="red", hue_hist=_hist(3), size=(80, 80))
    mgr.note_lost(tid=7, appearance=app, last_centroid_px=(100, 100),
                  vel_px_per_frame=(0.0, 0.0), last_field_xy=None, last_frame=100)
    # 5 frames later but 2000px away — way outside the gate
    tid = mgr.rebind(alliance="red", xyxy=(1980, 980, 2060, 1060),
                     centroid_px=(2020, 1020), frame_idx=105,
                     appearance=Appearance(alliance="red", hue_hist=_hist(3), size=(80, 80)))
    assert tid is None


def test_anchor_binds_team_and_survives_gap():
    """A bumper-number anchor permanently binds id<->team and is kept past the TTL."""
    mgr = ReIDManager({"lost_ttl_s": 0.1}, fps=FPS)  # tiny TTL
    assert mgr.anchor(7, 1678) == 1678
    assert mgr.team_for(7) == 1678
    assert mgr.track_of_team[1678] == 7
    # park it lost, then expire well past the TTL — anchored entries survive
    app = Appearance(alliance="red", hue_hist=_hist(3), size=(80, 80))
    mgr.note_lost(tid=7, appearance=app, last_centroid_px=(500, 300),
                  vel_px_per_frame=(0.0, 0.0), last_field_xy=None, last_frame=100)
    mgr.expire(frame_idx=100 + 10_000)   # ~166s later, way past 0.1s TTL
    assert 7 in mgr.lost_ids(), "anchored lost track must survive its TTL"
    # and the permanent registry is intact
    assert mgr.team_for(7) == 1678


def test_anchor_via_ocr_hook_stub():
    """The OCR hook binds id<->team without easyocr (stubbed)."""
    mgr = ReIDManager({}, fps=FPS, ocr_fn=lambda crop: 254)
    # frame is a dummy array-like; _crop just slices it. Use a tiny ndarray.
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    num = mgr.try_anchor(tid=3, xyxy=(10, 10, 90, 90), frame=frame)
    assert num == 254
    assert mgr.team_for(3) == 254
    # idempotent — no re-read needed
    assert mgr.try_anchor(tid=3, xyxy=(10, 10, 90, 90), frame=frame) == 254


def test_first_writer_wins_for_team():
    mgr = ReIDManager({}, fps=FPS)
    assert mgr.anchor(7, 1678) == 1678
    # a different track id can't steal team 1678
    assert mgr.anchor(9, 1678) is None
    assert mgr.track_of_team[1678] == 7


# ──────────────────── integration with RobotTracker ──────────────────────────
def test_tracker_rebinds_id_after_long_gap():
    """End-to-end: a robot tracked, occluded LONGER than max_age (so the Kalman
    track is culled), then reappearing nearby -> the tracker rebinds the same id
    via the ReIDManager rather than minting a new one."""
    # max_age 0.2s -> 12 frames @60; we inject a 20-frame gap (longer than max_age)
    reid = ReIDManager({"lost_ttl_s": 4.0, "base_gate_px": 60.0,
                        "per_frame_gate_px": 12.0, "min_appearance": 0.0}, fps=FPS)
    trk = RobotTracker({"iou_match": 0.3, "max_age_s": 0.2}, fps=FPS, reid=reid)
    assert trk.max_age == 12

    first_id = None
    cx = 400
    # phase 1: 10 frames present
    for f in range(10):
        out = trk.update(f, [_box(cx, 300, frame=f)])
        assert len(out) == 1
        first_id = out[0].tid

    # phase 2: 20-frame gap (no detections) — exceeds max_age=12, track culled
    for f in range(10, 30):
        trk.update(f, [])

    # the live track must be gone (culled), and the id parked as lost
    assert first_id in reid.lost_ids()

    # phase 3: reappears near where it was -> rebound to the SAME id
    out = trk.update(30, [_box(cx + 10, 305, frame=30)])
    assert len(out) == 1
    assert out[0].tid == first_id, (
        f"expected rebind to id {first_id}, got {out[0].tid}")


def test_tracker_without_reid_is_unchanged():
    """With no ReIDManager the tracker keeps the original behaviour: a gap longer
    than max_age culls the track and a reappearance gets a NEW id."""
    trk = RobotTracker({"iou_match": 0.3, "max_age_s": 0.2}, fps=FPS)
    first_id = None
    for f in range(10):
        out = trk.update(f, [_box(400, 300, frame=f)])
        first_id = out[0].tid
    for f in range(10, 30):
        trk.update(f, [])
    out = trk.update(30, [_box(410, 305, frame=30)])
    assert len(out) == 1
    assert out[0].tid != first_id, "without re-ID a long gap must mint a new id"
