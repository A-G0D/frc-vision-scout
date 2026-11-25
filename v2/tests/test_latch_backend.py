"""Real integration test for the optional latch-tracker backend on RobotTracker.

Builds actual BGR frames (a textured square moving across a synthetic field) and
drives RobotTracker with a detection gap long enough to exceed max_age. Without
the latch backend the track is culled and the reappearing robot gets a NEW id
(the documented ByteTrack weakness — see reid.py). With the latch backend
enabled, the visual tracker keeps finding the (still-visible, just undetected)
square every frame of the gap, so the SAME id survives straight through.

Skips (does not fail) if latch-tracker isn't checked out / cv2 lacks TrackerVit —
the backend must degrade gracefully, so absence of the sibling project is not a
test failure for the primary repo.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from frcscout.io.types import Detection
from frcscout.track.bytetrack import RobotTracker
from frcscout.track.latch_backend import LatchBackend, latch_available

FPS = 30.0
SIZE = 40


def _frame(cx: int, cy: int) -> np.ndarray:
    """A grey field with one textured square (robot) at (cx, cy)."""
    img = np.full((180, 320, 3), 60, np.uint8)
    h = SIZE // 2
    cv2.rectangle(img, (cx - h, cy - h), (cx + h, cy + h), (210, 210, 210), -1)
    # asymmetric dark patch inside gives the appearance model real texture
    cv2.rectangle(img, (cx - h + 4, cy - h + 4), (cx - h + 16, cy - h + 16),
                  (25, 25, 25), -1)
    return img


def _box(cx: int, cy: int):
    h = SIZE // 2
    return (float(cx - h), float(cy - h), float(cx + h), float(cy + h))


def _run(latch_backend):
    """Track a robot through a 7-frame detection gap (max_age=3). Returns the
    tid seen on the last frame with a detection before the gap, and the tid
    seen on the first frame with a detection after the gap."""
    cfg = {"max_age_frames": 3, "iou_match": 0.2, "min_hits": 1}
    trk = RobotTracker(cfg, fps=FPS, latch=latch_backend)

    before_gap_tid = None
    after_gap_tid = None
    f = 0
    # phase 1: robot visible + detected, moving right
    for i in range(10):
        cx, cy = 60 + 3 * i, 90
        frame = _frame(cx, cy)
        out = trk.update(f, [Detection("robot", _box(cx, cy), 0.9, f)], frame=frame)
        assert len(out) == 1
        before_gap_tid = out[0].tid
        f += 1

    # phase 2: robot keeps moving (still visible in the frame) but the detector
    # drops it for 7 straight frames — longer than max_age=3.
    cx, cy = 60 + 3 * 9, 90
    for i in range(1, 8):
        cx += 3
        frame = _frame(cx, cy)
        trk.update(f, [], frame=frame)
        f += 1

    # phase 3: detector picks the robot back up where it actually is.
    frame = _frame(cx, cy)
    out = trk.update(f, [Detection("robot", _box(cx, cy), 0.9, f)], frame=frame)
    assert len(out) == 1
    after_gap_tid = out[0].tid
    return before_gap_tid, after_gap_tid


def test_without_latch_gap_causes_id_switch():
    """Baseline: a gap longer than max_age culls the track -> new id on return."""
    before, after = _run(latch_backend=None)
    assert after != before, "expected the plain ByteTrack baseline to ID-switch"


def test_latch_backend_survives_the_same_gap():
    """With the visual backend, the still-visible robot is rescued every frame
    of the gap, so the id never dies."""
    if not latch_available():
        pytest.skip("latch-tracker not importable in this environment")
    before, after = _run(latch_backend=LatchBackend())
    assert after == before, "latch backend should have kept the id alive through the gap"


def test_latch_backend_disabled_by_default():
    """Constructing RobotTracker with no latch arg and no cfg flag never touches
    latch-tracker at all (no import side effects, identical to plain ByteTrack)."""
    trk = RobotTracker({"max_age_frames": 3}, fps=FPS)
    assert trk.latch is None
