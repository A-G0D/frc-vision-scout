"""Optional LatchTracker-backed visual rescue for RobotTracker's coasting tracks.

The documented weakness this plugs (see `reid.py`'s module docstring): ByteTrack's
Kalman filter only keeps an ID alive for `max_age` frames without a detection. On a
single broadcast camera the six FRC REBUILT robots routinely disappear for longer
than that — clustered at the HUB, behind one another, or the detector simply drops
a box on a frame where the robot is still plainly visible. `ReIDManager` covers the
LONG-gap case (mint-a-new-id -> rebind by appearance once the gap has already
happened). This module covers the frame-by-frame case: instead of coasting blind on
constant-velocity Kalman prediction while unmatched, ask a dedicated single-object
tracker to actually look at the raw frame the detector failed on.

latch-tracker (sibling project, `projects/latch-tracker`) is exactly that: a
click-to-latch tracker (TrackerVit appearance body + Kalman-IoU gating + optical-
flow-verified re-acquire), proven at 77fps on CPU. `LatchBackend` keeps one
`LatchTracker` instance per confirmed robot track, latched the first time the
detector confirms it, and consulted only on frames where the detector comes up
empty for that track.

Strictly optional and side-effect-free when unused:
  * `RobotTracker(... , latch=None)` (the default) never imports this module.
  * If latch-tracker isn't checked out / its cv2 build lacks `TrackerVit`, import
    fails quietly and `LatchBackend.enabled` is False — `RobotTracker` then runs
    exactly as plain ByteTrack, no exception, no behavior change.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

_LatchTracker = None
_import_error: Optional[BaseException] = None


def _try_import() -> None:
    """Best-effort, memoized import of `tracker_core.LatchTracker`.

    Looks for a `LATCH_TRACKER_PATH` env var first, then guesses the sibling
    checkout at `../../../../../../latch-tracker` relative to this file (i.e.
    `projects/latch-tracker` next to `projects/frc-vision-scout`).
    """
    global _LatchTracker, _import_error
    if _LatchTracker is not None or _import_error is not None:
        return
    here = os.path.dirname(os.path.abspath(__file__))
    # this file: v2/src/frcscout/track/latch_backend.py
    # up 6: track -> frcscout -> src -> v2 -> frc-vision-scout -> projects
    guess = os.path.normpath(os.path.join(
        here, "..", "..", "..", "..", "..", "latch-tracker"))
    candidates = [os.environ.get("LATCH_TRACKER_PATH", ""), guess]
    for path in candidates:
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
    try:
        from tracker_core import LatchTracker as _LT  # type: ignore
        _LatchTracker = _LT
    except Exception as exc:  # pragma: no cover - environment dependent
        _import_error = exc


def latch_available() -> bool:
    """True when latch-tracker was found and imported cleanly."""
    _try_import()
    return _LatchTracker is not None


class LatchBackend:
    """Per-track pool of `LatchTracker` instances, keyed by ByteTrack `tid`.

    Call `step()` once per live track per frame. It both maintains the visual
    tracker's own motion state (so it stays synced with real elapsed frames)
    and returns a rescue box exactly when the detector missed this track but
    the visual tracker still finds it.
    """

    def __init__(self) -> None:
        _try_import()
        self._by_tid: dict[int, object] = {}

    @property
    def enabled(self) -> bool:
        return _LatchTracker is not None

    def drop(self, tid: int) -> None:
        """Forget a track's visual tracker (call when the track is culled)."""
        self._by_tid.pop(tid, None)

    def step(self, tid: int, frame, xyxy, matched_this_frame: bool):
        """Advance (or create) the visual tracker for `tid` on this frame.

        Returns an `(x1, y1, x2, y2)` rescue box when the detector did NOT
        match this track this frame but the visual tracker still measured a
        confident location, else `None`.
        """
        if not self.enabled or frame is None:
            return None

        tracker = self._by_tid.get(tid)
        if tracker is None:
            if not matched_this_frame:
                return None  # nothing confirmed yet to latch onto
            x1, y1, x2, y2 = xyxy
            box = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
            if box[2] < 8 or box[3] < 8:
                return None  # too small for a stable template
            tracker = _LatchTracker()
            try:
                tracker.latch(frame, box)
            except Exception:
                return None
            self._by_tid[tid] = tracker
            return None

        try:
            result = tracker.update(frame)
        except Exception:
            self._by_tid.pop(tid, None)
            return None

        if matched_this_frame:
            return None  # detector already has it; keep the visual tracker warm only

        if not result.measured or result.state not in ("LOCKED", "COASTING"):
            return None
        x, y, w, h = result.box
        return (float(x), float(y), float(x + w), float(y + h))
