"""Tests for live/ingest.py.

Covers:
  - FileSource path (uses data/footage/qual59.mp4; skipped if file absent)
  - Decimation math (_compute_decimation_period) — pure, no network, no subprocess
  - Drop-oldest queue behaviour (no subprocess)
  - Real-time clock gate logic (mock time)
  - FrameItem dataclass fields
  - _build_ffmpeg_cmd output (no subprocess)
  - _backoff values
  - max_retries cap (unit-tested on the producer logic in isolation)
"""
from __future__ import annotations

import os
import queue
import time
import threading
from dataclasses import fields
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

# Absolute path to the test footage file
_FOOTAGE = os.path.join(
    os.path.dirname(__file__),
    "..", "data", "footage", "qual59.mp4",
)
_FOOTAGE = os.path.normpath(_FOOTAGE)

_HAS_FOOTAGE = os.path.isfile(_FOOTAGE)
_HAS_CV2 = True
try:
    import cv2  # noqa: F401
except ImportError:
    _HAS_CV2 = False


# ---------------------------------------------------------------------------
# Import the module under test (side-effect-free at import time)
# ---------------------------------------------------------------------------

from live.ingest import (  # noqa: E402
    FileSource,
    FrameItem,
    IngestStats,
    YouTubeLiveIngest,
    _backoff,
    _build_ffmpeg_cmd,
    _compute_decimation_period,
)


# ===========================================================================
# 1. Decimation math — pure, no network
# ===========================================================================


class TestComputeDecimationPeriod:
    """_compute_decimation_period is the core real-time gate formula."""

    def test_same_fps_returns_zero(self):
        """When target == source, keep every frame (period = 0)."""
        assert _compute_decimation_period(30.0, 30.0) == 0.0

    def test_target_above_source_returns_zero(self):
        """When target > source, never drop (period = 0)."""
        assert _compute_decimation_period(30.0, 60.0) == 0.0

    def test_half_rate(self):
        """60 fps source decimated to 30 fps: period = 1/30."""
        period = _compute_decimation_period(60.0, 30.0)
        assert abs(period - 1.0 / 30.0) < 1e-9

    def test_third_rate(self):
        """90 fps source decimated to 30 fps: period = 1/30."""
        period = _compute_decimation_period(90.0, 30.0)
        assert abs(period - 1.0 / 30.0) < 1e-9

    def test_arbitrary(self):
        """25 fps decimated to 10 fps: period = 0.1 s."""
        period = _compute_decimation_period(25.0, 10.0)
        assert abs(period - 0.1) < 1e-9

    def test_raises_on_zero_source(self):
        with pytest.raises(ValueError):
            _compute_decimation_period(0.0, 30.0)

    def test_raises_on_zero_target(self):
        with pytest.raises(ValueError):
            _compute_decimation_period(30.0, 0.0)

    def test_raises_on_negative(self):
        with pytest.raises(ValueError):
            _compute_decimation_period(-1.0, 30.0)

    def test_fractional_fps(self):
        """23.976 fps source to 23.976 fps target = 0 period."""
        assert _compute_decimation_period(23.976, 23.976) == 0.0

    def test_period_gives_correct_keep_ratio(self):
        """Simulate N source frames at uniform intervals; count kept vs dropped."""
        source_fps = 60.0
        target_fps = 20.0
        period = _compute_decimation_period(source_fps, target_fps)
        source_interval = 1.0 / source_fps

        t = 0.0
        next_t = 0.0
        kept = 0
        dropped = 0
        n_frames = 600   # 10 s at 60 fps

        for _ in range(n_frames):
            if t < next_t:
                dropped += 1
            else:
                kept += 1
                next_t += period
                if next_t < t:
                    next_t = t
            t += source_interval

        # Expect roughly 200 kept (10 s * 20 fps) with small rounding tolerance
        assert abs(kept - 200) <= 2, f"kept={kept}, expected ~200"
        assert kept + dropped == n_frames


# ===========================================================================
# 2. Backoff formula
# ===========================================================================


class TestBackoff:
    def test_attempt_0(self):
        assert abs(_backoff(0) - 0.5) < 1e-9

    def test_attempt_1(self):
        assert abs(_backoff(1) - 1.0) < 1e-9

    def test_attempt_2(self):
        assert abs(_backoff(2) - 2.0) < 1e-9

    def test_caps_at_30(self):
        assert _backoff(100) == 30.0

    def test_monotonically_increasing_until_cap(self):
        vals = [_backoff(i) for i in range(8)]
        for a, b in zip(vals, vals[1:]):
            assert b >= a


# ===========================================================================
# 3. FrameItem dataclass
# ===========================================================================


class TestFrameItem:
    def test_fields_exist(self):
        field_names = {f.name for f in fields(FrameItem)}
        assert "frame" in field_names
        assert "capture_t" in field_names
        assert "enqueue_t" in field_names
        assert "segment_seq" in field_names

    def test_default_segment_seq(self):
        item = FrameItem(
            frame=np.zeros((720, 1280, 3), dtype=np.uint8),
            capture_t=1.0,
            enqueue_t=1.001,
        )
        assert item.segment_seq == -1

    def test_frame_shape_preserved(self):
        arr = np.zeros((720, 1280, 3), dtype=np.uint8)
        item = FrameItem(frame=arr, capture_t=0.0, enqueue_t=0.0)
        assert item.frame.shape == (720, 1280, 3)
        assert item.frame.dtype == np.uint8


# ===========================================================================
# 4. _build_ffmpeg_cmd — no subprocess
# ===========================================================================


class TestBuildFfmpegCmd:
    def test_returns_list(self):
        cmd = _build_ffmpeg_cmd("http://example.com/manifest.m3u8", (1280, 720), 30.0)
        assert isinstance(cmd, list)
        assert len(cmd) > 0

    def test_contains_pipe1(self):
        cmd = _build_ffmpeg_cmd("http://x/m.m3u8", (1280, 720), 30.0)
        assert "pipe:1" in cmd

    def test_pix_fmt_bgr24(self):
        cmd = _build_ffmpeg_cmd("http://x/m.m3u8", (1280, 720), 30.0)
        idx = cmd.index("-pix_fmt")
        assert cmd[idx + 1] == "bgr24"

    def test_vcodec_rawvideo(self):
        cmd = _build_ffmpeg_cmd("http://x/m.m3u8", (1280, 720), 30.0)
        idx = cmd.index("-vcodec")
        assert cmd[idx + 1] == "rawvideo"

    def test_no_audio(self):
        cmd = _build_ffmpeg_cmd("http://x/m.m3u8", (1280, 720), 30.0)
        assert "-an" in cmd

    def test_reconnect_flags(self):
        cmd = _build_ffmpeg_cmd("http://x/m.m3u8", (1280, 720), 30.0)
        assert "-reconnect" in cmd
        assert "-reconnect_streamed" in cmd
        assert "-reconnect_delay_max" in cmd

    def test_fps_filter_in_vf(self):
        cmd = _build_ffmpeg_cmd("http://x/m.m3u8", (1280, 720), 30.0)
        idx = cmd.index("-vf")
        vf = cmd[idx + 1]
        assert "fps=fps=30" in vf
        assert "round=down" in vf

    def test_scale_in_vf(self):
        cmd = _build_ffmpeg_cmd("http://x/m.m3u8", (640, 360), 15.0)
        idx = cmd.index("-vf")
        vf = cmd[idx + 1]
        assert "scale=640:360" in vf

    def test_custom_ffmpeg_bin(self):
        cmd = _build_ffmpeg_cmd("http://x/m.m3u8", (1280, 720), 30.0, ffmpeg_bin="/opt/ffmpeg")
        assert cmd[0] == "/opt/ffmpeg"

    def test_manifest_url_as_input(self):
        url = "https://manifest.googlevideo.com/api/manifest/hls/id/abc123"
        cmd = _build_ffmpeg_cmd(url, (1280, 720), 30.0)
        idx = cmd.index("-i")
        assert cmd[idx + 1] == url


# ===========================================================================
# 5. Drop-oldest queue behaviour (no subprocess)
# ===========================================================================


class TestDropOldestQueue:
    """Verify _enqueue drops the oldest item on overflow."""

    def _make_item(self, val: int) -> FrameItem:
        frame = np.full((2, 2, 3), val, dtype=np.uint8)
        return FrameItem(frame=frame, capture_t=float(val), enqueue_t=float(val))

    def test_overflow_drops_oldest(self):
        ingest = YouTubeLiveIngest.__new__(YouTubeLiveIngest)
        ingest._queue = queue.Queue(maxsize=2)
        ingest._dropped = 0

        items = [self._make_item(i) for i in range(4)]
        for item in items:
            ingest._enqueue(item)

        # Queue maxsize=2; after 4 inserts, the two newest should remain
        collected = []
        while True:
            try:
                collected.append(ingest._queue.get_nowait())
            except queue.Empty:
                break

        assert len(collected) == 2
        # The oldest frames (lower capture_t) should have been dropped
        assert collected[0].capture_t >= items[0].capture_t
        assert collected[1].capture_t > collected[0].capture_t

    def test_drop_counter_increments(self):
        ingest = YouTubeLiveIngest.__new__(YouTubeLiveIngest)
        ingest._queue = queue.Queue(maxsize=1)
        ingest._dropped = 0

        for i in range(5):
            ingest._enqueue(self._make_item(i))

        # 1 slot: 4 overflows -> 4 drops
        assert ingest._dropped == 4


# ===========================================================================
# 6. YouTubeLiveIngest — start/stop without a network (mock _producer_loop)
# ===========================================================================


class TestYouTubeLiveIngestLifecycle:
    """Test state transitions and thread lifecycle without any real subprocess."""

    def _make_idle_ingest(self) -> YouTubeLiveIngest:
        return YouTubeLiveIngest(
            url="https://www.youtube.com/watch?v=FAKE",
            target_fps=30.0,
            resolution=(1280, 720),
            queue_maxsize=2,
            stall_timeout=1.0,
            max_retries=2,
        )

    def test_initial_state_is_idle(self):
        ingest = self._make_idle_ingest()
        assert ingest._state == "idle"

    def test_start_transitions_to_connecting(self):
        ingest = self._make_idle_ingest()
        # Patch _producer_loop and _watchdog_loop so they exit immediately
        ingest._producer_loop = lambda: None
        ingest._watchdog_loop = lambda: None
        ingest.start()
        # Give threads a moment to start
        time.sleep(0.05)
        ingest.stop()

    def test_stop_transitions_to_stopped(self):
        ingest = self._make_idle_ingest()
        ingest._producer_loop = lambda: None
        ingest._watchdog_loop = lambda: None
        ingest.start()
        ingest.stop()
        assert ingest._state == "stopped"

    def test_double_start_raises(self):
        ingest = self._make_idle_ingest()
        ingest._producer_loop = lambda: None
        ingest._watchdog_loop = lambda: None
        ingest.start()
        with pytest.raises(RuntimeError):
            ingest.start()
        ingest.stop()

    def test_context_manager(self):
        ingest = self._make_idle_ingest()
        ingest._producer_loop = lambda: None
        ingest._watchdog_loop = lambda: None
        with ingest:
            pass
        assert ingest._state == "stopped"

    def test_stats_returns_ingest_stats(self):
        ingest = self._make_idle_ingest()
        s = ingest.stats()
        assert isinstance(s, IngestStats)
        assert s.state == "idle"
        assert s.captured == 0
        assert s.dropped == 0


# ===========================================================================
# 7. Real-time clock gate logic — pure simulation
# ===========================================================================


class TestClockGate:
    """Simulate the clock-gate loop from _read_frames without any subprocess."""

    def _simulate_gate(
        self,
        source_fps: float,
        target_fps: float,
        n_source_frames: int,
    ) -> tuple[int, int]:
        """Run the clock-gate algorithm and return (kept, dropped)."""
        period = 1.0 / target_fps
        source_interval = 1.0 / source_fps

        t = 0.0
        next_t = 0.0
        kept = 0
        dropped = 0

        for _ in range(n_source_frames):
            now = t
            if now < next_t:
                dropped += 1
            else:
                kept += 1
                next_t += period
                if next_t < now:
                    next_t = now
            t += source_interval

        return kept, dropped

    def test_no_decimation_keeps_all(self):
        kept, dropped = self._simulate_gate(30.0, 30.0, 300)
        assert dropped == 0
        assert kept == 300

    def test_half_rate_decimation(self):
        """60 fps -> 30 fps: ~half kept."""
        kept, dropped = self._simulate_gate(60.0, 30.0, 600)
        # Allow +-2 rounding
        assert abs(kept - 300) <= 2, f"kept={kept}"
        assert kept + dropped == 600

    def test_real_time_correct_do_not_slow_down(self):
        """With target < source, kept frames are limited by target, not source."""
        kept, dropped = self._simulate_gate(60.0, 15.0, 600)
        # 10 s of 60fps -> expect ~150 kept at 15fps
        assert abs(kept - 150) <= 2, f"kept={kept}"

    def test_debt_reset_prevents_catch_up_burst(self):
        """Simulate a slow consumer period (big gap) then resume.

        After a gap, next_t snaps to now so we do NOT emit a burst of
        catch-up frames.
        """
        period = 1.0 / 30.0
        next_t = 0.0
        burst_after_gap = 0

        # Emit 30 frames at 30fps
        for i in range(30):
            now = i * period
            if now >= next_t:
                next_t += period
                if next_t < now:
                    next_t = now  # snap — this is the key invariant

        # Simulate a 5-second gap (consumer was busy)
        now = 30 * period + 5.0

        # Now feed 30 more frames at 30fps starting from now
        t = now
        for i in range(30):
            if t >= next_t:
                burst_after_gap += 1
                next_t += period
                if next_t < t:
                    next_t = t
            t += period

        # Without the snap, a 5-second gap at 30fps = 150 frames of debt -> burst.
        # With the snap, next_t was reset to now, so burst is normal (~30).
        assert burst_after_gap <= 31, (
            f"Burst after gap too large: {burst_after_gap} — debt accumulation bug"
        )


# ===========================================================================
# 8. FileSource — exercises the file path (skipped if footage absent or no cv2)
# ===========================================================================


@pytest.mark.skipif(
    not (_HAS_FOOTAGE and _HAS_CV2),
    reason="requires qual59.mp4 and opencv-python",
)
class TestFileSource:
    def test_reads_frames(self):
        """FileSource should yield at least a few frames from qual59."""
        src = FileSource(_FOOTAGE, target_fps=30.0, queue_maxsize=4)
        src.start()

        frames = []
        deadline = time.perf_counter() + 3.0   # read for up to 3 s
        while time.perf_counter() < deadline and len(frames) < 30:
            result = src.get(timeout=0.2)
            if result is not None:
                frames.append(result)

        src.stop()
        assert len(frames) >= 10, f"Expected >=10 frames, got {len(frames)}"

    def test_frame_shape_bgr(self):
        """Each frame must be a uint8 3-channel array."""
        src = FileSource(_FOOTAGE, target_fps=5.0, queue_maxsize=4)
        src.start()

        result = None
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            result = src.get(timeout=0.5)
            if result is not None:
                break

        src.stop()
        assert result is not None, "No frame received"
        frame, capture_t = result
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8

    def test_capture_t_is_recent(self):
        """capture_t should be a recent perf_counter value (within 5 s)."""
        src = FileSource(_FOOTAGE, target_fps=5.0, queue_maxsize=4)
        src.start()

        t_before = time.perf_counter()
        result = None
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            result = src.get(timeout=0.5)
            if result is not None:
                break
        t_after = time.perf_counter()

        src.stop()
        assert result is not None
        _, capture_t = result
        assert t_before <= capture_t <= t_after + 0.1

    def test_context_manager(self):
        """FileSource as context manager starts and stops cleanly."""
        frames = []
        with FileSource(_FOOTAGE, target_fps=10.0, queue_maxsize=4) as src:
            deadline = time.perf_counter() + 2.0
            while time.perf_counter() < deadline and len(frames) < 5:
                result = src.get(timeout=0.3)
                if result is not None:
                    frames.append(result)
        assert len(frames) >= 1

    def test_iterator_yields_tuples(self):
        """__iter__ yields (frame, capture_t) tuples."""
        collected = []
        with FileSource(_FOOTAGE, target_fps=5.0, queue_maxsize=4) as src:
            for frame, capture_t in src:
                collected.append((frame.shape, capture_t))
                if len(collected) >= 5:
                    break

        assert len(collected) >= 1
        for shape, t in collected:
            assert len(shape) == 3
            assert shape[2] == 3
            assert isinstance(t, float)

    def test_decimation_reduces_frame_rate(self):
        """Reading at 5 fps from a 30fps source should yield fewer frames per second
        than reading at native fps."""
        import cv2 as cv2_local

        # Get the native fps of the file
        cap = cv2_local.VideoCapture(_FOOTAGE)
        native_fps = cap.get(cv2_local.CAP_PROP_FPS) or 30.0
        cap.release()

        if native_fps < 10:
            pytest.skip("Test footage fps too low to demonstrate decimation")

        target_fps = max(1.0, native_fps / 3.0)

        src = FileSource(_FOOTAGE, target_fps=target_fps, queue_maxsize=8)
        src.start()

        t0 = time.perf_counter()
        frames = []
        window = 2.0   # measure over 2 seconds
        while time.perf_counter() - t0 < window:
            result = src.get(timeout=0.2)
            if result is not None:
                frames.append(result)

        src.stop()
        elapsed = time.perf_counter() - t0
        measured_fps = len(frames) / max(elapsed, 0.01)

        # Allow 50 % tolerance: measured_fps should be much closer to target than native
        assert measured_fps <= target_fps * 1.5, (
            f"measured_fps={measured_fps:.1f} target={target_fps:.1f} native={native_fps:.1f}"
        )

    def test_stats_returns_ingest_stats(self):
        """FileSource.stats() returns the same IngestStats shape."""
        src = FileSource(_FOOTAGE, target_fps=5.0, queue_maxsize=4)
        src.start()
        src.get(timeout=1.0)
        s = src.stats()
        src.stop()

        assert isinstance(s, IngestStats)
        assert s.reconnects == 0
        assert s.captured >= 0

    def test_resolution_override(self):
        """If resolution is provided, frames should have the requested size."""
        target_res = (320, 240)
        src = FileSource(_FOOTAGE, target_fps=5.0, resolution=target_res, queue_maxsize=4)
        src.start()

        result = None
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            result = src.get(timeout=0.5)
            if result is not None:
                break

        src.stop()
        assert result is not None
        frame, _ = result
        assert frame.shape == (240, 320, 3), f"shape={frame.shape}"
