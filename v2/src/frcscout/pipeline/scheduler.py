"""Async cadence scheduler — the mechanism that keeps the 60Hz frame clock from ever
blocking on a slow detector.

Each detector is registered with a cadence (Hz) and a compute device. A detector
runs in a worker thread; the engine NEVER awaits it on the hot path. Instead the
engine reads the most-recent COMPLETED result (a snapshot). A detector that takes
40ms but runs at 5Hz simply publishes a fresh result every ~12 frames; tracking
interpolates in between.

This is why p99 frame time can stay <16.6ms even though the ball detector is slow:
the slow work is off the critical path, overlapped on the iGPU/NPU.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class _Slot:
    """A registered detector and its latest published result."""
    name: str
    fn: Callable[[object, int], object]   # (frame, frame_idx) -> result
    period_frames: int                    # derived from cadence_hz and target_fps
    last_launch_frame: int = -10_000
    inflight: Optional[Future] = None
    latest: object = None                 # most recent completed result (snapshot)
    latest_frame: int = -1
    lock: threading.Lock = field(default_factory=threading.Lock)


class AsyncScheduler:
    def __init__(self, target_fps: float, max_workers: int = 4):
        self.target_fps = target_fps
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="det")
        self._slots: dict[str, _Slot] = {}

    def register(self, name: str, fn: Callable, cadence_hz: float) -> None:
        period = max(1, round(self.target_fps / max(cadence_hz, 0.01)))
        self._slots[name] = _Slot(name=name, fn=fn, period_frames=period)

    def tick(self, frame, frame_idx: int) -> None:
        """Called once per frame. Launches any detector whose period has elapsed and
        whose previous run has finished. Non-blocking."""
        for slot in self._slots.values():
            due = (frame_idx - slot.last_launch_frame) >= slot.period_frames
            free = slot.inflight is None or slot.inflight.done()
            if due and free:
                if slot.inflight is not None:           # harvest completed result
                    self._harvest(slot)
                slot.last_launch_frame = frame_idx
                slot.inflight = self._pool.submit(self._run, slot, frame, frame_idx)

    def _run(self, slot: _Slot, frame, frame_idx: int):
        return slot.fn(frame, frame_idx), frame_idx

    def _harvest(self, slot: _Slot) -> None:
        if slot.inflight and slot.inflight.done():
            try:
                result, fidx = slot.inflight.result()
                with slot.lock:
                    slot.latest, slot.latest_frame = result, fidx
            except Exception:
                pass  # a dropped detection frame must never crash the clock
            slot.inflight = None

    def latest(self, name: str):
        """Most-recent completed result + the frame it was computed on (for staleness)."""
        slot = self._slots[name]
        if slot.inflight and slot.inflight.done():
            self._harvest(slot)
        with slot.lock:
            return slot.latest, slot.latest_frame

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


class RateMeter:
    """EMA fps + p99 frame-time tracker for the budget gate."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.ema_fps: Optional[float] = None
        self._frame_times_ms: list[float] = []
        self._t_last = time.perf_counter()

    def mark(self) -> float:
        now = time.perf_counter()
        dt = now - self._t_last
        self._t_last = now
        self._frame_times_ms.append(dt * 1000.0)
        inst = 1.0 / dt if dt > 0 else 0.0
        self.ema_fps = inst if self.ema_fps is None else (
            (1 - self.alpha) * self.ema_fps + self.alpha * inst)
        return self.ema_fps

    def p99_ms(self) -> float:
        if not self._frame_times_ms:
            return 0.0
        s = sorted(self._frame_times_ms)
        return s[min(len(s) - 1, int(0.99 * len(s)))]
