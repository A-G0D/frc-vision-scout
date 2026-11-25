"""Budget + scheduler tests. The fps_budget test is the one that can ONLY pass on the
target laptop (Arc/NPU) — it reads the bench report rather than asserting in cloud CI.
"""
from __future__ import annotations

import json
import os

import pytest

from frcscout.pipeline.scheduler import AsyncScheduler, RateMeter


def test_scheduler_never_blocks_on_slow_detector():
    """A detector that sleeps far longer than the frame period must not stall tick();
    the engine reads stale-but-present snapshots instead."""
    import time
    sched = AsyncScheduler(target_fps=60, max_workers=2)
    sched.register("slow", lambda f, i: (time.sleep(0.05), "R")[1], cadence_hz=5)
    t0 = time.perf_counter()
    for i in range(120):              # 2s of frames at 60fps
        sched.tick(frame=None, frame_idx=i)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"tick() loop blocked on the slow detector ({elapsed:.2f}s)"
    sched.shutdown()


def test_period_from_cadence():
    sched = AsyncScheduler(target_fps=60)
    sched.register("a", lambda f, i: None, cadence_hz=5)
    assert sched._slots["a"].period_frames == 12


@pytest.mark.skipif(not os.path.exists("data/bench/laptop.json"),
                    reason="needs `make bench` on the target Intel laptop")
def test_fps_p99_budget():
    rep = json.load(open("data/bench/laptop.json"))
    assert rep["p99_frame_ms"] <= rep["budget_ms"], rep
