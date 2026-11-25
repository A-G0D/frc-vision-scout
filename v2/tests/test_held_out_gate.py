"""Held-out-match generalization gate — tests/test_held_out_gate.py.

Before this test, ``final1.mp4`` (the match ``configs/sources.yaml`` marks as
``split: test`` — "HELD OUT — never trained on") had only ever been run through the
A6.3 geometry audit (``scripts/audit_a63.py``); the actual detect->track->6-robot-prior
pipeline (``ScoutPipeline``, the SAME code path validated on qual56/qual59) had never
been run against it. "v2 generalizes to unseen footage" was therefore an unverified claim
(see README.md "Known gaps": "No validation against a live or unseen contest stream").

This is a real-footage end-to-end test (loads the OpenVINO ball/robot model, decodes real
video), so it is marked ``slow`` like ``test_run_live.py`` and skips cleanly when the
footage is absent. Frame counts are kept small (90 frames = 1.5s at 60fps) so the test
runs quickly; ``scripts/eval_held_out.py`` run standalone with a larger window is the
full report (see its module docstring / README for the measured numbers).
"""
from __future__ import annotations

import os
import sys

import pytest

_TESTS = os.path.dirname(os.path.abspath(__file__))
_V2 = os.path.dirname(_TESTS)
_SRC = os.path.join(_V2, "src")
_SCRIPTS = os.path.join(_V2, "scripts")
for _p in (_SRC, _V2, _SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BASELINE_VIDEO = os.path.join(_V2, "data", "footage", "qual56.mp4")   # split: val (trained-adjacent)
HELD_OUT_VIDEO = os.path.join(_V2, "data", "footage", "final1.mp4")   # split: test (never trained)
FOOTAGE_PRESENT = os.path.exists(BASELINE_VIDEO) and os.path.exists(HELD_OUT_VIDEO)

pytestmark = pytest.mark.slow

# Small window for test speed; a mid-match slice (robots + balls both active per the
# qual56 demo convention documented in v2/RESULTS.md).
START = 2000
FRAMES = 90


@pytest.mark.skipif(not FOOTAGE_PRESENT, reason="qual56.mp4 / final1.mp4 footage absent")
def test_pipeline_runs_clean_on_held_out_match():
    """The real sync pipeline must run to completion on final1.mp4 (never trained on,
    never previously run through this path) without raising, and must actually detect
    something — a crash or an all-zero run would mean v2 silently does not generalize."""
    from eval_held_out import measure

    held_out = measure(HELD_OUT_VIDEO, start=START, frames=FRAMES)

    assert held_out["n_frames"] == FRAMES, (
        f"expected {FRAMES} frames processed, got {held_out['n_frames']}")
    assert held_out["six_any_frac"] > 0, (
        "zero robot-tracking on the held-out match — the pipeline does not generalize")
    assert held_out["ball_nonzero_frac"] > 0, (
        "zero ball detections on the held-out match — the pipeline does not generalize")
    print(f"\n[held-out] final1.mp4 f{START}..{START + FRAMES}: {held_out}")


@pytest.mark.skipif(not FOOTAGE_PRESENT, reason="qual56.mp4 / final1.mp4 footage absent")
def test_held_out_does_not_collapse_vs_validated_baseline():
    """The held-out match's core detect/track health (six_any_frac, ball_nonzero_frac)
    must not collapse relative to the VALIDATED (train/val-split) qual56 match over the
    identical window — this is the actual regression/generalization gate. It may be
    materially worse (unseen footage, different camera framing/lighting) but not dead."""
    from eval_held_out import compare, measure

    baseline = measure(BASELINE_VIDEO, start=START, frames=FRAMES)
    held_out = measure(HELD_OUT_VIDEO, start=START, frames=FRAMES)
    report = compare(baseline, held_out, tolerance=0.5)

    print(f"\n[gate] baseline(qual56)={baseline}")
    print(f"[gate] held_out(final1)={held_out}")
    print(f"[gate] gated_metrics={report['gated_metrics']}")

    for metric, r in report["gated_metrics"].items():
        assert r["pass"], (
            f"{metric} collapsed on the held-out match: baseline={r['baseline']} "
            f"held_out={r['held_out']} (needed >= {report['tolerance']} x baseline)")
    assert report["pass"]
