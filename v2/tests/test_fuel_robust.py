"""Regression tests for OCR-noise robustness in overlay fuel/score reading.

Encodes the qual59 failure pattern the full-match-vs-TBA comparison exposed:
a single high OCR misread must NOT suppress later increments, and upward jitter must
NOT inflate the cumulative total. See metrics/scout.py::_robust_monotonic /
_fuel_from_overlay and autobuild_log.md cycle 5c.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from frcscout.metrics.scout import _robust_monotonic, MetricsAggregator  # noqa: E402


def test_robust_monotonic_rejects_high_spike():
    # a 200/400 spike must be skipped; series climbs cleanly to 92
    seq = [26, 26, 200, 51, 51, 9, 63, 63, 400, 75, 92, 92, 5]
    out = _robust_monotonic(seq)
    assert out[-1] == 92, out
    assert all(b >= a for a, b in zip(out, out[1:]))  # non-decreasing


def test_robust_monotonic_absorbs_downward_spike_and_nones():
    seq = [96, 162, 162, 221, 315, None, 105, 507, None]
    out = _robust_monotonic(seq)
    assert out[-1] == 507, out


def test_robust_monotonic_empty_and_allnone():
    assert _robust_monotonic([]) == []
    assert _robust_monotonic([None, None]) == []


def _ts(pairs, key="blue_fuel"):
    """pairs: list of (frame_idx, value, phase) -> overlay timeseries rows."""
    return [{"frame_idx": f, "t_s": None, "phase": ph, key: v} for f, v, ph in pairs]


def test_fuel_from_overlay_high_spike_not_suppressed():
    """Red-style failure: an early high misread must not zero out later counting."""
    agg = MetricsAggregator()
    # auto reaches 30, then a 999 misread, then teleop climbs 35..70
    ts = _ts([(0, 10, "auto"), (1, 30, "auto"), (2, 999, "teleop"),
              (3, 40, "teleop"), (4, 55, "teleop"), (5, 70, "teleop")], key="blue_fuel")
    out = agg._fuel_from_overlay("blue", ts)
    assert out is not None
    # total should reflect the real climb to ~70, NOT collapse to a tiny number
    assert out["total"] >= 65, out
    assert out["teleop"] >= 35, out


def test_fuel_from_overlay_jitter_not_inflated():
    """Blue-style failure: upward jitter must not inflate the cumulative total."""
    agg = MetricsAggregator()
    # true cumulative 0..92 with +/- jitter; final real value 92
    ts = _ts([(0, 0, "auto"), (1, 26, "auto"), (2, 20, "teleop"), (3, 51, "teleop"),
              (4, 48, "teleop"), (5, 63, "teleop"), (6, 75, "teleop"), (7, 92, "teleop")],
             key="blue_fuel")
    out = agg._fuel_from_overlay("blue", ts)
    assert out is not None
    # must be ~92, not inflated toward 140 by the dips/re-climbs
    assert 90 <= out["total"] <= 100, out
