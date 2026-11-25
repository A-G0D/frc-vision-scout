"""Tests for multi-view pane segmentation (composite → synchronized camera views)."""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from frcscout.geometry.panes import find_pane_ranges, segment_panes, remap_to_composite


def _stacked_frame(h=400, w=300, div0=190, div1=210):
    """Two bright panes (top 0..div0, bottom div1..h) split by a dark divider band."""
    f = np.full((h, w, 3), 180, dtype=np.uint8)          # bright, textured-ish content
    f[:, ::7] = 90                                       # add some column variance
    f[div0:div1, :] = 5                                  # dark uniform divider bar
    return f


def test_detects_two_panes():
    f = _stacked_frame()
    ranges = find_pane_ranges(f)
    assert len(ranges) == 2
    top, bot = ranges
    assert top[0] == 0 and top[1] <= 190
    assert bot[0] >= 210


def test_segment_offsets_and_crops():
    f = _stacked_frame()
    panes = segment_panes(f)
    assert len(panes) == 2
    assert panes[0]["offset"] == (0, 0)
    assert panes[1]["offset"][1] >= 210
    # crop height matches the range
    p = panes[1]
    assert p["crop"].shape[0] == p["y1"] - p["y0"]


def test_fraction_mode_and_remap():
    f = _stacked_frame()
    panes = segment_panes(f, fracs=[(0.0, 0.45), (0.55, 1.0)])
    assert len(panes) == 2
    box = remap_to_composite((10, 10, 20, 20), panes[1]["offset"])
    assert box[1] == 10 + panes[1]["offset"][1]


def test_single_pane_fallback():
    f = np.full((100, 100, 3), 150, dtype=np.uint8)
    f[:, ::5] = 60
    ranges = find_pane_ranges(f)
    assert len(ranges) >= 1
