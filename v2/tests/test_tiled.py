"""tests/test_tiled.py — pure-logic tests for refine/tiled.py (SAHI-style tiling).

NO model, NO GPU. numpy/cv2 are NOT required for the core tests; the single
pairwise_iou test is skipped if numpy is absent. Everything else runs in a bare
interpreter, which is the whole point of keeping the merge logic pure-Python.
"""
from __future__ import annotations

import pytest

from frcscout.io.types import Detection
from frcscout.refine import tiled
from frcscout.refine.tiled import (
    build_tiles,
    dedup,
    iou,
    merge_tiles,
    nms,
    remap,
)

try:
    import numpy as np  # noqa: F401
    HAVE_NUMPY = True
except Exception:
    HAVE_NUMPY = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _det(x1, y1, x2, y2, conf=0.9, cls="ball", fi=0) -> Detection:
    return Detection(cls=cls, xyxy=(float(x1), float(y1), float(x2), float(y2)),
                     conf=conf, frame_idx=fi)


def _covers(tiles, region, h, w):
    """True iff every integer pixel center in `region` lies inside some tile.

    region = (rx0, ry0, rx1, ry1), half-open. Checked on a coarse grid (every
    pixel) — regions in tests are small enough for this to be exact and cheap.
    """
    rx0, ry0, rx1, ry1 = region
    for py in range(ry0, ry1):
        for px in range(rx0, rx1):
            inside = any(x0 <= px < x1 and y0 <= py < y1 for (x0, y0, x1, y1) in tiles)
            if not inside:
                return False
    return True


# ===========================================================================
# A. build_tiles — coverage
# ===========================================================================

class TestCoverage:
    def test_full_frame_fully_covered_no_overlap(self):
        h, w = 64, 96
        tiles = build_tiles(h, w, 32, 0.0)
        assert _covers(tiles, (0, 0, w, h), h, w)

    def test_full_frame_fully_covered_with_overlap(self):
        h, w = 70, 100
        tiles = build_tiles(h, w, 32, 0.25)
        assert _covers(tiles, (0, 0, w, h), h, w)

    def test_non_divisible_tail_is_covered(self):
        """37 wide, 16 tile, 0 overlap: stepped tiles reach 32; the flush tile must
        cover the [32,37) tail. The v1-style clamp-and-break would have dropped it."""
        h, w = 16, 37
        tiles = build_tiles(h, w, 16, 0.0)
        assert _covers(tiles, (0, 0, w, h), h, w)
        # The right edge of the union must reach w.
        assert max(t[2] for t in tiles) == w

    def test_field_bbox_fully_covered(self):
        h, w = 200, 300
        bbox = (40, 30, 220, 160)
        tiles = build_tiles(h, w, 64, 0.2, field_bbox=bbox)
        assert _covers(tiles, bbox, h, w)

    def test_tiles_within_frame_bounds(self):
        h, w = 180, 320
        for (x0, y0, x1, y1) in build_tiles(h, w, 64, 0.2):
            assert 0 <= x0 < x1 <= w
            assert 0 <= y0 < y1 <= h

    def test_tiles_within_field_bbox(self):
        h, w = 180, 320
        bbox = (20, 15, 240, 150)
        for (x0, y0, x1, y1) in build_tiles(h, w, 64, 0.2, field_bbox=bbox):
            assert x0 >= bbox[0] and y0 >= bbox[1]
            assert x1 <= bbox[2] and y1 <= bbox[3]


# ===========================================================================
# B. build_tiles — overlap / count / edge cases
# ===========================================================================

class TestTileCount:
    def test_single_tile_when_region_smaller_than_tile(self):
        tiles = build_tiles(50, 80, 256, 0.2)
        assert tiles == [(0, 0, 80, 50)]

    def test_exact_grid_no_overlap(self):
        """128x128 frame, 64 tile, 0 overlap → exactly a 2x2 grid."""
        tiles = build_tiles(128, 128, 64, 0.0)
        assert len(tiles) == 4
        assert set(tiles) == {
            (0, 0, 64, 64), (64, 0, 128, 64),
            (0, 64, 64, 128), (64, 64, 128, 128),
        }

    def test_overlap_increases_count(self):
        no = build_tiles(256, 256, 64, 0.0)
        hi = build_tiles(256, 256, 64, 0.5)
        assert len(hi) > len(no)

    def test_field_bbox_reduces_count(self):
        full = build_tiles(1080, 1920, 256, 0.2)
        sub = build_tiles(1080, 1920, 256, 0.2, field_bbox=(300, 300, 900, 700))
        assert len(sub) < len(full)

    def test_no_duplicate_tiles(self):
        """Flush-edge alignment must not emit a tile identical to a stepped one."""
        for ov in (0.0, 0.1, 0.2, 0.5, 0.75):
            tiles = build_tiles(200, 350, 64, ov)
            assert len(tiles) == len(set(tiles)), f"dup tiles at overlap={ov}"

    def test_overlap_count_known_value(self):
        """1D reasoning lifted to 2D: span 256, tile 64, overlap 0.5 → step 32.
        Starts: 0,32,64,96,128,160,192 (=256-64) → 7 per axis → 49 tiles."""
        tiles = build_tiles(256, 256, 64, 0.5)
        assert len(tiles) == 49

    def test_degenerate_region_returns_empty(self):
        assert build_tiles(100, 100, 32, 0.2, field_bbox=(50, 50, 50, 90)) == []
        assert build_tiles(100, 100, 32, 0.2, field_bbox=(80, 10, 20, 90)) == []

    def test_bbox_clipped_to_frame(self):
        """A bbox extending past the frame is clipped, not over-tiled past edges."""
        tiles = build_tiles(100, 100, 64, 0.0, field_bbox=(50, 50, 500, 500))
        for (x0, y0, x1, y1) in tiles:
            assert x1 <= 100 and y1 <= 100
        assert _covers(tiles, (50, 50, 100, 100), 100, 100)

    def test_overlap_clamped_no_zero_step(self):
        """overlap >= 1 would zero the step; it must be clamped and still terminate."""
        tiles = build_tiles(128, 128, 64, 1.5)
        assert len(tiles) >= 4
        assert len(tiles) == len(set(tiles))

    def test_zero_overlap_tiles_do_not_share_pixels(self):
        """Half-open tiles at 0 overlap tile a clean partition (no shared columns)."""
        tiles = build_tiles(64, 128, 64, 0.0)
        # Row 0 only; check x-intervals are disjoint half-open.
        xs = sorted((x0, x1) for (x0, y0, x1, y1) in tiles)
        for (a0, a1), (b0, b1) in zip(xs, xs[1:]):
            assert a1 <= b0  # no overlap at zero-overlap setting


# ===========================================================================
# C. remap — correctness
# ===========================================================================

class TestRemap:
    def test_remap_tuple_boxes(self):
        boxes = [(10.0, 20.0, 30.0, 40.0)]
        out = remap(boxes, 100, 200)
        assert out == [(110.0, 220.0, 130.0, 240.0)]

    def test_remap_zero_offset_identity(self):
        boxes = [(1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0)]
        assert remap(boxes, 0, 0) == boxes

    def test_remap_detection_objects(self):
        d = _det(5, 5, 15, 15, conf=0.7)
        out = remap([d], 50, 60)
        assert len(out) == 1
        assert out[0].xyxy == (55.0, 65.0, 65.0, 75.0)
        # metadata preserved
        assert out[0].conf == 0.7
        assert out[0].cls == "ball"
        assert out[0].frame_idx == 0

    def test_remap_does_not_mutate_frozen_original(self):
        d = _det(5, 5, 15, 15)
        _ = remap([d], 50, 60)
        assert d.xyxy == (5.0, 5.0, 15.0, 15.0)  # original untouched

    def test_remap_preserves_width_height(self):
        d = _det(0, 0, 24, 12)
        out = remap([d], 7, 9)[0]
        x1, y1, x2, y2 = out.xyxy
        assert (x2 - x1, y2 - y1) == (24.0, 12.0)

    def test_remap_empty(self):
        assert remap([], 10, 10) == []

    def test_remap_then_build_tiles_roundtrip(self):
        """A detection found in a tile lands at the correct frame position."""
        tiles = build_tiles(256, 256, 64, 0.0)
        # Take a non-origin tile and place a box at tile-local (4,4,20,20).
        x0, y0, _, _ = tiles[3]
        local = [(4.0, 4.0, 20.0, 20.0)]
        framed = remap(local, x0, y0)[0]
        assert framed == (x0 + 4.0, y0 + 4.0, x0 + 20.0, y0 + 20.0)


# ===========================================================================
# D. iou / nms / merge
# ===========================================================================

class TestIoU:
    def test_identical(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_disjoint(self):
        assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_touching_edges_is_zero(self):
        assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0

    def test_half_overlap(self):
        # A=[0,0,10,10]=100, B=[5,0,15,10]=100, inter=50, union=150 → 1/3.
        assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)

    def test_zero_area_box(self):
        assert iou((0, 0, 0, 10), (0, 0, 10, 10)) == 0.0


class TestNMS:
    def test_empty(self):
        assert nms([]) == []

    def test_single(self):
        d = _det(0, 0, 10, 10)
        assert nms([d]) == [d]

    def test_identical_collapse_keeps_higher_conf(self):
        a = _det(0, 0, 10, 10, conf=0.9)
        b = _det(0, 0, 10, 10, conf=0.5)
        out = nms([b, a])  # deliberately low-conf first
        assert out == [a]

    def test_disjoint_all_kept(self):
        ds = [_det(0, 0, 10, 10), _det(100, 0, 110, 10), _det(0, 100, 10, 110)]
        assert len(nms(ds)) == 3

    def test_partial_overlap_below_thresh_both_kept(self):
        # IoU = 1/3 < 0.45 → both survive.
        ds = [_det(0, 0, 10, 10, conf=0.8), _det(5, 0, 15, 10, conf=0.7)]
        assert len(nms(ds, iou_thresh=0.45)) == 2

    def test_high_overlap_suppressed(self):
        ds = [_det(0, 0, 10, 10, conf=0.8), _det(0, 0, 10, 10, conf=0.7)]
        assert len(nms(ds, iou_thresh=0.45)) == 1

    def test_class_aware_keeps_overlapping_different_classes(self):
        ball = _det(0, 0, 10, 10, conf=0.9, cls="ball")
        robot = _det(0, 0, 10, 10, conf=0.8, cls="robot")
        out = nms([ball, robot], class_aware=True)
        assert len(out) == 2

    def test_class_agnostic_suppresses_across_classes(self):
        ball = _det(0, 0, 10, 10, conf=0.9, cls="ball")
        robot = _det(0, 0, 10, 10, conf=0.8, cls="robot")
        out = nms([ball, robot], class_aware=False)
        assert len(out) == 1
        assert out[0] is ball

    def test_deterministic_tie_break_on_equal_conf(self):
        a = _det(0, 0, 10, 10, conf=0.8)
        b = _det(1, 1, 11, 11, conf=0.8)  # IoU high vs a
        # Equal conf → insertion order decides; first one kept.
        assert nms([a, b]) == [a]
        assert nms([b, a]) == [b]

    def test_dedup_is_nms_alias(self):
        assert dedup is nms


class TestMergeTiles:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            merge_tiles([[_det(0, 0, 1, 1)]], [(0, 0), (5, 5)])

    def test_empty(self):
        assert merge_tiles([], []) == []

    def test_same_ball_in_two_overlapping_tiles_collapses(self):
        """The canonical seam case: one physical ball at frame (70,10,90,30) seen in
        two tiles whose origins are (0,0) and (50,0); after merge → exactly one box."""
        tile_a = [_det(70, 10, 90, 30, conf=0.9)]      # local == frame (origin 0,0)
        tile_b = [_det(20, 10, 40, 30, conf=0.85)]     # local; +50 → frame (70..90)
        out = merge_tiles([tile_a, tile_b], [(0, 0), (50, 0)])
        assert len(out) == 1
        assert out[0].conf == 0.9
        assert out[0].xyxy == (70.0, 10.0, 90.0, 30.0)

    def test_distinct_balls_across_tiles_all_survive(self):
        tile_a = [_det(5, 5, 15, 15)]
        tile_b = [_det(5, 5, 15, 15)]   # local; +200 → far away
        out = merge_tiles([tile_a, tile_b], [(0, 0), (200, 0)])
        assert len(out) == 2

    def test_merge_remaps_into_frame_coords(self):
        tile = [_det(1, 1, 3, 3)]
        out = merge_tiles([tile], [(10, 20)])
        assert out[0].xyxy == (11.0, 21.0, 13.0, 23.0)

    def test_merge_class_aware_default(self):
        # Same location, different classes, two tiles → both kept (class-aware).
        a = [_det(0, 0, 10, 10, cls="ball", conf=0.9)]
        b = [_det(0, 0, 10, 10, cls="robot", conf=0.9)]
        out = merge_tiles([a, b], [(0, 0), (0, 0)])
        assert len(out) == 2


# ===========================================================================
# E. optional numpy convenience
# ===========================================================================

@pytest.mark.skipif(not HAVE_NUMPY, reason="numpy not installed")
class TestPairwiseIoU:
    def test_matches_scalar_iou(self):
        boxes = [(0, 0, 10, 10), (5, 0, 15, 10), (100, 100, 110, 110)]
        m = tiled.pairwise_iou(boxes)
        assert m.shape == (3, 3)
        # diagonal == 1
        for i in range(3):
            assert m[i, i] == pytest.approx(1.0)
        # symmetric, and matches the scalar implementation
        assert m[0, 1] == pytest.approx(iou(boxes[0], boxes[1]))
        assert m[0, 1] == pytest.approx(m[1, 0])
        assert m[0, 2] == pytest.approx(0.0)

    def test_empty(self):
        m = tiled.pairwise_iou([])
        assert m.shape == (0, 0)
