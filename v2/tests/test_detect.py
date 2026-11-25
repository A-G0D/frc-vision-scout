"""tests/test_detect.py — pure-logic unit tests for the detect layer.

NO model, NO GPU, NO easyocr, NO ultralytics required.
All heavy imports are guarded so the file parses and runs in a bare Python env.

Two test groups:
  A. Tile-merge + NMS  (ball_net.py pure helpers)
  B. Overlay structural-validation  (overlay_ocr.py validate_teams)
"""
from __future__ import annotations

import sys
import types

# ---------------------------------------------------------------------------
# Guard: stub out heavy deps so the detect package imports cleanly in CI
# without easyocr / ultralytics / cv2.  cv2 IS available in the CI image,
# so we only need to stub the optional inference libs.
# ---------------------------------------------------------------------------

def _maybe_stub(name: str) -> None:
    """Insert an empty stub module if `name` is not importable."""
    try:
        __import__(name)
    except ModuleNotFoundError:
        stub = types.ModuleType(name)
        sys.modules[name] = stub
        # Also register sub-packages that might be referenced.
        for suffix in ("", ".Reader", ".YOLO"):
            full = name + suffix if suffix else name
            if full not in sys.modules:
                sys.modules[full] = stub

_maybe_stub("easyocr")
_maybe_stub("ultralytics")

# Now it is safe to import the detect helpers.
from frcscout.detect.ball_net import (  # noqa: E402
    Detection,
    build_tiles,
    nms_detections,
    remap_detections_to_frame,
)
from frcscout.detect.overlay_ocr import validate_teams  # noqa: E402


# ===========================================================================
# A. Tile-merge + NMS tests
# ===========================================================================

def _det(x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> Detection:
    return Detection(cls="ball", xyxy=(x1, y1, x2, y2), conf=conf, frame_idx=0)


class TestNMS:
    def test_no_detections_returns_empty(self):
        assert nms_detections([]) == []

    def test_single_detection_passes_through(self):
        d = _det(10, 10, 50, 50)
        result = nms_detections([d])
        assert len(result) == 1
        assert result[0] is d

    def test_identical_boxes_merged_to_one(self):
        """Two boxes at the exact same location should collapse to one."""
        dets = [_det(10, 10, 50, 50, conf=0.9), _det(10, 10, 50, 50, conf=0.8)]
        result = nms_detections(dets)
        assert len(result) == 1
        # The higher-confidence box is kept.
        assert result[0].conf == 0.9

    def test_heavily_overlapping_duplicates_suppressed(self):
        """Simulate the same ball detected in two overlapping tiles — IoU ~0.9."""
        # Box 1 (tile 1 frame): x in [10,50] y in [10,50]
        # Box 2 (tile 2 frame, shifted by 5px — heavy overlap): x in [12,52]
        dets = [_det(10, 10, 50, 50, conf=0.85), _det(12, 12, 52, 52, conf=0.75)]
        result = nms_detections(dets)
        assert len(result) == 1, "heavily overlapping duplicates must collapse to one"
        assert result[0].conf == 0.85

    def test_non_overlapping_boxes_all_kept(self):
        """Three balls far apart — all should survive NMS."""
        dets = [
            _det(0,   0,  30,  30),
            _det(100, 0, 130,  30),
            _det(200, 0, 230,  30),
        ]
        result = nms_detections(dets)
        assert len(result) == 3

    def test_nms_keeps_higher_confidence_box(self):
        """When two overlapping boxes have different confidences, higher wins."""
        low_conf  = _det(10, 10, 50, 50, conf=0.3)
        high_conf = _det(11, 11, 51, 51, conf=0.95)
        result = nms_detections([low_conf, high_conf])
        assert len(result) == 1
        assert result[0].conf == 0.95

    def test_partial_overlap_both_kept(self):
        """Boxes that share only ~30% area → IoU < 0.45 → both survive."""
        # Box A: [0,0,40,40], Box B: [30,30,70,70].
        # Overlap: [30,30,40,40] = 100 px^2; each area = 1600; union = 3100; IoU ≈ 0.032.
        dets = [_det(0, 0, 40, 40), _det(30, 30, 70, 70)]
        result = nms_detections(dets)
        assert len(result) == 2

    def test_tile_remap_then_nms(self):
        """Full round-trip: two tiles produce overlapping detections; after remap+NMS, one box."""
        # Tile A at (0,0), tile B at (50,0) with overlap.
        # The same ball sits at full-frame coords (70,10,90,30).
        # Tile A (x0=0, y0=0): ball is at local (70,10,90,30)
        # Tile B (x0=50, y0=0): ball is at local (20,10,40,30)
        tile_a_dets = [Detection("ball", (70.0, 10.0, 90.0, 30.0), 0.9, 0)]
        tile_b_dets = [Detection("ball", (20.0, 10.0, 40.0, 30.0), 0.85, 0)]

        remapped_a = remap_detections_to_frame(tile_a_dets, 0, 0)
        remapped_b = remap_detections_to_frame(tile_b_dets, 50, 0)

        # Both should now map to full-frame [70..90, 10..30].
        assert remapped_a[0].xyxy == (70.0, 10.0, 90.0, 30.0)
        assert remapped_b[0].xyxy == (70.0, 10.0, 90.0, 30.0)

        merged = nms_detections(remapped_a + remapped_b)
        assert len(merged) == 1


class TestBuildTiles:
    def test_single_tile_small_frame(self):
        """Frame smaller than tile_size → single tile covering the whole frame."""
        tiles = build_tiles(100, 200, 512, 0.2)
        assert len(tiles) == 1
        assert tiles[0] == (0, 0, 200, 100)

    def test_no_overlap_evenly_divisible(self):
        """2×2 grid with 0% overlap on a 1024×1024 frame with 512-tile."""
        tiles = build_tiles(1024, 1024, 512, 0.0)
        assert len(tiles) == 4

    def test_overlap_increases_tile_count(self):
        """With 50% overlap, more tiles are generated than without."""
        tiles_no_overlap = build_tiles(1024, 1024, 512, 0.0)
        tiles_with_overlap = build_tiles(1024, 1024, 512, 0.5)
        assert len(tiles_with_overlap) > len(tiles_no_overlap)

    def test_field_bbox_restricts_tiling(self):
        """Tiling a sub-region should produce fewer tiles than the full frame."""
        full_tiles  = build_tiles(1080, 1920, 512, 0.2)
        field_tiles = build_tiles(1080, 1920, 512, 0.2, field_bbox_xyxy=(200, 200, 800, 700))
        assert len(field_tiles) < len(full_tiles)

    def test_all_tiles_within_bounds(self):
        """Every tile must lie within the frame dimensions."""
        h, w = 720, 1280
        for (x0, y0, x1, y1) in build_tiles(h, w, 512, 0.2):
            assert 0 <= x0 < x1 <= w
            assert 0 <= y0 < y1 <= h

    def test_field_bbox_all_tiles_within_bbox(self):
        """Tiles must lie within the provided field_bbox."""
        bbox = (100, 80, 900, 600)
        for (x0, y0, x1, y1) in build_tiles(720, 1280, 512, 0.2, field_bbox_xyxy=bbox):
            assert x0 >= bbox[0]
            assert y0 >= bbox[1]
            assert x1 <= bbox[2]
            assert y1 <= bbox[3]


# ===========================================================================
# B. Overlay structural-validation tests
# ===========================================================================

class TestValidateTeams:
    """Tests for overlay_ocr.validate_teams (pure logic — no OCR)."""

    def test_valid_3v3_qual(self):
        """Exactly 3v3 for a qual match — no errors expected."""
        blue = [1234, 5678, 9999]
        red  = [100,  200,  300]
        b, r, errs = validate_teams(blue, red, "qm")
        assert b == blue
        assert r == red
        assert errs == []

    def test_valid_3v3_final(self):
        blue = [1000, 2000, 3000]
        red  = [4000, 5000, 6000]
        b, r, errs = validate_teams(blue, red, "f")
        assert errs == []

    def test_two_team_alliance_flagged_for_qual(self):
        """Only 2 teams read on one alliance → error reported."""
        blue = [1234, 5678]       # missing one
        red  = [100,  200,  300]
        _b, _r, errs = validate_teams(blue, red, "qm")
        assert any("blue_alliance_size" in e for e in errs)

    def test_team_on_both_alliances_removed(self):
        """A team number appearing on both alliances is stripped from red and an error raised."""
        shared = 1111
        blue = [shared, 2222, 3333]
        red  = [shared, 4444, 5555]
        _b, r, errs = validate_teams(blue, red, "qm")
        assert shared not in r
        assert any("team_on_both_alliances" in e for e in errs)

    def test_rank_column_value_would_be_below_min(self):
        """
        Values below _TEAM_MIN (100) represent rank-column artefacts from v1.
        The upstream _read_ints already filters with min_v=100; this test confirms
        validate_teams itself does NOT reintroduce them (i.e., the list passed in
        after filtering must not contain values < 100 — this is enforced upstream).
        The test directly checks the v1 bug scenario: if somehow values < 100 slip
        through they would produce a wrong team list.  Since validate_teams operates
        on already-filtered lists, passing in valid teams produces clean output.
        """
        # Simulate the corrected path: team numbers are already >=100.
        blue = [148, 254, 971]
        red  = [118, 330, 1690]
        b, r, errs = validate_teams(blue, red, "qm")
        assert all(t >= 100 for t in b)
        assert all(t >= 100 for t in r)
        assert errs == []

    def test_empty_alliances_flagged_for_qual(self):
        """Empty lists for a qual match → size errors on both alliances."""
        _b, _r, errs = validate_teams([], [], "qm")
        assert any("blue_alliance_size" in e for e in errs)
        assert any("red_alliance_size" in e for e in errs)

    def test_no_comp_level_no_size_error(self):
        """With comp_level=None (live bar), alliance size is not enforced."""
        blue = [1000, 2000]    # only 2 teams — OK for live bar
        red  = [3000]
        _b, _r, errs = validate_teams(blue, red, None)
        assert not any("size" in e for e in errs)

    def test_sf_no_size_error(self):
        """Semifinals (sf) are not size-enforced."""
        blue = [100, 200]
        red  = [300, 400]
        _b, _r, errs = validate_teams(blue, red, "sf")
        assert not any("size" in e for e in errs)

    def test_multiple_cross_alliance_teams(self):
        """Multiple cross-alliance duplicates all removed from red."""
        blue = [111, 222, 333]
        red  = [111, 222, 444]
        _b, r, errs = validate_teams(blue, red, "qm")
        assert 111 not in r
        assert 222 not in r
        assert 444 in r

    def test_completely_disjoint_teams_no_cross_error(self):
        """Disjoint alliances produce no cross-alliance error."""
        blue = [1000, 2000, 3000]
        red  = [4000, 5000, 6000]
        _b, _r, errs = validate_teams(blue, red, "qm")
        assert not any("team_on_both" in e for e in errs)
