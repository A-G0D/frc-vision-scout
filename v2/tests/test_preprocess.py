"""tests/test_preprocess.py — unit tests for the refine.preprocess layer.

Split into two groups:
  A. PURE LOGIC (no cv2): band geometry, scale clamping, kernel sizing, config parsing,
     BandMap remap math, and the validate-BGR guard. These run in a bare env.
  B. IMAGE OPS (cv2-gated): shape/dtype invariants of CLAHE / unsharp / process /
     upscale on a synthetic BGR image. SKIPPED automatically when cv2 is absent.

numpy is assumed present (it's a hard pipeline dependency). cv2 is optional: the module
must import and the pure-logic tests must pass without it.
"""
from __future__ import annotations

import numpy as np
import pytest

from frcscout.refine.preprocess import (
    BandMap,
    PreprocessConfig,
    Preprocessor,
    _odd,
    _validate_bgr,
    clamp_scale,
    compute_field_band,
    cv2_available,
)

HAS_CV2 = cv2_available()
requires_cv2 = pytest.mark.skipif(not HAS_CV2, reason="cv2 not installed")


def _synthetic_bgr(h: int = 120, w: int = 200, seed: int = 0) -> np.ndarray:
    """A deterministic synthetic BGR uint8 image with structure (gradient + noise + edges)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Horizontal luma gradient (dim top -> bright bottom) to exercise CLAHE.
    grad = np.linspace(0, 255, h, dtype=np.float32)[:, None]
    img[..., 0] = grad
    img[..., 1] = grad
    img[..., 2] = grad
    # A few bright edges to exercise the unsharp mask.
    img[:, w // 3] = 255
    img[h // 2, :] = 255
    # Light noise so CLAHE/threshold paths have something to act on.
    noise = rng.integers(0, 30, size=(h, w, 3), dtype=np.uint8)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ===========================================================================
# A. PURE-LOGIC TESTS  (no cv2)
# ===========================================================================

class TestOdd:
    def test_even_rounds_up(self):
        assert _odd(4) == 5

    def test_odd_unchanged(self):
        assert _odd(5) == 5

    def test_floor_at_one(self):
        assert _odd(0) == 1
        assert _odd(-3) == 1

    def test_rounds_float(self):
        assert _odd(4.4) == 5      # rounds to 4 -> bumped to 5
        assert _odd(2.6) == 3      # rounds to 3 (already odd)


class TestComputeFieldBand:
    def test_basic_band(self):
        x0, y0, x1, y1 = compute_field_band(1000, 1920, 0.15, 0.55)
        assert (x0, x1) == (0, 1920)          # full width
        assert y0 == 150
        assert y1 == 550
        assert y1 > y0

    def test_full_width_always(self):
        for tf, bf in [(0.0, 1.0), (0.2, 0.3), (0.9, 0.95)]:
            x0, _, x1, _ = compute_field_band(720, 1280, tf, bf)
            assert x0 == 0 and x1 == 1280

    def test_fractions_clamped(self):
        # Out-of-range fractions clamp into [0,1].
        x0, y0, x1, y1 = compute_field_band(100, 100, -0.5, 1.7)
        assert y0 == 0
        assert y1 == 100

    def test_swapped_order_corrected(self):
        # bottom < top should be swapped, not produce an empty/inverted band.
        x0, y0, x1, y1 = compute_field_band(100, 100, 0.6, 0.2)
        assert y0 < y1
        assert y0 == 20
        assert y1 == 60

    def test_never_empty(self):
        # Degenerate equal fractions still yields >=1px tall, in-bounds band.
        x0, y0, x1, y1 = compute_field_band(100, 100, 0.5, 0.5)
        assert y1 >= y0 + 1
        assert 0 <= y0 < y1 <= 100

    def test_in_bounds_for_tiny_frame(self):
        x0, y0, x1, y1 = compute_field_band(1, 1, 0.0, 1.0)
        assert 0 <= y0 < y1 <= 1
        assert 0 <= x0 < x1 <= 1

    def test_resolution_independence(self):
        # Same fractions -> proportional pixel bands across resolutions.
        _, y0_720, _, y1_720 = compute_field_band(720, 1280, 0.25, 0.5)
        _, y0_1080, _, y1_1080 = compute_field_band(1080, 1920, 0.25, 0.5)
        assert abs(y0_720 / 720 - y0_1080 / 1080) < 0.01
        assert abs(y1_720 / 720 - y1_1080 / 1080) < 0.01


class TestClampScale:
    def test_never_downscales(self):
        assert clamp_scale(0.5, 100, 100, 4096) == 1.0
        assert clamp_scale(-2.0, 100, 100, 4096) == 1.0

    def test_passthrough_when_under_cap(self):
        assert clamp_scale(2.0, 100, 100, 4096) == 2.0

    def test_clamped_to_max_side(self):
        # 1000px longer side, scale 8 -> 8000 > 4096 -> clamp to 4096/1000.
        s = clamp_scale(8.0, 1000, 500, 4096)
        assert s == pytest.approx(4.096)
        assert 1000 * s == pytest.approx(4096)

    def test_already_over_cap_returns_one(self):
        # Source already exceeds the cap: never upscale, but never downscale either.
        assert clamp_scale(2.0, 5000, 200, 4096) == 1.0

    def test_uses_longer_side(self):
        # Width is the longer side; clamp must key off it, not height.
        s = clamp_scale(10.0, 2000, 100, 4096)
        assert 2000 * s == pytest.approx(4096)


class TestPreprocessConfig:
    def test_defaults(self):
        c = PreprocessConfig()
        assert c.enabled is True
        assert c.clahe_grid == 8
        assert c.band_bottom_frac == 0.55

    def test_from_none(self):
        c = PreprocessConfig.from_dict(None)
        assert isinstance(c, PreprocessConfig)
        assert c.enabled is True

    def test_from_flat_dict(self):
        c = PreprocessConfig.from_dict({"clahe_clip": 3.5, "band_scale": 4.0,
                                        "unsharp_amount": 1.0})
        assert c.clahe_clip == 3.5
        assert c.band_scale == 4.0
        assert c.unsharp_amount == 1.0

    def test_from_nested_dict(self):
        c = PreprocessConfig.from_dict({
            "clahe": {"clip": 1.5, "grid": 16},
            "unsharp": {"amount": 0.9, "sigma": 2.0},
            "band": {"top_frac": 0.1, "bottom_frac": 0.6, "scale": 3.0},
        })
        assert c.clahe_clip == 1.5
        assert c.clahe_grid == 16
        assert c.unsharp_amount == 0.9
        assert c.unsharp_sigma == 2.0
        assert c.band_top_frac == 0.1
        assert c.band_bottom_frac == 0.6
        assert c.band_scale == 3.0

    def test_unknown_keys_ignored(self):
        c = PreprocessConfig.from_dict({"nonsense": 123, "clahe_clip": 2.5})
        assert c.clahe_clip == 2.5

    def test_disabled_flag(self):
        c = PreprocessConfig.from_dict({"enabled": False})
        assert c.enabled is False

    def test_is_frozen(self):
        c = PreprocessConfig()
        with pytest.raises(Exception):
            c.clahe_clip = 9.0  # type: ignore[misc]


class TestBandMap:
    def test_src_wh(self):
        bm = BandMap(x0=10, y0=20, x1=110, y1=70, scale=2.0)
        assert bm.src_wh == (100, 50)

    def test_remap_xyxy_identity_scale(self):
        bm = BandMap(x0=5, y0=7, x1=105, y1=57, scale=1.0)
        assert bm.remap_xyxy((0, 0, 10, 10)) == (5, 7, 15, 17)

    def test_remap_xyxy_with_scale_and_offset(self):
        # Crop was upscaled 2x and offset by (100, 50). A crop box at (20,10,40,30)
        # is (10,5,20,15) in source band coords + offset.
        bm = BandMap(x0=100, y0=50, x1=300, y1=150, scale=2.0)
        out = bm.remap_xyxy((20, 10, 40, 30))
        assert out == (100 + 10, 50 + 5, 100 + 20, 50 + 15)
        assert out == (110.0, 55.0, 120.0, 65.0)

    def test_remap_point(self):
        bm = BandMap(x0=100, y0=50, x1=300, y1=150, scale=2.0)
        assert bm.remap_point((20, 10)) == (110.0, 55.0)

    def test_remap_zero_scale_safe(self):
        # Defensive: a zero scale must not divide-by-zero (treated as 1.0).
        bm = BandMap(x0=0, y0=0, x1=10, y1=10, scale=0.0)
        assert bm.remap_xyxy((1, 2, 3, 4)) == (1, 2, 3, 4)

    def test_roundtrip_band_to_frame(self):
        # A point known in full-frame coords -> into crop coords -> back via remap.
        bm = BandMap(x0=40, y0=30, x1=240, y1=130, scale=4.0)
        frame_pt = (120.0, 80.0)                       # inside the band
        crop_pt = ((frame_pt[0] - bm.x0) * bm.scale,
                   (frame_pt[1] - bm.y0) * bm.scale)
        back = bm.remap_point(crop_pt)
        assert back == pytest.approx(frame_pt)


class TestValidateBgr:
    def test_accepts_valid(self):
        _validate_bgr(np.zeros((4, 4, 3), dtype=np.uint8))  # no raise

    def test_rejects_non_ndarray(self):
        with pytest.raises(ValueError):
            _validate_bgr([[1, 2, 3]])  # type: ignore[arg-type]

    def test_rejects_wrong_channels(self):
        with pytest.raises(ValueError):
            _validate_bgr(np.zeros((4, 4, 4), dtype=np.uint8))

    def test_rejects_grayscale(self):
        with pytest.raises(ValueError):
            _validate_bgr(np.zeros((4, 4), dtype=np.uint8))

    def test_rejects_wrong_dtype(self):
        with pytest.raises(ValueError):
            _validate_bgr(np.zeros((4, 4, 3), dtype=np.float32))


class TestDisabledNoCv2Path:
    """The fully-disabled path must work WITHOUT cv2 (pure copy) and preserve shape."""

    def test_process_disabled_returns_copy(self):
        frame = _synthetic_bgr(32, 48)
        pp = Preprocessor(PreprocessConfig(enabled=False))
        out = pp.process(frame)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8
        assert np.array_equal(out, frame)
        assert out is not frame                  # must be a copy, not an alias

    def test_field_band_rect_pure(self):
        # field_band_rect is pure geometry — no cv2 needed.
        pp = Preprocessor(PreprocessConfig(band_top_frac=0.2, band_bottom_frac=0.5))
        assert pp.field_band_rect(1000, 1920) == (0, 200, 1920, 500)


# ===========================================================================
# B. IMAGE-OP TESTS  (require cv2)
# ===========================================================================

@requires_cv2
class TestApplyClahe:
    def test_shape_dtype_preserved(self):
        from frcscout.refine.preprocess import apply_clahe
        frame = _synthetic_bgr()
        out = apply_clahe(frame, clip=2.0, grid=8)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_actually_changes_contrast(self):
        from frcscout.refine.preprocess import apply_clahe
        frame = _synthetic_bgr()
        out = apply_clahe(frame, clip=4.0, grid=8)
        # CLAHE on a gradient should change pixels (not a no-op).
        assert not np.array_equal(out, frame)

    def test_grid_one_is_valid(self):
        from frcscout.refine.preprocess import apply_clahe
        out = apply_clahe(_synthetic_bgr(), clip=2.0, grid=1)
        assert out.dtype == np.uint8


@requires_cv2
class TestUnsharpMask:
    def test_shape_dtype_preserved(self):
        from frcscout.refine.preprocess import unsharp_mask
        frame = _synthetic_bgr()
        out = unsharp_mask(frame, amount=0.6, sigma=1.2)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_amount_zero_is_noop_copy(self):
        from frcscout.refine.preprocess import unsharp_mask
        frame = _synthetic_bgr()
        out = unsharp_mask(frame, amount=0.0)
        assert np.array_equal(out, frame)
        assert out is not frame

    def test_sharpening_changes_edges(self):
        from frcscout.refine.preprocess import unsharp_mask
        frame = _synthetic_bgr()
        out = unsharp_mask(frame, amount=1.5, sigma=1.0)
        assert not np.array_equal(out, frame)

    def test_high_threshold_suppresses_changes(self):
        from frcscout.refine.preprocess import unsharp_mask
        frame = _synthetic_bgr()
        # An impossibly-high threshold means no pixel's detail qualifies -> unchanged.
        out = unsharp_mask(frame, amount=1.5, sigma=1.0, threshold=999)
        assert np.array_equal(out, frame)

    def test_no_overflow_wraparound(self):
        from frcscout.refine.preprocess import unsharp_mask
        # A near-white image sharpened hard must clip to 255, never wrap to 0.
        frame = np.full((20, 20, 3), 250, dtype=np.uint8)
        frame[10, 10] = 0   # one dark pixel creates a strong edge
        out = unsharp_mask(frame, amount=3.0, sigma=1.0)
        assert out.dtype == np.uint8
        assert out.max() <= 255 and out.min() >= 0


@requires_cv2
class TestUpscaleRoi:
    def test_upscaled_dims(self):
        from frcscout.refine.preprocess import upscale_roi
        frame = _synthetic_bgr(100, 200)
        crop, bm = upscale_roi(frame, (0, 20, 200, 70), scale=2.0)
        assert crop.shape[0] == 50 * 2          # height of band * scale
        assert crop.shape[1] == 200 * 2
        assert crop.shape[2] == 3
        assert crop.dtype == np.uint8
        assert bm.scale == 2.0

    def test_bandmap_offsets(self):
        from frcscout.refine.preprocess import upscale_roi
        frame = _synthetic_bgr(100, 200)
        _, bm = upscale_roi(frame, (10, 20, 110, 70), scale=1.0)
        assert (bm.x0, bm.y0, bm.x1, bm.y1) == (10, 20, 110, 70)

    def test_scale_clamped_against_max_side(self):
        from frcscout.refine.preprocess import upscale_roi
        frame = _synthetic_bgr(100, 200)
        # Band is 200 wide; scale 50 -> 10000 > max_side 400 -> clamp to 2.0.
        crop, bm = upscale_roi(frame, (0, 0, 200, 100), scale=50.0, max_side=400)
        assert bm.scale == pytest.approx(2.0)
        assert max(crop.shape[:2]) <= 400

    def test_rect_clipped_to_frame(self):
        from frcscout.refine.preprocess import upscale_roi
        frame = _synthetic_bgr(100, 200)
        # Rect exceeds bounds; must be clipped, crop stays in-frame & non-empty.
        crop, bm = upscale_roi(frame, (-10, -10, 500, 500), scale=1.0)
        assert bm.x1 <= 200 and bm.y1 <= 100
        assert crop.size > 0

    def test_remap_lands_inside_source_band(self):
        from frcscout.refine.preprocess import upscale_roi
        frame = _synthetic_bgr(100, 200)
        crop, bm = upscale_roi(frame, (20, 30, 180, 90), scale=3.0)
        ch, cw = crop.shape[:2]
        # A box spanning the whole crop must remap back to the source band rect.
        rm = bm.remap_xyxy((0, 0, cw, ch))
        assert rm == pytest.approx((20.0, 30.0, 180.0, 90.0))


@requires_cv2
class TestProcessFullFrame:
    def test_shape_dtype_invariant(self):
        frame = _synthetic_bgr(180, 320)
        pp = Preprocessor()                      # all stages on by default
        out = pp.process(frame)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_process_changes_image_by_default(self):
        frame = _synthetic_bgr(180, 320)
        out = Preprocessor().process(frame)
        assert not np.array_equal(out, frame)

    def test_clahe_only(self):
        frame = _synthetic_bgr(64, 64)
        pp = Preprocessor(PreprocessConfig(clahe_enabled=True, unsharp_enabled=False))
        out = pp.process(frame)
        assert out.shape == frame.shape and out.dtype == np.uint8

    def test_unsharp_only(self):
        frame = _synthetic_bgr(64, 64)
        pp = Preprocessor(PreprocessConfig(clahe_enabled=False, unsharp_enabled=True,
                                           unsharp_amount=1.0))
        out = pp.process(frame)
        assert out.shape == frame.shape and out.dtype == np.uint8

    def test_all_stages_off_is_copy(self):
        frame = _synthetic_bgr(64, 64)
        pp = Preprocessor(PreprocessConfig(clahe_enabled=False, unsharp_enabled=False))
        out = pp.process(frame)
        assert np.array_equal(out, frame)
        assert out is not frame

    def test_rejects_bad_input(self):
        pp = Preprocessor()
        with pytest.raises(ValueError):
            pp.process(np.zeros((10, 10), dtype=np.uint8))


@requires_cv2
class TestUpscaleFieldBand:
    def test_returns_enhanced_upscaled_crop(self):
        frame = _synthetic_bgr(200, 320)
        pp = Preprocessor(PreprocessConfig(band_top_frac=0.2, band_bottom_frac=0.6,
                                           band_scale=2.0))
        crop, bm = pp.upscale_field_band(frame, enhance=True)
        # Band height = (0.6-0.2)*200 = 80 -> *2 = 160; width 320 -> 640.
        assert crop.shape[0] == 80 * 2
        assert crop.shape[1] == 320 * 2
        assert crop.dtype == np.uint8
        assert bm.scale == 2.0
        assert (bm.y0, bm.y1) == (40, 120)

    def test_enhance_false_matches_plain_upscale(self):
        frame = _synthetic_bgr(200, 320)
        pp = Preprocessor(PreprocessConfig(band_scale=2.0))
        crop, bm = pp.upscale_field_band(frame, enhance=False)
        assert crop.dtype == np.uint8
        assert bm.scale == 2.0
        # enhance=False -> the crop equals the raw upscaled band (no contrast change).
        from frcscout.refine.preprocess import upscale_roi
        rect = pp.field_band_rect(*frame.shape[:2])
        plain, _ = upscale_roi(frame, rect, 2.0, pp.cfg.band_max_side)
        assert np.array_equal(crop, plain)

    def test_remap_box_into_full_frame(self):
        frame = _synthetic_bgr(200, 320)
        pp = Preprocessor(PreprocessConfig(band_top_frac=0.25, band_bottom_frac=0.5,
                                           band_scale=2.0))
        crop, bm = pp.upscale_field_band(frame, enhance=True)
        ch, cw = crop.shape[:2]
        full = bm.remap_xyxy((0, 0, cw, ch))
        # The full-frame box must equal the band rect and lie within the frame.
        assert full[0] == 0 and full[2] == 320
        assert full[1] == pytest.approx(50)      # 0.25*200
        assert full[3] == pytest.approx(100)     # 0.5*200
