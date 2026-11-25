"""Frame preprocessing to aid DISTANT detection — runs before the detectors.

Far-downfield robots/balls in a 1080p broadcast occupy very few pixels and sit in the
low-contrast haze of the stands lighting. Three cheap, classical operations recover most
of that lost signal without any neural net:

    1. CLAHE  — Contrast-Limited Adaptive Histogram Equalisation on the luma channel only
                (so colours/bumper hues survive for the alliance classifier). Local contrast
                lifts the dim far end without blowing out the bright near end.
    2. Unsharp mask — re-sharpens edges softened by JPEG/stream compression and by the
                CLAHE smoothing, so small objects keep a crisp boundary for the detector.
    3. ROI upscale — optionally crop the far "field band" (the upper strip where distant
                objects live) and upscale it, giving the detector more pixels on target.
                Returned as a SEPARATE crop + an offset/scale map so detections can be
                remapped to full-frame coords (the engine runs the detector on the crop).

Contract:
    * BGR uint8 ndarray in -> BGR uint8 ndarray out, SAME H×W×3 for `process()`
      (the full-frame contract the rest of the pipeline depends on is never broken).
    * `upscale_field_band()` returns a *different-sized* enhanced crop plus a `BandMap`
      describing how to map crop-pixels back to the source frame.

Guarding:
    * numpy is a hard dependency (used pipeline-wide as `np`).
    * cv2 is OPTIONAL here: the module PARSES and all pure-logic helpers
      (band geometry, scale clamping, kernel sizing, config) run with NO cv2.
      The actual pixel operations import cv2 lazily and raise a clear error if it is
      absent; `cv2_available()` lets callers/tests probe first.

Resolution independence: the field-band ROI is expressed as FRACTIONS of frame height
(matching the overlay_ocr.py ROI-as-fractions convention), never hardcoded pixel rows,
so the same config works on 720p and 1080p captures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# cv2 guard — keep the module importable without OpenCV
# ---------------------------------------------------------------------------

def cv2_available() -> bool:
    """True if OpenCV can be imported in this environment."""
    try:
        import cv2  # noqa: F401
        return True
    except Exception:
        return False


def _cv2():
    """Lazily import cv2, raising a clear, actionable error if it is missing."""
    try:
        import cv2
        return cv2
    except Exception as exc:  # pragma: no cover - exercised only in cv2-less envs
        raise RuntimeError(
            "frcscout.refine.preprocess requires OpenCV (cv2) for image operations; "
            "pure-logic helpers (compute_field_band, clamp_scale, PreprocessConfig) "
            "work without it."
        ) from exc


# ---------------------------------------------------------------------------
# Config — all knobs with sane defaults; resolution-independent
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreprocessConfig:
    """Preprocessing parameters. Spatial only — no timing lives here.

    CLAHE:
        clahe_clip:      contrast-limit; higher = stronger local contrast (typical 1.5–4).
        clahe_grid:      tile grid edge (NxN). Larger N = more local adaptation.
    Unsharp mask:
        unsharp_amount:  blend weight of the high-pass (0 disables; ~0.5–1.5 typical).
        unsharp_sigma:   Gaussian blur sigma (px) defining the "low-pass" to subtract.
        unsharp_threshold: only re-apply sharpening where |detail| exceeds this (0 = all).
    ROI field band (fractions of frame HEIGHT; the far/distant strip):
        band_top_frac:   top edge of the band (0 = image top).
        band_bottom_frac:bottom edge of the band (1 = image bottom).
        band_scale:      upscale factor for the band crop (>=1).
        band_max_side:   cap on the longer side of the upscaled crop (px) to bound cost.
    """
    enabled: bool = True
    clahe_enabled: bool = True
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    unsharp_enabled: bool = True
    unsharp_amount: float = 0.6
    unsharp_sigma: float = 1.2
    unsharp_threshold: int = 0
    band_top_frac: float = 0.15
    band_bottom_frac: float = 0.55
    band_scale: float = 2.0
    band_max_side: int = 4096

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "PreprocessConfig":
        """Build from a plain dict (e.g. a slice of deploy.yaml), ignoring unknown keys
        and falling back to defaults for any missing one."""
        cfg = cfg or {}
        defaults = cls()
        # Support a couple of friendly aliases without leaking them into the dataclass.
        clahe = cfg.get("clahe", {}) if isinstance(cfg.get("clahe"), dict) else {}
        unsharp = cfg.get("unsharp", {}) if isinstance(cfg.get("unsharp"), dict) else {}
        band = cfg.get("band", {}) if isinstance(cfg.get("band"), dict) else {}

        def pick(*keys, default):
            for src, key in keys:
                if key in src:
                    return src[key]
            return default

        return cls(
            enabled=bool(cfg.get("enabled", defaults.enabled)),
            clahe_enabled=bool(pick((cfg, "clahe_enabled"), (clahe, "enabled"),
                                    default=defaults.clahe_enabled)),
            clahe_clip=float(pick((cfg, "clahe_clip"), (clahe, "clip"),
                                  default=defaults.clahe_clip)),
            clahe_grid=int(pick((cfg, "clahe_grid"), (clahe, "grid"),
                                default=defaults.clahe_grid)),
            unsharp_enabled=bool(pick((cfg, "unsharp_enabled"), (unsharp, "enabled"),
                                      default=defaults.unsharp_enabled)),
            unsharp_amount=float(pick((cfg, "unsharp_amount"), (unsharp, "amount"),
                                      default=defaults.unsharp_amount)),
            unsharp_sigma=float(pick((cfg, "unsharp_sigma"), (unsharp, "sigma"),
                                     default=defaults.unsharp_sigma)),
            unsharp_threshold=int(pick((cfg, "unsharp_threshold"), (unsharp, "threshold"),
                                       default=defaults.unsharp_threshold)),
            band_top_frac=float(pick((cfg, "band_top_frac"), (band, "top_frac"),
                                     default=defaults.band_top_frac)),
            band_bottom_frac=float(pick((cfg, "band_bottom_frac"), (band, "bottom_frac"),
                                        default=defaults.band_bottom_frac)),
            band_scale=float(pick((cfg, "band_scale"), (band, "scale"),
                                  default=defaults.band_scale)),
            band_max_side=int(pick((cfg, "band_max_side"), (band, "max_side"),
                                   default=defaults.band_max_side)),
        )


@dataclass(frozen=True)
class BandMap:
    """Describes how an upscaled field-band crop maps back to the source frame.

    A crop-pixel (cx, cy) maps to a source-frame pixel:
        src_x = x0 + cx / scale
        src_y = y0 + cy / scale
    so a detector box found on the crop is remapped to full-frame via `remap_xyxy`.
    """
    x0: int
    y0: int
    x1: int
    y1: int
    scale: float

    @property
    def src_wh(self) -> tuple[int, int]:
        return self.x1 - self.x0, self.y1 - self.y0

    def remap_xyxy(
        self, box: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Map an [x1,y1,x2,y2] box from crop coords back to full-frame coords."""
        s = self.scale if self.scale else 1.0
        bx1, by1, bx2, by2 = box
        return (
            self.x0 + bx1 / s,
            self.y0 + by1 / s,
            self.x0 + bx2 / s,
            self.y0 + by2 / s,
        )

    def remap_point(self, pt: tuple[float, float]) -> tuple[float, float]:
        """Map a crop-space point back to full-frame coords."""
        s = self.scale if self.scale else 1.0
        return self.x0 + pt[0] / s, self.y0 + pt[1] / s


# ---------------------------------------------------------------------------
# Pure-logic helpers (NO cv2 required) — unit-testable in a bare env
# ---------------------------------------------------------------------------

def _odd(n: int) -> int:
    """Nearest odd integer >= 1 (kernel sizes must be odd & positive)."""
    n = int(round(n))
    if n < 1:
        return 1
    return n if n % 2 == 1 else n + 1


def compute_field_band(
    h: int,
    w: int,
    top_frac: float,
    bottom_frac: float,
) -> tuple[int, int, int, int]:
    """Pixel rectangle (x0, y0, x1, y1) of the far field band for a (h, w) frame.

    The band spans the full width and the vertical strip [top_frac, bottom_frac] of the
    height. Fractions are clamped to [0,1] and ordered; the result is guaranteed to be a
    valid, non-empty, in-bounds rectangle (at least 1px tall).
    """
    h = max(1, int(h))
    w = max(1, int(w))
    t = min(max(float(top_frac), 0.0), 1.0)
    b = min(max(float(bottom_frac), 0.0), 1.0)
    if b < t:
        t, b = b, t
    y0 = int(np.floor(t * h))
    y1 = int(np.ceil(b * h))
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))
    return 0, y0, w, y1


def clamp_scale(
    scale: float,
    src_w: int,
    src_h: int,
    max_side: int,
) -> float:
    """Clamp an upscale factor so the longer side of the result stays <= max_side.

    Never returns < 1.0 (we don't downscale the band). If the source already exceeds
    max_side, returns 1.0 (no upscale, but never a downscale here).
    """
    scale = float(scale)
    if scale < 1.0:
        scale = 1.0
    longer = max(int(src_w), int(src_h), 1)
    if max_side and longer * scale > max_side:
        scale = max_side / longer
    return max(1.0, scale)


def _validate_bgr(frame: np.ndarray) -> None:
    """Raise ValueError unless `frame` is an H×W×3 uint8 ndarray (BGR)."""
    if not isinstance(frame, np.ndarray):
        raise ValueError(f"frame must be a numpy ndarray, got {type(frame)!r}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must be H×W×3 (BGR); got shape {frame.shape!r}")
    if frame.dtype != np.uint8:
        raise ValueError(f"frame must be uint8; got dtype {frame.dtype!r}")


# ---------------------------------------------------------------------------
# Image operations (cv2-guarded) — each preserves uint8 BGR shape/dtype
# ---------------------------------------------------------------------------

def apply_clahe(
    frame: np.ndarray,
    clip: float = 2.0,
    grid: int = 8,
) -> np.ndarray:
    """CLAHE on the luma (L) channel in LAB space; chroma untouched.

    Operating on luma only keeps bumper hues intact for the alliance classifier while
    still lifting local contrast in the dim far field. Returns a new uint8 BGR array of
    identical shape.
    """
    _validate_bgr(frame)
    cv2 = _cv2()
    grid = max(1, int(grid))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(grid, grid))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def unsharp_mask(
    frame: np.ndarray,
    amount: float = 0.6,
    sigma: float = 1.2,
    threshold: int = 0,
) -> np.ndarray:
    """Unsharp mask: frame + amount * (frame - GaussianBlur(frame)).

    `threshold` (0–255) leaves low-contrast detail (noise) untouched: only pixels whose
    high-pass magnitude exceeds the threshold get sharpened. Returns uint8 BGR, same shape.
    `amount <= 0` is a no-op copy.
    """
    _validate_bgr(frame)
    cv2 = _cv2()
    if amount <= 0:
        return frame.copy()
    sigma = max(1e-3, float(sigma))
    # Kernel size derived from sigma (OpenCV will auto-size if ksize=(0,0)).
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma, sigmaY=sigma)
    # Work in float to avoid wrap-around, recombine, then clip back to uint8.
    f = frame.astype(np.float32)
    bl = blurred.astype(np.float32)
    detail = f - bl
    sharp = f + float(amount) * detail
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    if threshold and threshold > 0:
        # Only keep the sharpened result where the detail is strong enough.
        mag = np.abs(detail).max(axis=2)            # per-pixel detail strength
        mask = (mag >= float(threshold))[:, :, None]
        out = np.where(mask, sharp, frame)
        return out.astype(np.uint8)
    return sharp


def upscale_roi(
    frame: np.ndarray,
    rect_xyxy: tuple[int, int, int, int],
    scale: float,
    max_side: int = 4096,
) -> tuple[np.ndarray, BandMap]:
    """Crop `rect_xyxy` from `frame` and upscale it by (clamped) `scale`.

    Returns (crop, BandMap). The crop is uint8 BGR; BandMap carries the offset + the
    EFFECTIVE scale (after clamping to max_side) so detections can be remapped to the
    full frame. INTER_CUBIC is used for a small, detail-preserving upscale.
    """
    _validate_bgr(frame)
    cv2 = _cv2()
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = rect_xyxy
    x0 = max(0, min(int(x0), w - 1))
    y0 = max(0, min(int(y0), h - 1))
    x1 = max(x0 + 1, min(int(x1), w))
    y1 = max(y0 + 1, min(int(y1), h))
    crop = frame[y0:y1, x0:x1]
    sw, sh = x1 - x0, y1 - y0
    eff = clamp_scale(scale, sw, sh, max_side)
    if eff != 1.0:
        new_w = max(1, int(round(sw * eff)))
        new_h = max(1, int(round(sh * eff)))
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return crop, BandMap(x0=x0, y0=y0, x1=x1, y1=y1, scale=eff)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Preprocessor:
    """Stateless enhancer applied to frames before detection.

    Two entry points:
        process(frame)            -> full-frame enhanced BGR (SAME H×W×3 uint8).
        upscale_field_band(frame) -> (enhanced+upscaled band crop, BandMap).

    Construct from a dict (deploy.yaml slice) via `Preprocessor(cfg_dict)` or pass a
    PreprocessConfig directly.
    """

    def __init__(self, cfg: Optional[dict | PreprocessConfig] = None):
        if isinstance(cfg, PreprocessConfig):
            self.cfg = cfg
        else:
            self.cfg = PreprocessConfig.from_dict(cfg)

    # -- whole-frame contrast/sharpen (shape-preserving) -------------------- #
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE then unsharp to the whole frame. Returns same-shape uint8 BGR.

        If preprocessing is disabled (or a stage is disabled) the corresponding step is
        skipped; with everything disabled this returns a copy of the input unchanged.
        A NO-OP path never requires cv2.
        """
        _validate_bgr(frame)
        c = self.cfg
        if not c.enabled:
            return frame.copy()
        out = frame
        produced_new = False
        if c.clahe_enabled:
            out = apply_clahe(out, clip=c.clahe_clip, grid=c.clahe_grid)
            produced_new = True
        if c.unsharp_enabled and c.unsharp_amount > 0:
            out = unsharp_mask(
                out,
                amount=c.unsharp_amount,
                sigma=c.unsharp_sigma,
                threshold=c.unsharp_threshold,
            )
            produced_new = True
        # Guarantee we never hand back the caller's exact array (avoid aliasing surprises).
        return out if produced_new else frame.copy()

    # -- far-field ROI upscale (size-changing; returns remap) --------------- #
    def field_band_rect(self, h: int, w: int) -> tuple[int, int, int, int]:
        """The far-field band rectangle (x0,y0,x1,y1) for a (h, w) frame — pure geometry."""
        return compute_field_band(h, w, self.cfg.band_top_frac, self.cfg.band_bottom_frac)

    def upscale_field_band(
        self,
        frame: np.ndarray,
        enhance: bool = True,
    ) -> tuple[np.ndarray, BandMap]:
        """Crop the far-field band, optionally enhance it, and upscale it.

        Returns (crop, BandMap). When `enhance` is True the band is CLAHE+unsharp'd before
        upscaling so the detector sees both more pixels and more contrast on distant
        objects. Use BandMap.remap_xyxy to put crop detections back into full-frame coords.
        """
        _validate_bgr(frame)
        h, w = frame.shape[:2]
        rect = self.field_band_rect(h, w)
        if enhance and self.cfg.enabled:
            # Enhance only the band (cheaper than the whole frame), then upscale.
            x0, y0, x1, y1 = rect
            band = frame[y0:y1, x0:x1]
            band = self._enhance_only(band)
            cv2 = _cv2()
            sw, sh = x1 - x0, y1 - y0
            eff = clamp_scale(self.cfg.band_scale, sw, sh, self.cfg.band_max_side)
            if eff != 1.0:
                nw = max(1, int(round(sw * eff)))
                nh = max(1, int(round(sh * eff)))
                band = cv2.resize(band, (nw, nh), interpolation=cv2.INTER_CUBIC)
            return band, BandMap(x0=x0, y0=y0, x1=x1, y1=y1, scale=eff)
        return upscale_roi(frame, rect, self.cfg.band_scale, self.cfg.band_max_side)

    def _enhance_only(self, img: np.ndarray) -> np.ndarray:
        """CLAHE+unsharp on an arbitrary BGR crop (no resize). Same shape/dtype out."""
        out = img
        produced = False
        if self.cfg.clahe_enabled:
            out = apply_clahe(out, clip=self.cfg.clahe_clip, grid=self.cfg.clahe_grid)
            produced = True
        if self.cfg.unsharp_enabled and self.cfg.unsharp_amount > 0:
            out = unsharp_mask(
                out,
                amount=self.cfg.unsharp_amount,
                sigma=self.cfg.unsharp_sigma,
                threshold=self.cfg.unsharp_threshold,
            )
            produced = True
        return out if produced else img.copy()
