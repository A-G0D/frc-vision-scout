"""Bumper team-number OCR — read the printed team number off a robot's bumper.

This is a REFINEMENT layer, not a hot-path detector. The authoritative team list comes
from the broadcast overlay (`detect/overlay_ocr.py`); robot *identity* (which track is
which team) is assigned by the fusion layer. This module supplies an independent,
per-robot vote: crop the bumper band of a detected robot box and OCR the digits printed
on it. The fusion layer can use that vote to break ties / confirm assignments.

Why this is cheap and optional:
  • Runs only on robot boxes we already have (≤6), and only when asked — never at 60Hz.
  • The bumper number is small and motion-blurred most frames, so a single read is weak;
    the value is in *accumulating* votes across frames (the caller does that), each with a
    confidence this module returns.

Design mirrors `detect/overlay_ocr.py`:
  • easyocr is loaded LAZILY and GUARDED — the file parses with no easyocr/cv2 present.
  • All pure logic (crop-region geometry, digit plausibility, OCR-result parsing) lives in
    module-level functions that need NO heavy deps, so they are unit-testable with a stub.
  • Team-number plausibility caps reject garbage (a 5-digit read, "0", a stray rank digit).

The bumper band is the LOWER part of the robot box (bumpers sit low on the robot — the
same lower-40% convention the autolabel bumper-color gate uses), horizontally inset to
drop the box edges where neighbouring robots / field bleed in.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

# Heavy deps are imported lazily inside methods so this module PARSES without them.
# Pure logic below must never reference cv2 / numpy / easyocr at import time.

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# An OCR result item: (text, confidence). easyocr's native form is
# (bbox, text, conf); we normalise to (text, conf) at the boundary so the pure
# parser has no dependency on bbox geometry. `OcrFn` is the injectable stub seam
# used by tests and by `read(..., ocr_fn=...)`.
OcrItem = tuple[str, float]
OcrFn = Callable[[object], Sequence[OcrItem]]

# Bumper xyxy in absolute image pixels.
BBox = tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Plausibility — what is a real FRC team number?
# ---------------------------------------------------------------------------

# FRC team numbers are assigned sequentially starting at 1. As of the 2026 season
# the highest issued numbers are in the low 10000s; we allow generous head-room and
# reject anything past it as an OCR artefact (e.g. two numbers fused, a timer read).
TEAM_MIN = 1
TEAM_MAX = 12000

# FRC numbers have grown into the low five digits, so the issued range (TEAM_MAX=12000)
# spans up to 5 digits. A read with MORE digits than this is two numbers fused (or noise)
# and is rejected outright rather than truncated. Kept consistent with TEAM_MAX: any
# value that fits the digit cap but exceeds the ceiling is still rejected by plausible_team.
MAX_TEAM_DIGITS = len(str(TEAM_MAX))  # == 5


def plausible_team(n: int) -> bool:
    """True iff `n` is a syntactically plausible FRC team number.

    Rejects: non-positive values, values above the issued-number ceiling, and values
    with more digits than a real bumper number can have.
    """
    if n < TEAM_MIN or n > TEAM_MAX:
        return False
    if len(str(n)) > MAX_TEAM_DIGITS:
        return False
    return True


def _digits_only(text: str) -> str:
    """Keep only ASCII digits from an OCR fragment."""
    return "".join(ch for ch in text if ch.isdigit())


# ---------------------------------------------------------------------------
# Crop geometry (pure — no image data needed, just the box)
# ---------------------------------------------------------------------------

def bumper_roi_xyxy(
    robot_xyxy: BBox,
    *,
    band_top_frac: float = 0.60,
    band_bottom_frac: float = 1.0,
    x_inset_frac: float = 0.08,
    clip_wh: Optional[tuple[int, int]] = None,
) -> Optional[BBox]:
    """Compute the bumper-band sub-rectangle of a robot box, in absolute image pixels.

    The bumper (carrying the team number) sits low on the robot, so we take a vertical
    band from `band_top_frac`..`band_bottom_frac` of the box height (default lower 40%),
    and inset each side by `x_inset_frac` of the box width to avoid the box edges where
    adjacent robots / carpet bleed in.

    Args:
        robot_xyxy: (x1, y1, x2, y2) robot box in image px (x2>x1, y2>y1 expected).
        band_top_frac/band_bottom_frac: vertical band as fractions of box height (0..1).
        x_inset_frac: horizontal inset per side as a fraction of box width (0..<0.5).
        clip_wh: optional (W, H) frame size to clip the ROI into. When given, the ROI is
            clamped to the frame; if that leaves nothing, returns None.

    Returns:
        (x1, y1, x2, y2) ints-as-floats for the bumper band, or None if the input box is
        degenerate / the band collapses / it falls entirely outside the frame.
    """
    x1, y1, x2, y2 = (float(v) for v in robot_xyxy)
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None

    # Normalise band fractions defensively (callers may pass them from config).
    top = min(max(band_top_frac, 0.0), 1.0)
    bot = min(max(band_bottom_frac, 0.0), 1.0)
    if bot <= top:
        return None
    inset = min(max(x_inset_frac, 0.0), 0.49)

    bx1 = x1 + inset * w
    bx2 = x2 - inset * w
    by1 = y1 + top * h
    by2 = y1 + bot * h
    if bx2 <= bx1 or by2 <= by1:
        return None

    if clip_wh is not None:
        W, H = int(clip_wh[0]), int(clip_wh[1])
        bx1 = max(0.0, min(bx1, W))
        bx2 = max(0.0, min(bx2, W))
        by1 = max(0.0, min(by1, H))
        by2 = max(0.0, min(by2, H))
        if bx2 <= bx1 or by2 <= by1:
            return None

    return (bx1, by1, bx2, by2)


# ---------------------------------------------------------------------------
# OCR-result parsing (pure — operates on (text, conf) items, no easyocr needed)
# ---------------------------------------------------------------------------

def parse_team_from_ocr(
    items: Sequence[OcrItem],
    *,
    min_conf: float = 0.30,
) -> tuple[Optional[int], float]:
    """Pick the best plausible team number out of raw OCR fragments.

    Each item is (text, conf). We:
      1. strip non-digits from each fragment,
      2. drop empty / sub-threshold / implausible candidates,
      3. among survivors, prefer the HIGHEST-confidence read; ties broken by more digits
         (a 4-digit team beats a 1-digit fragment that happens to share confidence — long
         numbers are harder to hallucinate).

    Returns (team or None, confidence). Confidence is 0.0 when nothing plausible is found.
    Note: confidence-thresholding happens HERE, not in plausibility — `plausible_team`
    stays a pure syntactic check that tests can exercise independently.
    """
    best_team: Optional[int] = None
    best_conf = 0.0
    best_ndigits = 0

    for item in items:
        text, conf = item[0], float(item[1])
        if conf < min_conf:
            continue
        digits = _digits_only(text)
        if not digits:
            continue
        # Leading-zero / overlong reads are not real team numbers.
        if len(digits) > MAX_TEAM_DIGITS:
            continue
        value = int(digits)
        if not plausible_team(value):
            continue
        ndigits = len(digits)
        if conf > best_conf or (conf == best_conf and ndigits > best_ndigits):
            best_team, best_conf, best_ndigits = value, conf, ndigits

    if best_team is None:
        return None, 0.0
    return best_team, best_conf


# ---------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------

class BumperOCR:
    """Reads the team number printed on a robot's bumper.

    Lazily constructs one easyocr reader and reuses it across calls (construction is
    expensive). easyocr is imported only on first real read and is fully guarded, so the
    class can be instantiated and its pure helpers used with no easyocr installed.

    Config keys (all optional; sensible defaults):
        refine.bumper_ocr.band_top_frac     (default 0.60)  lower-40% band start
        refine.bumper_ocr.band_bottom_frac  (default 1.00)
        refine.bumper_ocr.x_inset_frac      (default 0.08)
        refine.bumper_ocr.upscale           (default 4)      OCR likes big glyphs
        refine.bumper_ocr.min_conf          (default 0.30)   per-read confidence floor
        refine.bumper_ocr.min_box_px        (default 18)     skip tiny/far robots
    """

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        # Support both a flat dict and a Config-style object; fall back to plain .get.
        get = cfg.get
        self.band_top_frac = float(get("band_top_frac", 0.60))
        self.band_bottom_frac = float(get("band_bottom_frac", 1.0))
        self.x_inset_frac = float(get("x_inset_frac", 0.08))
        self.upscale = int(get("upscale", 4))
        self.min_conf = float(get("min_conf", 0.30))
        self.min_box_px = int(get("min_box_px", 18))
        self._ocr = None  # lazy easyocr.Reader

    # ------------------------------------------------------------------
    # Lazy OCR loader (guarded — only needed when a real frame arrives)
    # ------------------------------------------------------------------

    def _ensure(self):
        if self._ocr is None:
            import easyocr  # guarded import
            self._ocr = easyocr.Reader(["en"], gpu=False, verbose=False)

    def _default_ocr_fn(self, crop) -> list[OcrItem]:
        """Run the real easyocr digit reader on a preprocessed crop, return (text, conf)."""
        self._ensure()
        # allowlist keeps the model on digits; detail=1 returns (bbox, text, conf).
        raw = self._ocr.readtext(crop, allowlist="0123456789", detail=1)
        out: list[OcrItem] = []
        for entry in raw:
            # easyocr returns (bbox, text, conf); be defensive about shape.
            if len(entry) >= 3:
                out.append((entry[1], float(entry[2])))
            elif len(entry) == 2:
                out.append((entry[0], float(entry[1])))
        return out

    # ------------------------------------------------------------------
    # Image preprocessing (guarded — cv2/numpy only touched here)
    # ------------------------------------------------------------------

    def _preprocess(self, crop):
        """Grayscale -> Otsu threshold -> upscale, to make small glyphs OCR-friendly.

        Returns the processed image, or the original crop if cv2 is unavailable (the
        real `_default_ocr_fn` would then still receive *something* usable). Never raises
        on a bad crop — returns None so `read` can bail cleanly.
        """
        try:
            import cv2
            import numpy as np
        except Exception:  # pragma: no cover - exercised only when cv2 absent
            return crop

        if crop is None or getattr(crop, "size", 0) == 0:
            return None

        img = crop
        if img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img if img.ndim == 2 else img[..., 0]

        # Otsu binarisation — bumper text is high-contrast (white on red/blue).
        _t, binar = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # If the field is darker than the glyphs we may need the inverse; pick the
        # variant whose ink (black) fraction is the smaller one (text is the minority).
        if (binar == 0).mean() > 0.5:
            binar = cv2.bitwise_not(binar)

        if self.upscale and self.upscale > 1:
            binar = cv2.resize(
                binar, None, fx=self.upscale, fy=self.upscale,
                interpolation=cv2.INTER_CUBIC,
            )
        return binar

    def _crop(self, frame, roi: BBox):
        """Slice the bumper ROI out of the frame (guarded; returns None if numpy/crop bad)."""
        x1, y1, x2, y2 = (int(round(v)) for v in roi)
        try:
            return frame[y1:y2, x1:x2]
        except Exception:  # pragma: no cover - non-array frame
            return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def read(
        self,
        frame,
        robot_xyxy: BBox,
        *,
        ocr_fn: Optional[OcrFn] = None,
    ) -> tuple[Optional[int], float]:
        """Read the bumper team number for one robot box.

        Steps:
          1. Compute the bumper-band ROI (lower part of the box, inset sides).
          2. Reject boxes too small to carry a legible number (min_box_px).
          3. Crop -> threshold -> upscale.
          4. OCR the digits (real easyocr by default; `ocr_fn` lets tests inject a stub
             that receives the preprocessed crop and returns [(text, conf), ...]).
          5. Parse the best plausible team number + confidence.

        Returns (team or None, confidence in [0,1]).

        `ocr_fn` is the seam that makes the WHOLE path testable without easyocr/cv2: pass
        a callable taking the crop and returning OCR items; geometry + parsing are then
        exercised end-to-end on a fake frame.
        """
        # Determine frame size for clipping if the frame looks like an array.
        clip_wh: Optional[tuple[int, int]] = None
        shape = getattr(frame, "shape", None)
        if shape is not None and len(shape) >= 2:
            clip_wh = (int(shape[1]), int(shape[0]))  # (W, H)

        roi = bumper_roi_xyxy(
            robot_xyxy,
            band_top_frac=self.band_top_frac,
            band_bottom_frac=self.band_bottom_frac,
            x_inset_frac=self.x_inset_frac,
            clip_wh=clip_wh,
        )
        if roi is None:
            return None, 0.0

        # Reject robots too small/distant for a legible number.
        if (roi[2] - roi[0]) < self.min_box_px or (roi[3] - roi[1]) < self.min_box_px:
            return None, 0.0

        crop = self._crop(frame, roi)
        if crop is None:
            return None, 0.0

        proc = self._preprocess(crop)
        if proc is None:
            return None, 0.0

        fn = ocr_fn if ocr_fn is not None else self._default_ocr_fn
        items = fn(proc)
        return parse_team_from_ocr(items, min_conf=self.min_conf)
