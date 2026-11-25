"""
results_extractor.py — hardcoded extractor for the FRC broadcast POST-MATCH RESULTS screen.

The results screen is a fixed-layout, large-text, once-per-match full-screen graphic. It is
far easier to read than the live in-match bar and carries everything we need to identify a
match and check it against The Blue Alliance:
    - match label ("Qualification 54 of 64")
    - 3 blue team numbers (left column) + 3 red team numbers (right column)
    - final BLUE / RED scores
    - score breakdown (AUTO/TELEOP fuel & tower, penalty)  [added incrementally]

No model API at inference: detection is pure OpenCV color/brightness signatures; digit
reading is easyocr (a local, offline OCR — no network, no key). ROIs are stored as
fractions of (W, H) so they're resolution-independent (measured on native 640x360).

This is the design-time-vision → hardcoded-algorithm step: a human/vision pass identified
the layout once; everything here runs deterministically thereafter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

# ----- ROIs (x0, y0, x1, y1) as fractions of frame W,H. Measured on the 640x360 results screen.
ROIS = {
    "label":      (0.30, 0.005, 0.70, 0.075),   # "Qualification 54 of 64"
    "blue_score": (0.31, 0.10,  0.50, 0.29),    # big blue score (read full text, not digit-allowlist)
    "red_score":  (0.50, 0.10,  0.69, 0.29),    # big red score
    "blue_teams": (0.040, 0.34, 0.205, 0.625),  # left column, 3 team numbers (rank col excluded)
    "red_teams":  (0.730, 0.34, 0.885, 0.625),  # team numbers sit LEFT of the red panel; rank col is further right
    "breakdown":  (0.30, 0.30,  0.70, 0.63),    # center AUTO/TELEOP table
}

# Detector probe regions
_BLUE_HDR = (0.33, 0.11, 0.49, 0.26)   # blue score background block
_RED_HDR  = (0.51, 0.11, 0.67, 0.26)   # red score background block
_TABLE    = (0.34, 0.33, 0.66, 0.55)   # white breakdown table


def _crop(img, roi):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = roi
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def _color_frac(bgr, which):
    if bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat = (s > 80) & (v > 60)
    if which == "blue":
        m = sat & (h >= 100) & (h <= 130)
    else:
        m = sat & ((h <= 10) | (h >= 170))
    return float(m.mean())


def _bright_lowsat_frac(bgr):
    """Fraction of near-white pixels (the breakdown table background)."""
    if bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(((hsv[..., 1] < 60) & (hsv[..., 2] > 150)).mean())


def is_results_screen(frame, *, blue_th=0.45, red_th=0.45, table_th=0.45) -> bool:
    """
    Post-match results screen signature: a solid blue score block (left of center), a solid
    red score block (right of center), and a bright low-saturation breakdown table below.
    Distinguishes the results screen from the live in-match view (where the same center band
    shows the green field) and from filler.
    """
    return (
        _color_frac(_crop(frame, _BLUE_HDR), "blue") >= blue_th
        and _color_frac(_crop(frame, _RED_HDR), "red") >= red_th
        and _bright_lowsat_frac(_crop(frame, _TABLE)) >= table_th
    )


@dataclass
class MatchResult:
    comp_level: str | None = None      # qm / sf / f
    match_number: int | None = None
    match_label: str | None = None
    blue_teams: list[int] = field(default_factory=list)
    red_teams: list[int] = field(default_factory=list)
    blue_score: int | None = None
    red_score: int | None = None
    raw_label: str | None = None


class ResultsExtractor:
    """Lazily loads one easyocr reader; reuse the instance across frames."""

    def __init__(self, gpu: bool = False):
        import easyocr
        self._reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    # --- OCR helpers -------------------------------------------------------
    def _read_text(self, bgr) -> str:
        bgr = cv2.resize(bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        out = self._reader.readtext(bgr, detail=0)
        return " ".join(out)

    def _read_ints(self, bgr, *, min_v=1, max_v=99999) -> list[int]:
        """Read integers from a region, top-to-bottom by vertical position."""
        bgr = cv2.resize(bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        res = self._reader.readtext(bgr, allowlist="0123456789")
        rows = []
        for bbox, text, conf in res:
            digits = re.sub(r"\D", "", text)
            if not digits:
                continue
            v = int(digits)
            if min_v <= v <= max_v:
                ys = [p[1] for p in bbox]
                rows.append((sum(ys) / len(ys), v))
        rows.sort(key=lambda r: r[0])
        return [v for _, v in rows]

    def _read_score(self, bgr) -> int | None:
        """Read a large score number. NO allowlist — easyocr's full model reads the bold
        stylized digits far better; the 'BLUE'/'RED' word above yields no digits, so the
        first/longest digit group is the score."""
        bgr = cv2.resize(bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        best = None
        for text in self._reader.readtext(bgr, detail=0):
            d = re.sub(r"\D", "", text)
            if d and (best is None or len(d) > len(best)):
                best = d
        return int(best) if best else None

    # --- main --------------------------------------------------------------
    def extract(self, frame) -> MatchResult:
        r = MatchResult()
        label = self._read_text(_crop(frame, ROIS["label"]))
        r.raw_label = label
        r.match_label, r.comp_level, r.match_number = _parse_label(label)
        # FRC team numbers are 1..~12000; rank/score columns are <100 so a wide team ROI
        # would also catch them — ROI is tightened, and we cap teams to plausible range.
        r.blue_teams = self._read_ints(_crop(frame, ROIS["blue_teams"]), min_v=1, max_v=12000)[:3]
        r.red_teams = self._read_ints(_crop(frame, ROIS["red_teams"]), min_v=1, max_v=12000)[:3]
        r.blue_score = self._read_score(_crop(frame, ROIS["blue_score"]))
        r.red_score = self._read_score(_crop(frame, ROIS["red_score"]))
        return r


def _parse_label(text: str):
    """'Qualification 54 of 64' -> ('Qualification 54 of 64', 'qm', 54). Tolerant of OCR noise."""
    if not text:
        return None, None, None
    low = text.lower()
    m = re.search(r"qual\w*\s+(\d+)", low)
    if m:
        n = int(m.group(1))
        return f"Qualification {n}", "qm", n
    m = re.search(r"final\s+(\d+)", low)
    if m:
        return f"Final {int(m.group(1))}", "f", int(m.group(1))
    if "playoff" in low or "bracket" in low or "alliance" in low:
        return text.strip(), "sf", None
    return text.strip(), None, None


if __name__ == "__main__":
    import sys, json
    img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else "frames/roi/native_011500.png")
    print("is_results_screen:", is_results_screen(img))
    ex = ResultsExtractor()
    res = ex.extract(img)
    print(json.dumps(res.__dict__, indent=2))
