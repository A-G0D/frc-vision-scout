"""Overlay reader — authoritative match data at 2Hz on the CPU.

The broadcast overlay is GROUND TRUTH for score / fuel count / teams / phase / timer.
The CV layers never need to *count* balls for the score — they read it here. This is
the reframing that lets the expensive ball detector run at 5Hz instead of 60Hz.

Reading is pure classical CV + a local offline OCR (easyocr) — NO inference API at
runtime. ROIs live in fractions of (W,H) so they are resolution-independent.

Per-layout ROI maps fix the v1 playoff bug: qual-layout ROIs were applied verbatim to
the playoff results screen, reading the rank column (1-8) as team numbers. The fix:
1. Classify which layout is on screen first.
2. Crop from THAT layout's ROI map.
3. Apply plausibility caps (team 100..12000; reject <100 as rank-column artefacts).
4. Enforce structural validity (3 teams per alliance, no team on both alliances).

Live-bar reading also extracts the fuel counter (X/100), timer, and phase.
"""
from __future__ import annotations

import re
from typing import Optional

import cv2
import numpy as np

from frcscout.io.types import OverlayState

# ---------------------------------------------------------------------------
# ROI maps — (x0, y0, x1, y1) as fractions of (W, H)
# Measured on 640×360 native broadcast resolution.
# ---------------------------------------------------------------------------

# ── Layout: live scoreboard bar (in-match) ──────────────────────────────────
_LIVE_BAR_ROIS: dict[str, tuple[float, float, float, float]] = {
    # Alliance team numbers shown in the left/right top-bar banners.
    "blue_teams":  (0.00, 0.00, 0.15, 0.08),
    "red_teams":   (0.85, 0.00, 1.00, 0.08),
    # Score digits in the scoreboard bar.
    "blue_score":  (0.15, 0.00, 0.40, 0.08),
    "red_score":   (0.60, 0.00, 0.85, 0.08),
    # Fuel counters (X/100 shown near each score).
    "blue_fuel":   (0.15, 0.00, 0.35, 0.08),
    "red_fuel":    (0.65, 0.00, 0.85, 0.08),
    # Timer sits in the centre of the bar.
    "timer":       (0.43, 0.00, 0.57, 0.08),
    # Phase label ("AUTO" / "TELEOP" / "ENDGAME") also centred.
    "phase":       (0.35, 0.00, 0.65, 0.10),
    # Probe regions to detect the live bar.
    "_probe_blue": (0.00, 0.00, 0.18, 0.09),
    "_probe_red":  (0.82, 0.00, 1.00, 0.09),
}

# ── Layout: qualification results screen (post-match) ──────────────────────
_QUAL_RESULTS_ROIS: dict[str, tuple[float, float, float, float]] = {
    "label":       (0.30, 0.005, 0.70, 0.075),
    "blue_score":  (0.31, 0.10,  0.50, 0.29),
    "red_score":   (0.50, 0.10,  0.69, 0.29),
    # Team-number columns: tightened to exclude the rank column (far left/right).
    "blue_teams":  (0.040, 0.34, 0.205, 0.625),
    "red_teams":   (0.730, 0.34, 0.885, 0.625),
    # Probe regions for layout detection.
    "_probe_blue": (0.33, 0.11, 0.49, 0.26),
    "_probe_red":  (0.51, 0.11, 0.67, 0.26),
    "_probe_table":(0.34, 0.33, 0.66, 0.55),
}

# ── Layout: playoff / alliance bracket results screen ──────────────────────
# The playoff screen has the same team-number column positions BUT:
#   • No ranking column (rank col disappears in playoffs) → min_v=100 guard is enough.
#   • Score blocks are in the same place as qual.
#   • Layout probe: colour blocks present but NO bright white breakdown table.
_PLAYOFF_RESULTS_ROIS: dict[str, tuple[float, float, float, float]] = {
    "label":       (0.30, 0.005, 0.70, 0.075),
    "blue_score":  (0.31, 0.10,  0.50, 0.29),
    "red_score":   (0.50, 0.10,  0.69, 0.29),
    # Playoff team columns are slightly wider (3 teams per alliance, no rank col).
    "blue_teams":  (0.030, 0.34, 0.220, 0.640),
    "red_teams":   (0.720, 0.34, 0.900, 0.640),
    "_probe_blue": (0.33, 0.11, 0.49, 0.26),
    "_probe_red":  (0.51, 0.11, 0.67, 0.26),
    "_probe_table":(0.34, 0.33, 0.66, 0.55),
}

# Team number plausibility.
_TEAM_MIN = 100    # reject rank-column reads (ranks are 1-99 at most).
_TEAM_MAX = 12000  # FRC numbers don't exceed ~12 000.

# Phase keyword map for normalisation.
_PHASE_KEYWORDS = {
    "auto": "auto",
    "teleop": "teleop",
    "tele": "teleop",
    "end": "endgame",
    "endgame": "endgame",
}


# ---------------------------------------------------------------------------
# Low-level image helpers
# ---------------------------------------------------------------------------

def _crop(img: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = roi
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def _color_frac(bgr: np.ndarray, which: str) -> float:
    """Fraction of blue or red saturated pixels."""
    if bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat = (s > 80) & (v > 60)
    if which == "blue":
        mask = sat & (h >= 100) & (h <= 130)
    else:  # red
        mask = sat & ((h <= 10) | (h >= 170))
    return float(mask.mean())


def _bright_lowsat_frac(bgr: np.ndarray) -> float:
    """Fraction of near-white pixels (breakdown table background)."""
    if bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(((hsv[..., 1] < 60) & (hsv[..., 2] > 150)).mean())


# ---------------------------------------------------------------------------
# Layout classifier
# ---------------------------------------------------------------------------

def _classify_layout(frame: np.ndarray) -> str:
    """Return one of "live_bar", "qual_results", "playoff_results", or "unknown"."""
    # --- Qual / playoff results: large solid blue + red score blocks in centre.
    blue_f = _color_frac(_crop(frame, _QUAL_RESULTS_ROIS["_probe_blue"]), "blue")
    red_f  = _color_frac(_crop(frame, _QUAL_RESULTS_ROIS["_probe_red"]),  "red")
    table_f = _bright_lowsat_frac(_crop(frame, _QUAL_RESULTS_ROIS["_probe_table"]))

    if blue_f >= 0.40 and red_f >= 0.40:
        if table_f >= 0.35:
            return "qual_results"
        # Large colour blocks without a white table → playoff bracket screen.
        return "playoff_results"

    # --- Live bar: blue top-left banner + red top-right banner (small fractions).
    blue_bar = _color_frac(_crop(frame, _LIVE_BAR_ROIS["_probe_blue"]), "blue")
    red_bar  = _color_frac(_crop(frame, _LIVE_BAR_ROIS["_probe_red"]),  "red")
    if blue_bar >= 0.08 and red_bar >= 0.08:
        return "live_bar"

    return "unknown"


# ---------------------------------------------------------------------------
# Structural validity
# ---------------------------------------------------------------------------

def _validate_teams(
    blue: list[int],
    red: list[int],
    comp_level: Optional[str],
) -> tuple[list[int], list[int], list[str]]:
    """
    Apply plausibility caps and structural rules.

    Rules:
    • team numbers must be in [_TEAM_MIN, _TEAM_MAX] (already filtered upstream).
    • For qm / f there must be exactly 3 teams per alliance; sf: 1-3 acceptable.
    • No team number may appear on both alliances.

    Returns (blue_out, red_out, errors).
    """
    errors: list[str] = []

    # Remove cross-alliance duplicates (keep first occurrence).
    blue_set = set(blue)
    red_clean = [t for t in red if t not in blue_set]
    if len(red_clean) != len(red):
        dropped = [t for t in red if t in blue_set]
        errors.append(f"team_on_both_alliances: {dropped}")
        red = red_clean

    # Alliance size check.
    if comp_level in ("qm", "f"):
        if len(blue) != 3:
            errors.append(f"blue_alliance_size={len(blue)}_expected_3")
        if len(red) != 3:
            errors.append(f"red_alliance_size={len(red)}_expected_3")

    return blue, red, errors


# ---------------------------------------------------------------------------
# Label parser (ported from v1 results_extractor)
# ---------------------------------------------------------------------------

def _parse_label(text: str) -> tuple[Optional[str], Optional[str], Optional[int]]:
    """'Qualification 54 of 64' → ('Qualification 54 of 64', 'qm', 54)."""
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


# ---------------------------------------------------------------------------
# Phase normaliser
# ---------------------------------------------------------------------------

def _parse_phase(text: str) -> Optional[str]:
    low = text.lower()
    for key, val in _PHASE_KEYWORDS.items():
        if key in low:
            return val
    return None


# ---------------------------------------------------------------------------
# Timer parser
# ---------------------------------------------------------------------------

def _parse_timer(text: str) -> Optional[float]:
    """'2:05' → 125.0; '45' → 45.0; returns None on failure."""
    text = text.strip()
    m = re.search(r"(\d+):(\d{2})", text)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    m = re.search(r"(\d{1,3})", text)
    if m:
        v = int(m.group(1))
        if 0 <= v <= 180:
            return float(v)
    return None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OverlayReader:
    """
    Lazily loads one easyocr reader; reuse the instance across frames.

    Layout detection happens first, then layout-specific ROI crops are OCR'd.
    The v1 playoff bug (rank column read as team numbers because qual ROIs were
    used on the playoff screen) is fixed by keeping separate ROI maps and
    selecting the correct one after _classify_layout().
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self._ocr = None

    # ------------------------------------------------------------------
    # Lazy OCR loader
    # ------------------------------------------------------------------

    @property
    def _template(self) -> bool:
        # cfg {"ocr": "template"}: CPU glyph matcher for score/timer, no easyocr.
        return self.cfg.get("ocr") == "template"

    def _ensure(self):
        if self._ocr is None and not self._template:
            import easyocr  # guarded — not needed until a frame arrives
            self._ocr = easyocr.Reader(["en"], gpu=False, verbose=False)

    # ------------------------------------------------------------------
    # OCR helpers
    # ------------------------------------------------------------------

    def _read_text(self, bgr: np.ndarray) -> str:
        if bgr.size == 0:
            return ""
        bgr = cv2.resize(bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        out = self._ocr.readtext(bgr, detail=0)
        return " ".join(out)

    def _read_ints(
        self,
        bgr: np.ndarray,
        *,
        min_v: int = _TEAM_MIN,
        max_v: int = _TEAM_MAX,
    ) -> list[int]:
        """Read integers top-to-bottom; filter to [min_v, max_v]."""
        if bgr.size == 0:
            return []
        bgr = cv2.resize(bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        res = self._ocr.readtext(bgr, allowlist="0123456789")
        rows: list[tuple[float, int]] = []
        for bbox, text, _conf in res:
            digits = re.sub(r"\D", "", text)
            if not digits:
                continue
            v = int(digits)
            if min_v <= v <= max_v:
                ys = [p[1] for p in bbox]
                rows.append((sum(ys) / len(ys), v))
        rows.sort(key=lambda r: r[0])
        return [v for _, v in rows]

    def _read_score(self, bgr: np.ndarray) -> Optional[int]:
        """Read a large stylised score digit block (no allowlist — full model)."""
        if bgr.size == 0:
            return None
        if self._template:
            from .digit_ocr import read_int
            return read_int(bgr)
        bgr = cv2.resize(bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        best: Optional[str] = None
        for text in self._ocr.readtext(bgr, detail=0):
            d = re.sub(r"\D", "", text)
            if d and (best is None or len(d) > len(best)):
                best = d
        return int(best) if best else None

    def _read_fuel(self, bgr: np.ndarray) -> Optional[int]:
        """Read 'X/100' fuel counter; return X capped at 100."""
        text = self._read_text(bgr)
        m = re.search(r"(\d{1,3})\s*/\s*100", text)
        if m:
            return min(int(m.group(1)), 100)
        # Fallback: single integer in sensible range.
        m = re.search(r"\b(\d{1,3})\b", text)
        if m:
            v = int(m.group(1))
            if 0 <= v <= 100:
                return v
        return None

    # ------------------------------------------------------------------
    # Per-layout readers
    # ------------------------------------------------------------------

    def _read_live_bar(self, frame: np.ndarray, frame_idx: int) -> OverlayState:
        rois = _LIVE_BAR_ROIS
        s = OverlayState()
        s.blue_score = self._read_score(_crop(frame, rois["blue_score"]))
        s.red_score  = self._read_score(_crop(frame, rois["red_score"]))
        if self._template:  # fuel, phase and teams need easyocr; left unset
            from .digit_ocr import read_timer
            s.timer_s = read_timer(_crop(frame, rois["timer"]))
            return s
        s.blue_fuel  = self._read_fuel(_crop(frame, rois["blue_fuel"]))
        s.red_fuel   = self._read_fuel(_crop(frame, rois["red_fuel"]))
        timer_text   = self._read_text(_crop(frame, rois["timer"]))
        s.timer_s    = _parse_timer(timer_text)
        phase_text   = self._read_text(_crop(frame, rois["phase"]))
        s.phase      = _parse_phase(phase_text)
        # Team numbers in the live bar are small; read with tighter caps.
        blue_raw = self._read_ints(_crop(frame, rois["blue_teams"]),
                                   min_v=_TEAM_MIN, max_v=_TEAM_MAX)
        red_raw  = self._read_ints(_crop(frame, rois["red_teams"]),
                                   min_v=_TEAM_MIN, max_v=_TEAM_MAX)

        blue, red, _errs = _validate_teams(blue_raw[:3], red_raw[:3], None)
        s.blue_teams = blue
        s.red_teams  = red
        return s

    def _read_results_screen(
        self,
        frame: np.ndarray,
        frame_idx: int,
        rois: dict,
    ) -> OverlayState:
        s = OverlayState()
        label_text = self._read_text(_crop(frame, rois["label"]))
        s.match_label, s.comp_level, s.match_number = _parse_label(label_text)
        s.blue_score = self._read_score(_crop(frame, rois["blue_score"]))
        s.red_score  = self._read_score(_crop(frame, rois["red_score"]))
        blue_raw = self._read_ints(_crop(frame, rois["blue_teams"]),
                                   min_v=_TEAM_MIN, max_v=_TEAM_MAX)[:3]
        red_raw  = self._read_ints(_crop(frame, rois["red_teams"]),
                                   min_v=_TEAM_MIN, max_v=_TEAM_MAX)[:3]
        blue, red, _errs = _validate_teams(blue_raw, red_raw, s.comp_level)
        s.blue_teams = blue
        s.red_teams  = red
        return s

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def read(self, frame: np.ndarray, frame_idx: int) -> OverlayState:
        """
        Return the current overlay state.

        Steps:
            1. Classify which overlay layout is on screen.
            2. Crop the ROIs for THAT layout.
            3. OCR digits (teams capped to plausible FRC range; reject <100).
            4. Enforce structural validity (3v3, no cross-alliance duplicates).
            5. Read live X/100 fuel counter, timer, phase (live bar only).
        """
        self._ensure()
        layout = _classify_layout(frame)

        if layout == "live_bar":
            return self._read_live_bar(frame, frame_idx)
        elif layout == "qual_results":
            return self._read_results_screen(frame, frame_idx, _QUAL_RESULTS_ROIS)
        elif layout == "playoff_results":
            return self._read_results_screen(frame, frame_idx, _PLAYOFF_RESULTS_ROIS)
        else:
            return OverlayState()


# ---------------------------------------------------------------------------
# Module-level helpers exposed for tests (pure logic, no OCR dependency)
# ---------------------------------------------------------------------------

def validate_teams(
    blue: list[int],
    red: list[int],
    comp_level: Optional[str] = None,
) -> tuple[list[int], list[int], list[str]]:
    """Public wrapper around _validate_teams for use in tests and eval."""
    return _validate_teams(blue, red, comp_level)
