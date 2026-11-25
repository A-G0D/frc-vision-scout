"""
frame_classify.py — Fast 3-class frame classifier for FRC broadcast footage.

Classes:
    'live'   - Active match field view (green overlay in pipeline)
    'info'   - Post-match results / info screens (yellow overlay)
    'filler' - Crowd, sponsors, non-game content (red overlay)

Strategy (no OCR, no ML, pure classical CV):
  1. results_extractor.is_results_screen() -> strong indicator for 'info'
  2. overlay_detect.detect_overlay() -> strong indicator for 'live'
  3. Supplementary signals to break edge cases:
     - Top-bar darkness: filler frames often lack the top overlay bar entirely
     - Yellow ball content: live frames have yellow fuel balls on grey carpet
     - Grey carpet texture: live frames show the grey-green field
     - Center brightness: info screens are bright white in the breakdown table area

All signals are cheap numpy/HSV operations. Target < 15 ms/frame at 360p.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

# Import existing detectors
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from overlay_detect import detect_overlay, crop_norm
from results_extractor import is_results_screen, _crop, _bright_lowsat_frac

# --------------------------------------------------------------------------- #
# Supplementary signal regions (fractions of W, H)
# --------------------------------------------------------------------------- #

# Top bar region — should be dark/colored in live, bright in info, absent in filler
_TOP_BAR    = (0.00, 0.00, 1.00, 0.10)

# Center field — live frames show grey carpet with yellow balls here
_FIELD_MID  = (0.20, 0.20, 0.80, 0.85)

# Info screen breakdown table (center white table — reuse results_extractor region)
_INFO_TABLE = (0.34, 0.33, 0.66, 0.55)

# Caption strip bottom — live frames always have this "NE District..." caption
_CAPTION    = (0.00, 0.92, 1.00, 1.00)

# Big color score blocks: live=blue score box left, info has large blue/red center blocks
_BLUE_SCORE_INFO = (0.31, 0.10, 0.50, 0.29)   # info screen blue score block
_RED_SCORE_INFO  = (0.50, 0.10, 0.69, 0.29)   # info screen red score block


def _hsv(bgr: np.ndarray):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def _yellow_frac(bgr: np.ndarray) -> float:
    """Fraction of strongly yellow pixels (FUEL balls)."""
    if bgr.size == 0:
        return 0.0
    h, s, v = _hsv(bgr)[..., 0], _hsv(bgr)[..., 1], _hsv(bgr)[..., 2]
    mask = (s > 80) & (v > 100) & (h >= 18) & (h <= 38)
    return float(mask.mean())


def _grey_frac(bgr: np.ndarray) -> float:
    """Fraction of grey/neutral pixels (carpet color)."""
    if bgr.size == 0:
        return 0.0
    hsv = _hsv(bgr)
    # Low saturation, mid brightness = grey carpet
    mask = (hsv[..., 1] < 55) & (hsv[..., 2] > 60) & (hsv[..., 2] < 210)
    return float(mask.mean())


def _blue_frac(bgr: np.ndarray) -> float:
    if bgr.size == 0:
        return 0.0
    hsv = _hsv(bgr)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return float(((s > 80) & (v > 60) & (h >= 100) & (h <= 130)).mean())


def _red_frac(bgr: np.ndarray) -> float:
    if bgr.size == 0:
        return 0.0
    hsv = _hsv(bgr)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return float(((s > 80) & (v > 60) & ((h <= 10) | (h >= 170))).mean())


@dataclass
class ClassifyResult:
    label: Literal["live", "info", "filler"]
    confidence: float
    signals: dict


def classify(frame: np.ndarray) -> ClassifyResult:
    """
    Classify a single frame as 'live', 'info', or 'filler'.

    Returns ClassifyResult with .label and .confidence (0-1).
    """
    # --- Signal 1: results_extractor overlay signature (large blue+red+white blocks)
    info_sig = is_results_screen(frame)

    # --- Signal 2: overlay_detect (blue-left / red-right scoreboard bar)
    ov = detect_overlay(frame, thresh=0.08)   # slightly lower thresh for sensitivity
    live_bar_score = ov.score                  # min(blue_left, red_right)

    # --- Signal 3: info screen has LARGE solid blue + red blocks in center
    blue_info = _blue_frac(_crop(frame, _BLUE_SCORE_INFO))
    red_info  = _red_frac(_crop(frame, _RED_SCORE_INFO))
    info_color_sig = (blue_info > 0.40 and red_info > 0.40)

    # --- Signal 4: yellow fuel balls in field area (live only)
    field_crop = _crop(frame, _FIELD_MID)
    yellow_f = _yellow_frac(field_crop)

    # --- Signal 5: grey carpet fraction (live field is grey)
    grey_f = _grey_frac(field_crop)

    # --- Signal 6: bright white breakdown table (info only)
    table_bright = _bright_lowsat_frac(_crop(frame, _INFO_TABLE))

    # --- Signal 7: top bar saturation (live has a colorful bar; filler often plain)
    top_crop = _crop(frame, _TOP_BAR)
    top_hsv = _hsv(top_crop)
    top_sat_mean = float(top_hsv[..., 1].mean())  # high = colored bar present

    # --- Signal 8: overall frame brightness / saturation for info screens
    # Info screens are very bright overall
    frame_v = float(_hsv(frame)[..., 2].mean())

    signals = {
        "info_sig": info_sig,
        "live_bar_score": round(live_bar_score, 3),
        "blue_left": round(ov.blue_left, 3),
        "red_right": round(ov.red_right, 3),
        "blue_info": round(blue_info, 3),
        "red_info": round(red_info, 3),
        "info_color_sig": info_color_sig,
        "yellow_f": round(yellow_f, 3),
        "grey_f": round(grey_f, 3),
        "table_bright": round(table_bright, 3),
        "top_sat_mean": round(top_sat_mean, 1),
        "frame_v": round(frame_v, 1),
    }

    # -----------------------------------------------------------------------
    # Decision tree: priority order info > live > filler
    # -----------------------------------------------------------------------

    # INFO: results screen detector is the strongest signal.
    # Also catch if large blue+red center blocks are present.
    # info_sig uses strict thresholds (0.45 each), very reliable.
    if info_sig or info_color_sig:
        # Verify it's not a live frame with incidentally colored regions
        # Live frames won't have bright white table
        if table_bright > 0.15 or info_sig:
            conf = min(1.0, 0.5 + 0.5 * table_bright + (0.3 if info_sig else 0.0))
            return ClassifyResult("info", round(min(conf, 1.0), 3), signals)

    # LIVE: scoreboard bar present (blue-left + red-right signature)
    # The bar score threshold of 0.08 is sensitive; also check yellow/grey field content
    if ov.is_match or live_bar_score > 0.07:
        # Cross-check: live frames also tend to have grey carpet or yellow balls
        field_evidence = yellow_f > 0.005 or grey_f > 0.10 or top_sat_mean > 35
        if field_evidence or live_bar_score > 0.12:
            conf = min(1.0, 0.4 + 2.0 * live_bar_score + 0.2 * (1 if field_evidence else 0))
            return ClassifyResult("live", round(min(conf, 1.0), 3), signals)

    # FILLER: everything else
    # But be careful: filler_01 shows the field from a wide angle without the top overlay
    # We add a soft check: if it has NO bar signature AND no info signature -> filler
    conf = max(0.5, 1.0 - live_bar_score * 5.0 - table_bright)
    return ClassifyResult("filler", round(min(conf, 1.0), 3), signals)


def classify_file(path: str) -> ClassifyResult:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return classify(img)


# --------------------------------------------------------------------------- #
# CLI runner + annotated output
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import glob, pathlib

    dataset_dir = pathlib.Path("frames/dataset")
    out_dir = dataset_dir / "frameclassify_annot"
    out_dir.mkdir(exist_ok=True)

    COLOR_MAP = {
        "live":   (0, 200, 0),    # GREEN
        "info":   (0, 200, 255),  # YELLOW
        "filler": (0, 0, 200),    # RED
    }

    frames = sorted(dataset_dir.glob("live_*.png")) + \
             sorted(dataset_dir.glob("info_*.png")) + \
             sorted(dataset_dir.glob("filler_*.png"))

    total_ms = 0.0
    correct = 0
    total = len(frames)
    results = []

    for path in frames:
        img = cv2.imread(str(path))
        t0 = time.perf_counter()
        res = classify(img)
        elapsed = (time.perf_counter() - t0) * 1000
        total_ms += elapsed

        expected = path.stem.split("_")[0]
        ok = res.label == expected
        if ok:
            correct += 1

        status = "OK" if ok else f"WRONG (expected {expected})"
        print(f"{path.name:40s} -> {res.label:6s} conf={res.confidence:.3f}  {status}  [{elapsed:.1f} ms]")

        # Annotate
        ann = img.copy()
        color = COLOR_MAP[res.label]
        # Thick colored border
        h, w = ann.shape[:2]
        thick = 8
        cv2.rectangle(ann, (thick//2, thick//2), (w - thick//2, h - thick//2), color, thick)
        # Label text
        label_str = f"{res.label.upper()}  conf={res.confidence:.2f}"
        if not ok:
            label_str += f"  [EXPECTED: {expected}]"
        cv2.putText(ann, label_str, (12, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # Signal values on top
        sig_text = (f"bar={res.signals['live_bar_score']:.3f} "
                    f"tbl={res.signals['table_bright']:.3f} "
                    f"yel={res.signals['yellow_f']:.3f} "
                    f"sat={res.signals['top_sat_mean']:.0f}")
        cv2.putText(ann, sig_text, (12, h - 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        out_path = out_dir / path.name
        cv2.imwrite(str(out_path), ann)
        results.append((path.name, expected, res.label, ok, elapsed))

    avg_ms = total_ms / total if total else 0
    print(f"\nAccuracy: {correct}/{total} = {100*correct/total:.1f}%  avg {avg_ms:.1f} ms/frame")
