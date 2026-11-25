"""
overlay_live.py — read the LIVE in-match scoreboard bar of the FRC REBUILT (2026)
broadcast (NE Waterbury stream), distinct from the post-match results screen handled
by results_extractor.py.

Observed live bar layout (measured on native 640x360 frames; fractions of W,H). The bar
sits in a thin strip across the top of the frame, roughly y in [0.07, 0.13]:

  +--------------------------------------------------------------------------------+
  |        FIRST/AGE logo |  "Qualification 52 of 64"  |  REBUILT logo             |  (dark label row, y~0.0-0.07)
  | [BLUE  | blue team | BIG  | TIMER | BIG | red team |  [RED                     |
  |  fuel] | 3 numbers |BLUE  | 0:28  |RED  | 3 numbers |  fuel]                    |  (color bar row, y~0.07-0.13)
  | 285/360|5856 228 176| 295 | 0:28  | 117 |9193 7694 |  117/360                  |
  +--------------------------------------------------------------------------------+

Reading strategy
----------------
* The two BIG score numbers are bold/stylized — read with easyocr's full model and NO
  digit allowlist (allowlist mangles the bold glyphs; see results_extractor._read_score).
* Timer "M:SS" and the "N / NNN" fuel boxes are plain text; read with full model and
  parse with regex.
* All ROIs are fractions of (W,H) so this is resolution independent.

Public API: read_bar(frame) -> dict(blue_score, red_score, timer, blue_fuel, red_fuel)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

# ---- ROIs (x0, y0, x1, y1) as fractions of frame W,H. Measured on 640x360 live bar.
# The color bar row sits at y~0.07..0.13; give a little vertical margin.
ROIS = {
    "blue_fuel":  (0.053, 0.075, 0.130, 0.135),  # box, far left (cube icon trimmed): "285 / 360"
    "blue_score": (0.372, 0.070, 0.470, 0.140),  # big blue number, right of blue panel
    "timer":      (0.470, 0.075, 0.535, 0.135),  # white center box: "0:28"
    "red_score":  (0.532, 0.070, 0.628, 0.140),  # big red number, left of red panel
    "red_fuel":   (0.900, 0.075, 0.995, 0.135),  # box, far right (cube icon trimmed): "117 / 360"
}

# Detector probe regions for is_live_bar() — blue block left of center, red block right.
_BLUE_BLK = (0.33, 0.085, 0.46, 0.125)
_RED_BLK  = (0.54, 0.085, 0.66, 0.125)


def _crop(img: np.ndarray, roi):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = roi
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def _color_frac(bgr, which):
    if bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat = (s > 90) & (v > 60)
    if which == "blue":
        m = sat & (h >= 100) & (h <= 130)
    else:
        m = sat & ((h <= 10) | (h >= 170))
    return float(m.mean())


def is_live_bar(frame, *, blue_th=0.4, red_th=0.4) -> bool:
    """The live bar shows a saturated blue score block left of center and red right of
    center. (The results screen has its blocks lower; filler has neither.)"""
    return (
        _color_frac(_crop(frame, _BLUE_BLK), "blue") >= blue_th
        and _color_frac(_crop(frame, _RED_BLK), "red") >= red_th
    )


@dataclass
class LiveBar:
    blue_score: int | None = None
    red_score: int | None = None
    timer: str | None = None
    blue_fuel: int | None = None
    red_fuel: int | None = None


class LiveBarReader:
    """Holds one easyocr reader; reuse across frames (construction is the slow part)."""

    def __init__(self, gpu: bool = False):
        import easyocr
        self._reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    # -- low-level OCR -----------------------------------------------------
    def _texts(self, bgr, scale=4):
        """Upscale (bold glyphs read better large) and return raw OCR strings."""
        if bgr.size == 0:
            return []
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return self._reader.readtext(bgr, detail=0)

    def _read_score(self, bgr, scale=5) -> int | None:
        """Big bold score. A small neighbouring team number can leak into the ROI, so we
        pick the digit group whose glyphs are TALLEST (the score is the large bold text),
        breaking ties by digit-group length."""
        if bgr.size == 0:
            return None
        big = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        best_v, best_key = None, (-1.0, -1)
        for bbox, text, _conf in self._reader.readtext(big):
            d = re.sub(r"\D", "", text)
            if not d:
                continue
            v = int(d)
            if not (0 <= v <= 9999):
                continue
            ys = [p[1] for p in bbox]
            height = max(ys) - min(ys)
            key = (height, len(d))
            if key > best_key:
                best_key, best_v = key, v
        return best_v

    def _read_timer(self, bgr) -> str | None:
        """Center timer 'M:SS'. OCR may drop the colon, so reconstruct from digits."""
        joined = " ".join(self._texts(bgr))
        m = re.search(r"(\d)\s*[:;.]\s*(\d{2})", joined)
        if m:
            return f"{m.group(1)}:{m.group(2)}"
        d = re.sub(r"\D", "", joined)
        if len(d) == 3:                       # e.g. "028" -> 0:28
            return f"{d[0]}:{d[1:]}"
        if len(d) == 2:                       # e.g. "28"  -> 0:28
            return f"0:{d}"
        if len(d) == 4:                       # e.g. "1 28" -> 1:28
            return f"{d[:2]}:{d[2:]}"
        return None

    @staticmethod
    def _norm_cap(s: str) -> int | None:
        """OCR of the denominator is noisy ('1c0','i00','10o','1OO'). It is always the
        fuel cap, which is 100 or 360. Map by leading digit / length."""
        digits = re.sub(r"\D", "", s)
        if s.startswith("3") or digits.startswith("3"):
            return 360
        if len(s) >= 2:                          # 'xx0' style -> 100
            return 100
        return None

    def _fuel_candidate(self, joined: str):
        """Parse one OCR string into (numerator, cap|None, has_slash). The fuel box reads
        'N / CAP' where CAP is always 100 or 360. The '/' may OCR as / | I l, be dropped to a
        space, or be glued into a digit ('22/100' -> '221100'); the left cube icon can also
        bleed a spurious leading digit. has_slash flags the clean explicit-slash form."""
        m = re.search(r"(\d{1,4})\s*([/|Il])\s*([\dILiOoCc]{2,4})", joined)
        has_slash = m is not None
        if m:
            num_s, cap = m.group(1), self._norm_cap(m.group(3))
        else:
            m = re.search(r"(\d{1,4})\s+([\dILiOoCc]{2,4})", joined)  # slash dropped to space
            if m:
                num_s, cap = m.group(1), self._norm_cap(m.group(2))
            else:
                # Everything glued together, e.g. "221100" (= 22 / 100) or "92100" (= 92 / 100).
                d = re.sub(r"\D", "", joined)
                cap = None
                num_s = d
                for c, cv in (("360", 360), ("100", 100)):
                    if d.endswith(c) and len(d) > 3:
                        cap, num_s = cv, d[:-3]
                        break
                if cap is None:
                    return (int(d) if d else None), None, False
                # Here the slash glues onto the RIGHT of the count ("22"+"1"+"100"); if the
                # remainder exceeds the cap, the trailing char is the slash — drop it.
                while num_s and int(num_s) > cap and len(num_s) > 1:
                    num_s = num_s[:-1]
                return (int(num_s) if num_s else None), cap, has_slash
        # slash / space branches: the numerator is at most 3 digits. A 4-digit reading means
        # the left cube icon bled a spurious leading digit (e.g. "7146" -> "146") — drop it.
        # NOTE we do NOT clamp to the cap: the far-right red box numerator mirrors the red
        # score, which legitimately exceeds 360 (e.g. "519 / 360").
        if not num_s:
            return None, cap, has_slash
        if len(num_s) == 4:
            num_s = num_s[1:]
        # Space branch with no real slash: '/' was often read as a trailing '1' glued onto the
        # count ("22/100" -> "221 100"). If the count exceeds its cap and ends in '1', that '1'
        # is the slash — drop it. (The explicit-slash branch never reaches here, so a true
        # over-cap value like "519 / 360" is unaffected.)
        if not has_slash and cap and num_s.endswith("1") and int(num_s) > cap:
            num_s = num_s[:-1]
        n = int(num_s)
        return (n if n <= 999 else None), cap, has_slash

    @staticmethod
    def _sharp_gray(bgr, scale=8):
        """Fuel digits are tiny white-on-color text. Grayscale + Lanczos upscale + an unsharp
        mask makes the thin strokes crisp; this reads the small 'N / NNN' far more reliably
        than colour-image cubic upscaling (single pass — no multi-scale voting needed)."""
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        return cv2.addWeighted(g, 1.8, cv2.GaussianBlur(g, (0, 0), 3), -0.8, 0)

    def _read_fuel(self, bgr) -> int | None:
        """Fuel box 'N / NNN' — return the numerator (scored fuel). One sharpened pass."""
        if bgr.size == 0:
            return None
        joined = " ".join(self._reader.readtext(self._sharp_gray(bgr), detail=0))
        num, _cap, _has_slash = self._fuel_candidate(joined)
        return num if (num is not None and 0 <= num <= 999) else None

    # -- main --------------------------------------------------------------
    def read(self, frame) -> LiveBar:
        return LiveBar(
            blue_score=self._read_score(_crop(frame, ROIS["blue_score"])),
            red_score=self._read_score(_crop(frame, ROIS["red_score"])),
            timer=self._read_timer(_crop(frame, ROIS["timer"])),
            blue_fuel=self._read_fuel(_crop(frame, ROIS["blue_fuel"])),
            red_fuel=self._read_fuel(_crop(frame, ROIS["red_fuel"])),
        )


# Module-level convenience: lazily build a single shared reader.
_READER: LiveBarReader | None = None


def read_bar(frame) -> dict:
    """Read the live scoreboard bar. Returns
    {blue_score, red_score, timer, blue_fuel, red_fuel}."""
    global _READER
    if _READER is None:
        _READER = LiveBarReader()
    r = _READER.read(frame)
    return {
        "blue_score": r.blue_score,
        "red_score": r.red_score,
        "timer": r.timer,
        "blue_fuel": r.blue_fuel,
        "red_fuel": r.red_fuel,
    }


def annotate(frame, result: dict) -> np.ndarray:
    """Draw the ROI boxes and the read values for visual verification."""
    out = frame.copy()
    h, w = out.shape[:2]
    colors = {
        "blue_fuel": (255, 100, 0), "blue_score": (255, 0, 0), "timer": (0, 0, 0),
        "red_score": (0, 0, 255), "red_fuel": (0, 80, 255),
    }
    for name, roi in ROIS.items():
        x0, y0, x1, y1 = roi
        p0 = (int(x0 * w), int(y0 * h))
        p1 = (int(x1 * w), int(y1 * h))
        cv2.rectangle(out, p0, p1, colors[name], 1)
    txt = (f"B:{result['blue_score']} R:{result['red_score']} T:{result['timer']} "
           f"bf:{result['blue_fuel']} rf:{result['red_fuel']}")
    cv2.putText(out, txt, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, txt, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 0), 1, cv2.LINE_AA)
    return out


if __name__ == "__main__":
    import sys, glob, os, time

    paths = sys.argv[1:] or sorted(glob.glob("frames/dataset/live_*.png"))
    out_dir = "frames/dataset/overlaylive_annot"
    os.makedirs(out_dir, exist_ok=True)

    reader = LiveBarReader()
    times = []
    for p in paths:
        frame = cv2.imread(p)
        if frame is None:
            print(f"SKIP {p}")
            continue
        t0 = time.perf_counter()
        r = reader.read(frame)
        ms = (time.perf_counter() - t0) * 1000
        times.append(ms)
        res = {"blue_score": r.blue_score, "red_score": r.red_score, "timer": r.timer,
               "blue_fuel": r.blue_fuel, "red_fuel": r.red_fuel}
        ann = annotate(frame, res)
        name = os.path.basename(p).replace(".png", "_bar.png")
        cv2.imwrite(os.path.join(out_dir, name), ann)
        print(f"{os.path.basename(p):28s} B={r.blue_score} R={r.red_score} "
              f"T={r.timer} bf={r.blue_fuel} rf={r.red_fuel}  {ms:.1f}ms")
    if times:
        print(f"\nMean {sum(times)/len(times):.1f} ms/frame (excl. reader init)  "
              f"max {max(times):.1f} ms")
