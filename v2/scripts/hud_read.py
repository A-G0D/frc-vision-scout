"""Broadcast-calibrated HUD reader (FIRST-AGE / REBUILT 'NE District' overlay, 1920x1080).

Reads the top scoreboard ROIs with easyocr -> OverlayState. ROI fractions were MEASURED
from the actual broadcast (a detail=1 easyocr pass over the top strip located every number:
blue teams ~cx 0.27-0.38 / red teams ~cx 0.62-0.75 / timer ~0.50 / fuel at the edges, all
on the scoreboard row at cy~0.09; match label on the title row at cy~0.035).

Footage-specific BY DESIGN — kept in scripts/ (not the tested src tree) because every
event's overlay layout differs. The reader degrades gracefully (fields stay None when a
ROI doesn't parse); the caller votes team rosters over many frames to beat transient OCR
misreads (e.g. a doubled '6320 6328').
"""
from __future__ import annotations

import re

from frcscout.io.types import OverlayState

# (x0, y0, x1, y1) as fractions of (W, H) — measured on this broadcast.
_ROIS = {
    "match":      (0.43, 0.018, 0.57, 0.055),
    "blue_fuel":  (0.03, 0.070, 0.135, 0.105),
    "blue_teams": (0.232, 0.070, 0.412, 0.105),
    "blue_score": (0.413, 0.068, 0.462, 0.107),
    "timer":      (0.462, 0.070, 0.548, 0.105),
    "red_score":  (0.545, 0.068, 0.598, 0.107),
    "red_teams":  (0.598, 0.070, 0.788, 0.105),
    "red_fuel":   (0.895, 0.070, 0.995, 0.105),
}


def _ints(texts, lo=1, hi=12000):
    out = []
    for t in texts:
        for m in re.findall(r"\d{1,5}", t.replace("O", "0").replace("o", "0")):
            v = int(m)
            if lo <= v <= hi:
                out.append(v)
    return out


class HudReader:
    """easyocr-backed reader for the calibrated HUD ROIs."""

    def __init__(self, gpu: bool = False):
        import easyocr
        self._r = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def _ocr(self, frame, roi):
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = roi
        crop = frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        if crop.size == 0:
            return []
        return self._r.readtext(crop, detail=0)

    def read(self, frame, frame_idx: int = 0) -> OverlayState:
        st = OverlayState()

        m = self._ocr(frame, _ROIS["match"])
        if m:
            st.match_label = " ".join(s.strip() for s in m).strip() or None

        st.blue_teams = list(dict.fromkeys(_ints(self._ocr(frame, _ROIS["blue_teams"]))))[:3]
        st.red_teams = list(dict.fromkeys(_ints(self._ocr(frame, _ROIS["red_teams"]))))[:3]

        bs = _ints(self._ocr(frame, _ROIS["blue_score"]), 0, 1000)
        st.blue_score = bs[0] if bs else None
        rs = _ints(self._ocr(frame, _ROIS["red_score"]), 0, 1000)
        st.red_score = rs[0] if rs else None

        bf = _ints(self._ocr(frame, _ROIS["blue_fuel"]), 0, 100)
        st.blue_fuel = bf[0] if bf else None        # "N /100" -> N (first int)
        rf = _ints(self._ocr(frame, _ROIS["red_fuel"]), 0, 100)
        st.red_fuel = rf[0] if rf else None

        tm = self._ocr(frame, _ROIS["timer"])
        if tm:
            tt = tm[0].replace(".", ":").replace(" ", "")
            mm = re.match(r"(\d+):(\d+)", tt)
            if mm:
                st.timer_s = float(int(mm.group(1)) * 60 + int(mm.group(2)))
            else:
                d = re.findall(r"\d+", tt)
                if d:
                    st.timer_s = float(d[0])
        return st


__all__ = ["HudReader"]
