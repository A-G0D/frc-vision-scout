"""Dependency-free digit reader for broadcast score/timer boxes (numpy + cv2 only).

Binarise (Otsu) -> connected components -> match each glyph against templates rendered
with cv2 Hershey fonts. Meant for large bright digits on a dark bar; it is NOT a general
OCR. shortcut: templates come from Hershey fonts, not from real broadcast glyphs; replace
them with crops from labelled real frames before trusting this on a live stream.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

_G = (20, 32)  # glyph size (w, h)
_FONTS = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX)


def _norm(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    g = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
    return cv2.resize(g, _G, interpolation=cv2.INTER_AREA).astype(np.float32)


def _templates() -> tuple[np.ndarray, list[str]]:
    ts, labels = [], []
    for font in _FONTS:
        for d in "0123456789":
            img = np.zeros((120, 100), np.uint8)
            cv2.putText(img, d, (10, 100), font, 3.0, 255, 6, cv2.LINE_AA)
            ts.append(_norm(img > 127)); labels.append(d)
    return np.stack(ts), labels


_T, _L = _templates()
_Tz = (_T - _T.mean((1, 2), keepdims=True))
_Tz /= np.linalg.norm(_Tz.reshape(len(_T), -1), axis=1)[:, None, None]


def _glyphs(bgr: np.ndarray) -> list[np.ndarray]:
    """Left-to-right digit masks from a ROI crop; colon dots and specks dropped."""
    if bgr.size == 0:
        return []
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    s = 96 / g.shape[0]
    g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if m.mean() > 127:  # text is the minority class
        m = 255 - m
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))  # heals thin strokes lost to blur/JPEG
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    comps = [st[i] for i in range(1, n) if st[i, 4] > 40]
    if not comps:
        return []
    hmax = max(c[3] for c in comps)
    out = []
    for x, y, w, h, _a in sorted((c for c in comps if c[3] >= 0.6 * hmax), key=lambda c: c[0]):
        k = max(1, round(w / (0.85 * h)))  # touching digits -> equal slices
        for i in range(k):
            sl = (lab[y:y + h, x + i * w // k:x + (i + 1) * w // k] > 0)
            if sl.sum() > 40:
                out.append(sl.astype(np.uint8) * 255)
    return out


def read_digits(bgr: np.ndarray) -> str:
    out = []
    for gm in _glyphs(bgr):
        v = _norm(gm > 127)
        v = v - v.mean()
        v /= (np.linalg.norm(v) + 1e-6)
        out.append(_L[int((_Tz.reshape(len(_T), -1) @ v.ravel()).argmax())])
    return "".join(out)


def read_int(bgr: np.ndarray) -> Optional[int]:
    s = read_digits(bgr)
    return int(s) if s else None


def read_timer(bgr: np.ndarray) -> Optional[float]:
    """'m:ss' (3 digits, colon dots dropped) or bare seconds (1-2 digits)."""
    s = read_digits(bgr)
    if len(s) == 3:
        return float(int(s[0]) * 60 + int(s[1:]))
    if 1 <= len(s) <= 2:
        return float(int(s))
    return None
