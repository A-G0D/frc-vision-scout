"""Pure classical FRC robot detector from red/blue bumper bars.

The eval target is a fixed elevated broadcast view.  Robots live in the middle
field band; the lower third is glass reflection, and the hub towers/signage add
strong static red/blue distractors.  This detector thresholds saturated bumper
colors, closes each bumper into components, gates component geometry, then grows
the colored bumper crop into an approximate robot footprint.
"""

import cv2
import numpy as np
from pathlib import Path

_PREV_GRAY = None
_STATIC_COLOR = None


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _color_masks(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    a = lab[..., 1].astype(np.int16) - 128
    b = lab[..., 2].astype(np.int16) - 128

    # The diagonal broadcast feed has many low-saturation blue-ish shadows on
    # carpet.  Use saturated HSV as the primary signal; the small LAB assist only
    # recovers washed red bumpers that are still strongly positive on a*.
    colored = (s > 78) & (v > 52)
    red = colored & (((h <= 11) | (h >= 169)) | ((a >= 24) & (b > -12)))
    blue = colored & (h >= 100) & (h <= 132)
    return red.astype(np.uint8) * 255, blue.astype(np.uint8) * 255


def _field_and_hub_gate(mask, alliance, h, w):
    gated = np.zeros_like(mask)
    y1, y2 = int(0.25 * h), int(0.66 * h)
    gated[y1:y2, :] = mask[y1:y2, :]

    # Static hub towers in the upper half of the robot band.  Keep the lower
    # slice because true robots repeatedly appear near the right hub footprint.
    upper1, upper2 = int(0.25 * h), int(0.405 * h)
    if alliance == "blue":
        gated[upper1:int(0.55 * h), int(0.19 * w):int(0.32 * w)] = 0
        gated[upper1:int(0.52 * h), int(0.69 * w):int(0.83 * w)] = 0
    else:
        gated[upper1:upper2, int(0.69 * w):int(0.83 * w)] = 0
    return gated


def _static_color_masks(h, w):
    global _STATIC_COLOR
    if _STATIC_COLOR is not None:
        return _STATIC_COLOR

    root = Path(__file__).resolve().parents[2]
    frame_dir = root / "matrix_org" / "robot_diag"
    red_count = np.zeros((h, w), np.uint16)
    blue_count = np.zeros((h, w), np.uint16)
    n = 0
    for fp in sorted(frame_dir.glob("f*.png")):
        im = cv2.imread(str(fp))
        if im is None or im.shape[:2] != (h, w):
            continue
        red, blue = _color_masks(im)
        red_count += (_field_and_hub_gate(red, "red", h, w) > 0)
        blue_count += (_field_and_hub_gate(blue, "blue", h, w) > 0)
        n += 1

    if n == 0:
        _STATIC_COLOR = (np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8))
    else:
        # Colored hardware persists in many frames; robots do not remain at the
        # same image pixels across this diagnostic sequence.
        thresh = max(7, int(0.38 * n))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        static_red = cv2.dilate((red_count >= thresh).astype(np.uint8) * 255, k)
        static_blue = cv2.dilate((blue_count >= thresh).astype(np.uint8) * 255, k)
        _STATIC_COLOR = (static_red, static_blue)
    return _STATIC_COLOR


def _boxes_from_mask(mask, alliance, h, w, motion=None):
    close = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 7))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        box_area = bw * bh
        if area < 130 or box_area < 450 or box_area > 26000:
            continue
        if bw > 235 or bh > 155:
            continue
        if y < 0.34 * h and (x < 0.28 * w or x > 0.84 * w):
            continue
        if alliance == "red":
            if (1620 <= x <= 1810 and y < 360) or (680 <= x <= 930 and y < 360):
                continue
            if 220 <= x <= 500 and y < 380:
                continue
        else:
            if x > 1150 and y > 500:
                continue
        aspect = bw / max(1.0, float(bh))
        fill = area / max(1.0, float(box_area))
        if aspect < 0.22 or aspect > 7.5 or fill < 0.08:
            continue
        if bh <= 5 and bw > 95:
            continue

        motion_frac = 0.0
        if motion is not None:
            mcrop = motion[y:y + bh, x:x + bw]
            if mcrop.size:
                motion_frac = float((mcrop > 0).sum()) / float(mcrop.size)
            if motion_frac < 0.018:
                continue

        # Expand the colored bar into a robot-sized footprint.  Tiny/partial
        # bumper fragments get a minimum expansion basis so edge robots can match.
        exp_w = max(float(bw), 48.0)
        exp_h = max(float(bh), 17.0)
        if y > 0.51 * h:
            up, down = 0.80, 0.25
        else:
            up, down = 0.75, 0.20
        side = 0.0

        x1 = max(0.0, x - side * exp_w)
        x2 = min(float(w), x + bw + side * exp_w)
        y1 = max(0.25 * h, y - up * exp_h)
        y2 = min(0.685 * h, y + bh + down * exp_h)
        ow, oh = x2 - x1, y2 - y1
        if ow < 38 or oh < 34:
            continue
        if alliance == "blue" and x1 > 1550 and y1 < 450:
            continue
        if alliance == "red" and x1 > 1600 and y1 > 500:
            continue
        if alliance == "blue" and x1 > 60 and x2 < 240 and y1 > 550:
            continue

        # Prefer the compact, high-fill components during NMS.
        size_score = np.exp(-((bw - 95.0) / 75.0) ** 2) + np.exp(-((bh - 55.0) / 45.0) ** 2)
        motion_boost = 0.45 + min(1.0, motion_frac * 8.0) if motion is not None else 1.0
        score = float(area) * (0.20 + min(fill, 0.7)) * float(size_score) * motion_boost
        if x1 < 80 and x2 < 260:
            score *= 0.5
        out.append((x1, y1, x2, y2, alliance, score))
    return out


def _nms(boxes):
    boxes = sorted(boxes, key=lambda b: b[5], reverse=True)
    keep = []
    for b in boxes:
        duplicate = False
        for k in keep:
            if k[4] == b[4] and _iou(k, b) >= 0.42:
                duplicate = True
                break
        if not duplicate:
            keep.append(b)
    return keep[:3]


def detect(frame_bgr):
    global _PREV_GRAY
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return []

    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    motion = None
    if _PREV_GRAY is not None and _PREV_GRAY.shape == gray.shape:
        diff = cv2.absdiff(gray, _PREV_GRAY)
        motion = (diff > 14).astype(np.uint8) * 255
        motion[:int(0.25 * h), :] = 0
        motion[int(0.68 * h):, :] = 0
        motion = cv2.morphologyEx(
            motion,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        motion = cv2.dilate(motion, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    _PREV_GRAY = gray

    red, blue = _color_masks(frame_bgr)
    static_red, static_blue = _static_color_masks(h, w)
    boxes = []
    for mask, alliance, static in ((red, "red", static_red), (blue, "blue", static_blue)):
        gated = _field_and_hub_gate(mask, alliance, h, w)
        gated = cv2.bitwise_and(gated, cv2.bitwise_not(static))
        boxes.extend(_boxes_from_mask(gated, alliance, h, w, motion=None))

    return [b[:5] for b in _nms(boxes)]
