"""
detect_fuel.py — locate FUEL (yellow foam balls) in a broadcast field frame.

Iteration 6 — carpet-polygon masked, perspective-aware, cluster-splitting.

Why this rewrite: the previous horizontal-band ROI (y 0.21–0.91) swallowed the upper
crowd and the entire lower reflected-stands region, so circles landed all over non-field
content. The fix is to mask to the actual CARPET POLYGON (a perspective trapezoid). Within
that mask:
  * HSV yellow gate (broadcast foam ball colour, margin for shadow).
  * connected components, routed by area with a PERSPECTIVE-AWARE floor (balls at the far
    /top edge are tiny ~3–5 px; near/bottom balls are ~8–12 px), so distant balls aren't
    discarded as noise while crowd speckle still is.
  * big blobs (touching balls) are split by distance-transform peak detection.
  * NMS de-duplicates.

CALIBRATION: FIELD_POLY_NORM is the carpet trapezoid for the static wide camera used in the
playoff footage. Pass field_poly=... to override for a different camera. Set field_poly=None
to fall back to the legacy horizontal band.

Signature (unchanged + optional arg):
    detect(frame, draw=False, field_poly=FIELD_POLY_NORM) -> (list[(cx,cy,r)], annotated|None)
"""

from __future__ import annotations

import cv2
import numpy as np

# Carpet trapezoid (normalized x,y), clockwise from top-left. Tight to the grey carpet:
# excludes crowd above (~<0.30) and the barrier/reflection below (~>0.55).
FIELD_POLY_NORM = [(0.045, 0.305), (0.955, 0.305), (0.998, 0.560), (0.002, 0.560)]
LEGACY_BAND = (0.0, 0.21, 1.0, 0.91)

# HSV yellow — foam balls H≈25-33; widen a touch for shadowed balls.
YELLOW_LO = np.array([18, 75, 85], dtype=np.uint8)
YELLOW_HI = np.array([42, 255, 255], dtype=np.uint8)

# Area routing (px^2 at 360p, BEFORE perspective scaling).
ABS_MIN_AREA     = 3        # hard floor (anything smaller is noise even far away)
SINGLE_MAX_AREA  = 550      # <= this (after row-scaling) -> single ball
CLUSTER_MAX_AREA = 24000    # > this -> discard (HUB body / seam)
MIN_CIRCULARITY  = 0.12     # carpet mask already guarantees yellow=ball, so be permissive
BALL_AREA_EST    = 95       # px^2 per ball inside a cluster (tuned to carpet balls)
NMS_OVERLAP      = 0.50


def _poly_px(field_poly, w, h):
    return np.array([(int(x * w), int(y * h)) for x, y in field_poly], np.int32)


def _carpet_mask(field_poly, w, h):
    m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m, _poly_px(field_poly, w, h), 255)
    return m


def _row_scale(y, y_top, y_bot):
    """Perspective scale in [~0.35 .. 1.0]: far/top balls are smaller, so allow smaller area."""
    if y_bot <= y_top:
        return 1.0
    f = (y - y_top) / (y_bot - y_top)          # 0 at top edge, 1 at bottom edge
    return 0.35 + 0.65 * max(0.0, min(1.0, f))


def _split_cluster(blob, area):
    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return []
    d8 = (dist * 255 / dist.max()).astype(np.uint8)
    ball_r = max(3, int(np.sqrt(BALL_AREA_EST / np.pi)))
    ksz = ball_r * 2 + 1
    dil = cv2.dilate(d8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz)))
    floor = max(12, int(d8[d8 > 0].mean() * 0.5) if (d8 > 0).any() else 12)
    peaks = (((d8 >= dil - 3) & (d8 > floor)).astype(np.uint8)) * 255
    n, _, _, cents = cv2.connectedComponentsWithStats(peaks)
    out = [(int(cents[i][0]), int(cents[i][1]), ball_r) for i in range(1, n)]
    if not out:                                # fallback: 1 centre
        M = cv2.moments(blob)
        if M["m00"] > 0:
            out = [(int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]),
                    max(3, int(np.sqrt(area / np.pi / 2))))]
    return out


def _nms(dets):
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: -d[2])
    keep, sup = [], [False] * len(dets)
    for i, (xi, yi, ri) in enumerate(dets):
        if sup[i]:
            continue
        keep.append(dets[i])
        for j in range(i + 1, len(dets)):
            if not sup[j]:
                xj, yj, rj = dets[j]
                if np.hypot(xi - xj, yi - yj) < NMS_OVERLAP * (ri + rj):
                    sup[j] = True
    return keep


def detect(frame, draw=False, field_poly=FIELD_POLY_NORM):
    h, w = frame.shape[:2]
    if field_poly is None:                     # legacy band
        x0, y0, x1, y1 = LEGACY_BAND
        carpet = np.zeros((h, w), np.uint8)
        carpet[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 255
        y_top, y_bot = int(y0 * h), int(y1 * h)
    else:
        carpet = _carpet_mask(field_poly, w, h)
        ys = [int(y * h) for _, y in field_poly]
        y_top, y_bot = min(ys), max(ys)

    blur = cv2.GaussianBlur(frame, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
    mask = cv2.bitwise_and(mask, carpet)
    # Despeckle ONLY (2x2 open). Deliberately NO close: closing merges adjacent balls into
    # clumps that the splitter undercounts, which was the main recall loss.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > CLUSTER_MAX_AREA:
            continue
        (cx, cy), _ = cv2.minEnclosingCircle(c)
        scale = _row_scale(cy, y_top, y_bot)
        if area < max(ABS_MIN_AREA, 7 * scale):      # perspective-aware noise floor
            continue
        if area <= SINGLE_MAX_AREA * scale:
            perim = cv2.arcLength(c, True)
            circ = (4 * np.pi * area / (perim * perim)) if perim > 0 else 0.0
            if circ < MIN_CIRCULARITY:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            dets.append((int(x), int(y), max(2, int(r))))
        else:
            bx, by, bw, bh = cv2.boundingRect(c)
            blob = np.zeros((bh, bw), np.uint8)
            cv2.drawContours(blob, [c - np.array([bx, by])], -1, 255, -1)
            for sx, sy, r in _split_cluster(blob, area):
                dets.append((sx + bx, sy + by, r))

    balls = _nms(dets)

    annotated = None
    if draw:
        annotated = frame.copy()
        cv2.polylines(annotated, [_poly_px(field_poly or
                      [(LEGACY_BAND[0], LEGACY_BAND[1]), (LEGACY_BAND[2], LEGACY_BAND[1]),
                       (LEGACY_BAND[2], LEGACY_BAND[3]), (LEGACY_BAND[0], LEGACY_BAND[3])], w, h)],
                      True, (255, 255, 255), 1)
        for (x, y, r) in balls:
            cv2.circle(annotated, (x, y), max(2, r), (0, 255, 255), 1)
        cv2.putText(annotated, f"FUEL: {len(balls)}", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return balls, annotated


if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else "data/sim/_balltest.png")
    balls, ann = detect(img, draw=True)
    out = (sys.argv[2] if len(sys.argv) > 2 else "data/sim/_ball_v6.png")
    cv2.imwrite(out, cv2.resize(ann, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_LANCZOS4))
    print(f"detected {len(balls)} FUEL -> {out}")
