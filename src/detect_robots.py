"""
detect_robots.py — locate ROBOTS by bumper color in a live FRC REBUILT (2026) field view.

Robots wear saturated RED or BLUE alliance bumpers. On the grey-carpet broadcast view they
appear as compact, solidly-colored blobs — bigger than FUEL balls, smaller and squarer than
the wide HUB goal bases, and far more solid than thin field tape or banners.

Scene reality (measured from the Waterbury frames, 640x360):
  * The two alliance HUB goals are colored box structures (blue left, red right). Their bases
    read as the LARGEST, WIDEST saturated blobs (area 600-1000, aspect 2-3). Robots are the
    medium, near-square blobs (area ~100-450, aspect <~2).
  * Top scoreboard bar (0..18%) and bottom caption / split-camera (>~58%) are also red/blue
    and are excluded by ROI.

Discriminators (classical CV only — OpenCV + numpy):
  * COLOR: saturated red (H~0/180) or blue (H~100-128) with high S and high V (drops dark navy
    HUB bodies and muted tape).
  * REGION: field band only.
  * SHAPE: area band + aspect cap (rejects the wide HUB bases) + fill + solidity. Big-but-square
    blobs that exceed the robot area cap are treated as merged robots / HUB tops and split or
    dropped by a width-aware rule.

detect(frame, draw=False) -> (list[{'bbox':(x,y,w,h),'alliance':'red'|'blue'}], annotated)
Counts are approximate (occlusion, robots clustered at the HUB, camera cuts).
"""

from __future__ import annotations

import cv2
import numpy as np

# Playing-field band (fractions of W x H). Excludes top scoreboard bar and the bottom
# split-camera / caption where wall banners + crowd add false red+blue.
FIELD_ROI = (0.0, 0.18, 1.0, 0.58)

# --- HSV gates (OpenCV H is 0..179) -----------------------------------------------------
# High S and high V isolate vivid bumper color and reject dark navy HUB bodies / shadow tape.
BLUE_LO = np.array([100, 110, 95])
BLUE_HI = np.array([128, 255, 255])

RED1_LO = np.array([0, 120, 95])
RED1_HI = np.array([9, 255, 255])
RED2_LO = np.array([169, 120, 95])
RED2_HI = np.array([179, 255, 255])

# --- blob shape filters (px at native 640x360) -----------------------------------------
MIN_AREA = 90          # robots are big; fuel balls / tape specks / crowd dots are smaller
MAX_AREA = 520         # above this it is a HUB base / merged structure (split-checked below)
MIN_SOLIDITY = 0.55    # filled blob vs. thin jagged structure edge
MIN_FILL = 0.40        # contour area / bbox area: reject sparse scatter
MAX_ASPECT = 2.2       # bumpers are squat-to-square; HUB bases are wide (asp 2-3)
MIN_DIM = 9            # smallest bbox side; kills slivers and stray ball clumps
MAX_W = 40             # robots aren't wider than this at this zoom; wider = HUB base

# Big square blobs (area in [MAX_AREA, BIG_AREA], aspect<~1.6) are likely two robots merged
# or a HUB top; keep as a single robot if reasonably square, else drop.
BIG_AREA = 900
BIG_MAX_ASPECT = 1.7

_KERNEL = np.ones((3, 3), np.uint8)


def _field_crop(frame):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = FIELD_ROI
    y0p, y1p = int(y0 * h), int(y1 * h)
    x0p, x1p = int(x0 * w), int(x1 * w)
    return frame[y0p:y1p, x0p:x1p], (x0p, y0p)


def _accept_shape(a, w, h):
    """Return True if a contour's geometry looks like a robot bumper blob."""
    if w < MIN_DIM or h < MIN_DIM:
        return False
    aspect = max(w / h, h / w)
    fill = a / float(w * h)
    if fill < MIN_FILL:
        return False
    if a <= MAX_AREA:
        return aspect <= MAX_ASPECT and w <= MAX_W
    # Larger blob: only keep if it is big-but-square (merged robots / HUB top facing us),
    # never the wide HUB base.
    if a <= BIG_AREA and aspect <= BIG_MAX_ASPECT and w <= MAX_W + 8:
        return True
    return False


def _blobs(mask, ox, oy, alliance):
    # Open to kill speckle, then close to consolidate a bumper broken up by an arm / fuel.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        hull = cv2.convexHull(c)
        ha = cv2.contourArea(hull)
        solidity = a / ha if ha > 0 else 0.0
        if solidity < MIN_SOLIDITY:
            continue
        if not _accept_shape(a, w, h):
            continue
        out.append({"bbox": (x + ox, y + oy, w, h), "alliance": alliance})
    return out


def detect(frame, draw: bool = False):
    crop, (ox, oy) = _field_crop(frame)
    blur = cv2.GaussianBlur(crop, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    blue = cv2.inRange(hsv, BLUE_LO, BLUE_HI)
    red = cv2.inRange(hsv, RED1_LO, RED1_HI) | cv2.inRange(hsv, RED2_LO, RED2_HI)

    robots = _blobs(blue, ox, oy, "blue") + _blobs(red, ox, oy, "red")

    annotated = None
    if draw:
        annotated = frame.copy()
        h_full, w_full = frame.shape[:2]
        cv2.rectangle(annotated, (0, int(FIELD_ROI[1] * h_full)),
                      (w_full - 1, int(FIELD_ROI[3] * h_full) - 1), (0, 200, 0), 1)
        for r in robots:
            x, y, w, h = r["bbox"]
            col = (255, 0, 0) if r["alliance"] == "blue" else (0, 0, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), col, 2)
        nb = sum(1 for r in robots if r["alliance"] == "blue")
        nr = sum(1 for r in robots if r["alliance"] == "red")
        cv2.putText(annotated, f"ROBOTS b:{nb} r:{nr}", (8, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return robots, annotated


if __name__ == "__main__":
    import sys, glob, os, time
    paths = sys.argv[1:] or sorted(glob.glob("frames/dataset/live_*.png"))
    outdir = "frames/dataset/robots_annot"
    os.makedirs(outdir, exist_ok=True)
    times = []
    for p in paths:
        frame = cv2.imread(p)
        t0 = time.perf_counter()
        robots, annotated = detect(frame, draw=True)
        ms = (time.perf_counter() - t0) * 1000
        times.append(ms)
        out = os.path.join(outdir, os.path.basename(p))
        cv2.imwrite(out, annotated)
        nb = sum(1 for r in robots if r["alliance"] == "blue")
        nr = sum(1 for r in robots if r["alliance"] == "red")
        print(f"{os.path.basename(p):28s}: {len(robots)} robots (blue {nb}, red {nr})  {ms:.1f}ms")
    if times:
        print(f"\nMean {sum(times)/len(times):.2f} ms/frame  max {max(times):.2f} ms")
