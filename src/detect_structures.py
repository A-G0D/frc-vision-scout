"""
detect_structures.py — locate static REBUILT (2026) field STRUCTURES in a
broadcast field-view frame, primarily the HUBs.

Object
------
HUBs are the dark navy / near-black tall box structures standing near
center field (about four of them in the Waterbury views, in alliance-coloured
pairs). They are fed FUEL through an opening, carry alliance-coloured trim and
white/grey panel markings, and sit in a horizontal band across mid-field.

Strategy (classical CV, deterministic, fast)
--------------------------------------------
1. Work on the mid-field ROI only — exclude the scoreboard bar (top ~18%),
   the caption strip (bottom ~7%), and the lower crowd / scoring-table band
   where dark seating dominates. The HUBs live in y ~ [0.27, 0.63].
2. Build a "structure" mask in HSV: low Value (the boxes read much darker than
   the grey carpet, yellow fuel and bright crowd), OR dark-ish with strong
   blue/red alliance trim. Both count as structure pixels.
3. Morphology to consolidate each box (close gaps from white markings / fuel
   occlusion) and drop carpet speckle.
4. Contour filter: area band, plausible box aspect ratio, minimum fill, and a
   genuine-darkness check on the original pixels inside the box.
5. Light NMS, then label every survivor as kind="hub". (TOWER climbers are not
   reliably separable from the HUBs at this scale in these views, so we emit
   HUBs only.)

Signature
---------
    detect(frame, draw=False)
        -> (list[{"bbox":(x,y,w,h), "kind":str, "score":float}], annotated_or_None)
"""

from __future__ import annotations

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Tunables (native 640x360; ROI is fractional so it scales)
# ---------------------------------------------------------------------------

# Mid-field carpet band where the HUB boxes stand. The grey carpet (V~120) is
# bright; the HUBs are isolated dark boxes sitting ON it. Top excludes the dark
# textured crowd/stands; bottom excludes the scoring-table / ref band + caption.
FIELD_ROI = (0.02, 0.30, 0.98, 0.61)

# Darkness gate (HSV Value). Carpet median V ~123; box bodies sit in the bottom
# few percent (V ~38). A threshold well below carpet isolates the boxes.
DARK_V_MAX = 70          # pixel is "dark body" if V <= this

# Morphology: an ASYMMETRIC (taller-than-wide) CLOSE fills the white face-panel
# holes inside each box vertically, while bridging little horizontally — so the
# two boxes of an alliance pair (sitting side-by-side ~15-20px apart) stay
# separate. A small OPEN then removes carpet speckle.
CLOSE_K = (5, 9)
OPEN_K  = (3, 3)

# Contour filters (px / px^2 at 640x360 inside the ROI-cropped image)
MIN_AREA   = 230
MAX_AREA   = 5500
MIN_W, MAX_W = 18, 92
MIN_H, MAX_H = 16, 82
# Aspect = w/h. HUB boxes are squarish-to-slightly-wide; reject long thin
# horizontal shadow bands and tall sliver artifacts.
MIN_ASPECT = 0.55
MAX_ASPECT = 2.4
MIN_FILL   = 0.42        # solid-blob fill after CLOSE
MIN_DARK_FRAC = 0.26     # fraction of bbox pixels genuinely dark (in raw V)
# Carpet-context check: a real HUB is an isolated dark box surrounded by bright
# carpet, so its immediate border ring should be markedly brighter than its body.
MIN_BORDER_CONTRAST = 22 # mean(border V) - mean(body V) must exceed this
# The REBUILT field carries four HUBs (two alliance pairs). Keep the four best
# scoring isolated-dark-box candidates; this drops weaker false positives
# (dark robots/people, edge clutter) which score lower than the real HUBs.
MAX_HUBS = 4
NMS_IOU = 0.30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _roi_crop(frame: np.ndarray):
    h, w = frame.shape[:2]
    x0f, y0f, x1f, y1f = FIELD_ROI
    x0, y0 = int(x0f * w), int(y0f * h)
    x1, y1 = int(x1f * w), int(y1f * h)
    return frame[y0:y1, x0:x1], (x0, y0)


def _structure_mask(crop_bgr: np.ndarray):
    """Return (binary structure mask, value channel) for the ROI crop."""
    blur = cv2.GaussianBlur(crop_bgr, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]

    mask = (v <= DARK_V_MAX).astype(np.uint8) * 255          # navy/black body

    # Strong close fills the white face-panel holes; open removes carpet speckle.
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSE_K), iterations=1)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, OPEN_K), iterations=1)
    return mask, v


def _nms(dets):
    """dets: list of (score, (x,y,w,h)). Greedy IoU suppression."""
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: -d[0])
    keep = []
    for d in dets:
        x1, y1, w1, h1 = d[1]
        a1 = w1 * h1
        ok = True
        for k in keep:
            x2, y2, w2, h2 = k[1]
            ix = max(x1, x2); iy = max(y1, y2)
            ax = min(x1 + w1, x2 + w2); ay = min(y1 + h1, y2 + h2)
            iw = max(0, ax - ix); ih = max(0, ay - iy)
            inter = iw * ih
            union = a1 + w2 * h2 - inter
            if union > 0 and inter / union > NMS_IOU:
                ok = False
                break
        if ok:
            keep.append(d)
    return keep


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(frame: np.ndarray, draw: bool = False):
    """
    Detect REBUILT field STRUCTURES (HUBs) in one broadcast frame.

    Returns
    -------
    structures : list of {"bbox": (x, y, w, h), "kind": "hub", "score": float}
                 in original frame pixel coordinates, left-to-right.
    annotated  : annotated BGR copy if draw=True, else None.
    """
    crop, (ox, oy) = _roi_crop(frame)
    mask, vch = _structure_mask(crop)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cands = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not (MIN_W <= w <= MAX_W and MIN_H <= h <= MAX_H):
            continue
        aspect = w / float(h)
        if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            continue
        fill = area / float(w * h)
        if fill < MIN_FILL:
            continue
        patch = vch[y:y + h, x:x + w]
        dark_frac = float((patch <= DARK_V_MAX).mean())
        if dark_frac < MIN_DARK_FRAC:
            continue
        body_v = float(patch[patch <= DARK_V_MAX].mean()) if dark_frac > 0 else 255.0
        # Border-ring brightness: a real HUB sits on bright carpet, so the ring
        # of pixels just outside the bbox is much brighter than the dark body.
        m = 4
        H0, W0 = vch.shape
        ry0, ry1 = max(0, y - m), min(H0, y + h + m)
        rx0, rx1 = max(0, x - m), min(W0, x + w + m)
        ring = vch[ry0:ry1, rx0:rx1]
        ring_mean = float(ring.mean())
        if ring_mean - body_v < MIN_BORDER_CONTRAST:
            continue
        # Prefer larger, solid, dark, well-isolated boxes. Also prefer boxes in
        # the upper-middle of the carpet band: the real HUBs stand mid-field,
        # while near-camera dark robots/people fall toward the band bottom.
        contrast = ring_mean - body_v
        cy_band = (y + h / 2) / float(H0)            # 0 = band top, 1 = band bottom
        vpos = 1.0 - 0.6 * max(0.0, cy_band - 0.45)  # gentle penalty below 45%
        score = (area * (0.4 + 0.6 * fill) * (0.5 + dark_frac)
                 * (1.0 + contrast / 120.0) * vpos)
        cands.append((score, (x + ox, y + oy, w, h)))

    kept = _nms(cands)[:MAX_HUBS]

    structures = [
        {"bbox": bbox, "kind": "hub", "score": round(float(s), 1)}
        for s, bbox in kept
    ]
    structures.sort(key=lambda d: d["bbox"][0])

    annotated = None
    if draw:
        annotated = frame.copy()
        H, W = frame.shape[:2]
        rx0, ry0 = int(FIELD_ROI[0] * W), int(FIELD_ROI[1] * H)
        rx1, ry1 = int(FIELD_ROI[2] * W), int(FIELD_ROI[3] * H)
        cv2.rectangle(annotated, (rx0, ry0), (rx1, ry1), (120, 120, 120), 1)
        for sdct in structures:
            x, y, w, h = sdct["bbox"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(annotated, sdct["kind"], (x, max(10, y - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        cv2.putText(annotated, f"HUBs: {len(structures)}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    return structures, annotated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, glob, os, time

    paths = sys.argv[1:] or sorted(glob.glob("frames/dataset/live_*.png"))
    out_dir = "frames/dataset/structures_annot"
    os.makedirs(out_dir, exist_ok=True)

    times = []
    for p in paths:
        frame = cv2.imread(p)
        if frame is None:
            print(f"SKIP: {p}")
            continue
        t0 = time.perf_counter()
        structs, _ = detect(frame, draw=False)
        ms = (time.perf_counter() - t0) * 1000
        times.append(ms)
        _, annotated = detect(frame, draw=True)
        name = os.path.basename(p).replace(".png", "_struct.png")
        cv2.imwrite(os.path.join(out_dir, name), annotated)
        print(f"{os.path.basename(p):32s}  hubs={len(structs):2d}  {ms:.2f} ms")

    if times:
        print(f"\nMean {sum(times)/len(times):.2f} ms/frame  max {max(times):.2f} ms")
