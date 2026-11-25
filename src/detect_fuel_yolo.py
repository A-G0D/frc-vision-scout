"""
detect_fuel_yolo.py — FUEL ball detection with a pretrained DEEP model (YOLO11x, COCO).

Replaces HSV color-thresholding. YOLO's 'sports ball' class (learned on real balls) finds
the foam FUEL even in dense clusters / partial occlusion / odd lighting — far higher recall
than color blobs. Runs locally (weights cached, no API/network after first download).

Two tricks that matter:
  * high inference resolution (imgsz=1920) so the small dense balls are resolvable — this is
    the difference between ~28 and ~108 detections on a 640x360 broadcast frame.
  * carpet-polygon gate on the OUTPUT: keep only detections whose centre is on the field,
    so an occasional crowd/stands false positive is dropped (deep recall + geometric precision).

Model is a lazy singleton; reuse across frames. CPU inference at imgsz=1920 is ~10s/frame, so
for whole-video runs use imgsz=1280 (faster, fewer) or a GPU.

API:  detect(frame, draw=False, imgsz=1920, conf=0.06, carpet=True) -> (list[(cx,cy,r)], annotated|None)
"""
from __future__ import annotations
import cv2, numpy as np

# reuse the calibrated carpet trapezoid from the HSV engine
try:
    from detect_fuel import FIELD_POLY_NORM
except Exception:
    FIELD_POLY_NORM = [(0.045, 0.305), (0.955, 0.305), (0.998, 0.560), (0.002, 0.560)]

_MODEL = None
def _model(weights="yolo11x.pt"):
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO(weights)
    return _MODEL


def _carpet_poly_px(w, h):
    return np.array([(int(x * w), int(y * h)) for x, y in FIELD_POLY_NORM], np.int32)

# PLAY COLUMN (fractions of height): the whole field volume INCLUDING the air above the
# carpet (so airborne shots are kept) and the front barrier (close balls). Excludes only the
# scoreboard bar (top) and the reflection strip (bottom) — the only places that produce
# false 'balls'. Diagnostics showed YOLO does NOT fire on the deep crowd, so a band is safe.
PLAY_Y = (0.125, 0.62)


def detect(frame, draw=False, imgsz=1920, conf=0.04, gate="play"):
    """gate: 'play' (air column + carpet + front, recommended) | 'carpet' (tight, drops airborne) | None."""
    h, w = frame.shape[:2]
    r = _model().predict(frame, imgsz=imgsz, conf=conf, classes=[32], verbose=False)[0]
    poly = _carpet_poly_px(w, h)
    y0, y1 = int(PLAY_Y[0] * h), int(PLAY_Y[1] * h)
    balls = []
    for b in r.boxes.xyxy.cpu().numpy():
        x1, yy1, x2, yy2 = b
        cx, cy = int((x1 + x2) / 2), int((yy1 + yy2) / 2)
        rad = max(2, int((x2 - x1) / 2))
        if gate == "carpet" and cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) < 0:
            continue
        if gate == "play" and not (y0 <= cy <= y1):   # keep air column + carpet + front; drop bar/reflection
            continue
        balls.append((cx, cy, rad))

    annotated = None
    if draw:
        annotated = frame.copy()
        if gate == "play":
            cv2.rectangle(annotated, (0, y0), (w, y1), (255, 255, 255), 1)
        elif gate == "carpet":
            cv2.polylines(annotated, [poly], True, (255, 255, 255), 1)
        for (x, y, rr) in balls:
            cv2.circle(annotated, (x, y), max(2, rr), (0, 255, 255), 1)
        cv2.putText(annotated, f"YOLO FUEL: {len(balls)}", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return balls, annotated


if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else "data/sim/_yolotest.png")
    imgsz = int(sys.argv[3]) if len(sys.argv) > 3 else 1920
    balls, ann = detect(img, draw=True, imgsz=imgsz)
    out = sys.argv[2] if len(sys.argv) > 2 else "data/sim/_fuel_yolo.png"
    cv2.imwrite(out, cv2.resize(ann, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_LANCZOS4))
    print(f"YOLO carpet-masked FUEL: {len(balls)} -> {out}")
