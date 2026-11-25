"""Decide the robot-detection strategy with two measurements:

1. CAMERA MOTION: mean abs frame-diff inside the static field band over a window.
   If high, the broadcast cam pans/zooms -> MOG2 motion-gating is invalid.
2. YOLO ROBOT SEPARATION: run the box detector on each footage and report how many
   distinct (post-NMS) robot boxes appear per frame, and how much they overlap.
   Tells us whether match_1080 really gives a clean ~6 vs qual59's merge/split.
"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def cam_motion(video, start=3000, n=40):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    prev = None
    diffs = []
    H = W = None
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        if H is None:
            H, W = f.shape[:2]
        # field band only (avoid scoreboard overlay + crowd)
        band = f[int(0.30*H):int(0.62*H), :]
        g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            diffs.append(float(np.mean(cv2.absdiff(g, prev))))
        prev = g
    cap.release()
    return float(np.mean(diffs)) if diffs else -1.0


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    if inter <= 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def yolo_counts(model, video, start=3000, n=30, conf=0.12):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    counts, overlaps = [], []
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            break
        if i % 3:
            continue
        r = model.predict(f, imgsz=960, conf=conf, device="cpu", verbose=False)[0]
        xy = r.boxes.xyxy.cpu().numpy()
        cl = r.boxes.cls.cpu().numpy().astype(int)
        boxes = [b for b, c in zip(xy, cl) if c == 1]
        counts.append(len(boxes))
        # max pairwise IoU = how badly boxes pile on the same robot
        mx = 0.0
        for j in range(len(boxes)):
            for k in range(j+1, len(boxes)):
                mx = max(mx, iou(boxes[j], boxes[k]))
        overlaps.append(mx)
    cap.release()
    return counts, overlaps


def main():
    from ultralytics import YOLO
    model = YOLO(str(ROOT / "weights/ball_openvino_model"), task="detect")
    for name in ("qual59", "match_1080"):
        v = ROOT / f"data/footage/{name}.mp4"
        cm = cam_motion(v)
        counts, ov = yolo_counts(model, v)
        print(f"\n=== {name} ===")
        print(f"  camera motion (field-band mean |diff|/frame): {cm:.2f}  "
              f"({'STATIC' if cm < 4 else 'PANS/ZOOMS — MOG2 invalid'})")
        if counts:
            print(f"  YOLO robot boxes/frame: mean={np.mean(counts):.1f} "
                  f"median={np.median(counts):.0f} min={min(counts)} max={max(counts)}")
            print(f"  max box-pair IoU/frame: mean={np.mean(ov):.2f} max={max(ov):.2f}  "
                  f"(high => boxes piling on one robot)")


if __name__ == "__main__":
    main()
