"""
yolo_test.py — prove a pretrained deep detector beats the HSV ball engine.

Runs two big local models on one frame (no API, weights auto-download once):
  1. YOLO11x (COCO) filtered to 'sports ball' (cls 32) — should catch the FUEL foam balls.
  2. YOLO-World (open-vocab) prompted with ['ball','robot'] — no training needed.
High inference resolution (imgsz) is the key for the small, dense balls.
"""
import sys, time
import cv2
from ultralytics import YOLO

IMG = sys.argv[1] if len(sys.argv) > 1 else "data/sim/_yolotest.png"
img = cv2.imread(IMG)
h, w = img.shape[:2]
print(f"frame {w}x{h}")

# ---- 1. COCO sports-ball ----
t = time.time()
m = YOLO("yolo11x.pt")
print(f"loaded yolo11x in {time.time()-t:.1f}s")
for imgsz in (1280, 1920):
    t = time.time()
    r = m.predict(IMG, conf=0.06, classes=[32], imgsz=imgsz, verbose=False)[0]
    n = len(r.boxes)
    print(f"COCO sports-ball @imgsz={imgsz}, conf=0.06 -> {n} balls in {time.time()-t:.1f}s")
    ann = img.copy()
    for b in r.boxes.xyxy.cpu().numpy().astype(int):
        x1, y1, x2, y2 = b
        cv2.circle(ann, ((x1+x2)//2, (y1+y2)//2), max(3, (x2-x1)//2), (0, 255, 255), 1)
    cv2.putText(ann, f"YOLO11x sports-ball: {n} (imgsz={imgsz})", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.imwrite(f"data/sim/_yolo_balls_{imgsz}.png", cv2.resize(ann, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4))

# ---- 2. YOLO-World open-vocab (ball + robot) ----
try:
    t = time.time()
    mw = YOLO("yolov8x-worldv2.pt")
    mw.set_classes(["ball", "robot"])
    print(f"loaded YOLO-World in {time.time()-t:.1f}s")
    rw = mw.predict(IMG, conf=0.02, imgsz=1280, verbose=False)[0]
    ann = img.copy()
    nb = nr = 0
    for box, cls in zip(rw.boxes.xyxy.cpu().numpy().astype(int), rw.boxes.cls.cpu().numpy().astype(int)):
        x1, y1, x2, y2 = box
        if cls == 0:
            cv2.circle(ann, ((x1+x2)//2, (y1+y2)//2), max(3, (x2-x1)//2), (0, 255, 255), 1); nb += 1
        else:
            cv2.rectangle(ann, (x1, y1), (x2, y2), (255, 90, 0), 2); nr += 1
    print(f"YOLO-World ['ball','robot'] -> {nb} balls, {nr} robots")
    cv2.putText(ann, f"YOLO-World ball:{nb} robot:{nr}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.imwrite("data/sim/_yoloworld.png", cv2.resize(ann, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4))
except Exception as e:
    print("YOLO-World failed:", e)
