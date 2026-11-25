"""Test whether imgsz limits robot recall (small/distant robots) and compare the
dedicated robot2 single-class net vs the ball-model's robot class.

Runs the SAME frames at imgsz {960, 1280, 1536} on the ball model (class 1), and also
runs robot2_openvino_model (single-class @ its native 640 and pushed to 1280) for a
head-to-head. Reports robot counts at conf {0.05,0.10,0.25}.
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cv2, numpy as np
from ultralytics import YOLO

ROOT = os.path.join(os.path.dirname(__file__), "..")
ball = YOLO(os.path.join(ROOT, "weights", "ball_openvino_model"), task="detect")
try:
    robot2 = YOLO(os.path.join(ROOT, "weights", "robot2_openvino_model"), task="detect")
    print("robot2.names =", robot2.names, flush=True)
except Exception as e:
    robot2 = None
    print("robot2 load FAILED:", e, flush=True)

# a spread incl. weak frames (small/distant robots) and strong frames
SAMPLES = [
    ("qual50", 6606), ("qual50", 8808),   # weak (max_conf .10-.13)
    ("match_1080", 6480), ("qual59", 11010),  # weak
    ("qual56", 8700), ("ubr1m1", 4404),    # strong (has .78-.88)
    ("final1", 6525),                      # busy
]
CONF = [0.05, 0.10, 0.25]

def count(model, frame, imgsz, cls_id):
    try:
        r = model.predict(frame, imgsz=imgsz, conf=0.01, device="cpu", verbose=False)[0]
    except Exception as e:
        return {f"c{int(t*100):02d}": -1 for t in CONF}, "ERR(%s)" % type(e).__name__
    confs = []
    if r.boxes is not None and len(r.boxes):
        xyxy = r.boxes.xyxy.cpu().numpy(); cl = r.boxes.cls.cpu().numpy().astype(int)
        cf = r.boxes.conf.cpu().numpy()
        for c, cc in zip(cl, cf):
            if cls_id is None or c == cls_id:
                confs.append(float(cc))
    a = np.array(confs) if confs else np.array([])
    return {f"c{int(t*100):02d}": int((a >= t).sum()) for t in CONF}, (round(float(a.max()),3) if confs else None)

frames = {}
for v, f in SAMPLES:
    cap = cv2.VideoCapture(os.path.join(ROOT, "data", "footage", f"{v}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, fr = cap.read(); cap.release()
    frames[(v, f)] = fr if ok else None

print(f"\n{'sample':18s} | {'ball@960':>22s} | {'ball@1280':>22s} | {'ball@1536':>22s} | {'robot2@640':>16s} | {'robot2@1280':>16s}", flush=True)
for (v, f), fr in frames.items():
    if fr is None:
        print(f"{v}:{f} READ FAIL"); continue
    b960, m960 = count(ball, fr, 960, 1)
    b1280, m1280 = count(ball, fr, 1280, 1)
    b1536, m1536 = count(ball, fr, 1536, 1)
    if robot2 is not None:
        r640, rm640 = count(robot2, fr, 640, 0)
        r1280, rm1280 = count(robot2, fr, 1280, 0)
    else:
        r640 = r1280 = {"c05":-1,"c10":-1,"c25":-1}; rm640=rm1280=None
    def fmt(d, m): return f"{d['c05']}/{d['c10']}/{d['c25']} (mx{m})"
    print(f"{v+':'+str(f):18s} | {fmt(b960,m960):>22s} | {fmt(b1280,m1280):>22s} | {fmt(b1536,m1536):>22s} | {fmt(r640,rm640):>16s} | {fmt(r1280,rm1280):>16s}", flush=True)
print("\n(counts are robots at conf .05/.10/.25 ; mx=max conf)", flush=True)
