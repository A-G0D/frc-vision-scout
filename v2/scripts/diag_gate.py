"""Decompose the field gate's rejections: pane-test vs field-backprojection-test.

For each (video, frame), take the class-1 robot dets at conf>=0.10, and for each
report whether it passed/failed (a) the main-pane test (y2 < 0.52*H) and (b) the
field back-projection test (contains_image_pt). This isolates WHICH half of the gate
is destroying recall.
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cv2, numpy as np
from ultralytics import YOLO
from frcscout.io.types import Detection
from frcscout.geometry.field import FieldModel
from frcscout.geometry.apriltag import PoseEstimator
from frcscout.io.config import Config

ROOT = os.path.join(os.path.dirname(__file__), "..")
cfg = Config.load(os.path.join(ROOT, "configs", "deploy.yaml"))
field = FieldModel(os.path.join(ROOT, cfg.get("field_config")))
model = YOLO(os.path.join(ROOT, "weights", "ball_openvino_model"), task="detect")
MAIN = float(cfg.get("main_pane_frac", 0.52))
VIDEOS = ["qual50", "qual56", "qual59", "final1", "match_1080", "ubr1m1"]
FRACS = [0.15, 0.30, 0.45, 0.60, 0.75]

print(f"{'video':12s} {'frame':>6s} | n>=.10  pane_fail  field_fail  both_pass(kept)", flush=True)
agg = {"n": 0, "pane_fail": 0, "field_fail": 0, "kept": 0}
for v in VIDEOS:
    cap = cv2.VideoCapture(os.path.join(ROOT, "data", "footage", f"{v}.mp4"))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pe = PoseEstimator(field)
    for frac in FRACS:
        fidx = int(total * frac)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok: continue
        r = model.predict(frame, imgsz=960, conf=0.10, device="cpu", verbose=False)[0]
        boxes = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy(); cls = r.boxes.cls.cpu().numpy().astype(int)
            boxes = [tuple(map(float, b)) for b, c in zip(xyxy, cls) if c == 1]
        pose = pe.estimate(frame, fidx, region=(0.0, MAIN))
        y_main = MAIN * Hh
        if pose is None or not pose.valid:
            print(f"{v:12s} {fidx:>6d} | NO VALID POSE (gate no-op, keeps {len(boxes)})", flush=True)
            continue
        pane_fail = field_fail = kept = 0
        for b in boxes:
            x1, y1, x2, y2 = b
            cx = (x1 + x2) / 2.0
            in_pane = y2 < y_main
            on_field = field.contains_image_pt((cx, y2), pose)
            if not in_pane:
                pane_fail += 1
            elif not on_field:
                field_fail += 1
            else:
                kept += 1
        agg["n"] += len(boxes); agg["pane_fail"] += pane_fail
        agg["field_fail"] += field_fail; agg["kept"] += kept
        print(f"{v:12s} {fidx:>6d} | {len(boxes):>5d}  {pane_fail:>9d}  {field_fail:>10d}  {kept:>14d}", flush=True)
    cap.release()
print("\nAGGREGATE:", json.dumps(agg), flush=True)
print(f"of {agg['n']} robot dets(>=.10): pane_fail={agg['pane_fail']} "
      f"({100*agg['pane_fail']/max(1,agg['n']):.0f}%), "
      f"field_fail={agg['field_fail']} ({100*agg['field_fail']/max(1,agg['n']):.0f}%), "
      f"kept={agg['kept']} ({100*agg['kept']/max(1,agg['n']):.0f}%)", flush=True)
