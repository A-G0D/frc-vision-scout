"""Robot-detector diagnostic (READ-ONLY measurement; does not modify the pipeline).

For each footage video, sample frames across the match, run the ACTUAL deployed
ball_openvino_model (classes {0:ball,1:robot}) at imgsz 960, and report:
  - robot (class-1) detection COUNT at conf thresholds {0.05, 0.10, 0.25}
  - the raw confidence values of every class-1 box (so we see how strong/weak they are)
  - field-gate effect: did PoseEstimator recover a valid pose? how many class-1 dets
    survive FieldModel.gate_detections() (the gate run_sync/MatchRunner apply)?
  - bumper_alliance() label per robot box (to explain the all-"red" export)
Saves an annotated PNG per (video, frame) to data/scout_out/diag_robot_<video>_f<n>.png.
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np
from ultralytics import YOLO

from frcscout.io.types import Detection
from frcscout.geometry.field import FieldModel
from frcscout.geometry.apriltag import PoseEstimator
from frcscout.detect.robot_net import RobotNet
from frcscout.io.config import Config

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "data", "scout_out")
os.makedirs(OUT, exist_ok=True)

VIDEOS = ["qual50", "qual56", "qual59", "final1", "match_1080", "ubr1m1"]
FRACS = [0.15, 0.30, 0.45, 0.60, 0.75]   # sample points across each match
CONF_LEVELS = [0.05, 0.10, 0.25]
IMGSZ = 960
ROBOT_CLS = 1

cfg = Config.load(os.path.join(ROOT, "configs", "deploy.yaml"))
field = FieldModel(os.path.join(ROOT, cfg.get("field_config")))
model = YOLO(os.path.join(ROOT, "weights", "ball_openvino_model"), task="detect")
print("model.names =", model.names, flush=True)

# main-pane fraction matches run_sync MAIN_PANE = (0.0, 0.52)
MAIN_PANE_FRAC = float(cfg.get("main_pane_frac", 0.52))

summary = []

for v in VIDEOS:
    path = os.path.join(ROOT, "data", "footage", f"{v}.mp4")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[{v}] cannot open", flush=True)
        continue
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pose_est = PoseEstimator(field)
    for fi, frac in enumerate(FRACS):
        fidx = int(total * frac)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            print(f"[{v}] read fail @ {fidx}", flush=True)
            continue

        # --- run model at a very low conf to capture ALL candidate robot boxes ---
        r = model.predict(frame, imgsz=IMGSZ, conf=0.01, device="cpu", verbose=False)[0]
        confs = []   # class-1 confidences
        boxes = []   # class-1 xyxy
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            for b, c, cc in zip(xyxy, cls, cf):
                if c == ROBOT_CLS:
                    boxes.append(tuple(map(float, b)))
                    confs.append(float(cc))
        confs_arr = np.array(confs) if confs else np.array([])

        counts = {f"c{int(t*100):02d}": int((confs_arr >= t).sum()) for t in CONF_LEVELS}

        # --- field gate: recover pose for the MAIN pane, then gate the >=0.10 dets ---
        # qual59 is a stacked composite -> restrict pose to the top pane band like run_sync.
        region = (0.0, MAIN_PANE_FRAC)
        pose = pose_est.estimate(frame, fidx, region=region)
        pose_valid = bool(pose is not None and pose.valid)
        n_tags = int(pose.n_tags) if pose is not None else 0
        reproj = round(float(pose.reproj_err_px), 2) if pose is not None else None

        # build Detection list at the deploy robot conf gate (0.12 ~ use 0.10 bucket boxes)
        gate_conf = 0.10
        det_boxes = [b for b, cc in zip(boxes, confs) if cc >= gate_conf]
        dets = [Detection("robot", b, 1.0, fidx) for b in det_boxes]
        y_main = MAIN_PANE_FRAC * Hh
        if pose_valid:
            kept = field.gate_detections(dets, pose, y_main)
            n_after_gate = len(kept)
        else:
            kept = dets
            n_after_gate = len(dets)  # gate is a NO-OP without valid pose

        # --- alliance vote per >=0.10 box (explain all-red export) ---
        alliances = [RobotNet.bumper_alliance(frame, b) for b in det_boxes]
        ally_hist = {}
        for a in alliances:
            ally_hist[str(a)] = ally_hist.get(str(a), 0) + 1

        # --- annotate: draw every class-1 box, color by conf bucket, label conf+alliance ---
        ann = frame.copy()
        # draw main-pane boundary + field polygon used by run_sync
        cv2.line(ann, (0, int(y_main)), (W, int(y_main)), (0, 255, 255), 2)
        for b, cc in zip(boxes, confs):
            x1, y1, x2, y2 = (int(z) for z in b)
            if cc >= 0.25:   color = (0, 255, 0)      # green strong
            elif cc >= 0.10: color = (0, 165, 255)    # orange mid
            else:            color = (0, 0, 255)      # red weak (<0.10)
            cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)
            al = RobotNet.bumper_alliance(frame, b)
            cv2.putText(ann, f"{cc:.2f}/{al}", (x1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        hud = (f"{v} f{fidx} pose={'V' if pose_valid else 'x'}({n_tags}t,re={reproj}) "
               f"n@.05={counts['c05']} .10={counts['c10']} .25={counts['c25']} "
               f"gate:{len(dets)}->{n_after_gate}")
        cv2.putText(ann, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(ann, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
        outp = os.path.join(OUT, f"diag_robot_{v}_f{fi}.png")
        cv2.imwrite(outp, ann)

        rec = {
            "video": v, "frame": fidx, "frac": frac,
            "counts": counts,
            "n_total_class1": len(confs),
            "confs_sorted": [round(c, 3) for c in sorted(confs, reverse=True)][:12],
            "max_conf": round(float(confs_arr.max()), 3) if confs else None,
            "pose_valid": pose_valid, "n_tags": n_tags, "reproj": reproj,
            "gate_before": len(dets), "gate_after": n_after_gate,
            "ally_hist": ally_hist,
            "png": os.path.relpath(outp, ROOT),
        }
        summary.append(rec)
        print(json.dumps(rec), flush=True)
    cap.release()

with open(os.path.join(OUT, "diag_robot_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\n=== WROTE", os.path.join(OUT, "diag_robot_summary.json"), flush=True)
