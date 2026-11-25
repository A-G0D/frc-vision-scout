"""Score robot detection against hand-labeled ground truth (data/robot_gt/*.json).

The keystone the project lacked: a MEASURED precision/recall for robot detection, so
every detector variant is scored against truth instead of guessed. Runs a detector
function over each labeled frame, matches predictions to GT boxes by IoU, and reports
precision / recall / alliance-accuracy overall + per frame.

Detectors (select with --detector):
  net        : raw YOLO class-1 boxes (conf gate) — the OLD live behaviour
  net_poly   : + ROBOT_POLY field gate
  net_color  : + ROBOT_POLY + robot_filter (the CURRENT pipeline behaviour)
A detector is `frame_bgr -> [(x1,y1,x2,y2,alliance_or_None), ...]`.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from frcscout.detect.robot_filter import filter_robot_dets  # noqa: E402

GT = ROOT / "data/robot_gt/qual59_diag_labels.json"
FRAMES_DIR = ROOT / "matrix_org/robot_diag"
STATIC_MASK = ROOT / "data/robot_gt/qual59_static_mask.png"
ROBOT_POLY = np.array([(0.02, 0.25), (0.98, 0.25), (1.00, 0.64), (0.00, 0.64)], np.float32)
_MODEL = None
_STATIC_MASK = None
_USE_STATIC_MASK = True


def _model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO(str(ROOT / "weights/ball_openvino_model"), task="detect")
    return _MODEL


def _net_boxes(frame, conf=0.12, return_conf=False):
    r = _model().predict(frame, imgsz=960, conf=conf, device="cpu", verbose=False)[0]
    out = []
    scores = []
    if r.boxes is not None:
        xy = r.boxes.xyxy.cpu().numpy()
        cl = r.boxes.cls.cpu().numpy().astype(int)
        cf = r.boxes.conf.cpu().numpy()
        for b, c, score in zip(xy, cl, cf):
            if c == 1:
                out.append(tuple(map(float, b)))
                scores.append(float(score))
    if return_conf:
        return out, scores
    return out


def _static_mask():
    global _STATIC_MASK
    if not _USE_STATIC_MASK or not STATIC_MASK.exists():
        return None
    if _STATIC_MASK is None:
        _STATIC_MASK = cv2.imread(str(STATIC_MASK), cv2.IMREAD_GRAYSCALE)
        if _STATIC_MASK is None:
            raise RuntimeError(f"Could not load static mask: {STATIC_MASK}")
    return _STATIC_MASK


def _in_poly(box, W, H):
    cx, y2 = (box[0] + box[2]) / 2.0, box[3]
    poly = (ROBOT_POLY * [W, H]).astype(np.float32)
    return cv2.pointPolygonTest(poly, (float(cx), float(y2)), False) >= 0


def det_net(frame):
    return [(b[0], b[1], b[2], b[3], None) for b in _net_boxes(frame)]


def det_net_poly(frame):
    H, W = frame.shape[:2]
    return [(b[0], b[1], b[2], b[3], None) for b in _net_boxes(frame) if _in_poly(b, W, H)]


def det_net_color(frame):
    H, W = frame.shape[:2]
    boxes, confs = _net_boxes(frame, return_conf=True)
    gated = [(b, c) for b, c in zip(boxes, confs) if _in_poly(b, W, H)]
    if not gated:
        return []
    raw, raw_confs = zip(*gated)
    # _static_mask() returns None when --no-static-mask (cross-clip) — single toggle path.
    return filter_robot_dets(frame, raw, W, H, confs=raw_confs, static_mask=_static_mask())


DETECTORS = {"net": det_net, "net_poly": det_net_poly, "net_color": det_net_color}


def _load_detector_module(path):
    """Load a detector from a .py file exposing `detect(frame_bgr) -> [(x1,y1,x2,y2,alliance|None), ...]`.
    Lets each fleet agent drop in its own variant and self-score without touching this file.
    The module may import `_net_boxes`, `_in_poly`, `_model` from this module (scripts on path)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_fleet_det", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.detect


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="net_color", choices=list(DETECTORS))
    ap.add_argument("--detector-module", default=None,
                    help="path to a .py exposing detect(frame_bgr)->[(x1,y1,x2,y2,alliance|None)]")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--no-static-mask", action="store_true",
                    help="disable auto-loading data/robot_gt/qual59_static_mask.png")
    ap.add_argument("--gt", default=None, help="GT json path (default qual59)")
    ap.add_argument("--frames-dir", default=None, help="frames dir (default qual59 diag)")
    args = ap.parse_args()
    global _USE_STATIC_MASK
    _USE_STATIC_MASK = not args.no_static_mask
    if args.detector_module:
        det = _load_detector_module(args.detector_module)
        args.detector = Path(args.detector_module).stem
    else:
        det = DETECTORS[args.detector]
    gt_path = Path(args.gt) if args.gt else GT
    frames_dir = Path(args.frames_dir) if args.frames_dir else FRAMES_DIR
    gt = json.loads(gt_path.read_text())["frames"]

    TP = FP = FN = 0
    alliance_ok = alliance_tot = 0
    rows = []
    for fname, gboxes in gt.items():
        fp = frames_dir / fname
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        preds = det(frame)
        gused = [False] * len(gboxes)
        ftp = ffp = 0
        for p in preds:
            best, bi = args.iou, -1
            for i, g in enumerate(gboxes):
                if gused[i]:
                    continue
                v = iou(p[:4], g[:4])
                if v >= best:
                    best, bi = v, i
            if bi >= 0:
                gused[bi] = True
                ftp += 1
                if p[4] is not None:
                    alliance_tot += 1
                    if p[4] == gboxes[bi][4]:
                        alliance_ok += 1
            else:
                ffp += 1
        ffn = gused.count(False)
        TP += ftp; FP += ffp; FN += ffn
        rows.append((fname, len(gboxes), len(preds), ftp, ffp, ffn))

    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"=== robot eval: detector={args.detector} iou>={args.iou} ===")
    print(f"frames={len(rows)}  GT_boxes={TP+FN}  preds={TP+FP}")
    print(f"TP={TP} FP={FP} FN={FN}")
    print(f"PRECISION={prec:.3f}  RECALL={rec:.3f}  F1={f1:.3f}")
    if alliance_tot:
        print(f"alliance accuracy (on TPs w/ a colour)={alliance_ok}/{alliance_tot}={alliance_ok/alliance_tot:.3f}")
    worst = sorted(rows, key=lambda r: (r[4] + r[5]), reverse=True)[:6]
    print("worst frames (FP+FN):  frame  gt preds tp fp fn")
    for r in worst:
        print(f"   {r[0]:10s} {r[1]:3d} {r[2]:4d} {r[3]:3d} {r[4]:3d} {r[5]:3d}")


if __name__ == "__main__":
    main()
