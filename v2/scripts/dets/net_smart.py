from __future__ import annotations

from pathlib import Path

import numpy as np

from frcscout.detect.robot_net import RobotNet


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "weights" / "ball_openvino_model"

CONF = 0.12
IMG_SIZE = 960
NMS_IOU = 0.55

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO

        _MODEL = YOLO(str(MODEL_PATH), task="detect")
    return _MODEL


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _overlap_frac(box, region) -> float:
    ix1, iy1 = max(box[0], region[0]), max(box[1], region[1])
    ix2, iy2 = min(box[2], region[2]), min(box[3], region[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    return inter / area


def _tighten_box(box, width: int, height: int):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) * 0.5

    # The net usually sees a person/structure-sized extent around a robot.
    # Keep the lower footprint where bumpers live and halve the side-to-side span.
    anchor_y = y1 + 0.80 * h
    new_w = 0.50 * w
    new_h = 0.85 * h
    ny1 = anchor_y - 0.80 * new_h
    return (
        max(0.0, cx - 0.5 * new_w),
        max(0.0, ny1),
        min(float(width), cx + 0.5 * new_w),
        min(float(height), ny1 + new_h),
    )


def _static_hub_or_person(box, alliance, width: int, height: int) -> bool:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return True
    cx = (x1 + x2) * 0.5

    # People and drive-team/referee hits sit in the glass/reflection band or side lanes.
    if y2 > 0.648 * height:
        return True
    if y1 < 0.333 * height:
        return True
    if x1 > 0.823 * width:
        return True
    if x1 > 0.052 * width and x2 < 0.161 * width and y1 > 0.481 * height:
        return True
    if h / w > 2.35:
        return True

    # Size prior after tightening: robots in these diagnostic frames are compact.
    if w < 0.023 * width or w > 0.099 * width:
        return True
    if h > 0.176 * height:
        return True

    blue_hub = (0.19 * width, 0.30 * height, 0.34 * width, 0.52 * height)
    red_hub = (0.69 * width, 0.30 * height, 0.82 * width, 0.52 * height)
    if _overlap_frac(box, blue_hub) > 0.45 or _overlap_frac(box, red_hub) > 0.45:
        return True

    # Residual static tower/truss flashes that survive the broad hub boxes.
    if alliance == "blue" and 0.260 * width < cx < 0.365 * width and y1 < 0.491 * height and y2 < 0.574 * height:
        return True
    if alliance == "red" and 0.625 * width < cx < 0.677 * width and y2 < 0.431 * height and w < 0.031 * width:
        return True

    return False


def _nms(dets):
    keep = []
    for det in sorted(dets, key=lambda d: d[5], reverse=True):
        if all(_iou(det[:4], prev[:4]) < NMS_IOU for prev in keep):
            keep.append(det)
    return keep


def detect(frame_bgr):
    height, width = frame_bgr.shape[:2]
    result = _model().predict(
        frame_bgr,
        imgsz=IMG_SIZE,
        conf=CONF,
        device="cpu",
        verbose=False,
    )[0]

    dets = []
    if result.boxes is None:
        return []

    xyxy = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    for raw_box, cls, conf in zip(xyxy, classes, confs):
        if cls != 1:
            continue
        raw = tuple(map(float, raw_box))
        alliance = RobotNet.bumper_alliance(frame_bgr, raw)
        if alliance is None:
            continue

        box = _tighten_box(raw, width, height)
        if _static_hub_or_person(box, alliance, width, height):
            continue
        dets.append((*box, alliance, float(conf)))

    return [(x1, y1, x2, y2, alliance) for x1, y1, x2, y2, alliance, _ in _nms(dets)]
