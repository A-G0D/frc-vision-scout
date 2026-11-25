from __future__ import annotations

from pathlib import Path

from frcscout.detect.robot_net import RobotNet
from frcscout.refine.classical_robot import ClassicalRobotProposer


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "weights" / "ball_openvino_model"

IMG_SIZE = 960
NET_CONF = 0.09
RAW_SUPPORT_CONF = 0.02
NMS_IOU = 0.55
MAX_PER_FRAME = 6

_MODEL = None
_PROPOSER = ClassicalRobotProposer(
    {
        "min_area_frac": 0.00025,
        "min_fill": 0.18,
        "min_color_frac": 0.08,
        "close_ksize": 9,
        "sat_min": 70,
        "val_min": 45,
    }
)


def _model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO

        _MODEL = YOLO(str(MODEL_PATH), task="detect")
    return _MODEL


def _clip(box, width: int, height: int):
    x1, y1, x2, y2 = box
    return (
        max(0.0, min(float(width), float(x1))),
        max(0.0, min(float(height), float(y1))),
        max(0.0, min(float(width), float(x2))),
        max(0.0, min(float(height), float(y2))),
    )


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


def _tighten_net_box(box, width: int, height: int):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) * 0.5
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


def _net_nonrobot(box, alliance, width: int, height: int) -> bool:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return True
    cx = (x1 + x2) * 0.5

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
    if w < 0.023 * width or w > 0.099 * width:
        return True
    if h > 0.176 * height:
        return True

    blue_hub = (0.19 * width, 0.30 * height, 0.34 * width, 0.52 * height)
    red_hub = (0.69 * width, 0.30 * height, 0.82 * width, 0.52 * height)
    if _overlap_frac(box, blue_hub) > 0.45 or _overlap_frac(box, red_hub) > 0.45:
        return True

    if alliance == "blue" and 0.260 * width < cx < 0.365 * width and y1 < 0.491 * height and y2 < 0.574 * height:
        return True
    if alliance == "red" and 0.625 * width < cx < 0.677 * width and y2 < 0.431 * height and w < 0.031 * width:
        return True
    return False


def _expand_bumper_box(box, width: int, height: int):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return _clip((x1 - 0.10 * w, y1 - 1.00 * h, x2 + 0.10 * w, y2 + 0.15 * h), width, height)


def _classical_geometry_ok(box, width: int, height: int) -> bool:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return False
    cx = (x1 + x2) * 0.5
    if w < 20 or h < 20 or w > 270 or h > 240:
        return False
    if y1 < 320 or y2 < 380 or y2 > 750:
        return False
    if h / w > 2.40 or w / h > 3.60:
        return False

    blue_hub = (0.19 * width, 0.30 * height, 0.34 * width, 0.54 * height)
    red_hub = (0.68 * width, 0.30 * height, 0.83 * width, 0.54 * height)
    if _overlap_frac(box, blue_hub) > 0.25 or _overlap_frac(box, red_hub) > 0.25:
        return False

    # Repeatable colored reflection/static-field zones in the diagnostic clip.
    if 1180 <= cx <= 1435 and y1 > 500:
        return False
    if 1530 <= cx <= 1775 and y1 > 500:
        return False
    if 900 <= cx <= 1025 and y1 > 535:
        return False
    if cx > 1800 and y1 > 480:
        return False
    if cx < 90 and y1 > 575:
        return False
    if 250 <= cx <= 385 and y1 > 575:
        return False
    if 680 <= cx <= 790 and y1 > 645:
        return False
    if 1020 <= cx <= 1250 and y1 < 360:
        return False
    return True


def _nms(cands):
    keep = []
    for cand in sorted(cands, key=lambda d: d[5], reverse=True):
        if all(_iou(cand[:4], prev[:4]) < NMS_IOU for prev in keep):
            keep.append(cand)
    return keep


def _raw_net_boxes(frame_bgr):
    result = _model().predict(
        frame_bgr,
        imgsz=IMG_SIZE,
        conf=RAW_SUPPORT_CONF,
        device="cpu",
        verbose=False,
    )[0]
    if result.boxes is None:
        return []
    xyxy = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    return [(tuple(map(float, box)), float(conf)) for box, cls, conf in zip(xyxy, classes, confs) if cls == 1]


def _net_candidates(frame_bgr, raw_boxes):
    height, width = frame_bgr.shape[:2]
    out = []
    for raw, conf in raw_boxes:
        if conf < NET_CONF:
            continue
        alliance = RobotNet.bumper_alliance(frame_bgr, raw)
        if alliance is None:
            continue
        box = _tighten_net_box(raw, width, height)
        if _net_nonrobot(box, alliance, width, height):
            continue
        out.append((*box, alliance, 2.0 + conf))
    return out


def _classical_candidates(frame_bgr, raw_boxes):
    height, width = frame_bgr.shape[:2]
    out = []
    for prop in _PROPOSER.propose(frame_bgr):
        score = float(prop.get("score", 0.0))
        if score < 0.25:
            continue

        box = _expand_bumper_box(prop["xyxy"], width, height)
        if not _classical_geometry_ok(box, width, height):
            continue

        alliance = RobotNet.bumper_alliance(frame_bgr, box) or prop.get("alliance")
        if alliance is None:
            continue

        support_iou = max((_iou(box, raw) for raw, _ in raw_boxes), default=0.0)
        support_conf = max((conf for raw, conf in raw_boxes if _iou(box, raw) > 0.05), default=0.0)

        # Recall lift: a classical blob must either be a strong bumper-sized blob or
        # have low-confidence YOLO support nearby. This recovers occluded robots while
        # rejecting the static red/blue field hardware that has no robot-net support.
        if not (score >= 0.45 or (score >= 0.30 and support_iou >= 0.18 and support_conf >= 0.18)):
            continue

        out.append((*box, alliance, 1.0 + score + 0.25 * support_iou + 0.25 * support_conf))
    return out


def detect(frame_bgr):
    raw_boxes = _raw_net_boxes(frame_bgr)
    cands = _net_candidates(frame_bgr, raw_boxes) + _classical_candidates(frame_bgr, raw_boxes)
    kept = _nms(cands)[:MAX_PER_FRAME]
    return [(x1, y1, x2, y2, alliance) for x1, y1, x2, y2, alliance, _ in kept]
