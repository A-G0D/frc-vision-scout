"""Robot-detection precision filter — the measured winner of the detector fleet.

The raw single-class YOLO `robot` head is bad on this broadcast (eval vs hand-labeled
ground truth: F1 0.25, precision 0.30 — most boxes are balls/people/HUB structures).
This filter, scored on the same ground truth, lifts it to **F1 0.51, precision 0.90**
(`scripts/eval_robots.py`, data/robot_gt/qual59_diag_labels.json). It keeps only boxes
that look like a real robot:

  1. bumper-colour present (RobotNet.bumper_alliance != None)  -> rejects balls/people/grey structures
  2. tighten the box to the lower bumper footprint (the net over-boxes person/structure extent)
  3. reject people / reflections / wrong shapes (floor-point below the carpet line, or tall-thin)
  4. reject the two static HUB scoring towers (fractional x/y regions)
  5. size sanity (a robot is a bounded fraction of the frame)
  6. NMS dedup

All gates are expressed as FRACTIONS of (W,H) so they are resolution-independent. The
qual59-specific hardcoded pixel lanes from the fleet prototype were intentionally dropped;
the HUB regions are fractional and hold for this near-fixed wide field shot — for other
camera framings they should come from the field model (TODO: config-drive per clip).
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from frcscout.detect.robot_net import RobotNet

NMS_IOU = 0.55
Box = Tuple[float, float, float, float]


def saturated_red_or_blue(frame_bgr: np.ndarray) -> np.ndarray:
    """Return pixels that look like saturated FRC red or blue material."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    red_lo = cv2.inRange(hsv, (0, 85, 55), (12, 255, 255))
    red_hi = cv2.inRange(hsv, (168, 85, 55), (179, 255, 255))
    blue = cv2.inRange(hsv, (92, 85, 55), (132, 255, 255))
    red = cv2.bitwise_or(red_lo, red_hi)
    mask = cv2.bitwise_or(red, blue)
    return mask != 0


def build_static_structure_mask(
    vote_counts: np.ndarray,
    frame_count: int,
    persistence: float = 0.60,
    dilate: int = 5,
) -> np.ndarray:
    """Build the canonical static red/blue structure mask from persistence votes.

    ``vote_counts`` is a per-pixel count of frames where
    :func:`saturated_red_or_blue` was true. Pixels are kept only when they were
    saturated red/blue in at least ``persistence`` of the sampled frames, then the
    same morphology used by the offline mask builder is applied.
    """
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if vote_counts.ndim != 2:
        raise ValueError("vote_counts must be a 2D per-pixel count array")

    threshold = int(math.ceil(float(persistence) * int(frame_count)))
    mask = (vote_counts >= threshold).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if dilate > 0:
        k = max(1, int(dilate))
        if k % 2 == 0:
            k += 1
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=1)
    return mask


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _overlap_frac(box, region) -> float:
    ix1, iy1 = max(box[0], region[0]), max(box[1], region[1])
    ix2, iy2 = min(box[2], region[2]), min(box[3], region[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    return inter / area


def _mask_coverage(mask: np.ndarray, rect: Box) -> float:
    h, w = mask.shape[:2]
    x1 = max(0, min(w, int(rect[0])))
    y1 = max(0, min(h, int(rect[1])))
    x2 = max(0, min(w, int(np.ceil(rect[2]))))
    y2 = max(0, min(h, int(np.ceil(rect[3]))))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = mask[y1:y2, x1:x2]
    return float(np.count_nonzero(roi)) / float(roi.size)


def _static_structure_hit(
    box: Box,
    static_mask: np.ndarray,
    W: int,
    H: int,
    box_thresh: float,
    center_thresh: float,
) -> bool:
    if static_mask.ndim == 3:
        if static_mask.shape[2] == 1:
            static_mask = static_mask[:, :, 0]
        else:
            static_mask = cv2.cvtColor(static_mask, cv2.COLOR_BGR2GRAY)
    if static_mask.shape[:2] != (H, W):
        static_mask = cv2.resize(static_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    static_mask = static_mask > 0

    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return True

    lower_box = (x1, y1 + 0.45 * bh, x2, y2)
    cx = (x1 + x2) * 0.5
    lower_center = (cx - 0.22 * bw, y2 - 0.35 * bh, cx + 0.22 * bw, y2)
    return (
        _mask_coverage(static_mask, box) >= box_thresh
        or _mask_coverage(static_mask, lower_box) >= box_thresh
        or _mask_coverage(static_mask, lower_center) >= center_thresh
    )


def tighten_box(box: Box, W: int, H: int) -> Box:
    """The net boxes a person/structure-sized extent around a robot; keep the lower
    footprint where the bumpers live and halve the side-to-side span."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) * 0.5
    anchor_y = y1 + 0.80 * h
    new_w, new_h = 0.50 * w, 0.85 * h
    ny1 = anchor_y - 0.80 * new_h
    return (max(0.0, cx - 0.5 * new_w), max(0.0, ny1),
            min(float(W), cx + 0.5 * new_w), min(float(H), ny1 + new_h))


def _is_nonrobot(box: Box, W: int, H: int) -> bool:
    """True if the tightened box is a person / reflection / HUB / wrong-shape -> reject."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return True
    # People / drive team hits: floor-point below the carpet line, or top-band
    # glass/reflection hits that are above the playable carpet in this broadcast view.
    if y2 > 0.648 * H:
        return True
    if y1 < 0.333 * H:
        return True
    if h / w > 1.80:                       # tall-thin = a standing person, not a robot
        return True
    # size sanity (resolution-independent)
    if w < 0.023 * W or w > 0.099 * W:
        return True
    if h > 0.176 * H:
        return True
    # the two large static HUB scoring towers (fractional regions; TODO config-drive)
    blue_hub = (0.19 * W, 0.30 * H, 0.34 * W, 0.52 * H)
    red_hub = (0.69 * W, 0.30 * H, 0.82 * W, 0.52 * H)
    if _overlap_frac(box, blue_hub) > 0.45 or _overlap_frac(box, red_hub) > 0.45:
        return True
    return False


def _nms(dets):
    keep = []
    for det in sorted(dets, key=lambda d: d[5], reverse=True):
        if all(_iou(det[:4], prev[:4]) < NMS_IOU for prev in keep):
            keep.append(det)
    return keep


def filter_robot_dets(frame, raw_boxes, W: int, H: int,
                      confs: Optional[Sequence[float]] = None,
                      static_mask: Optional[np.ndarray] = None,
                      static_mask_box_thresh: float = 0.80,
                      static_mask_center_thresh: float = 0.90,
                      ) -> List[Tuple[float, float, float, float, str]]:
    """Filter raw net class-1 boxes into clean robot detections.

    Parameters
    ----------
    frame : BGR ndarray
    raw_boxes : list of (x1,y1,x2,y2) — the net's class-1 boxes (already field-poly gated is fine)
    W, H : frame size
    confs : optional per-box confidence (used only for NMS ordering; defaults to 1.0)
    static_mask : optional binary mask of fixed red/blue structures, derived from video
    static_mask_*_thresh : mask coverage thresholds for lower box / lower-center reject

    Returns list of (x1,y1,x2,y2, alliance) — tightened, deduped, precision-filtered.
    """
    if confs is None:
        confs = [1.0] * len(raw_boxes)
    dets = []
    for raw, cf in zip(raw_boxes, confs):
        alliance = RobotNet.bumper_alliance(frame, raw)
        if alliance is None:               # no red/blue bumper -> ball / person / structure
            continue
        box = tighten_box(tuple(map(float, raw)), W, H)
        if _is_nonrobot(box, W, H):
            continue
        if static_mask is not None and _static_structure_hit(
            box,
            static_mask,
            W,
            H,
            static_mask_box_thresh,
            static_mask_center_thresh,
        ):
            continue
        dets.append((box[0], box[1], box[2], box[3], alliance, float(cf)))
    return [(d[0], d[1], d[2], d[3], d[4]) for d in _nms(dets)]
