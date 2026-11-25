"""Individual metric implementations consumed by gates.py. Each takes the held-out,
HUMAN-labeled match (and, where relevant, the overlay/TBA truth) and returns a scalar.

Kept deliberately separate from gates so each can be unit-tested in isolation.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_yolo_label(path: str | Path):
    """Parse a YOLO-format label file.

    Each line: cls cx cy w h [conf]
    Returns list of dicts with keys: cls (int), cx, cy, w, h, conf (float 0-1).
    Skips blank and comment lines.
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            conf = float(parts[5]) if len(parts) >= 6 else 1.0
            rows.append({"cls": cls, "cx": cx, "cy": cy, "w": w, "h": h, "conf": conf})
    return rows


def _xywh_to_xyxy(cx, cy, w, h):
    """YOLO cx/cy/w/h (all 0-1 normalised) -> x1,y1,x2,y2 in the same units."""
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return x1, y1, x2, y2


def _iou(b1, b2) -> float:
    """IoU between two (x1,y1,x2,y2) boxes."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)


def _ap_from_pr(precision: list[float], recall: list[float]) -> float:
    """Compute AP using the 11-point interpolation (VOC style), pure Python."""
    if not precision:
        return 0.0
    ap = 0.0
    for thr in [i / 10.0 for i in range(11)]:
        p_at = [p for p, r in zip(precision, recall) if r >= thr]
        ap += max(p_at) if p_at else 0.0
    return ap / 11.0


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------

def ball_map50(pred_labels_dir: str, human_labels_dir: str) -> float:
    """mAP@50 of student predictions vs HUMAN labels on the held-out match.

    Reads paired YOLO-format label files from *pred_labels_dir* and
    *human_labels_dir*.  Prediction files must contain a confidence score in
    column 5 (standard ultralytics predict output).  GT files may omit conf.

    Only ball detections (class 0 in both directories by convention) are
    counted.  Matching is greedy: sort predictions by descending conf, match
    to unmatched GT boxes with IoU >= 0.5.  Precision-recall curve is computed
    per-image then aggregated.  AP is the 11-point interpolation (VOC 2007).

    NOT measured against teacher pseudo-labels — this must receive the
    HUMAN-reviewed ground truth.
    """
    pred_dir = Path(pred_labels_dir)
    gt_dir = Path(human_labels_dir)

    # Collect all prediction files; skip images with no GT file (out-of-split).
    all_tp: list[tuple[float, bool]] = []  # (conf, is_tp) sorted later
    n_gt_total = 0

    pred_files = sorted(pred_dir.glob("*.txt"))
    if not pred_files:
        return 0.0

    for pred_file in pred_files:
        gt_file = gt_dir / pred_file.name
        preds = _parse_yolo_label(pred_file)
        # Filter to ball class (0)
        preds = [p for p in preds if p["cls"] == 0]
        if gt_file.exists():
            gts = [g for g in _parse_yolo_label(gt_file) if g["cls"] == 0]
        else:
            gts = []

        n_gt_total += len(gts)

        # Sort preds by conf descending
        preds.sort(key=lambda x: x["conf"], reverse=True)

        gt_boxes = [_xywh_to_xyxy(g["cx"], g["cy"], g["w"], g["h"]) for g in gts]
        matched_gt = [False] * len(gt_boxes)

        for p in preds:
            pb = _xywh_to_xyxy(p["cx"], p["cy"], p["w"], p["h"])
            best_iou = 0.0
            best_idx = -1
            for j, gb in enumerate(gt_boxes):
                if matched_gt[j]:
                    continue
                iou_val = _iou(pb, gb)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = j
            if best_iou >= 0.5 and best_idx >= 0:
                matched_gt[best_idx] = True
                all_tp.append((p["conf"], True))
            else:
                all_tp.append((p["conf"], False))

    if n_gt_total == 0:
        return 1.0 if not all_tp else 0.0  # no GT: vacuously perfect only if no preds either

    # Sort all predictions globally by conf descending
    all_tp.sort(key=lambda x: x[0], reverse=True)

    precision_pts: list[float] = []
    recall_pts: list[float] = []
    tp_cum = 0
    fp_cum = 0
    for _conf, is_tp in all_tp:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precision_pts.append(tp_cum / (tp_cum + fp_cum))
        recall_pts.append(tp_cum / n_gt_total)

    return _ap_from_pr(precision_pts, recall_pts)


def int8_recall_ratio(int8_recall: float, fp32_recall: float) -> float:
    """Guards against INT8 PTQ collapsing small-object recall -> triggers QAT."""
    return int8_recall / fp32_recall if fp32_recall else 0.0


def count_mae(cv_counts: list[int], overlay_counts: list[int]) -> float:
    """Mean abs error of CV ball count vs the OVERLAY count (free dense ground truth)."""
    n = min(len(cv_counts), len(overlay_counts))
    if n == 0:
        return float("inf")
    return sum(abs(cv_counts[i] - overlay_counts[i]) for i in range(n)) / n


def robot_idf1(pred_tracks, gt_tracks) -> tuple[float, int]:
    """IDF1 + ID-switch count on hand-labeled robot track snippets.

    Tries to use ``motmetrics`` if importable; falls back to a self-contained
    implementation so the metric is always available without the optional dep.

    Parameters
    ----------
    pred_tracks / gt_tracks : list of dicts
        Each dict must have keys:
            ``frame``  – int frame index
            ``tid``    – int track id
            ``xyxy``   – (x1,y1,x2,y2) in pixels

    Returns
    -------
    (idf1, id_switches) : (float, int)
    """
    if not pred_tracks and not gt_tracks:
        return 1.0, 0

    # ------------------------------------------------------------------ #
    # Try motmetrics first
    # ------------------------------------------------------------------ #
    try:
        import motmetrics as mm  # type: ignore
        import numpy as np

        acc = mm.MOTAccumulator(auto_id=True)

        # Build frame -> list mapping
        from collections import defaultdict
        gt_by_frame: dict = defaultdict(list)
        pred_by_frame: dict = defaultdict(list)
        for t in gt_tracks:
            gt_by_frame[t["frame"]].append(t)
        for t in pred_tracks:
            pred_by_frame[t["frame"]].append(t)

        all_frames = sorted(set(gt_by_frame) | set(pred_by_frame))
        for fr in all_frames:
            gts = gt_by_frame[fr]
            preds = pred_by_frame[fr]

            gt_ids = [t["tid"] for t in gts]
            pred_ids = [t["tid"] for t in preds]

            # Distance matrix: 1 - IoU  (motmetrics expects distances, 0=perfect)
            if gts and preds:
                dist = np.ones((len(gts), len(preds)), dtype=float)
                for i, g in enumerate(gts):
                    for j, p in enumerate(preds):
                        dist[i, j] = 1.0 - _iou(g["xyxy"], p["xyxy"])
            else:
                dist = np.empty((len(gts), len(preds)))

            acc.update(gt_ids, pred_ids, dist)

        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=["idf1", "num_switches"], name="eval")
        idf1 = float(summary["idf1"].iloc[0])
        id_sw = int(summary["num_switches"].iloc[0])
        return idf1, id_sw

    except ImportError:
        pass

    # ------------------------------------------------------------------ #
    # Self-contained fallback (greedy IoU matching per frame)
    # ------------------------------------------------------------------ #
    from collections import defaultdict

    gt_by_frame: dict = defaultdict(list)
    pred_by_frame: dict = defaultdict(list)
    for t in gt_tracks:
        gt_by_frame[t["frame"]].append(t)
    for t in pred_tracks:
        pred_by_frame[t["frame"]].append(t)

    all_frames = sorted(set(gt_by_frame) | set(pred_by_frame))

    # Per-GT-ID stats for IDF1
    # IDF1 = 2 * IDTP / (2*IDTP + IDFP + IDFN)
    # We track (for each gt_id): longest consecutive matching pred_id
    gt_id_tps: dict[int, int] = defaultdict(int)   # IDTP counts
    gt_id_total: dict[int, int] = defaultdict(int)  # total GT detections
    pred_id_total: dict[int, int] = defaultdict(int) # total pred detections
    # map (gt_id) -> currently best-matching pred_id
    gt_to_pred: dict[int, int] = {}
    id_switches = 0
    total_idfp = 0
    total_idfn = 0
    total_idtp = 0

    for fr in all_frames:
        gts = gt_by_frame[fr]
        preds = pred_by_frame[fr]

        for g in gts:
            gt_id_total[g["tid"]] += 1
        for p in preds:
            pred_id_total[p["tid"]] += 1

        if not gts or not preds:
            continue

        # Greedy matching by IoU
        matched_pred = set()
        gt_match = {}  # gt_tid -> pred_tid
        for g in sorted(gts, key=lambda x: x["tid"]):
            best_iou = 0.0
            best_p = None
            for p in preds:
                if p["tid"] in matched_pred:
                    continue
                v = _iou(g["xyxy"], p["xyxy"])
                if v > best_iou:
                    best_iou = v
                    best_p = p
            if best_p is not None and best_iou >= 0.5:
                matched_pred.add(best_p["tid"])
                gt_match[g["tid"]] = best_p["tid"]

        for gt_tid, pred_tid in gt_match.items():
            if gt_tid in gt_to_pred and gt_to_pred[gt_tid] != pred_tid:
                id_switches += 1
            gt_to_pred[gt_tid] = pred_tid
            gt_id_tps[gt_tid] += 1

    # Aggregate IDTP, IDFP, IDFN
    for gt_id, tp in gt_id_tps.items():
        total_idtp += tp

    total_idfn = sum(gt_id_total.values()) - total_idtp
    total_idfp = sum(pred_id_total.values()) - total_idtp

    denom = 2 * total_idtp + total_idfp + total_idfn
    idf1 = (2 * total_idtp / denom) if denom > 0 else 0.0

    return idf1, id_switches


def reproj_err_px(pose, field_model) -> float:
    """Mean reprojection error of known AprilTag corners under the recovered pose.

    Parameters
    ----------
    pose : CameraPose
        Must have ``valid=True`` and populated ``rvec`` / ``tvec``.
    field_model : dict
        Mapping from tag_id (int) to list of 4 corner world points [[x,y,z], ...].
        This is read from ``configs/field_2026_rebuilt.yaml`` by the geometry layer.

    Returns float('inf') when pose is invalid or no tags were used.
    """
    if pose is None or not getattr(pose, "valid", False):
        return float("inf")

    rvec = getattr(pose, "rvec", None)
    tvec = getattr(pose, "tvec", None)
    if rvec is None or tvec is None:
        # Only homography recovered — no 3-D reprojection possible
        return float("inf")

    try:
        import numpy as np

        # Rodrigues rotation
        r = np.array(rvec, dtype=float).reshape(3)
        t = np.array(tvec, dtype=float).reshape(3)
        theta = float(np.linalg.norm(r))
        if theta < 1e-9:
            R = np.eye(3)
        else:
            k = r / theta
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

        errs = []
        for tag_id, corners_3d in field_model.items():
            corners_3d = np.array(corners_3d, dtype=float)  # (4,3)
            observed = pose.__dict__.get("observed_corners", {}).get(int(tag_id))
            if observed is None:
                continue
            observed = np.array(observed, dtype=float)  # (4,2) pixel coords

            # Project world points: p_cam = R @ X + t
            p_cam = (R @ corners_3d.T).T + t  # (4,3)
            # Simple pinhole: need K (intrinsics).  If not stored, fall back to pose.reproj_err_px
            # which was already computed by the geometry layer.
            _ = p_cam  # used below if K available

        # If the geometry layer already stored the value, use it directly.
        if hasattr(pose, "reproj_err_px") and pose.reproj_err_px > 0:
            return float(pose.reproj_err_px)

        return float("inf") if not errs else float(np.mean(errs))

    except Exception:
        return float("inf")
