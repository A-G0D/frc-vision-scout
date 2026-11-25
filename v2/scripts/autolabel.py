"""Reusable teacher-ensemble auto-labeler for FRC Vision Scout v2.

Two labelers:
  1. BALL — YOLO11x + SAHI tiling.  COCO class 32 ('sports ball') -> YOLO class 0.
  2. ROBOT — MOG2 large-blob motion + bumper-color HSV gate.  class 1.
             NOT YOLO-World (gave 0 labels in v1).

Both write YOLO-format label files alongside each image.

Usage (programmatic, importable by Modal function and Kaggle notebook):
    from scripts.autolabel import autolabel_clip

Usage (CLI):
    python scripts/autolabel.py \\
        --clips data/footage/match.mp4 \\
        --out_dir data/dataset \\
        --step_s 0.5

GPU-only paths are guarded with try/import blocks and documented.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional

# ---------------------------------------------------------------------------
# Field-band gate — vertical fraction of frame that is field (not scoreboard /
# crowd / floor reflection).  These match the v1+v2 notebook constants.
# ---------------------------------------------------------------------------
PLAY_TOP = 0.125   # top of the play area (air column above carpet)
PLAY_BOT = 0.620   # bottom of the play area (below = pit / floor)

# Robot blob must be at least this fraction of frame area to be a robot candidate.
ROBOT_MIN_AREA_FRAC = 0.004   # ~0.4% of frame; avoids small motion noise

# Bumper-color detector thresholds (HSV).
BUMPER_SAT_MIN  = 90
BUMPER_VAL_MIN  = 60
BUMPER_RED_HUE_LO = 0
BUMPER_RED_HUE_HI = 10
BUMPER_RED_HUE_WRAP = 170   # red wraps at 180
BUMPER_BLUE_HUE_LO = 100
BUMPER_BLUE_HUE_HI = 130
BUMPER_COLOR_THRESH = 0.06   # 6% of bumper ROI pixels must be saturated red or blue


# ---------------------------------------------------------------------------
# Bumper-color helper — reused in both Kaggle notebook and Modal function
# ---------------------------------------------------------------------------

def is_bumper_color(crop_bgr) -> bool:
    """Return True if the lower-half crop of a robot candidate has strongly
    red or blue pixels (bumper color).  Works on CPU, no ML required.

    Args:
        crop_bgr: numpy array (H, W, 3) in BGR, the lower half of a blob.

    Returns:
        True if the ROI passes the bumper-color gate.
    """
    # Import guarded so the module is importable without cv2 installed.
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise RuntimeError("cv2 / numpy required for bumper_color check")

    if crop_bgr is None or crop_bgr.size == 0:
        return False
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat = (s > BUMPER_SAT_MIN) & (v > BUMPER_VAL_MIN)
    red  = (sat & ((h <= BUMPER_RED_HUE_HI) | (h >= BUMPER_RED_HUE_WRAP))).mean()
    blue = (sat & (h >= BUMPER_BLUE_HUE_LO) & (h <= BUMPER_BLUE_HUE_HI)).mean()
    return float(max(red, blue)) > BUMPER_COLOR_THRESH


# ---------------------------------------------------------------------------
# Ball labeler (YOLO11x + SAHI) — GPU-only path
# ---------------------------------------------------------------------------

def label_balls_frame(
    frame_bgr,
    ball_teacher,          # sahi.AutoDetectionModel, already loaded
    W: int,
    H: int,
    tmp_path: str = "/tmp/_autolabel_frame.png",
) -> list[str]:
    """Run SAHI sliced inference on one frame and return YOLO label lines.

    Each line: "0 cx cy w h" (normalised, class 0 = ball).
    Only detections inside the vertical field band [PLAY_TOP, PLAY_BOT] are kept.
    COCO class 32 = 'sports ball'.
    """
    try:
        import cv2
        from sahi.predict import get_sliced_prediction
    except ImportError:
        raise RuntimeError("cv2 and sahi required for ball labeling")

    cv2.imwrite(tmp_path, frame_bgr)
    sp = get_sliced_prediction(
        tmp_path, ball_teacher,
        slice_height=384, slice_width=384,
        overlap_height_ratio=0.25, overlap_width_ratio=0.25,
        verbose=0,
    )
    lines = []
    for o in sp.object_prediction_list:
        if o.category.id != 32:
            continue
        b = o.bbox
        cx = (b.minx + b.maxx) / 2
        cy = (b.miny + b.maxy) / 2
        if not (PLAY_TOP * H <= cy <= PLAY_BOT * H):
            continue
        lines.append(
            f"0 {cx/W:.6f} {cy/H:.6f} "
            f"{(b.maxx - b.minx)/W:.6f} {(b.maxy - b.miny)/H:.6f}"
        )
    return lines


# ---------------------------------------------------------------------------
# Robot labeler (MOG2 + bumper-color) — CPU
# ---------------------------------------------------------------------------

def build_robot_labeler():
    """Return a stateful callable that wraps a fresh MOG2 background subtractor.

    The returned function has signature:
        label_robots_frame(frame_bgr, W, H) -> list[str]

    Call it on EVERY frame (not just sampled frames) so the background model
    accumulates properly; only use the returned labels at the sampled frames.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise RuntimeError("cv2 / numpy required for robot labeling")

    bg = cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=30, detectShadows=False
    )
    kernel_open  = None   # lazy init (shape depends on first frame)

    def label_robots_frame(frame_bgr, W: int, H: int) -> list[str]:
        nonlocal kernel_open
        import numpy as np
        if kernel_open is None:
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        fg = bg.apply(frame_bgr)
        fg_clean = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel_open)
        # Also close small holes so large robots register as one blob.
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_CLOSE, kernel_close)

        cnts, _ = cv2.findContours(
            fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        lines = []
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < ROBOT_MIN_AREA_FRAC * W * H:
                continue
            cy_abs = y + h / 2
            if not (PLAY_TOP * H <= cy_abs <= PLAY_BOT * H):
                continue
            # Bumper color gate: check the lower half of the bounding box.
            bumper_crop = frame_bgr[y + h // 2 : y + h, x : x + w]
            if not is_bumper_color(bumper_crop):
                continue
            lines.append(
                f"1 {(x + w/2)/W:.6f} {(y + h/2)/H:.6f} {w/W:.6f} {h/H:.6f}"
            )
        return lines

    return label_robots_frame


# ---------------------------------------------------------------------------
# Main clip-level entry point
# ---------------------------------------------------------------------------

def autolabel_clip(
    clip: str,
    out_images_dir: str,
    out_labels_dir: str,
    step_s: float = 0.5,
    ball_conf: float = 0.05,
    device: str = "cuda:0",
    tmp_dir: Optional[str] = None,
    gid_start: int = 0,
    clip_idx: int = 0,
) -> list[dict]:
    """Auto-label a single clip and write YOLO dataset files.

    Args:
        clip:            Path to input .mp4 / .mkv clip.
        out_images_dir:  Where to write PNG frames.
        out_labels_dir:  Where to write YOLO .txt label files.
        step_s:          Sampling interval in seconds (default 0.5 -> ~2 fps).
        ball_conf:       SAHI confidence threshold for ball teacher.
        device:          torch device string, e.g. "cuda:0" or "cpu".
        tmp_dir:         Temp directory for intermediate SAHI image writes.
        gid_start:       Global frame index offset (for multi-clip datasets).
        clip_idx:        Index of this clip in a multi-clip dataset.

    Returns:
        List of dicts: {"clip_idx", "frame_idx", "t_s", "gid",
                        "img_path", "lbl_path", "n_balls", "n_robots"}

    NOTE: Ball labeling requires CUDA + ultralytics + sahi (GPU-only path).
          Robot labeling runs on CPU only.
          Import guards prevent crashes when running locally without ML deps.
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError("opencv-python required for autolabel_clip")

    # Ensure output dirs exist.
    Path(out_images_dir).mkdir(parents=True, exist_ok=True)
    Path(out_labels_dir).mkdir(parents=True, exist_ok=True)

    # ── load ball teacher (GPU) ──────────────────────────────────────────────
    ball_teacher = None
    try:
        from sahi import AutoDetectionModel
        ball_teacher = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path="yolo11x.pt",
            confidence_threshold=ball_conf,
            device=device,
        )
    except Exception as e:
        print(f"[autolabel] WARNING: ball teacher not loaded ({e}); ball labels will be empty")

    # ── open clip ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(clip)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open clip: {clip}")
    fps  = cap.get(cv2.CAP_PROP_FPS) or 60.0
    W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(fps / (1.0 / step_s) if step_s > 0 else fps / 2))
    # Correct: step_s is seconds-per-sample, so frames-per-sample = fps * step_s
    step = max(1, round(fps * step_s))

    tmp_png = os.path.join(tmp_dir or out_images_dir, "_sahi_tmp.png")

    # ── per-clip MOG2 robot labeler ──────────────────────────────────────────
    label_robots = build_robot_labeler()

    metadata: list[dict] = []
    gid = gid_start
    frame_idx = 0

    while True:
        ok = cap.grab()
        if not ok:
            break

        # Robot labeler runs on every frame to keep MOG2 background current.
        ok2, frame = cap.retrieve()
        if not ok2:
            frame_idx += 1
            continue

        robot_lines_now = label_robots(frame, W, H)

        if frame_idx % step == 0:
            img_path = os.path.join(out_images_dir, f"{gid:06d}.png")
            lbl_path = os.path.join(out_labels_dir, f"{gid:06d}.txt")

            # Save frame.
            cv2.imwrite(img_path, frame)

            # Ball labels (GPU, may be empty if teacher not loaded).
            ball_lines: list[str] = []
            if ball_teacher is not None:
                try:
                    ball_lines = label_balls_frame(
                        frame, ball_teacher, W, H, tmp_path=tmp_png
                    )
                except Exception as e:
                    print(f"[autolabel] ball labeling error at gid={gid}: {e}")

            # Robot labels (CPU, may be empty if no blobs pass).
            robot_lines = robot_lines_now

            all_lines = ball_lines + robot_lines
            Path(lbl_path).write_text("\n".join(all_lines))

            metadata.append({
                "clip_idx":  clip_idx,
                "frame_idx": frame_idx,
                "t_s":       frame_idx / fps,
                "gid":       gid,
                "img_path":  img_path,
                "lbl_path":  lbl_path,
                "n_balls":   len(ball_lines),
                "n_robots":  len(robot_lines),
            })
            gid += 1

            if gid % 25 == 0:
                print(f"[autolabel] {gid} frames labeled "
                      f"(balls={sum(m['n_balls'] for m in metadata)}, "
                      f"robots={sum(m['n_robots'] for m in metadata)})")

        frame_idx += 1

    cap.release()
    print(
        f"[autolabel] clip done: {gid - gid_start} frames "
        f"| balls_detected={sum(m['n_balls'] for m in metadata)} "
        f"| robots_detected={sum(m['n_robots'] for m in metadata)}"
    )
    return metadata


def autolabel_clips(
    clips: Iterable[str],
    out_dir: str,
    step_s: float = 0.5,
    ball_conf: float = 0.05,
    device: str = "cuda:0",
) -> list[dict]:
    """Autolabel multiple clips.  Images/labels go into out_dir/images/ and out_dir/labels/.

    Returns combined metadata list usable by the temporal-split step.
    """
    images_dir = os.path.join(out_dir, "images", "train")
    labels_dir = os.path.join(out_dir, "labels", "train")
    all_meta: list[dict] = []
    for ci, clip in enumerate(clips):
        meta = autolabel_clip(
            clip=clip,
            out_images_dir=images_dir,
            out_labels_dir=labels_dir,
            step_s=step_s,
            ball_conf=ball_conf,
            device=device,
            gid_start=len(all_meta),
            clip_idx=ci,
        )
        all_meta.extend(meta)
    return all_meta


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli():
    ap = argparse.ArgumentParser(description="Teacher-ensemble auto-labeler for FRC Vision Scout")
    ap.add_argument("--clips", nargs="+", required=True, help="Input .mp4 clip(s)")
    ap.add_argument("--out_dir", default="data/dataset", help="Output dataset root")
    ap.add_argument("--step_s", type=float, default=0.5, help="Sampling interval in seconds")
    ap.add_argument("--ball_conf", type=float, default=0.05, help="Ball teacher confidence threshold")
    ap.add_argument("--device", default="cuda:0", help="Torch device (cuda:0 | cpu)")
    args = ap.parse_args()

    meta = autolabel_clips(
        clips=args.clips,
        out_dir=args.out_dir,
        step_s=args.step_s,
        ball_conf=args.ball_conf,
        device=args.device,
    )
    print(f"[autolabel] total: {len(meta)} labeled frames")


if __name__ == "__main__":
    _cli()
