"""Active-learning loop: train -> mine -> relabel -> retrain -> distill.

Pipeline:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  PHASE 1 — BOOTSTRAP (one pass, human/teacher labels only)          │
  │    autolabel -> temporal split -> train_student (YOLO11s ball,      │
  │    YOLO11n robot)                                                   │
  │                                                                     │
  │  PHASE 2 — ACTIVE-LEARNING LOOP  (repeat until plateau)            │
  │    for each round r:                                                │
  │      1. run student on unlabeled frames (mine_unlabeled)            │
  │      2. compare student vs teacher -> find disagreements &          │
  │         low-confidence detections (mine_disagreements)              │
  │      3. save mined frames to review queue (queue_for_review)        │
  │      4. teacher auto-labels the queue frames (autolabel_mined)      │
  │      5. retrain student with base + mined data (train_one)          │
  │      6. measure val mAP; stop when improvement < epsilon            │
  │                                                                     │
  │  PHASE 3 — DISTILLATION  (runs after PHASE 2 plateau)              │
  │    teacher (large model, e.g. YOLO11x) generates soft-label         │
  │    training signals for the small CPU students. Each student is     │
  │    retrained with KD loss.  Deploy targets:                         │
  │      ball student:  YOLO11s @ 960px  (Arc iGPU, 5Hz)               │
  │      robot student: YOLO11n @ 512px  (NPU, every frame)            │
  └─────────────────────────────────────────────────────────────────────┘

Usage (import, e.g. from modal_app.py):
    from scripts.active_learning import run_active_learning_loop
    run_active_learning_loop(cfg)

Usage (CLI):
    python scripts/active_learning.py \\
        --ball_cfg  configs/train_ball.yaml \\
        --robot_cfg configs/train_robot.yaml \\
        --data_root data/dataset \\
        --unlabeled data/unlabeled_frames \\
        --out       runs/active_learning \\
        --teacher   yolo11x.pt \\
        --max_rounds 10 \\
        --epsilon 0.002

GPU-ONLY paths are guarded and documented. The module is import-clean without ML deps.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ActiveLearningConfig:
    """All settings for one complete active-learning + distillation run.

    GPU-ONLY note: all training/inference fields are unused until Phase 2/3;
    the dataclass itself can be instantiated without ML deps.
    """
    # ── data paths ───────────────────────────────────────────────────────────
    data_root: str = "data/dataset"           # YOLO dataset root (has train/val)
    unlabeled_dir: str = "data/unlabeled"     # frames not yet in any split
    out_root: str = "runs/active_learning"    # per-round output directories

    # ── model configs ────────────────────────────────────────────────────────
    ball_cfg: str  = "configs/train_ball.yaml"
    robot_cfg: str = "configs/train_robot.yaml"

    # ── teacher model ────────────────────────────────────────────────────────
    teacher_weights: str = "yolo11x.pt"       # large slow model; runs in cloud only

    # ── loop control ─────────────────────────────────────────────────────────
    max_rounds: int   = 10                    # hard cap on AL iterations
    epsilon: float    = 0.002                 # stop when mAP gain < epsilon
    max_mine: int     = 500                   # max frames to mine per round
    conf_low: float   = 0.15                  # below this -> low-confidence (mine it)
    conf_high: float  = 0.6                   # above this -> skip (student is sure)
    iou_disagree: float = 0.3                 # IoU with teacher below this = disagreement

    # ── distillation ─────────────────────────────────────────────────────────
    distill: bool = True                      # run Phase 3 after plateau

    # ── misc ─────────────────────────────────────────────────────────────────
    device: int   = 0                         # CUDA device index
    step_s: float = 1.0                       # unlabeled sampling interval (seconds)
    history: list = field(default_factory=list)  # per-round metrics log


# ---------------------------------------------------------------------------
# Phase 2a — Mine disagreements / low-confidence from unlabeled frames
# ---------------------------------------------------------------------------

def mine_unlabeled(
    student_weights: str,
    unlabeled_dir: str,
    teacher_weights: str,
    out_dir: str,
    conf_low: float = 0.15,
    iou_disagree: float = 0.3,
    max_mine: int = 500,
    device: int = 0,
) -> list[str]:
    """Run student + teacher on unlabeled frames; find hard/uncertain ones.

    Mining strategy (in order of priority):
      1. Frames where the student and teacher disagree (IoU < iou_disagree).
      2. Frames where the student has low-confidence detections (< conf_low).
      3. Frames where the teacher finds detections but the student finds none.

    Args:
        student_weights:  Path to student best.pt (or OpenVINO dir).
        unlabeled_dir:    Directory of PNG/JPG frames to mine.
        teacher_weights:  Path to teacher weights (or "yolo11x.pt" auto-download).
        out_dir:          Where to copy the mined frames.
        conf_low:         Student confidence below this is "uncertain".
        iou_disagree:     Max IoU for student/teacher box pair to count as disagreement.
        max_mine:         Hard limit on number of frames to mine per round.
        device:           CUDA device index.

    Returns:
        List of frame paths that were selected for review.

    GPU-ONLY: requires ultralytics.
    """
    try:
        from ultralytics import YOLO
        import numpy as np
    except ImportError:
        raise RuntimeError("[active_learning] ultralytics/numpy required for mining.")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    student = YOLO(student_weights)
    teacher = YOLO(teacher_weights)

    frame_paths = sorted(
        p for p in Path(unlabeled_dir).iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not frame_paths:
        print(f"[mine] no frames found in {unlabeled_dir}")
        return []

    mined: list[str] = []
    scores: list[tuple[float, str]] = []  # (priority_score, path)

    for fpath in frame_paths:
        if len(mined) >= max_mine:
            break

        fpath_str = str(fpath)

        # Student inference.
        s_res = student.predict(fpath_str, conf=0.05, verbose=False)
        s_boxes = (
            s_res[0].boxes.xyxyn.cpu().numpy()
            if s_res and s_res[0].boxes is not None and len(s_res[0].boxes) > 0
            else np.empty((0, 4))
        )
        s_confs = (
            s_res[0].boxes.conf.cpu().numpy()
            if s_res and s_res[0].boxes is not None and len(s_res[0].boxes) > 0
            else np.empty(0)
        )

        # Teacher inference.
        t_res = teacher.predict(fpath_str, conf=0.05, verbose=False)
        t_boxes = (
            t_res[0].boxes.xyxyn.cpu().numpy()
            if t_res and t_res[0].boxes is not None and len(t_res[0].boxes) > 0
            else np.empty((0, 4))
        )

        # Score this frame by how uncertain / disagreeing it is.
        priority = 0.0

        # Low-confidence student detections.
        if len(s_confs) > 0:
            low_conf_count = int((s_confs < conf_low).sum())
            priority += low_conf_count * 1.0

        # Teacher finds objects the student misses completely.
        if len(t_boxes) > 0 and len(s_boxes) == 0:
            priority += len(t_boxes) * 2.0

        # Boxes present but IoU disagrees.
        if len(t_boxes) > 0 and len(s_boxes) > 0:
            disagree_count = _count_disagreements(s_boxes, t_boxes, iou_disagree)
            priority += disagree_count * 1.5

        if priority > 0:
            scores.append((priority, fpath_str))

    # Sort by priority descending, take top max_mine.
    scores.sort(key=lambda x: -x[0])
    for _, fpath_str in scores[:max_mine]:
        dst = os.path.join(out_dir, Path(fpath_str).name)
        shutil.copy2(fpath_str, dst)
        mined.append(dst)

    print(f"[mine] mined {len(mined)} frames from {len(frame_paths)} unlabeled "
          f"(max_mine={max_mine})")
    return mined


def _count_disagreements(
    s_boxes,   # numpy (N, 4) normalised xyxy
    t_boxes,   # numpy (M, 4) normalised xyxy
    iou_threshold: float,
) -> int:
    """Count teacher boxes with no matching student box above iou_threshold."""
    import numpy as np

    disagree = 0
    for tb in t_boxes:
        best_iou = max(
            (_iou(tb, sb) for sb in s_boxes), default=0.0
        )
        if best_iou < iou_threshold:
            disagree += 1
    return disagree


def _iou(a, b) -> float:
    """Compute IoU between two normalised xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (ax2 - ax1) * (ay2 - ay1)
    ub = (bx2 - bx1) * (by2 - by1)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Phase 2b — Auto-label the mined frames (teacher ensemble)
# ---------------------------------------------------------------------------

def autolabel_mined(
    mined_frames: list[str],
    out_images_dir: str,
    out_labels_dir: str,
    device: str = "cuda:0",
    ball_conf: float = 0.05,
) -> list[str]:
    """Run the teacher-ensemble auto-labeler on the mined frames.

    Delegates to scripts/autolabel.py's per-frame labelers.  Each mined frame
    is a still image (no video context), so robot MOG2 runs with a single-frame
    background model (less accurate; label quality is reviewed in the queue).

    Returns list of label file paths written.

    GPU-ONLY for ball labels; robot labels run on CPU.
    """
    # Import autolabel from sibling module.
    try:
        from scripts.autolabel import (
            label_balls_frame,
            build_robot_labeler,
        )
    except ImportError:
        try:
            # Allow running from the repo root.
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from scripts.autolabel import label_balls_frame, build_robot_labeler
        except ImportError:
            raise RuntimeError("[active_learning] cannot import scripts.autolabel")

    try:
        import cv2
        from sahi import AutoDetectionModel
    except ImportError:
        raise RuntimeError("[active_learning] cv2/sahi required for autolabeling")

    Path(out_images_dir).mkdir(parents=True, exist_ok=True)
    Path(out_labels_dir).mkdir(parents=True, exist_ok=True)

    ball_teacher = None
    try:
        ball_teacher = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path="yolo11x.pt",
            confidence_threshold=ball_conf,
            device=device,
        )
    except Exception as e:
        print(f"[autolabel_mined] ball teacher not loaded: {e}")

    # Single-frame robot labeler (no temporal context).
    label_robots = build_robot_labeler()
    label_files: list[str] = []

    for i, fpath in enumerate(mined_frames):
        frame = cv2.imread(fpath)
        if frame is None:
            continue
        H, W = frame.shape[:2]
        stem  = Path(fpath).stem
        img_p = os.path.join(out_images_dir, f"{stem}.png")
        lbl_p = os.path.join(out_labels_dir, f"{stem}.txt")

        # Copy image.
        cv2.imwrite(img_p, frame)

        ball_lines: list[str] = []
        if ball_teacher:
            try:
                ball_lines = label_balls_frame(frame, ball_teacher, W, H, tmp_path=img_p)
            except Exception as e:
                print(f"[autolabel_mined] ball error on {fpath}: {e}")

        robot_lines = label_robots(frame, W, H)

        Path(lbl_p).write_text("\n".join(ball_lines + robot_lines))
        label_files.append(lbl_p)

        if (i + 1) % 50 == 0:
            print(f"[autolabel_mined] {i+1}/{len(mined_frames)} frames labeled")

    print(f"[autolabel_mined] {len(label_files)} label files written")
    return label_files


# ---------------------------------------------------------------------------
# Phase 2c — Queue for human / Claude vision review
# ---------------------------------------------------------------------------

def queue_for_review(
    mined_frames: list[str],
    review_dir: str,
    round_idx: int,
) -> str:
    """Move mined frames + draft labels into a review queue directory.

    The review queue follows the layout expected by scripts/label_review.py.
    After review, the corrected labels are merged back into the main dataset.

    Returns the review directory path.
    """
    q = os.path.join(review_dir, f"round_{round_idx:02d}")
    Path(q).mkdir(parents=True, exist_ok=True)
    for fp in mined_frames:
        dst = os.path.join(q, Path(fp).name)
        if not os.path.exists(dst):
            shutil.copy2(fp, dst)
    # Write a manifest for label_review.py.
    manifest = [os.path.basename(f) for f in mined_frames]
    Path(os.path.join(q, "manifest.json")).write_text(
        json.dumps({"round": round_idx, "frames": manifest}, indent=2)
    )
    print(f"[queue] round {round_idx}: {len(mined_frames)} frames queued in {q}")
    return q


# ---------------------------------------------------------------------------
# Phase 2 — Val mAP measurement
# ---------------------------------------------------------------------------

def measure_val_map(weights: str, data: str, imgsz: int, device: int = 0) -> float:
    """Run YOLO val and return mAP50-95.

    Returns -1.0 on failure (so the loop doesn't stop prematurely).

    GPU-ONLY.
    """
    try:
        from ultralytics import YOLO
        m = YOLO(weights)
        metrics = m.val(data=data, imgsz=imgsz, device=device, verbose=False)
        return float(metrics.box.map)
    except Exception as e:
        print(f"[active_learning] val mAP measurement failed: {e}")
        return -1.0


# ---------------------------------------------------------------------------
# Phase 3 — Knowledge Distillation
# ---------------------------------------------------------------------------

def distill_student(
    teacher_weights: str,
    data: str,
    student_cfg_path: str,
    out: str,
    device: int = 0,
) -> str:
    """Use the large teacher model to distill a small CPU student.

    Distillation strategy:
      - Teacher generates soft labels (class probability distributions) on
        every training image.
      - Student is trained with a KD loss combining hard CE and soft KL terms.

    Ultralytics >=8.3 supports `kd_teacher=<path>` in .train() to enable
    built-in knowledge distillation.  Falls back to standard training from
    the teacher's pseudo-labels if KD is not supported.

    Returns path to student best.pt.

    GPU-ONLY.
    """
    try:
        from ultralytics import YOLO
        import inspect
    except ImportError:
        raise RuntimeError("[distill] ultralytics required for distillation.")

    from frcscout.io.config import Config
    cfg = Config.load(student_cfg_path)
    imgsz   = int(cfg.get("imgsz", 960))
    epochs  = int(cfg.get("epochs", 150))
    batch   = int(cfg.get("batch", 16))
    patience = int(cfg.get("patience", 50))
    aug     = cfg.get("aug", {}) or {}
    aug_kwargs = {k: v for k, v in aug.items() if k != "comment"}

    m = YOLO(cfg.get("base_model"))

    # Check if ultralytics supports kd_teacher= arg.
    train_sig = inspect.signature(m.train)
    kd_supported = "kd_teacher" in train_sig.parameters

    if kd_supported:
        print(f"[distill] ultralytics KD supported: training with teacher={teacher_weights}")
        m.train(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            project=os.path.dirname(out),
            name=os.path.basename(out),
            exist_ok=True,
            device=device,
            cache=False,
            kd_teacher=teacher_weights,
            kd_alpha=0.5,   # 50% hard labels, 50% soft; tune if needed
            **aug_kwargs,
        )
    else:
        # Fallback: use the teacher to generate pseudo-labels, then train on them.
        print(
            "[distill] ultralytics KD (kd_teacher=) not available; "
            "falling back to pseudo-label distillation"
        )
        pseudo_data = _generate_pseudo_labels(teacher_weights, data, imgsz, device)
        m.train(
            data=pseudo_data or data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            project=os.path.dirname(out),
            name=os.path.basename(out),
            exist_ok=True,
            device=device,
            cache=False,
            **aug_kwargs,
        )

    best = os.path.join(out, "weights", "best.pt")
    print(f"[distill] student saved: {best}")
    return best


def _generate_pseudo_labels(
    teacher_weights: str,
    data_yaml: str,
    imgsz: int,
    device: int,
) -> Optional[str]:
    """Run teacher inference on all training images and write pseudo labels.

    This lets a student learn from teacher predictions without built-in KD
    support.  Returns path to new data.yaml with pseudo-labeled images, or
    None on failure.
    """
    try:
        import yaml
        from ultralytics import YOLO

        with open(data_yaml) as f:
            data = yaml.safe_load(f)

        data_path = Path(data.get("path", "."))
        train_rel  = data.get("train", "images/train")
        train_dir  = data_path / train_rel

        pseudo_label_dir = data_path / "labels_pseudo" / "train"
        pseudo_label_dir.mkdir(parents=True, exist_ok=True)

        teacher = YOLO(teacher_weights)
        img_files = sorted(
            p for p in train_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

        for imgp in img_files:
            res = teacher.predict(str(imgp), conf=0.1, imgsz=imgsz,
                                  device=device, verbose=False)
            if res and res[0].boxes is not None and len(res[0].boxes) > 0:
                lines = []
                boxes = res[0].boxes
                for cls, xywhn in zip(
                    boxes.cls.cpu().numpy(),
                    boxes.xywhn.cpu().numpy()
                ):
                    cx, cy, w, h = xywhn
                    lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                (pseudo_label_dir / (imgp.stem + ".txt")).write_text("\n".join(lines))
            else:
                (pseudo_label_dir / (imgp.stem + ".txt")).write_text("")

        # Build a new data.yaml pointing to pseudo labels.
        pseudo_yaml = str(data_path / "data_pseudo.yaml")
        new_data = dict(data)
        new_data["train"] = str(train_rel)   # images unchanged
        # Override label path via label_dir (ultralytics >=8.3 supports this).
        new_data["label_dir"] = str(pseudo_label_dir.relative_to(data_path))
        with open(pseudo_yaml, "w") as f:
            yaml.safe_dump(new_data, f)

        print(f"[distill] pseudo labels written: {pseudo_label_dir}")
        return pseudo_yaml

    except Exception as e:
        print(f"[distill] pseudo-label generation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main loop — Phase 1 + 2 + 3
# ---------------------------------------------------------------------------

def run_active_learning_loop(al_cfg: ActiveLearningConfig) -> dict:
    """Run the full active-learning + distillation pipeline.

    Returns a summary dict with per-round mAP history and final model paths.

    GPU-ONLY: calls train_one, mine_unlabeled, and distill_student.
    """
    from train.train_student import train_one

    os.makedirs(al_cfg.out_root, exist_ok=True)
    history: list[dict] = []

    # ── Phase 1: bootstrap training ─────────────────────────────────────────
    print("\n[AL] PHASE 1 — bootstrap training")
    data_yaml = os.path.join(al_cfg.data_root, "data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(
            f"[AL] data.yaml not found at {data_yaml}. "
            "Run autolabel first (scripts/autolabel.py or modal autolabel)."
        )

    ball_out  = os.path.join(al_cfg.out_root, "r0", "ball")
    robot_out = os.path.join(al_cfg.out_root, "r0", "robot")

    ball_weights  = train_one(al_cfg.ball_cfg,  data_yaml, ball_out,  resume=True)
    robot_weights = train_one(al_cfg.robot_cfg, data_yaml, robot_out, resume=True)

    ball_map  = measure_val_map(
        ball_weights, data_yaml,
        imgsz=_cfg_imgsz(al_cfg.ball_cfg), device=al_cfg.device
    )
    robot_map = measure_val_map(
        robot_weights, data_yaml,
        imgsz=_cfg_imgsz(al_cfg.robot_cfg), device=al_cfg.device
    )
    prev_map = (ball_map + robot_map) / 2 if ball_map >= 0 and robot_map >= 0 else -1.0

    history.append({"round": 0, "ball_map": ball_map, "robot_map": robot_map,
                    "avg_map": prev_map, "mined": 0})
    print(f"[AL] round 0 — ball mAP50-95={ball_map:.4f}, robot mAP50-95={robot_map:.4f}")

    # ── Phase 2: active-learning rounds ─────────────────────────────────────
    print("\n[AL] PHASE 2 — active-learning loop")
    review_dir = os.path.join(al_cfg.out_root, "review_queue")

    for r in range(1, al_cfg.max_rounds + 1):
        print(f"\n[AL] round {r}/{al_cfg.max_rounds}")

        round_out     = os.path.join(al_cfg.out_root, f"r{r}")
        mined_frames_dir = os.path.join(round_out, "mined_frames")
        mined_labels_dir = os.path.join(round_out, "mined_labels")

        # ── 2a: mine disagreements from unlabeled frames ─────────────────────
        if not os.path.isdir(al_cfg.unlabeled_dir) or not os.listdir(al_cfg.unlabeled_dir):
            print(f"[AL] no unlabeled frames in {al_cfg.unlabeled_dir}; stopping early")
            break

        mined = mine_unlabeled(
            student_weights=ball_weights,   # mine primarily on ball model
            unlabeled_dir=al_cfg.unlabeled_dir,
            teacher_weights=al_cfg.teacher_weights,
            out_dir=mined_frames_dir,
            conf_low=al_cfg.conf_low,
            iou_disagree=al_cfg.iou_disagree,
            max_mine=al_cfg.max_mine,
            device=al_cfg.device,
        )
        if not mined:
            print(f"[AL] round {r}: no frames mined; stopping early")
            break

        # ── 2b: queue for review ─────────────────────────────────────────────
        queue_for_review(mined, review_dir, round_idx=r)

        # ── 2c: auto-label mined frames with teacher ─────────────────────────
        autolabel_mined(
            mined_frames=mined,
            out_images_dir=mined_frames_dir,
            out_labels_dir=mined_labels_dir,
            device=f"cuda:{al_cfg.device}",
        )

        # Build extra data.yaml for the mined round.
        mined_data_yaml = _write_mined_data_yaml(
            mined_frames_dir, mined_labels_dir, round_out, data_yaml
        )

        # ── 2d: retrain students with base + mined data ───────────────────────
        ball_out_r  = os.path.join(round_out, "ball")
        robot_out_r = os.path.join(round_out, "robot")

        ball_weights  = train_one(al_cfg.ball_cfg,  data_yaml, ball_out_r,
                                   resume=False, extra_data=mined_data_yaml)
        robot_weights = train_one(al_cfg.robot_cfg, data_yaml, robot_out_r,
                                   resume=False, extra_data=mined_data_yaml)

        # ── 2e: measure improvement ───────────────────────────────────────────
        ball_map  = measure_val_map(
            ball_weights, data_yaml,
            imgsz=_cfg_imgsz(al_cfg.ball_cfg), device=al_cfg.device
        )
        robot_map = measure_val_map(
            robot_weights, data_yaml,
            imgsz=_cfg_imgsz(al_cfg.robot_cfg), device=al_cfg.device
        )
        avg_map = (ball_map + robot_map) / 2 if ball_map >= 0 and robot_map >= 0 else -1.0

        improvement = avg_map - prev_map if prev_map >= 0 else float("inf")
        history.append({
            "round": r, "ball_map": ball_map, "robot_map": robot_map,
            "avg_map": avg_map, "improvement": improvement,
            "mined": len(mined),
        })
        print(
            f"[AL] round {r} — ball={ball_map:.4f}, robot={robot_map:.4f}, "
            f"avg={avg_map:.4f}, delta={improvement:+.4f}, mined={len(mined)}"
        )

        if avg_map >= 0 and improvement < al_cfg.epsilon:
            print(
                f"[AL] PLATEAU: improvement {improvement:.6f} < epsilon {al_cfg.epsilon} "
                "-> stopping active-learning loop"
            )
            break
        prev_map = avg_map

    # ── Phase 3: distillation ────────────────────────────────────────────────
    if al_cfg.distill:
        print("\n[AL] PHASE 3 — knowledge distillation")
        distill_out_ball  = os.path.join(al_cfg.out_root, "distilled", "ball")
        distill_out_robot = os.path.join(al_cfg.out_root, "distilled", "robot")

        print("[AL] distilling ball student (YOLO11s@960) from teacher ...")
        ball_distilled = distill_student(
            teacher_weights=al_cfg.teacher_weights,
            data=data_yaml,
            student_cfg_path=al_cfg.ball_cfg,
            out=distill_out_ball,
            device=al_cfg.device,
        )

        print("[AL] distilling robot student (YOLO11n@512) from teacher ...")
        robot_distilled = distill_student(
            teacher_weights=al_cfg.teacher_weights,
            data=data_yaml,
            student_cfg_path=al_cfg.robot_cfg,
            out=distill_out_robot,
            device=al_cfg.device,
        )

        ball_map_final  = measure_val_map(
            ball_distilled, data_yaml,
            imgsz=_cfg_imgsz(al_cfg.ball_cfg), device=al_cfg.device
        )
        robot_map_final = measure_val_map(
            robot_distilled, data_yaml,
            imgsz=_cfg_imgsz(al_cfg.robot_cfg), device=al_cfg.device
        )
        history.append({
            "phase": "distillation",
            "ball_distilled_map": ball_map_final,
            "robot_distilled_map": robot_map_final,
        })
        print(
            f"[AL] distilled — ball mAP50-95={ball_map_final:.4f}, "
            f"robot mAP50-95={robot_map_final:.4f}"
        )
    else:
        ball_distilled = ball_weights
        robot_distilled = robot_weights

    # Write history.
    history_path = os.path.join(al_cfg.out_root, "al_history.json")
    Path(history_path).write_text(json.dumps(history, indent=2))
    print(f"[AL] history written: {history_path}")

    return {
        "history": history,
        "ball_weights": ball_distilled,
        "robot_weights": robot_distilled,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg_imgsz(cfg_path: str) -> int:
    from frcscout.io.config import Config
    cfg = Config.load(cfg_path)
    return int(cfg.get("imgsz", 960))


def _write_mined_data_yaml(
    images_dir: str,
    labels_dir: str,
    out_dir: str,
    base_data_yaml: str,
) -> str:
    """Write a minimal data.yaml for the mined frames."""
    try:
        import yaml
        with open(base_data_yaml) as f:
            base = yaml.safe_load(f)
        names = base.get("names", {0: "ball", 1: "robot"})
    except Exception:
        names = {0: "ball", 1: "robot"}

    yaml_path = os.path.join(out_dir, "mined_data.yaml")
    os.makedirs(out_dir, exist_ok=True)
    content = {
        "path": out_dir,
        "train": os.path.relpath(images_dir, out_dir),
        "val":   os.path.relpath(images_dir, out_dir),   # tiny val from same set
        "label_dir": os.path.relpath(labels_dir, out_dir),
        "names": names,
    }
    try:
        import yaml
        with open(yaml_path, "w") as f:
            yaml.safe_dump(content, f)
    except ImportError:
        # Manual YAML write without pyyaml.
        lines = [f"path: {out_dir}",
                 f"train: {content['train']}",
                 f"val: {content['val']}",
                 "names:"]
        for k, v in names.items():
            lines.append(f"  {k}: {v}")
        Path(yaml_path).write_text("\n".join(lines) + "\n")

    return yaml_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    ap = argparse.ArgumentParser(description="Active-learning loop for FRC Vision Scout")
    ap.add_argument("--ball_cfg",    default="configs/train_ball.yaml")
    ap.add_argument("--robot_cfg",   default="configs/train_robot.yaml")
    ap.add_argument("--data_root",   default="data/dataset")
    ap.add_argument("--unlabeled",   default="data/unlabeled")
    ap.add_argument("--out",         default="runs/active_learning")
    ap.add_argument("--teacher",     default="yolo11x.pt")
    ap.add_argument("--max_rounds",  type=int,   default=10)
    ap.add_argument("--epsilon",     type=float, default=0.002)
    ap.add_argument("--max_mine",    type=int,   default=500)
    ap.add_argument("--device",      type=int,   default=0)
    ap.add_argument("--no_distill",  action="store_true")
    args = ap.parse_args()

    cfg = ActiveLearningConfig(
        data_root=args.data_root,
        unlabeled_dir=args.unlabeled,
        out_root=args.out,
        ball_cfg=args.ball_cfg,
        robot_cfg=args.robot_cfg,
        teacher_weights=args.teacher,
        max_rounds=args.max_rounds,
        epsilon=args.epsilon,
        max_mine=args.max_mine,
        device=args.device,
        distill=not args.no_distill,
    )
    result = run_active_learning_loop(cfg)
    print("\n[AL] DONE")
    print(f"  ball weights:  {result['ball_weights']}")
    print(f"  robot weights: {result['robot_weights']}")


if __name__ == "__main__":
    _cli()
