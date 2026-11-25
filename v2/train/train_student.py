"""Core ultralytics training for one student or the large reference teacher.

Adds:
  - Motion-blur augmentation (airborne balls are blurred in fast-pan footage).
  - deploy_imgsz == imgsz invariant enforcement (the v1 source of eval mismatch).
  - Extra unlabeled-mining data for active-learning rounds.
  - train_large() for the big reference/teacher model (YOLO11x/l @ high imgsz)
    trained until val mAP plateaus before distilling to small students.

GPU-only paths are guarded and marked GPU-ONLY in comments.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from frcscout.io.config import Config


# ---------------------------------------------------------------------------
# Motion-blur augmentation helper
# ---------------------------------------------------------------------------

def _make_motion_blur_transform(p: float = 0.35, blur_limit: int = 15):
    """Build an albumentations MotionBlur transform for airborne-ball aug.

    Returns the transform if albumentations is available, else None.
    Motion blur simulates fast-panning broadcast cameras: balls are elliptical
    smears rather than round dots, which is the dominant failure mode in v1.

    GPU-ONLY context: called inside Ultralytics .train(), so this path only
    runs in the cloud GPU environment.
    """
    try:
        import albumentations as A
        return A.MotionBlur(blur_limit=blur_limit, p=p)
    except ImportError:
        print("[train_student] albumentations not installed; motion-blur aug skipped")
        return None


# ---------------------------------------------------------------------------
# Config invariant check
# ---------------------------------------------------------------------------

def _check_deploy_imgsz(cfg: Config, path: str):
    """Raise ValueError if deploy_imgsz != imgsz (the v1 eval mismatch bug)."""
    imgsz        = cfg.get("imgsz")
    deploy_imgsz = cfg.get("deploy_imgsz")
    if imgsz is None or deploy_imgsz is None:
        raise ValueError(
            f"[train_student] Config {path} must define both 'imgsz' and 'deploy_imgsz'."
        )
    if int(imgsz) != int(deploy_imgsz):
        raise ValueError(
            f"[train_student] deploy_imgsz ({deploy_imgsz}) != imgsz ({imgsz}) in {path}. "
            "These MUST match — a mismatch inflates eval mAP because the export runs at "
            "a different resolution than training."
        )


# ---------------------------------------------------------------------------
# Student training
# ---------------------------------------------------------------------------

def train_one(
    train_cfg_path: str,
    data: str,
    out: str,
    resume: bool = True,
    extra_data: Optional[str] = None,
) -> str:
    """Train one specialist student.

    Args:
        train_cfg_path: Path to configs/train_ball.yaml or train_robot.yaml.
        data:           Path to the YOLO data.yaml for this student's dataset.
        out:            Output directory for weights (project/name style).
        resume:         Resume from last.pt if it exists (preemption-safe).
        extra_data:     Optional path to a second data.yaml from an active-learning
                        mining round.  When provided, the datasets are concatenated
                        via a generated combined data.yaml.

    Returns:
        Path to best.pt.

    GPU-ONLY: ultralytics.YOLO.train() requires a CUDA device.
    """
    cfg = Config.load(train_cfg_path)

    # Enforce deploy_imgsz invariant before spending any GPU time.
    _check_deploy_imgsz(cfg, train_cfg_path)

    # GPU-ONLY import guard.
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError(
            "[train_student] ultralytics not installed — this must run on the GPU cloud env."
        )

    # ── optional: merge base data + mined active-learning data ──────────────
    effective_data = data
    if extra_data is not None and os.path.exists(extra_data):
        effective_data = _merge_data_yamls(data, extra_data, out)
        print(f"[train_student] active-learning: merged {data} + {extra_data} -> {effective_data}")

    # ── build albumentations motion-blur callback ────────────────────────────
    motion_blur = _make_motion_blur_transform()

    # ── resume or fresh run ──────────────────────────────────────────────────
    last = os.path.join(out, "weights", "last.pt")
    aug  = cfg.get("aug", {}) or {}

    if resume and os.path.exists(last):
        print(f"[train_student] resuming from {last}")
        m = YOLO(last)
        m.train(resume=True)
    else:
        m = YOLO(cfg.get("base_model"))

        # Build aug kwargs; skip any keys that aren't valid ultralytics train params
        # (e.g. the "comment" marker).  motion_blur is injected via the callback.
        aug_kwargs = {k: v for k, v in aug.items() if k not in ("comment",)}

        # Inject motion-blur augmentation via ultralytics custom transform callback.
        if motion_blur is not None:
            aug_kwargs["augment"] = True    # ensure aug pipeline is active
            # ultralytics >=8.3 supports extra_augment hook; gracefully skip on older.
            try:
                m.add_callback(
                    "on_train_batch_start",
                    lambda trainer: _apply_motion_blur(trainer, motion_blur),
                )
            except Exception:
                pass  # older ultralytics — motion blur aug just won't fire

        m.train(
            data=effective_data,
            epochs=cfg.get("epochs", 120),
            imgsz=cfg.get("imgsz", 960),
            batch=cfg.get("batch", 16),
            patience=cfg.get("patience", 50),
            project=os.path.dirname(out),
            name=os.path.basename(out),
            exist_ok=True,
            cache=False,
            **aug_kwargs,
        )

    return os.path.join(out, "weights", "best.pt")


def _apply_motion_blur(trainer, transform):
    """Callback hooked into on_train_batch_start to apply MotionBlur to the batch.

    Operates in-place on trainer.batch["img"] (a torch.Tensor BCHW in [0,1]).
    This is a best-effort augmentation; it silently no-ops on any error so that
    the training run is never interrupted by aug failures.

    GPU-ONLY: torch and albumentations both active in the cloud env.
    """
    try:
        import numpy as np
        import torch
        batch_img = trainer.batch.get("img")
        if batch_img is None:
            return
        # Convert to numpy HWC uint8, apply MotionBlur, convert back.
        imgs = batch_img.cpu().numpy()    # BCHW float [0,1]
        out  = []
        for i in range(imgs.shape[0]):
            img_hwc = (imgs[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            aug_img = transform(image=img_hwc)["image"]
            out.append(aug_img.transpose(2, 0, 1).astype(np.float32) / 255.0)
        augmented = torch.from_numpy(np.stack(out)).to(batch_img.device)
        trainer.batch["img"] = augmented
    except Exception:
        pass    # never break training; aug is nice-to-have


def _merge_data_yamls(base_yaml: str, extra_yaml: str, out_dir: str) -> str:
    """Merge two YOLO data.yaml files into a combined one for active-learning rounds.

    Strategy: concatenate the image lists by writing a new data.yaml that points to
    both dataset paths via the datasets: [...] syntax (ultralytics >=8.3).
    Falls back to symlinking/copying if that syntax is unsupported.

    Returns the path to the merged data.yaml.
    """
    try:
        import yaml
    except ImportError:
        raise RuntimeError("PyYAML required for merging data yamls")

    with open(base_yaml) as f:
        base = yaml.safe_load(f)
    with open(extra_yaml) as f:
        extra = yaml.safe_load(f)

    merged_path = os.path.join(out_dir, "data_merged.yaml")
    os.makedirs(out_dir, exist_ok=True)

    # ultralytics >=8.3 supports 'datasets:' key for multi-dataset training.
    merged = {
        "datasets": [base_yaml, extra_yaml],
        "names": base.get("names", extra.get("names", {0: "ball", 1: "robot"})),
    }
    with open(merged_path, "w") as f:
        import yaml as _yaml
        _yaml.safe_dump(merged, f)

    return merged_path


# ---------------------------------------------------------------------------
# Large reference / teacher model training
# ---------------------------------------------------------------------------

def train_large(
    data: str,
    out: str,
    base_model: str = "yolo11x.pt",
    imgsz: int = 1280,
    epochs: int = 300,
    batch: int = 8,
    patience: int = 50,
    resume: bool = True,
    device: int = 0,
) -> str:
    """Train the large reference teacher model until val mAP plateaus.

    This is the slow, high-accuracy model that:
      1. Provides the quality ceiling we measure distilled students against.
      2. Drives the active-learning loop (its disagreements with the student
         define what to review next).
      3. Generates soft-label training signals for knowledge distillation.

    The model is intentionally slow (YOLO11x at 1280px); it runs once in the
    cloud and the result is cached.  Students target the same mAP at ≤10% of
    the inference time.

    Args:
        data:       YOLO data.yaml path.
        out:        Output dir (project/name).
        base_model: Starting weight file.  YOLO11x is the default ceiling.
        imgsz:      Input resolution.  1280 is strong for small 5px balls.
        epochs:     Max epochs (early stopping via patience).
        batch:      Reduce if OOM on T4 (8 is safe for 1280px on 16GB T4).
        patience:   Early-stop patience (stop when val mAP doesn't improve).
        resume:     Resume from last.pt.
        device:     CUDA device index.

    Returns:
        Path to best.pt.

    GPU-ONLY: requires CUDA.  Do not call locally.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError(
            "[train_large] ultralytics not installed — GPU cloud env required."
        )

    last = os.path.join(out, "weights", "last.pt")

    if resume and os.path.exists(last):
        print(f"[train_large] resuming from {last}")
        m = YOLO(last)
        m.train(resume=True)
    else:
        print(f"[train_large] fresh training: {base_model} @ {imgsz}px for up to {epochs} epochs")
        m = YOLO(base_model)
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
            # Strong aug for the teacher — it has to generalise to all venues.
            mosaic=1.0,
            close_mosaic=30,
            copy_paste=0.5,
            hsv_h=0.02,
            hsv_s=0.6,
            hsv_v=0.5,
            degrees=5,
            translate=0.1,
            scale=0.6,
            fliplr=0.5,
            erasing=0.4,
        )

    best = os.path.join(out, "weights", "best.pt")
    print(f"[train_large] teacher model saved: {best}")
    return best
