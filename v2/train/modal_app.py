"""Cloud-GPU training on Modal — the browser-free replacement for Colab/Kaggle cells.

Why Modal instead of Colab: Modal is defined in Python and launched from the shell
(`modal run train/modal_app.py::train_all`). Claude can drive every step via Bash with
NO browser, NO notebook cells, NO "are you a robot" datacenter gate. One-time auth is
`modal token new` (the only browser touch in the whole project).

Data + weights move through a Modal Volume (persistent cloud storage), so:
    autolabel -> train -> export  all read/write the same volume; `make pull` downloads
    the final OpenVINO artifacts to the laptop.

Entry points (each a `modal run` target):
    autolabel    teacher ensemble -> draft pseudo-labels on the volume
    train_all    train ball + robot students (resumable on preemption)
    export_all   OpenVINO INT8 export (+QAT fallback) per device
"""
from __future__ import annotations

import modal

VOL = modal.Volume.from_name("frcscout-data", create_if_missing=True)
MODELS = modal.Volume.from_name("frcscout-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1")
    .pip_install("ultralytics>=8.3", "sahi>=0.11.20", "nncf>=2.12",
                 "supervision>=0.22", "openvino>=2024.4", "opencv-python-headless")
)
app = modal.App("frcscout-train", image=image)

GPU = "T4"          # bump to "A10G"/"L4" for faster epochs; T4 is the free-tier sweet spot
DATA = "/data"
OUT = "/exports"


@app.function(gpu=GPU, volumes={DATA: VOL, OUT: MODELS}, timeout=60 * 60 * 6)
def autolabel(clip: str = "match.mp4", step_s: float = 0.5):
    """Teacher ensemble -> draft YOLO labels on the volume.
    Balls: YOLO11x + SAHI tiling. Robots: bumper-color + motion proposals (NOT YOLO-World,
    which gave 0 labels in v1). Writes images/labels to {DATA}/dataset (draft split only).

    Delegates to scripts/autolabel.py so the logic is shared with the Kaggle notebook.
    """
    import sys, os
    # The repo is mounted at /root inside the Modal container via the image build.
    sys.path.insert(0, "/root")
    from scripts.autolabel import autolabel_clip
    import json

    clip_path = os.path.join(DATA, clip)
    out_images = os.path.join(DATA, "dataset", "images", "train")
    out_labels = os.path.join(DATA, "dataset", "labels", "train")

    # autolabel_clip writes images + labels and returns per-frame metadata.
    meta = autolabel_clip(
        clip=clip_path,
        out_images_dir=out_images,
        out_labels_dir=out_labels,
        step_s=step_s,
        ball_conf=0.05,
        device="cuda:0",
        tmp_dir=DATA,
        gid_start=0,
        clip_idx=0,
    )

    # Write metadata for the temporal-split step.
    meta_path = os.path.join(DATA, "dataset", "frames_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    # Temporal split: last 15% of each clip's frames -> val.
    import shutil
    os.makedirs(os.path.join(DATA, "dataset", "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(DATA, "dataset", "labels", "val"), exist_ok=True)
    cut = int(0.85 * len(meta))
    for m in meta[cut:]:
        for sub, ext in (("images", "png"), ("labels", "txt")):
            src = os.path.join(DATA, "dataset", sub, "train", f"{m['gid']:06d}.{ext}")
            dst = os.path.join(DATA, "dataset", sub, "val",   f"{m['gid']:06d}.{ext}")
            if os.path.exists(src):
                shutil.move(src, dst)

    # Write data.yaml.
    data_yaml = os.path.join(DATA, "dataset", "data.yaml")
    with open(data_yaml, "w") as f:
        f.write(
            f"path: {DATA}/dataset\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"names:\n  0: ball\n  1: robot\n"
        )

    print(f"[modal autolabel] done: {len(meta)} frames, split {cut} train / {len(meta)-cut} val")
    print(f"[modal autolabel] data.yaml: {data_yaml}")
    VOL.commit()


@app.function(gpu=GPU, volumes={DATA: VOL, OUT: MODELS}, timeout=60 * 60 * 12)
def train_all(resume: bool = True):
    """Train ball (YOLO11s@960) + robot (YOLO11n@512) students from the REVIEWED,
    match-split dataset. Checkpoints to the volume every epoch -> preemption-safe; re-run
    to resume from last.pt. Reads configs/train_*.yaml (uploaded with the code)."""
    from train.train_student import train_one
    train_one("configs/train_ball.yaml", data=f"{DATA}/dataset_ball/data.yaml",
              out=f"{OUT}/ball", resume=resume)
    train_one("configs/train_robot.yaml", data=f"{DATA}/dataset_robot/data.yaml",
              out=f"{OUT}/robot", resume=resume)
    MODELS.commit()


@app.function(gpu=GPU, volumes={DATA: VOL, OUT: MODELS}, timeout=60 * 60 * 2)
def export_all():
    """Export OpenVINO INT8 for both students, per target device, with QAT fallback if
    INT8 PTQ drops small-ball recall past the configured threshold."""
    from train.export_openvino import export_one
    export_one(f"{OUT}/ball/weights/best.pt", "configs/train_ball.yaml", f"{OUT}/ball_openvino_model")
    export_one(f"{OUT}/robot/weights/best.pt", "configs/train_robot.yaml", f"{OUT}/robot_openvino_model")
    MODELS.commit()


@app.local_entrypoint()
def main():
    """`modal run train/modal_app.py` runs the full chain. Or call a single function:
    `modal run train/modal_app.py::train_all`."""
    autolabel.remote()
    train_all.remote()
    export_all.remote()
