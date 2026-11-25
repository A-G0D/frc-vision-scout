"""Drive the whole Kaggle training run from the shell — so Claude does it, not you.

With a Kaggle API token in place (~/.kaggle/kaggle.json — created once on your account
page), this uploads footage as a Dataset, pushes the notebook as a GPU+internet kernel
(headless Save&Run-All), polls until it finishes, and downloads the model zip. No browser.

Reads your username from kaggle.json automatically and fills the *-metadata.json slugs.

    python scripts/kaggle_drive.py up      # create/version the footage dataset
    python scripts/kaggle_drive.py push    # push + run the notebook (GPU, internet on)
    python scripts/kaggle_drive.py status  # poll run state
    python scripts/kaggle_drive.py pull    # download /kaggle/working output -> weights/

Fully browser-free path. The only thing you do once: create the account + download the API
token. (Phone-verification to unlock GPU is also a one-time account step.)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
KAGGLE_DIR = HERE / "kaggle"
DATASET_SLUG = "frc-footage"
KERNEL_SLUG = "frc-train"


def _username() -> str:
    for p in (Path.home() / ".kaggle" / "kaggle.json", Path("kaggle.json")):
        if p.exists():
            return json.load(open(p))["username"]
    sys.exit("No kaggle.json found. Create an API token on kaggle.com/settings -> Create New Token, "
             "then put it at ~/.kaggle/kaggle.json.")


def _run(cmd: list[str]):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def up(args):
    user = _username()
    data_dir = HERE / "data" / "footage"
    assert any(data_dir.glob("*.mp4")), f"No .mp4 in {data_dir} — run `make footage` first."
    meta = {"title": DATASET_SLUG, "id": f"{user}/{DATASET_SLUG}", "licenses": [{"name": "CC0-1.0"}]}
    (data_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    # create on first run, version on subsequent runs
    try:
        _run(["kaggle", "datasets", "create", "-p", str(data_dir), "--dir-mode", "zip"])
    except subprocess.CalledProcessError:
        _run(["kaggle", "datasets", "version", "-p", str(data_dir), "-m", "update", "--dir-mode", "zip"])


def push(args):
    user = _username()
    meta = {
        "id": f"{user}/{KERNEL_SLUG}",
        "title": "frc-train",
        "code_file": "frc_train_kaggle.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,            # <- the accelerator step, set via API (no editor click)
        "enable_internet": True,       # <- Internet On, via API
        "dataset_sources": [f"{user}/{DATASET_SLUG}"],
    }
    (KAGGLE_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    _run(["kaggle", "kernels", "push", "-p", str(KAGGLE_DIR)])
    print(f"\nrunning headless: https://www.kaggle.com/code/{user}/{KERNEL_SLUG}")


def status(args):
    user = _username()
    _run(["kaggle", "kernels", "status", f"{user}/{KERNEL_SLUG}"])


def pull(args):
    user = _username()
    out = HERE / "weights"
    out.mkdir(exist_ok=True)
    _run(["kaggle", "kernels", "output", f"{user}/{KERNEL_SLUG}", "-p", str(out)])
    print(f"\nunzip {out}/frc_model_openvino.zip and run: make bench SOURCE=clip.mp4")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["up", "push", "status", "pull"])
    a = ap.parse_args()
    {"up": up, "push": push, "status": status, "pull": pull}[a.cmd](a)
