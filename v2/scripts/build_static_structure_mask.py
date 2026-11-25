"""Build a per-video mask for fixed saturated red/blue field structures.

The mask is derived from temporal persistence: a pixel is marked static only when it
is saturated red or blue in a high fraction of sampled frames. Moving robots should
not remain at the same image pixel often enough to survive this vote.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from frcscout.detect.robot_filter import build_static_structure_mask, saturated_red_or_blue


def build_mask(
    video_path: Path,
    samples: int = 120,
    persistence: float = 0.60,
    dilate: int = 5,
) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"Could not determine frame count for {video_path}")

    frame_ids = np.linspace(0, max(0, total - 1), samples, dtype=np.int64)
    acc: np.ndarray | None = None
    read = 0
    for frame_id in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        vote = saturated_red_or_blue(frame)
        if acc is None:
            acc = np.zeros(vote.shape, dtype=np.uint16)
        acc += vote.astype(np.uint16)
        read += 1
    cap.release()

    if acc is None or read == 0:
        raise RuntimeError(f"No frames could be sampled from {video_path}")

    return build_static_structure_mask(
        acc,
        read,
        persistence=persistence,
        dilate=dilate,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(ROOT / "data/footage/qual59.mp4"))
    ap.add_argument("--out", default=str(ROOT / "data/robot_gt/qual59_static_mask.png"))
    ap.add_argument("--samples", type=int, default=120)
    ap.add_argument("--persistence", type=float, default=0.60)
    ap.add_argument("--dilate", type=int, default=5)
    args = ap.parse_args()

    mask = build_mask(
        Path(args.video),
        samples=args.samples,
        persistence=args.persistence,
        dilate=args.dilate,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), mask):
        raise RuntimeError(f"Could not write mask: {out}")
    print(str(out))


if __name__ == "__main__":
    main()
