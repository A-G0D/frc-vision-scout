"""Sample frames from footage for labeling. Uniform + event-biased sampling so the label
set covers dense piles and airborne moments, not just easy static frames.

Output is gitignored (frames/). Claude runs this via Bash.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from frcscout.io.config import Config


def sample(video: str, out_dir: str, fps_sample: float = 2.0) -> int:
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    step = max(1, round(fps / fps_sample))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    i = n = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok2, fr = cap.retrieve()
            if ok2:
                cv2.imwrite(f"{out_dir}/{Path(video).stem}_{n:05d}.png", fr)
                n += 1
        i += 1
    cap.release()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sources.yaml")
    a = ap.parse_args()
    cfg = Config.load(a.config)
    for m in cfg.get("matches", []):
        n = sample(m["out"], f"frames/{Path(m['out']).stem}")
        print(f"{m['out']}: {n} frames")


if __name__ == "__main__":
    main()
