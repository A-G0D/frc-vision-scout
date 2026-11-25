"""_profile_pipeline.py — drive ScoutPipeline over N frames with SCOUT_PROFILE=1 to print
the per-block ms breakdown. Throwaway profiling harness (STEP 1)."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("SCOUT_PROFILE", "1")

import cv2
from frcscout.io.config import Config
from frcscout.pipeline.scout_pipeline import ScoutPipeline


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "data/scout_out/qual59_action.mp4"
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nframes = int(os.environ.get("SCOUT_PROFILE_FRAMES", "40")) + 5
    device = os.environ.get("PROF_DEVICE", "cpu")

    cfg = Config.load("configs/deploy.yaml")
    dets = cfg.get("detectors", {}) or {}
    for k in ("robot", "ball_inventory"):
        if isinstance(dets.get(k), dict):
            dets[k]["device"] = device

    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    print(f"[prof] {src} {W}x{H} fps={fps:.1f} device={device} frames={nframes}", flush=True)

    pipe = ScoutPipeline(cfg, frame_wh=(W, H), fps=fps, device=device)
    # apply optional live config from env
    if os.environ.get("PROF_LIVE"):
        pipe.configure_live(live_mode=True,
                            net_every=int(os.environ.get("PROF_NET", "1")),
                            blob_every=int(os.environ.get("PROF_BLOB", "1")),
                            pile_every=int(os.environ.get("PROF_PILE", "1")),
                            ocr_every=int(os.environ.get("PROF_OCR", "1")),
                            pose_every=int(os.environ.get("PROF_POSE", "1")),
                            fast=bool(os.environ.get("PROF_FAST")))

    t0 = time.perf_counter()
    n = 0
    for i in range(nframes):
        ok, frame = cap.read()
        if not ok:
            break
        pipe.process(frame, (start or 0) + i)
        n += 1
    dt = time.perf_counter() - t0
    cap.release()
    pipe.close()
    print(f"[prof] {n} frames in {dt:.1f}s -> {n/dt if dt else 0:.2f} fps (wall, incl warmup)",
          flush=True)


if __name__ == "__main__":
    main()
