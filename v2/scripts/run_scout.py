"""run_scout.py — drive the integrated scouting pipeline on real footage.

Two modes:

  --sync  (RECOMMENDED for file/demo rendering): a SYNCHRONOUS pipeline. Runs the YOLO11s
          ball model ONCE per frame @960 (non-tiled — proven to catch dense piles), splits
          class 0 -> balls / class 1 -> robots, feeds the real trackers + 6-robot prior +
          phase SM + metrics, renders the scouting overlay, and exports the match JSON/CSV.
          Deterministic, every-frame detection, progress printed unbuffered.

  default: the live async Engine (detect->track->geometry->fuse) via MatchRunner.run_video.
           Built for real-time; detections lag the loop on a fast file source.

    # synchronous demo on a mid-match window (robots + balls present), render + export:
    PYTHONPATH=src python scripts/run_scout.py --sync --source data/footage/qual56.mp4 \
        --start 2000 --frames 600 --out-mp4 data/scout_out/qual56_demo.mp4 \
        --out-dir data/scout_out --match qual56

    # quick smoke (engine path), no render:
    PYTHONPATH=src python scripts/run_scout.py --source data/footage/qual56.mp4 --smoke --frames 200
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

# make `frcscout` importable when run as a plain script (PYTHONPATH=src also works)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from frcscout.io.config import Config


def force_device(cfg: Config, device: str) -> Config:
    dets = cfg.get("detectors", {}) or {}
    for k in ("robot", "ball_inventory"):
        if isinstance(dets.get(k), dict):
            dets[k]["device"] = device
    return cfg


def _limited(it, n):
    for i, x in enumerate(it):
        if n and i >= n:
            break
        yield x


# --------------------------------------------------------------------------- #
# SYNCHRONOUS demo pipeline (the reliable file/demo path)
#
# The per-frame logic now lives in ONE place — frcscout.pipeline.scout_pipeline.ScoutPipeline.
# run_sync owns ONLY video IO (capture, seek, the mp4 writer), progress printing, and export;
# it delegates every per-frame computation + the overlay render to the pipeline. The MPP
# scale constant, the _BallTracker, and _project_h that used to live here moved into the
# pipeline module as well (the single source of truth).
# --------------------------------------------------------------------------- #
def run_sync(a, cfg) -> int:
    import cv2

    from frcscout.pipeline.scout_pipeline import ScoutPipeline

    dev = a.device

    # Accept a livestream/VOD URL (the "directly from livestream" path): resolve a
    # YouTube/yt-dlp URL to a direct stream URL; otherwise use the source as-is (a local
    # file, or an already-direct http stream). cv2.VideoCapture opens files AND http streams.
    src = a.source
    if "://" in src and ("youtu" in src):
        try:
            from frcscout.pipeline.decode import resolve_yt_url
            src = resolve_yt_url(a.source)
            print("[sync] resolved YouTube URL -> direct stream", flush=True)
        except Exception as e:
            print(f"[sync] URL resolve failed ({type(e).__name__}: {e}); using raw source", flush=True)
            src = a.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[sync] cannot open {a.source}")
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if a.start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)
    print(f"[sync] {a.source} {W}x{H} fps={fps:.1f} total={total} "
          f"start={a.start} frames={a.frames} dev={dev}", flush=True)

    writer = None
    if a.out_mp4:
        os.makedirs(os.path.dirname(a.out_mp4) or ".", exist_ok=True)
        writer = cv2.VideoWriter(a.out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    pipe = ScoutPipeline(cfg, frame_wh=(W, H), fps=fps, device=dev,
                         render=writer is not None)
    sync_perf = cfg.get("sync_performance", {}) or {}
    if sync_perf.get("enabled", False):
        pipe.configure_live(
            live_mode=True,
            fast=bool(sync_perf.get("fast", False)),
            net_every=int(sync_perf.get("net_every", pipe.net_every)),
            blob_every=int(sync_perf.get("blob_every", pipe.blob_every)),
            pile_every=int(sync_perf.get("pile_every", pipe.pile_every)),
            ocr_every=int(sync_perf.get("ocr_every", pipe.ocr_every)),
            pose_every=int(sync_perf.get("pose_every", pipe.pose_every_live)),
        )
        if not bool(sync_perf.get("hud_ocr", True)):
            pipe.hud = None

    t0 = time.perf_counter()
    n = 0
    try:
        for i in range(a.frames):
            ok, frame = cap.read()
            if not ok:
                break
            fidx = (a.start or 0) + i

            ann = pipe.process(frame, fidx)

            if writer is not None and ann is not None:
                writer.write(ann)
            n += 1
            if i % 30 == 0:
                st = pipe.last_state
                res = pipe.last_res
                pile = pipe.last_pile
                phases = {rid: rec.get("phase") for rid, rec in res["phase"].items()}
                r0xy = [round(v, 1) for v in (st.robots[0].field_xy_m or ())] if st.robots else []
                Hpose = (pipe.cur_pose.H if (pipe.cur_pose is not None and pipe.cur_pose.valid)
                         else None)
                ov = pipe.cur_ov
                print(f"  f{fidx}: six={len(st.robots)} balls={pipe.last_ball_det_count} "
                      f"fuel={pile.label()} flight={pipe.last_flight_count} "
                      f"geo={'TAG' if Hpose is not None else 'px'} R0_fieldxy={r0xy} "
                      f"phases={phases} | HUD B{ov.blue_teams}:{ov.blue_score} "
                      f"R{ov.red_teams}:{ov.red_score} t={ov.timer_s}", flush=True)
    finally:
        if writer is not None:
            writer.release()
        pipe.close()
        cap.release()

    dt = time.perf_counter() - t0
    print(f"[sync] {n} frames in {dt:.1f}s ({n/dt if dt else 0:.1f} fps) | "
          f"max_six={pipe.max_six} pose_frames={pipe.pose_frames}/{n} "
          f"crowd_rejected={pipe.crowd_rejected} phase_hist={pipe.phase_hist}", flush=True)
    if a.out_mp4:
        print(f"[sync] wrote {a.out_mp4}", flush=True)
    if a.out_dir:
        exp = pipe.runner.export(a.out_dir, a.match)
        print(f"[export] {exp}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# Engine (async) path — smoke + run_video
# --------------------------------------------------------------------------- #
def run_engine(a, cfg) -> int:
    from frcscout.pipeline.engine import Engine
    from frcscout.pipeline.match_runner import MatchRunner

    t0 = time.perf_counter()
    try:
        eng = Engine(cfg, a.source)
    except Exception:
        print("[engine] CONSTRUCTION FAILED:\n" + traceback.format_exc())
        return 2
    print(f"[engine] constructed in {time.perf_counter()-t0:.1f}s; "
          f"fps={eng.fps}, frame={eng.src.w}x{eng.src.h}, device={a.device}", flush=True)
    runner = MatchRunner(field_model=eng.field, hub_localizer=eng.field, cfg=cfg, fps=eng.fps)
    try:
        if a.smoke:
            n = 0
            max_six = 0
            for i, frame in enumerate(_limited(eng.src.frames(), a.frames)):
                st = eng.step(frame, i)
                res = runner.process(st)
                max_six = max(max_six, len(st.robots))
                if i % 20 == 0:
                    print(f"  f{i:>4}: six={len(st.robots)} balls_inv={len(st.balls_inventory)} "
                          f"flight={len(st.balls_flight)} ovl={st.overlay.phase} "
                          f"pose={getattr(st.pose,'valid',None)}", flush=True)
                n += 1
            print(f"[smoke] {n} frames OK | max_six={max_six}", flush=True)
            return 0
        summary = runner.run_video(eng, _limited(eng.src.frames(), a.frames),
                                   out_mp4=a.out_mp4, out_dir=a.out_dir, match_key=a.match)
        print(f"[done] {summary}", flush=True)
        return 0
    finally:
        try:
            eng.sched.shutdown()
        except Exception:
            pass
        try:
            eng.src.release()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the integrated FRC scouting pipeline.")
    ap.add_argument("--config", default="configs/deploy.yaml")
    ap.add_argument("--source", required=True)
    ap.add_argument("--frames", type=int, default=600, help="frames to process")
    ap.add_argument("--start", type=int, default=0, help="seek to this frame first (sync mode)")
    ap.add_argument("--device", default="intel:cpu", help="intel:cpu|cpu|intel:gpu|intel:npu")
    ap.add_argument("--out-mp4", dest="out_mp4", default=None)
    ap.add_argument("--out-dir", dest="out_dir", default=None)
    ap.add_argument("--match", default="match")
    ap.add_argument("--sync", action="store_true", help="synchronous demo pipeline (recommended)")
    ap.add_argument("--smoke", action="store_true", help="engine smoke: step frames, print, exit")
    a = ap.parse_args()

    cfg = Config.load(a.config)
    force_device(cfg, a.device)
    if a.sync:
        return run_sync(a, cfg)
    return run_engine(a, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
