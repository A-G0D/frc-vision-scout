"""
pipeline_sim.py — Full simulation pipeline for the FRC REBUILT (2026) playoff
match (Upper Bracket R4 M11, match_full.mp4).

Runs the complete chain:
  1.  shots.segment_shots()       -> identify wide camera segment(s)
  2.  detect.build_background()   -> median background plate
  3.  Per-frame detect + RobotTracker + BallTracker  (wide segment only)
  4.  events.detect_events()      -> PICKUP / SHOT / SCORE events
  5.  attribute.attribute_scores()-> per-robot score attribution
  6.  gamestate.simulate()        -> per-second timeline + per-robot totals
  7.  verify.verify()             -> MAE + accuracy metrics

Outputs
-------
  data/sim/pipeline_robot_tracks.json  — robot_tracks_by_frame (sampled)
  data/sim/pipeline_ball_tracks.json   — ball_tracks_by_frame (sampled)
  data/sim/events.json                 — full events list
  data/sim/gamestate.json              — game-state simulation result
  data/sim/verify_report.json          — verification metrics

Usage
-----
  python src/sim/pipeline_sim.py [video_path]

Falls back to data/sim/match_full.mp4 if no argument given.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_SRC  = _HERE.parent.parent       # .../src/
_ROOT = _SRC.parent               # project root
_DATA = _ROOT / "data" / "sim"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

# ---------------------------------------------------------------------------
# Import sim modules
# ---------------------------------------------------------------------------
from sim import field, detect as sim_detect, shots as sim_shots
from sim.track import RobotTracker
from sim.track_ball import BallTracker
from sim import events as sim_events
from sim import attribute as sim_attribute
from sim import gamestate as sim_gamestate
from sim import verify as sim_verify

# TBA ground truth
TBA_BLUE = 268
TBA_RED  = 506


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_overlay_ts(path: Path) -> list[dict]:
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    return raw.get("timeseries", [])


def _save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Wrote {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(video_path: Path | str) -> dict:
    """Run the full pipeline. Returns the verify result dict."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"FRC REBUILT Pipeline — {video_path.name}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    # ---- 0. Load pre-existing overlay_ts oracle --------------------------
    overlay_ts_path = _DATA / "overlay_ts.json"
    if not overlay_ts_path.exists():
        print("[pipeline] overlay_ts.json not found — re-generating...", file=sys.stderr)
        from sim.overlay_ts import read_timeseries
        overlay_ts = read_timeseries(str(video_path))
    else:
        print(f"[pipeline] Loading overlay oracle from {overlay_ts_path}", file=sys.stderr)
        overlay_ts = _load_overlay_ts(overlay_ts_path)
    print(f"[pipeline] Oracle rows: {len(overlay_ts)}", file=sys.stderr)

    # ---- 1. Identify camera segments ------------------------------------
    t0_step = time.perf_counter()
    print("\n[1/7] Segmenting camera shots...", file=sys.stderr)
    segments = sim_shots.segment_shots(str(video_path))
    wide_segs = [s for s in segments if s.get("kind") == "wide"]
    print(f"      Total segments: {len(segments)}, wide: {len(wide_segs)}", file=sys.stderr)
    for s in wide_segs:
        dur = (s["end_f"] - s["start_f"]) / 30.0
        print(f"      Wide: frame {s['start_f']}..{s['end_f']}  ({dur:.1f}s)", file=sys.stderr)

    # Use the longest wide segment; fall back to [0, 4170] if none found.
    if wide_segs:
        wide_segs_sorted = sorted(wide_segs, key=lambda s: s["end_f"] - s["start_f"], reverse=True)
        wide_start = wide_segs_sorted[0]["start_f"]
        wide_end   = wide_segs_sorted[0]["end_f"]
    else:
        # Hardcode from shots.json: the 139s wide segment is frames 0..4170
        wide_start, wide_end = 0, 4170
        print(f"      No wide segments found, using hardcoded 0..4170", file=sys.stderr)

    # Cap to reasonable bounds
    cap_probe = cv2.VideoCapture(str(video_path))
    total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_probe.release()
    wide_end = min(wide_end, total_frames)
    print(f"      Processing wide segment: frames {wide_start}..{wide_end}", file=sys.stderr)
    print(f"      Step done in {time.perf_counter() - t0_step:.1f}s", file=sys.stderr)

    # ---- 2. Build background -------------------------------------------
    t0_step = time.perf_counter()
    print("\n[2/7] Building median background...", file=sys.stderr)
    t0_bg = wide_start / 30.0
    t1_bg = wide_end / 30.0
    bg = sim_detect.build_background(str(video_path), t0=t0_bg, t1=t1_bg,
                                     max_samples=50)
    bg_path = _DATA / "pipeline_bg.png"
    cv2.imwrite(str(bg_path), bg)
    print(f"      Background saved to {bg_path}", file=sys.stderr)
    print(f"      Step done in {time.perf_counter() - t0_step:.1f}s", file=sys.stderr)

    # ---- 3. Per-frame detection + tracking ------------------------------
    t0_step = time.perf_counter()
    n_frames = wide_end - wide_start
    print(f"\n[3/7] Per-frame detection + tracking ({n_frames} frames)...", file=sys.stderr)

    robot_tracker = RobotTracker(
        max_age=60,
        min_hits=4,
        max_dist=90.0,
        iou_weight=0.5,
        max_tracks=6,
        reacquire_frames=600,
        reacquire_dist=220.0,
    )
    # track_ball.BallTracker: uses speed-adaptive gate + EMA velocity + multi-
    # criteria shot detection (speed + travel + directness). Parameters:
    #   shot_speed=4.5  — lower floor (px/frame) to catch the gradual ball arcs
    #   shot_min_frames=3 — must sustain speed for 3+ frames (rejects noise)
    #   shot_min_travel=18.0 — minimum total travel since shot began (px)
    ball_tracker = BallTracker(
        shot_speed=4.5,
        shot_min_frames=3,
        shot_min_travel=18.0,
    )

    robot_tracks_by_frame: dict[int, list] = {}
    ball_tracks_by_frame: dict[int, list] = {}

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, wide_start)

    fi = wide_start
    log_interval = max(1, n_frames // 20)
    while fi < wide_end:
        ok, frame = cap.read()
        if not ok:
            break

        rdets = sim_detect.detect_robots(frame, bg)
        bdets = sim_detect.detect_balls(frame)

        rtracked = robot_tracker.update(fi, rdets, frame)
        btracked = ball_tracker.update(fi, bdets)

        robot_tracks_by_frame[fi] = rtracked
        ball_tracks_by_frame[fi] = btracked

        if (fi - wide_start) % log_interval == 0:
            pct = 100.0 * (fi - wide_start) / n_frames
            nr = len(rtracked)
            nb_balls = len(btracked)
            elapsed = time.perf_counter() - t0_step
            eta = (elapsed / max(1, fi - wide_start + 1)) * (wide_end - fi - 1)
            print(f"      f{fi}  {pct:.0f}%  robots={nr}  balls={nb_balls}"
                  f"  ETA={eta:.0f}s", file=sys.stderr)
        fi += 1
    cap.release()

    n_tracked = sum(1 for v in robot_tracks_by_frame.values() if v)
    avg_robots = (sum(len(v) for v in robot_tracks_by_frame.values()) / max(1, len(robot_tracks_by_frame)))
    print(f"      Tracked {fi - wide_start} frames, "
          f"avg robots/frame={avg_robots:.2f}, "
          f"id_switches={robot_tracker.id_switches}", file=sys.stderr)

    # Track persistence metric
    all_track_ids = set()
    for tracks in robot_tracks_by_frame.values():
        for t in tracks:
            all_track_ids.add(t["id"])
    n_unique_ids = len(all_track_ids)
    # Ideal: 6 robots -> 6 stable IDs. More IDs = more fragmenting.
    track_persistence_str = f"{n_unique_ids} unique IDs (ideal=6), {robot_tracker.id_switches} alliance-vote flips"
    print(f"      {track_persistence_str}", file=sys.stderr)
    print(f"      Step done in {time.perf_counter() - t0_step:.1f}s", file=sys.stderr)

    # ---- 4. Hub regions -------------------------------------------------
    cap_probe2 = cv2.VideoCapture(str(video_path))
    ok2, sample_frame = cap_probe2.read()
    cap_probe2.release()
    if ok2:
        h_fr, w_fr = sample_frame.shape[:2]
    else:
        h_fr, w_fr = 360, 640
    hub_regs = field.hub_regions(w_fr, h_fr)

    # ---- 5. Event detection ---------------------------------------------
    t0_step = time.perf_counter()
    print("\n[4/7] Detecting events...", file=sys.stderr)
    events = sim_events.detect_events(
        robot_tracks_by_frame,
        ball_tracks_by_frame,
        hub_regs,
        overlay_ts,
    )
    pickups = [e for e in events if e["type"] == "pickup"]
    shots   = [e for e in events if e["type"] == "shot"]
    scores  = [e for e in events if e["type"] == "score"]
    print(f"      Events: {len(pickups)} pickups, {len(shots)} shots, "
          f"{len(scores)} score ticks", file=sys.stderr)
    _save_json(events, _DATA / "events.json")
    print(f"      Step done in {time.perf_counter() - t0_step:.1f}s", file=sys.stderr)

    # ---- 6. Score attribution -------------------------------------------
    t0_step = time.perf_counter()
    print("\n[5/7] Attributing scores...", file=sys.stderr)
    attributed_events = sim_attribute.attribute_scores(events, overlay_ts, hub_regs)
    attr_summary = sim_attribute.summarise_attribution(attributed_events)
    print(f"      Attribution coverage: "
          f"blue={attr_summary['coverage_pct']['blue']:.1f}% "
          f"red={attr_summary['coverage_pct']['red']:.1f}%", file=sys.stderr)
    sim_attribute.print_attribution_report(attributed_events, overlay_ts)
    print(f"      Step done in {time.perf_counter() - t0_step:.1f}s", file=sys.stderr)

    # ---- 7. Game state simulation ---------------------------------------
    t0_step = time.perf_counter()
    print("\n[6/7] Simulating game state...", file=sys.stderr)
    sim_result = sim_gamestate.simulate(attributed_events, overlay_ts)
    sim_gamestate.simulate_and_save(attributed_events, overlay_ts,
                                    out_path=_DATA / "gamestate.json")
    final = sim_result["final"]
    print(f"      Sim final: blue={final['blue']}, red={final['red']}", file=sys.stderr)
    print(f"      TBA final: blue={TBA_BLUE}, red={TBA_RED}", file=sys.stderr)
    print(f"      Step done in {time.perf_counter() - t0_step:.1f}s", file=sys.stderr)

    # ---- 8. Verification -----------------------------------------------
    t0_step = time.perf_counter()
    print("\n[7/7] Verifying simulation...", file=sys.stderr)
    verif = sim_verify.verify(sim_result, overlay_ts, {"blue": TBA_BLUE, "red": TBA_RED})
    print(f"\n{verif['notes']}", file=sys.stderr)
    _save_json(verif, _DATA / "verify_report.json")
    print(f"      Step done in {time.perf_counter() - t0_step:.1f}s", file=sys.stderr)

    # ---- Summary -------------------------------------------------------
    print(f"\n{'=' * 60}", file=sys.stderr)
    print("PIPELINE COMPLETE", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"  score_timeline_mae:      {verif['score_timeline_mae']:.3f} pts", file=sys.stderr)
    print(f"  timeline_accuracy_pct:   {verif['timeline_accuracy_pct']:.1f}%", file=sys.stderr)
    print(f"  final_blue:              {verif['final_blue']}  (TBA={TBA_BLUE})", file=sys.stderr)
    print(f"  final_red:               {verif['final_red']}  (TBA={TBA_RED})", file=sys.stderr)
    print(f"  final_vs_tba_ok:         {verif['final_vs_tba_ok']}", file=sys.stderr)
    print(f"  track_persistence:       {track_persistence_str}", file=sys.stderr)

    per_robot = sim_result.get("per_robot", {})
    if per_robot:
        print(f"\n  Per-robot breakdown:", file=sys.stderr)
        for rid, info in sorted(per_robot.items(),
                                key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0):
            print(f"    Robot {rid} ({info['alliance']:4s}): "
                  f"{info['points']:3d} pts  {info['shots']:3d} shots  "
                  f"{info['pickups']:3d} pickups  acc={info['accuracy']:.3f}", file=sys.stderr)

    # Return a compact result for callers
    return {
        "verif": verif,
        "sim": sim_result,
        "track_persistence": track_persistence_str,
        "n_unique_ids": n_unique_ids,
        "id_switches": robot_tracker.id_switches,
        "robot_tracks_by_frame": robot_tracks_by_frame,
        "ball_tracks_by_frame": ball_tracks_by_frame,
        "hub_regions": hub_regs,
        "events": events,
        "attributed_events": attributed_events,
        "overlay_ts": overlay_ts,
        "wide_start": wide_start,
        "wide_end": wide_end,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else str(_DATA / "match_full.mp4")
    result = run_pipeline(video)
    print("\n[pipeline] All outputs written to data/sim/")
