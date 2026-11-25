"""live/run_live.py — production entry point for the FRC Vision Scout live pipeline.

Ties together:
    ingest (FileSource or YouTubeLiveIngest)
    -> ScoutPipeline.process per frame
    -> OverlayObs (from pipe.cur_ov) -> MatchSegmenter
    -> on each finalized match: MetricsAggregator.aggregate -> MatchRecordBuilder.build
    -> LivePublisher.publish(final=True)

Also publishes in-progress snapshots every SNAPSHOT_EVERY_S seconds while a match is live.

CLI
---
    python live/run_live.py --source data/footage/qual59.mp4 \\
        --event-key 2026ctwat --frames 300 --store data/live_store

    --source   : file path or YouTube URL (default data/footage/qual59.mp4)
    --event-key: TBA event key (default 2026ctwat)
    --store    : publish store directory (default data/live_store)
    --frames   : max frames to process, 0=all (default 0)
    --calib    : path to a CalibrationArtifact JSON (optional)
    --device   : YOLO device string (default cpu)
    --snapshot-every : seconds between in-progress snapshots (default 10)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Bootstrap: allow running from the v2 root without a full package install.
_HERE = os.path.dirname(os.path.abspath(__file__))
_V2 = os.path.dirname(_HERE)
_SRC = os.path.join(_V2, "src")
for _p in (_SRC, _V2):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SNAPSHOT_EVERY_S_DEFAULT = 10.0


def _load_cfg(device: str):
    """Load deploy.yaml and force device onto detectors (mirrors run_scout.force_device)."""
    from frcscout.io.config import Config
    cfg_path = os.path.join(_V2, "configs", "deploy.yaml")
    cfg = Config.load(cfg_path)
    dets = cfg.get("detectors", {}) or {}
    for k in ("robot", "ball_inventory"):
        if isinstance(dets.get(k), dict):
            dets[k]["device"] = device
    return cfg


def _open_source(source: str, fps_cap: float = 30.0):
    """Return a started FileSource (for files) or YouTubeLiveIngest (for URLs)."""
    if "://" in source and ("youtu" in source or "youtube" in source):
        from live.ingest import YouTubeLiveIngest
        src = YouTubeLiveIngest(source, target_fps=fps_cap)
        src.start()
        return src, "youtube"
    else:
        from live.ingest import FileSource
        # target_fps=None → use native fps (no artificial decimation; pipeline consumes
        # frames as fast as they arrive, which is what we want for a capped-frames demo).
        src = FileSource(source, target_fps=None)
        src.start()
        return src, "file"


def _stop_source(src) -> None:
    try:
        src.stop()
    except Exception:
        pass


def _ov_to_obs(ov, frame_idx: int):
    """Map a ScoutPipeline OverlayState to a MatchSegmenter OverlayObs."""
    from live.match_segment import OverlayObs
    # Determine layout: if there is a phase or timer it's a live_bar; results screens
    # are not surfaced in cur_ov yet (HUD reader emits them as phase changes), so we
    # map any active live-bar read to "live_bar" and fall through to "unknown".
    layout = "unknown"
    if ov is not None and (ov.timer_s is not None or ov.phase is not None):
        layout = "live_bar"
    timer = getattr(ov, "timer_s", None) if ov else None
    phase = getattr(ov, "phase", None) if ov else None
    blue_score = getattr(ov, "blue_score", None) if ov else None
    red_score = getattr(ov, "red_score", None) if ov else None
    comp_level = getattr(ov, "comp_level", None) if ov else None
    match_number = getattr(ov, "match_number", None) if ov else None
    blue_teams = list(getattr(ov, "blue_teams", []) or []) if ov else []
    red_teams = list(getattr(ov, "red_teams", []) or []) if ov else []
    return OverlayObs(
        idx=frame_idx,
        layout=layout,
        timer_s=timer,
        phase=phase,
        blue_score=blue_score,
        red_score=red_score,
        comp_level=comp_level,
        match_number=match_number,
        blue_teams=blue_teams,
        red_teams=red_teams,
    )


def _get_vision_metrics(pipe, match_key: str, overlay_timeseries=None) -> dict:
    """Derive the vision_metrics dict (with tba_comparable) from the pipeline runner.

    ScoutPipeline.runner is a MatchRunner; MatchRunner.finalize() returns a match_record
    dict with {match_key, robots, events, ...} but NOT the tba_comparable block directly.
    We feed that through MetricsAggregator.aggregate() which returns the §5 schema
    including tba_comparable.
    """
    from frcscout.metrics.scout import MetricsAggregator
    # MatchRunner.finalize() -> match_record
    match_record = pipe.runner.finalize(match_key)
    # MetricsAggregator.aggregate() -> vision_metrics with tba_comparable
    agg = MetricsAggregator()
    vision_metrics = agg.aggregate(match_record, overlay_timeseries=overlay_timeseries)
    return vision_metrics


def _load_tba_match(event_key: str, match_key: str):
    """Load the TBA match dict from the on-disk cache at data/tba/<event_key>_matches.json."""
    tba_path = os.path.join(_V2, "data", "tba", f"{event_key}_matches.json")
    if not os.path.exists(tba_path):
        print(f"[tba] cache not found: {tba_path}", flush=True)
        return None
    try:
        with open(tba_path, encoding="utf-8") as f:
            matches = json.load(f)
        for m in matches:
            if m.get("key") == match_key:
                return m
        print(f"[tba] match {match_key} not found in {tba_path}", flush=True)
    except Exception as e:
        print(f"[tba] load error: {e}", flush=True)
    return None


def _infer_match_key(source: str, event_key: str) -> str:
    """Derive a best-effort match_key from the source filename.

    Convention: qualN.mp4 -> <event_key>_qmN
                matchN.mp4 -> <event_key>_qmN  (common alias)
    Returns <event_key>_live if no match.
    """
    import re
    basename = os.path.splitext(os.path.basename(source))[0].lower()
    m = re.search(r"qual(\d+)", basename)
    if m:
        return f"{event_key}_qm{m.group(1)}"
    m = re.search(r"match(\d+)", basename)
    if m:
        return f"{event_key}_qm{m.group(1)}"
    return f"{event_key}_live"


def _load_calib(calib_path):
    """Load a CalibrationArtifact from path, or return None."""
    if not calib_path:
        return None
    try:
        from calib.error_model import CalibrationArtifact
        art = CalibrationArtifact.load(calib_path)
        print(f"[calib] loaded artifact from {calib_path}", flush=True)
        return art
    except Exception as e:
        print(f"[calib] load failed ({type(e).__name__}: {e}); continuing without calib", flush=True)
        return None


def _build_and_publish(
    pipe,
    seg_match_key: str,
    event_key: str,
    publisher,
    calib,
    overlay_timeseries,
    final: bool,
) -> dict:
    """Get vision metrics, look up TBA, build the record, and publish it."""
    from metrics.aggregator import MatchRecordBuilder

    # Determine match_key: prefer the segmenter's resolved key; fall back to a synthetic one.
    match_key = seg_match_key or f"{event_key}_live_unknown"

    print(f"[publish] building record for {match_key} (final={final})", flush=True)
    vision_metrics = _get_vision_metrics(pipe, match_key, overlay_timeseries)

    tba_match = _load_tba_match(event_key, match_key)
    if tba_match is not None:
        print(f"[tba] found match dict for {match_key}", flush=True)
    else:
        print(f"[tba] no TBA match for {match_key} (offline; publishing vision-only)", flush=True)

    builder = MatchRecordBuilder()
    record = builder.build(match_key, vision_metrics, tba_match=tba_match, calib=calib)

    result = publisher.publish(match_key, record, final=final)
    if result.error:
        print(f"[publish] WARNING: {result.error}", flush=True)
    else:
        print(f"[publish] wrote {len(result.wrote)} file(s) for {match_key}", flush=True)

    return record


def _print_fuel_deltas(record: dict) -> None:
    """Print per-alliance fuel vision-vs-TBA comparison from the published record."""
    per_alliance = record.get("per_alliance", {})
    print("\n=== Fuel Vision-vs-TBA Deltas ===", flush=True)
    for alliance in ("blue", "red"):
        a = per_alliance.get(alliance, {})
        comp = a.get("comparison", {})
        if comp:
            print(f"  [{alliance.upper()}]", flush=True)
            for label in ("fuel_auto", "fuel_teleop", "fuel_total"):
                c = comp.get(label, {}).get("comparison", {})
                v_est = c.get("vision_est")
                tba_act = c.get("tba_actual")
                delta = c.get("delta")
                pct = c.get("pct_error")
                pct_str = f"{pct:+.1f}%" if pct is not None else "N/A"
                print(f"    {label}: vision={v_est}  tba={tba_act}  "
                      f"delta={delta}  pct_err={pct_str}", flush=True)
        else:
            # No TBA comparison block — print raw vision estimates
            vision = a.get("vision", {})
            print(f"  [{alliance.upper()}] (no TBA comparison available)", flush=True)
            for key in ("auto_fuel", "teleop_fuel", "total_fuel"):
                v = vision.get(key, {})
                print(f"    {key}: vision={v.get('value')}  "
                      f"available={v.get('available')}", flush=True)
    print("=================================\n", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run(args) -> int:
    import cv2

    from frcscout.pipeline.scout_pipeline import ScoutPipeline
    from live.match_segment import MatchSegmenter
    from publish.repo_writer import LivePublisher, PublishConfig

    # ── Config + calibration ──────────────────────────────────────────────────
    print(f"[live] source={args.source} event={args.event_key} "
          f"store={args.store} frames={args.frames} device={args.device}", flush=True)
    cfg = _load_cfg(args.device)
    calib = _load_calib(args.calib)

    # ── Open video source to get frame geometry / fps ──────────────────────
    # For files we use cv2 first to read metadata, then switch to FileSource for the
    # real-time cadence iteration. This mirrors how run_scout.run_sync works.
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[live] cannot open source: {args.source}")
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"[live] {W}x{H} fps={fps:.1f} total_frames={total}", flush=True)

    # ── Pipeline + segmenter + publisher ─────────────────────────────────────
    pipe = ScoutPipeline(cfg, frame_wh=(W, H), fps=fps, device=args.device)
    segmenter = MatchSegmenter(event_key=args.event_key)
    publisher = LivePublisher(PublishConfig(store_dir=args.store, push=False))

    # ── Overlay timeseries accumulator (per-match, reset on each segment) ──
    overlay_timeseries: list[dict] = []
    # Full, never-reset copy of every overlay row — dumped at the end so the fuel/score
    # OCR series can be inspected + fuel logic iterated OFFLINE without a full re-run.
    all_overlay_rows: list[dict] = []

    # ── Snapshot timer ────────────────────────────────────────────────────────
    snapshot_every_s = float(args.snapshot_every)
    last_snapshot_t = time.perf_counter()

    # Determine the initial fallback match_key (used when segmenter cannot identify).
    # Priority: explicit --match-key > filename inference > <event_key>_live
    if getattr(args, "match_key", None):
        cur_seg_match_key: str = args.match_key
    else:
        cur_seg_match_key = _infer_match_key(args.source, args.event_key)
    print(f"[live] initial match_key={cur_seg_match_key} "
          f"(will be overridden by segmenter if OCR identifies the match)", flush=True)
    last_record = None

    # ── Main frame loop ───────────────────────────────────────────────────────
    # We use cv2.VideoCapture directly (not FileSource) because the --frames cap and
    # per-frame index tracking are most straightforward with direct cap.read(). The
    # FileSource path is equivalent but adds threading overhead; for the demo/smoke
    # path the synchronous loop is more debuggable.
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[live] cannot re-open source: {args.source}")
        pipe.close()
        return 2

    start_frame = int(getattr(args, "start", 0) or 0)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"[live] seeking to start frame {start_frame}", flush=True)

    n_frames = 0
    t0 = time.perf_counter()
    max_frames = int(args.frames) if args.frames and int(args.frames) > 0 else None

    try:
        while True:
            if max_frames is not None and n_frames >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx = start_frame + n_frames
            n_frames += 1

            # ── Process frame ──────────────────────────────────────────────
            pipe.process(frame, frame_idx)

            # ── Build OverlayObs + accumulate timeseries ───────────────────
            obs = _ov_to_obs(pipe.cur_ov, frame_idx)
            ov = pipe.cur_ov
            overlay_timeseries.append({
                "frame_idx": frame_idx,
                "t_s": getattr(ov, "timer_s", None),
                "phase": getattr(ov, "phase", None),
                "blue_fuel": getattr(ov, "blue_fuel", None),
                "red_fuel": getattr(ov, "red_fuel", None),
                "blue_score": getattr(ov, "blue_score", None),
                "red_score": getattr(ov, "red_score", None),
            })
            all_overlay_rows.append(overlay_timeseries[-1])

            # ── Segmenter: check for finalized match ───────────────────────
            seg = segmenter.push(obs)
            if seg is not None:
                # A match segment just closed.
                seg_key = seg.match_key or cur_seg_match_key
                print(f"[seg] match closed: {seg_key} "
                      f"frames={seg.start_idx}..{seg.end_idx} "
                      f"id_confidence={seg.id_confidence:.2f}", flush=True)
                record = _build_and_publish(
                    pipe, seg_key, args.event_key, publisher, calib,
                    overlay_timeseries, final=True)
                last_record = record
                _print_fuel_deltas(record)
                # Reset per-match accumulators
                overlay_timeseries = []
                last_snapshot_t = time.perf_counter()

            # ── Update the running match_key from segmenter's in-progress seg ──
            if segmenter.cur is not None:
                # mid-match: use any partial identity the segmenter has accumulated
                obs_key = (segmenter.cur.match_key
                           if segmenter.cur.match_key
                           else cur_seg_match_key)
                cur_seg_match_key = obs_key

            # ── In-progress snapshot every N seconds ─────────────────────────
            now = time.perf_counter()
            if (segmenter.state == "IN_MATCH"
                    and now - last_snapshot_t >= snapshot_every_s):
                # Snapshot: do NOT call runner.finalize in a side-effectful way — just
                # build a lightweight vision_metrics from the current runner state.
                try:
                    snap_record = _build_and_publish(
                        pipe, cur_seg_match_key, args.event_key, publisher, calib,
                        overlay_timeseries, final=False)
                    _ = snap_record  # snapshot only; no delta print for noise reduction
                    print(f"[snap] in-progress snapshot @ f{frame_idx}", flush=True)
                except Exception as snap_err:
                    print(f"[snap] snapshot error: {snap_err}", flush=True)
                last_snapshot_t = now

            if n_frames % 60 == 0:
                print(f"  f{frame_idx}: seg_state={segmenter.state} "
                      f"seg_key={cur_seg_match_key} "
                      f"ov_phase={getattr(ov, 'phase', None)} "
                      f"ov_timer={getattr(ov, 'timer_s', None)}", flush=True)

    finally:
        cap.release()
        pipe.close()

    dt = time.perf_counter() - t0
    print(f"\n[live] processed {n_frames} frames in {dt:.1f}s "
          f"({n_frames / dt:.1f} fps)", flush=True)

    # ── Flush: handle an unclosed match at stream end ──────────────────────
    seg = segmenter.flush()
    if seg is not None:
        seg_key = seg.match_key or cur_seg_match_key
        print(f"[flush] match flushed: {seg_key} "
              f"id_confidence={seg.id_confidence:.2f}", flush=True)
        record = _build_and_publish(
            pipe, seg_key, args.event_key, publisher, calib,
            overlay_timeseries, final=True)
        last_record = record
        _print_fuel_deltas(record)
    elif segmenter.state == "IN_MATCH" and n_frames > 0:
        # Stream ended mid-match without a results screen AND segmenter didn't flush
        # (e.g. too few obs to transition). Publish whatever we have.
        print(f"[flush] stream ended mid-match (no results screen); publishing best-effort", flush=True)
        record = _build_and_publish(
            pipe, cur_seg_match_key, args.event_key, publisher, calib,
            overlay_timeseries, final=True)
        last_record = record
        _print_fuel_deltas(record)
    elif last_record is None and n_frames > 0:
        # No segment closed (short clip / no segmenter triggers) but we processed
        # frames. Publish the current pipeline state so the store is not empty.
        print(f"[flush] no segment closed; publishing current pipeline state as {cur_seg_match_key}", flush=True)
        record = _build_and_publish(
            pipe, cur_seg_match_key, args.event_key, publisher, calib,
            overlay_timeseries, final=True)
        last_record = record
        _print_fuel_deltas(record)

    if last_record is not None:
        print(f"[done] record match_key={last_record.get('match_key')}", flush=True)
    # Dump the full overlay OCR series for offline inspection / fuel-logic iteration.
    try:
        os.makedirs(args.store, exist_ok=True)
        ts_path = os.path.join(args.store, "overlay_timeseries.json")
        with open(ts_path, "w", encoding="utf-8") as fh:
            json.dump(all_overlay_rows, fh)
        print(f"[done] overlay series: {ts_path} ({len(all_overlay_rows)} rows)", flush=True)
    except Exception as _e:
        print(f"[warn] could not save overlay series: {_e}", flush=True)
    print(f"[done] store: {os.path.abspath(args.store)}", flush=True)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="FRC Vision Scout — live production pipeline")
    ap.add_argument("--source", default="data/footage/qual59.mp4",
                    help="video file path or YouTube URL")
    ap.add_argument("--event-key", dest="event_key", default="2026ctwat",
                    help="TBA event key (default 2026ctwat)")
    ap.add_argument("--match-key", dest="match_key", default=None,
                    help="Override match key (e.g. 2026ctwat_qm59). When absent, derived "
                         "from source filename (qualN -> qmN) or falls back to "
                         "<event_key>_live if OCR cannot identify.")
    ap.add_argument("--store", default="data/live_store",
                    help="publish store directory (default data/live_store)")
    ap.add_argument("--frames", type=int, default=0,
                    help="max frames to process (0=all)")
    ap.add_argument("--start", type=int, default=0,
                    help="seek to this source frame before processing (skip pre-match)")
    ap.add_argument("--calib", default=None,
                    help="CalibrationArtifact JSON path (optional)")
    ap.add_argument("--device", default="cpu",
                    help="YOLO device string (default cpu)")
    ap.add_argument("--snapshot-every", dest="snapshot_every",
                    type=float, default=SNAPSHOT_EVERY_S_DEFAULT,
                    help="seconds between in-progress snapshots (default 10)")
    args = ap.parse_args()

    # Resolve relative paths against the v2 root so the script can be run from
    # anywhere (mirrors run_scout's convention).
    if not os.path.isabs(args.source) and not ("://" in args.source):
        args.source = os.path.join(_V2, args.source)
    if not os.path.isabs(args.store):
        args.store = os.path.join(_V2, args.store)

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
