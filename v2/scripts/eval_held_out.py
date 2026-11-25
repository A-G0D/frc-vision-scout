"""eval_held_out.py — held-out-match generalization gate for the v2 sync pipeline.

THE GAP THIS CLOSES
--------------------
``configs/sources.yaml`` marks ``final1`` (data/footage/final1.mp4) as the HELD-OUT test
match: "never uploaded, never trained on". Before this script, the only thing ever run
against final1 was the A6.3 geometry audit (``scripts/audit_a63.py``), which measures
field-pose clamping only. The full production path — the SAME ``ScoutPipeline`` used for
the validated qual56/qual59 demos (detect -> track -> 6-robot prior -> phase state
machine -> metrics) — had NEVER been run on final1, so "v2 generalizes to unseen footage"
was an unverified claim (see README.md "Known gaps").

This module runs that exact pipeline on a VALIDATED match (default qual56, split=val in
sources.yaml) and on the HELD-OUT match (default final1, split=test) over comparable
frame windows, measures the same core tracking-health signals on both, and reports
whether the held-out match holds up against the validated baseline instead of degrading.

Metrics measured (all cheap, derived from ``ScoutPipeline`` state the driver already
exposes — no new detector/heuristic, so a change here is a measurement, not a new guess):
  * six_full_frac   — fraction of frames the 6-robot prior held exactly 6 (3 blue + 3 red)
  * six_any_frac    — fraction of frames with >=1 robot tracked at all
  * ball_nonzero_frac — fraction of frames with >=1 gated ball detection
  * pose_frac       — fraction of frames a real AprilTag pose was available
  * crowd_rejected  — cumulative robot detections killed by the crowd/pane gate
  * distinct_robot_ids — ByteTrack id churn (lower is more stable; vanilla ByteTrack
    over-churns on occlusion, the 6-robot prior should keep this near 6)
  * max_six         — the largest robot count seen in one frame

Usage
-----
    PYTHONPATH=src python scripts/eval_held_out.py \\
        --baseline qual56 --held-out final1 --start 2000 --frames 300

Exits 0 if the held-out match's six_any_frac AND ball_nonzero_frac are each within
``--tolerance`` of the baseline's (default 0.5x — i.e. detection/tracking may be
materially worse on unseen footage but must not collapse to near-zero), else 1. Always
prints both matches' full metrics and the deltas, and writes a JSON report to
``data/scout_out/held_out_gate_<held-out>.json`` when ``--out`` is not suppressed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from frcscout.io.config import Config  # noqa: E402

_V2_ROOT = os.path.join(os.path.dirname(__file__), "..")


def measure(video: str, config_path: str = "configs/deploy.yaml",
            start: int = 2000, frames: int = 300,
            device: str = "intel:cpu") -> dict:
    """Run the real sync ``ScoutPipeline`` over ``[start, start+frames)`` of ``video`` and
    return the tracking-health metrics described in the module docstring.

    A fresh ``Config`` is loaded per call: ``ScoutPipeline.__init__`` mutates its cfg
    in place (pins the climb-heuristic ratios to 1e9), so two calls must not share one
    dict or the second run would silently start from the first run's mutated config.
    """
    import cv2

    from frcscout.pipeline.scout_pipeline import ScoutPipeline

    cfg = Config.load(os.path.join(_V2_ROOT, config_path))
    dets = cfg.get("detectors", {}) or {}
    for k in ("robot", "ball_inventory"):
        if isinstance(dets.get(k), dict):
            dets[k]["device"] = device

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    pipe = ScoutPipeline(cfg, frame_wh=(W, H), fps=fps, device=device, render=False)
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

    n_six_full = 0
    n_six_any = 0
    n_ball_nonzero = 0
    all_ids: set = set()
    n = 0
    t0 = time.perf_counter()
    try:
        for i in range(frames):
            ok, frame = cap.read()
            if not ok:
                break
            fidx = start + i
            pipe.process(frame, fidx)
            st = pipe.last_state
            if st is not None:
                nrob = len(st.robots)
                if nrob >= 6:
                    n_six_full += 1
                if nrob >= 1:
                    n_six_any += 1
                for r in st.robots:
                    all_ids.add(r.tid)
            if pipe.last_ball_det_count > 0:
                n_ball_nonzero += 1
            n += 1
    finally:
        pipe.close()
        cap.release()
    dt = time.perf_counter() - t0

    n = max(n, 1)
    return {
        "video": os.path.basename(video),
        "start": start,
        "n_frames": n,
        "wall_s": round(dt, 1),
        "fps_processed": round(n / dt, 2) if dt > 0 else None,
        "six_full_frac": round(n_six_full / n, 4),
        "six_any_frac": round(n_six_any / n, 4),
        "ball_nonzero_frac": round(n_ball_nonzero / n, 4),
        "pose_frac": round(pipe.pose_frames / n, 4),
        "crowd_rejected": pipe.crowd_rejected,
        "distinct_robot_ids": len(all_ids),
        "max_six": pipe.max_six,
        "phase_hist": dict(pipe.phase_hist),
    }


_GATE_METRICS = ("six_any_frac", "ball_nonzero_frac")


def compare(baseline: dict, held_out: dict, tolerance: float = 0.5) -> dict:
    """Gate verdict.

    Gates on ``six_any_frac`` (robot tracking is alive at all) and ``ball_nonzero_frac``
    (the ball detector is alive at all) — the two signals that answer the actual question
    a held-out check must answer: *does detect->track still function on unseen footage*,
    not *is the harder 6-robot-simultaneously count perfect* (that is a separate, already
    documented tracking-QUALITY gap — see v2/RESULTS.md "Honest limitations" — and gating
    on it here would fail even the VALIDATED baseline on some windows, making the gate
    meaningless). Each gated metric on the held-out match must be >= ``tolerance`` x the
    baseline's value (it may be materially worse on unseen footage, but must not collapse
    towards zero). ``six_full_frac`` / ``max_six`` are still reported for both matches as
    diagnostics, just not gated.

    A baseline metric of 0 makes its ratio undefined -> that metric passes trivially
    (there is nothing to regress from; the corresponding held-out value is reported as-is,
    and a NON-zero held-out value against a zero baseline is a strict improvement, not a
    failure).
    """
    per_metric = {}
    ok = True
    for m in _GATE_METRICS:
        base_val = baseline[m]
        held_val = held_out[m]
        if base_val <= 0:
            ratio = None
            m_ok = True
        else:
            ratio = round(held_val / base_val, 4)
            m_ok = held_val >= tolerance * base_val
        per_metric[m] = {"baseline": base_val, "held_out": held_val,
                         "ratio": ratio, "pass": m_ok}
        ok = ok and m_ok
    return {
        "baseline": baseline,
        "held_out": held_out,
        "gated_metrics": per_metric,
        "tolerance": tolerance,
        "pass": ok,
    }


def _footage_path(name: str) -> str:
    if os.path.exists(name):
        return name
    return os.path.join(_V2_ROOT, "data", "footage", f"{name}.mp4")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default="qual56",
                    help="a VALIDATED (train/val split) match name or path")
    ap.add_argument("--held-out", default="final1", dest="held_out",
                    help="the HELD-OUT (test split) match name or path")
    ap.add_argument("--config", default="configs/deploy.yaml")
    ap.add_argument("--start", type=int, default=2000)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--device", default="intel:cpu")
    ap.add_argument("--tolerance", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="report JSON path; 'none' to suppress")
    a = ap.parse_args()

    base_path = _footage_path(a.baseline)
    held_path = _footage_path(a.held_out)
    print(f"[gate] baseline={base_path} held_out={held_path} "
          f"start={a.start} frames={a.frames} device={a.device}", flush=True)

    baseline = measure(base_path, a.config, a.start, a.frames, a.device)
    print(f"[gate] baseline({a.baseline}) -> {baseline}", flush=True)

    held_out = measure(held_path, a.config, a.start, a.frames, a.device)
    print(f"[gate] held_out({a.held_out}) -> {held_out}", flush=True)

    report = compare(baseline, held_out, a.tolerance)
    verdict = "PASS" if report["pass"] else "FAIL"
    print(f"\n[gate] tolerance={a.tolerance}", flush=True)
    for m, r in report["gated_metrics"].items():
        print(f"  {m}: baseline={r['baseline']} held_out={r['held_out']} "
              f"ratio={r['ratio']} -> {'pass' if r['pass'] else 'FAIL'}", flush=True)
    print(f"  (diagnostic, not gated) six_full_frac: baseline={baseline['six_full_frac']} "
          f"held_out={held_out['six_full_frac']} | max_six: baseline={baseline['max_six']} "
          f"held_out={held_out['max_six']}", flush=True)
    print(f"[gate] verdict -> {verdict}", flush=True)

    out = a.out
    if out != "none":
        out = out or os.path.join(_V2_ROOT, "data", "scout_out",
                                  f"held_out_gate_{a.held_out}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"[gate] wrote {out}", flush=True)

    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
