"""Release gate — a model does NOT ship unless it clears every threshold on a HELD-OUT,
HUMAN-LABELED match. This is the structural fix for v1, where every metric was measured
against the system's own outputs (teacher labels = grader, tuning set = test set).

Thresholds live in configs/eval.yaml. `--report` prints the table; `--enforce` exits
nonzero on any failure (use in CI).

Gates:
    ball_map50        ball mAP@50 vs HUMAN labels (not teacher labels)
    int8_recall_ratio INT8 recall / FP32 recall  (catches small-object PTQ collapse)
    count_mae         CV ball-count error vs the OVERLAY count (free dense truth)
    robot_idf1        robot tracking IDF1 + ID-switch cap
    reproj_err_px     AprilTag corner reprojection error
    tba_match_acc     end-to-end teams+scores vs TBA, PLAYOFFS INCLUDED
    fps_p99_ms        sustained p99 frame time <= 16.6ms on the TARGET laptop
"""
from __future__ import annotations

import argparse
import json
import sys

from frcscout.io.config import Config


GATES = {
    "ball_map50":        (">=", 0.60),
    "int8_recall_ratio": (">=", 0.95),
    "count_mae":         ("<=", 3.0),
    "robot_idf1":        (">=", 0.80),
    "robot_id_switches": ("<=", 10),
    "reproj_err_px":     ("<=", 2.0),
    "tba_match_acc":     (">=", 0.95),
    "fps_p99_ms":        ("<=", 16.6),
}


def _cmp(op: str, val: float, thr: float) -> bool:
    return val >= thr if op == ">=" else val <= thr


def run(cfg: Config) -> dict:
    """Invoke each metric module on the held-out match + target-hardware bench json.

    Reads paths from eval.yaml.  Tolerates missing artifacts: if a required
    file does not exist, that gate is simply omitted from the returned dict
    (the reporting loop in main() marks it as NOT MEASURED).

    Gates wired here:
        ball_map50        – vs human labels
        int8_recall_ratio – passed through if cfg supplies both recall values
        count_mae         – cv counts vs overlay truth
        robot_idf1        – tracking IDF1 vs hand-labeled snippets
        robot_id_switches – companion to robot_idf1
        reproj_err_px     – AprilTag reprojection (reads from pose artifact)
        tba_match_acc     – end-to-end vs TBA (ALL comp levels)
        fps_p99_ms        – read from the laptop bench report JSON
    """
    from frcscout.eval.metrics import ball_map50, count_mae, robot_idf1, reproj_err_px
    from frcscout.eval import tba_verify
    import json
    from pathlib import Path

    measured: dict = {}

    # ------------------------------------------------------------------ #
    # ball_map50
    # ------------------------------------------------------------------ #
    pred_dir = cfg.get("predictions_dir")
    human_dir = cfg.get("human_labels_dir")
    if pred_dir and human_dir and Path(pred_dir).is_dir() and Path(human_dir).is_dir():
        try:
            measured["ball_map50"] = ball_map50(pred_dir, human_dir)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # int8_recall_ratio  – supplied externally (two scalars in bench report or cfg)
    # ------------------------------------------------------------------ #
    bench_path = cfg.get("fps_bench_report")
    if bench_path and Path(bench_path).exists():
        try:
            bench = json.loads(Path(bench_path).read_text())
            fps_p99 = bench.get("fps_p99_ms") or bench.get("p99_ms")
            if fps_p99 is not None:
                measured["fps_p99_ms"] = float(fps_p99)
            # int8 / fp32 recall may be stored in the same report
            int8_r = bench.get("int8_recall")
            fp32_r = bench.get("fp32_recall")
            if int8_r is not None and fp32_r is not None:
                from frcscout.eval.metrics import int8_recall_ratio
                measured["int8_recall_ratio"] = int8_recall_ratio(
                    float(int8_r), float(fp32_r)
                )
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # count_mae  – overlay truth JSON: list of {cv: int, overlay: int} or
    #              two parallel arrays keyed "cv_counts"/"overlay_counts"
    # ------------------------------------------------------------------ #
    overlay_path = cfg.get("overlay_truth")
    if overlay_path and Path(overlay_path).exists():
        try:
            ov = json.loads(Path(overlay_path).read_text())
            if isinstance(ov, dict):
                cv_counts = ov.get("cv_counts", [])
                overlay_counts = ov.get("overlay_counts", [])
            else:
                # list of {cv, overlay}
                cv_counts = [r["cv"] for r in ov]
                overlay_counts = [r["overlay"] for r in ov]
            measured["count_mae"] = count_mae(cv_counts, overlay_counts)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # robot_idf1 / robot_id_switches
    # Path convention: data/tracks/test_pred.json + data/tracks/test_gt.json
    # ------------------------------------------------------------------ #
    tracks_pred_path = cfg.get("robot_tracks_pred")
    tracks_gt_path = cfg.get("robot_tracks_gt")
    if (tracks_pred_path and tracks_gt_path
            and Path(tracks_pred_path).exists()
            and Path(tracks_gt_path).exists()):
        try:
            pred_tracks = json.loads(Path(tracks_pred_path).read_text())
            gt_tracks = json.loads(Path(tracks_gt_path).read_text())
            idf1, id_sw = robot_idf1(pred_tracks, gt_tracks)
            measured["robot_idf1"] = idf1
            measured["robot_id_switches"] = id_sw
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # reproj_err_px  – read from a serialised CameraPose JSON artifact
    # ------------------------------------------------------------------ #
    pose_path = cfg.get("pose_artifact")
    if pose_path and Path(pose_path).exists():
        try:
            from frcscout.io.types import CameraPose
            pose_data = json.loads(Path(pose_path).read_text())
            pose = CameraPose(**pose_data)
            # field_model is optional for this gate — if missing, use stored value
            measured["reproj_err_px"] = reproj_err_px(pose, {})
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # tba_match_acc
    # ------------------------------------------------------------------ #
    detected_path = cfg.get("detected_matches")
    tba_path = cfg.get("tba_truth")
    if (detected_path and tba_path
            and Path(detected_path).exists()
            and Path(tba_path).exists()):
        try:
            result = tba_verify.verify(detected_path, tba_path)
            acc = result.get("overall_accuracy")
            if acc is not None:
                measured["tba_match_acc"] = acc
        except Exception:
            pass

    return measured


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--enforce", action="store_true")
    a = ap.parse_args()
    measured = run(Config.load(a.config))
    failures = []
    rows = []
    for name, (op, thr) in GATES.items():
        if name not in measured:
            rows.append((name, "—", f"{op}{thr}", "NOT MEASURED"))
            failures.append(name)
            continue
        ok = _cmp(op, measured[name], thr)
        rows.append((name, measured[name], f"{op}{thr}", "PASS" if ok else "FAIL"))
        if not ok:
            failures.append(name)
    if a.report or not a.enforce:
        print(f"{'gate':22} {'measured':>10} {'threshold':>10}  status")
        for n, m, t, s in rows:
            print(f"{n:22} {str(m):>10} {t:>10}  {s}")
    print(json.dumps({"measured": measured, "failures": failures}, indent=2))
    if a.enforce and failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
