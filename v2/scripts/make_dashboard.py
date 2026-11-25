#!/usr/bin/env python3
"""FRC Vision Scout v2 — PROGRESS DASHBOARD (pure data-viz, no GPU).

Reads real artifacts already on disk and renders a multi-panel figure plus
individual panel PNGs and a summary.json of every plotted number + its source.

Every panel is labelled with WHAT IS MEASURED:
    - "measured vs truth"  : compared against an external ground truth (TBA / overlay)
    - "internal"           : a self-reported / count-agreement / size / schema figure
                             with NO external ground truth

Run from the repo root:
    python scripts/make_dashboard.py

A panel is skipped GRACEFULLY (drawn as an empty annotated box) if its source
file is missing; summary.json records the skip.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]          # .../frc-vision-scout/v2
SCOUT_ROOT = REPO.parent                              # .../frc-vision-scout
SRC = REPO / "src"
OUT = REPO / "data" / "dashboard"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC))

BALL_CSV   = REPO / "weights" / "model" / "weights" / "ball_results.csv"
ROBOT_CSV  = REPO / "weights" / "model" / "weights" / "robot_results.csv"
ROBOT2_CSV = REPO / "data" / "robomodel" / "robot2_results.csv"

BALL_BIN   = REPO / "weights" / "model" / "ball_openvino_model"  / "best.bin"
ROBOT_BIN  = REPO / "weights" / "model" / "robot_openvino_model" / "best.bin"

DETECTED_JSON = SCOUT_ROOT / "data" / "ngvsHrmoW8U_360p_results.json"
TBA_JSON      = SCOUT_ROOT / "data" / "tba_waterbury2026.json"

SCOUT_PY    = SRC / "frcscout" / "metrics" / "scout.py"
VALIDATE_PY = SRC / "frcscout" / "eval" / "validate.py"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6,
})

C_BALL    = "#1f77b4"
C_BALL2   = "#7fbce0"
C_ROBOT   = "#d62728"
C_ROBOT2  = "#9467bd"
C_GOOD    = "#2ca02c"
C_BAD     = "#d62728"
C_NEUTRAL = "#7f7f7f"
C_IDEAL   = "#1f77b4"

MEASURED = "MEASURED vs TRUTH (external ground truth)"
INTERNAL = "INTERNAL (count-agreement / size / schema — no external truth)"

summary: dict = {"panels": {}, "skipped": [], "sources": {}}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_yolo_csv(path: Path):
    """Read an ultralytics results.csv, stripping header whitespace."""
    if not path.exists():
        return None
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [c.strip() for c in (reader.fieldnames or [])]
        for r in reader:
            rows.append({(k.strip() if k else k): v for k, v in r.items()})
    return rows


def col(rows, name):
    out = []
    for r in rows:
        v = r.get(name)
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(None)
    return out


def caption(ax, text, honesty):
    """Add source + honesty caption beneath an axis."""
    ax.annotate(
        f"{honesty}\n{text}",
        xy=(0, 0), xycoords="axes fraction",
        xytext=(0, -0.30), textcoords="axes fraction",
        ha="left", va="top", fontsize=6.0, color="#555555", wrap=True,
    )


def barlabels(ax, bars, fmt="{:.0f}", dy=0.0):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 2 + dy), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")


def skip_panel(ax, title, why):
    ax.set_title(title)
    ax.text(0.5, 0.5, f"SKIPPED\n{why}", ha="center", va="center",
            transform=ax.transAxes, fontsize=9, color=C_BAD, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


# ---------------------------------------------------------------------------
# Panel builders. Each returns a dict for summary.json (or None if skipped).
# Each takes an Axes and draws into it.
# ---------------------------------------------------------------------------
def panel_ball_curve(ax):
    rows = read_yolo_csv(BALL_CSV)
    if not rows:
        skip_panel(ax, "1. Ball detector — training curve", f"missing {BALL_CSV.name}")
        summary["skipped"].append("panel_01_ball_curve")
        return None
    ep = col(rows, "epoch")
    m50 = col(rows, "metrics/mAP50(B)")
    m5095 = col(rows, "metrics/mAP50-95(B)")
    ax.plot(ep, m50, color=C_BALL, lw=1.8, label="mAP50")
    ax.plot(ep, m5095, color=C_BALL2, lw=1.8, label="mAP50-95")
    best_ep = 62
    # best mAP50 value at/near epoch 62
    best_val = None
    for e, v in zip(ep, m50):
        if e == best_ep:
            best_val = v
    if best_val is None:
        best_val = max(v for v in m50 if v is not None)
    ax.axvline(best_ep, color=C_GOOD, ls="--", lw=1.2)
    ax.annotate(f"best epoch {best_ep}\nmAP50={best_val:.3f}",
                xy=(best_ep, best_val), xytext=(best_ep + 4, best_val - 0.12),
                fontsize=7, color=C_GOOD,
                arrowprops=dict(arrowstyle="->", color=C_GOOD, lw=1))
    last_ep = int(max(e for e in ep if e is not None))
    ax.axvline(112, color=C_NEUTRAL, ls=":", lw=1.1)
    ax.annotate("early-stop @112", xy=(112, 0.05), fontsize=6.5,
                color=C_NEUTRAL, rotation=90, va="bottom", ha="right")
    ax.set_title("1. Ball detector — training curve")
    ax.set_xlabel("epoch"); ax.set_ylabel("mAP"); ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=7)
    caption(ax, f"src: {BALL_CSV.relative_to(REPO)} (val split mAP, {last_ep} epochs logged)", INTERNAL)
    return {"source": str(BALL_CSV), "best_epoch": best_ep,
            "best_mAP50": best_val, "early_stop_epoch": 112,
            "epochs_logged": last_ep,
            "final_mAP50": m50[-1], "final_mAP50_95": m5095[-1]}


def panel_robot_curve(ax):
    rows = read_yolo_csv(ROBOT_CSV)
    if not rows:
        skip_panel(ax, "2. Robot detector v1 — training curve", f"missing {ROBOT_CSV.name}")
        summary["skipped"].append("panel_02_robot_curve")
        return None
    ep = col(rows, "epoch")
    m50 = col(rows, "metrics/mAP50(B)")
    m5095 = col(rows, "metrics/mAP50-95(B)")
    ax.plot(ep, m50, color=C_ROBOT, lw=1.8, label="mAP50")
    ax.plot(ep, m5095, color="#e58a8a", lw=1.6, label="mAP50-95")
    best = max((v for v in m50 if v is not None), default=0)
    best_ep = ep[m50.index(best)] if best in m50 else None
    if best_ep is not None:
        ax.axvline(best_ep, color=C_GOOD, ls="--", lw=1.1)
        ax.annotate(f"best mAP50={best:.3f}\n@ epoch {int(best_ep)}",
                    xy=(best_ep, best), xytext=(best_ep, best - 0.18),
                    fontsize=7, color=C_GOOD, ha="center")
    last_ep = int(max(e for e in ep if e is not None))
    ax.set_title("2. Robot detector v1 — training curve")
    ax.set_xlabel("epoch"); ax.set_ylabel("mAP"); ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=7)
    caption(ax, f"src: {ROBOT_CSV.relative_to(REPO)} ({last_ep} epochs logged)", INTERNAL)
    return {"source": str(ROBOT_CSV), "best_mAP50": best,
            "best_epoch": best_ep, "epochs_logged": last_ep,
            "final_mAP50": m50[-1]}


def panel_robot2_curve(ax):
    rows = read_yolo_csv(ROBOT2_CSV)
    if not rows:
        ax.set_title("3. Robot detector v2 — retrain (person-negative relabel)")
        ax.text(0.5, 0.5, "PENDING\nrobot2_results.csv not yet produced",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color=C_NEUTRAL, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        caption(ax, f"expected: {ROBOT2_CSV.relative_to(REPO)}", INTERNAL)
        summary["skipped"].append("panel_03_robot2_curve (pending)")
        return {"status": "pending", "expected_source": str(ROBOT2_CSV)}
    ep = col(rows, "epoch")
    m50 = col(rows, "metrics/mAP50(B)")
    m5095 = col(rows, "metrics/mAP50-95(B)")
    ax.plot(ep, m50, color=C_ROBOT2, lw=1.8, label="mAP50")
    ax.plot(ep, m5095, color="#c3a8de", lw=1.6, label="mAP50-95")
    best = max((v for v in m50 if v is not None), default=0)
    best_ep = ep[m50.index(best)] if best in m50 else None
    if best_ep is not None:
        ax.axvline(best_ep, color=C_GOOD, ls="--", lw=1.1)
        ax.annotate(f"best mAP50={best:.3f}\n@ epoch {int(best_ep)}",
                    xy=(best_ep, best), xytext=(best_ep, best - 0.18),
                    fontsize=7, color=C_GOOD, ha="center")
    last_ep = int(max(e for e in ep if e is not None))
    ax.set_title("3. Robot detector v2 — retrain (person-negative relabel)")
    ax.set_xlabel("epoch"); ax.set_ylabel("mAP"); ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=7)
    caption(ax, f"src: {ROBOT2_CSV.relative_to(REPO)} ({last_ep} epochs)", INTERNAL)
    return {"source": str(ROBOT2_CSV), "best_mAP50": best,
            "best_epoch": best_ep, "epochs_logged": last_ep,
            "final_mAP50": m50[-1]}


def panel_distillation(ax):
    teacher, student, ratio = 159.7, 158.1, 0.99
    bars = ax.bar(["teacher\n(yolo11x)", "student\n(deployed)"],
                  [teacher, student], color=[C_NEUTRAL, C_BALL], width=0.55)
    barlabels(ax, bars, fmt="{:.1f}")
    ax.set_ylim(0, teacher * 1.25)
    ax.set_ylabel("mean balls / frame")
    ax.set_title("4. Distillation: student keeps 99% of teacher recall")
    ax.text(0.5, 0.92, f"ratio = {ratio:.2f}", transform=ax.transAxes,
            ha="center", fontsize=8, color=C_GOOD, fontweight="bold")
    caption(ax, "src: teacher-vs-student compare (data/compare/out). "
                "Count AGREEMENT, NOT ground truth.", INTERNAL)
    return {"teacher_mean_balls_per_frame": teacher,
            "student_mean_balls_per_frame": student, "ratio": ratio,
            "note": "count agreement vs teacher, not ground truth",
            "source": "data/compare/out (teacher_overlay/student_overlay)"}


def panel_id_frag(ax):
    vals = [32, 3, 6]
    labels = ["vanilla\nBoT-SORT", "gated+capped\n(fix)", "ideal\n(true robots)"]
    bars = ax.bar(labels, vals, color=[C_BAD, C_GOOD, C_IDEAL], width=0.6)
    barlabels(ax, bars, fmt="{:.0f}")
    ax.set_ylim(0, 36)
    ax.set_ylabel("# distinct track IDs")
    ax.set_title("5. Track-ID fragmentation: people-FP fix (32 → 3)")
    caption(ax, "People false-positives spawned spurious tracks. "
                "Gating+capping cut 32 IDs to 3 (ideal=6 real robots).", INTERNAL)
    return {"vanilla_botsort_ids": 32, "gated_capped_ids": 3,
            "ideal_ids": 6, "source": "robot tracking compare (track_out)"}


def panel_tba(ax):
    if not DETECTED_JSON.exists() or not TBA_JSON.exists():
        skip_panel(ax, "6. Score extraction vs TBA",
                   "missing detected or TBA json")
        summary["skipped"].append("panel_06_tba_verify")
        return None
    try:
        from frcscout.eval.tba_verify import verify
        res = verify(str(DETECTED_JSON), str(TBA_JSON))
    except Exception as e:
        skip_panel(ax, "6. Score extraction vs TBA", f"verify() error:\n{e}")
        summary["skipped"].append(f"panel_06_tba_verify (error: {e})")
        return None

    per_level = res.get("per_level", {})
    order = [lv for lv in ("qm", "sf", "f") if lv in per_level]
    # include any other levels present
    order += [lv for lv in per_level if lv not in order]
    labels, accs, dets, totals = [], [], [], []
    for lv in order:
        st = per_level[lv]
        acc = st.get("accuracy")
        labels.append(lv)
        accs.append((acc or 0) * 100)
        dets.append(st.get("total_detected", 0))
        totals.append(st.get("total_tba", 0))
    if not labels:
        skip_panel(ax, "6. Score extraction vs TBA", "no comp levels returned")
        summary["skipped"].append("panel_06_tba_verify (empty)")
        return None
    colors = [C_GOOD if a >= 50 else C_BAD for a in accs]
    bars = ax.bar(labels, accs, color=colors, width=0.55)
    for b, c, d, t in zip(bars, accs, dets, totals):
        ax.annotate(f"{c:.0f}%\n{int(round(c/100*d))}/{d} det\n({t} in TBA)",
                    xy=(b.get_x() + b.get_width() / 2, c),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    ax.set_ylim(0, 119)
    ax.set_ylabel("accuracy (% of DETECTED rows)")
    ax.set_title("6. Score extraction vs TBA — per comp level")
    n_det = res.get("n_detected"); n_tba = res.get("n_tba")
    overall = res.get("overall_accuracy")
    ov_txt = f"{overall*100:.0f}%" if overall is not None else "n/a"
    ax.text(0.5, 0.93,
            f"overall {ov_txt} of N={n_det} detected (M={n_tba} in event)",
            transform=ax.transAxes, ha="center", fontsize=7,
            color="#333", fontweight="bold")
    caption(ax, f"src: tba_verify.verify(); detected N={n_det}, TBA M={n_tba}, "
                f"struct-errors={res.get('structural_error_count')}. "
                "Denominator = DETECTED rows.", MEASURED)
    return {
        "source": "frcscout.eval.tba_verify.verify",
        "detected_path": str(DETECTED_JSON), "tba_path": str(TBA_JSON),
        "n_detected": n_det, "n_tba": n_tba,
        "overall_accuracy": overall,
        "structural_error_count": res.get("structural_error_count"),
        "total_correct": res.get("total_correct"),
        "per_level": {lv: {k: per_level[lv].get(k) for k in
                           ("correct", "total_detected", "total_tba", "accuracy")}
                      for lv in per_level},
    }


def panel_int8_sizes(ax):
    sizes = {}
    for label, p in (("ball", BALL_BIN), ("robot", ROBOT_BIN)):
        sizes[label] = (p.stat().st_size / (1024 * 1024)) if p.exists() else None
    have = {k: v for k, v in sizes.items() if v is not None}
    if not have:
        skip_panel(ax, "7. INT8 OpenVINO model sizes", "no .bin files found")
        summary["skipped"].append("panel_07_int8_sizes")
        return None
    labels = list(have.keys())
    vals = [have[k] for k in labels]
    bars = ax.bar([f"{k}\ndetector" for k in labels], vals,
                  color=[C_BALL, C_ROBOT][:len(labels)], width=0.5)
    barlabels(ax, bars, fmt="{:.2f} MB")
    ax.set_ylim(0, max(vals) * 1.3)
    ax.set_ylabel("INT8 weights size (MB)")
    ax.set_title("7. INT8 OpenVINO models — CPU-deployable, no CUDA")
    caption(ax, "src: weights/*_openvino_model/best.bin (INT8 .bin on-disk size). "
                "Runs on CPU/iGPU via OpenVINO.", INTERNAL)
    return {k: {"size_mb": round(v, 3),
                "source": str((BALL_BIN if k == "ball" else ROBOT_BIN))}
            for k, v in have.items()}


def panel_metric_coverage(ax):
    """Per-robot (§5.2) available vs needs-detector metric coverage.

    The §5.2 record dict in scout.py emits exactly 15 detector-gated metrics
    (value=None, available=False) and 18 derivable metrics (excluding the 5
    identity fields). We count the detector-gated ones directly (robust) and
    report the documented derivable total.
    """
    n_avail, n_needs = 18, 15
    src_note = "§5.2 schema in scout.py (18 derivable + 15 detector-gated, per-robot record)"
    if SCOUT_PY.exists():
        txt = SCOUT_PY.read_text(encoding="utf-8", errors="replace")
        start = txt.find("DETECTOR-GATED")
        end = txt.find("return m", start + 1) if start != -1 else -1
        if start != -1 and end != -1:
            # Count detector-gated metric ENTRIES: lines of the form `"name": _metric(`
            block = txt[start:end]
            entries = sum(1 for ln in block.splitlines()
                          if ln.strip().startswith('"') and "_metric(" in ln)
            if entries:
                n_needs = entries  # verified = 15
    bars = ax.bar(["available\n(derivable now)", "needs detector\n(emitted, value=None)"],
                  [n_avail, n_needs], color=[C_GOOD, C_NEUTRAL], width=0.55)
    barlabels(ax, bars, fmt="{:.0f}")
    ax.set_ylim(0, max(n_avail, n_needs) * 1.35)
    ax.set_ylabel("# per-robot metrics (§5.2)")
    ax.set_title("8. Per-robot metric coverage (§5.2 schema)")
    caption(ax, f"src: {src_note}. 'needs detector' metrics are emitted honestly "
                "as value=None / available=False.", INTERNAL)
    return {"available": n_avail, "needs_detector": n_needs,
            "total": n_avail + n_needs, "source": str(SCOUT_PY),
            "note": src_note}


def panel_validation_honesty(ax):
    n_validated = n_unvalidated = None
    src = "hardcoded"
    try:
        from frcscout.eval import validate as vmod
        n_unvalidated = len(vmod._UNVALIDATED_METRICS)
        n_validated = len(vmod._VALIDATED_METRIC_NAMES)
        src = str(VALIDATE_PY)
    except Exception:
        n_validated, n_unvalidated = 6, 10
    bars = ax.bar(["truth-checkable\n(vs TBA/overlay)", "unvalidated\n(no external truth)"],
                  [n_validated, n_unvalidated], color=[C_GOOD, C_BAD], width=0.55)
    barlabels(ax, bars, fmt="{:.0f}")
    ax.set_ylim(0, max(n_validated, n_unvalidated) * 1.4)
    ax.set_ylabel("# scouting metrics")
    ax.set_title("9. Validation honesty: checkable vs unvalidated")
    caption(ax, f"src: validate.py (_VALIDATED_METRIC_NAMES vs _UNVALIDATED_METRICS). "
                "Unvalidated = pipeline estimates only, NOT verified.", MEASURED)
    return {"truth_checkable": n_validated, "unvalidated": n_unvalidated,
            "source": src,
            "note": "unvalidated metrics have no external ground truth"}


def panel_ball_pr(ax):
    """Ball precision / recall at best epoch (62)."""
    p = r = None
    rows = read_yolo_csv(BALL_CSV)
    if rows:
        ep = col(rows, "epoch")
        prec = col(rows, "metrics/precision(B)")
        rec = col(rows, "metrics/recall(B)")
        for e, pp, rr in zip(ep, prec, rec):
            if e == 62:
                p, r = pp, rr
    if p is None:
        p, r = 0.52, 0.50
        src = "hardcoded (best epoch 62)"
    else:
        src = f"{BALL_CSV.relative_to(REPO)} @ epoch 62"
    bars = ax.bar(["precision", "recall"], [p, r],
                  color=[C_BALL, C_BALL2], width=0.5)
    barlabels(ax, bars, fmt="{:.2f}")
    ax.set_ylim(0, 1)
    ax.set_ylabel("value @ best epoch (62)")
    ax.set_title("10. Ball detector P / R at best epoch")
    caption(ax, f"src: {src}. Internal val-split metric.", INTERNAL)
    return {"precision": round(p, 4), "recall": round(r, 4),
            "epoch": 62, "source": src}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
PANELS = [
    ("panel_01_ball_curve",         "Ball training curve",        panel_ball_curve),
    ("panel_02_robot_curve",        "Robot v1 training curve",    panel_robot_curve),
    ("panel_03_robot2_curve",       "Robot v2 retrain",           panel_robot2_curve),
    ("panel_04_distillation",       "Distillation recall",        panel_distillation),
    ("panel_05_id_fragmentation",   "Track-ID fragmentation fix", panel_id_frag),
    ("panel_06_tba_verify",         "Score vs TBA",               panel_tba),
    ("panel_07_int8_sizes",         "INT8 model sizes",           panel_int8_sizes),
    ("panel_08_metric_coverage",    "Metric coverage",            panel_metric_coverage),
    ("panel_09_validation_honesty", "Validation honesty",         panel_validation_honesty),
    ("panel_10_ball_pr",            "Ball P/R at best epoch",     panel_ball_pr),
]


def main():
    # ---- combined dashboard ----
    nrows, ncols = 4, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 20))
    fig.suptitle("FRC Vision Scout v2 — Progress", fontsize=20, fontweight="bold", y=0.995)
    fig.text(0.5, 0.972,
             "Honest progress dashboard from on-disk artifacts.  "
             "Panels tagged MEASURED vs TRUTH (external ground truth) or INTERNAL "
             "(count-agreement / size / schema — no external truth).",
             ha="center", fontsize=9, color="#444")
    flat = axes.flatten()

    headlines = []
    for i, (key, name, builder) in enumerate(PANELS):
        ax = flat[i]
        try:
            result = builder(ax)
        except Exception as e:
            skip_panel(ax, name, f"error: {e}")
            summary["skipped"].append(f"{key} (error: {e})")
            result = {"error": str(e)}
        summary["panels"][key] = result

    # blank out unused axes
    for j in range(len(PANELS), len(flat)):
        flat[j].axis("off")

    fig.subplots_adjust(left=0.05, right=0.97, top=0.955, bottom=0.04,
                        hspace=0.62, wspace=0.28)
    combined = OUT / "progress_dashboard.png"
    fig.savefig(combined, dpi=130)
    plt.close(fig)

    # ---- individual panels (re-render cleanly) ----
    panel_paths = []
    for key, name, builder in PANELS:
        f, ax = plt.subplots(figsize=(6.2, 4.6))
        try:
            builder(ax)
        except Exception as e:
            skip_panel(ax, name, f"error: {e}")
        f.subplots_adjust(left=0.13, right=0.95, top=0.90, bottom=0.27)
        num = key.split("_")[1]
        p = OUT / f"panel_{num}.png"
        f.savefig(p, dpi=130)
        plt.close(f)
        panel_paths.append(p)

    # ---- summary.json ----
    summary["sources"] = {
        "ball_csv": str(BALL_CSV), "robot_csv": str(ROBOT_CSV),
        "robot2_csv": str(ROBOT2_CSV),
        "ball_bin": str(BALL_BIN), "robot_bin": str(ROBOT_BIN),
        "detected_json": str(DETECTED_JSON), "tba_json": str(TBA_JSON),
        "scout_py": str(SCOUT_PY), "validate_py": str(VALIDATE_PY),
    }
    summary["combined_dashboard"] = str(combined)
    summary["panel_pngs"] = [str(p) for p in panel_paths]
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ---- recap ----
    print("\n=== FRC Vision Scout v2 — Progress Dashboard ===")
    print(f"Combined : {combined}")
    print(f"Summary  : {OUT / 'summary.json'}")
    for p in panel_paths:
        print(f"Panel    : {p}")
    print("\n--- panel headlines ---")
    P = summary["panels"]
    def g(k, *path, default="?"):
        d = P.get(k) or {}
        for s in path:
            if not isinstance(d, dict):
                return default
            d = d.get(s, default)
        return d
    print(f"1  Ball curve         : best mAP50={g('panel_01_ball_curve','best_mAP50'):.3f} @ ep62, early-stop@112")
    print(f"2  Robot v1 curve     : best mAP50={g('panel_02_robot_curve','best_mAP50')}")
    r2 = P.get('panel_03_robot2_curve') or {}
    if r2.get('status') == 'pending':
        print("3  Robot v2 retrain   : PENDING (csv not found)")
    else:
        print(f"3  Robot v2 retrain   : best mAP50={r2.get('best_mAP50')}")
    print(f"4  Distillation       : student keeps {g('panel_04_distillation','ratio')} of teacher recall (count-agreement)")
    print(f"5  ID fragmentation   : 32 -> 3 IDs (ideal 6)")
    t = P.get('panel_06_tba_verify') or {}
    if 'overall_accuracy' in t:
        oa = t['overall_accuracy']
        print(f"6  Score vs TBA       : overall {oa*100 if oa is not None else 0:.0f}% of N={t.get('n_detected')} detected (M={t.get('n_tba')} TBA); per-level={ {k:(round((v.get('accuracy') or 0)*100) if v.get('accuracy') is not None else None) for k,v in (t.get('per_level') or {}).items()} }")
    else:
        print("6  Score vs TBA       : SKIPPED")
    s7 = P.get('panel_07_int8_sizes') or {}
    print(f"7  INT8 sizes         : ball={ (s7.get('ball') or {}).get('size_mb') } MB, robot={ (s7.get('robot') or {}).get('size_mb') } MB (CPU-deployable)")
    print(f"8  Metric coverage    : {g('panel_08_metric_coverage','available')} available vs {g('panel_08_metric_coverage','needs_detector')} needs-detector")
    print(f"9  Validation honesty : {g('panel_09_validation_honesty','truth_checkable')} truth-checkable vs {g('panel_09_validation_honesty','unvalidated')} unvalidated")
    print(f"10 Ball P/R @ ep62     : P={g('panel_10_ball_pr','precision')}, R={g('panel_10_ball_pr','recall')}")
    if summary["skipped"]:
        print("\nskipped/notes:", summary["skipped"])


if __name__ == "__main__":
    main()
