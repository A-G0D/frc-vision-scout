"""Validation layer: compares vision-system computed metrics against ground truth.

Given:
    (a) computed  – dict of per-match estimates from the vision pipeline
    (b) tba_path  – path to a TBA JSON export ({matches:[...]})
    (c) overlay_truth – optional per-frame fuel-count series for CV-count MAE

Produces a VALIDATION REPORT that honestly separates:
    - truth-checkable metrics: score error, winner accuracy, fuel totals, overlay count
    - unvalidated metrics: shot_accuracy, intake rates, sourcing mix, cycle time, etc.
      (TBA simply does not record them — we MUST NOT imply they're verified)

HONESTY REQUIREMENT:
    Many scouting metrics have no external source of truth.  They are listed under
    `unvalidated_metrics` with an explicit reason.  The report headline states
    exactly how many metrics are truth-checkable vs unvalidated.

Public API
----------
validate(computed, tba_path, overlay_truth=None) -> dict
format_report(report) -> str
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

from frcscout.eval.metrics import count_mae
from frcscout.eval.tba_verify import _load_tba, _match_key


# ---------------------------------------------------------------------------
# Metric honesty catalogue
# ---------------------------------------------------------------------------

# Metrics the vision system computes but for which TBA provides NO ground truth.
# Reason strings must be specific — not just "no TBA field."
_UNVALIDATED_METRICS = [
    {
        "metric": "shot_accuracy",
        "reason": (
            "TBA records only total fuel scored — it does NOT record shots attempted,"
            " so miss rate / shot accuracy cannot be verified against any external source."
        ),
    },
    {
        "metric": "intake_rate",
        "reason": (
            "TBA has no field for pick-up events or intake timing; intake rate is"
            " entirely inferred by the vision pipeline with no independent ground truth."
        ),
    },
    {
        "metric": "cycle_time",
        "reason": (
            "Cycle time (pickup→score) is a vision-pipeline-internal timing estimate;"
            " TBA does not record individual cycle events."
        ),
    },
    {
        "metric": "sourcing_mix",
        "reason": (
            "Fuel-source breakdown (DEPOT vs OUTPOST/human vs floor) is not in TBA."
            " The vision system infers sourcing from zone geometry — not externally verifiable."
        ),
    },
    {
        "metric": "fuel_into_inactive_hub",
        "reason": (
            "TBA's score fields count only scoring-eligible (active-HUB) fuel."
            " Fuel deposited into an inactive HUB scores 0 pts and is not recorded by TBA."
        ),
    },
    {
        "metric": "active_shift_efficiency",
        "reason": (
            "Requires per-shift breakdown of fuel scored vs total fuel attempted;"
            " TBA provides only the aggregate match total, not shift-level splits."
        ),
    },
    {
        "metric": "climb_start_time",
        "reason": (
            "Climb timing within the match is not in TBA; TBA records only the final"
            " climb level achieved."
        ),
    },
    {
        "metric": "foul_candidates",
        "reason": (
            "Foul candidates are advisory-only — referees never consult video"
            " (2026 Game Manual §6.7). TBA records assessed fouls, but the vision system"
            " only flags candidate events, not confirmed calls."
        ),
    },
    {
        "metric": "robot_defense_time",
        "reason": (
            "Time spent in opponent zone is vision-pipeline-inferred via field geometry;"
            " TBA does not record defensive behavior at all."
        ),
    },
    {
        "metric": "auto_mobility",
        "reason": (
            "AUTO line-crossing detection is vision-inferred; TBA does not expose"
            " per-robot mobility flags in the standard match JSON format."
        ),
    },
]

# Metrics that ARE truth-checkable (used for the headline count)
_VALIDATED_METRIC_NAMES = [
    "blue_score_est",
    "red_score_est",
    "winner",
    "auto_fuel",
    "teleop_fuel",
    "total_fuel",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pct_err(est: float, true: float) -> Optional[float]:
    """Signed percentage error: (est - true) / true * 100.  None if true == 0."""
    if true == 0:
        return None
    return (est - true) / true * 100.0


def _abs_err(est: float, true: float) -> float:
    return abs(est - true)


def _winner(blue_score: Optional[float], red_score: Optional[float]) -> Optional[str]:
    """Return 'blue', 'red', or 'tie'.  None if either score is missing."""
    if blue_score is None or red_score is None:
        return None
    if blue_score > red_score:
        return "blue"
    if red_score > blue_score:
        return "red"
    return "tie"


def _safe_mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# Core validate function
# ---------------------------------------------------------------------------

def validate(
    computed: dict[str, Any],
    tba_path: str,
    overlay_truth: Optional[dict] = None,
) -> dict:
    """Compare vision-system estimates against TBA ground truth.

    Parameters
    ----------
    computed : dict
        Output of the vision pipeline.  Expected sub-keys:

        ``matches`` (list of per-match dicts):
            Each entry must have a ``key`` field (matching TBA match key format:
            e.g. 'qm1', 'sf1m1') and any of:
                blue_score_est, red_score_est
                blue_auto_fuel, red_auto_fuel
                blue_teleop_fuel, red_teleop_fuel
                blue_total_fuel, red_total_fuel
                blue_estimated_match_points, red_estimated_match_points

        ``alliances`` (dict, optional):
            Aggregate per-alliance totals (keyed 'blue'/'red'), each optionally
            containing: auto_fuel, teleop_fuel, total_fuel, estimated_match_points.

    tba_path : str
        Path to a TBA JSON file with format ``{matches:[...]}`` or a bare list.
        Each TBA match dict must have: blue_score, red_score (for score comparison).
        Optional: blue_fuel / red_fuel for fuel total comparison.

    overlay_truth : dict, optional
        If given, must have keys ``cv_counts`` and ``overlay_counts`` (parallel
        integer lists, one entry per frame).  Used to compute overlay count MAE
        via metrics.count_mae.

    Returns
    -------
    dict with keys:
        per_match       – list of per-match comparison rows
        aggregates      – score MAE, %err, winner accuracy, fuel %err
        overlay         – count MAE + n (None if overlay_truth not given)
        validated_metrics   – list of metric names that were truth-checked
        unvalidated_metrics – list of {metric, reason} for metrics with no ground truth
        headline        – summary string stating validated vs unvalidated counts
    """
    tba_matches = _load_tba(Path(tba_path))

    # Build lookup: match_key -> TBA row
    tba_by_key: dict[str, dict] = {}
    for m in tba_matches:
        k = _match_key(m)
        if k:
            tba_by_key[k] = m

    # --- Per-match comparison ---
    computed_matches: list[dict] = computed.get("matches", [])

    per_match: list[dict] = []
    score_abs_errors: list[float] = []
    score_pct_errors: list[float] = []
    fuel_pct_errors: list[float] = []
    winner_correct: list[bool] = []
    validated_metric_names: set[str] = set()

    for cm in computed_matches:
        key = cm.get("key")
        if not key:
            continue

        tba_row = tba_by_key.get(key)
        row: dict[str, Any] = {"key": key}

        # Score comparison
        for alliance in ("blue", "red"):
            est_key = f"{alliance}_score_est"
            true_key = f"{alliance}_score"
            est = cm.get(est_key)
            true = tba_row.get(true_key) if tba_row else None

            row[f"{alliance}_score_est"] = est
            row[f"{alliance}_score_true"] = true

            if est is not None and true is not None:
                err = _abs_err(est, true)
                pct = _pct_err(est, true)
                row[f"{alliance}_score_err"] = err
                row[f"{alliance}_score_pct_err"] = pct
                score_abs_errors.append(err)
                if pct is not None:
                    score_pct_errors.append(abs(pct))
                validated_metric_names.add(f"{alliance}_score_est")
            else:
                row[f"{alliance}_score_err"] = None
                row[f"{alliance}_score_pct_err"] = None

        # Winner comparison
        est_winner = _winner(
            cm.get("blue_score_est"),
            cm.get("red_score_est"),
        )
        true_winner = (
            _winner(tba_row.get("blue_score"), tba_row.get("red_score"))
            if tba_row
            else None
        )
        row["est_winner"] = est_winner
        row["true_winner"] = true_winner

        if est_winner is not None and true_winner is not None:
            correct = est_winner == true_winner
            row["winner_correct"] = correct
            winner_correct.append(correct)
            validated_metric_names.add("winner")
        else:
            row["winner_correct"] = None

        # Fuel comparison (where TBA provides it)
        for alliance in ("blue", "red"):
            for phase in ("auto", "teleop", "total"):
                est_key = f"{alliance}_{phase}_fuel"
                est_fuel = cm.get(est_key)

                # TBA may store fuel as blue_fuel / red_fuel (total only)
                if phase == "total":
                    true_fuel = tba_row.get(f"{alliance}_fuel") if tba_row else None
                else:
                    # TBA does NOT provide auto/teleop fuel split — unvalidatable
                    true_fuel = None

                row[f"{alliance}_{phase}_fuel_est"] = est_fuel
                row[f"{alliance}_{phase}_fuel_true"] = true_fuel

                if est_fuel is not None and true_fuel is not None:
                    fuel_err_pct = _pct_err(est_fuel, true_fuel)
                    row[f"{alliance}_{phase}_fuel_pct_err"] = fuel_err_pct
                    if fuel_err_pct is not None:
                        fuel_pct_errors.append(abs(fuel_err_pct))
                    validated_metric_names.add(f"{alliance}_{phase}_fuel")
                else:
                    row[f"{alliance}_{phase}_fuel_pct_err"] = None

        row["tba_found"] = tba_row is not None
        per_match.append(row)

    # --- Aggregates ---
    score_mae = _safe_mean(score_abs_errors)
    score_pct_err_mean = _safe_mean(score_pct_errors)
    winner_accuracy = (
        sum(winner_correct) / len(winner_correct) if winner_correct else None
    )
    fuel_pct_err_mean = _safe_mean(fuel_pct_errors) if fuel_pct_errors else None

    aggregates: dict[str, Any] = {
        "score_mae": score_mae,
        "score_pct_err": score_pct_err_mean,
        "winner_accuracy": winner_accuracy,
        "fuel_pct_err": fuel_pct_err_mean,
        "n_matches_compared": len(per_match),
        "n_matches_tba_found": sum(1 for r in per_match if r.get("tba_found")),
    }

    # --- Overlay count comparison ---
    overlay_result: Optional[dict] = None
    if overlay_truth is not None:
        cv_counts: list[int] = overlay_truth.get("cv_counts", [])
        ov_counts: list[int] = overlay_truth.get("overlay_counts", [])
        n = min(len(cv_counts), len(ov_counts))
        if n > 0:
            mae = count_mae(cv_counts[:n], ov_counts[:n])
            overlay_result = {"count_mae": mae, "n": n}
            validated_metric_names.add("overlay_ball_count")
        else:
            overlay_result = {"count_mae": None, "n": 0}

    # --- Finalize validated / unvalidated lists ---
    # Union of names actually confirmed during this run
    final_validated = sorted(validated_metric_names)

    # Build headline
    n_validated = len(final_validated)
    n_unvalidated = len(_UNVALIDATED_METRICS)

    headline = (
        f"VALIDATION SUMMARY: {n_validated} metric(s) truth-checked against TBA / overlay "
        f"ground truth; {n_unvalidated} metric(s) have no external source of truth and are "
        f"listed as UNVALIDATED — their values are pipeline estimates only."
    )

    return {
        "per_match": per_match,
        "aggregates": aggregates,
        "overlay": overlay_result,
        "validated_metrics": final_validated,
        "unvalidated_metrics": _UNVALIDATED_METRICS,
        "headline": headline,
    }


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------

def format_report(report: dict) -> str:
    """Render a validation report dict as a human-readable string.

    No external dependencies — uses only the standard library.
    """
    lines: list[str] = []

    sep = "=" * 72
    thin = "-" * 72

    lines.append(sep)
    lines.append("FRC VISION SCOUT — VALIDATION REPORT")
    lines.append(sep)
    lines.append(report.get("headline", ""))
    lines.append("")

    # --- Per-match table ---
    per_match = report.get("per_match", [])
    if per_match:
        lines.append("PER-MATCH SCORE COMPARISON")
        lines.append(thin)
        hdr = (
            f"{'Key':<12} "
            f"{'B-est':>6} {'B-true':>6} {'B-err':>6} {'B-%err':>7}  "
            f"{'R-est':>6} {'R-true':>6} {'R-err':>6} {'R-%err':>7}  "
            f"{'Winner-est':<10} {'Winner-true':<11} {'W-ok'}"
        )
        lines.append(hdr)
        lines.append(thin)
        for row in per_match:
            def _fmt(v: Any, width: int = 6, prec: int = 1) -> str:
                if v is None:
                    return "N/A".rjust(width)
                if isinstance(v, float):
                    return f"{v:.{prec}f}".rjust(width)
                return str(v).rjust(width)

            key = str(row.get("key", "?"))
            b_est   = _fmt(row.get("blue_score_est"))
            b_true  = _fmt(row.get("blue_score_true"))
            b_err   = _fmt(row.get("blue_score_err"))
            b_pct   = _fmt(row.get("blue_score_pct_err"), width=7)
            r_est   = _fmt(row.get("red_score_est"))
            r_true  = _fmt(row.get("red_score_true"))
            r_err   = _fmt(row.get("red_score_err"))
            r_pct   = _fmt(row.get("red_score_pct_err"), width=7)
            w_est   = str(row.get("est_winner") or "N/A")
            w_true  = str(row.get("true_winner") or "N/A")
            w_ok_raw = row.get("winner_correct")
            w_ok    = "YES" if w_ok_raw is True else ("NO" if w_ok_raw is False else "N/A")

            lines.append(
                f"{key:<12} "
                f"{b_est} {b_true} {b_err} {b_pct}  "
                f"{r_est} {r_true} {r_err} {r_pct}  "
                f"{w_est:<10} {w_true:<11} {w_ok}"
            )
        lines.append("")

    # --- Aggregates ---
    agg = report.get("aggregates", {})
    lines.append("AGGREGATES")
    lines.append(thin)

    def _agg(label: str, val: Any, unit: str = "", prec: int = 3) -> str:
        if val is None:
            return f"  {label:<30} N/A"
        if isinstance(val, float):
            return f"  {label:<30} {val:.{prec}f}{unit}"
        return f"  {label:<30} {val}{unit}"

    lines.append(_agg("Score MAE", agg.get("score_mae"), " pts"))
    lines.append(_agg("Score mean |%err|", agg.get("score_pct_err"), " %"))
    lines.append(_agg("Winner accuracy", agg.get("winner_accuracy")))
    lines.append(_agg("Fuel mean |%err|", agg.get("fuel_pct_err"), " %"))
    lines.append(
        f"  {'Matches compared':<30} {agg.get('n_matches_compared', 0)}"
        f" (TBA match found: {agg.get('n_matches_tba_found', 0)})"
    )
    lines.append("")

    # --- Overlay ---
    ov = report.get("overlay")
    if ov is not None:
        lines.append("OVERLAY COUNT VALIDATION")
        lines.append(thin)
        mae = ov.get("count_mae")
        n = ov.get("n", 0)
        lines.append(
            f"  Overlay count MAE: "
            + (f"{mae:.3f} balls/frame" if mae is not None else "N/A")
            + f"  (n={n} frames)"
        )
        lines.append("")

    # --- Validated metrics ---
    validated = report.get("validated_metrics", [])
    lines.append(f"VALIDATED METRICS ({len(validated)} total)")
    lines.append(thin)
    if validated:
        for m in validated:
            lines.append(f"  [OK] {m}")
    else:
        lines.append("  (none — no ground truth could be matched)")
    lines.append("")

    # --- Unvalidated metrics ---
    unvalidated = report.get("unvalidated_metrics", [])
    lines.append(f"UNVALIDATED METRICS ({len(unvalidated)} total)")
    lines.append(thin)
    lines.append("  These metrics are pipeline estimates ONLY — no external source of truth.")
    lines.append("  Do NOT treat them as verified.")
    lines.append("")
    for entry in unvalidated:
        lines.append(f"  [UNVALIDATED] {entry['metric']}")
        lines.append(f"    Reason: {entry['reason']}")
    lines.append("")

    lines.append(sep)
    return "\n".join(lines)
