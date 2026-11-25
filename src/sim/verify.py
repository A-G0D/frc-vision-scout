"""
verify.py — verification harness for the FRC REBUILT (2026) playoff match
(Upper Bracket R4 M11) simulation.

Compares the simulated score timeline against:
  1. The per-second overlay oracle (from overlay_ts.json) at every second.
  2. The TBA final score (blue=268, red=506).

Interface (exact):
    verify(sim, overlay_ts, tba_final) ->
        {
          'score_timeline_mae':  float,   # mean |sim - oracle| per second
          'final_blue':          int,
          'final_red':           int,
          'final_vs_tba_ok':     bool,    # True iff BOTH final scores match TBA
          'timeline_accuracy_pct': float, # pct of seconds where |sim - oracle| <= 5
          'notes':               str,
        }

tba_final: {'blue': int, 'red': int}
sim: the dict returned by gamestate.simulate()
overlay_ts: the list of per-second dicts from overlay_ts.read_timeseries()
"""

from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# Tolerance thresholds for accuracy counting
# ---------------------------------------------------------------------------
_CLOSE_THRESH = 5       # |sim - oracle| <= 5 pts => "on target"
_TBA_TOLERANCE = 0      # must be exact match to TBA to count as ok


def verify(
    sim: dict[str, Any],
    overlay_ts: list[dict],
    tba_final: dict[str, int],
) -> dict[str, Any]:
    """Verify the simulation result against overlay oracle and TBA.

    Parameters
    ----------
    sim        : output of gamestate.simulate()
    overlay_ts : list of per-second dicts from overlay_ts.read_timeseries()
                 (may include the nested 'timeseries' key if loaded raw)
    tba_final  : {'blue': int, 'red': int}

    Returns
    -------
    {
      'score_timeline_mae':    float,
      'final_blue':            int,
      'final_red':             int,
      'final_vs_tba_ok':       bool,
      'timeline_accuracy_pct': float,
      'notes':                 str,
    }
    """
    # Handle both raw JSON form (with nested 'timeseries') and plain list.
    ts_list: list[dict]
    if isinstance(overlay_ts, dict):
        ts_list = overlay_ts.get("timeseries", [])
    else:
        ts_list = list(overlay_ts)

    # Build a fast lookup: t -> (blue_score, red_score) for oracle.
    # Only include rows where at least one score is non-None and read_ok.
    oracle: dict[float, tuple[int, int]] = {}
    # Also track running last-known for forward-fill.
    _last_b, _last_r = 0, 0
    for row in sorted(ts_list, key=lambda r: r.get("t", 0)):
        b = row.get("blue_score")
        r = row.get("red_score")
        if b is None:
            b = _last_b
        else:
            _last_b = int(b)
            b = _last_b
        if r is None:
            r = _last_r
        else:
            _last_r = int(r)
            r = _last_r
        # Only include "meaningful" times (skip t=0 which may be stale carry-in).
        oracle[float(row.get("t", 0))] = (b, r)

    # Build sim timeline lookup: t -> (blue_score, red_score).
    sim_timeline = sim.get("timeline", [])
    sim_lookup: dict[float, tuple[int, int]] = {}
    for entry in sim_timeline:
        sim_lookup[float(entry.get("t", 0))] = (
            int(entry.get("blue_score", 0)),
            int(entry.get("red_score", 0)),
        )

    # Match oracle seconds to sim seconds.
    # For oracle times with a live_bar reading (read_ok), compare to sim.
    live_seconds: list[float] = []
    for row in sorted(ts_list, key=lambda r: r.get("t", 0)):
        if row.get("live_bar", False) and row.get("read_ok", False):
            live_seconds.append(float(row.get("t", 0)))

    errors_b: list[float] = []
    errors_r: list[float] = []
    close_count = 0
    total_count = 0

    for t in live_seconds:
        ob, or_ = oracle.get(t, (None, None))
        if ob is None or or_ is None:
            continue
        # Find nearest sim entry.
        sb, sr = _nearest_sim(sim_lookup, t)
        if sb is None:
            continue
        eb = abs(sb - ob)
        er = abs(sr - or_)
        errors_b.append(eb)
        errors_r.append(er)
        # Count as "on target" if BOTH alliances are within threshold.
        if eb <= _CLOSE_THRESH and er <= _CLOSE_THRESH:
            close_count += 1
        total_count += 1

    all_errors = errors_b + errors_r
    mae = float(sum(all_errors) / len(all_errors)) if all_errors else 0.0
    timeline_acc_pct = (100.0 * close_count / total_count) if total_count > 0 else 0.0

    # Final scores.
    sim_final = sim.get("final", {})
    final_blue = int(sim_final.get("blue", 0))
    final_red = int(sim_final.get("red", 0))
    tba_b = int(tba_final.get("blue", 268))
    tba_r = int(tba_final.get("red", 506))
    final_ok = (abs(final_blue - tba_b) <= _TBA_TOLERANCE and
                abs(final_red - tba_r) <= _TBA_TOLERANCE)

    # Build notes.
    lines = [
        f"Compared {total_count} live-bar oracle seconds.",
        f"Score MAE (per-second, both alliances combined): {mae:.2f} pts",
        f"Timeline accuracy (|sim-oracle|<=5 for BOTH alliances): "
        f"{timeline_acc_pct:.1f}% ({close_count}/{total_count})",
        f"Sim final: blue={final_blue}, red={final_red}",
        f"TBA final: blue={tba_b}, red={tba_r}",
        f"Final vs TBA: {'MATCH' if final_ok else 'MISMATCH'}",
    ]

    # Per-robot summary.
    per_robot = sim.get("per_robot", {})
    if per_robot:
        lines.append("Per-robot totals:")
        for rid, info in sorted(per_robot.items(),
                                key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0]):
            al = info.get("alliance", "?")
            pts = info.get("points", 0)
            shots = info.get("shots", 0)
            pickups = info.get("pickups", 0)
            acc = info.get("accuracy", 0.0)
            lines.append(
                f"  Robot {rid} ({al}): {pts} pts, {shots} shots, "
                f"{pickups} pickups, acc={acc:.3f}"
            )
    else:
        lines.append("Per-robot: (no per-robot data — events were empty or unattributed)")

    notes = "\n".join(lines)
    return {
        "score_timeline_mae": round(mae, 3),
        "final_blue": final_blue,
        "final_red": final_red,
        "final_vs_tba_ok": final_ok,
        "timeline_accuracy_pct": round(timeline_acc_pct, 2),
        "notes": notes,
    }


def _nearest_sim(
    sim_lookup: dict[float, tuple[int, int]],
    t: float,
    tolerance: float = 1.5,
) -> tuple[int | None, int | None]:
    """Return the sim entry nearest to t (within tolerance seconds)."""
    if t in sim_lookup:
        return sim_lookup[t]
    best_dt = 1e9
    best = (None, None)
    for st, sv in sim_lookup.items():
        dt = abs(st - t)
        if dt < best_dt and dt <= tolerance:
            best_dt = dt
            best = sv
    return best


# ---------------------------------------------------------------------------
# CLI: run verify against stored data files
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    _HERE = Path(__file__).resolve()
    _SRC  = _HERE.parent.parent
    _ROOT = _SRC.parent
    _DATA = _ROOT / "data" / "sim"

    overlay_path = _DATA / "overlay_ts.json"
    gamestate_path = _DATA / "gamestate.json"

    with open(overlay_path) as f:
        ots_raw = json.load(f)
    with open(gamestate_path) as f:
        gs_raw = json.load(f)

    # gamestate.json stores per_robot with string keys; fix for simulate output shape.
    sim_result = {
        "timeline": gs_raw.get("timeline", []),
        "final": gs_raw.get("final", {"blue": 0, "red": 0}),
        "per_robot": {
            int(k): v for k, v in gs_raw.get("per_robot", {}).items()
        },
    }

    result = verify(sim_result, ots_raw, {"blue": 268, "red": 506})

    print("=" * 60)
    print("VERIFICATION REPORT")
    print("=" * 60)
    print(result["notes"])
    print("-" * 60)
    print(f"score_timeline_mae:      {result['score_timeline_mae']:.3f}")
    print(f"timeline_accuracy_pct:   {result['timeline_accuracy_pct']:.2f}%")
    print(f"final_blue:              {result['final_blue']}")
    print(f"final_red:               {result['final_red']}")
    print(f"final_vs_tba_ok:         {result['final_vs_tba_ok']}")
    print("=" * 60)
