"""Match-record exporter — the livestream → STORAGE endpoint.

Serializes a finalized match (per-frame state rolled up to per-robot scouting metrics +
alliance/match reconstruction + resolved team identities) to disk as:
  - match_<key>.json   : the full structured record (everything, for replay/analysis)
  - scouting_<key>.csv : one flat row per robot with the canonical objective scouting columns
                         (ScoutingPASS / Citrus-style), so it drops straight into a scouting DB.

Unwraps the metrics {value, available, needs} envelope to plain values for the CSV. Pure-python
(json + csv), no heavy deps.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

# Canonical per-robot scouting columns — the OBJECTIVE, vision-derivable subset of the
# real-scouter schema (subjective ratings like driver-skill are human-only, omitted here).
SCOUT_COLUMNS = [
    "team", "alliance",
    "fuel_scored", "scoring_rate_per_min", "mean_cycle_time_s",
    "intake_attempts", "intake_successes", "intake_success_rate", "mean_time_to_intake_s",
    "max_held", "balls_shot_at_once", "shot_attempts", "shots_made", "shot_accuracy",
    "possession_time_s", "climb_level", "defense_time_s", "jitter_index",
]


def _val(stats: dict, key: str):
    """Pull a metric value, unwrapping the {value, available, ...} envelope if present."""
    v = stats.get(key) if isinstance(stats, dict) else None
    if isinstance(v, dict) and "value" in v:
        return v["value"]
    return v


def write_match(match_record: dict, outdir: str = "data/scouting", match_key: str = "match") -> dict:
    """Write the full JSON + the per-robot CSV scouting sheet. Returns the output paths."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    jpath = out / f"match_{match_key}.json"
    cpath = out / f"scouting_{match_key}.csv"

    with open(jpath, "w") as f:
        json.dump(match_record, f, indent=1, default=str)

    robots = match_record.get("robots", {}) or {}
    with open(cpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCOUT_COLUMNS)
        w.writeheader()
        for rid, stats in robots.items():
            stats = stats if isinstance(stats, dict) else {}
            row = {c: _val(stats, c) for c in SCOUT_COLUMNS}
            row["team"] = stats.get("team") or rid           # real team# if resolved, else track id
            if row.get("alliance") is None:
                row["alliance"] = stats.get("alliance")
            w.writerow(row)

    return {"json": str(jpath), "csv": str(cpath), "n_robots": len(robots)}
