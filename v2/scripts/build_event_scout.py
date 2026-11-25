"""build_event_scout — roll every per-match scout JSON in an event into a per-TEAM
scouting table (data/scout_out/event_teams.json + event_teams.csv).

This is the EVENT-LEVEL scouting product (R7). It:

  1. globs ``data/scout_out/match_*.json`` (the files scripts/run_scout.py writes),
  2. ADAPTS each raw file into the ``match_record`` shape MetricsAggregator.aggregate
     consumes (raw files are track-keyed robots + an ``overlay`` block, NOT the
     events/possession shape the aggregator documents),
  3. runs ``aggregate()`` per match, then ``team_aggregates()`` across the event,
  4. computes OPR / DPR / CCWM from AUTHORITATIVE alliance scores (overlay or TBA),
     never from our fuel ESTIMATE — fuel is gated as not-truth,
  5. writes one row per TEAM (their aggregated metrics + OPR/DPR/CCWM) to JSON+CSV.

KEYING / ATTRIBUTION (the core fix this product needed)
-------------------------------------------------------
The raw match files carry ``team: null`` / ``id_method: "unknown"`` for every track
(the identity layer is not resolving teams on this footage yet). MetricsAggregator's
``team_aggregates`` now gates per-team offensive attribution on
``_identity_resolved`` (id_method in {anchor,elimination} AND id_confidence ≥ floor).
So unresolved tracks land in ``track:N`` buckets tagged
``attribution_level='alliance'`` with offense set available:False — they never get
credited to a guessed team number. When the overlay (or TBA) DOES give us an
authoritative alliance roster, this script additionally emits a ROSTER row per real
team (matches_played from the overlay rosters) so the picklist has every team, with
offense honestly gated as alliance-level until per-robot identity is resolved.

Run:
    PYTHONPATH=src python scripts/build_event_scout.py
    PYTHONPATH=src python scripts/build_event_scout.py --scout-dir data/scout_out \
        --tba data/tba/2026ctwat_matches.json --event 2026ctwat
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from frcscout.metrics import MetricsAggregator, MetricsConfig


# ───────────────────────────── raw-file adapters ─────────────────────────────

def _overlay_alliances(raw: dict) -> dict:
    """Authoritative rosters {alliance: [team,...]}. Prefer ``expected_alliances``
    (already color-resolved upstream), else the overlay teams lists."""
    exp = raw.get("expected_alliances") or {}
    ov = raw.get("overlay") or {}
    blue = exp.get("blue") or ov.get("blue_teams") or []
    red = exp.get("red") or ov.get("red_teams") or []
    return {"blue": [int(t) for t in blue if t is not None],
            "red": [int(t) for t in red if t is not None]}


def _to_match_record(raw: dict) -> dict:
    """Adapt a raw run_scout match_*.json into the aggregate() match_record shape.

    aggregate() wants ``robots`` whose values carry shots_made / shot_attempts /
    intake_* / possession_time_s / etc. (RobotStats.to_dict() + identity). The raw
    file's per-track dicts already carry those scalar keys at top level (plus a
    nested ``metrics`` mirror we ignore), so we pass them through, normalizing the
    intake_success_rate / shot_accuracy ENVELOPES back down to scalars/None the way
    RobotStats.to_dict() would, and stamping a stable ``track_id``.
    """
    robots: dict[str, dict] = {}
    for tid, r in (raw.get("robots") or {}).items():
        stats = dict(r)  # shallow copy; keep scalar possession/intake/shot keys
        stats.pop("metrics", None)
        # collapse envelope-valued fields the aggregator expects as scalars/None
        for k in ("intake_success_rate", "shot_accuracy", "miss_rate",
                  "fuel_scored", "scoring_rate_per_min", "mean_cycle_time_s"):
            v = stats.get(k)
            if isinstance(v, dict):
                stats[k] = v.get("value")
        stats["track_id"] = _as_int(tid)
        stats.setdefault("robot_id", _as_int(tid))
        robots[str(tid)] = stats
    return {
        "match_key": raw.get("match_key"),
        "events": raw.get("events") or [],
        "robots": robots,
        "overlay": raw.get("overlay") or {},
        "expected_alliances": raw.get("expected_alliances") or {},
    }


def _authoritative_alliance_points(raw: dict, tba_by_key: Optional[dict]
                                   ) -> Optional[dict]:
    """Authoritative {blue, red} alliance points for the OPR solve.

    Priority: TBA final score_breakdown.totalPoints (truth) > overlay blue/red_score
    (authoritative but possibly a mid-match snapshot). Returns None when neither
    side is a usable full-match score, so that match is simply excluded from OPR
    (we never feed our fuel ESTIMATE into OPR)."""
    key = raw.get("match_key")
    if tba_by_key and key in tba_by_key:
        row = tba_by_key[key]
        sb = row.get("score_breakdown") or {}
        al = row.get("alliances") or {}
        b = (sb.get("blue") or {}).get("totalPoints")
        r = (sb.get("red") or {}).get("totalPoints")
        if b is None:
            b = (al.get("blue") or {}).get("score")
        if r is None:
            r = (al.get("red") or {}).get("score")
        if b is not None and r is not None and b >= 0 and r >= 0:
            return {"blue": float(b), "red": float(r), "source": "tba_final"}
    ov = raw.get("overlay") or {}
    b, r = ov.get("blue_score"), ov.get("red_score")
    if b is not None and r is not None:
        return {"blue": float(b), "red": float(r), "source": "overlay_snapshot"}
    return None


def _tba_key_for(raw: dict, event: Optional[str]) -> Optional[str]:
    """Map a raw match_key like 'qual56' to a TBA key like '2026ctwat_qm56'."""
    mk = (raw.get("match_key") or "").lower().strip()
    if not mk:
        return None
    import re
    m = re.match(r"(?:qual|qm|q)\D*(\d+)", mk)
    if m and event:
        return f"{event}_qm{int(m.group(1))}"
    return None


# ───────────────────────────────── builder ───────────────────────────────────

def build_event(scout_dir: str, tba_path: Optional[str] = None,
                event: Optional[str] = None,
                config: Optional[MetricsConfig] = None) -> dict:
    agg = MetricsAggregator(config or MetricsConfig())

    tba_by_key: Optional[dict] = None
    if tba_path and os.path.exists(tba_path):
        obj = json.load(open(tba_path))
        rows = obj["matches"] if isinstance(obj, dict) and "matches" in obj else obj
        tba_by_key = {r.get("key"): r for r in rows if r.get("key")}

    paths = sorted(glob.glob(os.path.join(scout_dir, "match_*.json")))
    match_metrics: list[dict] = []
    opr_inputs: list[dict] = []
    roster_matches: dict[int, dict] = defaultdict(lambda: {"matches": 0, "alliance": None})
    matches_meta: list[dict] = []

    for p in paths:
        raw = json.load(open(p))
        if event is None:
            # try to infer event from a tba key field if present
            event = raw.get("event_key") or event
        rec = _to_match_record(raw)
        mm = agg.aggregate(rec)
        match_metrics.append(mm)

        # authoritative roster bookkeeping (for the per-team roster rows + OPR)
        alliances = _overlay_alliances(raw)
        for color in ("blue", "red"):
            for t in alliances[color]:
                roster_matches[t]["matches"] += 1
                roster_matches[t]["alliance"] = color

        # OPR feed: authoritative points only (TBA final > overlay snapshot)
        tk = _tba_key_for(raw, event)
        if tk and tba_by_key and tk in tba_by_key:
            raw["match_key_tba"] = tk
        pts = _authoritative_alliance_points(
            {**raw, "match_key": tk or raw.get("match_key")}, tba_by_key)
        if pts and alliances["blue"] and alliances["red"]:
            opr_inputs.append({"teams": alliances["blue"], "points": pts["blue"],
                               "opp_points": pts["red"]})
            opr_inputs.append({"teams": alliances["red"], "points": pts["red"],
                               "opp_points": pts["blue"]})
        matches_meta.append({
            "match_key": raw.get("match_key"),
            "alliances": alliances,
            "authoritative_points": pts,
        })

    team_table = agg.team_aggregates(match_metrics, opr_inputs=opr_inputs)

    # Merge: ensure EVERY authoritative-roster team has a row (offense gated alliance)
    teams_out = dict(team_table["teams"])
    resolved_team_numbers = {
        v.get("team") for v in teams_out.values()
        if v.get("attribution_level") == "team" and v.get("team") is not None
    }
    for t, info in sorted(roster_matches.items()):
        if t in resolved_team_numbers:
            continue
        teams_out[str(t)] = _roster_only_row(t, info)

    opr_block = team_table.get("opr_dpr_ccwm") or {}

    out = {
        "schema_version": "rebuilt-2026.event_scout.v1",
        "event": event,
        "n_matches": len(paths),
        "match_files": [os.path.basename(p) for p in paths],
        "attribution_note": (
            "Per-team OFFENSIVE means are populated only for tracks whose identity "
            "was resolved (id_method in {anchor,elimination}, id_confidence ≥ 0.5). "
            "On current footage no track is identity-resolved, so every team row is "
            "attribution_level='alliance' with offense available:False. OPR/DPR/CCWM "
            "are computed from AUTHORITATIVE alliance scores (TBA final / overlay), "
            "NOT our fuel estimate."),
        "opr_dpr_ccwm": opr_block,
        "matches": matches_meta,
        "teams": teams_out,
    }
    return out


def _roster_only_row(team: int, info: dict) -> dict:
    from frcscout.metrics.scout import _metric
    unattr = ("offense unattributed: per-robot identity unresolved on this footage "
              "(team known from overlay roster, but which track is this team is not)")
    return {
        "team": team,
        "attribution_level": "alliance",
        "alliance": info.get("alliance"),
        "matches_played": info.get("matches", 0),
        "avg_fuel_per_match": _metric(None, available=False, unit="fuel", needs=unattr),
        "fuel_variance": _metric(None, available=False, unit="fuel²", needs=unattr),
        "avg_auto_fuel": _metric(None, available=False, unit="fuel", needs=unattr),
        "active_shift_conversion": _metric(None, available=False, unit="fraction",
                                           needs=unattr),
        "ranking_score": _metric(None, available=False,
                                 needs="per-match RP totals (need TOWER points)"),
        "avg_match_points_excl_fouls": _metric(
            None, available=False, needs="complete match_points (TOWER) to average"),
        "avg_tower_points": _metric(None, available=False,
                                    needs="climb-level detector (ranking tiebreaker #4)"),
        "climb_rate": _metric(None, available=False, unit="fraction",
                              needs="climb-level detector"),
    }


# ───────────────────────────────── CSV writer ────────────────────────────────

_CSV_COLS = [
    "team", "attribution_level", "alliance", "matches_played",
    "avg_fuel_per_match", "fuel_variance", "avg_auto_fuel",
    "active_shift_conversion", "opr", "dpr", "ccwm",
]


def _cell(metric_or_val: Any) -> str:
    """Render a metric envelope or scalar into a CSV cell — NEVER blank/0 for an
    unavailable metric; emit 'n/a (needs …)'."""
    if isinstance(metric_or_val, dict) and "available" in metric_or_val:
        if metric_or_val.get("available") and metric_or_val.get("value") is not None:
            return str(metric_or_val["value"])
        needs = (metric_or_val.get("needs") or "detector").split(";")[0].split("(")[0].strip()
        # If the note already reads as a status ('offense unattributed: …'), surface
        # it verbatim; otherwise frame it as a requirement.
        if needs.lower().startswith("offense unattributed"):
            return f"n/a ({needs})"
        return f"n/a (needs {needs})"
    return "" if metric_or_val is None else str(metric_or_val)


def write_csv(event_doc: dict, path: str) -> None:
    opr = ((event_doc.get("opr_dpr_ccwm") or {}).get("value") or {})
    opr_map = opr.get("opr") or {}
    dpr_map = opr.get("dpr") or {}
    ccwm_map = opr.get("ccwm") or {}
    rows = []
    for key, t in event_doc["teams"].items():
        team = t.get("team")
        rows.append({
            "team": team if team is not None else key,
            "attribution_level": t.get("attribution_level", ""),
            "alliance": t.get("alliance") or "",
            "matches_played": t.get("matches_played", 0),
            "avg_fuel_per_match": _cell(t.get("avg_fuel_per_match")),
            "fuel_variance": _cell(t.get("fuel_variance")),
            "avg_auto_fuel": _cell(t.get("avg_auto_fuel")),
            "active_shift_conversion": _cell(t.get("active_shift_conversion")),
            "opr": opr_map.get(team, opr_map.get(str(team), "n/a (needs OPR rows)")),
            "dpr": dpr_map.get(team, dpr_map.get(str(team), "n/a (needs OPR rows)")),
            "ccwm": ccwm_map.get(team, ccwm_map.get(str(team), "n/a (needs OPR rows)")),
        })

    def _sortkey(r):
        v = r["opr"]
        return -v if isinstance(v, (int, float)) else 1e9
    rows.sort(key=_sortkey)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS)
        w.writeheader()
        w.writerows(rows)


def validate_vs_tba(scout_dir: str, tba_path: str, event: Optional[str]) -> dict:
    """Compare OUR per-match read to TBA truth: winner agreement + score agreement
    over the matches we have, with the FUEL-ONLY caveat explicit.

    We compare the overlay-derived blue/red score we captured against TBA's final
    totalPoints. ``winner_agree`` checks the higher-scoring alliance matches; it is
    only meaningful when our overlay score is a FINAL read — many of our reads are
    mid-match snapshots, which we count and flag rather than scoring as agreement.
    """
    obj = json.load(open(tba_path))
    rows = obj["matches"] if isinstance(obj, dict) and "matches" in obj else obj
    tba_by_key = {r.get("key"): r for r in rows if r.get("key")}

    n = winner_agree = score_exact = snapshot_flagged = 0
    details = []
    for p in sorted(glob.glob(os.path.join(scout_dir, "match_*.json"))):
        raw = json.load(open(p))
        tk = _tba_key_for(raw, event)
        if not tk or tk not in tba_by_key:
            continue
        n += 1
        trow = tba_by_key[tk]
        al = trow.get("alliances") or {}
        tba_b = (al.get("blue") or {}).get("score")
        tba_r = (al.get("red") or {}).get("score")
        ov = raw.get("overlay") or {}
        our_b, our_r = ov.get("blue_score"), ov.get("red_score")
        timer = ov.get("timer_s")
        # our roster color may be swapped vs TBA; align by roster set
        our_al = _overlay_alliances(raw)
        tba_blue = {int(t[3:]) for t in (al.get("blue") or {}).get("team_keys", [])}
        swapped = bool(our_al["red"]) and set(our_al["red"]) == tba_blue
        if swapped:
            our_b, our_r = our_r, our_b
        is_snapshot = our_b is None or our_r is None or (timer not in (None, 0))
        if is_snapshot:
            snapshot_flagged += 1
        wa = sa = None
        if our_b is not None and our_r is not None and tba_b is not None and tba_r is not None:
            wa = (our_b > our_r) == (tba_b > tba_r)
            sa = (our_b == tba_b and our_r == tba_r)
            if not is_snapshot:
                winner_agree += int(wa)
                score_exact += int(sa)
        details.append({"match": raw.get("match_key"), "tba_key": tk,
                        "our": [our_b, our_r], "tba": [tba_b, tba_r],
                        "color_swapped": swapped, "snapshot": is_snapshot,
                        "winner_agree": wa, "score_exact": sa})

    final_n = n - snapshot_flagged
    return {
        "n_compared": n,
        "n_final_reads": final_n,
        "n_snapshot_reads": snapshot_flagged,
        "winner_agreement": (winner_agree / final_n) if final_n else None,
        "score_exact_agreement": (score_exact / final_n) if final_n else None,
        "caveat": ("Our overlay score is FUEL-derived and many reads are MID-MATCH "
                   "snapshots (timer != 0), not finals; winner/score agreement is "
                   "computed ONLY over final reads. TBA score includes TOWER+foul "
                   "points our fuel-only estimate cannot reconstruct."),
        "details": details,
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scout-dir", default="data/scout_out")
    ap.add_argument("--tba", default=None, help="TBA matches json for OPR truth")
    ap.add_argument("--event", default=None, help="event key e.g. 2026ctwat")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--level", default="regional",
                    help="MetricsConfig level (regional/dcmp/champs)")
    ap.add_argument("--validate", action="store_true",
                    help="also compare our reads vs TBA (needs --tba)")
    a = ap.parse_args(argv)

    out_json = a.out_json or os.path.join(a.scout_dir, "event_teams.json")
    out_csv = a.out_csv or os.path.join(a.scout_dir, "event_teams.csv")

    doc = build_event(a.scout_dir, tba_path=a.tba, event=a.event,
                      config=MetricsConfig.for_level(a.level))
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(out_json, "w"), indent=2)
    write_csv(doc, out_csv)
    n_resolved = sum(1 for t in doc["teams"].values()
                     if t.get("attribution_level") == "team")
    print(f"event={doc['event']} matches={doc['n_matches']} teams={len(doc['teams'])} "
          f"(offense-resolved={n_resolved}) opr_available="
          f"{(doc['opr_dpr_ccwm'] or {}).get('available')}")
    print(f"wrote {out_json}")
    print(f"wrote {out_csv}")

    if a.validate:
        if not a.tba:
            print("--validate needs --tba"); return 2
        v = validate_vs_tba(a.scout_dir, a.tba, a.event)
        out_val = os.path.join(a.scout_dir, "tba_validation.json")
        json.dump(v, open(out_val, "w"), indent=2)
        print(f"[validate] compared={v['n_compared']} final={v['n_final_reads']} "
              f"snapshot={v['n_snapshot_reads']} "
              f"winner_agree={v['winner_agreement']} score_exact={v['score_exact_agreement']}")
        print(f"[validate] caveat: {v['caveat']}")
        print(f"wrote {out_val}")
    return 0


def _as_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return x


if __name__ == "__main__":
    raise SystemExit(main())
