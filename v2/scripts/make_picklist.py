"""make_picklist — turn event_teams.json into a ranked picklist + per-team cards.

Outputs (default into the same dir as event_teams.json):
  - rankings.csv   one row per TEAM, sorted by a composite z-score, with the z of
                   each component, the composite, and a low_confidence flag.
  - picklist.csv   the ordered draft list (rank, team, composite, confidence, why).
  - team_cards.txt human-readable per-team cards, each with an AVAILABILITY STRIP
                   that prints 'n/a (needs X)' for every gated metric — never blank/0.

Composite (z-score over the signals we actually have, §5.3 picklist intent):
    composite = w_opr·z(OPR) + w_rate·z(scoring_rate) − w_var·z(fuel_variance)

z() is computed over the population of teams for which that component is available;
a team missing a component contributes 0 for that term (and is flagged). With the
current footage NONE of the per-team offensive components (scoring_rate / variance)
are available — identity is unresolved — so the composite collapses to OPR alone and
EVERY team is flagged ``low_confidence: True``. That is honest: the picklist is an
OPR-only ranking until the identity layer resolves per-robot teams.

Run:
    PYTHONPATH=src python scripts/make_picklist.py
    PYTHONPATH=src python scripts/make_picklist.py --event-teams data/scout_out/event_teams.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any, Optional


# component weights (composite intent: reward OPR + scoring rate, penalize variance)
W_OPR = 1.0
W_RATE = 1.0
W_VAR = 0.5


def _val(metric: Any) -> Optional[float]:
    """Pull a numeric value from a metric envelope iff available, else None."""
    if isinstance(metric, dict):
        if metric.get("available") and isinstance(metric.get("value"), (int, float)):
            return float(metric["value"])
        return None
    return float(metric) if isinstance(metric, (int, float)) else None


def _zscores(values: dict[Any, Optional[float]]) -> dict[Any, Optional[float]]:
    present = [v for v in values.values() if v is not None]
    if len(present) < 2:
        # not enough spread to standardize; everyone present gets 0 (neutral)
        return {k: (0.0 if v is not None else None) for k, v in values.items()}
    mu = statistics.fmean(present)
    sd = statistics.pstdev(present)
    if sd == 0:
        return {k: (0.0 if v is not None else None) for k, v in values.items()}
    return {k: ((v - mu) / sd if v is not None else None) for k, v in values.items()}


def _opr_map(doc: dict) -> dict[Any, float]:
    opr = (doc.get("opr_dpr_ccwm") or {})
    if not opr.get("available"):
        return {}
    m = (opr.get("value") or {}).get("opr") or {}
    out: dict[Any, float] = {}
    for k, v in m.items():
        try:
            out[int(k)] = float(v)
        except (TypeError, ValueError):
            out[k] = float(v)
    return out


def _real_team_rows(doc: dict) -> list[tuple[int, dict]]:
    """Rows keyed by a real team number (skip the synthetic track:N rows)."""
    rows = []
    for key, t in doc["teams"].items():
        team = t.get("team")
        if team is None:
            continue  # track:N alliance-attribution rows aren't draftable by number
        rows.append((int(team), t))
    return rows


def build_rankings(doc: dict) -> list[dict]:
    opr_map = _opr_map(doc)
    teams = _real_team_rows(doc)

    opr_vals = {tn: opr_map.get(tn) for tn, _ in teams}
    rate_vals = {tn: _val(t.get("scoring_rate_per_min") or t.get("avg_fuel_per_match"))
                 for tn, t in teams}
    var_vals = {tn: _val(t.get("fuel_variance")) for tn, t in teams}

    z_opr = _zscores(opr_vals)
    z_rate = _zscores(rate_vals)
    z_var = _zscores(var_vals)

    ranked = []
    for tn, t in teams:
        zo = z_opr.get(tn)
        zr = z_rate.get(tn)
        zv = z_var.get(tn)
        composite = (W_OPR * (zo or 0.0)) + (W_RATE * (zr or 0.0)) - (W_VAR * (zv or 0.0))
        # low confidence when offense components are missing OR OPR system is thin
        offense_missing = (zr is None) or (zv is None)
        opr_thin = (doc.get("opr_dpr_ccwm") or {}).get("available") and \
            doc.get("n_matches", 0) < 6   # OPR under-determined with few matches
        low_conf = bool(offense_missing or opr_thin or zo is None)
        reasons = []
        if zo is None:
            reasons.append("no OPR")
        if offense_missing:
            reasons.append("offense unattributed (identity unresolved)")
        if opr_thin:
            reasons.append(f"OPR under-determined ({doc.get('n_matches')} matches)")
        ranked.append({
            "team": tn,
            "alliance_seen": t.get("alliance"),
            "matches_played": t.get("matches_played", 0),
            "opr": opr_vals.get(tn),
            "z_opr": _round(zo),
            "z_scoring_rate": _round(zr),
            "z_fuel_variance": _round(zv),
            "composite": round(composite, 4),
            "low_confidence": low_conf,
            "confidence_reasons": reasons,
        })
    ranked.sort(key=lambda r: r["composite"], reverse=True)
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    return ranked


def _round(x: Optional[float]) -> Optional[float]:
    return round(x, 4) if isinstance(x, (int, float)) else None


# ───────────────────────────────── writers ───────────────────────────────────

_RANK_COLS = ["rank", "team", "alliance_seen", "matches_played", "opr",
              "z_opr", "z_scoring_rate", "z_fuel_variance", "composite",
              "low_confidence", "confidence_reasons"]


def write_rankings_csv(ranked: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_RANK_COLS)
        w.writeheader()
        for r in ranked:
            row = dict(r)
            row["confidence_reasons"] = "; ".join(r["confidence_reasons"]) or "—"
            for c in ("opr", "z_scoring_rate", "z_fuel_variance"):
                if row.get(c) is None:
                    row[c] = "n/a"
            w.writerow({k: row.get(k, "") for k in _RANK_COLS})


def write_picklist_csv(ranked: list[dict], path: str) -> None:
    cols = ["pick_order", "team", "composite", "confidence", "why"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in ranked:
            conf = "LOW" if r["low_confidence"] else "OK"
            why = "; ".join(r["confidence_reasons"]) if r["confidence_reasons"] else \
                f"OPR {r['opr']}, composite {r['composite']}"
            w.writerow({"pick_order": r["rank"], "team": r["team"],
                        "composite": r["composite"], "confidence": conf, "why": why})


_CARD_METRICS = [
    ("avg_fuel_per_match", "Avg fuel / match"),
    ("fuel_variance", "Fuel variance (consistency)"),
    ("avg_auto_fuel", "Avg AUTO fuel"),
    ("active_shift_conversion", "Active-shift conversion"),
    ("avg_tower_points", "Avg TOWER points"),
    ("climb_rate", "Climb rate"),
    ("ranking_score", "Ranking score (avg RP)"),
]


def _strip_cell(metric: Any) -> str:
    """Availability-strip cell. NEVER blank/0 for an unavailable metric."""
    if isinstance(metric, dict) and "available" in metric:
        if metric.get("available") and metric.get("value") is not None:
            unit = metric.get("unit")
            return f"{metric['value']}" + (f" {unit}" if unit else "")
        needs = (metric.get("needs") or "detector").split(";")[0].split("(")[0].strip()
        if needs.lower().startswith("offense unattributed"):
            return f"n/a ({needs})"
        return f"n/a (needs {needs})"
    return "n/a" if metric is None else str(metric)


def render_cards(doc: dict, ranked: list[dict]) -> str:
    rank_by_team = {r["team"]: r for r in ranked}
    opr_map = _opr_map(doc)
    lines: list[str] = []
    lines.append(f"EVENT SCOUTING CARDS — {doc.get('event')}  "
                 f"({doc.get('n_matches')} match files)")
    lines.append("=" * 72)
    note = doc.get("attribution_note")
    if note:
        lines.append("NOTE: " + note)
        lines.append("-" * 72)
    for r in ranked:
        tn = r["team"]
        t = next((row for key, row in doc["teams"].items()
                  if row.get("team") == tn), {})
        lines.append("")
        conf = "LOW CONFIDENCE" if r["low_confidence"] else "ok"
        lines.append(f"#{r['rank']:>2}  TEAM {tn}   [{conf}]   "
                     f"composite={r['composite']}   alliance_seen={t.get('alliance')}")
        lines.append(f"     matches_played: {t.get('matches_played', 0)}   "
                     f"attribution: {t.get('attribution_level','?')}")
        lines.append(f"     OPR: {opr_map.get(tn, 'n/a (needs OPR rows)')}   "
                     f"composite_reasons: {'; '.join(r['confidence_reasons']) or '—'}")
        lines.append("     availability strip:")
        for key, label in _CARD_METRICS:
            lines.append(f"        {label:<28} {_strip_cell(t.get(key))}")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--event-teams", default="data/scout_out/event_teams.json")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)

    doc = json.load(open(a.event_teams))
    out_dir = a.out_dir or os.path.dirname(a.event_teams) or "."
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ranked = build_rankings(doc)
    rankings_csv = os.path.join(out_dir, "rankings.csv")
    picklist_csv = os.path.join(out_dir, "picklist.csv")
    cards_txt = os.path.join(out_dir, "team_cards.txt")

    write_rankings_csv(ranked, rankings_csv)
    write_picklist_csv(ranked, picklist_csv)
    open(cards_txt, "w", encoding="utf-8").write(render_cards(doc, ranked))

    n_low = sum(1 for r in ranked if r["low_confidence"])
    print(f"teams={len(ranked)} low_confidence={n_low}")
    print(f"wrote {rankings_csv}")
    print(f"wrote {picklist_csv}")
    print(f"wrote {cards_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
