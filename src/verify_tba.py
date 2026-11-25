"""
verify_tba.py — score the video-extracted match data against the TBA ground-truth oracle.

Loads:
  - detected results  : data/<stem>_results.json   (from scan_results.py)
  - TBA ground truth  : data/tba_waterbury2026.json (from the TBA fetch agent)

For each detected qualification match, find the same match_number in TBA and compare:
  - teams: blue set vs TBA blue set, red set vs TBA red set (order-independent — alliance
    membership is what matters; also checks the color-swapped case and reports it).
  - scores: blue/red final scores.

Reports per-match pass/fail and aggregate accuracy (teams, scores), and lists mismatches so
the next loop iteration can target them (ROI/threshold fixes, OCR errors, wrong frame).

"Relatively accurate is okay" — so we report rates, not a hard all-or-nothing gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def _tba_index(tba: list[dict]) -> dict:
    """Map qual match_number -> {'blue': set, 'red': set, 'blue_score', 'red_score'}."""
    idx = {}
    for m in tba:
        if m.get("comp_level") != "qm":
            continue
        idx[int(m["match_number"])] = {
            "blue": set(int(t) for t in m["blue"]),
            "red": set(int(t) for t in m["red"]),
            "blue_score": m.get("blue_score"),
            "red_score": m.get("red_score"),
        }
    return idx


def _load_tba(tba_path: Path) -> list[dict]:
    """Accept either a bare match array or a {..., 'matches': [...]} wrapper."""
    obj = json.load(open(tba_path))
    return obj["matches"] if isinstance(obj, dict) and "matches" in obj else obj


def verify(results_path: Path, tba_path: Path) -> dict:
    detected = json.load(open(results_path))
    tba = _tba_index(_load_tba(tba_path))

    n_quals = teams_ok = scores_ok = matched = swapped = 0
    mismatches = []

    for d in detected:
        if d.get("comp_level") != "qm" or d.get("match_number") is None:
            continue
        n_quals += 1
        num = int(d["match_number"])
        truth = tba.get(num)
        if not truth:
            mismatches.append(f"qm{num}: not in TBA (detected blue={d['blue_teams']} red={d['red_teams']})")
            continue
        matched += 1
        db, dr = set(d["blue_teams"]), set(d["red_teams"])

        direct = (db == truth["blue"] and dr == truth["red"])
        swap = (db == truth["red"] and dr == truth["blue"])
        if direct or swap:
            teams_ok += 1
            if swap:
                swapped += 1
        else:
            mismatches.append(
                f"qm{num} TEAMS: got blue={sorted(db)} red={sorted(dr)} | "
                f"tba blue={sorted(truth['blue'])} red={sorted(truth['red'])}")

        if truth["blue_score"] is not None:
            sd = (d.get("blue_score") == truth["blue_score"] and d.get("red_score") == truth["red_score"])
            ss = (d.get("blue_score") == truth["red_score"] and d.get("red_score") == truth["blue_score"])
            if sd or ss:
                scores_ok += 1
            else:
                mismatches.append(
                    f"qm{num} SCORE: got {d.get('blue_score')}-{d.get('red_score')} | "
                    f"tba {truth['blue_score']}-{truth['red_score']}")

    pct = lambda a, b: f"{(100*a/b):.0f}%" if b else "n/a"
    summary = {
        "detected_quals": n_quals,
        "matched_to_tba": matched,
        "teams_correct": teams_ok,
        "teams_accuracy": pct(teams_ok, matched),
        "color_swapped": swapped,
        "scores_correct": scores_ok,
        "scores_accuracy": pct(scores_ok, matched),
        "mismatches": mismatches,
    }
    return summary


if __name__ == "__main__":
    res = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "ngvsHrmoW8U_360p_results.json"
    tba = Path(sys.argv[2]) if len(sys.argv) > 2 else DATA / "tba_waterbury2026.json"
    if not res.exists():
        print(f"missing detected results: {res}"); sys.exit(1)
    if not tba.exists():
        print(f"missing TBA ground truth: {tba}"); sys.exit(1)
    s = verify(res, tba)
    print(f"detected quals:   {s['detected_quals']}")
    print(f"matched to TBA:   {s['matched_to_tba']}")
    print(f"TEAMS accuracy:   {s['teams_correct']}/{s['matched_to_tba']}  ({s['teams_accuracy']})"
          f"{'  (' + str(s['color_swapped']) + ' color-swapped)' if s['color_swapped'] else ''}")
    print(f"SCORES accuracy:  {s['scores_correct']}/{s['matched_to_tba']}  ({s['scores_accuracy']})")
    if s["mismatches"]:
        print(f"\n{len(s['mismatches'])} mismatch(es):")
        for m in s["mismatches"][:40]:
            print("  " + m)
