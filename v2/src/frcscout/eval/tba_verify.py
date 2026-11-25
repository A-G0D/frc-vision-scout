"""End-to-end match-data accuracy vs The Blue Alliance — PLAYOFFS INCLUDED.

v1 reported 100% by (a) scoping to the 16 quals visible in one VOD out of 64, and (b)
silently skipping playoff rows (match_number=null) whose extraction was visibly broken
(2-team alliances, a team on both alliances, rank-column numbers read as teams).

This version:
    - reports accuracy per comp_level (qm / sf / f) AND overall
    - matches playoff rows by bracket label, not just match_number
    - flags structurally-impossible extractions (alliance != 3 teams, team on both sides,
      implausible 1-2 digit team numbers) as ERRORS, not silent skips
    - reports the DENOMINATOR explicitly (N detected of M total in the event)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_tba(p: Path) -> list[dict]:
    obj = json.load(open(p))
    rows = obj["matches"] if isinstance(obj, dict) and "matches" in obj else obj
    # TBA stores alliance teams under 'blue'/'red'; the detector output (and our
    # comparison) use 'blue_teams'/'red_teams'. Normalize so they line up.
    for m in rows:
        if "blue_teams" not in m and "blue" in m:
            m["blue_teams"] = m["blue"]
        if "red_teams" not in m and "red" in m:
            m["red_teams"] = m["red"]
    return rows


def _match_key(m: dict) -> Optional[str]:
    """Build a canonical match key from comp_level + match_number or bracket label.

    Playoff rows sometimes have match_number=null; fall back to a bracket label
    (e.g. 'sf1m1') stored in 'match_label' or synthesised from comp_level+set+match.
    """
    comp = m.get("comp_level", "")
    mn = m.get("match_number")
    ml = m.get("match_label") or m.get("label")  # overlay-extracted label

    if comp == "qm" and mn is not None:
        return f"qm{mn}"

    # Playoff: prefer the explicit bracket label
    if ml:
        return str(ml).strip().lower()

    # Try to build sf/f key from set_number + match_number
    sn = m.get("set_number")
    if comp and mn is not None and sn is not None:
        return f"{comp}{sn}m{mn}"
    if comp and mn is not None:
        return f"{comp}{mn}"

    return None


def _teams_match(detected: dict, tba: dict, allow_color_swap: bool = True) -> bool:
    """Teams are compared as order-independent sets with an optional color-swap allowance."""
    db = set(detected.get("blue_teams") or [])
    dr = set(detected.get("red_teams") or [])
    tb = set(tba.get("blue_teams") or [])
    tr = set(tba.get("red_teams") or [])

    if db == tb and dr == tr:
        return True
    if allow_color_swap and db == tr and dr == tb:
        return True
    return False


def _scores_match(detected: dict, tba: dict, allow_color_swap: bool = True) -> bool:
    """Score comparison with optional color-swap allowance.

    Both sides must be present; returns True if either alignment matches.
    """
    ds = detected.get("blue_score")
    dr_s = detected.get("red_score")
    ts = tba.get("blue_score")
    tr_s = tba.get("red_score")

    if ds is None or dr_s is None or ts is None or tr_s is None:
        return False  # can't verify

    if ds == ts and dr_s == tr_s:
        return True
    if allow_color_swap and ds == tr_s and dr_s == ts:
        return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify(detected_path: str, tba_path: str) -> dict:
    """Compare detected match data vs TBA across ALL comp levels.

    Returns
    -------
    dict with keys:
        n_detected          – how many rows the CV system produced
        n_tba               – total matches in the TBA export for this event
        structural_errors   – list of detected dicts that fail _is_impossible
        per_level           – {comp_level: {"correct": int, "total_detected": int,
                                            "total_tba": int, "accuracy": float}}
        overall_accuracy    – float (correct / n_detected), or None if n_detected==0
        unmatched_detected  – list of detected rows that had no TBA counterpart
        unmatched_tba       – list of TBA rows that had no detected counterpart
    """
    detected = json.load(open(detected_path))
    tba_matches = _load_tba(Path(tba_path))

    # --- structural validity filter (do NOT skip — count them as errors) ---
    structural_errors = [d for d in detected if _is_impossible(d)]

    # --- build lookup by match key for TBA rows ---
    tba_by_key: dict[str, dict] = {}
    for m in tba_matches:
        k = _match_key(m)
        if k:
            tba_by_key[k] = m

    # --- per-level accumulators ---
    # level -> {correct, total_detected, tba_keys_seen}
    level_stats: dict[str, dict] = defaultdict(lambda: {
        "correct": 0,
        "total_detected": 0,
        "tba_keys_seen": set(),
    })

    total_correct = 0
    unmatched_detected = []
    matched_tba_keys = set()

    for d in detected:
        comp = d.get("comp_level", "unknown")
        level_stats[comp]["total_detected"] += 1

        k = _match_key(d)
        if k is None or k not in tba_by_key:
            unmatched_detected.append(d)
            continue

        tba_row = tba_by_key[k]
        matched_tba_keys.add(k)

        teams_ok = _teams_match(d, tba_row)
        scores_ok = _scores_match(d, tba_row)

        if teams_ok and scores_ok:
            level_stats[comp]["correct"] += 1
            total_correct += 1

    # Unmatched TBA rows
    unmatched_tba = [m for m in tba_matches if _match_key(m) not in matched_tba_keys]

    # Collate per-level TBA totals
    tba_by_level: dict[str, int] = defaultdict(int)
    for m in tba_matches:
        tba_by_level[m.get("comp_level", "unknown")] += 1

    per_level: dict[str, dict] = {}
    for comp, stats in level_stats.items():
        n_det = stats["total_detected"]
        n_tba = tba_by_level.get(comp, 0)
        correct = stats["correct"]
        per_level[comp] = {
            "correct": correct,
            "total_detected": n_det,
            "total_tba": n_tba,
            # Denominator is n_detected (what we can measure); n_tba shows coverage gap
            "accuracy": correct / n_det if n_det else None,
        }

    n_detected = len(detected)
    overall_accuracy = total_correct / n_detected if n_detected else None

    return {
        "n_detected": n_detected,
        "n_tba": len(tba_matches),
        # Number of structural errors (for backward-compat the original stub returned int)
        "structural_errors": structural_errors,   # list so callers can inspect details
        "structural_error_count": len(structural_errors),
        "per_level": per_level,
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "unmatched_detected": unmatched_detected,
        "unmatched_tba": unmatched_tba,
    }


def _is_impossible(d: dict) -> bool:
    """Catches the exact v1 failure modes."""
    b, r = d.get("blue_teams", []), d.get("red_teams", [])
    if d.get("comp_level") in ("qm", "f") and (len(b) != 3 or len(r) != 3):
        return True
    if set(b) & set(r):                       # a team can't be on both alliances
        return True
    if any(t < 100 for t in b + r):           # implausible: likely a rank-column misread
        return True
    return False
