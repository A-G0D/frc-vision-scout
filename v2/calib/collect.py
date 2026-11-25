"""collect.py — REAL (vision, TBA) pair collector feeding error_model.py.

Scans vision scout-output files under data/scout_out/ (or a supplied directory),
pairs each FINAL end-of-match overlay read against the TBA score_breakdown for
the corresponding alliance, and writes calib/data/pairs.jsonl.

HARD RULES
----------
* End-of-match gate:  overlay.timer_s must be 0 (or the sentinel "FINAL"/None
  with explicit is_final flag).  Any read where timer_s > 0 is a mid-match
  snapshot and is REJECTED — it would poison the calibration.
* Only the ONE valid calibration component is emitted:  component="fuel_total",
  vision_est = overlay.<alliance>_fuel, tba_actual = hubScore.totalCount.
  Sub-components "fuel_auto" and "fuel_teleop" are emitted when available.
* Color-swap resolution: if the alliance team-sets in the vision overlay match
  the TBA match with blue/red swapped, the alliance labels are swapped before
  pairing (reusing the _teams_match logic from src/frcscout/eval/tba_verify.py).

Usage (command-line)
--------------------
    python calib/collect.py
    python calib/collect.py --scout-dir data/scout_out --tba-json data/tba/2026ctwat_matches.json
    python calib/collect.py --out calib/data/pairs.jsonl --fit

Public API
----------
    VisionRead          dataclass — parsed overlay from one scout-output file
    collect_pairs(...)  -> CollectResult  (the main entry point)
    CollectResult       dataclass — pairs, skips, stats
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_V2_ROOT = Path(__file__).resolve().parents[1]

_DEFAULT_SCOUT_DIR = _V2_ROOT / "data" / "scout_out"
_DEFAULT_TBA_JSON = _V2_ROOT / "data" / "tba" / "2026ctwat_matches.json"
_DEFAULT_OUT_JSONL = _V2_ROOT / "calib" / "data" / "pairs.jsonl"

# FRC 2026 REBUILT — a regulation match runs 150 seconds (15s auto + 135s teleop).
# Any overlay read with timer_s > 0 is a MID-MATCH snapshot.
# A timer_s == 0 or an explicit is_final=True flag means end-of-match.
_END_OF_MATCH_TIMER_S = 0.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VisionRead:
    """Parsed overlay data from one scout-output JSON file.

    Fields
    ------
    source_file : str
        Basename of the scout-output file this came from.
    match_key : str
        Vision-side match key (e.g. "qual59", "m1080").
    tba_match_key : str | None
        Resolved TBA match key (e.g. "2026ctwat_qm59"), if determinable.
    timer_s : float | None
        Timer value extracted from overlay.  0 = end of match.
    is_final : bool
        True if this read passes the end-of-match gate (timer_s == 0).
    blue_fuel : float | None
        overlay.blue_fuel — vision estimate of blue alliance fuel.
    red_fuel : float | None
        overlay.red_fuel — vision estimate of red alliance fuel.
    blue_teams : list[int]
        overlay.blue_teams
    red_teams : list[int]
        overlay.red_teams
    color_swapped : bool
        True if vision blue/red were swapped vs TBA to achieve team alignment.
    event_key : str
        TBA event key (e.g. "2026ctwat"), extracted from tba_match_key.
    comp_level : str
        "qm", "sf", or "f" (extracted from tba_match_key).
    """
    source_file: str
    match_key: str
    tba_match_key: Optional[str]
    timer_s: Optional[float]
    is_final: bool
    blue_fuel: Optional[float]
    red_fuel: Optional[float]
    blue_teams: List[int] = field(default_factory=list)
    red_teams: List[int] = field(default_factory=list)
    color_swapped: bool = False
    event_key: str = ""
    comp_level: str = "qm"


@dataclass
class CollectResult:
    """Result of collect_pairs().

    Attributes
    ----------
    pairs : List[PairedSample]
        All valid (vision, TBA) pairs that passed the end-of-match gate.
    n_pairs : int
        len(pairs)
    n_skipped_midmatch : int
        Reads rejected because timer_s > 0 (mid-match snapshot).
    n_skipped_no_tba : int
        Reads that were final but had no matching TBA breakdown.
    n_skipped_no_fuel : int
        Reads that were final + TBA matched but vision fuel was None.
    reads : List[VisionRead]
        All parsed VisionRead objects (final + skipped).
    """
    pairs: List  # List[PairedSample]
    n_pairs: int
    n_skipped_midmatch: int
    n_skipped_no_tba: int
    n_skipped_no_fuel: int
    reads: List[VisionRead] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TBA match index builder
# ---------------------------------------------------------------------------

def _build_tba_index(tba_matches: List[dict]) -> Dict[str, dict]:
    """Build a lookup: tba_match_key -> match dict.

    Keys are the full TBA match key strings (e.g. "2026ctwat_qm59").
    """
    idx: Dict[str, dict] = {}
    for m in tba_matches:
        k = m.get("key")
        if k:
            idx[str(k)] = m
    return idx


def _infer_tba_key(vision_match_key: str, event_key: str) -> Optional[str]:
    """Map a vision match_key to a TBA match key.

    Vision match keys are informal strings like "qual59", "m1080", "q59final".
    We try heuristic patterns:
      - "qual<N>"  → "<event>_qm<N>"
      - "q<N>..."  → "<event>_qm<N>"
      - "m<N>..."  → numeric N, try as qm<N> for 4-digit match (1080 → qm80? no—
                      match_1080 is the filename for a match with 3-digit VOD index;
                      the actual TBA match number is different — we can't safely infer it)

    Returns None if the pattern is not recognized or ambiguous.
    """
    mk = vision_match_key.lower().strip()

    # "qual59" → qm59
    if mk.startswith("qual") and mk[4:].isdigit():
        return f"{event_key}_qm{mk[4:]}"

    # "q59..." where digits follow q directly
    if mk.startswith("q") and not mk.startswith("qm"):
        # extract leading digits after 'q'
        digits = ""
        for ch in mk[1:]:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            return f"{event_key}_qm{digits}"

    return None


# ---------------------------------------------------------------------------
# Alliance-level color-swap resolution
# ---------------------------------------------------------------------------

def _resolve_color_swap(
    vision_blue_teams: List[int],
    vision_red_teams: List[int],
    tba_match: dict,
) -> Tuple[bool, bool]:
    """Determine if vision colors are swapped relative to TBA.

    Returns
    -------
    (teams_resolved, color_swapped)
        teams_resolved : True if team sets matched at all (normal or swapped).
        color_swapped  : True if a swap was needed.
    """
    def _team_set(m: dict, side: str) -> frozenset:
        raw = m.get("alliances", {}).get(side, {}).get("team_keys", [])
        nums = set()
        for t in raw:
            # TBA team keys are "frc<N>" strings
            t_str = str(t)
            if t_str.startswith("frc"):
                try:
                    nums.add(int(t_str[3:]))
                except ValueError:
                    pass
            else:
                try:
                    nums.add(int(t_str))
                except ValueError:
                    pass
        return frozenset(nums)

    tba_blue = _team_set(tba_match, "blue")
    tba_red = _team_set(tba_match, "red")
    vis_blue = frozenset(vision_blue_teams)
    vis_red = frozenset(vision_red_teams)

    # If neither side has team data, we can't resolve — assume no swap
    if not (tba_blue or tba_red or vis_blue or vis_red):
        return False, False

    # Normal alignment
    if vis_blue == tba_blue and vis_red == tba_red:
        return True, False

    # Swapped alignment
    if vis_blue == tba_red and vis_red == tba_blue:
        return True, True

    # Teams don't match — can't confirm swap; proceed without swap (best-effort)
    return False, False


# ---------------------------------------------------------------------------
# HubScore extraction from TBA breakdown
# ---------------------------------------------------------------------------

def _hub_score(tba_match: dict, alliance: str) -> Dict[str, Optional[int]]:
    """Extract hubScore sub-fields for one alliance from a TBA match dict.

    Returns dict with keys: total_count, auto_count, teleop_count
    (all may be None if breakdown unavailable).
    """
    sb = tba_match.get("score_breakdown") or {}
    alliance_sb = sb.get(alliance) or {}
    hub = alliance_sb.get("hubScore") or {}
    return {
        "total_count": hub.get("totalCount"),
        "auto_count": hub.get("autoCount"),
        "teleop_count": hub.get("teleopCount"),
    }


# ---------------------------------------------------------------------------
# Vision file parsing
# ---------------------------------------------------------------------------

def _parse_scout_file(path: Path, event_key: str) -> Optional[VisionRead]:
    """Parse one scout-output JSON file into a VisionRead.

    Returns None if the file has no overlay block or cannot be parsed.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    # Skip files whose root is not a dict (e.g. diag_robot_summary.json is a list)
    if not isinstance(data, dict):
        return None

    overlay = data.get("overlay")
    if not overlay or not isinstance(overlay, dict):
        return None

    mk = str(data.get("match_key", path.stem))
    timer_raw = overlay.get("timer_s")

    # Normalise timer_s
    try:
        timer_s = float(timer_raw) if timer_raw is not None else None
    except (TypeError, ValueError):
        timer_s = None

    # End-of-match gate: timer_s must be exactly 0, OR an explicit is_final flag.
    is_final_flag = bool(data.get("is_final", False))
    is_final = is_final_flag or (timer_s is not None and timer_s == _END_OF_MATCH_TIMER_S)

    # Extract fuel values
    def _to_fuel(v) -> Optional[float]:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    blue_fuel = _to_fuel(overlay.get("blue_fuel"))
    red_fuel = _to_fuel(overlay.get("red_fuel"))

    # Team lists
    blue_teams = [int(t) for t in (overlay.get("blue_teams") or []) if t is not None]
    red_teams = [int(t) for t in (overlay.get("red_teams") or []) if t is not None]

    tba_key = _infer_tba_key(mk, event_key)
    ev_key = event_key
    comp_level = "qm"
    if tba_key:
        # Parse comp_level from key suffix (e.g. "2026ctwat_qm59" -> "qm")
        suffix = tba_key.split("_", 1)[-1]  # "qm59"
        for lvl in ("qm", "sf", "f"):
            if suffix.startswith(lvl):
                comp_level = lvl
                break

    return VisionRead(
        source_file=path.name,
        match_key=mk,
        tba_match_key=tba_key,
        timer_s=timer_s,
        is_final=is_final,
        blue_fuel=blue_fuel,
        red_fuel=red_fuel,
        blue_teams=blue_teams,
        red_teams=red_teams,
        event_key=ev_key,
        comp_level=comp_level,
    )


# ---------------------------------------------------------------------------
# Core collector
# ---------------------------------------------------------------------------

def collect_pairs(
    scout_dir: Path = _DEFAULT_SCOUT_DIR,
    tba_json: Path = _DEFAULT_TBA_JSON,
    event_key: str = "2026ctwat",
    out_jsonl: Optional[Path] = _DEFAULT_OUT_JSONL,
    *,
    verbose: bool = False,
) -> "CollectResult":
    """Collect (vision, TBA) calibration pairs from disk.

    Parameters
    ----------
    scout_dir : Path
        Directory containing scout-output JSON files (searched non-recursively
        plus one level of subdirectories).
    tba_json : Path
        Path to the TBA event matches JSON file (list of match dicts OR
        ``{"matches": [...]}``) that provides score_breakdown ground truth.
    event_key : str
        TBA event key used to build full match keys (e.g. "2026ctwat").
    out_jsonl : Path | None
        Where to write pairs.jsonl.  None = dry-run (don't write).
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    CollectResult
        .pairs     — List[PairedSample] that passed all gates
        .n_pairs   — total valid pairs written
        .n_skipped_midmatch — rejected by end-of-match gate
        .n_skipped_no_tba   — final read but no TBA match found
        .n_skipped_no_fuel  — final + TBA matched but vision fuel None
        .reads     — all VisionRead objects parsed
    """
    # Deferred import to avoid circular imports in test scenarios
    from calib.error_model import PairedSample

    # --- 1. Load TBA matches ---
    tba_matches: List[dict] = []
    if Path(tba_json).exists():
        raw = json.loads(Path(tba_json).read_text(encoding="utf-8"))
        if isinstance(raw, list):
            tba_matches = raw
        elif isinstance(raw, dict) and "matches" in raw:
            tba_matches = raw["matches"]
    tba_index = _build_tba_index(tba_matches)

    # --- 2. Find all scout-output JSON files ---
    scout_path = Path(scout_dir)
    candidate_files: List[Path] = []
    if scout_path.is_dir():
        # Non-recursive top level
        candidate_files.extend(p for p in scout_path.glob("*.json"))
        # One level of subdirectories (e.g. refactor_check/, anchor/)
        for sub in scout_path.iterdir():
            if sub.is_dir():
                candidate_files.extend(sub.glob("*.json"))

    # --- 3. Parse each file ---
    reads: List[VisionRead] = []
    n_skipped_midmatch = 0
    n_skipped_no_tba = 0
    n_skipped_no_fuel = 0
    pairs: List = []

    for fpath in sorted(candidate_files):
        vr = _parse_scout_file(fpath, event_key)
        if vr is None:
            continue
        reads.append(vr)

        if verbose:
            print(f"  {vr.source_file:45s} timer={vr.timer_s!r:8} is_final={vr.is_final}")

        # Gate 1: end-of-match only
        if not vr.is_final:
            n_skipped_midmatch += 1
            continue

        # Gate 2: must have a TBA match key and a match in the index
        if not vr.tba_match_key or vr.tba_match_key not in tba_index:
            n_skipped_no_tba += 1
            if verbose:
                print(f"    -> SKIP no TBA match for tba_key={vr.tba_match_key!r}")
            continue

        tba_match = tba_index[vr.tba_match_key]

        # Resolve color swap
        _, color_swapped = _resolve_color_swap(
            vr.blue_teams, vr.red_teams, tba_match
        )
        vr.color_swapped = color_swapped

        # Determine which vision fuel maps to which TBA alliance
        # After swap: vision_blue_fuel -> tba_blue, vision_red_fuel -> tba_red
        # With swap:  vision_blue_fuel -> tba_red,  vision_red_fuel -> tba_blue
        if color_swapped:
            alliance_fuel_pairs = [
                ("blue", vr.red_fuel),   # vision red is TBA blue
                ("red", vr.blue_fuel),   # vision blue is TBA red
            ]
        else:
            alliance_fuel_pairs = [
                ("blue", vr.blue_fuel),
                ("red", vr.red_fuel),
            ]

        match_produced_pair = False
        for tba_alliance, vis_fuel in alliance_fuel_pairs:
            if vis_fuel is None:
                n_skipped_no_fuel += 1
                if verbose:
                    print(f"    -> SKIP {tba_alliance}: vision fuel is None")
                continue

            hub = _hub_score(tba_match, tba_alliance)

            # fuel_total
            if hub["total_count"] is not None:
                pairs.append(PairedSample(
                    match_key=vr.tba_match_key,
                    comp_level=vr.comp_level,
                    alliance=tba_alliance,
                    component="fuel_total",
                    vision_est=vis_fuel,
                    tba_actual=float(hub["total_count"]),
                    event_key=vr.event_key,
                ))
                match_produced_pair = True

            # fuel_auto (only if we have a vision auto breakdown)
            # Note: current overlay only emits total fuel, not auto/teleop split.
            # We skip auto/teleop sub-pairs unless a future overlay provides them.

        if not match_produced_pair and verbose:
            print(f"    -> no pairs produced for {vr.tba_match_key}")

    # --- 4. Write pairs.jsonl ---
    if out_jsonl is not None and pairs:
        out_path = Path(out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for p in pairs:
            obj = {
                "match_key": p.match_key,
                "comp_level": p.comp_level,
                "alliance": p.alliance,
                "component": p.component,
                "vision_est": p.vision_est,
                "tba_actual": p.tba_actual,
                "event_key": p.event_key,
            }
            lines.append(json.dumps(obj))
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if verbose:
            print(f"\nWrote {len(pairs)} pairs to {out_path}")
    elif out_jsonl is not None:
        # Write empty file so downstream tools don't fail on missing path
        out_path = Path(out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
        if verbose:
            print(f"\nNo pairs collected — wrote empty {out_path}")

    return CollectResult(
        pairs=pairs,
        n_pairs=len(pairs),
        n_skipped_midmatch=n_skipped_midmatch,
        n_skipped_no_tba=n_skipped_no_tba,
        n_skipped_no_fuel=n_skipped_no_fuel,
        reads=reads,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Collect (vision, TBA) calibration pairs from scout-output JSON files."
    )
    parser.add_argument(
        "--scout-dir", default=str(_DEFAULT_SCOUT_DIR),
        help="Directory of scout-output JSON files."
    )
    parser.add_argument(
        "--tba-json", default=str(_DEFAULT_TBA_JSON),
        help="Path to the TBA event matches JSON file."
    )
    parser.add_argument(
        "--event-key", default="2026ctwat",
        help="TBA event key (default: 2026ctwat)."
    )
    parser.add_argument(
        "--out", default=str(_DEFAULT_OUT_JSONL),
        help="Output JSONL path for pairs."
    )
    parser.add_argument(
        "--fit", action="store_true",
        help="After collecting pairs, run ErrorCorrectionModel.fit() and print results."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-file progress."
    )
    args = parser.parse_args()

    # Ensure calib/ is importable
    v2_root = Path(__file__).resolve().parents[1]
    for p in (str(v2_root), str(v2_root / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)

    result = collect_pairs(
        scout_dir=Path(args.scout_dir),
        tba_json=Path(args.tba_json),
        event_key=args.event_key,
        out_jsonl=Path(args.out),
        verbose=args.verbose,
    )

    print("\n=== collect_pairs result ===")
    print(f"  n_pairs             : {result.n_pairs}")
    print(f"  n_skipped_midmatch  : {result.n_skipped_midmatch}  (timer_s > 0 — not end-of-match)")
    print(f"  n_skipped_no_tba    : {result.n_skipped_no_tba}   (final read but no TBA match found)")
    print(f"  n_skipped_no_fuel   : {result.n_skipped_no_fuel}   (final + TBA matched, vision fuel None)")
    print(f"  total reads parsed  : {len(result.reads)}")

    if result.n_pairs == 0:
        print("\n  HONEST RESULT: 0 usable calibration pairs found.")
        print("  All on-disk vision reads are mid-match snapshots (timer_s > 0).")
        print("  Calibration requires end-of-match reads (timer_s == 0).")
        print("  Model will return identity calibration (a=1, b=0) — no fit possible.")

    if args.fit:
        from calib.error_model import ErrorCorrectionModel
        print("\n=== ErrorCorrectionModel.fit() ===")
        model = ErrorCorrectionModel(event_key=args.event_key)
        artifact = model.fit(Path(args.out))
        print(f"  n_total_pairs : {artifact.n_total_pairs}")
        for w in artifact.warnings:
            print(f"  WARNING: {w}")
        for comp, calib in artifact.components.items():
            if comp == "fuel_total" or not calib.is_identity:
                print(f"  component={comp}: a={calib.a:.4f} b={calib.b:.4f} "
                      f"sigma={calib.sigma:.2f} n_train={calib.n_train} "
                      f"identity={calib.is_identity}")


if __name__ == "__main__":
    _cli()
