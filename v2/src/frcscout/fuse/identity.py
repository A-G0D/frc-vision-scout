"""Retroactive robot identity resolution + closure-by-elimination.

The user's insight: each robot carries an internal track ID the whole match; if we read
its bumper team-number even ONCE, we know which physical team that track is — and because
events store the *track ID* (not a baked-in team number), resolving track->team at REPORT
time relabels the track's entire history, backward and forward, for free.

Two cues, in priority order:
  1. ANCHOR    — a confident bumper-number OCR read bound track_id<->team# (ReIDManager
                 registry). Highest confidence; survives any gap.
  2. ELIMINATION — the broadcast overlay gives the exact 6 teams (3 red + 3 blue), and
                 bumper color splits the robot tracks into those two buckets. So we don't
                 need to read every number: if an alliance has one unanchored track and one
                 unassigned team, that track IS that team by elimination.

Anything still ambiguous (≥2 unanchored tracks sharing ≥2 unassigned teams, or more track
identities than teams = a Re-ID under-merge) is left team=None, method="unknown" — we never
guess between equally-likely teams.

Pure logic; no cv2/model. Wire `resolve_identities` at report time in StateFuser using the
final ReID registry, the per-track alliance votes, and the overlay's team lists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from frcscout.io.types import Alliance


@dataclass
class ResolvedId:
    track_id: int
    team: Optional[int]          # the real FRC team number, or None if unresolved
    alliance: Optional[Alliance]
    method: str                  # "anchor" | "elimination" | "unknown"
    confidence: float            # 1.0 anchor, ~0.9 elimination, 0.0 unknown


def resolve_identities(
    track_alliances: dict[int, Optional[Alliance]],
    anchored: dict[int, int],                 # track_id -> team# (ReIDManager.team_of_track)
    overlay_teams: dict[str, list[int]],      # {"red":[...3], "blue":[...3]} from the overlay
) -> dict[int, ResolvedId]:
    """Resolve every known robot track to a real team number where possible.

    Retroactive by construction: callers map event/stat track_ids through this result, so a
    late OCR read relabels the whole track timeline.
    """
    out: dict[int, ResolvedId] = {}

    # 1) anchors — confident OCR binds
    for tid, team in anchored.items():
        out[tid] = ResolvedId(tid, team, track_alliances.get(tid), "anchor", 1.0)

    # 2) elimination, per alliance
    for alliance in ("red", "blue"):
        known = list(overlay_teams.get(alliance, []) or [])
        tracks_here = [t for t, a in track_alliances.items() if a == alliance]

        assigned_teams = {out[t].team for t in tracks_here if t in out and out[t].team is not None}
        remaining_teams = [tm for tm in known if tm not in assigned_teams]
        unassigned_tracks = [t for t in tracks_here if t not in out]

        if len(unassigned_tracks) == 1 and len(remaining_teams) == 1:
            t = unassigned_tracks[0]
            out[t] = ResolvedId(t, remaining_teams[0], alliance, "elimination", 0.9)
        else:
            # ambiguous (multiple candidates) or over-segmented (more tracks than teams)
            for t in unassigned_tracks:
                out[t] = ResolvedId(t, None, alliance, "unknown", 0.0)

    # tracks with no alliance at all
    for t, a in track_alliances.items():
        if t not in out:
            out[t] = ResolvedId(t, None, a, "unknown", 0.0)

    return out


def apply_identities(match_record: dict, resolved: dict[int, ResolvedId]) -> dict:
    """Retroactively relabel a match_record's events + per-robot stats with team numbers.

    Adds `team` / `id_method` / `id_confidence` alongside the original `robot_id` (we keep the
    track id so nothing is lost). Returns the same dict (mutated) for convenience.
    """
    def stamp(d: dict, tid_key: str = "robot_id"):
        tid = d.get(tid_key)
        r = resolved.get(tid) if tid is not None else None
        if r is not None:
            d["team"] = r.team
            d["id_method"] = r.method
            d["id_confidence"] = r.confidence

    for ev in match_record.get("events", []):
        stamp(ev)

    robots = match_record.get("robots", {})
    relabeled = {}
    for tid, stats in robots.items():
        r = resolved.get(tid if isinstance(tid, int) else _as_int(tid))
        if r is not None and isinstance(stats, dict):
            stats = {**stats, "team": r.team, "id_method": r.method, "id_confidence": r.confidence}
        relabeled[tid] = stats
    match_record["robots"] = relabeled
    return match_record


def _as_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return x
