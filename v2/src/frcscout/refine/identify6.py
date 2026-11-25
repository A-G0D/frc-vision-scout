"""Six-robot team-ID resolver (report-time, retroactive).

Where this sits
---------------
`fuse/identity.py` consumes an ALREADY-decided ``{track_id: team#}`` anchor map (one
clean number per track, produced first-writer-wins inside the Re-ID registry) and closes
the rest by elimination. This module is the layer just UPSTREAM of that: it turns the raw,
noisy, time-series *bumper OCR reads* — many per track, frequently disagreeing, sometimes
off-roster garbage — into trustworthy anchors, and only then closes by elimination.

The problem it actually solves
------------------------------
In a wide broadcast shot the bumper digits are tiny and motion-blurred, so a single track
accumulates reads like ``[254, 254, 24, 254, 1114]`` across the match. Picking the team for
that track is a *voting + confidence* problem, not a lookup:

  * The broadcast OVERLAY is ground truth for *which* six teams are on the field (3 red,
    3 blue). A read that is not one of those six is OCR noise and is discarded outright —
    this alone removes most spurious reads (e.g. a "24" that is really a partial "254").
  * Reads are further gated to the track's OWN alliance roster (bumper colour is
    unambiguous at this zoom), so a blue number misread on a red robot cannot anchor it.
  * Per team, evidence accumulates as a confidence-weighted tally. A track LOCKS to a team
    (method ``anchor``) only when that team's evidence both clears ``anchor_min_conf`` and
    beats the runner-up by ``anchor_margin``. A track whose reads are genuinely split
    (e.g. 254 vs 255 with no clear winner) stays ``unknown`` rather than guessing.
  * If two tracks lock onto the SAME team, the stronger evidence keeps it; the weaker track
    is released back to ``unknown`` (it may still be recovered by elimination below).

Closure by elimination (iterated to a fixpoint)
------------------------------------------------
Per alliance: with the three roster teams known and tracks split by bumper colour, an
alliance that has exactly one unresolved track and exactly one still-unassigned team forces
that pairing (method ``elimination``). Because filling one slot can expose another, the
sweep repeats until nothing changes. Genuine ambiguity (>=2 unresolved tracks sharing >=2
free teams) and Re-ID over-segmentation (more track identities than roster teams) are left
``unknown`` — we never coin-flip between equally likely teams.

Retroactive by construction
---------------------------
Resolution happens ONCE, at report time, over the whole match. The output is a
``track_id -> {team, method, confidence}`` map; callers relabel every event/stat through it,
so a number read at second 130 retro-labels that track's actions from second 0. ``apply``
performs that relabel on a match record (same contract as ``fuse.identity.apply_identities``).

Pure logic — no cv2 / numpy / scipy / easyocr. The whole module is unit-testable with plain
Python ints; OCR reads are passed in as data, never produced here.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence, Union

from frcscout.io.types import Alliance

# Method labels (kept identical to fuse.identity.ResolvedId.method for a uniform downstream).
ANCHOR = "anchor"
ELIMINATION = "elimination"
UNKNOWN = "unknown"

# Tunable defaults. Two quantities are tracked per (track, team):
#   * evidence  — noisy-OR of the reads' confidences, a saturating score in [0, 1] used as
#     the lock-strength / reported confidence ("how sure are we this track is this team").
#   * mass      — plain SUM of the reads' confidences (~ a confidence-weighted read count),
#     used to decide the WINNER's margin over the runner-up. Mass, not evidence, drives the
#     margin: three agreeing reads (mass 2.4) decisively beat one competing read (mass 0.8),
#     yet a genuine 1-vs-1 split (equal mass) is correctly indecisive — whereas noisy-OR
#     saturates so a single mediocre read looks deceptively close to a triple-confirmed one.
DEFAULT_ANCHOR_MIN_CONF = 0.55   # min winner evidence (noisy-OR) for a team to lock a track
DEFAULT_ANCHOR_MARGIN = 0.20     # winner mass must exceed runner-up mass by this (read-count units)
DEFAULT_MIN_READ_CONF = 0.0      # drop individual reads below this OCR confidence
ELIMINATION_CONF = 0.9           # fixed confidence stamped on an elimination fill
# A single read with no confidence supplied is treated as this weight, so callers can pass
# bare team numbers (ints) and still get sane behaviour.
DEFAULT_READ_CONF = 0.8


@dataclass(frozen=True)
class BumperRead:
    """One bumper-number OCR observation for a track.

    ``conf`` is the OCR engine's confidence in [0, 1]; ``frame_idx`` is optional and used
    only for diagnostics / tie ordering (resolution itself is order-independent)."""
    team: int
    conf: float = DEFAULT_READ_CONF
    frame_idx: Optional[int] = None


@dataclass
class ResolvedId:
    """Per-track resolution. Mirrors fuse.identity.ResolvedId plus a small evidence trail
    so a report can show *why* a track was (or was not) identified."""
    track_id: int
    team: Optional[int]
    alliance: Optional[Alliance]
    method: str                              # ANCHOR | ELIMINATION | UNKNOWN
    confidence: float
    votes: dict[int, float] = field(default_factory=dict)   # team -> accumulated evidence
    n_reads: int = 0                                         # on-roster reads counted

    def as_dict(self) -> dict:
        """The compact ``{team, method, confidence}`` record the task asks for (alliance and
        the evidence trail are included; downstream may ignore the extras)."""
        return {
            "team": self.team,
            "method": self.method,
            "confidence": self.confidence,
            "alliance": self.alliance,
        }


# A reading for one track may be given as: a list of BumperRead, a list of plain ints
# (each an equal-weight vote), a list of (team, conf) pairs, or a Counter-like
# {team: weight} mapping. We normalise all of these.
TrackReads = Union[
    Sequence[BumperRead],
    Sequence[int],
    Sequence[tuple],
    Mapping[int, float],
]


def _normalise_reads(reads: Optional[TrackReads], min_read_conf: float) -> list[BumperRead]:
    """Coerce any accepted read container into a list[BumperRead], dropping low-conf reads."""
    if not reads:
        return []
    out: list[BumperRead] = []
    if isinstance(reads, Mapping):
        for team, w in reads.items():
            try:
                conf = float(w)
            except (TypeError, ValueError):
                continue
            if conf >= min_read_conf and conf > 0.0:
                out.append(BumperRead(int(team), conf))
        return out
    for r in reads:
        if isinstance(r, BumperRead):
            br = r
        elif isinstance(r, (tuple, list)) and len(r) >= 1:
            team = int(r[0])
            conf = float(r[1]) if len(r) >= 2 and r[1] is not None else DEFAULT_READ_CONF
            br = BumperRead(team, conf)
        else:
            # a bare team number
            try:
                br = BumperRead(int(r), DEFAULT_READ_CONF)
            except (TypeError, ValueError):
                continue
        if br.conf >= min_read_conf:
            out.append(br)
    return out


def _evidence(confs: Iterable[float]) -> float:
    """Combine several independent OCR confidences for the SAME team into one evidence
    score in [0, 1] via noisy-OR: ``1 - prod(1 - c)``. Two mediocre 0.5 reads (-> 0.75)
    outweigh one, and many agreeing reads saturate toward 1 — exactly the "read it once,
    you basically know it; read it a lot, you're certain" behaviour we want. Confidences
    are clamped to [0, 1) so a stray 1.0 cannot make the product collapse to a hard 1."""
    prod_miss = 1.0
    for c in confs:
        c = 0.0 if c < 0.0 else (0.999 if c >= 1.0 else c)
        prod_miss *= (1.0 - c)
    return 1.0 - prod_miss


def _tally(reads: list[BumperRead], roster: set[int]) -> tuple[dict[int, float], dict[int, float]]:
    """Per-team (evidence, mass) for one track, restricted to teams on this track's roster
    (the overlay's six, optionally narrowed to the track's alliance). Off-roster reads —
    the bulk of OCR noise — are dropped here. Returns (evidence_by_team, mass_by_team)."""
    by_team: dict[int, list[float]] = defaultdict(list)
    for r in reads:
        if r.team in roster:
            by_team[r.team].append(max(0.0, r.conf))
    evidence = {team: _evidence(cs) for team, cs in by_team.items()}
    mass = {team: float(sum(cs)) for team, cs in by_team.items()}
    return evidence, mass


def _best_two(evidence: dict[int, float], mass: dict[int, float]):
    """Pick the winning team by mass (confidence-weighted read count), tie-broken by the
    smaller team number for determinism. Return
    ``(winning_team, winner_evidence, winner_mass, runner_up_mass)``."""
    if not mass:
        return None, 0.0, 0.0, 0.0
    ordered = sorted(mass.items(), key=lambda kv: (kv[1], -kv[0]), reverse=True)
    win_team, win_mass = ordered[0]
    runner_mass = ordered[1][1] if len(ordered) > 1 else 0.0
    return win_team, evidence.get(win_team, 0.0), win_mass, runner_mass


def resolve_six(
    track_alliances: Mapping[int, Optional[Alliance]],
    bumper_reads: Mapping[int, TrackReads],
    overlay_teams: Mapping[str, Sequence[int]],
    *,
    anchor_min_conf: float = DEFAULT_ANCHOR_MIN_CONF,
    anchor_margin: float = DEFAULT_ANCHOR_MARGIN,
    min_read_conf: float = DEFAULT_MIN_READ_CONF,
) -> dict[int, ResolvedId]:
    """Resolve up to six robot tracks to real FRC team numbers.

    Args:
      track_alliances: ``{track_id: 'red'|'blue'|None}`` — the SixRobotTracker's per-track
        alliance (held bumper-colour vote). Defines the universe of tracks to resolve.
      bumper_reads: ``{track_id: reads}`` — time-series OCR reads per track (any of the
        accepted forms; see ``TrackReads``). Tracks absent here simply have no reads.
      overlay_teams: ``{'red':[3], 'blue':[3]}`` — the six roster teams from the overlay
        (ground truth). Reads not in a track's alliance roster are discarded as noise.
      anchor_min_conf / anchor_margin: lock thresholds (evidence units, see module docs).
      min_read_conf: drop individual OCR reads weaker than this before tallying.

    Returns ``{track_id: ResolvedId}`` for every track in ``track_alliances``. Resolution is
    order-independent and idempotent, and is meant to run once at report time so its result
    relabels each track's whole timeline (use ``apply``).
    """
    red_roster = [int(t) for t in (overlay_teams.get("red") or [])]
    blue_roster = [int(t) for t in (overlay_teams.get("blue") or [])]
    roster_by_alliance: dict[Alliance, set[int]] = {
        "red": set(red_roster),
        "blue": set(blue_roster),
    }
    all_roster = set(red_roster) | set(blue_roster)

    # ---- per-track tally (alliance-gated when alliance known, else full roster) ----
    evid: dict[int, dict[int, float]] = {}   # track -> {team: noisy-OR evidence}
    mass: dict[int, dict[int, float]] = {}   # track -> {team: confidence-weighted count}
    n_reads: dict[int, int] = {}
    for tid, alliance in track_alliances.items():
        reads = _normalise_reads(bumper_reads.get(tid), min_read_conf)
        roster = roster_by_alliance.get(alliance, all_roster) if alliance else all_roster
        evid[tid], mass[tid] = _tally(reads, roster)
        n_reads[tid] = sum(1 for r in reads if r.team in roster)

    out: dict[int, ResolvedId] = {
        tid: ResolvedId(
            track_id=tid, team=None, alliance=track_alliances.get(tid),
            method=UNKNOWN, confidence=0.0, votes=evid.get(tid, {}),
            n_reads=n_reads.get(tid, 0),
        )
        for tid in track_alliances
    }

    # ---- 1) ANCHOR: each track's confident, unambiguous winner ----
    # A track locks to a team when that team's evidence clears anchor_min_conf AND its mass
    # exceeds the runner-up's mass by anchor_margin (so a split vote stays unknown).
    # Provisional anchors first, then resolve any team claimed by two tracks (keep the
    # stronger; release the weaker to UNKNOWN so elimination can still recover it).
    provisional: list[tuple[int, int, float]] = []   # (track_id, team, evidence)
    for tid in track_alliances:
        win_team, win_ev, win_mass, runner_mass = _best_two(evid[tid], mass[tid])
        if (win_team is not None
                and win_ev >= anchor_min_conf
                and (win_mass - runner_mass) >= anchor_margin):
            provisional.append((tid, win_team, win_ev))

    # strongest evidence wins a contested team number
    provisional.sort(key=lambda x: x[2], reverse=True)
    team_owner: dict[int, int] = {}                  # team -> winning track_id
    for tid, team, ev in provisional:
        if team not in team_owner:
            team_owner[team] = tid
            r = out[tid]
            r.team = team
            r.method = ANCHOR
            r.confidence = min(1.0, ev)
        # else: this track lost the contest for `team`; it stays UNKNOWN for now.

    # ---- 2) CLOSURE BY ELIMINATION, iterated to a fixpoint ----
    changed = True
    while changed:
        changed = False
        for alliance in ("red", "blue"):
            roster = roster_by_alliance.get(alliance, set())  # type: ignore[arg-type]
            tracks_here = [t for t in track_alliances if track_alliances.get(t) == alliance]

            assigned_teams = {
                out[t].team for t in tracks_here if out[t].team is not None
            }
            free_teams = [tm for tm in roster if tm not in assigned_teams]
            unresolved = [t for t in tracks_here if out[t].team is None]

            if len(unresolved) == 1 and len(free_teams) == 1:
                t = unresolved[0]
                out[t].team = free_teams[0]
                out[t].method = ELIMINATION
                out[t].confidence = ELIMINATION_CONF
                changed = True
            # >1 unresolved (ambiguous) or 0 free with leftovers (over-segmented): leave as-is.

    return out


def resolutions_to_dict(resolved: Mapping[int, ResolvedId]) -> dict[int, dict]:
    """Project the resolver output to the exact ``{track_id: {team, method, confidence}}``
    shape the task specifies (alliance carried along for convenience)."""
    return {tid: r.as_dict() for tid, r in resolved.items()}


def apply(match_record: dict, resolved: Mapping[int, ResolvedId]) -> dict:
    """Retroactively relabel a match record's events + per-robot stats with the resolved
    team numbers. Keeps the original ``robot_id`` (track id) and adds ``team`` / ``id_method``
    / ``id_confidence`` beside it, so nothing is lost and a late read propagates backward.

    Same contract as ``fuse.identity.apply_identities``; mutates and returns ``match_record``.
    """
    def stamp(d: dict, tid_key: str = "robot_id") -> None:
        tid = d.get(tid_key)
        r = resolved.get(tid) if tid is not None else None
        if r is not None:
            d["team"] = r.team
            d["id_method"] = r.method
            d["id_confidence"] = r.confidence

    for ev in match_record.get("events", []):
        if isinstance(ev, dict):
            stamp(ev)

    robots = match_record.get("robots", {})
    relabeled: dict = {}
    for tid, stats in robots.items():
        key = tid if isinstance(tid, int) else _as_int(tid)
        r = resolved.get(key)
        if r is not None and isinstance(stats, dict):
            stats = {**stats, "team": r.team,
                     "id_method": r.method, "id_confidence": r.confidence}
        relabeled[tid] = stats
    match_record["robots"] = relabeled
    return match_record


def _as_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return x
