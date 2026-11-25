"""StateFuser — ties event detection + attribution into the per-frame game state, and
accumulates the match-level scouting record (robot trajectories, shot maps, cycle times).
"""
from __future__ import annotations

from frcscout.fuse.attribution import Attributor
from frcscout.fuse.events import EventDetector
from frcscout.io.types import Event, FrameState


class StateFuser:
    def __init__(self, field, events_config: str, fps: float):
        self.events = EventDetector(field, events_config, fps)
        self.attrib = Attributor.from_config(self.events.cfg)
        self._shot_buffer: list[Event] = []     # recent shots awaiting score correlation
        self.match_record: dict = {"robots": {}, "shots": [], "events": []}

    def update(self, st: FrameState) -> list[Event]:
        evs = self.events.step(st)
        # PossessionTracker emits "shot_attempt"/"shot_success" (not the legacy
        # "shot" literal) — these are the candidate shots the Attributor windows
        # against overlay score deltas.
        self._shot_buffer += [e for e in evs if e.type in ("shot_attempt", "shot_success")]
        # bound the buffer: drop shots older than the attribution look-back window.
        cutoff_t = st.t_s - self.attrib.window_s
        self._shot_buffer = [e for e in self._shot_buffer if e.t_s >= cutoff_t]
        score_evs = [e for e in evs if e.type == "score"]
        attributed = self.attrib.attribute(score_evs, self._shot_buffer)
        out = [e for e in evs if e.type != "score"] + attributed
        self.match_record["events"] += [e.__dict__ for e in out]
        # roll the live per-robot scouting counters into the match record. The
        # PossessionTracker owns the authoritative running totals; we snapshot
        # them each frame so the record reflects the latest state.
        self.match_record["robots"] = self.events.possession.stats_dict()
        # accumulate each robot track's alliance over the match (for report-time identity)
        self._track_alliances = getattr(self, "_track_alliances", {})
        for t in st.robots:
            if t.alliance is not None:
                self._track_alliances[t.tid] = t.alliance
        return out

    def finalize_identities(self, reid_registry, overlay_teams, track_alliances=None):
        """Report-time pass: resolve internal track IDs to real team numbers (OCR anchors +
        closure-by-elimination against the overlay's 6 teams) and relabel the WHOLE
        match_record retroactively — a number read once, anywhere, propagates backward.

        Inputs:
          reid_registry   : RobotTracker.reid.team_of_track ({track_id: team#}) or {}
          overlay_teams   : {"red":[...3], "blue":[...3]} from the overlay
          track_alliances : optional override; defaults to the alliances accumulated here.
        Returns the team-labeled match_record (also stored in self.identities)."""
        from frcscout.fuse.identity import resolve_identities, apply_identities
        tal = track_alliances if track_alliances is not None else getattr(self, "_track_alliances", {})
        resolved = resolve_identities(tal, reid_registry or {}, overlay_teams or {})
        self.identities = {t: r.__dict__ for t, r in resolved.items()}
        return apply_identities(self.match_record, resolved)
