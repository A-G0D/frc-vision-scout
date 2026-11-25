"""Event detection in FIELD coordinates (camera-invariant).

PICKUP : a tracked ball vanishes within pickup.radius_m of a robot (field meters).
SHOT   : a flight track sustains > shot.min_speed_mps over shot.min_travel_m, directed
         at a HUB volume (ballistic arc impact inside the HUB rim).
SCORE  : derived from the overlay fuel/score delta — the AUTHORITATIVE count. CV never
         counts the score; it only explains WHO/HOW.

All thresholds come from configs/events.yaml in SECONDS/METERS and are converted to
frames with the real fps. No hardcoded frame counts (the v1 bug).

Richer scouting events
----------------------
Beyond the original pickup/shot/score, this detector now drives a
``PossessionTracker`` (fuse/possession.py) and surfaces its per-frame events:
``intake_attempt``, ``intake_success``, ``shot_attempt``, ``shot_success``. These
are the per-robot ball-interaction signals the scouting record aggregates. The
overlay remains AUTHORITATIVE for the count (``score`` events are unchanged); the
possession layer adds *who/how* and feeds overlay score pulses back so a shot can
be confirmed against the real count when the arc test is inconclusive.
"""
from __future__ import annotations

from frcscout.io.config import Config, seconds_to_frames
from frcscout.io.types import Event, FrameState
from frcscout.fuse.possession import PossessionTracker


class EventDetector:
    def __init__(self, field, events_config: str, fps: float):
        self.field = field
        self.cfg = Config.load(events_config)
        self.fps = fps
        self.pickup_dedup = seconds_to_frames(self.cfg.get("pickup.dedup_per_robot_s", 1.5), fps)
        self.shot_lookahead = seconds_to_frames(self.cfg.get("shot.lookahead_s", 4.0), fps)
        self._last_overlay = {"blue_fuel": None, "red_fuel": None,
                              "blue_score": None, "red_score": None}
        # per-robot possession state machine (intake/shot scouting metrics)
        poss_cfg = dict(self.cfg.get("possession", {}) or {})
        # share the events.yaml shot lookahead with the possession resolver
        poss_cfg.setdefault("shot_lookahead_s", self.cfg.get("shot.lookahead_s", 4.0))
        poss_cfg.setdefault("intake_zone_m", self.cfg.get("pickup.radius_m", 0.6))
        poss_cfg.setdefault("intake_dedup_s", self.cfg.get("pickup.dedup_per_robot_s", 1.0))
        self.possession = PossessionTracker(field, poss_cfg, fps)

    def step(self, st: FrameState) -> list[Event]:
        """Detect score deltas from the overlay (authoritative) and the richer
        per-robot intake/shot events from the possession state machine."""
        events: list[Event] = []
        score_evs = self._score_from_overlay(st)
        # feed authoritative score pulses to the possession layer for shot
        # confirmation BEFORE stepping it this frame.
        for e in score_evs:
            if e.alliance is not None:
                self.possession.note_overlay_score(st.frame_idx, e.alliance)
        events += score_evs
        events += self.possession.step(st, self.fps)
        return events

    def _score_from_overlay(self, st: FrameState) -> list[Event]:
        out: list[Event] = []
        for al in ("blue", "red"):
            cur = getattr(st.overlay, f"{al}_fuel")
            prev = self._last_overlay[f"{al}_fuel"]
            if cur is not None and prev is not None and cur > prev:
                out.append(Event(t_s=st.t_s, frame_idx=st.frame_idx, type="score",
                                 alliance=al, detail={"fuel_delta": cur - prev}))
            if cur is not None:
                self._last_overlay[f"{al}_fuel"] = cur
        return out
