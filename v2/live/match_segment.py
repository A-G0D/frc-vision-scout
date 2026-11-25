"""Match segmentation + identification from the overlay stream (RUBRIC_v2 D2).

A continuous broadcast/livestream is a sequence of matches separated by results
screens, replays, and downtime. This module turns the per-frame `OverlayState`
(from `detect/overlay_ocr.OverlayReader`) into discrete, identified matches:

  * SEGMENT: a debounced state machine IDLE -> IN_MATCH -> ENDED driven by the live
    timer (counts down ~150->0), phase (auto->teleop->endgame), and the results
    screen. Emits one MatchSegment per real match; replays/timeouts don't double-count.
  * IDENTIFY: vote comp_level + match_number + alliance teams across the segment's
    frames and build the TBA match key  f"{event_key}_{comp}{num}"  (e.g. 2026ctwat_qm59).

The state-machine logic is a PURE function of a lightweight per-frame tuple, so it is
unit-testable with synthetic overlay sequences and no OCR/cv2. The driver class wraps
an OverlayReader for live use.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional


# --------------------------------------------------------------------------- #
# Lightweight per-frame observation (decoupled from OverlayState so the core is
# testable without importing cv2/easyocr).
# --------------------------------------------------------------------------- #
@dataclass
class OverlayObs:
    idx: int                                  # frame index (or stream tick)
    layout: str = "unknown"                   # live_bar | qual_results | playoff_results | unknown
    timer_s: Optional[float] = None
    phase: Optional[str] = None               # auto | teleop | endgame
    blue_score: Optional[int] = None
    red_score: Optional[int] = None
    comp_level: Optional[str] = None          # qm | sf | f
    match_number: Optional[int] = None
    blue_teams: list[int] = field(default_factory=list)
    red_teams: list[int] = field(default_factory=list)


@dataclass
class MatchSegment:
    start_idx: int
    end_idx: Optional[int] = None
    obs: list[OverlayObs] = field(default_factory=list)
    match_key: Optional[str] = None
    comp_level: Optional[str] = None
    match_number: Optional[int] = None
    blue_teams: list[int] = field(default_factory=list)
    red_teams: list[int] = field(default_factory=list)
    final_blue_score: Optional[int] = None
    final_red_score: Optional[int] = None
    id_confidence: float = 0.0


# --------------------------------------------------------------------------- #
# Tunables (debounce + timer logic). Fractions/seconds, resolution-independent.
# --------------------------------------------------------------------------- #
_CONFIRM_FRAMES = 3          # consecutive confirmations to flip a state
_TIMER_START_MIN = 120.0     # a live timer this high = a fresh match (2026 ~150s match)
_TIMER_LOW = 2.0             # timer at/below this = match basically over
_RESULTS_LAYOUTS = ("qual_results", "playoff_results")


class MatchSegmenter:
    """Debounced IDLE->IN_MATCH->ENDED machine over an OverlayObs stream."""

    def __init__(self, event_key: Optional[str] = None,
                 confirm_frames: int = _CONFIRM_FRAMES):
        self.event_key = event_key
        self.confirm = confirm_frames
        self.state = "IDLE"
        self.cur: Optional[MatchSegment] = None
        self._pending = 0
        self._pending_to: Optional[str] = None
        self._last_timer: Optional[float] = None
        self.completed: list[MatchSegment] = []

    # ----- pure transition decision ---------------------------------------- #
    def _want_state(self, o: OverlayObs) -> str:
        """What state this single observation argues for (pre-debounce)."""
        if o.layout in _RESULTS_LAYOUTS:
            return "ENDED" if self.state == "IN_MATCH" else "IDLE"
        if o.layout == "live_bar" and o.timer_s is not None:
            # A high live timer means a match is on (or just started).
            if o.timer_s >= _TIMER_START_MIN:
                return "IN_MATCH"
            # timer mid-range while already in a match -> stay
            if self.state == "IN_MATCH":
                # near-zero timer for real -> heading to ENDED (results will confirm)
                if o.timer_s <= _TIMER_LOW and self._last_timer is not None \
                        and self._last_timer <= _TIMER_LOW + 5:
                    return "IN_MATCH"   # let results screen drive the actual end
                return "IN_MATCH"
        return self.state  # no opinion -> hold

    def push(self, o: OverlayObs) -> Optional[MatchSegment]:
        """Feed one observation. Returns a finalized MatchSegment when one closes."""
        want = self._want_state(o)
        finalized: Optional[MatchSegment] = None

        # debounce: count consecutive identical "want != state" votes
        if want != self.state:
            if want == self._pending_to:
                self._pending += 1
            else:
                self._pending_to, self._pending = want, 1
            if self._pending >= self.confirm:
                finalized = self._transition(want, o)
                self._pending, self._pending_to = 0, None
        else:
            self._pending, self._pending_to = 0, None

        # accumulate frames while in a match
        if self.state == "IN_MATCH" and self.cur is not None:
            self.cur.obs.append(o)
            if o.blue_score is not None:
                self.cur.final_blue_score = o.blue_score
            if o.red_score is not None:
                self.cur.final_red_score = o.red_score
        # capture identity off the results screen too
        if o.layout in _RESULTS_LAYOUTS and self.cur is not None and self.state != "IDLE":
            self.cur.obs.append(o)

        self._last_timer = o.timer_s if o.timer_s is not None else self._last_timer
        return finalized

    def _transition(self, want: str, o: OverlayObs) -> Optional[MatchSegment]:
        finalized = None
        if want == "IN_MATCH" and self.state in ("IDLE", "ENDED"):
            self.state = "IN_MATCH"
            self.cur = MatchSegment(start_idx=o.idx)
        elif want == "ENDED" and self.state == "IN_MATCH":
            self.state = "ENDED"
            if self.cur is not None:
                self.cur.end_idx = o.idx
                identify_segment(self.cur, self.event_key)
                self.completed.append(self.cur)
                finalized = self.cur
                self.cur = None
        elif want == "IDLE":
            self.state = "IDLE"
        return finalized

    def flush(self) -> Optional[MatchSegment]:
        """Close an open match at stream end (no trailing results screen)."""
        if self.state == "IN_MATCH" and self.cur is not None and self.cur.obs:
            self.cur.end_idx = self.cur.obs[-1].idx
            identify_segment(self.cur, self.event_key)
            self.completed.append(self.cur)
            seg, self.cur, self.state = self.cur, None, "IDLE"
            return seg
        return None


def _vote(values: Iterable) -> Optional[object]:
    c = Counter(v for v in values if v is not None)
    return c.most_common(1)[0][0] if c else None


def _vote_teams(team_lists: Iterable[list[int]]) -> list[int]:
    """Most-common team at each of the 3 slots, then de-dup preserving order."""
    slots: list[Counter] = [Counter(), Counter(), Counter()]
    for tl in team_lists:
        for i, t in enumerate(tl[:3]):
            slots[i][t] += 1
    out: list[int] = []
    for s in slots:
        if s:
            t = s.most_common(1)[0][0]
            if t not in out:
                out.append(t)
    return out


def identify_segment(seg: MatchSegment, event_key: Optional[str]) -> MatchSegment:
    """Vote identity across the segment's frames and build the TBA match key."""
    comp = _vote(o.comp_level for o in seg.obs)
    num = _vote(o.match_number for o in seg.obs)
    seg.comp_level = comp
    seg.match_number = num
    seg.blue_teams = _vote_teams(o.blue_teams for o in seg.obs)
    seg.red_teams = _vote_teams(o.red_teams for o in seg.obs)
    # confidence = agreement on (comp, num) among frames that reported them
    reps = [(o.comp_level, o.match_number) for o in seg.obs
            if o.comp_level is not None and o.match_number is not None]
    if reps:
        seg.id_confidence = Counter(reps).most_common(1)[0][1] / len(reps)
    if event_key and comp and num is not None:
        seg.match_key = f"{event_key}_{comp}{num}"
    return seg


__all__ = ["OverlayObs", "MatchSegment", "MatchSegmenter", "identify_segment"]
