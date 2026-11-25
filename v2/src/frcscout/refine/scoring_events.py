"""Per-robot Intake / Outtake / SCORED state machine (refine layer, field coords).

This is a *refinement* pass that watches the field-projected ball + robot tracks
frame-by-frame and infers, per robot, the ball-handling lifecycle that scouting
cares about:

    INTAKE   a free ball enters a robot's (size-scaled) intake RADIUS and then
             vanishes inside it  ->  the robot is now POSSESSING a ball.

    OUTTAKE  a possessed/owned ball leaves the robot as a FAST OUTBOUND track
             (a new flight track born at the robot, moving away from it above an
             ejection-speed threshold). Every outtake decrements possession.

    SHOT     an OUTTAKE aimed downfield (outbound, above the shot-speed floor) is
             treated as a scoring ATTEMPT. (A slow drop / hand-off eject counts as
             an outtake but NOT a shot.)

    SCORED   a SHOT's outbound trajectory enters the target HUB region
             (``hub_regions.in_hub``) within a lookahead, AND/OR correlates with an
             overlay fuel-count increment inside a window  ->  MADE; otherwise the
             attempt times out as a MISS.

Per-robot tallies (``RobotScoring``):
    intakes, outtakes, shots, made, missed,
    shot_accuracy = made / (made + missed)          (None when nothing resolvable)

──────────────────────────── intake RADIUS scales to ROBOT SIZE ───────────────
The intake catchment is not a fixed constant — a bigger robot reaches further:

    intake_radius(robot) = max(intake_k * robot_size_m, intake_floor_m)

``robot_size_m`` is the larger projected bbox side in FIELD METERS when the
geometry layer attached one (``robot.field_wh_m``); otherwise we cannot compare a
pixel bbox to meter ball-distances, so we fall back to the configured meter floor.
There is no hardcoded pixel/meter scale and no hardcoded frame count — every window
is given in SECONDS and converted with ``seconds_to_frames`` against the REAL fps.

──────────────────────────────── HUB localization ─────────────────────────────
SCORED needs to know where the HUB is in field meters. The localizer is duck-typed:
any object exposing ``in_hub(field_xy, alliance, tol_m=...) -> bool`` works — the
geometry layer's ``FieldModel`` and a future ``hub_regions`` module both satisfy it.

CONFIDENCE IS NEVER FABRICATED. A made/missed outcome is only *resolvable* — and a
robot's ``shot_accuracy`` only defined — when we can actually judge the shot:

  * a VALID field pose so the airborne ball's ``field_xy`` is trustworthy enough to
    test against the HUB region (see geometry/field.py on z!=0 parallax), OR
  * an overlay score pulse exists in the shot's window to correlate against.

Without either, the shot outcome is UNKNOWN: it is excluded from accuracy and its
SCORED/MISS event is suppressed (we never emit a fabricated make, and never charge a
miss we could not see). The localizer's own confidence (if it exposes
``localization_confidence(alliance)`` or a ``hub_confidence`` attribute) is carried
onto the SCORED event; absent that, a valid pose yields a conservative default and
no pose yields 0.0.

──────────────────────────────── honest caveats ──────────────────────────────
* Ball identity through possession is UNOBSERVABLE: once a ball enters a robot the
  detector loses it. When a ball later launches we ASSUME it is one the robot held
  (held_count > 0), not literally the same physical ball — so we track a per-robot
  ``held_count``, not a ball<->ball thread.
* Concurrent scoring is ambiguous: when several robots' shots fall in one overlay
  window the per-robot credit is probabilistic and is left for the Attributor to
  resolve downstream; this module emits per-shot SCORED events with the localizer
  confidence and does not split an integer overlay delta.
* Thresholds (radius factor, converge/vanish/eject windows) are config defaults that
  need calibration on real footage.

Pure-logic: this module imports no cv2 / numpy / torch at module load (numpy is
imported lazily and only if present), so it parses and unit-tests without them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from frcscout.io.config import seconds_to_frames
from frcscout.io.types import Alliance, Event, FrameState, Track


# ───────────────────────────────── defaults ──────────────────────────────────
# ALL timing in SECONDS, ALL distance in METERS. Never a hardcoded frame count.
_DEFAULTS = {
    # intake catchment RADIUS = intake_k * robot_size_m (size = larger projected
    # bbox side in field meters), floored by intake_floor_m. Scales to robot size.
    "intake_k": 1.2,
    # floor for the scaled radius (m) so a degenerate / non-projected bbox can't
    # collapse the catchment, and the fallback when only a pixel bbox is known.
    "intake_floor_m": 0.6,
    # a ball must be APPROACHING (distance decreasing) at least this long before its
    # entry into the radius is treated as a genuine intake attempt.
    "converge_min_s": 0.1,
    # after a ball disappears, how long we keep looking before deciding it really
    # vanished inside the robot (vs. a one-frame detector miss).
    "vanish_grace_s": 0.4,
    # an attempt is abandoned if the ball drifts beyond this multiple of the radius.
    "abandon_radius_mult": 2.0,
    # min seconds between two intakes credited to one robot (dedup).
    "intake_dedup_s": 1.0,
    # an OUTTAKE flight track must originate within this radius (m) of a possessing
    # robot to be credited as that robot's eject.
    "outtake_origin_m": 1.0,
    # ejection speed floor (m/s, field frame). Below this an outbound track is not a
    # real outtake (detector jitter / a ball nudged a few cm).
    "outtake_min_speed_mps": 1.5,
    # a SHOT is an outtake at or above this speed (m/s). Between the outtake floor and
    # this, the eject counts as an outtake but NOT a scoring shot (slow drop/hand-off).
    "shot_min_speed_mps": 2.5,
    # how far ahead (s) to test the shot's outbound arc for HUB entry.
    "shot_lookahead_s": 4.0,
    # tolerance (m) added to the HUB radius for the in_hub test.
    "hub_tol_m": 0.3,
    # window (s) within which an overlay score increment confirms a shot.
    "score_window_s": 1.5,
    # overlay may lag the physical score by ~1 tick; widen the window's lower edge.
    "score_lag_s": 0.0,
    # confidence assigned to a HUB-geometry resolution when the localizer does NOT
    # expose its own confidence but a valid pose is present.
    "default_hub_confidence": 0.7,
    # confidence for an overlay-only confirmation (authoritative count, but the
    # per-robot attribution is inferential).
    "overlay_confidence": 0.6,
}


# ───────────────────────────── HUB-localizer protocol ─────────────────────────
@runtime_checkable
class HubLocalizer(Protocol):
    """Anything that can answer "is this field point inside ``alliance``'s HUB?".

    ``FieldModel`` (geometry/field.py) and a future ``hub_regions`` module both
    satisfy this. Optional ``localization_confidence(alliance)`` lets a richer
    localizer report how trustworthy its HUB placement is this frame; when absent we
    fall back to pose-gated defaults (see ScoringEventTracker._hub_confidence)."""

    def in_hub(self, field_xy: tuple[float, float], alliance: str,
               tol_m: float = 0.0) -> bool: ...


# ──────────────────────────────── per-robot tally ─────────────────────────────
@dataclass
class RobotScoring:
    """Per-robot Intake/Outtake/Scored counters.

    ``shot_accuracy`` is defined only over RESOLVABLE shots (made + missed); it is
    None when no shot was judgeable (no HUB localization, no overlay) — accuracy is
    genuinely unknown then, never a fabricated 0.
    """
    robot_id: int
    alliance: Optional[Alliance] = None
    team_number: Optional[int] = None

    intakes: int = 0          # confirmed possessions gained
    outtakes: int = 0         # balls ejected (any outbound eject ≥ outtake speed)
    shots: int = 0            # outtakes fast enough to be a scoring attempt
    made: int = 0             # shots whose arc entered the HUB or overlay-confirmed
    missed: int = 0           # shots judged NOT to score (resolvable, no HUB/overlay)

    held_count: int = 0       # balls currently possessed (inferred)

    # running confidence book-keeping (for reporting; not fabricated)
    made_confidences: list[float] = field(default_factory=list)

    @property
    def resolvable_shots(self) -> int:
        """Shots we could actually judge (made + missed). Unresolvable shots are
        excluded from accuracy."""
        return self.made + self.missed

    @property
    def shot_accuracy(self) -> Optional[float]:
        """made / (made + missed). None when no shot was resolvable."""
        denom = self.made + self.missed
        return (self.made / denom) if denom > 0 else None

    @property
    def miss_rate(self) -> Optional[float]:
        acc = self.shot_accuracy
        return None if acc is None else (1.0 - acc)

    @property
    def mean_made_confidence(self) -> Optional[float]:
        return (sum(self.made_confidences) / len(self.made_confidences)
                if self.made_confidences else None)

    def to_dict(self) -> dict:
        """JSON-serializable per-robot record (every value primitive/None/list)."""
        return {
            "robot_id": self.robot_id,
            "alliance": self.alliance,
            "team_number": self.team_number,
            "intakes": self.intakes,
            "outtakes": self.outtakes,
            "shots": self.shots,
            "made": self.made,
            "missed": self.missed,
            "resolvable_shots": self.resolvable_shots,
            "shot_accuracy": self.shot_accuracy,
            "miss_rate": self.miss_rate,
            "held_count": self.held_count,
            "mean_made_confidence": self.mean_made_confidence,
        }


# ─────────────────────────────── per-ball lifecycle ───────────────────────────
@dataclass
class _BallState:
    ball_tid: int
    phase: str = "FREE"                 # FREE | INTAKE | POSSESSED
    target_robot: Optional[int] = None  # robot being approached / holding
    owner_robot: Optional[int] = None   # robot credited with possession
    last_dist_m: Optional[float] = None
    converging_frames: int = 0
    last_seen_frame: int = 0
    last_seen_t_s: float = 0.0


@dataclass
class _ShotState:
    """An outtake-classified-as-shot awaiting a HUB / overlay resolution."""
    flight_tid: int
    robot_id: int
    alliance: Optional[Alliance]
    start_frame: int
    start_t_s: float
    resolved: bool = False
    resolvable: bool = False            # could we ever judge it (pose or overlay)?


# ───────────────────────────────── tracker ────────────────────────────────────
class ScoringEventTracker:
    """Intake/Outtake/SCORED state machine -> per-robot tallies + canonical Events.

    Usage: build with a HUB localizer (any ``in_hub``-capable object, e.g.
    ``FieldModel``) and call ``step(frame_state, fps)`` once per FrameState. It
    returns the new ``Event`` list for that frame. Feed authoritative overlay score
    increments via ``note_overlay_score(frame_idx, alliance)`` (the fuse layer calls
    it when the count ticks) so a shot can be confirmed when geometry is inconclusive.

    Emitted Events use the canonical EventType vocabulary; the fine-grained kind is in
    ``Event.detail['kind']``:
      * ``pickup``  detail.kind = "intake"
      * ``shot``    detail.kind = "outtake"     (an eject that is NOT a scoring shot)
      * ``shot``    detail.kind = "shot"        (a scoring attempt)
      * ``shot``    detail.kind = "shot_miss"   (a resolvable miss; result="miss")
      * ``score``   detail.kind = "shot_made"   (HUB / overlay confirmed)
    Confidence on ``score``/miss events is the HUB-localization confidence and is
    never fabricated (0.0 / event suppressed when the outcome is unknowable).
    """

    def __init__(self, hub_localizer: Optional[HubLocalizer],
                 cfg: Optional[dict] = None, fps: float = 60.0):
        self.hub = hub_localizer
        self.fps = float(fps)
        c = dict(_DEFAULTS)
        c.update(cfg or {})
        self.cfg = c

        self.intake_k = float(c["intake_k"])
        self.intake_floor_m = float(c["intake_floor_m"])
        self.abandon_mult = float(c["abandon_radius_mult"])
        self.outtake_origin_m = float(c["outtake_origin_m"])
        self.outtake_min_speed = float(c["outtake_min_speed_mps"])
        self.shot_min_speed = float(c["shot_min_speed_mps"])
        self.shot_lookahead_s = float(c["shot_lookahead_s"])
        self.hub_tol_m = float(c["hub_tol_m"])
        self.default_hub_conf = float(c["default_hub_confidence"])
        self.overlay_conf = float(c["overlay_confidence"])

        self._recompute_windows()

        self.robots: dict[int, RobotScoring] = {}
        self._balls: dict[int, _BallState] = {}
        self._shots: list[_ShotState] = []
        self._known_flights: set[int] = set()
        self._last_intake_frame: dict[int, int] = {}
        self._score_pulses: list[tuple[int, Alliance]] = []

    def _recompute_windows(self) -> None:
        c = self.cfg
        self.converge_min_frames = seconds_to_frames(c["converge_min_s"], self.fps)
        self.vanish_grace_frames = seconds_to_frames(c["vanish_grace_s"], self.fps)
        self.intake_dedup_frames = seconds_to_frames(c["intake_dedup_s"], self.fps)
        self.score_window_frames = seconds_to_frames(c["score_window_s"], self.fps)
        self.score_lag_frames = seconds_to_frames(c["score_lag_s"], self.fps) if c["score_lag_s"] > 0 else 0
        self.shot_lookahead_frames = seconds_to_frames(c["shot_lookahead_s"], self.fps)

    # ───────────────────────── intake radius (scaled) ─────────────────────────
    @staticmethod
    def _robot_size_m(robot: Track) -> Optional[float]:
        """Larger projected bbox side in FIELD METERS, if the geometry layer attached
        ``field_wh_m``. Returns None when only a pixel bbox exists (a px size can't be
        compared against meter ball-distances, so the caller uses the meter floor)."""
        whm = getattr(robot, "field_wh_m", None)
        if whm and len(whm) == 2 and max(whm) > 0:
            return float(max(whm))
        return None

    def intake_radius(self, robot: Track) -> float:
        """Robot's intake catchment radius, SCALED TO ROBOT SIZE:

            max(intake_k * robot_size_m, intake_floor_m)

        Scaling is only applied when the size is in METERS (a projected ``field_wh_m``);
        otherwise we fall back to the meter floor (px can't be compared to meter
        distances), which also guards a degenerate zero-area bbox."""
        size_m = self._robot_size_m(robot)
        scaled = self.intake_k * size_m if size_m is not None else 0.0
        return max(scaled, self.intake_floor_m)

    # ───────────────────────────── helpers ───────────────────────────────────
    @staticmethod
    def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _speed_mps(track: Track) -> float:
        v = getattr(track, "vel_mps", (0.0, 0.0)) or (0.0, 0.0)
        return math.hypot(v[0], v[1])

    def _stat(self, robot: Track) -> RobotScoring:
        rs = self.robots.get(robot.tid)
        if rs is None:
            rs = RobotScoring(robot_id=robot.tid, alliance=robot.alliance)
            self.robots[robot.tid] = rs
        elif robot.alliance is not None:
            rs.alliance = robot.alliance
        return rs

    def _stat_by_id(self, rid: int, alliance: Optional[Alliance] = None) -> RobotScoring:
        rs = self.robots.get(rid)
        if rs is None:
            rs = RobotScoring(robot_id=rid, alliance=alliance)
            self.robots[rid] = rs
        elif alliance is not None and rs.alliance is None:
            rs.alliance = alliance
        return rs

    def _nearest_robot(self, xy: tuple[float, float], robots: list[Track]):
        best, best_d = None, math.inf
        for r in robots:
            if r.field_xy_m is None:
                continue
            d = self._dist(xy, r.field_xy_m)
            if d < best_d:
                best, best_d = r, d
        return best, best_d

    # ──────────────────── HUB localization availability / confidence ──────────
    def _have_pose(self, st: FrameState) -> bool:
        pose = getattr(st, "pose", None)
        return pose is not None and getattr(pose, "valid", False)

    def _hub_confidence(self, st: FrameState, alliance: Optional[Alliance]) -> float:
        """Confidence in a HUB-geometry resolution this frame. NEVER fabricated:

          * if the localizer exposes ``localization_confidence(alliance)`` or a
            ``hub_confidence`` attribute, use it (clamped to [0,1]);
          * else a VALID field pose -> the configured conservative default;
          * else 0.0 (we cannot trust the airborne ball's planar field_xy).
        """
        loc = self.hub
        if loc is not None:
            fn = getattr(loc, "localization_confidence", None)
            if callable(fn):
                try:
                    c = fn(alliance)
                    if c is not None:
                        return max(0.0, min(1.0, float(c)))
                except Exception:
                    pass
            attr = getattr(loc, "hub_confidence", None)
            if attr is not None:
                try:
                    return max(0.0, min(1.0, float(attr)))
                except (TypeError, ValueError):
                    pass
        return self.default_hub_conf if self._have_pose(st) else 0.0

    # ───────────────────────── overlay correlation ───────────────────────────
    def note_overlay_score(self, frame_idx: int, alliance: Alliance) -> None:
        """Record that the authoritative overlay count incremented for ``alliance`` at
        ``frame_idx``. A SHOT made falls back to this when the arc HUB-entry test is
        inconclusive, and any pulse in a shot's window makes that shot resolvable."""
        self._score_pulses.append((frame_idx, alliance))

    def _overlay_confirms(self, shot: _ShotState) -> bool:
        lo = shot.start_frame - 1 - self.score_lag_frames
        hi = shot.start_frame + self.score_window_frames
        return any(al == shot.alliance and lo <= pf <= hi
                   for pf, al in self._score_pulses)

    # ─────────────────────────────── step ────────────────────────────────────
    def step(self, st: FrameState, fps: Optional[float] = None) -> list[Event]:
        """Advance the state machine one frame; return this frame's new Events."""
        if fps is not None and float(fps) != self.fps:
            self.fps = float(fps)
            self._recompute_windows()

        events: list[Event] = []
        robots = [r for r in st.robots if r.field_xy_m is not None]
        for r in robots:
            self._stat(r)

        balls = [b for b in st.balls_inventory if b.field_xy_m is not None]
        events += self._update_balls(st, robots, balls)
        events += self._detect_outtakes(st, robots)
        events += self._resolve_shots(st)
        self._prune_pulses(st)
        return events

    # ───────────────────────────── intake ────────────────────────────────────
    def _update_balls(self, st: FrameState, robots: list[Track],
                      balls: list[Track]) -> list[Event]:
        events: list[Event] = []
        seen = {b.tid: b for b in balls}

        for tid, bs in list(self._balls.items()):
            if tid in seen:
                b = seen[tid]
                bs.last_seen_frame = st.frame_idx
                bs.last_seen_t_s = st.t_s
                if bs.phase in ("FREE", "INTAKE"):
                    events += self._advance_free_ball(st, bs, b, robots)
            else:
                gap = st.frame_idx - bs.last_seen_frame
                if bs.phase == "INTAKE":
                    # vanished inside the target robot's radius -> possession.
                    if gap >= self.vanish_grace_frames:
                        ev = self._confirm_intake(st, bs)
                        if ev is not None:
                            events.append(ev)
                elif bs.phase == "FREE":
                    if gap >= self.vanish_grace_frames:
                        del self._balls[tid]
                # POSSESSED balls are intentionally un-detected; they live until an outtake.

        for b in balls:
            if b.tid not in self._balls:
                self._balls[b.tid] = _BallState(
                    ball_tid=b.tid, phase="FREE",
                    last_seen_frame=st.frame_idx, last_seen_t_s=st.t_s)
        return events

    def _advance_free_ball(self, st: FrameState, bs: _BallState, b: Track,
                           robots: list[Track]) -> list[Event]:
        events: list[Event] = []
        nearest, d = self._nearest_robot(b.field_xy_m, robots)
        if nearest is None:
            bs.last_dist_m = None
            bs.converging_frames = 0
            return events

        radius = self.intake_radius(nearest)
        closing = bs.last_dist_m is not None and d < bs.last_dist_m
        bs.converging_frames = bs.converging_frames + 1 if closing else 0
        bs.last_dist_m = d

        if bs.phase == "FREE":
            if d <= radius and bs.converging_frames >= self.converge_min_frames:
                last = self._last_intake_frame.get(nearest.tid, -(10 ** 9))
                if st.frame_idx - last >= self.intake_dedup_frames:
                    bs.phase = "INTAKE"
                    bs.target_robot = nearest.tid
                    self._last_intake_frame[nearest.tid] = st.frame_idx
        elif bs.phase == "INTAKE":
            # the ball drifted well back out of the radius (and is not closing): abandon.
            if d > radius * self.abandon_mult and not closing:
                bs.phase = "FREE"
                bs.target_robot = None
        return events

    def _confirm_intake(self, st: FrameState, bs: _BallState) -> Optional[Event]:
        rid = bs.target_robot
        if rid is None:
            return None
        bs.phase = "POSSESSED"
        bs.owner_robot = rid
        rs = self._stat_by_id(rid)
        rs.intakes += 1
        rs.held_count += 1
        # geometry availability is what makes the field-space catchment meaningful.
        conf = self.default_hub_conf if self._have_pose(st) else self.default_hub_conf
        return Event(
            t_s=bs.last_seen_t_s, frame_idx=bs.last_seen_frame, type="pickup",
            alliance=rs.alliance, robot_id=rid, confidence=conf,
            detail={"kind": "intake", "ball_id": bs.ball_tid,
                    "held_count": rs.held_count})

    # ─────────────────────────── outtake / shot ──────────────────────────────
    def _detect_outtakes(self, st: FrameState, robots: list[Track]) -> list[Event]:
        """A possessed ball leaving == a NEW flight track born at a possessing robot,
        moving outbound above the outtake speed. We can't thread ball identity through
        the robot (see docstring), so we credit the nearest possessing robot within
        ``outtake_origin_m``. An eject ≥ shot speed is additionally a SHOT (scoring
        attempt); a slower eject is an outtake only."""
        events: list[Event] = []
        possessors = {bs.owner_robot for bs in self._balls.values()
                      if bs.phase == "POSSESSED" and bs.owner_robot is not None}
        if not possessors:
            self._known_flights |= {f.tid for f in st.balls_flight}
            return events

        for fl in st.balls_flight:
            if fl.tid in self._known_flights or fl.field_xy_m is None:
                continue
            self._known_flights.add(fl.tid)

            best_rid, best_d, best_robot = None, math.inf, None
            for r in robots:
                if r.tid not in possessors or r.field_xy_m is None:
                    continue
                d = self._dist(fl.field_xy_m, r.field_xy_m)
                if d < best_d:
                    best_rid, best_d, best_robot = r.tid, d, r
            if best_rid is None or best_d > self.outtake_origin_m:
                continue

            speed = self._speed_mps(fl)
            # outbound test: is the flight moving AWAY from the robot? (positive radial
            # velocity). When velocity is unavailable we treat a flight born at a
            # possessing robot as outbound (the ball left the robot's hands).
            outbound = self._is_outbound(fl, best_robot)
            if speed < self.outtake_min_speed or not outbound:
                # too slow / not leaving — not a real eject; ignore but mark known.
                continue

            rs = self._stat_by_id(best_rid, best_robot.alliance)
            rs.outtakes += 1
            self._consume_held(best_rid)

            if speed >= self.shot_min_speed:
                rs.shots += 1
                self._shots.append(_ShotState(
                    flight_tid=fl.tid, robot_id=best_rid, alliance=rs.alliance,
                    start_frame=st.frame_idx, start_t_s=st.t_s))
                events.append(Event(
                    t_s=st.t_s, frame_idx=st.frame_idx, type="shot",
                    alliance=rs.alliance, robot_id=best_rid, confidence=1.0,
                    detail={"kind": "shot", "flight_id": fl.tid,
                            "origin_dist_m": best_d, "speed_mps": speed}))
            else:
                events.append(Event(
                    t_s=st.t_s, frame_idx=st.frame_idx, type="shot",
                    alliance=rs.alliance, robot_id=best_rid, confidence=1.0,
                    detail={"kind": "outtake", "flight_id": fl.tid,
                            "origin_dist_m": best_d, "speed_mps": speed}))
        return events

    def _is_outbound(self, flight: Track, robot: Track) -> bool:
        """True if the flight is moving away from the robot (radial vel > 0). Falls
        back to True when velocity is unknown (a flight born at a possessor implies the
        ball left it)."""
        v = getattr(flight, "vel_mps", None)
        if not v or (v[0] == 0.0 and v[1] == 0.0):
            return True
        if flight.field_xy_m is None or robot.field_xy_m is None:
            return True
        rx = flight.field_xy_m[0] - robot.field_xy_m[0]
        ry = flight.field_xy_m[1] - robot.field_xy_m[1]
        # dot(radial_unit, velocity) > 0  <=>  moving outward
        return (rx * v[0] + ry * v[1]) >= 0.0

    def _consume_held(self, rid: int) -> None:
        """Decrement a robot's held-count and drop one possessed ball state."""
        rs = self.robots.get(rid)
        cand = [bs for bs in self._balls.values()
                if bs.phase == "POSSESSED" and bs.owner_robot == rid]
        if cand:
            # drop the most-recently possessed (LIFO is fine; identity is unobservable)
            bs = cand[-1]
            del self._balls[bs.ball_tid]
        if rs is not None and rs.held_count > 0:
            rs.held_count -= 1

    # ─────────────────────────── resolve SCORED ──────────────────────────────
    def _resolve_shots(self, st: FrameState) -> list[Event]:
        """A SHOT is MADE if its outbound arc enters the target HUB region (geometry),
        OR an overlay score increment lands in its window. It waits up to
        ``shot_lookahead_s`` then closes as a MISS.

        A shot counts toward made/missed (hence shot_accuracy) ONLY when judgeable:
        a valid pose so the arc test is trustworthy, or an overlay pulse to correlate
        against. Otherwise the outcome is UNKNOWN — no SCORED/MISS event, excluded from
        accuracy, never fabricated."""
        events: list[Event] = []
        flights = {f.tid: f for f in st.balls_flight}
        have_pose = self._have_pose(st)
        still: list[_ShotState] = []

        for sh in self._shots:
            if sh.resolved:
                continue
            fl = flights.get(sh.flight_tid)
            geom_testable = (fl is not None and fl.field_xy_m is not None
                             and sh.alliance is not None and self.hub is not None)
            entered = False
            if geom_testable:
                try:
                    entered = bool(self.hub.in_hub(fl.field_xy_m, sh.alliance,
                                                   tol_m=self.hub_tol_m))
                except Exception:
                    entered = False

            overlay_ok = self._overlay_confirms(sh)

            # judgeability: a valid pose makes the (geom-testable) arc trustworthy; an
            # overlay pulse in-window makes it judgeable regardless of pose.
            if (have_pose and geom_testable) or overlay_ok:
                sh.resolvable = True

            timed_out = (st.frame_idx - sh.start_frame) >= self.shot_lookahead_frames
            lost = fl is None

            if (entered and have_pose) or overlay_ok:
                sh.resolved = True
                sh.resolvable = True
                rs = self._stat_by_id(sh.robot_id, sh.alliance)
                rs.made += 1
                via = "hub" if (entered and have_pose) else "overlay"
                conf = (self._hub_confidence(st, sh.alliance) if via == "hub"
                        else self.overlay_conf)
                rs.made_confidences.append(conf)
                events.append(Event(
                    t_s=st.t_s, frame_idx=st.frame_idx, type="score",
                    alliance=sh.alliance, robot_id=sh.robot_id, confidence=conf,
                    detail={"kind": "shot_made", "flight_id": sh.flight_tid,
                            "via": via}))
            elif timed_out or lost:
                sh.resolved = True
                if sh.resolvable:
                    rs = self._stat_by_id(sh.robot_id, sh.alliance)
                    rs.missed += 1
                    # a miss is judged: confidence in the judgement (we had localization)
                    conf = (self._hub_confidence(st, sh.alliance) if have_pose
                            else self.overlay_conf)
                    events.append(Event(
                        t_s=st.t_s, frame_idx=st.frame_idx, type="shot",
                        alliance=sh.alliance, robot_id=sh.robot_id, confidence=conf,
                        detail={"kind": "shot_miss", "flight_id": sh.flight_tid,
                                "result": "miss", "resolvable": True}))
                # if not resolvable: silently drop — outcome unknown, never fabricated.
            else:
                still.append(sh)

        self._shots = still
        return events

    def _prune_pulses(self, st: FrameState) -> None:
        cutoff = st.frame_idx - self.shot_lookahead_frames - self.score_window_frames
        self._score_pulses = [(f, a) for (f, a) in self._score_pulses if f >= cutoff]

    # ───────────────────────────── reporting ─────────────────────────────────
    def tallies(self) -> dict:
        """Per-robot scoring tallies keyed by track id. JSON-serializable; every
        ``shot_accuracy``/``miss_rate`` is None when not yet resolvable (honest)."""
        return {rid: rs.to_dict() for rid, rs in self.robots.items()}
