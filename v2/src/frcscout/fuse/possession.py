"""Per-robot ball-possession state machine + scouting metrics (field coords).

This is the scouting brain: it watches the field-projected ball tracks and robot
tracks frame-by-frame and infers, per ball, a possession lifecycle:

    FREE --(ball trajectory converges on a robot intake RADIUS)--> INTAKE_ATTEMPT(robot)
        --(ball vanishes inside that robot's radius)--> POSSESSED(robot)  [intake SUCCESS]
        --(ball drifts away / reappears free)-------> FREE               [intake FAIL]

    POSSESSED(robot) --(a flight track is born at that robot)--> SHOT_ATTEMPT(robot)
        --(arc enters the target HUB volume, and/or correlates with an
           overlay score increment in a window)-----------------> SCORED  [shot SUCCESS]
        --(neither)--------------------------------------------> MISSED  [shot FAIL]

and accumulates per-robot scouting counters: intake_attempts / intake_successes
(+ rate + mean time_to_intake), shot_attempts / shots_made (+ accuracy),
possession_time, cycle_time, and a per-frame ``balls_in_radius`` opportunity
measure (peak retained).

────────────────────────── intake RADIUS scales to ROBOT SIZE ─────────────────
A robot can only realistically intake balls that are physically near it, and a
big robot reaches further than a small one. So the intake catchment is NOT a
fixed pixel/meter constant — it is

    intake_radius(robot) = INTAKE_K * robot_size

where ``robot_size`` = max(bbox_w, bbox_h) of the robot, measured in FIELD METERS
when a pose has projected the bbox corners (preferred) and falling back to image
PIXELS otherwise. ``INTAKE_K`` comes from config (``intake_k``, default ~1.2).
Because the radius derives from the *observed* robot extent it self-calibrates to
whatever units the geometry layer is producing — there is no hardcoded scale.
The legacy fixed ``intake_zone_m`` still acts as a floor so the radius never
collapses to zero for a degenerate/zero-area bbox.

All HUB geometry is in FIELD METERS via the geometry layer (``field.in_hub``).
All timing is in SECONDS; the caller passes the real fps and this module converts
to frames via ``seconds_to_frames`` — it never hardcodes a frame count.

────────────────────────────── honest caveats ───────────────────────────────
The hard, *inferential* parts (flagged inline):

* Ball identity through possession is UNOBSERVABLE. Once a ball enters a robot the
  detector loses it; when a ball later launches we ASSUME it is "a" ball the robot
  held (held_count > 0) rather than literally the same physical ball. We therefore
  track per-robot a held_count, not a ball<->ball thread.

* "Made" requires HUB LOCALIZATION (geometry). ``shots_made`` / ``shot_accuracy``
  are only meaningful when we have a valid field pose so the flight arc can be
  tested against ``field.in_hub`` (or an authoritative overlay increment confirms
  the shot). With no geometry and no overlay we DO NOT fabricate a make:
  ``shot_accuracy`` is reported as ``None`` (unknown), never 0 or a guess.

* Concurrent scoring is ambiguous. SHOT made prefers the geometric HUB-entry test,
  but the broadcast overlay count is the AUTHORITATIVE score. When the arc test is
  inconclusive we fall back to correlating the shot with an overlay score increment
  inside a window; if several robots' shots land in the same window the per-robot
  credit is PROBABILISTIC and should be resolved downstream by the Attributor. We
  expose ``note_overlay_score`` and emit shot_attempt/shot_success events the fuse
  layer reconciles against the overlay; we never fabricate an integer split here.

* Without real footage the radius factor, converge window and "vanished" grace are
  config defaults; they need calibration on actual matches (TODO).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from frcscout.io.config import seconds_to_frames
from frcscout.io.types import Alliance, Event, FrameState, Track


# ─────────────────────────────── defaults ────────────────────────────────────
_DEFAULTS = {
    # intake catchment RADIUS = intake_k * robot_size (robot_size = max bbox side
    # in field meters if projected, else px). The catchment scales to robot size.
    "intake_k": 1.2,
    # floor for the scaled radius (m or px, matching robot_size units) so a
    # degenerate / zero-area bbox can't collapse the catchment to zero. Also the
    # legacy fixed intake zone, kept for back-compat with events.yaml.
    "intake_zone_m": 0.6,
    # the ball must be approaching (distance decreasing) for at least this long.
    "converge_min_s": 0.1,
    # after a ball disappears, how long (s) we keep looking for it before we
    # decide it really vanished (vs. a one-frame detector miss).
    "vanish_grace_s": 0.4,
    # a shot's flight track must originate within this radius (m) of the robot
    # that holds a ball to be credited as that robot's shot.
    "shot_origin_m": 1.0,
    # how far ahead (s) to test the flight arc for HUB entry.
    "shot_lookahead_s": 4.0,
    # HUB-entry tolerance (m) added to the hub radius for the in_hub test.
    "hub_tol_m": 0.3,
    # window (s) within which an overlay score increment confirms a shot.
    "score_window_s": 1.5,
    # min cycle (s) between two intakes credited to one robot (dedup).
    "intake_dedup_s": 1.0,
}


def _bbox_size(t: Track) -> Optional[float]:
    """max(width, height) of a track's bbox, in whatever units ``xyxy`` carries
    (image px). Returns None for a degenerate / missing box."""
    box = getattr(t, "xyxy", None)
    if not box or len(box) != 4:
        return None
    x0, y0, x1, y1 = box
    w, h = abs(x1 - x0), abs(y1 - y0)
    s = max(w, h)
    return s if s > 0 else None


def robot_size(robot: Track) -> tuple[Optional[float], str]:
    """The robot's characteristic extent = max(bbox_w, bbox_h), with its units.

    Returns ``(size, units)`` where units is "m" or "px":
      1. field meters — if the geometry layer recorded the projected bbox extent on
         the track (``field_wh_m`` attribute), use the larger side -> ("m");
      2. else the image-pixel bbox size from ``xyxy`` -> ("px").
    Returns ``(None, "none")`` when neither is available.

    IMPORTANT (units): the intake radius is compared against ball distances that
    live in FIELD METERS. Scaling is only physically meaningful when ``robot_size``
    is itself in meters. A pixel bbox cannot be compared to meter distances, so the
    caller does NOT scale off pixels — it uses the configured meter floor instead.
    The pixel path is still surfaced (for diagnostics) but flagged "px".
    """
    # 1) projected field-meter extent, if the geometry layer attached one.
    whm = getattr(robot, "field_wh_m", None)
    if whm and len(whm) == 2 and max(whm) > 0:
        return float(max(whm)), "m"
    # 2) pixel bbox extent.
    px = _bbox_size(robot)
    if px is not None:
        return px, "px"
    return None, "none"


@dataclass
class RobotStats:
    """Per-robot scouting counters.

    Times are SECONDS; rates are dimensionless. ``shots_made`` / ``shot_accuracy``
    require HUB localization (see module docstring) — when no shot has ever been
    resolvable (no pose, no overlay) ``shot_accuracy`` is reported as None, never a
    fabricated number.
    """
    robot_id: int
    alliance: Optional[Alliance] = None
    team_number: Optional[int] = None

    # ── intake ──
    intake_attempts: int = 0
    intake_successes: int = 0
    time_to_intake_s: list[float] = field(default_factory=list)  # entered-radius -> possessed

    # ── shots ──
    shot_attempts: int = 0
    shots_made: int = 0                 # arc entered HUB OR overlay-confirmed
    shots_resolvable: int = 0           # shots we could actually judge (pose/overlay)

    # ── possession / opportunity ──
    held_count: int = 0                 # balls currently held (inferred)
    possession_time_s: float = 0.0      # cumulative seconds holding >=1 ball
    cycle_times_s: list[float] = field(default_factory=list)  # intake->shot gaps
    balls_in_radius_peak: int = 0       # max simultaneous balls inside intake_radius

    # ── back-compat aliases (events.py / state.py read shot_success*) ──
    @property
    def shot_successes(self) -> int:
        """Alias for shots_made — preserves the historical field name used by the
        fuse layer / match record."""
        return self.shots_made

    @property
    def intake_success_rate(self) -> float:
        return self.intake_successes / self.intake_attempts if self.intake_attempts else 0.0

    @property
    def mean_time_to_intake_s(self) -> Optional[float]:
        return (sum(self.time_to_intake_s) / len(self.time_to_intake_s)
                if self.time_to_intake_s else None)

    @property
    def shot_accuracy(self) -> Optional[float]:
        """made / attempts, but ONLY over shots we could localize (pose or overlay).
        None when no shot was resolvable — accuracy is genuinely unknown, not 0."""
        if self.shots_resolvable == 0:
            return None
        return self.shots_made / self.shots_resolvable

    @property
    def miss_rate(self) -> Optional[float]:
        """Fraction of resolvable shots that did NOT enter the HUB (§5.2 miss rate).
        None when no shot was resolvable."""
        acc = self.shot_accuracy
        return None if acc is None else (1.0 - acc)

    @property
    def shot_success_rate(self) -> float:
        """Back-compat: made / attempts over ALL attempts (0 when none). Prefer
        ``shot_accuracy`` (which excludes unresolvable shots)."""
        return self.shots_made / self.shot_attempts if self.shot_attempts else 0.0

    @property
    def avg_cycle_time_s(self) -> Optional[float]:
        return sum(self.cycle_times_s) / len(self.cycle_times_s) if self.cycle_times_s else None

    def to_dict(self) -> dict:
        return {
            "robot_id": self.robot_id,
            "alliance": self.alliance,
            "team_number": self.team_number,
            # intake
            "intake_attempts": self.intake_attempts,
            "intake_successes": self.intake_successes,
            "intake_success_rate": self.intake_success_rate,
            "mean_time_to_intake_s": self.mean_time_to_intake_s,
            "time_to_intake_s": list(self.time_to_intake_s),
            # shots
            "shot_attempts": self.shot_attempts,
            "shots_made": self.shots_made,
            "shots_resolvable": self.shots_resolvable,
            "shot_accuracy": self.shot_accuracy,
            "miss_rate": self.miss_rate,
            # back-compat names
            "shot_successes": self.shot_successes,
            "shot_success_rate": self.shot_success_rate,
            # possession / opportunity
            "held_count": self.held_count,
            "possession_time_s": self.possession_time_s,
            "balls_in_radius_peak": self.balls_in_radius_peak,
            "avg_cycle_time_s": self.avg_cycle_time_s,
            "cycle_times_s": list(self.cycle_times_s),
        }


# ─────────────────────────── per-ball lifecycle ──────────────────────────────
@dataclass
class _BallState:
    """Tracks one ball's possession lifecycle across frames."""
    ball_tid: int
    phase: str = "FREE"                 # FREE | INTAKE_ATTEMPT | POSSESSED
    target_robot: Optional[int] = None  # robot being approached / holding
    intake_robot: Optional[int] = None  # robot credited with possession
    last_field_xy: Optional[tuple[float, float]] = None
    last_dist_m: Optional[float] = None     # distance to target_robot last frame
    converging_frames: int = 0
    enter_radius_t_s: Optional[float] = None  # when the ball first entered the radius
    attempt_t_s: Optional[float] = None     # when the intake attempt began
    possess_t_s: Optional[float] = None     # when possession was confirmed
    last_seen_frame: int = 0
    last_seen_t_s: float = 0.0


@dataclass
class _ShotState:
    """An in-flight shot awaiting a HUB-entry / overlay-score resolution."""
    flight_tid: int
    robot_id: int
    alliance: Optional[Alliance]
    start_t_s: float
    start_frame: int
    resolved: bool = False
    entered_hub: bool = False
    # Could this shot ever be judged? True once we have geometry (pose so the arc
    # can be tested vs the HUB) or an overlay pulse to correlate against.
    resolvable: bool = False


class PossessionTracker:
    """Per-ball possession state machine -> per-robot scouting metrics + events.

    Usage: call ``step(state, fps)`` once per FrameState. It returns the list of
    new Events (intake_attempt / intake_success / shot_attempt / shot_success) for
    that frame. The overlay-correlation path is fed by ``note_overlay_score`` which
    the fuse layer calls when the authoritative count increments.

    Intake catchment is per-robot and SCALED TO ROBOT SIZE — see ``intake_radius``.
    """

    def __init__(self, field, cfg: Optional[dict] = None, fps: float = 60.0):
        self.field = field
        self.fps = float(fps)
        c = dict(_DEFAULTS)
        c.update(cfg or {})
        self.cfg = c

        self.intake_k = float(c["intake_k"])
        self.intake_zone_m = float(c["intake_zone_m"])   # floor + legacy fixed zone
        self.shot_origin_m = float(c["shot_origin_m"])
        self.hub_tol_m = float(c["hub_tol_m"])
        self.shot_lookahead_s = float(c["shot_lookahead_s"])

        self.converge_min_frames = seconds_to_frames(c["converge_min_s"], self.fps)
        self.vanish_grace_frames = seconds_to_frames(c["vanish_grace_s"], self.fps)
        self.score_window_frames = seconds_to_frames(c["score_window_s"], self.fps)
        self.intake_dedup_frames = seconds_to_frames(c["intake_dedup_s"], self.fps)

        self.robots: dict[int, RobotStats] = {}
        self._balls: dict[int, _BallState] = {}
        self._shots: list[_ShotState] = []
        self._known_flights: set[int] = set()
        self._last_intake_frame: dict[int, int] = {}
        # recent overlay score increments: list of (frame_idx, alliance)
        self._score_pulses: list[tuple[int, Alliance]] = []

    # ───────────────────────── intake radius (scaled) ─────────────────────────
    def intake_radius(self, robot: Track) -> float:
        """The robot's intake catchment radius, SCALED TO ROBOT SIZE.

            intake_radius = max(intake_k * robot_size_m, intake_zone_m floor)

        ``robot_size_m`` is the larger bbox side in FIELD METERS when the geometry
        layer projected it (``robot.field_wh_m``). A robot can only realistically
        intake balls within this catchment, so ``balls_in_radius`` uses it to
        normalize intake opportunity, and a bigger robot reaches further.

        Units guard: ball distances are in field meters, so scaling is only applied
        when ``robot_size`` is itself in meters. When the only size we have is a
        pixel bbox (no pose projection), scaling px against meter distances would be
        nonsense, so we fall back to the configured meter floor (``intake_zone_m``,
        the legacy fixed zone). The floor also protects against a degenerate
        zero-area bbox collapsing the radius to zero."""
        size, units = robot_size(robot)
        scaled = self.intake_k * size if (size is not None and units == "m") else 0.0
        return max(scaled, self.intake_zone_m)

    # ───────────────────────────── helpers ───────────────────────────────────
    def _stat(self, robot: Track) -> RobotStats:
        rs = self.robots.get(robot.tid)
        if rs is None:
            rs = RobotStats(robot_id=robot.tid, alliance=robot.alliance)
            self.robots[robot.tid] = rs
        elif robot.alliance is not None:
            rs.alliance = robot.alliance
        return rs

    @staticmethod
    def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _nearest_robot(self, xy: tuple[float, float], robots: list[Track]):
        """Nearest robot whose intake radius the point is *closest to entering* —
        we return the robot with the smallest distance and that distance (the caller
        compares it against that robot's own scaled intake_radius)."""
        best, best_d = None, math.inf
        for r in robots:
            if r.field_xy_m is None:
                continue
            d = self._dist(xy, r.field_xy_m)
            if d < best_d:
                best, best_d = r, d
        return best, best_d

    # ───────────────────────── overlay correlation ───────────────────────────
    def note_overlay_score(self, frame_idx: int, alliance: Alliance) -> None:
        """Record that the authoritative overlay count incremented for ``alliance``
        at ``frame_idx``. SHOT made falls back to this when the arc HUB-entry test
        is inconclusive (see module docstring). A pulse also makes outstanding shots
        of that alliance *resolvable* (we now have something to judge them by)."""
        self._score_pulses.append((frame_idx, alliance))

    def _overlay_confirms(self, shot: _ShotState, frame_idx: int) -> bool:
        lo = shot.start_frame - 1
        hi = shot.start_frame + self.score_window_frames
        for pf, al in self._score_pulses:
            if al == shot.alliance and lo <= pf <= hi:
                return True
        return False

    def _overlay_in_window(self, shot: _ShotState) -> bool:
        """Is there ANY overlay pulse for this alliance within the score window?
        Used to mark a shot resolvable even when it didn't confirm (e.g. a miss in a
        window where a *different* robot scored is still judgeable)."""
        lo = shot.start_frame - 1
        hi = shot.start_frame + self.score_window_frames
        return any(al == shot.alliance and lo <= pf <= hi
                   for pf, al in self._score_pulses)

    # ─────────────────────────────── step ────────────────────────────────────
    def step(self, st: FrameState, fps: Optional[float] = None) -> list[Event]:
        """Advance the state machine one frame. Returns this frame's new events."""
        if fps is not None and float(fps) != self.fps:
            # allow the caller to pass the real fps; reconverts windows
            self.fps = float(fps)
            self.converge_min_frames = seconds_to_frames(self.cfg["converge_min_s"], self.fps)
            self.vanish_grace_frames = seconds_to_frames(self.cfg["vanish_grace_s"], self.fps)
            self.score_window_frames = seconds_to_frames(self.cfg["score_window_s"], self.fps)
            self.intake_dedup_frames = seconds_to_frames(self.cfg["intake_dedup_s"], self.fps)
        events: list[Event] = []
        robots = [r for r in st.robots if r.field_xy_m is not None]
        balls = [b for b in st.balls_inventory if b.field_xy_m is not None]
        ball_ids = {b.tid for b in balls}

        # ensure stats exist for visible robots; accrue possession time
        for r in robots:
            self._stat(r)
        held_robots = {bs.intake_robot for bs in self._balls.values()
                       if bs.phase == "POSSESSED" and bs.intake_robot is not None}
        for rid in held_robots:
            if rid in self.robots:
                self.robots[rid].possession_time_s += 1.0 / self.fps

        # 0) opportunity measure: balls inside each robot's scaled intake radius
        self._update_balls_in_radius(robots, balls)

        # 1) update / advance each tracked free or attempting ball
        events += self._update_balls(st, robots, balls, ball_ids)

        # 2) detect shots: new flight tracks originating at a possessing robot
        events += self._detect_shots(st, robots)

        # 3) resolve outstanding shots (HUB entry along the arc, or overlay)
        events += self._resolve_shots(st)

        return events

    # ─────────────────────── balls-in-radius opportunity ──────────────────────
    def balls_in_radius(self, robot: Track, balls: list[Track]) -> int:
        """Number of (free/inventory) balls whose centre lies within this robot's
        SCALED intake_radius this frame. This normalizes intake rate by opportunity
        — a robot can only intake what is near it."""
        if robot.field_xy_m is None:
            return 0
        r = self.intake_radius(robot)
        rxy = robot.field_xy_m
        return sum(1 for b in balls
                   if b.field_xy_m is not None and self._dist(b.field_xy_m, rxy) <= r)

    def _update_balls_in_radius(self, robots: list[Track], balls: list[Track]) -> None:
        for r in robots:
            n = self.balls_in_radius(r, balls)
            rs = self._stat(r)
            if n > rs.balls_in_radius_peak:
                rs.balls_in_radius_peak = n

    # ───────────────────────────── balls ─────────────────────────────────────
    def _update_balls(self, st: FrameState, robots: list[Track],
                      balls: list[Track], ball_ids: set[int]) -> list[Event]:
        events: list[Event] = []
        seen = {b.tid: b for b in balls}

        # advance existing ball states
        for tid, bs in list(self._balls.items()):
            if tid in seen:
                b = seen[tid]
                bs.last_seen_frame = st.frame_idx
                bs.last_seen_t_s = st.t_s
                if bs.phase in ("FREE", "INTAKE_ATTEMPT"):
                    ev = self._advance_free_ball(st, bs, b, robots)
                    events += ev
                bs.last_field_xy = b.field_xy_m
            else:
                # ball not detected this frame — disappearance bookkeeping
                gap = st.frame_idx - bs.last_seen_frame
                if bs.phase == "INTAKE_ATTEMPT" and gap >= 0:
                    # vanished inside (or near) the target robot's radius -> SUCCESS.
                    # The ball entry is KEPT (now phase POSSESSED) so the robot
                    # counts as a possessor until it shoots.
                    if gap >= self.vanish_grace_frames:
                        ev = self._confirm_intake(st, bs)
                        if ev:
                            events.append(ev)
                elif bs.phase == "FREE":
                    if gap >= self.vanish_grace_frames:
                        del self._balls[tid]
                # POSSESSED balls have no detection; they live until a shot.

        # register newly-seen balls as FREE
        for b in balls:
            if b.tid not in self._balls:
                self._balls[b.tid] = _BallState(
                    ball_tid=b.tid, phase="FREE", last_field_xy=b.field_xy_m,
                    last_seen_frame=st.frame_idx, last_seen_t_s=st.t_s)

        return events

    def _advance_free_ball(self, st: FrameState, bs: _BallState,
                           b: Track, robots: list[Track]) -> list[Event]:
        events: list[Event] = []
        nearest, d = self._nearest_robot(b.field_xy_m, robots)
        if nearest is None:
            bs.last_dist_m = None
            bs.converging_frames = 0
            bs.enter_radius_t_s = None
            return events

        radius = self.intake_radius(nearest)

        # time_to_intake clock starts the moment the ball is INSIDE the radius.
        if d <= radius:
            if bs.enter_radius_t_s is None:
                bs.enter_radius_t_s = st.t_s
        else:
            bs.enter_radius_t_s = None

        closing = bs.last_dist_m is not None and d < bs.last_dist_m
        if closing:
            bs.converging_frames += 1
        else:
            bs.converging_frames = 0
        bs.last_dist_m = d

        if bs.phase == "FREE":
            # INTAKE_ATTEMPT: a free ball enters the robot's (scaled) intake radius
            # while CLOSING (approaching) for long enough.
            if (d <= radius
                    and bs.converging_frames >= self.converge_min_frames):
                last = self._last_intake_frame.get(nearest.tid, -10**9)
                if st.frame_idx - last >= self.intake_dedup_frames:
                    bs.phase = "INTAKE_ATTEMPT"
                    bs.target_robot = nearest.tid
                    bs.attempt_t_s = st.t_s
                    if bs.enter_radius_t_s is None:
                        bs.enter_radius_t_s = st.t_s
                    self._last_intake_frame[nearest.tid] = st.frame_idx
                    rs = self._stat(nearest)
                    rs.intake_attempts += 1
                    events.append(self._mk(st, "intake_attempt", nearest,
                                           {"ball_id": b.tid, "dist_m": d,
                                            "intake_radius_m": radius}))
        elif bs.phase == "INTAKE_ATTEMPT":
            # if the ball drifts well back out of the radius, the attempt FAILED.
            if d > radius * 2.0 and not closing:
                bs.phase = "FREE"
                bs.target_robot = None
                bs.attempt_t_s = None
                bs.enter_radius_t_s = None
        return events

    def _confirm_intake(self, st: FrameState, bs: _BallState) -> Optional[Event]:
        """The ball vanished inside the target robot's radius -> possession."""
        rid = bs.target_robot
        if rid is None:
            return None
        bs.phase = "POSSESSED"
        bs.intake_robot = rid
        bs.possess_t_s = bs.last_seen_t_s
        rs = self.robots.get(rid)
        if rs is None:
            rs = RobotStats(robot_id=rid)
            self.robots[rid] = rs
        rs.intake_successes += 1
        rs.held_count += 1
        # time_to_intake: from entering the radius to possession (seconds).
        if bs.enter_radius_t_s is not None:
            rs.time_to_intake_s.append(max(0.0, bs.last_seen_t_s - bs.enter_radius_t_s))
        return Event(
            t_s=bs.last_seen_t_s, frame_idx=bs.last_seen_frame, type="intake_success",  # type: ignore[arg-type]
            alliance=rs.alliance, robot_id=rid,
            detail={"ball_id": bs.ball_tid, "held_count": rs.held_count,
                    "time_to_intake_s": (rs.time_to_intake_s[-1]
                                         if rs.time_to_intake_s else None)})

    # ───────────────────────────── shots ─────────────────────────────────────
    def _detect_shots(self, st: FrameState, robots: list[Track]) -> list[Event]:
        """A possessed ball launching == a NEW flight track born at a robot that
        holds >=1 ball. We can't thread ball identity through the robot (see
        docstring), so we credit the nearest possessing robot within
        ``shot_origin_m``."""
        events: list[Event] = []
        possessors = {bs.intake_robot for bs in self._balls.values()
                      if bs.phase == "POSSESSED" and bs.intake_robot is not None}
        if not possessors:
            self._known_flights |= {f.tid for f in st.balls_flight}
            return events

        for fl in st.balls_flight:
            if fl.tid in self._known_flights or fl.field_xy_m is None:
                continue
            self._known_flights.add(fl.tid)
            # find the closest possessing robot to the flight origin
            best_rid, best_d = None, math.inf
            for r in robots:
                if r.tid not in possessors or r.field_xy_m is None:
                    continue
                d = self._dist(fl.field_xy_m, r.field_xy_m)
                if d < best_d:
                    best_rid, best_d = r.tid, d
            if best_rid is None or best_d > self.shot_origin_m:
                continue
            rs = self.robots[best_rid]
            rs.shot_attempts += 1
            # consume one held ball + record cycle time from its intake
            self._consume_held(best_rid, st)
            self._shots.append(_ShotState(
                flight_tid=fl.tid, robot_id=best_rid, alliance=rs.alliance,
                start_t_s=st.t_s, start_frame=st.frame_idx))
            events.append(self._mk(st, "shot_attempt", None,
                                   {"flight_id": fl.tid, "origin_dist_m": best_d},
                                   robot_id=best_rid, alliance=rs.alliance))
        return events

    def _consume_held(self, rid: int, st: FrameState) -> None:
        """Decrement a robot's held-count for a shot and log the cycle time from the
        most-recent possession this robot established (pickup -> shot, §5.2)."""
        rs = self.robots.get(rid)
        if rs is None:
            return
        # find the possessed ball held by this robot with the latest possess time
        cand = [bs for bs in self._balls.values()
                if bs.phase == "POSSESSED" and bs.intake_robot == rid]
        if cand:
            bs = max(cand, key=lambda b: b.possess_t_s or 0.0)
            if bs.attempt_t_s is not None:
                rs.cycle_times_s.append(max(0.0, st.t_s - bs.attempt_t_s))
            del self._balls[bs.ball_tid]
        if rs.held_count > 0:
            rs.held_count -= 1

    def _resolve_shots(self, st: FrameState) -> list[Event]:
        """A shot is MADE if its flight arc enters the target HUB volume, OR (when
        geometry is inconclusive) an overlay score increment lands in the window.
        Resolution waits up to ``shot_lookahead_s`` then closes as MISSED.

        A shot is only counted toward ``shots_resolvable`` (hence shot_accuracy /
        miss_rate) when we could actually judge it: a valid field pose so the arc
        can be tested vs the HUB, or an overlay pulse to correlate against. Without
        either, the shot's outcome is UNKNOWN and excluded from accuracy — we never
        fabricate a make or a miss."""
        events: list[Event] = []
        flights = {f.tid: f for f in st.balls_flight}
        lookahead = seconds_to_frames(self.shot_lookahead_s, self.fps)
        have_pose = getattr(st, "pose", None) is not None and getattr(st.pose, "valid", False)
        still: list[_ShotState] = []
        for sh in self._shots:
            if sh.resolved:
                continue
            # geometric: does the live flight position fall inside the alliance HUB?
            fl = flights.get(sh.flight_tid)
            entered = False
            geom_testable = (fl is not None and fl.field_xy_m is not None
                             and sh.alliance is not None)
            if geom_testable:
                if self.field.in_hub(fl.field_xy_m, sh.alliance, tol_m=self.hub_tol_m):
                    entered = True
            overlay_ok = self._overlay_confirms(sh, st.frame_idx)
            confirmed = entered or overlay_ok

            # A shot becomes judgeable once we can trust the arc test (a VALID field
            # pose so the airborne ball's field_xy is meaningful — see field.py on
            # z!=0 parallax) OR an overlay pulse exists in its window. Without a pose
            # the planar fix of a high ball is unreliable, so geometry alone (with no
            # pose) does NOT make a shot resolvable; the outcome stays unknown.
            if have_pose and geom_testable:
                sh.resolvable = True
            if self._overlay_in_window(sh):
                sh.resolvable = True

            timed_out = (st.frame_idx - sh.start_frame) >= lookahead
            lost = fl is None  # flight track ended (ball landed / left FOV)

            if confirmed:
                sh.resolved = True
                sh.resolvable = True
                rs = self.robots.get(sh.robot_id)
                if rs:
                    rs.shots_resolvable += 1
                    rs.shots_made += 1
                events.append(self._mk(
                    st, "shot_success", None,
                    {"flight_id": sh.flight_tid, "via": "hub" if entered else "overlay"},
                    robot_id=sh.robot_id, alliance=sh.alliance))
            elif timed_out or lost:
                sh.resolved = True   # MISS — no success event, but attempt stands
                rs = self.robots.get(sh.robot_id)
                if rs and sh.resolvable:
                    rs.shots_resolvable += 1   # a judged miss counts in accuracy
                events.append(self._mk(
                    st, "shot_attempt", None,
                    {"flight_id": sh.flight_tid, "result": "miss",
                     "resolvable": sh.resolvable},
                    robot_id=sh.robot_id, alliance=sh.alliance, miss=True))
            else:
                still.append(sh)
        self._shots = still
        # prune old score pulses
        cutoff = st.frame_idx - lookahead - self.score_window_frames
        self._score_pulses = [(f, a) for (f, a) in self._score_pulses if f >= cutoff]
        return events

    # ───────────────────────────── event factory ─────────────────────────────
    def _mk(self, st: FrameState, etype: str, robot: Optional[Track],
            detail: dict, robot_id: Optional[int] = None,
            alliance: Optional[Alliance] = None, miss: bool = False) -> Event:
        if robot is not None:
            robot_id = robot.tid
            alliance = robot.alliance
        if miss:
            detail = {**detail, "result": "miss"}
        return Event(
            t_s=st.t_s, frame_idx=st.frame_idx, type=etype,  # type: ignore[arg-type]
            alliance=alliance, robot_id=robot_id,
            confidence=1.0, detail=detail)

    # ───────────────────────────── reporting ─────────────────────────────────
    def stats_dict(self) -> dict:
        """Per-robot scouting record keyed by track_id. JSON-serializable: every
        value is an int/float/str/None/list of those. ``shot_accuracy`` /
        ``miss_rate`` / ``mean_time_to_intake_s`` are None when not yet computable."""
        return {rid: rs.to_dict() for rid, rs in self.robots.items()}
