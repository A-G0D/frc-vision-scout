"""Per-robot PHASE state machine — the *activity classifier* (refine layer).

While ``sixbot`` answers *which* robot is where and ``scoring_events`` answers *what
ball-handling happened*, this module answers the scout-facing question the demo video
makes obvious to a human eye: **what is each robot DOING right now?** Every frame it
labels each of the (up to six) robots with exactly one PHASE and accumulates the
scouting METRICS the user asked for.

PHASES and their detection signatures (from REBUILT-2026-game-model.md)
-----------------------------------------------------------------------
The label is chosen by a fixed PRIORITY order (an action that is happening *now*
out-ranks a steady state). Highest first:

  SCORED      a shot this robot owns was just CONFIRMED into a HUB (scoring_events
              emitted a ``score`` for it, OR the overlay fuel count ticked up in the
              shot's window). A brief, latched pulse — it decays back to the
              underlying state after ``scored_hold_s``.
  SHOOTING    a fast OUTBOUND ball track was just born at this robot, aimed at a HUB
              (scoring_events emitted a ``shot`` for it). Latched for ``shot_hold_s``
              so the renderer can actually show it (one-frame events flicker).
  CLIMBING    ENDGAME phase AND the robot is adjacent to its own TOWER AND its bumper
              height is rising past the rung heights -> latch L1 / L2 / L3. The level
              only ever ratchets UP and never un-latches (a climb is monotonic).
  INTAKING    a loose ball is inside this robot's size-scaled intake zone and is being
              consumed (scoring_events is mid-intake on it, or just confirmed one);
              equivalently the inferred held-count is rising. REFINED (the user's exact
              spec): an embedded image-space ``BinScorer`` additionally gates intake on a
              ball lying within radius R of the robot AND in the robot's DIRECTION OF
              TRAVEL (a velocity dot-product cone) — so a ball merely *near* a robot that
              is driving away does not read INTAKING. Either signal (field-meter scoring
              authority OR the motion-gated bbox cone) lights INTAKING.
  DEFENSE     the robot is geofenced inside the OPPONENT's half of the field, sitting
              near an opponent robot, and is not itself scoring/intaking.
  HOLDING(n)  carrying n>=1 balls (n = intaken − shot, inferred — held balls are not
              directly visible once swallowed).
  TRAVERSING  the default: moving (or idle) with no ball, not at a hub/tower, not on
              defense. The catch-all "just driving".

`label_for(robot_id)` renders the chosen phase for the overlay, e.g.
``"R3 SHOOTING"`` / ``"R1 INTAKING(2)"`` / ``"R5 CLIMB L2"`` / ``"B2 HOLDING(1)"``.

What this layer CONSUMES (it does NOT reimplement any of it)
------------------------------------------------------------
* ``sixbot.SixRobotTracker`` tracks — read off ``FrameState.robots`` as ``Track``
  objects (id / alliance / field_xy_m / xyxy / vel_mps); positions + heading-over-time
  drive jitter and motion.
* ``scoring_events.ScoringEventTracker`` — the authority on intake / outtake / shot /
  scored. PhaseTracker OWNS one internally and ``step()``s it on the same FrameState,
  then reads its per-frame Events + per-robot tallies (``held_count``, shots, etc.).
  This is the single source of truth for held-count and shot outcomes; we never
  re-derive them from raw balls here.
* ``collector_bin.BinScorer`` (OPTIONAL, the user's bbox/velocity spec) — an image-space
  ball<->robot<->bin state machine PhaseTracker also OWNS and ``step()``s on the same
  FrameState. It is a *refinement signal*, NOT a second scoring authority: it supplies
  (a) a motion-gated INTAKING cue (radius + velocity dot-product), (b) a HELD-by-bounding
  -box cue that is robust when a ball rides the bumper, (c) SHOOTING via a ball leaving
  the box at high exit speed, and (d) bin-entry SCORE *candidates* (never counted while a
  ball is inside a robot box). Held-count + shot accuracy still come from the field-meter
  authority above (no double-count); the bbox cues only OR-in to light a phase the demo
  eye expects, and the bin scores are exposed as candidates with their confidence.
* geometry — a duck-typed field model (``FieldModel`` satisfies it) for HUB / TOWER
  placement and the field half-line that defines "opponent zone". ``hub_regions`` is
  used by the scoring tracker we embed.
* the match clock / phase — ``FrameState.overlay.phase`` ("auto"|"teleop"|"endgame")
  is authoritative; ``overlay.timer_s`` is the fallback when phase is absent.

HONEST about what is inferential / what needs geometry|overlay (never fabricated)
---------------------------------------------------------------------------------
* **held-count is INFERRED**, not seen: a swallowed ball is invisible to the detector,
  so HOLDING(n) is ``intaken − shot`` carried by ``scoring_events`` — it can drift if
  an intake or eject was missed. Flagged ``inferred: True`` on the held metric.
* **CLIMB level from a single view is a HEURISTIC.** True rung clearance is judged in
  3-D (z), which a monocular broadcast frame does not give. When a track carries a
  real ``z_m`` (e.g. from ballistics/PnP) we threshold on it; otherwise we fall back
  to BUMPER-PIXEL-HEIGHT GROWTH vs the robot's own on-ground baseline — a proxy that a
  zoom or camera move can fool. The latched level always carries ``method:
  "z" | "bumper_px_heuristic"`` and ``needs`` text so downstream knows which.
* **CLIMBING / DEFENSE need a valid field pose** (zones + tower distance live in field
  meters). Without ``field_xy_m`` on the robot we cannot geofence, so those phases are
  simply unreachable that frame and the corresponding metrics are flagged
  ``available: False`` with a ``needs`` note — they are NOT silently counted as 0.
* **jitter_index needs field-meter positions** to be a real m/s wobble; with only px
  it is reported but flagged px-scaled (``needs`` a pose for metric units).
* **shot accuracy / made** are exactly ``scoring_events``' resolvable-only figures
  (None when no HUB pose and no overlay pulse) — carried through unchanged, never a
  fabricated 0.

Pure-logic: imports no cv2 / numpy / torch (matches the rest of the refine layer), so
it parses and unit-tests on a bare interpreter.
"""
from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from frcscout.io.config import seconds_to_frames
from frcscout.io.types import Alliance, Event, FrameState, Track
from frcscout.refine.collector_bin import BinScorer, CollectorBin
from frcscout.refine.scoring_events import ScoringEventTracker


# ───────────────────────────────── phase vocabulary ──────────────────────────
TRAVERSING = "TRAVERSING"
INTAKING = "INTAKING"
HOLDING = "HOLDING"
SHOOTING = "SHOOTING"
SCORED = "SCORED"
CLIMBING = "CLIMBING"
DEFENSE = "DEFENSE"

PHASES = (SCORED, SHOOTING, CLIMBING, INTAKING, DEFENSE, HOLDING, TRAVERSING)


# ───────────────────────────────── defaults ──────────────────────────────────
# ALL timing in SECONDS, ALL distance/height in METERS. Never a hardcoded frame count
# (everything time-like is converted with seconds_to_frames against the REAL fps).
_DEFAULTS = {
    # ── motion ──
    # field speed (m/s) at/above which a robot counts as "moving" (else idle). Used
    # only to distinguish an idle TRAVERSING robot from a moving one in metrics; the
    # phase itself is TRAVERSING either way.
    "moving_min_speed_mps": 0.25,

    # ── jitter ──
    # rolling window (s) over which we measure positional + heading wobble. A jittery
    # driver oscillates within this window; a smooth driver does not.
    "jitter_window_s": 1.0,
    # weight (meters of equivalent position-stddev per radian of heading-stddev) used
    # to fold heading wobble into the single scalar jitter_index. Heading is noisier
    # and unitless-vs-position, so it is down-weighted.
    "jitter_heading_w_m_per_rad": 0.3,

    # ── latches (so one-frame events are visible on the overlay) ──
    "shot_hold_s": 0.6,            # show SHOOTING for this long after a shot event
    "scored_hold_s": 0.8,          # show SCORED for this long after a make
    # show INTAKING for this long after a CONFIRMED consumption (a pickup is a brief event;
    # this keeps it visible on the overlay without re-firing every frame on proximity).
    "intake_hold_s": 0.4,
    # max seconds a SHOOTING latch can persist before we stop waiting for a SCORED/MISS
    # (mirrors scoring_events' shot lookahead; the latch is purely cosmetic).
    "shoot_latch_max_s": 4.0,

    # ── intake burst window (balls_shot_at_once) ──
    # shots landing within this window of each other count as one "burst" (a salvo).
    "shot_burst_window_s": 1.0,

    # ── climb (TOWER) ──
    # a robot must be within this (m) of its TOWER centre to be a climb candidate.
    "tower_adjacency_m": 1.5,
    # REBUILT TOWER rung heights (m) — LOW 27in, MID 45in, HIGH 63in (game model §2).
    # A robot's bumper clearing a rung latches the corresponding LEVEL.
    "rung_low_m": 0.686,           # 27 in
    "rung_mid_m": 1.143,           # 45 in
    "rung_high_m": 1.600,          # 63 in  (only needed to bound L3)
    # bumper-pixel-height GROWTH ratio (vs the robot's on-ground baseline px height)
    # that we treat as "off the ground" when no real z is available. A robot lifted
    # onto a rung changes apparent bbox height; a modest growth/shrink ratio vs the
    # on-ground baseline is a coarse proxy only. Mapped to L1/L2/L3 by growth bands below.
    # NOTE: monocular heuristic — see module docstring. Real climb judging needs z.
    "climb_px_l1_ratio": 1.06,     # >=6% bbox-height change from baseline -> off ground
    "climb_px_l2_ratio": 1.14,
    "climb_px_l3_ratio": 1.24,
    # frames of sustained adjacency+rise before we believe a climb has started
    # (debounce a one-frame bbox flicker). In SECONDS.
    "climb_confirm_s": 0.3,
    # baseline bumper px height = median over this many seconds of recent on-ground,
    # tower-adjacent observations before the rise begins.
    "climb_baseline_s": 1.0,

    # ── intake source gating (the dense-field INTAKING-trust fix) ──
    # Whether the image-space BinScorer's INTAKING signal may light the INTAKING phase.
    # The field-meter ScoringEventTracker (converge + VANISH-into-radius + dedup) is the
    # TRUSTWORTHY intake authority; the bbox/cone bin signal is image-space and, on a 100+
    # loose-ball pile, is noisy even with the consumption gate (ball-id churn + bbox jitter
    # keep manufacturing cone->box "pickups"). So on a dense REAL field set this False
    # (deploy.yaml does) and let INTAKING come ONLY from the field-meter consumption — exactly
    # how SHOOTING defers to the correlated flight signal (require_flight_for_shot). On clean/
    # synthetic input leave True (the bin cue is then a useful, low-noise OR-in).
    "intake_from_bin": True,
    # Whether the image-space BinScorer's HELD cue (a ball centre inside the robot bbox) may
    # light the HOLDING phase. Same dense-field problem as intake: a robot merely PARKED OVER
    # a ball pile has loose litter overlapping its bbox every frame, so the bbox HELD cue read
    # HOLDING for ~83% of robot-frames — "sitting in litter", not possession. On a dense REAL
    # field set this False (deploy.yaml): HOLDING then requires TRUE tracked possession (the
    # field-meter authority's held_count > 0 from a confirmed converge->vanish intake), and a
    # robot driving through litter with no tracked intake reads TRAVERSING. The bbox HELD cue
    # remains an internal candidate (held-count display when on, bin tallies) but does not
    # light the phase. On clean/synthetic input leave True (the bbox cue is low-noise there).
    "hold_from_bin": True,

    # ── defense ──
    # a defender must be within this (m) of an OPPONENT robot to count as defending.
    "defense_proximity_m": 2.0,
    # field length (m) fallback for the half-line geofence when the field model does
    # not expose one. Overridden by field_model.length_m when present.
    "field_length_m": 16.46,
    # tolerance (m) past the field extent [0, length] within which a back-projected robot
    # pose is still trusted for the opponent-half geofence. Beyond it the pose is treated as
    # an unreliable homography EXTRAPOLATION and DEFENSE is not asserted (see _in_opponent_zone).
    "field_extent_tol_m": 1.5,

    # ── HELD(n) sanity bound ──
    # held-count is INFERRED (intaken − shot) and can DRIFT upward if intakes are over-
    # counted or shots/ejects are missed. Cap the SHOWN held at a plausible robot capacity
    # (REBUILT robots carry only a few FUEL) so a drifted count never displays an absurd n.
    "max_plausible_held": 5,
    # if a robot has shown held>=1 but no NEW intake (and no shot) for this long, the
    # inferred possession is treated as STALE and decayed by one (a swallowed ball we never
    # saw leave). 0 disables decay. Keeps HOLDING(n) honest on a long, eventless track.
    "held_stale_decay_s": 8.0,
}


# ───────────────────────── tower localizer protocol (duck-typed) ──────────────
# Anything exposing EITHER ``towers`` (a {alliance: {center_m, climb_height_m}} dict,
# as FieldModel does) OR a ``tower_for(alliance) -> dict`` works. We never import
# FieldModel — we duck-type, so the synthetic tests can pass a tiny stub.
def _tower_for(field_model: object, alliance: str) -> Optional[dict]:
    if field_model is None or alliance is None:
        return None
    fn = getattr(field_model, "tower_for", None)
    if callable(fn):
        try:
            t = fn(alliance)
            return t or None
        except Exception:
            return None
    towers = getattr(field_model, "towers", None)
    if isinstance(towers, dict):
        return towers.get(alliance)
    return None


def _hub_center(field_model: object, alliance: str) -> Optional[tuple[float, float]]:
    if field_model is None or alliance is None:
        return None
    fn = getattr(field_model, "hub_for", None)
    hub = None
    if callable(fn):
        try:
            hub = fn(alliance)
        except Exception:
            hub = None
    if not hub:
        hubs = getattr(field_model, "hubs", None)
        if isinstance(hubs, dict):
            hub = hubs.get(alliance)
    if hub and "center_m" in hub:
        c = hub["center_m"]
        return (float(c[0]), float(c[1]))
    return None


def _field_length(field_model: object, default: float) -> float:
    if field_model is not None:
        lm = getattr(field_model, "length_m", None)
        if lm:
            try:
                return float(lm)
            except (TypeError, ValueError):
                pass
    return float(default)


# ───────────────────────────── per-robot rolling state ───────────────────────
@dataclass
class _RobotPhase:
    """Mutable per-robot phase/metric accumulator (NOT serialized directly; see
    ``snapshot``). Times are SECONDS; distances/heights METERS."""
    robot_id: int
    alliance: Optional[Alliance] = None

    # the phase chosen on the latest frame + the level if CLIMBING
    phase: str = TRAVERSING
    climb_level: int = 0                       # 0 = not climbing; 1/2/3 latched
    climb_method: Optional[str] = None         # "z" | "bumper_px_heuristic"

    # ── latches (frame indices the latch is valid UNTIL) ──
    shot_latch_until: int = -1
    scored_latch_until: int = -1
    intake_latch_until: int = -1               # confirmed-consumption INTAKING latch
    shoot_started_frame: int = -1              # when the current SHOOTING latch began

    # ── jitter rolling buffers (field xy + heading rad, with t) ──
    _pos: deque = field(default_factory=deque)   # (t_s, x, y)
    _hdg: deque = field(default_factory=deque)   # (t_s, heading_rad)
    jitter_index: Optional[float] = None
    jitter_units: Optional[str] = None         # None (no sample yet) | "m" (pose) | "px"

    # ── intake / possession timing ──
    intake_duration_s: float = 0.0            # cumulative time INTAKING
    time_to_intake_s: Optional[float] = None  # first-seen -> first confirmed intake
    first_seen_t_s: Optional[float] = None
    max_held: int = 0

    # ── HELD(n) sanity bound bookkeeping (does NOT mutate the scoring authority) ──
    held_decay: int = 0                       # balls we've decayed off a stale inferred hold
    _last_raw_held: int = 0                   # scoring held_count seen last frame (change edge)
    _held_change_t_s: Optional[float] = None  # when the raw held last changed

    # ── shots ──
    _shot_times: deque = field(default_factory=deque)   # recent shot t_s (burst calc)
    balls_shot_at_once: int = 0               # max shots inside one burst window

    # ── possession / defense durations ──
    possession_time_s: float = 0.0            # cumulative seconds holding >=1 ball
    defense_time_s: float = 0.0               # cumulative seconds in DEFENSE
    _last_t_s: Optional[float] = None         # for dt integration

    # ── climb bookkeeping ──
    _ground_px_h: deque = field(default_factory=deque)   # (t_s, bbox_px_height) on-ground
    _climb_rise_frames: int = 0               # consecutive adjacent+rising frames
    climb_start_t_s: Optional[float] = None

    # ── phase timeline: list of {phase, t_start, t_end} ──
    timeline: list = field(default_factory=list)

    # availability flags (honest about geometry/overlay gating)
    pose_seen: bool = False                   # ever had a field pose this match
    climb_geom_seen: bool = False             # ever had tower geometry to test against


# ─────────────────────────────────── tracker ─────────────────────────────────
class PhaseTracker:
    """Per-robot activity-PHASE classifier + scouting-metric accumulator.

    Usage::

        pt = PhaseTracker(field_model, hub_localizer, fps=fps)   # both optional
        for st in frame_states:
            out = pt.update(st)            # {robot_id: {phase, held, metrics...}}
            label = pt.label_for(3)        # "R3 SHOOTING"

    ``field_model`` (duck-typed; ``FieldModel`` fits) supplies HUB/TOWER positions and
    the field half-line for the opponent-zone geofence. ``hub_localizer`` (anything
    with ``in_hub``; ``FieldModel`` or ``HubRegions`` fit) is handed to the embedded
    ``ScoringEventTracker`` so shots/scores resolve exactly as that module defines —
    we do not re-implement scoring. Either may be None: the dependent phases/metrics
    then degrade honestly (flagged unavailable), they are never fabricated.

    The returned dict and ``metrics(...)`` are JSON-serializable (primitives / None /
    lists / nested dicts only).
    """

    def __init__(
        self,
        field_model: object | None = None,
        hub_localizer: object | None = None,
        cfg: Optional[dict] = None,
        fps: float = 60.0,
        scoring_tracker: Optional[ScoringEventTracker] = None,
        bin_scorer: Optional[BinScorer] = None,
        collector_bin: Optional[CollectorBin] = None,
    ):
        self.field_model = field_model
        c = dict(_DEFAULTS)
        c.update(cfg or {})
        self.cfg = c
        self.fps = float(fps)

        self.moving_min_speed = float(c["moving_min_speed_mps"])
        self.jitter_hdg_w = float(c["jitter_heading_w_m_per_rad"])
        self.tower_adj_m = float(c["tower_adjacency_m"])
        self.rung_low = float(c["rung_low_m"])
        self.rung_mid = float(c["rung_mid_m"])
        self.rung_high = float(c["rung_high_m"])
        self.px_l1 = float(c["climb_px_l1_ratio"])
        self.px_l2 = float(c["climb_px_l2_ratio"])
        self.px_l3 = float(c["climb_px_l3_ratio"])
        self.defense_prox_m = float(c["defense_proximity_m"])

        # The embedded scoring authority. We hand it the SAME hub localizer + fps so
        # its windows match ours. Callers may inject one (e.g. to share state with the
        # rest of the pipeline); otherwise we own a fresh instance.
        self.scoring = scoring_tracker or ScoringEventTracker(
            hub_localizer, cfg=self._scoring_cfg(cfg), fps=self.fps)

        # OPTIONAL image-space ball<->robot<->bin refinement (the user's bbox/velocity
        # spec). NOT a second scoring authority — a refinement-cue + bin-score-candidate
        # source. Off unless a caller hands us one, OR cfg asks us to build a default one
        # (``enable_bin_scorer: True``). When ``collector_bin`` is given we wrap it;
        # otherwise the default CollectorBin (measured qual59 HUB regions) is used.
        if bin_scorer is not None:
            self.bin_scorer: Optional[BinScorer] = bin_scorer
        elif collector_bin is not None or c.get("enable_bin_scorer"):
            self.bin_scorer = BinScorer(
                bin_loc=collector_bin, cfg=self._bin_cfg(cfg), fps=self.fps)
        else:
            self.bin_scorer = None

        self._recompute_windows()
        self.robots: dict[int, _RobotPhase] = {}
        self._field_len = _field_length(field_model, c["field_length_m"])
        self._bin_scores: list = []                # accumulated BinScoreEvents (candidates)
        # optional INTAKING source attribution (FRCSCOUT_PHASE_DEBUG=1) — see update().
        import os as _os
        self._debug = ({"intake_fieldmeter": 0, "intake_bin": 0}
                       if _os.environ.get("FRCSCOUT_PHASE_DEBUG") else None)

    @staticmethod
    def _scoring_cfg(cfg: Optional[dict]) -> Optional[dict]:
        """Pass through only the scoring-relevant keys a caller put in our cfg, so the
        embedded tracker can be tuned without us shadowing its many own defaults."""
        if not cfg:
            return None
        passthru = {
            k: v for k, v in cfg.items()
            if k in (
                "intake_k", "intake_floor_m", "converge_min_s", "vanish_grace_s",
                "abandon_radius_mult", "intake_dedup_s", "outtake_origin_m",
                "outtake_min_speed_mps", "shot_min_speed_mps", "shot_lookahead_s",
                "hub_tol_m", "score_window_s", "score_lag_s",
                "default_hub_confidence", "overlay_confidence",
            )
        }
        return passthru or None

    @staticmethod
    def _bin_cfg(cfg: Optional[dict]) -> Optional[dict]:
        """Pass through only the BinScorer-relevant keys a caller put in our cfg (image/
        bbox-space knobs), so the embedded bin scorer can be tuned without shadowing its
        own defaults."""
        if not cfg:
            return None
        passthru = {
            k: v for k, v in cfg.items()
            if k in (
                "intake_k_box", "intake_floor_px", "intake_cone_halfangle_rad",
                "intake_min_robot_speed_px_s", "intake_confirm_s", "held_pad_px",
                "held_confirm_s", "shot_min_exit_speed_px_s", "shot_to_score_max_s",
                "bin_tol_px", "bin_tol_m", "score_clear_px", "ball_ttl_s",
                "image_score_confidence", "field_score_confidence",
                "require_flight_for_shot", "min_held_s_for_shot", "intake_dedup_s",
                "require_consumption_for_intake", "intake_signal_hold_s",
            )
        }
        return passthru or None

    def _recompute_windows(self) -> None:
        c = self.cfg
        f = self.fps
        self.jitter_window_frames = seconds_to_frames(c["jitter_window_s"], f)
        self.shot_hold_frames = seconds_to_frames(c["shot_hold_s"], f)
        self.scored_hold_frames = seconds_to_frames(c["scored_hold_s"], f)
        self.intake_hold_frames = seconds_to_frames(c["intake_hold_s"], f)
        self.shoot_latch_max_frames = seconds_to_frames(c["shoot_latch_max_s"], f)
        self.shot_burst_frames = seconds_to_frames(c["shot_burst_window_s"], f)
        self.climb_confirm_frames = seconds_to_frames(c["climb_confirm_s"], f)
        self.climb_baseline_frames = seconds_to_frames(c["climb_baseline_s"], f)

    # ─────────────────────────────── helpers ─────────────────────────────────
    @staticmethod
    def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _speed_mps(t: Track) -> float:
        v = getattr(t, "vel_mps", (0.0, 0.0)) or (0.0, 0.0)
        return math.hypot(v[0], v[1])

    @staticmethod
    def _heading_rad(t: Track) -> Optional[float]:
        """Robot heading in radians. There is no heading field on ``Track``; we accept
        an explicit ``heading_rad`` dynamic attribute if a consumer attached one, else
        derive it from the velocity direction when the robot is actually moving (a
        stationary robot has no meaningful instantaneous heading from velocity)."""
        h = getattr(t, "heading_rad", None)
        if h is not None:
            try:
                return float(h)
            except (TypeError, ValueError):
                pass
        v = getattr(t, "vel_mps", None)
        if v and (abs(v[0]) + abs(v[1])) > 1e-6:
            return math.atan2(v[1], v[0])
        return None

    @staticmethod
    def _bbox_px_height(t: Track) -> Optional[float]:
        box = getattr(t, "xyxy", None)
        if not box or len(box) != 4:
            return None
        h = abs(box[3] - box[1])
        return h if h > 0 else None

    @staticmethod
    def _robot_z_m(t: Track) -> Optional[float]:
        """A real height (m) if some upstream layer attached ``z_m`` (PnP/ballistics).
        Almost never present from a monocular broadcast frame -> we fall back to the
        bumper-pixel heuristic. Honest: see module docstring."""
        z = getattr(t, "z_m", None)
        if z is None:
            return None
        try:
            return float(z)
        except (TypeError, ValueError):
            return None

    def _state(self, robot: Track) -> _RobotPhase:
        rp = self.robots.get(robot.tid)
        if rp is None:
            rp = _RobotPhase(robot_id=robot.tid, alliance=robot.alliance)
            self.robots[robot.tid] = rp
        elif robot.alliance is not None:
            rp.alliance = robot.alliance
        return rp

    # ─────────────────────────────── main step ───────────────────────────────
    def update(self, st: FrameState, fps: Optional[float] = None) -> dict:
        """Advance one frame: classify each robot's PHASE and accumulate metrics.

        Returns ``{robot_id: {phase, held, level, label, metrics...}}`` —
        JSON-serializable. Robots absent from the frame are not updated (their state
        persists; the timeline is closed lazily on the next finalize/snapshot)."""
        if fps is not None and float(fps) != self.fps:
            self.fps = float(fps)
            self._recompute_windows()
            self.scoring.fps = self.fps
            self.scoring._recompute_windows()

        # 1) Drive the embedded scoring authority on this same frame. Its events are
        #    the source of truth for intake/shot/scored; its tallies hold held_count.
        events = self.scoring.step(st, fps=self.fps)

        # 1b) Drive the OPTIONAL image-space bin scorer on the same frame (the user's
        #     bbox/velocity spec). It is a refinement-cue + bin-score-candidate source,
        #     NOT a second scoring authority, so its events do not feed held-count/accuracy.
        bin_intaking: set[int] = set()        # robots the bbox cone says are intaking
        bin_held: dict[int, list] = {}        # robot -> [ball ids inside its box]
        bin_shooting: set[int] = set()        # robots whose held ball just left fast
        if self.bin_scorer is not None:
            bin_events = self.bin_scorer.step(st, fps=self.fps)
            for be in bin_events:
                self._bin_scores.append(be)
            sig = self.bin_scorer.robot_signals()
            for rid, s in sig.items():
                if s.get("intaking"):
                    bin_intaking.add(rid)
                if s.get("held_balls"):
                    bin_held[rid] = list(s["held_balls"])
                if s.get("shooting"):
                    bin_shooting.add(rid)

        # 2) Per-robot event flags this frame (from the canonical Event stream).
        intaking_now: set[int] = set()       # mid/confirmed intake this frame
        shot_now: dict[int, int] = {}         # robot_id -> #shots fired this frame
        scored_now: set[int] = set()         # robot_id with a confirmed make this frame
        for e in events:
            rid = e.robot_id
            if rid is None:
                continue
            kind = (e.detail or {}).get("kind")
            if e.type == "pickup" and kind == "intake":
                intaking_now.add(rid)
            elif e.type == "shot" and kind == "shot":
                shot_now[rid] = shot_now.get(rid, 0) + 1
            elif e.type == "score" and kind == "shot_made":
                scored_now.add(rid)

        # The bbox/cone INTAKING signal OR-s into the canonical set ONLY when ``intake_from_bin``
        # is enabled. On a dense real field it is disabled (deploy.yaml) because the image-space
        # bin signal is noisy in a 100+ ball pile; INTAKING then comes solely from the trustworthy
        # field-meter consumption authority (converge + vanish-into-radius). The field-meter and
        # the bin cue are independent evidence; when both are trusted either lighting it is correct.
        if self.cfg.get("intake_from_bin", True):
            intaking_now |= bin_intaking

        # OPTIONAL source attribution (off unless FRCSCOUT_PHASE_DEBUG=1): tally which
        # signal lit INTAKING this frame so the dense-field over-fire can be measured by
        # source (field-meter consumption vs bbox consumption). Zero-cost when off.
        if self._debug is not None:
            fm_only = intaking_now - bin_intaking
            self._debug["intake_fieldmeter"] += len(fm_only)
            self._debug["intake_bin"] += len(bin_intaking)

        # 3) Robots present this frame, indexed; opponents for the defense test.
        robots = list(st.robots)
        by_id = {r.tid: r for r in robots}
        self._bin_held_now = bin_held         # used by _classify (HELD-by-box cue)
        self._bin_shooting_now = bin_shooting

        out: dict = {}
        for r in robots:
            rp = self._state(r)
            self._accumulate(st, r, rp, intaking_now, shot_now, scored_now)
            phase, level = self._classify(st, r, rp, robots, by_id,
                                          intaking_now, shot_now, scored_now)
            self._set_phase(rp, phase, level, st.t_s)
            out[r.tid] = self._frame_record(rp)
        return out

    # ───────────────────────── metric accumulation ───────────────────────────
    def _accumulate(self, st: FrameState, r: Track, rp: _RobotPhase,
                    intaking_now: set, shot_now: dict, scored_now: set) -> None:
        """Fold this frame's observations into the rolling metrics (independent of the
        phase finally chosen)."""
        t = st.t_s
        dt = 0.0 if rp._last_t_s is None else max(0.0, t - rp._last_t_s)

        if rp.first_seen_t_s is None:
            rp.first_seen_t_s = t

        # ── HELD(n) sanity bookkeeping (computed BEFORE reading the bounded held) ──
        # held-count is INFERRED (intaken − shot) and drifts; bound it without corrupting
        # the scoring authority. (1) any change in the raw held (intake or eject) restarts
        # the staleness clock; (2) decay one ball off a hold that has gone eventless for
        # ``held_stale_decay_s`` (a swallowed ball we likely never saw leave).
        rs = self.scoring.robots.get(r.tid)
        raw_held = int(rs.held_count) if rs is not None else 0
        if raw_held != rp._last_raw_held:
            rp._held_change_t_s = t          # a real change -> restart the staleness clock
            if raw_held < rp._last_raw_held:
                rp.held_decay = max(0, rp.held_decay - (rp._last_raw_held - raw_held))
        rp._last_raw_held = raw_held
        decay_s = float(self.cfg["held_stale_decay_s"])
        if (decay_s > 0 and rp._held_change_t_s is not None
                and (raw_held - rp.held_decay) >= 1
                and (t - rp._held_change_t_s) >= decay_s):
            rp.held_decay += 1               # treat one inferred ball as a missed eject
            rp._held_change_t_s = t          # re-arm so we decay at most ~1 per window

        # ── held-count (INFERRED + sanity-bounded) + possession time ──
        held = self._held(r.tid)
        rp.max_held = max(rp.max_held, held)
        if held >= 1:
            rp.possession_time_s += dt

        # ── jitter rolling buffers (position + heading) ──
        # NEVER mix scales in one buffer: meters take precedence (a real metric); if a
        # pose lands after px frames we clear the px history and switch to meters.
        have_pose = r.field_xy_m is not None
        rp.pose_seen = rp.pose_seen or have_pose
        if have_pose:
            if rp.jitter_units != "m":
                rp._pos.clear()
            rp.jitter_units = "m"
            rp._pos.append((t, r.field_xy_m[0], r.field_xy_m[1]))
        elif rp.jitter_units != "m":
            # px fallback: use the bbox centroid so jitter is still *reported* (flagged px)
            cx, cy = getattr(r, "centroid_px", (0.0, 0.0)) or (0.0, 0.0)
            rp.jitter_units = "px"
            rp._pos.append((t, float(cx), float(cy)))
        hdg = self._heading_rad(r)
        if hdg is not None:
            rp._hdg.append((t, hdg))
        self._trim(rp._pos, t)
        self._trim(rp._hdg, t)
        rp.jitter_index = self._jitter(rp)

        # ── intake duration + time-to-intake ──
        # Count duration only while INTAKING is *actually* the phase this robot reads —
        # i.e. a confirmed consumption this frame OR inside the brief intake latch. We no
        # longer add time for mere proximity (``_mid_intake``); that inflated the metric for
        # any robot near loose balls, the same noise the phase over-fire came from.
        if r.tid in intaking_now or st.frame_idx <= rp.intake_latch_until:
            rp.intake_duration_s += dt
            if rp.time_to_intake_s is None and r.tid in intaking_now:
                rp.time_to_intake_s = max(0.0, t - (rp.first_seen_t_s or t))

        # ── shots: burst tracking (balls_shot_at_once) ──
        n_shots = shot_now.get(r.tid, 0)
        for _ in range(n_shots):
            rp._shot_times.append(t)
            rp.shot_latch_until = st.frame_idx + self.shot_hold_frames
            if rp.shoot_started_frame < 0 or st.frame_idx > rp.shoot_started_frame + self.shoot_latch_max_frames:
                rp.shoot_started_frame = st.frame_idx
        if n_shots:
            rp.balls_shot_at_once = max(rp.balls_shot_at_once, self._burst_size(rp, st.frame_idx))

        # ── scored latch ──
        if r.tid in scored_now:
            rp.scored_latch_until = st.frame_idx + self.scored_hold_frames

        rp._last_t_s = t

    def _trim(self, buf: deque, now: float) -> None:
        window_s = self.cfg["jitter_window_s"]
        while buf and (now - buf[0][0]) > window_s:
            buf.popleft()

    def _jitter(self, rp: _RobotPhase) -> Optional[float]:
        """Single scalar = stddev of position over the window + a down-weighted stddev
        of heading. High when the driver wobbles/oscillates, low when smooth. Needs
        >=2 samples; None otherwise. In meters when a pose existed, else px (flagged)."""
        if len(rp._pos) < 2:
            return None
        xs = [p[1] for p in rp._pos]
        ys = [p[2] for p in rp._pos]
        # de-trend: stddev about the path *mean* captures wobble, not steady travel,
        # only weakly — a fast straight run still has spread. We additionally subtract
        # the linear drift by using the stddev of frame-to-frame steps' direction via
        # heading; position spread stays a simple, robust wobble proxy.
        pos_sd = math.hypot(_pstdev(xs), _pstdev(ys))
        hdg_sd = 0.0
        if len(rp._hdg) >= 2:
            hs = [h[1] for h in rp._hdg]
            hdg_sd = _circ_stdev(hs)
        return pos_sd + self.jitter_hdg_w * hdg_sd

    def _burst_size(self, rp: _RobotPhase, frame_idx: int) -> int:
        """How many shots fall inside the most recent burst window ending now."""
        window_s = self.cfg["shot_burst_window_s"]
        if not rp._shot_times:
            return 0
        now = rp._shot_times[-1]
        # drop shots older than the window from the FRONT (keep the deque bounded)
        while rp._shot_times and (now - rp._shot_times[0]) > window_s:
            rp._shot_times.popleft()
        return len(rp._shot_times)

    # ───────────────────────── held-count accessors ──────────────────────────
    def _raw_held(self, rid: int) -> int:
        """The scoring authority's UNBOUNDED inferred held-count (intaken − shot)."""
        rs = self.scoring.robots.get(rid)
        return int(rs.held_count) if rs is not None else 0

    def _held(self, rid: int) -> int:
        """The SANITY-BOUNDED held-count to use everywhere a phase decision / display
        reads possession: the authority's inferred count minus any stale-decay this robot
        accrued, clamped to a plausible capacity and floored at 0. Never negative, never
        an absurd n. Does NOT mutate the scoring authority (shot accuracy stays intact)."""
        raw = self._raw_held(rid)
        rp = self.robots.get(rid)
        decay = rp.held_decay if rp is not None else 0
        cap = int(self.cfg["max_plausible_held"])
        return max(0, min(raw - decay, cap))

    def _mid_intake(self, rid: int) -> bool:
        """True if the scoring authority currently has a ball in the INTAKE phase
        targeting this robot (i.e. a ball converging/entering its radius but not yet
        swallowed). Lets INTAKING show during the approach, not only on confirmation."""
        for bs in self.scoring._balls.values():
            if getattr(bs, "phase", None) == "INTAKE" and getattr(bs, "target_robot", None) == rid:
                return True
        return False

    # ───────────────────────────── classification ────────────────────────────
    def _classify(self, st: FrameState, r: Track, rp: _RobotPhase,
                  robots: list, by_id: dict,
                  intaking_now: set, shot_now: dict, scored_now: set) -> tuple[str, int]:
        """Choose ONE phase by priority. Returns (phase, climb_level)."""
        fidx = st.frame_idx

        # 1) SCORED — a confirmed make this frame, or still inside the scored latch.
        if r.tid in scored_now or fidx <= rp.scored_latch_until:
            return SCORED, rp.climb_level

        # 2) SHOOTING — a shot fired this frame (field-meter authority OR the bbox
        #    high-exit-speed cue), or still inside the shot latch. The bbox cue is the
        #    user's exact discriminator: a HELD ball leaving the box FAST (vs a slow drop).
        bin_shooting = getattr(self, "_bin_shooting_now", ())
        if r.tid in shot_now or r.tid in bin_shooting or fidx <= rp.shot_latch_until:
            if (r.tid in bin_shooting) and fidx > rp.shot_latch_until:
                rp.shot_latch_until = fidx + self.shot_hold_frames   # latch the bbox shot
            return SHOOTING, rp.climb_level

        # 3) CLIMBING — endgame + adjacent to own tower + rising. Latches a level.
        level = self._climb_level(st, r, rp)
        if level > 0:
            return CLIMBING, level
        # If a level was already latched earlier (monotonic), keep showing CLIMBING
        # while still in endgame and adjacent (a held climb).
        if rp.climb_level > 0 and self._endgame(st) and self._tower_adjacent(r):
            return CLIMBING, rp.climb_level

        # 4) INTAKING — a CONFIRMED CONSUMPTION this frame, or still inside the brief
        #    intake latch. "Confirmed" means real evidence of a ball being taken IN:
        #      * the field-meter ScoringEventTracker emitted a ``pickup/intake`` (a ball
        #        converged then VANISHED inside the robot's size-scaled radius), and/or
        #      * the bbox BinScorer saw a ball cross from its tracked approach cone INTO the
        #        robot box / vanish at it (``require_consumption_for_intake``).
        #    We deliberately do NOT light INTAKING on mere proximity-in-heading: in a 100+
        #    ball field a robot ALWAYS has a ball within radius in its travel direction, so
        #    the old cone-membership / ``_mid_intake`` (a ball merely approaching) fired for
        #    nearly every robot every frame — that was noise, not intaking. The latch keeps
        #    the brief, real pickup visible without re-deriving it every frame.
        if r.tid in intaking_now:
            rp.intake_latch_until = max(rp.intake_latch_until,
                                        fidx + self.intake_hold_frames)
            return INTAKING, rp.climb_level
        if fidx <= rp.intake_latch_until:
            return INTAKING, rp.climb_level

        # 5) DEFENSE — in the opponent's half, near an opponent, not scoring/intaking.
        if self._is_defending(st, r, rp, robots):
            return DEFENSE, rp.climb_level

        # 6) HOLDING(n) — TRUE tracked possession: the field-meter ScoringEventTracker
        #    inferred a held ball (a real converge->vanish intake the authority confirmed,
        #    ``held_count > 0``). The bbox HELD cue (a ball centre inside the robot box) ONLY
        #    OR-s in when ``hold_from_bin`` is enabled. On a dense REAL field it is DISABLED
        #    (deploy.yaml), because a robot merely PARKED OVER a ball pile has loose litter
        #    overlapping its bbox every frame — that is "sitting in litter", NOT possession,
        #    and it lit HOLDING for ~83% of robot-frames (the over-fire just moved here from
        #    INTAKING). A robot driving through litter with no TRACKED intake must read
        #    TRAVERSING. Mirrors ``intake_from_bin``: the bbox cue stays an internal candidate
        #    (held-count DISPLAY, bin tallies) but does not light the phase on the real field.
        #    Honest trade: HOLDING is now LOW-RECALL (only when we actually tracked an intake)
        #    — far better than over-asserting possession most of the match.
        bin_held = getattr(self, "_bin_held_now", {})
        if self._held(r.tid) >= 1:
            return HOLDING, rp.climb_level
        if self.cfg.get("hold_from_bin", True) and bin_held.get(r.tid):
            return HOLDING, rp.climb_level

        # 7) TRAVERSING — the default (a robot just driving, incl. through loose litter).
        return TRAVERSING, rp.climb_level

    # ── phase helpers ──
    def _endgame(self, st: FrameState) -> bool:
        ph = getattr(st.overlay, "phase", None)
        if ph is not None:
            return str(ph).lower() == "endgame"
        # fallback: timer counts DOWN within teleop; last 30s is endgame. We only know
        # the displayed timer, which in REBUILT resets per period — without the period
        # we cannot be sure, so be conservative and treat <=30s as endgame ONLY when a
        # timer is present and small. (Authoritative path is overlay.phase above.)
        timer = getattr(st.overlay, "timer_s", None)
        return timer is not None and timer <= 30.0

    def _tower_adjacent(self, r: Track) -> bool:
        if r.field_xy_m is None or r.alliance is None:
            return False
        tower = _tower_for(self.field_model, r.alliance)
        if not tower or "center_m" not in tower:
            return False
        c = tower["center_m"]
        return self._dist(r.field_xy_m, (float(c[0]), float(c[1]))) <= self.tower_adj_m

    def _climb_level(self, st: FrameState, r: Track, rp: _RobotPhase) -> int:
        """Detect/ratchet the climb LEVEL. Returns the (monotonic) latched level, or 0.

        Endgame-gated and tower-adjacency-gated. Prefers a real z when present; else a
        BUMPER-PIXEL-HEIGHT heuristic vs the robot's own on-ground baseline. The level
        only ever ratchets UP (a climb does not reverse on the scoreboard)."""
        if not self._endgame(st):
            return rp.climb_level
        if not self._tower_adjacent(r):
            # not at the tower: keep any prior latch but don't progress; also reset the
            # rise debounce so a later approach starts fresh.
            rp._climb_rise_frames = 0
            return rp.climb_level
        rp.climb_geom_seen = True

        # ── preferred: a real height in meters ──
        z = self._robot_z_m(r)
        if z is not None:
            rp.climb_method = "z"
            new_level = self._z_to_level(z)
            return self._latch_climb(rp, new_level, st.t_s)

        # ── fallback: bumper-pixel-height growth vs on-ground baseline ──
        rp.climb_method = "bumper_px_heuristic"
        h = self._bbox_px_height(r)
        if h is None:
            return rp.climb_level
        baseline = self._climb_baseline(rp)
        if baseline is None:
            # still collecting on-ground samples; record this (assumed-ground) sample
            # only while no rise has begun, then bail.
            rp._ground_px_h.append((st.t_s, h))
            self._trim_baseline(rp, st.t_s)
            return rp.climb_level
        ratio = h / baseline if baseline > 0 else 1.0
        # treat shrink the same as growth in magnitude (camera may be above/below):
        growth = max(ratio, (1.0 / ratio) if ratio > 0 else 1.0)
        cand = self._px_growth_to_level(growth)
        if cand > 0:
            rp._climb_rise_frames += 1
            if rp._climb_rise_frames >= self.climb_confirm_frames:
                return self._latch_climb(rp, cand, st.t_s)
            return rp.climb_level
        # not rising -> this is an on-ground sample; refresh the baseline buffer.
        rp._climb_rise_frames = 0
        rp._ground_px_h.append((st.t_s, h))
        self._trim_baseline(rp, st.t_s)
        return rp.climb_level

    def _latch_climb(self, rp: _RobotPhase, new_level: int, t_s: float) -> int:
        if new_level > rp.climb_level:
            if rp.climb_start_t_s is None:
                rp.climb_start_t_s = t_s
            rp.climb_level = new_level
        return rp.climb_level

    def _z_to_level(self, z: float) -> int:
        if z >= self.rung_mid:
            return 3
        if z >= self.rung_low:
            return 2
        # off the ground but below the LOW rung clearance == LEVEL 1 (off carpet)
        if z > 0.05:
            return 1
        return 0

    def _px_growth_to_level(self, growth: float) -> int:
        if growth >= self.px_l3:
            return 3
        if growth >= self.px_l2:
            return 2
        if growth >= self.px_l1:
            return 1
        return 0

    def _climb_baseline(self, rp: _RobotPhase) -> Optional[float]:
        if len(rp._ground_px_h) < 2:
            return None
        return statistics.median(h for _, h in rp._ground_px_h)

    def _trim_baseline(self, rp: _RobotPhase, now: float) -> None:
        window_s = self.cfg["climb_baseline_s"]
        while rp._ground_px_h and (now - rp._ground_px_h[0][0]) > window_s:
            rp._ground_px_h.popleft()

    def _is_defending(self, st: FrameState, r: Track, rp: _RobotPhase,
                      robots: list) -> bool:
        """Geofence: robot in the OPPONENT half of the field, within proximity of an
        opponent robot. Needs field poses (its own + an opponent's). If we cannot
        geofence (no pose), DEFENSE is unreachable this frame (NOT counted)."""
        if r.field_xy_m is None or r.alliance is None:
            return False
        if not self._in_opponent_zone(r):
            return False
        opp = "red" if r.alliance == "blue" else "blue"
        for o in robots:
            if o.tid == r.tid or o.alliance != opp or o.field_xy_m is None:
                continue
            if self._dist(r.field_xy_m, o.field_xy_m) <= self.defense_prox_m:
                return True
        return False

    def _in_opponent_zone(self, r: Track) -> bool:
        """Half-field geofence. Blue defends near the RED end (+x half) and vice-versa.
        We split the field at length/2; a robot past the midline into the other
        alliance's half is "in the opponent zone". We orient halves by HUB x when the
        field model exposes hubs (robust to coordinate convention), else by raw x.

        SANITY: a homography fit on a tight tag cluster EXTRAPOLATES badly toward the field
        edges — robots routinely back-project to x well OUTSIDE [0, length] (measured ~18 m
        on a 16.46 m field). A half-field call on such an off-field x is untrustworthy, so we
        reject the geofence (and hence DEFENSE) when x is beyond the field extent by more than
        ``field_extent_tol_m``. DEFENSE then degrades to unreachable that frame, never faked."""
        if r.field_xy_m is None or r.alliance is None:
            return False
        x = r.field_xy_m[0]
        tol = float(self.cfg.get("field_extent_tol_m", 1.5))
        if x < -tol or x > (self._field_len + tol):
            return False
        mid = self._field_len / 2.0
        own_hub = _hub_center(self.field_model, r.alliance)
        if own_hub is not None:
            # opponent zone = the half NOT containing the robot's own hub.
            own_on_low = own_hub[0] < mid
            robot_on_low = x < mid
            return robot_on_low != own_on_low
        # fallback by convention: blue starts low-x, defends high-x (red) half.
        if r.alliance == "blue":
            return x > mid
        return x < mid

    # ───────────────────────── phase set / timeline ──────────────────────────
    def _set_phase(self, rp: _RobotPhase, phase: str, level: int, t_s: float) -> None:
        rp.phase = phase
        if phase == CLIMBING:
            rp.climb_level = max(rp.climb_level, level)
        # timeline: open a new segment whenever the (phase,level) label changes.
        label_key = (phase, rp.climb_level if phase == CLIMBING else 0)
        if not rp.timeline:
            rp.timeline.append({"phase": phase, "level": label_key[1],
                                "t_start": t_s, "t_end": t_s})
        else:
            last = rp.timeline[-1]
            last_key = (last["phase"], last.get("level", 0))
            if last_key == label_key:
                last["t_end"] = t_s
            else:
                rp.timeline.append({"phase": phase, "level": label_key[1],
                                    "t_start": t_s, "t_end": t_s})

    # ───────────────────────────── reporting ─────────────────────────────────
    def _held_display(self, rid: int) -> int:
        """The held-count to SHOW: the (sanity-bounded) field-meter authority count, and —
        only when ``hold_from_bin`` is enabled — at least the number of balls the bbox cue
        sees inside the box (so HOLDING(n) reflects a visibly-carried ball when the field
        intake was missed, on clean input).

        On the dense REAL field ``hold_from_bin`` is OFF: the bbox count is loose LITTER
        overlapping a parked robot, NOT possession (it showed absurd "HOLDING(8)+"), so we
        report ONLY the tracked field-meter count. Both inputs are still capped at
        ``max_plausible_held``. Metrics report the authority's count + ``held_inferred``;
        this is display-only."""
        cap = int(self.cfg["max_plausible_held"])
        base = self._held(rid)                       # already capped + decayed
        if not self.cfg.get("hold_from_bin", True):
            return base                              # litter-free: tracked possession only
        bin_held = getattr(self, "_bin_held_now", {})
        seen = min(len(bin_held.get(rid, [])) if bin_held else 0, cap)
        return max(base, seen)

    def _frame_record(self, rp: _RobotPhase) -> dict:
        """The compact per-frame dict returned from ``update`` for one robot
        (JSON-serializable)."""
        return {
            "robot_id": rp.robot_id,
            "alliance": rp.alliance,
            "phase": rp.phase,
            "held": self._held_display(rp.robot_id),
            "climb_level": rp.climb_level if rp.phase == CLIMBING or rp.climb_level else 0,
            "label": self._label(rp),
            "metrics": self._metrics_for(rp),
        }

    def _label(self, rp: _RobotPhase) -> str:
        """Renderer string, e.g. 'R3 SHOOTING' / 'R1 INTAKING(2)' / 'R5 CLIMB L2'."""
        prefix = self._prefix(rp)
        if rp.phase == CLIMBING:
            lvl = rp.climb_level if rp.climb_level > 0 else 1
            return f"{prefix} CLIMB L{lvl}"
        if rp.phase == HOLDING:
            return f"{prefix} HOLDING({self._held_display(rp.robot_id)})"
        if rp.phase == INTAKING:
            held = self._held_display(rp.robot_id)
            return f"{prefix} INTAKING({held})" if held else f"{prefix} INTAKING"
        return f"{prefix} {rp.phase}"

    def _prefix(self, rp: _RobotPhase) -> str:
        """'R3' style id prefix; uses the alliance initial when known (R=red, B=blue),
        else a neutral '#'."""
        a = rp.alliance
        letter = "R" if a == "red" else ("B" if a == "blue" else "#")
        return f"{letter}{rp.robot_id}"

    def label_for(self, robot_id: int) -> str:
        """Public renderer helper: the current label for ``robot_id`` (empty string for
        an unknown robot)."""
        rp = self.robots.get(robot_id)
        return self._label(rp) if rp is not None else ""

    def _metrics_for(self, rp: _RobotPhase) -> dict:
        """Per-robot METRICS (the user's parameters), JSON-serializable. Metrics that
        need geometry/overlay carry ``available``/``needs`` so a missing input reads as
        UNKNOWN, never a fabricated number. Scoring figures are pulled straight from
        the embedded ScoringEventTracker (single source of truth)."""
        rs = self.scoring.robots.get(rp.robot_id)
        shot_attempts = int(rs.shots) if rs else 0
        shots_made = int(rs.made) if rs else 0
        shot_acc = rs.shot_accuracy if rs else None        # None when unresolvable
        intakes = int(rs.intakes) if rs else 0

        jitter_available = rp.jitter_index is not None
        # jitter is only a *metric* (m) with a pose; px-scaled otherwise (still useful
        # for relative comparison, but flagged so it isn't read as meters).
        jitter_needs = None
        if not jitter_available:
            jitter_needs = ">=2 frames in the rolling window"
        elif rp.jitter_units != "m":
            jitter_needs = "a valid field pose for metric (m) units; value is px-scaled"

        return {
            # ── motion stability ──
            "jitter_index": {
                "value": (round(rp.jitter_index, 4) if rp.jitter_index is not None else None),
                "available": jitter_available,
                "units": rp.jitter_units,
                "needs": jitter_needs,
                "provenance": "stddev(field xy)+w*stddev(heading) over rolling window",
            },
            # ── intake ──
            "intake_duration_s": round(rp.intake_duration_s, 3),
            "time_to_intake_s": (round(rp.time_to_intake_s, 3)
                                 if rp.time_to_intake_s is not None else None),
            "intakes": intakes,
            # ── possession (held is INFERRED; flagged) ──
            "max_held": rp.max_held,
            "held_inferred": True,
            "possession_time_s": round(rp.possession_time_s, 3),
            # ── shots ──
            "balls_shot_at_once": rp.balls_shot_at_once,
            "shot_attempts": shot_attempts,
            "shots_made": shots_made,
            "shot_accuracy": {
                "value": shot_acc,
                "available": shot_acc is not None,
                "needs": (None if shot_acc is not None
                          else "a valid HUB pose or an overlay fuel pulse to resolve a shot"),
                "provenance": "scoring_events resolvable shots (made/(made+missed))",
            },
            # ── defense (geofence; needs pose) ──
            "defense_time_s": {
                "value": round(rp.defense_time_s, 3),
                "available": rp.pose_seen,
                "needs": (None if rp.pose_seen
                          else "a valid field pose to geofence the opponent zone"),
                "provenance": "cumulative time in DEFENSE phase",
            },
            # ── climb (single-view heuristic unless z; flagged) ──
            "climb_level": rp.climb_level,
            "climb_method": rp.climb_method,
            "climb_start_t_s": (round(rp.climb_start_t_s, 3)
                                if rp.climb_start_t_s is not None else None),
            "time_to_climb_s": self._time_to_climb(rp),
            "climb_available": rp.climb_geom_seen,
            "climb_needs": (None if rp.climb_geom_seen
                            else "endgame + TOWER geometry + a field pose; level via z "
                                 "(else bumper-px heuristic, monocular — can be fooled)"),
            # ── timeline ──
            "phase_timeline": [dict(seg) for seg in rp.timeline],
        }

    def _time_to_climb(self, rp: _RobotPhase) -> Optional[float]:
        """Onset (first tower-adjacent rise) -> first latched level, in seconds. None
        until a level is latched."""
        if rp.climb_level <= 0 or rp.climb_start_t_s is None:
            return None
        # the latch happens at/after start; we approximate duration as start->now via
        # the timeline's last CLIMBING segment end if present.
        for seg in reversed(rp.timeline):
            if seg["phase"] == CLIMBING:
                return round(max(0.0, seg["t_end"] - rp.climb_start_t_s), 3)
        return 0.0

    def metrics(self, robot_id: Optional[int] = None) -> dict:
        """Full metrics snapshot. With ``robot_id`` -> that robot's metrics dict; else
        ``{robot_id: metrics}`` for every robot seen. JSON-serializable."""
        if robot_id is not None:
            rp = self.robots.get(robot_id)
            return self._metrics_for(rp) if rp is not None else {}
        return {rid: self._metrics_for(rp) for rid, rp in self.robots.items()}

    def snapshot(self) -> dict:
        """Everything, keyed by robot id: phase + held + label + full metrics. The
        natural end-of-match dump. JSON-serializable."""
        return {rid: self._frame_record(rp) for rid, rp in self.robots.items()}

    # ───────────────────────── collector-bin (candidates) ────────────────────
    def bin_score_events(self) -> list:
        """All bin-entry SCORE *candidates* accumulated so far (the user's bin-scoring
        signal), as JSON-serializable dicts. These are NOT the authoritative score (the
        overlay FUEL counter is) — each carries its confidence + hub-active flags. Empty
        when no bin scorer is embedded."""
        return [be.to_event().detail | {
            "t_s": be.t_s, "frame_idx": be.frame_idx,
            "alliance": be.alliance, "robot_id": be.robot_id}
            for be in self._bin_scores]

    def bin_tallies(self) -> dict:
        """Per-alliance bin-interaction tallies from the embedded BinScorer
        (intakes/shots/losses/bin_scores/points), or ``{}`` when none is embedded.
        ``points`` honors HUB active/inactive gating (manual Table 6-4)."""
        return self.bin_scorer.tallies() if self.bin_scorer is not None else {}


# ───────────────────────────── small numeric helpers ─────────────────────────
def _pstdev(xs) -> float:
    """Population stddev, 0 for <2 samples (pure python; no numpy)."""
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    return statistics.pstdev(xs)


def _circ_stdev(angles) -> float:
    """Circular standard deviation of angles (radians) — the right spread measure for
    a heading that wraps at ±π. Returns 0 for <2 samples. Uses the mean resultant
    length R: circ_std = sqrt(-2 ln R)."""
    angs = list(angles)
    if len(angs) < 2:
        return 0.0
    s = sum(math.sin(a) for a in angs) / len(angs)
    c = sum(math.cos(a) for a in angs) / len(angs)
    r = math.hypot(s, c)
    if r <= 1e-9:
        return math.pi  # maximally dispersed
    if r >= 1.0:
        return 0.0
    return math.sqrt(-2.0 * math.log(r))


__all__ = [
    "PhaseTracker",
    "PHASES",
    "TRAVERSING", "INTAKING", "HOLDING", "SHOOTING", "SCORED", "CLIMBING", "DEFENSE",
]
