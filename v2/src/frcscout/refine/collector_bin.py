"""COLLECTOR BIN (the REBUILT *HUB*) localization + ball<->robot<->bin interaction scoring.

What the user calls the "collector bin" is, in the official 2026 *REBUILT* manual, the
**HUB** — the static receptacle a ROBOT shoots FUEL (the yellow foam balls) into. This
module is the field-element + scoring half of the per-robot interaction state machine
(``refine/phase.py`` is the per-robot PHASE half and consumes what is here).

Grounded in the manual (Version TU22, ``2026_REBUILT_GameManual.pdf``) — verified, not guessed
------------------------------------------------------------------------------------------------
* **FUEL** is the only SCORING ELEMENT: a 5.91 in (15.0 cm) foam ball  (§5.10 / §5.10.1, p.32).
* The **HUB** is a 47×47 in (~1.19 m) structure with a **41.7 in (~1.06 m) hexagonal top
  opening whose front edge is 72 in (~1.83 m) off the carpet**. There are **two** — one per
  ALLIANCE — each centered 158.6 in (~4.03 m) from its ALLIANCE WALL  (§5.4, pp.22-23).
  So the user's "single central bin" is really *two static alliance HUBs*; both are fixed
  field elements present the whole match, which is what matters here.
* **Scoring criterion:** "A FUEL is scored in the HUB once it passes through the top opening
  of the HUB and through the sensor array."  (§6.5.1, p.46).
* **Point value:** FUEL in an **active** HUB = **1 point** in BOTH AUTO and TELEOP; FUEL in
  an **inactive** HUB = **0 points**  (Table 6-4, §6.5.3, p.47). TOWER/climb points are
  separate and are NOT handled here.
* **HUB active/inactive** (§6.4.1, pp.44-45): both HUBs are active during AUTO, the
  TRANSITION SHIFT, and END GAME; during the four ALLIANCE SHIFTS exactly one alliance's HUB
  is active. We accept the active-state per frame (the caller derives it from the HUB DMX
  lights or the shift schedule); when it is unknown we *assume active* but FLAG the award
  ``hub_active_known: False`` so a 0-pt inactive score is never silently counted as 1.

Two responsibilities
---------------------
1. :class:`CollectorBin` — **localize** the static bin. The broadcast camera on the demo
   footage (qual59) is effectively fixed, so each HUB's top-opening occupies a fixed image
   region. We store an approximate **per-alliance opening polygon** in FRACTIONS of (W, H)
   (resolution-independent, same convention as ``hub_regions``/overlay ROIs), scale to the
   live frame, and run a pure point-in-polygon test — ``which_bin(px) -> "blue"|"red"|None``.
   A valid :class:`~frcscout.io.types.CameraPose` + a field model upgrades the test to field
   coordinates (survives pan/zoom). An optional one-time ``auto_locate`` snaps the polygons
   to the colored HUB structures in a frame (honest: a coarse colour heuristic, off by
   default; the measured fractions are the reliable path for this fixed shot).

2. :class:`BinScorer` — the **ball<->robot<->bin interaction** the user specified, EXACTLY:
     * **INTAKING** — a ball is within radius ``R`` of a robot AND lies in the robot's
       direction of travel (a velocity dot-product gate, not mere proximity).
     * **HELD**     — a ball's centre is inside the robot's bounding box.
     * **SHOOTING** — a ball transitions from INSIDE the robot box to LEAVING it at high
       speed (a real launch), discriminated from a slow drop / loss of grip by exit speed.
     * **SCORING**  — a (post-shot) ball enters the bounds of the COLLECTOR BIN -> award the
       FUEL its point value to the shooting robot's alliance. A score is **NEVER** counted
       while the ball is still inside the robot bounding box (the user's hard rule).

   This is an IMAGE/BBOX-space state machine — complementary to
   ``refine/scoring_events.ScoringEventTracker``, which does the same lifecycle in FIELD
   METERS via a size-scaled radius and a HUB ``in_hub`` circle. ``PhaseTracker`` keeps the
   field-meter tracker as its scoring *authority* (held-count, shot accuracy) and layers the
   bbox signals here on top for the cases the user called out (occlusion-robust HELD, a
   motion-gated INTAKING, and a bin-entry SCORE that is exact in pixels for a fixed camera).

Honest about the broadcast angle
--------------------------------
From this elevated wide shot the HUB *opening* is small and is frequently occluded by balls
in flight, the structure itself, and crowd behind it. Bin-entry SCORING from video is
therefore the least-reliable of the four signals; the authoritative per-alliance score is
still the overlay FUEL counter (``ScoringEventTracker.note_overlay_score``). ``BinScorer``
reports a bin-entry score as a *candidate* with its confidence and never claims it is the
official count. See ``validate`` / the saved annotated frames for measured behaviour.

Pure-logic: imports no cv2 / numpy / torch at module load (numpy is imported lazily and only
inside ``auto_locate``), so it parses and unit-tests on a bare interpreter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple

from frcscout.io.config import seconds_to_frames
from frcscout.io.types import Alliance, Event, FrameState, Track
# Reuse the vetted pure-python point-in-polygon + fraction scaler from hub_regions
# (do NOT reimplement geometry — this is the same test the HUB image localizer uses).
from frcscout.refine.hub_regions import point_in_polygon, scale_fraction_polygon

Point = Tuple[float, float]
Polygon = list

ALLIANCES: Tuple[str, str] = ("blue", "red")

# ─────────────────────────────────────────────────────────────────────────────
# MEASURED HUB top-opening polygons for the qual59 broadcast (1920×1080, fixed cam).
# Vertices are FRACTIONS of (W, H). Traced by eye against a fine fraction grid on real
# frames (mid-teleop f7340 + verified stable from end-of-AUTO f1761 through late-teleop
# f11200; blue-HUB patch NCC f1761↔f11200 = 0.82 -> camera effectively static). These are
# the *opening/funnel* of each HUB (where FUEL enters), not the whole structure.
# Override per-video via field-config key ``collector_bin_image_regions`` (same shape).
# !! footage-specific !! re-trace for a different camera/venue.
_QUAL59_BIN_REGIONS: dict[str, list] = {
    # Traced against a labeled 0.02 fraction grid onto the actual sloped FUNNEL surfaces
    # (where FUEL enters), as parallelograms receding from the centre camera. blue tilts
    # strongly up-to-the-right; red is the red sloped top (orange ball-entry glow ~0.76,0.31).
    "blue": [(0.206, 0.346), (0.286, 0.324), (0.290, 0.367), (0.212, 0.387)],
    "red":  [(0.710, 0.330), (0.816, 0.320), (0.822, 0.348), (0.706, 0.347)],
}

# Manual point value for one FUEL scored in an ACTIVE hub (Table 6-4, §6.5.3, p.47).
# Same in AUTO and TELEOP. Inactive hub = 0. Config-overridable but defaulted from the manual.
FUEL_POINTS_ACTIVE = 1
FUEL_POINTS_INACTIVE = 0


# ─────────────────────────────────────────────────────────────────────────────
# CollectorBin — static HUB localization (image polygons, optional field upgrade)
# ─────────────────────────────────────────────────────────────────────────────
class CollectorBin:
    """Localize the static COLLECTOR BIN (REBUILT HUB) openings in image space.

    Per-alliance opening polygons live as (W,H) fractions; ``which_bin(px)`` / ``in_bin``
    scale them to the live frame and run a point-in-polygon test. With a field model + a
    valid :class:`CameraPose`, the test upgrades to the exact field-coords HUB circle
    (``FieldModel.in_hub``), which survives camera pan/zoom.

    Parameters
    ----------
    regions : dict | None
        ``{alliance: [(fx,fy), ...]}`` opening polygons in (W,H) fractions. Defaults to the
        measured qual59 regions. Pass your own (or a field-config block) for other footage.
    frame_wh : (W,H) | None
        Frame size used to scale the fractional polygons to pixels; may be given per call.
    field_model : object | None
        Anything exposing ``in_hub(field_xy, alliance, tol_m)`` (+ ``image_to_field(px,pose)``
        for the pixel path). Enables the field-coords (Mode-B) test when a valid pose exists.
    """

    def __init__(
        self,
        regions: Optional[dict] = None,
        frame_wh: Optional[Tuple[int, int]] = None,
        field_model: object | None = None,
    ):
        src = regions if regions is not None else _QUAL59_BIN_REGIONS
        self._frac: dict[str, list] = {}
        for ally, verts in (src or {}).items():
            self._frac[str(ally)] = [(float(v[0]), float(v[1])) for v in verts]
        self.frame_wh = frame_wh
        self.field_model = field_model

    # ---- construction helpers ----
    @classmethod
    def from_config(cls, config, frame_wh=None, field_model=None,
                    key: str = "collector_bin_image_regions") -> "CollectorBin":
        """Build from a Config/dict, reading ``collector_bin_image_regions`` if present;
        falls back to the measured defaults when the key is absent."""
        regions = None
        if config is not None:
            try:
                regions = config.get(key) if hasattr(config, "get") else None
            except Exception:
                regions = None
            if isinstance(config, dict) and not regions:
                regions = config.get(key)
        return cls(regions=regions or None, frame_wh=frame_wh, field_model=field_model)

    def alliances(self) -> list:
        """Alliances that have a usable (>=3-vertex) opening polygon configured."""
        return [a for a, p in self._frac.items() if len(p) >= 3]

    def has_region(self, alliance: Optional[str] = None) -> bool:
        if alliance is None:
            return any(len(p) >= 3 for p in self._frac.values())
        return len(self._frac.get(alliance, [])) >= 3

    # ---- pixel polygon access (for drawing + the image test) ----
    def pixel_polygon(self, alliance: str,
                      frame_wh: Optional[Tuple[int, int]] = None) -> Optional[list]:
        """Opening polygon for ``alliance`` in PIXELS, or None if unavailable."""
        frac = self._frac.get(alliance)
        if not frac or len(frac) < 3:
            return None
        wh = frame_wh if frame_wh is not None else self.frame_wh
        if wh is None:
            return None
        return scale_fraction_polygon(frac, int(wh[0]), int(wh[1]))

    def polygons_px(self, frame_wh: Optional[Tuple[int, int]] = None) -> dict:
        """All opening polygons in PIXELS keyed by alliance (for the renderer)."""
        out = {}
        for a in self._frac:
            p = self.pixel_polygon(a, frame_wh)
            if p is not None:
                out[a] = p
        return out

    # ---- the core membership test ----
    def in_bin(self, px: Point, alliance: str,
               pose=None, frame_wh: Optional[Tuple[int, int]] = None,
               tol_px: float = 0.0, tol_m: float = 0.0) -> bool:
        """Is image point ``px`` inside ``alliance``'s bin opening?

        Field-coords path when a field model + a valid pose exist (unproject ``px`` to
        meters, defer to ``FieldModel.in_hub``); else the scaled-polygon image test.
        """
        if self.field_model is not None and _pose_valid(pose):
            try:
                fxy = self.field_model.image_to_field((float(px[0]), float(px[1])), pose)
                return bool(self.field_model.in_hub(fxy, alliance, tol_m=tol_m))
            except Exception:
                pass  # fall through to the image test
        poly = self.pixel_polygon(alliance, frame_wh)
        if poly is None:
            return False
        return point_in_polygon(px, poly, tol=tol_px)

    def which_bin(self, px: Point, pose=None,
                  frame_wh: Optional[Tuple[int, int]] = None,
                  tol_px: float = 0.0, tol_m: float = 0.0) -> Optional[str]:
        """Which alliance bin (if any) contains ``px``. Returns the alliance, or None."""
        for a in self._frac:
            if self.in_bin(px, a, pose=pose, frame_wh=frame_wh, tol_px=tol_px, tol_m=tol_m):
                return a
        return None

    # ---- optional one-time auto-locate (honest: coarse colour heuristic) ----
    def auto_locate(self, frame, save_overlay: Optional[str] = None) -> dict:
        """Best-effort one-time snap of each opening polygon to the colored HUB structure
        in ``frame`` (BGR ndarray). COARSE heuristic — the measured fractions are the
        reliable path on this fixed shot; this only refines/sanity-checks them and is OFF
        by default. Returns the (possibly-updated) per-alliance fraction polygons; on any
        failure it leaves the existing regions untouched and returns them.

        It finds the largest blue / red structure blob in each HUB's half of the frame and
        places a small opening polygon at the TOP of that blob (the opening sits atop the
        structure). numpy + cv2 are imported HERE only.
        """
        try:
            import numpy as np  # noqa: F401  (lazy heavy import; never at module load)
            import cv2
        except Exception:
            return dict(self._frac)
        if frame is None:
            return dict(self._frac)
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bands = {
            "blue": (cv2.inRange(hsv, (100, 90, 40), (130, 255, 255)), (0.05, 0.45)),
            "red": ((cv2.inRange(hsv, (0, 110, 60), (10, 255, 255))
                     | cv2.inRange(hsv, (165, 110, 60), (180, 255, 255))), (0.55, 0.95)),
        }
        updated = dict(self._frac)
        for al, (mask, (xlo, xhi)) in bands.items():
            band = mask.copy()
            band[:, : int(xlo * w)] = 0
            band[:, int(xhi * w):] = 0
            band[: int(0.25 * h), :] = 0  # ignore the very top (HUD/crowd)
            band[int(0.6 * h):, :] = 0    # ignore the lower pane
            cnts, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) < 0.0008 * w * h:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            # opening = a band across the TOP ~18% of the structure
            ox0, ox1 = bx + 0.04 * bw, bx + 0.96 * bw
            oy0, oy1 = by, by + 0.20 * bh
            updated[al] = [(ox0 / w, oy0 / h), (ox1 / w, oy0 / h),
                           (ox1 / w, oy1 / h), (ox0 / w, oy1 / h)]
        self._frac = {k: [(float(a), float(b)) for a, b in v] for k, v in updated.items()}
        if save_overlay is not None:
            try:
                vis = frame.copy()
                for al in self._frac:
                    poly = self.pixel_polygon(al, (w, h))
                    if poly:
                        import numpy as _np2
                        pts = _np2.array([(int(x), int(y)) for x, y in poly], dtype=int)
                        cv2.polylines(vis, [pts.reshape(-1, 1, 2)], True,
                                      (255, 80, 0) if al == "blue" else (0, 0, 220), 3)
                cv2.imwrite(save_overlay, vis)
            except Exception:
                pass
        return dict(self._frac)


# ─────────────────────────────────────────────────────────────────────────────
# Per-ball interaction lifecycle (the bbox/velocity state machine)
# ─────────────────────────────────────────────────────────────────────────────
_FREE = "FREE"
_INTAKING = "INTAKING"
_HELD = "HELD"
_SHOT = "SHOT"          # left the box fast — a real launch, in flight toward a bin
_SCORED = "SCORED"
_LOST = "LOST"          # left the box slowly (drop / loss of grip) — NOT a shot

INTERACTION_STATES = (_FREE, _INTAKING, _HELD, _SHOT, _SCORED, _LOST)


@dataclass
class _Ball:
    """Mutable per-ball interaction record (image/px space)."""
    tid: int
    state: str = _FREE
    robot_id: Optional[int] = None        # robot it is held by / was last held by
    alliance: Optional[Alliance] = None   # that robot's alliance (for scoring credit)
    # px history for speed/dwell
    last_px: Optional[Point] = None
    last_t_s: float = 0.0
    last_seen_frame: int = 0
    held_frames: int = 0                  # consecutive frames inside the box
    intaking_frames: int = 0              # consecutive frames in the intake cone
    exit_speed_px_s: float = 0.0          # speed at the moment it left the box
    shot_frame: int = -1                  # frame the shot was detected
    scored: bool = False


@dataclass
class BinScoreEvent:
    """A bin-entry SCORING candidate (NOT the authoritative count — see module docs)."""
    t_s: float
    frame_idx: int
    alliance: Optional[Alliance]
    robot_id: Optional[int]
    ball_id: int
    points: int
    hub_active: bool
    hub_active_known: bool
    confidence: float

    def to_event(self) -> Event:
        """As a canonical :class:`~frcscout.io.types.Event` (``type='score'``)."""
        return Event(
            t_s=self.t_s, frame_idx=self.frame_idx, type="score",
            alliance=self.alliance, robot_id=self.robot_id, confidence=self.confidence,
            detail={"kind": "bin_score", "ball_id": self.ball_id, "points": self.points,
                    "hub_active": self.hub_active, "hub_active_known": self.hub_active_known,
                    "source": "collector_bin_image"})


# ─────────────────────────────────────────────────────────────────────────────
# defaults — all timing in SECONDS, all distance in PIXELS (image-space machine)
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    # INTAKING catchment radius. Scaled to the robot's bbox so a bigger/closer robot
    # reaches further: R = max(intake_k_box * robot_bbox_diag_px, intake_floor_px).
    "intake_k_box": 0.9,
    "intake_floor_px": 40.0,
    # the velocity DOT-PRODUCT gate: the ball must lie within this half-angle (radians)
    # of the robot's heading (direction of travel). cos(gate) is the dot threshold.
    "intake_cone_halfangle_rad": math.radians(75.0),
    # the robot must actually be MOVING (px/s) for the direction gate to apply; below this
    # a (near-)stationary robot has no travel direction, so proximity alone gates intake.
    "intake_min_robot_speed_px_s": 8.0,
    # sustained frames in the cone before INTAKING latches (debounce one-frame flicker).
    "intake_confirm_s": 0.10,

    # min seconds between two intakes credited to one robot (dedup; a dense ball field
    # would otherwise re-trigger intake every frame as litter grazes the box edge).
    "intake_dedup_s": 0.5,

    # CONSUMPTION GATE (the dense-field INTAKING fix). When True, the per-frame INTAKING
    # *signal* (``robot_signals()[rid]["intaking"]``) is NOT raised by mere cone-membership
    # (a ball within radius + in the heading) — in a 100+ ball field that is true for nearly
    # every robot every frame, which is noise, not intaking. Instead the signal is raised
    # ONLY on a confirmed CONSUMPTION: a ball that was converging in the cone then crosses
    # INTO the robot's box (a real pickup transition), or vanishes right at the box while it
    # was being intaken by this robot. The signal then LATCHES for ``intake_signal_hold_s``
    # so the brief event is visible, and re-arms only after ``intake_dedup_s`` (so it cannot
    # fire every frame). Set False to restore the legacy proximity-cone behaviour (clean/
    # synthetic input where every loose ball near a moving robot really is an intake attempt).
    "require_consumption_for_intake": True,
    # how long (s) a confirmed-consumption INTAKING signal stays raised after the pickup,
    # so a 1-frame transition is actually visible to the phase layer / overlay.
    "intake_signal_hold_s": 0.30,
    # Under the consumption gate, also require the consuming robot to be MOVING (>=
    # ``intake_min_robot_speed_px_s``) for the INTAKING *signal* to fire: a real intake is
    # ACTIVE — the robot drives onto / sucks in the ball. A robot PARKED in a dense pile has
    # loose balls drifting into its bbox every frame; that is sitting in litter, not intaking.
    # This gate removes those parked-in-pile false positives. (The possession TALLY still
    # counts — the robot does gain the ball — but the PHASE does not read INTAKING.)
    "intake_require_robot_motion": True,

    # HELD: a ball centre inside the robot box. Box may be padded by this px margin
    # (a held ball rides the bumper edge); sustained this long to confirm (debounce).
    "held_pad_px": 6.0,
    "held_confirm_s": 0.05,

    # SHOOTING vs LOSS: when a held ball leaves the box, its exit SPEED (px/s) decides.
    # >= shot_min_exit_speed_px_s -> a real SHOT; below -> a slow drop / loss of grip.
    "shot_min_exit_speed_px_s": 220.0,
    # a ball must have been HELD at least this long before a fast exit counts as a SHOT —
    # filters litter that merely GRAZES a box edge for a frame (key on dense broadcast).
    "min_held_s_for_shot": 0.0,
    # ROBUSTNESS (broadcast dense-field): require the exiting ball to be a CONFIRMED flight
    # track (is_flight) for the exit to count as a SHOT. On clean/synthetic input leave
    # False (the exit-speed gate alone suffices); on a dense real field set True so ball
    # id-churn in the pile cannot masquerade as shots. A shot still needs the speed gate.
    "require_flight_for_shot": False,
    # after a SHOT, how long (s) the ball may travel before a bin-entry counts as ITS score
    # (a fuel arc is short). Past this the shot is abandoned (a miss / lost track).
    "shot_to_score_max_s": 2.5,

    # SCORING: padding (px) on the bin polygon for the entry test (a ball on the lip counts).
    "bin_tol_px": 6.0,
    # a ball must NOT be inside any robot box to score (the user's hard rule). Extra guard:
    # require the scoring ball to be at least this px clear of its shooter's box centre —
    # belt-and-suspenders so a ball resting on a robot near the bin can't false-score.
    # (0 disables the extra distance guard; the inside-box veto always applies.)
    "score_clear_px": 0.0,

    # housekeeping: drop a ball record unseen this long.
    "ball_ttl_s": 1.5,
    # confidence stamped on an image-polygon bin score (field-coords path scores higher).
    "image_score_confidence": 0.55,
    "field_score_confidence": 0.75,
}


class BinScorer:
    """Ball<->robot<->bin interaction state machine in IMAGE/BBOX space (the user's spec).

    Call :meth:`step(frame_state)` once per frame. It updates each ball's state
    (FREE→INTAKING→HELD→SHOT→SCORED / LOST), and when a post-shot ball enters the
    COLLECTOR BIN it emits a :class:`BinScoreEvent` crediting the shooter's alliance the
    FUEL point value — but only when the ball is OUTSIDE every robot box (the hard rule).

    State for the renderer/PhaseTracker is exposed per frame:
      * :meth:`ball_states` -> ``{ball_id: state}``
      * :meth:`robot_signals` -> ``{robot_id: {"intaking", "held_balls", "shooting"}}``
      * :meth:`tallies` -> per-alliance ``{intakes, shots, losses, bin_scores, points}``.

    ``hub_active`` per alliance may be supplied via :meth:`set_hub_active` (from HUB lights
    or the shift schedule). Unknown -> assumed active, flagged ``hub_active_known: False``.
    """

    def __init__(self, bin_loc: Optional[CollectorBin] = None,
                 cfg: Optional[dict] = None, fps: float = 60.0,
                 fuel_points_active: int = FUEL_POINTS_ACTIVE,
                 fuel_points_inactive: int = FUEL_POINTS_INACTIVE):
        self.bin = bin_loc if bin_loc is not None else CollectorBin()
        c = dict(_DEFAULTS)
        c.update(cfg or {})
        self.cfg = c
        self.fps = float(fps)
        self.fuel_pts_active = int(fuel_points_active)
        self.fuel_pts_inactive = int(fuel_points_inactive)

        self.intake_k_box = float(c["intake_k_box"])
        self.intake_floor_px = float(c["intake_floor_px"])
        self.cone_half = float(c["intake_cone_halfangle_rad"])
        self.intake_min_speed = float(c["intake_min_robot_speed_px_s"])
        self.held_pad = float(c["held_pad_px"])
        self.shot_min_exit = float(c["shot_min_exit_speed_px_s"])
        self.require_flight_for_shot = bool(c["require_flight_for_shot"])
        self.require_consumption_for_intake = bool(c["require_consumption_for_intake"])
        self.intake_require_robot_motion = bool(c["intake_require_robot_motion"])
        self.bin_tol_px = float(c["bin_tol_px"])
        self.score_clear_px = float(c["score_clear_px"])

        self._recompute_windows()

        self._balls: dict[int, _Ball] = {}
        self._hub_active: dict[str, bool] = {}      # alliance -> bool (if known)
        self._frame_wh: Optional[Tuple[int, int]] = None
        self._last_intake_frame: dict[int, int] = {}   # robot -> last credited intake frame
        # robot -> frame index up to which a confirmed-consumption INTAKING signal is held
        # (so a 1-frame pickup transition stays visible for ``intake_signal_hold_s``).
        self._intake_signal_until: dict[int, int] = {}

        # per-alliance tallies
        self.intakes: dict[str, int] = {"blue": 0, "red": 0}
        self.shots: dict[str, int] = {"blue": 0, "red": 0}
        self.losses: dict[str, int] = {"blue": 0, "red": 0}
        self.bin_scores: dict[str, int] = {"blue": 0, "red": 0}
        self.points: dict[str, int] = {"blue": 0, "red": 0}

        # per-frame view (refreshed each step)
        self._intaking_now: set = set()
        self._held_now: dict[int, list] = {}
        self._shooting_now: set = set()
        self._scores_this_frame: list = []

    def _recompute_windows(self) -> None:
        c = self.cfg
        f = self.fps
        self.intake_confirm_frames = seconds_to_frames(c["intake_confirm_s"], f)
        self.intake_dedup_frames = seconds_to_frames(c["intake_dedup_s"], f)
        self.intake_signal_hold_frames = seconds_to_frames(c["intake_signal_hold_s"], f)
        self.held_confirm_frames = seconds_to_frames(c["held_confirm_s"], f)
        self.min_held_frames_for_shot = (seconds_to_frames(c["min_held_s_for_shot"], f)
                                         if c["min_held_s_for_shot"] > 0 else 0)
        self.shot_to_score_max_frames = seconds_to_frames(c["shot_to_score_max_s"], f)
        self.ball_ttl_frames = seconds_to_frames(c["ball_ttl_s"], f)

    # ───────────────────────────── public knobs ──────────────────────────────
    def set_frame_wh(self, w: int, h: int) -> None:
        """Tell the scorer the frame size so the bin's fractional polygons scale to px."""
        self._frame_wh = (int(w), int(h))
        if self.bin.frame_wh is None:
            self.bin.frame_wh = self._frame_wh

    def set_hub_active(self, alliance: str, active: bool) -> None:
        """Set whether ``alliance``'s HUB is currently active (from lights / shift schedule).
        FUEL into an inactive HUB scores 0 (manual §6.4.1 / Table 6-4)."""
        self._hub_active[alliance] = bool(active)

    # ───────────────────────────── geometry helpers ──────────────────────────
    @staticmethod
    def _box_contains(box, px: Point, pad: float = 0.0) -> bool:
        x1, y1, x2, y2 = box
        lo_x, hi_x = (x1, x2) if x1 <= x2 else (x2, x1)
        lo_y, hi_y = (y1, y2) if y1 <= y2 else (y2, y1)
        return (lo_x - pad) <= px[0] <= (hi_x + pad) and (lo_y - pad) <= px[1] <= (hi_y + pad)

    @staticmethod
    def _box_diag(box) -> float:
        x1, y1, x2, y2 = box
        return math.hypot(x2 - x1, y2 - y1)

    @staticmethod
    def _dist(a: Point, b: Point) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _robot_dir_px(self, r: Track) -> Optional[Point]:
        """Robot travel direction as a unit vector in IMAGE px. We have field velocity
        (m/s) but the interaction is in px; for the *direction* the field-frame velocity
        and the image-frame motion are consistent enough on this near-top-down broadcast
        (the homography is close to a similarity over the play area). Prefer an explicit
        px velocity if a consumer attached one (``vel_px``), else use the field velocity
        direction, else None (stationary)."""
        vpx = getattr(r, "vel_px", None)
        if vpx and (abs(vpx[0]) + abs(vpx[1])) > 1e-6:
            n = math.hypot(vpx[0], vpx[1])
            return (vpx[0] / n, vpx[1] / n)
        v = getattr(r, "vel_mps", None)
        if v and (abs(v[0]) + abs(v[1])) > 1e-6:
            # image y grows DOWNWARD; field +y is up/left. We only need a consistent
            # direction for the cone test, and the dot-product gate is symmetric to a
            # global sign flip on one axis as long as we apply it consistently to the
            # ball-relative vector too — which we do (both from field velocity sign here
            # would be ideal, but the ball offset is measured in px). To keep the gate
            # meaningful we flip vy to image convention.
            n = math.hypot(v[0], v[1])
            return (v[0] / n, -v[1] / n)
        return None

    def _robot_moving(self, r: Optional[Track], dt: float) -> bool:
        """True if ``r`` is moving fast enough to be ACTIVELY intaking (vs parked in a
        pile). When the motion gate is off, always True. A missing robot is treated as
        moving (no info -> don't suppress)."""
        if not self.intake_require_robot_motion:
            return True
        if r is None:
            return True
        return self._robot_speed_px_s(r, dt) >= self.intake_min_speed

    def _raise_intake_signal(self, rid: int, frame_idx: int) -> None:
        """Raise + LATCH the per-frame INTAKING signal for ``rid`` on a confirmed
        consumption. The latch (``intake_signal_hold_frames``) keeps a 1-frame pickup
        visible to the phase layer; ``robot_signals()`` reads the latch each frame."""
        self._intaking_now.add(rid)
        self._intake_signal_until[rid] = frame_idx + self.intake_signal_hold_frames

    def _robot_speed_px_s(self, r: Track, dt: float) -> float:
        """Approx robot image speed (px/s). Uses an attached ``vel_px`` if present; else
        derives from the centroid step when we have a previous centroid + dt."""
        vpx = getattr(r, "vel_px", None)
        if vpx:
            return math.hypot(vpx[0], vpx[1])
        prev = self._robot_prev_centroid.get(r.tid) if hasattr(self, "_robot_prev_centroid") else None
        if prev is not None and dt > 0:
            return self._dist(prev, r.centroid_px) / dt
        return 0.0

    # ─────────────────────────────── step ────────────────────────────────────
    def step(self, st: FrameState, fps: Optional[float] = None) -> list:
        """Advance the interaction machine one frame; return this frame's BinScoreEvents."""
        if fps is not None and float(fps) != self.fps:
            self.fps = float(fps)
            self._recompute_windows()
        t = st.t_s

        robots = list(st.robots or [])
        by_id = {r.tid: r for r in robots}
        # all balls we can see this frame (inventory + flight; both carry centroid_px)
        balls = list(st.balls_inventory or []) + list(st.balls_flight or [])

        # track previous robot centroids for px-speed derivation
        if not hasattr(self, "_robot_prev_centroid"):
            self._robot_prev_centroid: dict[int, Point] = {}

        self._intaking_now = set()
        self._held_now = {}
        self._shooting_now = set()
        self._scores_this_frame = []
        events: list = []

        seen_ids = set()
        for b in balls:
            seen_ids.add(b.tid)
            bs = self._balls.get(b.tid)
            if bs is None:
                bs = _Ball(tid=b.tid, last_px=b.centroid_px, last_t_s=t,
                           last_seen_frame=st.frame_idx)
                self._balls[b.tid] = bs
            ev = self._advance_ball(st, bs, b, robots, by_id)
            if ev is not None:
                events.append(ev)
                self._scores_this_frame.append(ev)
            bs.last_px = b.centroid_px
            bs.last_t_s = t
            bs.last_seen_frame = st.frame_idx

        # balls that VANISHED while HELD stay HELD (a swallowed ball is invisible — same
        # honesty as scoring_events); FREE/INTAKING balls that disappear are pruned by ttl.
        for tid, bs in list(self._balls.items()):
            if tid in seen_ids:
                continue
            gap = st.frame_idx - bs.last_seen_frame
            if bs.state == _INTAKING and gap == 1 and bs.robot_id is not None:
                # CONSUMPTION-by-vanish: a ball that was converging in this robot's cone and
                # then DISAPPEARED right at it (one-frame loss, where it sat) was very likely
                # swallowed — the strongest "vanished INTO the robot" evidence (mirrors the
                # field-meter scorer's vanish_grace intake). Promote to HELD + credit once
                # (dedup-gated) and raise the INTAKING signal.
                rid = bs.robot_id
                r = by_id.get(rid)
                last = self._last_intake_frame.get(rid, -(10 ** 9))
                if (st.frame_idx - last) >= self.intake_dedup_frames:
                    bs.state = _HELD
                    bs.alliance = getattr(r, "alliance", bs.alliance)
                    if bs.alliance in self.intakes:
                        self.intakes[bs.alliance] += 1
                    self._last_intake_frame[rid] = st.frame_idx
                    # active-intake motion gate (same as the cone->box path)
                    if self._robot_moving(r, 1.0 / self.fps):
                        self._raise_intake_signal(rid, st.frame_idx)
                    self._held_now.setdefault(rid, []).append(tid)
                else:
                    # already credited an intake very recently; just keep it held & invisible
                    bs.state = _HELD
                    self._held_now.setdefault(rid, []).append(tid)
            elif bs.state in (_FREE, _INTAKING, _SCORED, _LOST) and gap >= self.ball_ttl_frames:
                del self._balls[tid]
            elif bs.state == _SHOT and gap >= self.shot_to_score_max_frames:
                # shot whose ball was lost before reaching a bin -> abandon (a miss/occlusion)
                del self._balls[tid]
            elif bs.state == _HELD:
                # still held (invisible); surface it so HOLDING persists
                if bs.robot_id is not None:
                    self._held_now.setdefault(bs.robot_id, []).append(tid)

        # apply the INTAKING signal LATCH: a confirmed consumption stays "intaking" for
        # ``intake_signal_hold_s`` so a 1-frame pickup is visible (and never re-fires every
        # frame — only a NEW dedup-gated consumption re-raises it).
        if self.require_consumption_for_intake:
            for rid, until in self._intake_signal_until.items():
                if st.frame_idx <= until:
                    self._intaking_now.add(rid)

        # refresh robot prev-centroids
        for r in robots:
            self._robot_prev_centroid[r.tid] = r.centroid_px

        return events

    # ───────────────────────── per-ball transitions ──────────────────────────
    def _advance_ball(self, st: FrameState, bs: _Ball, b: Track,
                      robots: list, by_id: dict) -> Optional[BinScoreEvent]:
        t = st.t_s
        dt = max(1e-3, t - bs.last_t_s) if bs.last_t_s else (1.0 / self.fps)
        px = b.centroid_px

        # ---- TERMINAL states: a SCORED or LOST ball is inert. A SCORED ball is done; a
        #      LOST (slow-drop) ball returns to the FREE pool only if it is picked up again
        #      (re-enters a box) — until then it is just a loose ball, not re-credited. ----
        if bs.state == _SCORED:
            return None
        if bs.state == _LOST:
            # allow recovery: if it lands inside a robot box again it can be re-held.
            rid = self._robot_box_containing(px, robots)
            if rid is None:
                return None
            # fall through to the FREE/held re-acquire logic below by resetting to FREE
            bs.state = _FREE

        # which robot box currently contains the ball (if any)?
        inside_rid = self._robot_box_containing(px, robots)

        # ---- SHOT in flight: look for bin entry (and ENFORCE the not-inside-box rule) ----
        if bs.state == _SHOT:
            if inside_rid is not None:
                # ball went back inside a robot box: not a scoring path — revert to HELD.
                bs.state = _HELD
                bs.robot_id = inside_rid
                r = by_id.get(inside_rid)
                bs.alliance = getattr(r, "alliance", bs.alliance)
                self._held_now.setdefault(inside_rid, []).append(bs.tid)
                return None
            self._shooting_now.add(bs.robot_id) if bs.robot_id is not None else None
            ev = self._try_bin_score(st, bs, px)
            if ev is not None:
                return ev
            # timed out?
            if (st.frame_idx - bs.shot_frame) >= self.shot_to_score_max_frames:
                bs.state = _FREE
                bs.robot_id = None
            return None

        # ---- HELD: is the ball still inside the box? ----
        if bs.state == _HELD:
            r = by_id.get(bs.robot_id)
            if r is not None and self._box_contains(r.xyxy, px, pad=self.held_pad):
                bs.held_frames += 1
                self._held_now.setdefault(bs.robot_id, []).append(bs.tid)
                return None
            # the ball LEFT the box this frame -> SHOT (fast) or LOSS (slow), by exit speed.
            exit_speed = self._dist(px, bs.last_px) / dt if bs.last_px else 0.0
            bs.exit_speed_px_s = exit_speed
            # robustness gates (matter on a dense real field, no-ops on clean input):
            #  * the ball must have been HELD for a minimum dwell (filters litter grazing a
            #    box edge for one frame);  * optionally the exiting ball must be a confirmed
            #    flight track (is_flight) so ball-id churn in the pile can't fake a shot.
            held_long_enough = bs.held_frames >= self.min_held_frames_for_shot
            flight_ok = (not self.require_flight_for_shot) or bool(getattr(b, "is_flight", False))
            if exit_speed >= self.shot_min_exit and held_long_enough and flight_ok:
                bs.state = _SHOT
                bs.shot_frame = st.frame_idx
                if bs.alliance in self.shots:
                    self.shots[bs.alliance] += 1
                self._shooting_now.add(bs.robot_id) if bs.robot_id is not None else None
                # it may already be over the bin on this very frame (fast arc) — but only
                # score if it is OUTSIDE every robot box (checked inside _try_bin_score).
                ev = self._try_bin_score(st, bs, px)
                return ev
            else:
                # left the box but NOT a credited shot (slow, too-brief, or unconfirmed):
                # a LOSS of grip. Only tally a loss if it was genuinely held a moment.
                bs.state = _LOST
                if held_long_enough and bs.alliance in self.losses:
                    self.losses[bs.alliance] += 1
                bs.robot_id = None
                return None

        # ---- FREE / INTAKING / LOST: can the ball get intaken or held? ----
        # HELD takes priority: a ball centre inside a robot box is held.
        if inside_rid is not None:
            bs.held_frames += 1
            if bs.held_frames >= self.held_confirm_frames:
                prev_state = bs.state
                # CONSUMPTION evidence: a ball CROSSING from a tracked approach (it was
                # converging in this robot's intake cone, ``_INTAKING``) INTO the box is a
                # real pickup. A ball the box merely overlapped while the robot drove through
                # loose litter was never ``_INTAKING`` for this robot -> it is HELD/carried,
                # but it is NOT a fresh consumption event and must not light INTAKING.
                was_converging = (prev_state == _INTAKING and bs.robot_id == inside_rid)
                bs.state = _HELD
                bs.robot_id = inside_rid
                r = by_id.get(inside_rid)
                bs.alliance = getattr(r, "alliance", None)
                # a fresh possession (was not already this robot's held/intaking) counts an
                # intake — but dedup per robot so a dense ball field can't re-credit every
                # frame as litter grazes the box edge.
                last = self._last_intake_frame.get(inside_rid, -(10 ** 9))
                fresh = ((st.frame_idx - last) >= self.intake_dedup_frames)
                if prev_state in (_FREE, _LOST, _INTAKING) and bs.alliance in self.intakes and fresh:
                    self.intakes[bs.alliance] += 1
                    self._last_intake_frame[inside_rid] = st.frame_idx
                    # raise (and latch) the per-frame INTAKING signal ONLY on confirmed
                    # consumption: the ball crossed from this robot's tracked approach
                    # (``was_converging``) into its box, AND (optionally) the robot is moving
                    # (an ACTIVE intake, not a robot parked in a pile). When the gate is off,
                    # the cone path below raises the signal directly (legacy proximity).
                    if self.require_consumption_for_intake and was_converging \
                            and self._robot_moving(r, dt):
                        self._raise_intake_signal(inside_rid, st.frame_idx)
                self._held_now.setdefault(inside_rid, []).append(bs.tid)
            return None
        else:
            bs.held_frames = 0

        # INTAKING — within scaled radius AND in the robot's travel direction (dot gate).
        cand = self._intake_candidate(px, robots, dt)
        if cand is not None:
            bs.intaking_frames += 1
            if bs.intaking_frames >= self.intake_confirm_frames:
                bs.state = _INTAKING
                bs.robot_id = cand.tid
                # The cone alone lights INTAKING only in LEGACY mode. With the consumption
                # gate on, cone-membership is just a tracked approach (used to recognise the
                # later cross-into-box pickup); it does NOT raise the signal — that is what
                # killed the dense-field over-fire (a ball is always in some robot's cone).
                if not self.require_consumption_for_intake:
                    self._intaking_now.add(cand.tid)
        else:
            bs.intaking_frames = 0
            if bs.state == _INTAKING:
                # an approach that fizzled (ball drifted out of the cone without being taken).
                # If it VANISHES next frame near the robot the step() vanish handler credits a
                # consumption; here it simply reverts to a free/loose ball.
                bs.state = _FREE
                bs.robot_id = None
        return None

    def _robot_box_containing(self, px: Point, robots: list) -> Optional[int]:
        """The id of a robot whose (padded) bbox contains ``px`` — the HELD test. If
        several overlap, the SMALLEST box wins (the nearer/owning robot)."""
        best, best_area = None, math.inf
        for r in robots:
            if self._box_contains(r.xyxy, px, pad=self.held_pad):
                x1, y1, x2, y2 = r.xyxy
                area = abs((x2 - x1) * (y2 - y1))
                if area < best_area:
                    best, best_area = r.tid, area
        return best

    def _intake_candidate(self, px: Point, robots: list, dt: float) -> Optional[Track]:
        """The robot (if any) currently intaking the ball at ``px``: within the scaled
        radius AND the ball lies in the robot's direction of travel (velocity dot gate).

        A (near-)stationary robot has no travel direction -> proximity alone qualifies."""
        cos_gate = math.cos(self.cone_half)
        best, best_d = None, math.inf
        for r in robots:
            radius = max(self.intake_k_box * self._box_diag(r.xyxy), self.intake_floor_px)
            d = self._dist(px, r.centroid_px)
            if d > radius or d >= best_d:
                continue
            speed = self._robot_speed_px_s(r, dt)
            if speed >= self.intake_min_speed:
                dirv = self._robot_dir_px(r)
                if dirv is not None:
                    # unit vector robot -> ball, in image px
                    bx = px[0] - r.centroid_px[0]
                    by = px[1] - r.centroid_px[1]
                    bn = math.hypot(bx, by)
                    if bn > 1e-6:
                        dot = (bx / bn) * dirv[0] + (by / bn) * dirv[1]
                        if dot < cos_gate:
                            continue  # ball is NOT in the direction the robot is moving
            best, best_d = r, d
        return best

    def _try_bin_score(self, st: FrameState, bs: _Ball, px: Point) -> Optional[BinScoreEvent]:
        """If the (in-flight, post-shot) ball at ``px`` is inside a bin AND outside every
        robot box, award the FUEL points to its shooter's alliance and emit the event.

        The not-inside-any-box check is the user's hard rule: NEVER count a score while the
        ball is still inside a robot bounding box."""
        # hard rule: never score a ball that is inside any robot box this frame.
        if self._robot_box_containing(px, st.robots or []) is not None:
            return None
        frame_wh = self._frame_wh
        if frame_wh is None and st.robots is None:
            return None
        pose = getattr(st, "pose", None)
        used_field = (self.bin.field_model is not None and _pose_valid(pose))
        which = self.bin.which_bin(px, pose=pose, frame_wh=frame_wh,
                                   tol_px=self.bin_tol_px, tol_m=self.cfg.get("bin_tol_m", 0.3))
        if which is None:
            return None
        # extra distance guard (optional): the ball must be clear of its shooter's box.
        if self.score_clear_px > 0 and bs.robot_id is not None:
            r = next((x for x in (st.robots or []) if x.tid == bs.robot_id), None)
            if r is not None and self._dist(px, r.centroid_px) < self.score_clear_px:
                return None

        # score it. credit the SHOOTER's alliance (bs.alliance); the bin's alliance is which.
        credit = bs.alliance or which
        active, known = self._hub_state(credit)
        pts = self.fuel_pts_active if active else self.fuel_pts_inactive
        bs.state = _SCORED
        bs.scored = True
        if credit in self.bin_scores:
            self.bin_scores[credit] += 1
            self.points[credit] += pts
        conf = (self.cfg["field_score_confidence"] if used_field
                else self.cfg["image_score_confidence"])
        return BinScoreEvent(
            t_s=st.t_s, frame_idx=st.frame_idx, alliance=credit, robot_id=bs.robot_id,
            ball_id=bs.tid, points=pts, hub_active=active, hub_active_known=known,
            confidence=conf)

    def _hub_state(self, alliance: Optional[str]) -> Tuple[bool, bool]:
        """(active, known) for ``alliance``'s HUB. Unknown -> assume active but flag it so
        a 0-pt inactive score is never silently counted as 1 (manual §6.4.1)."""
        if alliance in self._hub_active:
            return self._hub_active[alliance], True
        return True, False

    # ───────────────────────────── per-frame views ───────────────────────────
    def ball_states(self) -> dict:
        """``{ball_id: state}`` for every tracked ball (FREE/INTAKING/HELD/SHOT/SCORED/LOST)."""
        return {tid: bs.state for tid, bs in self._balls.items()}

    def robot_signals(self) -> dict:
        """Per-robot interaction signals THIS frame:
        ``{robot_id: {"intaking": bool, "held_balls": [ids], "shooting": bool}}``."""
        out: dict = {}
        rids = set(self._intaking_now) | set(self._held_now) | set(self._shooting_now)
        for rid in rids:
            out[rid] = {
                "intaking": rid in self._intaking_now,
                "held_balls": list(self._held_now.get(rid, [])),
                "shooting": rid in self._shooting_now,
            }
        return out

    def scores_this_frame(self) -> list:
        """The BinScoreEvents emitted on the latest :meth:`step` (also returned by it)."""
        return list(self._scores_this_frame)

    def tallies(self) -> dict:
        """Per-alliance cumulative tallies. ``points`` honors active/inactive HUB gating
        (manual Table 6-4): an inactive-HUB score adds 0 to points but still counts in
        ``bin_scores`` (a tracked-but-wasted shot). JSON-serializable."""
        return {
            al: {
                "intakes": self.intakes.get(al, 0),
                "shots": self.shots.get(al, 0),
                "losses": self.losses.get(al, 0),
                "bin_scores": self.bin_scores.get(al, 0),
                "points": self.points.get(al, 0),
            }
            for al in ("blue", "red")
        }


# ─────────────────────────────────────────────────────────────────────────────
def _pose_valid(pose) -> bool:
    return pose is not None and bool(getattr(pose, "valid", False))


__all__ = [
    "CollectorBin",
    "BinScorer",
    "BinScoreEvent",
    "INTERACTION_STATES",
    "FUEL_POINTS_ACTIVE",
    "FUEL_POINTS_INACTIVE",
]
