"""MatchRunner — the scouting integration layer on top of the validated engine.

This is the CENTERPIECE wiring: it takes the engine's per-frame
:class:`~frcscout.io.types.FrameState` (detection -> ByteTrack output, up to ~32 raw
robot ids churning frame to frame) and refines it into the scouting story the demo and
the match files care about:

    raw FrameState.robots (ByteTrack)                       ── engine ──┐
        │  (a) convert to Detection + per-det alliance                  │
        ▼                                                               │
    SixRobotTracker.update()  ──► <=6 STABLE tracks (the 6-robot prior) │  process()
        │  (b) OVERWRITE st.robots with these <=6 as Track objects      │
        │  (c) attach team numbers where bumper OCR resolves them       │
        ▼                                                               │
    PhaseTracker.update(st)   ──► {robot_id: {phase, held, metrics...}} ┘
        │  (embeds ScoringEventTracker — the single scoring authority)
        ▼
    finalize()/export()       ──► match_<key>.json + scouting_<key>.csv  (write_match)

Decoupled from video IO so it is unit-testable with synthetic FrameStates: ``process``
takes a FrameState and returns plain dicts; ``run_video`` is a thin driver that loops an
engine over frames, renders, and writes an mp4 (cv2 imported LAZILY there, never at
module import). ``finalize`` builds the ``write_match`` record straight from
``PhaseTracker.snapshot()`` / ``metrics()`` — it does NOT re-derive any scoring (that
would double-count the embedded ScoringEventTracker).

Honesty about gated metrics is preserved end to end: PhaseTracker emits the
``{value, available, needs}`` envelope for metrics that need a field pose / overlay
(shot_accuracy, defense_time_s, jitter_index, climb), and we carry that envelope through
to the record unchanged — ``export.write_match`` unwraps it for the CSV. A metric whose
input never arrived stays ``available: False``; we never fabricate a 0.
"""
from __future__ import annotations

from typing import Iterable, Optional

from frcscout.io.types import Alliance, Detection, FrameState, OverlayState, Track
from frcscout.refine.phase import PhaseTracker
from frcscout.refine.scoring_events import ScoringEventTracker
from frcscout.refine.sixbot import SixRobotTracker

# identify6 is pure-python (no heavy deps); safe to import at module load.
from frcscout.refine import identify6


class MatchRunner:
    """Layers scouting (6-robot prior -> identity -> phase -> metrics -> export) onto the
    validated engine, decoupled from video IO.

    Parameters
    ----------
    field_model : object | None
        The geometry ``FieldModel`` (or any duck-typed stand-in). Supplies HUB/TOWER
        positions + the field half-line to ``PhaseTracker``. May be None (dependent
        phases/metrics then degrade honestly, flagged unavailable).
    hub_localizer : object | None
        Anything with ``in_hub`` (``FieldModel`` or ``HubRegions``). Handed to the
        embedded ``ScoringEventTracker`` so shots/scores resolve as that module defines.
    cfg : dict | None
        Tuning knobs. Keys under ``sixbot``/``six`` are routed to SixRobotTracker; the
        rest are passed to PhaseTracker (which forwards the scoring-relevant subset to
        its embedded ScoringEventTracker).
    fps : float
        The capture's REAL fps (all seconds->frames timing flows from this).
    expected_alliances : dict | None
        Known 3-red/3-blue composition when available from the overlay, e.g.
        ``{"red": [254, 1678, 973], "blue": [118, 2056, 1114]}``. Used as the identity
        roster (the overlay's six teams) so bumper OCR reads resolve to real teams.
    """

    def __init__(
        self,
        field_model: object | None = None,
        hub_localizer: object | None = None,
        cfg: Optional[dict] = None,
        fps: float = 60.0,
        expected_alliances: Optional[dict] = None,
    ):
        self.field_model = field_model
        self.hub_localizer = hub_localizer if hub_localizer is not None else field_model
        self.cfg = dict(cfg or {})
        self.fps = float(fps)
        self.expected_alliances = self._norm_alliances(expected_alliances)

        # --- SixRobotTracker (the 6-robot structural prior). Accept its knobs under a
        #     "sixbot"/"six" sub-key so they don't collide with phase/scoring keys.
        six_cfg = self.cfg.get("sixbot") or self.cfg.get("six") or {}
        self.sixbot = SixRobotTracker(six_cfg, fps=self.fps)

        # --- FIELD GATE knobs. The robot detections are gated by the recovered pose BEFORE
        #     the 6-robot prior, so the prior never sees crowd / person / opposite-pane
        #     boxes. main_pane_frac is the lower y-boundary of the MAIN broadcast pane as a
        #     fraction of frame height (matches run_scout's MAIN_PANE = (0.0, 0.52)).
        gate_cfg = self.cfg.get("gate") or {}
        self.main_pane_frac = float(gate_cfg.get("main_pane_frac", 0.52))
        self.gate_tol_px = float(gate_cfg.get("tol_px", 24.0))
        # frame height in px (for the pane boundary). Unknown until set; when unknown the
        # pane test is skipped and the gate relies on field back-projection alone (which
        # already rejects opposite-pane/crowd boxes). Set via set_frame_h or cfg.gate.frame_h.
        self._frame_h: Optional[float] = (
            float(gate_cfg["frame_h"]) if gate_cfg.get("frame_h") else None)

        # --- PhaseTracker. It OWNS a ScoringEventTracker internally; we do NOT create a
        #     second one (that would double-count). We pass the rest of cfg through.
        phase_cfg = {k: v for k, v in self.cfg.items() if k not in ("sixbot", "six")}
        self.phase = PhaseTracker(
            field_model=field_model,
            hub_localizer=self.hub_localizer,
            cfg=phase_cfg or None,
            fps=self.fps,
        )

        # --- identity: accumulate per-track bumper-number OCR reads over the match, plus
        #     each track's held alliance (bumper colour), resolved retroactively at
        #     report time via identify6.resolve_six.
        self._bumper_reads: dict[int, list] = {}
        self._track_alliance: dict[int, Optional[Alliance]] = {}
        self._overlay_teams: dict[str, list[int]] = dict(self.expected_alliances)
        self._identities: dict[int, dict] = {}   # track_id -> {team, method, confidence, alliance}

        self._events: list = []                   # accumulated canonical Events
        self._last_overlay: Optional[OverlayState] = None

    # ───────────────────────── construction helpers ──────────────────────────
    @staticmethod
    def _norm_alliances(expected: Optional[dict]) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {"red": [], "blue": []}
        if not expected:
            return out
        for ally in ("red", "blue"):
            vals = expected.get(ally) or []
            out[ally] = [int(t) for t in vals]
        return out

    # ────────────────────────────── per-frame ────────────────────────────────
    def process(self, st: FrameState) -> dict:
        """Refine one engine FrameState in place and return the per-frame scouting view.

        Steps (see module docstring):
          (a) convert ``st.robots`` (ByteTrack Tracks) -> Detection + alliance form;
          (b) run SixRobotTracker -> <=6 STABLE tracks, and OVERWRITE ``st.robots`` with
              them (as Track objects carrying the stable id + alliance) so PHASE + render
              use the 6-robot prior, not the raw churning ids;
          (c) attach resolved team numbers where bumper OCR has resolved them;
          (d) run PhaseTracker on the (now 6-robot) FrameState.

        Returns ``{"state": st, "six": six, "phase": phase_out}``.
        """
        raw_robots = list(st.robots or [])

        # (a) ByteTrack Track -> Detection (+ parallel alliance list). Carry each raw
        #     track's centroid so we can re-attach geometry to the matched six below.
        dets: list[Detection] = []
        allies: list[Optional[Alliance]] = []
        for t in raw_robots:
            dets.append(Detection(cls="robot", xyxy=tuple(t.xyxy),
                                   conf=float(getattr(t, "conf", 1.0) or 1.0),
                                   frame_idx=st.frame_idx))
            allies.append(t.alliance)

        # (b') FIELD GATE — drop crowd / person / opposite-pane boxes BEFORE the prior.
        #     A detection survives only if its floor-contact point (cx, y2) is in the main
        #     pane AND back-projects onto the field (FieldModel.gate_detections). With no
        #     valid pose, or no gate-capable field model, this is a NO-OP (we never blank
        #     the prior just because a pose is missing this frame). The prior thus only ever
        #     sees plausible on-field robots, so it stops birthing crowd FPs into slots.
        dets, allies = self._gate_field(st, dets, allies)

        # (b) the 6-robot prior.
        six = self.sixbot.update(st.frame_idx, dets, allies)

        # rebuild st.robots from the <=6 stable tracks. Re-attach field geometry (and any
        # bumper OCR reads) by matching each stable track back to the nearest raw track
        # that fed it (so the field_xy_m / field_wh_m / z the geometry layer computed —
        # and the OCR reads a consumer attached — survive the swap; SixRobotTracker only
        # carries pixels + the stable id).
        st.robots = [self._six_to_track(d, raw_robots) for d in six]

        # accumulate identity evidence keyed by the STABLE id (the id that survives onto
        # st.robots and into the phase snapshot), then resolve. Doing this on the stable
        # tracks — not the raw ones — is what makes the resolved team land on the right id.
        for t in st.robots:
            self._record_identity(t)
        self._refresh_identities()

        # (d) PHASE + metrics on the refined frame.
        phase_out = self.phase.update(st, fps=self.fps)

        # stamp team numbers onto the per-frame phase records (so the renderer can show
        # them) and accumulate this frame's events.
        for rid, rec in phase_out.items():
            ident = self._identities.get(rid)
            if ident and ident.get("team") is not None:
                rec["team"] = ident["team"]
                rec["id_method"] = ident.get("method")
                rec["id_confidence"] = ident.get("confidence")
        self._accumulate_events(st)
        self._last_overlay = st.overlay

        return {"state": st, "six": six, "phase": phase_out}

    # ── field gate ──
    def set_frame_h(self, frame_h: float) -> None:
        """Tell the runner the frame height in px so the gate can apply the main-pane
        boundary (``main_pane_frac * frame_h``). Optional: without it the gate still runs,
        relying on field back-projection alone (which already rejects off-field boxes)."""
        self._frame_h = float(frame_h)

    def _gate_field(self, st: FrameState, dets: list[Detection],
                    allies: list[Optional[Alliance]]):
        """Apply the field gate to (dets, allies) in lock-step. No-op when there is no
        gate-capable field model or no valid pose. Infers frame height from the pose-time
        boxes only via the configured fraction; if frame height is unknown the pane
        boundary is set to +inf (pane test passes, field back-projection still gates)."""
        fm = self.field_model
        pose = getattr(st, "pose", None)
        if fm is None or not hasattr(fm, "gate_detections") or pose is None \
                or not getattr(pose, "valid", False):
            return dets, allies
        if self._frame_h is not None:
            y_main = self.main_pane_frac * self._frame_h
        else:
            y_main = float("inf")  # unknown frame height -> rely on field bounds only
        return fm.gate_detections(dets, pose, y_main, tol_px=self.gate_tol_px,
                                  alliances=allies)

    # ── scoring passthrough (for the tech lead / run_scout) ──
    def note_overlay_score(self, frame_idx: int, alliance: Alliance) -> None:
        """Passthrough to the embedded scoring authority. run_scout calls this when the
        overlay fuel/score counter ticks up for ``alliance`` at ``frame_idx`` so a shot can
        be overlay-confirmed when geometry is inconclusive. Forwards to
        ``self.phase.scoring.note_overlay_score`` (the single ScoringEventTracker)."""
        scoring = getattr(self.phase, "scoring", None)
        if scoring is not None and hasattr(scoring, "note_overlay_score"):
            scoring.note_overlay_score(int(frame_idx), alliance)

    # ── identity bookkeeping ──
    def _record_identity(self, t: Track) -> None:
        """Remember a STABLE track's held alliance + any bumper-number reads carried onto
        it (from the matched raw track) this frame.

        A consumer may attach OCR reads as a dynamic ``bumper_reads`` attribute on the
        ByteTrack tracks; ``_six_to_track`` forwards them onto the stable track. We
        accumulate them per STABLE id so identify6 can vote over the whole match at report
        time. Absent any reads, identity still closes by elimination from the alliance
        split + the overlay roster (``expected_alliances``)."""
        if t.alliance is not None:
            self._track_alliance[t.tid] = t.alliance
        else:
            self._track_alliance.setdefault(t.tid, None)
        reads = getattr(t, "bumper_reads", None)
        if reads:
            self._bumper_reads.setdefault(t.tid, [])
            self._bumper_reads[t.tid].extend(reads)

    def _refresh_identities(self) -> None:
        """Re-run the retroactive resolver over everything seen so far. Cheap (<=6
        tracks) and order-independent, so calling it per frame is fine; in practice the
        result only changes when a new read lands or the roster fills."""
        if not self._track_alliance:
            return
        # the SixRobotTracker emits stable ids; resolve over those (they ARE the track
        # ids carried on st.robots after the swap).
        resolved = identify6.resolve_six(
            self._track_alliance, self._bumper_reads, self._overlay_teams)
        self._identities = identify6.resolutions_to_dict(resolved)

    def _six_to_track(self, d: dict, raw_robots: list[Track]) -> Track:
        """Build a Track from one SixRobotTracker dict, re-attaching field geometry from
        the nearest raw track (by pixel centroid) so downstream geometry survives."""
        cx, cy = d["centroid"]
        src = self._nearest_raw(cx, cy, raw_robots, d.get("alliance"))
        # Track has no `est` field (see io/types.py); construct the documented fields and
        # carry the coast flag as a dynamic attr below.
        t = Track(
            tid=int(d["id"]),
            cls="robot",
            xyxy=tuple(d["xyxy"]),
            centroid_px=(float(cx), float(cy)),
            alliance=d.get("alliance"),
            hits=int(d.get("hits", 1)),
            age=int(d.get("age", 0)),
        )
        if src is not None:
            t.field_xy_m = src.field_xy_m
            t.vel_mps = getattr(src, "vel_mps", (0.0, 0.0)) or (0.0, 0.0)
            for attr in ("field_wh_m", "z_m", "heading_rad"):
                val = getattr(src, attr, None)
                if val is not None:
                    setattr(t, attr, val)
            # forward bumper OCR reads (if a consumer attached them) onto the stable track
            # so identity accumulates against the STABLE id.
            reads = getattr(src, "bumper_reads", None)
            if reads:
                t.bumper_reads = reads
        # expose the coast flag as a dynamic attr so the renderer can dim a predicted box
        # without depending on a Track field that doesn't exist.
        t.est = bool(d.get("est"))
        # carry the SixRobotTracker honesty flag through: a slot coasting past its grace is
        # emitted (id continuity) but not a CONFIDENT robot. Default True if absent.
        t.confident = bool(d.get("confident", True))
        return t

    @staticmethod
    def _nearest_raw(cx: float, cy: float, raw_robots: list[Track],
                     alliance: Optional[str]) -> Optional[Track]:
        """Nearest raw ByteTrack track to a stable track's centroid, gated to the same
        alliance when known (so geometry is re-attached from the right robot)."""
        best, best_d = None, float("inf")
        for r in raw_robots:
            if alliance is not None and r.alliance is not None and r.alliance != alliance:
                continue
            rcx, rcy = r.centroid_px
            d = (rcx - cx) ** 2 + (rcy - cy) ** 2
            if d < best_d:
                best, best_d = r, d
        return best

    def _accumulate_events(self, st: FrameState) -> None:
        """Append this frame's events to the match-long stream.

        ``PhaseTracker.update`` already stepped its embedded ScoringEventTracker (the
        single scoring authority) and consumed the events it returned — that return is
        not re-exposed, and re-stepping the scorer here would DOUBLE-COUNT. So the events
        we accumulate are the ones already present on ``st.events`` (the engine's fuser
        writes the canonical attributed stream there in the live path; synthetic callers
        may pre-populate it). This keeps us strictly a consumer — we never re-derive
        scoring."""
        for ev in (st.events or []):
            self._events.append(ev)

    # ────────────────────────────── finalize ─────────────────────────────────
    def finalize(self, match_key: str, overlay: Optional[OverlayState] = None) -> dict:
        """Build the match record ``export.write_match`` expects.

        Pulls per-robot metrics from ``PhaseTracker.snapshot()`` / ``metrics()``, merges
        the resolved team number + alliance onto each robot, and carries the
        ``{value, available, needs}`` envelope for gated metrics through UNCHANGED (so an
        unresolved metric stays ``available: False`` — never a fabricated 0). Includes the
        accumulated events.

        Returns ``{"match_key", "robots": {track_id: stats}, "events": [...], ...}``.
        """
        ov = overlay if overlay is not None else self._last_overlay
        self._refresh_identities()
        snapshot = self.phase.snapshot()          # {rid: {phase, held, label, metrics}}

        robots: dict[int, dict] = {}
        for rid, rec in snapshot.items():
            metrics = rec.get("metrics", {}) or {}
            ident = self._identities.get(rid, {})
            stats = self._robot_stats(rid, rec, metrics, ident)
            robots[rid] = stats

        record = {
            "match_key": match_key,
            "fps": self.fps,
            "robots": robots,
            "events": [self._event_to_dict(e) for e in self._events],
        }
        if ov is not None:
            record["overlay"] = self._overlay_to_dict(ov)
        if any(self._overlay_teams.values()):
            record["expected_alliances"] = dict(self._overlay_teams)
        return record

    def _robot_stats(self, rid: int, rec: dict, metrics: dict, ident: dict) -> dict:
        """Flatten one robot's phase record + metrics into the per-robot scouting stats
        dict, mapping PhaseTracker metric names onto the export SCOUT_COLUMNS where they
        differ, and carrying every gated metric's {value, available} envelope through.

        Mapping (PhaseTracker metric  ->  SCOUT column):
            intakes            -> intake_attempts
            (scoring) intakes  -> intake_successes        (confirmed possessions == intakes)
            shot_attempts      -> shot_attempts
            shots_made         -> shots_made
            shot_accuracy{}    -> shot_accuracy  (envelope preserved)
            max_held           -> max_held
            balls_shot_at_once -> balls_shot_at_once
            possession_time_s  -> possession_time_s
            climb_level        -> climb_level
            defense_time_s{}   -> defense_time_s (envelope preserved)
            jitter_index{}     -> jitter_index (envelope preserved)
            time_to_intake_s   -> mean_time_to_intake_s
        Columns with no vision-derivable source yet (fuel_scored, scoring_rate_per_min,
        mean_cycle_time_s, intake_success_rate) are emitted as the {value:None,
        available:False} envelope — UNKNOWN, never a fabricated 0.
        """
        intakes = metrics.get("intakes")
        stats: dict = {
            # identity
            "team": ident.get("team"),
            "alliance": rec.get("alliance") or ident.get("alliance"),
            "id_method": ident.get("method"),
            "id_confidence": ident.get("confidence"),
            # live phase (handy for replay / debugging; not a scout column)
            "phase": rec.get("phase"),
            "held": rec.get("held"),
            # ── intake ──
            "intake_attempts": intakes,
            "intake_successes": intakes,           # a confirmed intake IS a success here
            "intake_success_rate": self._gated(None, False,
                                                "intake attempts vs. failed-attempt "
                                                "detection (not vision-derived yet)"),
            "mean_time_to_intake_s": metrics.get("time_to_intake_s"),
            # ── possession ──
            "max_held": metrics.get("max_held"),
            "possession_time_s": metrics.get("possession_time_s"),
            # ── shots ──
            "balls_shot_at_once": metrics.get("balls_shot_at_once"),
            "shot_attempts": metrics.get("shot_attempts"),
            "shots_made": metrics.get("shots_made"),
            "shot_accuracy": metrics.get("shot_accuracy"),     # {value,available} envelope
            # ── climb / defense / jitter (envelopes preserved) ──
            "climb_level": metrics.get("climb_level"),
            "defense_time_s": metrics.get("defense_time_s"),   # {value,available} envelope
            "jitter_index": metrics.get("jitter_index"),       # {value,available} envelope
            # ── columns with no vision source yet: UNKNOWN, never 0 ──
            "fuel_scored": self._gated(None, False,
                                       "authoritative per-robot fuel needs overlay "
                                       "per-robot attribution (overlay gives alliance totals)"),
            "scoring_rate_per_min": self._gated(None, False,
                                                "derived from resolved shots_made over match "
                                                "duration; needs a resolvable shot stream"),
            "mean_cycle_time_s": self._gated(None, False,
                                             "intake->score cycle timing; needs paired "
                                             "intake/score events with localization"),
            # full per-robot metrics blob + timeline (kept in JSON, ignored by the CSV)
            "metrics": metrics,
        }
        return stats

    @staticmethod
    def _gated(value, available: bool, needs: str) -> dict:
        """Build a ``{value, available, needs}`` envelope (the same shape PhaseTracker
        uses and ``export._val`` unwraps). For a metric we cannot derive yet we emit
        ``available: False`` with a ``needs`` note — NEVER a fabricated 0."""
        return {"value": value, "available": bool(available), "needs": needs}

    # ── serialization helpers ──
    @staticmethod
    def _event_to_dict(e) -> dict:
        if isinstance(e, dict):
            return e
        return {
            "t_s": getattr(e, "t_s", None),
            "frame_idx": getattr(e, "frame_idx", None),
            "type": getattr(e, "type", None),
            "alliance": getattr(e, "alliance", None),
            "robot_id": getattr(e, "robot_id", None),
            "confidence": getattr(e, "confidence", None),
            "detail": getattr(e, "detail", None) or {},
        }

    @staticmethod
    def _overlay_to_dict(ov: OverlayState) -> dict:
        return {
            "match_label": getattr(ov, "match_label", None),
            "comp_level": getattr(ov, "comp_level", None),
            "match_number": getattr(ov, "match_number", None),
            "blue_teams": list(getattr(ov, "blue_teams", []) or []),
            "red_teams": list(getattr(ov, "red_teams", []) or []),
            "blue_score": getattr(ov, "blue_score", None),
            "red_score": getattr(ov, "red_score", None),
            "blue_fuel": getattr(ov, "blue_fuel", None),
            "red_fuel": getattr(ov, "red_fuel", None),
            "phase": getattr(ov, "phase", None),
            "timer_s": getattr(ov, "timer_s", None),
        }

    # ─────────────────────────────── export ──────────────────────────────────
    def export(self, outdir: str, match_key: str) -> dict:
        """Finalize and write the match files. Returns ``write_match``'s path dict."""
        from frcscout.fuse.export import write_match
        return write_match(self.finalize(match_key), outdir, match_key)

    # ─────────────────────────────── driver ──────────────────────────────────
    def run_video(self, engine, frames_iterable: Iterable, out_mp4: Optional[str] = None,
                  out_dir: Optional[str] = None, match_key: str = "match") -> dict:
        """Thin video driver: step the engine over frames, refine + render each, write an
        mp4 (cv2 imported LAZILY here so importing this module never needs cv2), then
        export the match files at the end.

        ``engine`` must expose ``step(frame, idx) -> FrameState``. ``frames_iterable``
        yields decoded BGR frames. Returns a dict with the render/export summary.
        """
        from frcscout.viz.scout_render import draw_scout_overlay  # lazy (guards cv2/numpy)

        writer = None
        n = 0
        try:
            for i, frame in enumerate(frames_iterable):
                st = engine.step(frame, i)
                res = self.process(st)
                if out_mp4 is not None:
                    annotated = draw_scout_overlay(
                        frame, res["state"], res["six"], res["phase"],
                        self.phase.metrics(), fps=getattr(self, "fps", None))
                    writer = self._ensure_writer(writer, out_mp4, annotated)
                    if writer is not None and annotated is not None:
                        writer.write(annotated)
                n += 1
        finally:
            if writer is not None:
                writer.release()

        summary = {"frames": n, "mp4": out_mp4}
        if out_dir is not None:
            summary["export"] = self.export(out_dir, match_key)
        return summary

    def _ensure_writer(self, writer, out_mp4: str, annotated):
        """Open a cv2.VideoWriter on the first frame (so the size matches the annotated
        output). cv2 is imported HERE — never at module import — and guarded."""
        if writer is not None or annotated is None:
            return writer
        try:
            import cv2  # lazy + guarded
        except Exception:
            return None
        h, w = annotated.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(out_mp4, fourcc, float(self.fps), (int(w), int(h)))


__all__ = ["MatchRunner"]
