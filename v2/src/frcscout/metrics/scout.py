"""MetricsAggregator — turns a finalized ``match_record`` into the full §5 schema.

Inputs (all produced upstream; this layer is pure-python, no model / cv2 / GPU):

  match_record : dict
      {
        "events":  [ Event.__dict__, ... ]        # score / shot_* / intake_* / climb
        "robots":  { track_id: RobotStats.to_dict()+identity, ... }
        "shots":   [...]                           # (optional, unused here)
        ... possibly "identities", "overlay_timeseries", "meta" ...
      }
  overlay_timeseries : list[dict] | None
      Per-frame authoritative read from the broadcast overlay:
        { "t_s", "frame_idx", "phase", "blue_fuel", "red_fuel",
          "blue_score", "red_score" }
      ``phase`` is one of "auto" | "teleop" | "endgame" (the §1 FSM). When the
      timeseries is absent we fall back to deriving phase from event ``t_s`` via
      the match-clock config, and to deriving fuel totals from the score-event
      stream — both clearly flagged in provenance.
  identities : dict[int|str, dict] | None
      {track_id: ResolvedId.__dict__}. Optional; the per-robot rollup also reads
      the ``team`` already stamped onto the robot stats by ``apply_identities``.

Output: ``aggregate()`` returns the §5.1 (per-alliance) + §5.2 (per-robot) schema
plus a ``tba_comparable`` block. ``team_aggregates()`` rolls many match outputs
into the §5.3 team table (avg fuel, climb-rate, consistency variance, active-shift
conversion, OPR/DPR/CCWM hook).

All scoring values + RP thresholds come from ``MetricsConfig`` (game-model
defaults), never hardcoded inline.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Optional


# ───────────────────────────── RP threshold tables (§3.2, Table 6-5) ──────────
# {energized_fuel, supercharged_fuel, traversal_tower_points}
REGIONAL_THRESHOLDS = {"energized": 100, "supercharged": 360, "traversal": 50}
DISTRICT_CHAMPS_THRESHOLDS = {"energized": 240, "supercharged": 360, "traversal": 50}
FIRST_CHAMPS_THRESHOLDS = {"energized": 360, "supercharged": 500, "traversal": 50}


@dataclass
class MetricsConfig:
    """All scoring values + thresholds, config-driven (§3, §3.2). Defaults mirror
    the game model; override per event level (Regional / DCMP / Champs)."""
    # §3.1 MATCH point values
    fuel_point: int = 1                       # active-HUB fuel, AUTO or TELEOP
    tower_level_points: dict = field(default_factory=lambda: {1: 10, 2: 20, 3: 30})
    # §4.1 penalty currency
    minor_foul_points: int = 5                # to OPPONENT
    major_foul_points: int = 15               # to OPPONENT
    # §3.2 win/tie RP
    win_rp: int = 3
    tie_rp: int = 1
    energized_rp: int = 1
    supercharged_rp: int = 1
    traversal_rp: int = 1
    # §3.2 bonus thresholds — default Regional (config-driven, see tables above)
    rp_thresholds: dict = field(default_factory=lambda: dict(REGIONAL_THRESHOLDS))

    # §1 match-clock fallback (only used to derive phase when no overlay timeseries).
    # Timer counts DOWN from 2:40. t_s here is elapsed seconds from match start.
    auto_len_s: float = 20.0
    teleop_len_s: float = 140.0               # 2:20
    endgame_len_s: float = 30.0               # last 0:30 of teleop

    @property
    def match_len_s(self) -> float:
        return self.auto_len_s + self.teleop_len_s

    @property
    def endgame_start_s(self) -> float:
        """Elapsed seconds at which END GAME begins (auto + teleop − endgame)."""
        return self.auto_len_s + (self.teleop_len_s - self.endgame_len_s)

    @classmethod
    def for_level(cls, level: str = "regional", **overrides) -> "MetricsConfig":
        table = {
            "regional": REGIONAL_THRESHOLDS,
            "district": REGIONAL_THRESHOLDS,
            "district_champs": DISTRICT_CHAMPS_THRESHOLDS,
            "dcmp": DISTRICT_CHAMPS_THRESHOLDS,
            "champs": FIRST_CHAMPS_THRESHOLDS,
            "first_champs": FIRST_CHAMPS_THRESHOLDS,
        }.get(level.lower(), REGIONAL_THRESHOLDS)
        cfg = cls(rp_thresholds=dict(table))
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg


def _metric(value: Any, available: bool = True, needs: Optional[str] = None,
            unit: Optional[str] = None, provenance: Optional[str] = None) -> dict:
    """Uniform metric envelope. An UNAVAILABLE metric carries value=None,
    available=False and a one-line ``needs:`` note — never a fabricated number."""
    m: dict = {"value": value, "available": bool(available)}
    if unit is not None:
        m["unit"] = unit
    if needs is not None:
        m["needs"] = needs
    if provenance is not None:
        m["provenance"] = provenance
    return m


# Back-compat handle for tests/callers that want the envelope type.
Metric = dict


def _robust_monotonic(values: list) -> list:
    """Clean a noisy OCR'd CUMULATIVE counter into a monotonic non-decreasing series.

    The broadcast score/fuel counters only ever increase during a match, but OCR
    injects two failure modes (measured on qual59): a single spuriously-HIGH read
    (which, in a naive delta-sum, suppresses all later real increments) and small
    upward jitter (which inflates the sum). We reject both:
      * drop None/negative samples,
      * reject a sample that JUMPS above the recent floor by more than a plausible
        per-step amount (treat as a misread spike), then
      * enforce non-decreasing (a low misread below the running max is pulled up to it).
    Returns the cleaned list (same order). Pure function — unit-testable, no cv2.
    """
    nums = [v for v in values if isinstance(v, (int, float)) and v >= 0]
    if not nums:
        return []
    # Robust per-step ceiling: generous but finite, from the data's own spread.
    # A real fuel/score step between ~0.5s overlay samples is modest; a 5x-median
    # jump is a misread. Use an absolute floor so early small values still admit growth.
    import statistics
    diffs = [b - a for a, b in zip(nums, nums[1:]) if b >= a]
    med_step = statistics.median(diffs) if diffs else 0.0
    max_step = max(40.0, 6.0 * med_step)   # generous; only catches gross spikes
    out: list = []
    running = nums[0]
    for v in nums:
        if v > running + max_step:          # implausible upward spike -> skip
            continue
        running = max(running, v)           # non-decreasing
        out.append(running)
    return out


def _robust_final(values: list, pct: float = 0.98) -> Optional[float]:
    """Robust FINAL value of a monotonic-increasing OCR'd counter (e.g. the match
    SCORE). Unlike `_robust_monotonic` (which step-caps and is right for the small-step
    FUEL delta-sum), the score takes legitimately LARGE jumps, so step-capping wrongly
    stalls it (measured: red_score raw max 593 ≈ TBA 613, but step-capped stalled at 399).
    A score only rises, so the final ≈ the max — we just drop the top (1-pct) as possible
    single-frame OCR over-reads by taking the pct-percentile of the sorted values.
    Returns None if no numeric samples. Pure function, unit-testable.
    """
    nums = sorted(v for v in values if isinstance(v, (int, float)) and v >= 0)
    if not nums:
        return None
    idx = min(len(nums) - 1, int(round(pct * (len(nums) - 1))))
    return float(nums[idx])


class MetricsAggregator:
    """Consumes one finalized match_record -> full §5.1 + §5.2 schema + TBA block,
    and rolls many such outputs into §5.3 team aggregates."""

    def __init__(self, config: Optional[MetricsConfig] = None):
        self.cfg = config or MetricsConfig()

    # ───────────────────────────── public API ────────────────────────────────
    def aggregate(
        self,
        match_record: dict,
        overlay_timeseries: Optional[list[dict]] = None,
        identities: Optional[dict] = None,
    ) -> dict:
        """Produce the §5.1 + §5.2 schema + tba_comparable for ONE match."""
        events = match_record.get("events", []) or []
        robots = match_record.get("robots", {}) or {}
        identities = identities or match_record.get("identities") or {}
        ts = overlay_timeseries
        if ts is None:
            ts = match_record.get("overlay_timeseries")

        # per-robot rollup (§5.2)
        per_robot = {
            str(rid): self._robot_metrics(rid, stats, events, ts, identities)
            for rid, stats in robots.items()
        }

        # per-alliance reconstruction (§5.1)
        per_alliance = {
            al: self._alliance_metrics(al, events, robots, ts)
            for al in ("blue", "red")
        }

        # TBA-comparable block the validation layer consumes
        tba = self._tba_comparable(per_alliance, ts)

        return {
            "schema_version": "rebuilt-2026.§5",
            "per_robot": per_robot,
            "per_alliance": per_alliance,
            "tba_comparable": tba,
            "provenance": {
                "overlay_timeseries": ts is not None,
                "n_events": len(events),
                "n_robots": len(robots),
                "rp_thresholds": dict(self.cfg.rp_thresholds),
            },
        }

    # ───────────────────────── phase / fuel helpers ──────────────────────────
    def _phase_at(self, t_s: Optional[float], ts: Optional[list[dict]],
                  frame_idx: Optional[int] = None) -> Optional[str]:
        """Phase ("auto"|"teleop"|"endgame") at an event. Prefer the authoritative
        overlay timeseries (nearest sample); else derive from the match clock."""
        if ts:
            sample = self._nearest_sample(ts, t_s, frame_idx)
            if sample is not None and sample.get("phase"):
                return sample["phase"]
        if t_s is None:
            return None
        if t_s < self.cfg.auto_len_s:
            return "auto"
        if t_s >= self.cfg.endgame_start_s:
            return "endgame"
        return "teleop"

    @staticmethod
    def _nearest_sample(ts: list[dict], t_s: Optional[float],
                        frame_idx: Optional[int]) -> Optional[dict]:
        if not ts:
            return None
        if frame_idx is not None:
            cand = [s for s in ts if s.get("frame_idx") is not None]
            if cand:
                return min(cand, key=lambda s: abs(s["frame_idx"] - frame_idx))
        if t_s is not None:
            cand = [s for s in ts if s.get("t_s") is not None]
            if cand:
                return min(cand, key=lambda s: abs(s["t_s"] - t_s))
        return None

    def _hub_active(self, alliance: str, phase: Optional[str],
                    sample: Optional[dict]) -> Optional[bool]:
        """Whether ``alliance``'s HUB is active (§1.1). Authoritative source is the
        HUB-light detector (not built) surfaced on the timeseries sample as
        ``{alliance}_hub_active``. AUTO / TRANSITION / END GAME -> both active. The
        per-shift alternation needs that detector, so during teleop shifts we return
        None (unknown) unless the sample carries it."""
        if sample is not None:
            v = sample.get(f"{alliance}_hub_active")
            if v is not None:
                return bool(v)
        if phase in ("auto", "endgame"):
            return True
        return None  # teleop shift alternation needs the HUB-light detector

    def _fuel_from_overlay(self, alliance: str,
                           ts: Optional[list[dict]]) -> Optional[dict]:
        """Authoritative cumulative fuel from the overlay timeseries, split by
        phase. Returns {"auto", "teleop", "endgame", "total"} or None if no ts.

        The overlay fuel count is monotonic-ish cumulative; we attribute each
        positive increment to the phase of the sample in which it occurred."""
        if not ts:
            return None
        key = f"{alliance}_fuel"
        samples = [s for s in ts if s.get(key) is not None]
        if not samples:
            return None
        samples = sorted(samples, key=lambda s: (s.get("frame_idx") or 0, s.get("t_s") or 0.0))
        buckets = {"auto": 0, "teleop": 0, "endgame": 0}
        # Robust monotonic clean of the cumulative counter BEFORE delta-summing: a single
        # high OCR misread otherwise suppresses all later increments (observed: red teleop
        # collapsed to 5) and upward jitter inflates the sum (observed: blue 140 vs 66).
        import statistics as _stats
        raw = [float(s[key]) for s in samples]
        diffs = [b - a for a, b in zip(raw, raw[1:]) if b >= a]
        max_step = max(40.0, 6.0 * (_stats.median(diffs) if diffs else 0.0))
        prev = None
        running = None
        for s in samples:
            cur = float(s[key])
            if running is not None and cur > running + max_step:
                continue                         # implausible spike -> skip this sample
            running = cur if running is None else max(running, cur)
            if prev is not None and running > prev:
                ph = s.get("phase") or self._phase_at(s.get("t_s"), ts, s.get("frame_idx")) or "teleop"
                if ph not in buckets:
                    ph = "teleop"
                buckets[ph] += running - prev
            prev = running
        total = buckets["auto"] + buckets["teleop"] + buckets["endgame"]
        # if the first (cleaned) sample already had fuel (missed the ramp), treat as auto preload
        first_clean = next((float(s[key]) for s in samples), 0.0)
        if first_clean > 0:
            buckets["auto"] += first_clean
            total += first_clean
        # CAP detection: the broadcast fuel readout is a 0-100 RP-progress counter (measured:
        # red_fuel saturates at 100 on qual59 while the true total was 518). When the cleaned
        # series reaches the cap, the overlay total is a LOWER BOUND, not the true cumulative —
        # flag it so downstream marks the fuel metric capped/uncertain (use SCORE or CV instead).
        capped = (running is not None and running >= 99)
        return {**buckets, "total": total, "capped": bool(capped)}

    def _fuel_from_score_events(self, alliance: str,
                                events: list[dict], ts: Optional[list[dict]]) -> dict:
        """Fallback fuel attribution from the ``score`` event stream (each carries
        detail.fuel_delta). Phase per event from clock/overlay."""
        buckets = {"auto": 0, "teleop": 0, "endgame": 0}
        for e in events:
            if e.get("type") != "score" or e.get("alliance") != alliance:
                continue
            delta = (e.get("detail") or {}).get("fuel_delta", 1)
            ph = self._phase_at(e.get("t_s"), ts, e.get("frame_idx")) or "teleop"
            if ph not in buckets:
                ph = "teleop"
            buckets[ph] += delta
        total = sum(buckets.values())
        return {**buckets, "total": total, "capped": False}

    # ───────────────────────────── §5.2 per-robot ─────────────────────────────
    def _robot_metrics(self, rid: Any, stats: dict, events: list[dict],
                       ts: Optional[list[dict]], identities: dict) -> dict:
        """The §5.2 per-robot scouting record: ≥20 metrics. Derivable ones carry a
        value + available:True; detector-gated ones are emitted available:False."""
        cfg = self.cfg
        # identity (already stamped by apply_identities, else from identities map)
        team = stats.get("team")
        if team is None:
            ident = identities.get(rid) or identities.get(str(rid)) or identities.get(_as_int(rid))
            if isinstance(ident, dict):
                team = ident.get("team")
        alliance = stats.get("alliance")

        # this robot's events (matched on robot_id == track id)
        tid = _as_int(rid)
        rev = [e for e in events if e.get("robot_id") == tid or e.get("robot_id") == rid]

        # ── fuel scored = this robot's confirmed shots that scored ──
        shots_made = int(stats.get("shots_made", 0) or 0)
        fuel_scored = shots_made * cfg.fuel_point  # 1 pt/fuel; made shot == 1 fuel in HUB

        # ── per-phase fuel from this robot's shot_success events ──
        auto_fuel = endgame_fuel = teleop_fuel = 0
        success_evs = [e for e in rev if e.get("type") == "shot_success"]
        for e in success_evs:
            ph = self._phase_at(e.get("t_s"), ts, e.get("frame_idx")) or "teleop"
            if ph == "auto":
                auto_fuel += cfg.fuel_point
            elif ph == "endgame":
                endgame_fuel += cfg.fuel_point
            else:
                teleop_fuel += cfg.fuel_point
        # If the possession layer credited makes but we have no per-event phase
        # (events list pruned), don't fabricate the phase split — fall back to a
        # single teleop bucket only when event count matches, else report unknown.
        evt_made = len(success_evs)

        # ── active-shift vs wasted-inactive fuel (§5.2 active-shift efficiency) ──
        # active = own HUB active when the shot scored; needs HUB-light detector to
        # know teleop-shift alternation, so it's only certain in auto/endgame or when
        # the sample carries {alliance}_hub_active.
        active_fuel = wasted_inactive_fuel = 0
        active_known = True
        for e in success_evs:
            sample = self._nearest_sample(ts, e.get("t_s"), e.get("frame_idx")) if ts else None
            ph = self._phase_at(e.get("t_s"), ts, e.get("frame_idx"))
            act = self._hub_active(alliance, ph, sample) if alliance else None
            if act is True:
                active_fuel += cfg.fuel_point
            elif act is False:
                wasted_inactive_fuel += cfg.fuel_point
            else:
                active_known = False  # teleop-shift alternation unknown w/o HUB lights

        # ── scoring rate (fuel / minute) over the match window ──
        rate = None
        if cfg.match_len_s > 0:
            rate = fuel_scored / (cfg.match_len_s / 60.0)

        # possession / cycle
        mean_cycle = stats.get("avg_cycle_time_s")
        possession_time = stats.get("possession_time_s", 0.0)

        # intake
        intake_attempts = int(stats.get("intake_attempts", 0) or 0)
        intake_successes = int(stats.get("intake_successes", 0) or 0)
        intake_rate = stats.get("intake_success_rate")
        mean_tti = stats.get("mean_time_to_intake_s")

        # opportunity: balls in radius peak (real) + mean (derive from per-frame? we
        # only retain the peak from possession — mean needs the per-frame series we
        # don't persist, so it's available:False).
        balls_peak = int(stats.get("balls_in_radius_peak", 0) or 0)

        # shots
        shot_attempts = int(stats.get("shot_attempts", 0) or 0)
        shot_accuracy = stats.get("shot_accuracy")           # None when unresolvable
        miss_rate = stats.get("miss_rate")                   # None when unresolvable

        m: dict = {
            # ── identity ──
            "track_id": tid,
            "team": team,
            "alliance": alliance,
            "id_method": stats.get("id_method"),
            "id_confidence": stats.get("id_confidence"),

            # ── fuel cycle (§5.2) — DERIVABLE ──
            "fuel_scored": _metric(fuel_scored, unit="fuel",
                                   provenance="possession.shots_made"),
            "scoring_rate_per_min": _metric(rate, unit="fuel/min",
                                            provenance="fuel_scored / match_len"),
            "mean_cycle_time_s": _metric(mean_cycle, available=mean_cycle is not None,
                                         needs=None if mean_cycle is not None
                                         else "≥1 completed pickup→shot cycle",
                                         unit="s", provenance="possession.cycle_times"),
            # ── intake (§5.2 / possession) — DERIVABLE ──
            "intake_attempts": _metric(intake_attempts, provenance="possession"),
            "intake_successes": _metric(intake_successes, provenance="possession"),
            "intake_success_rate": _metric(
                intake_rate, available=intake_attempts > 0,
                needs=None if intake_attempts > 0 else "≥1 intake attempt",
                provenance="possession"),
            "mean_time_to_intake_s": _metric(
                mean_tti, available=mean_tti is not None,
                needs=None if mean_tti is not None else "≥1 confirmed intake",
                unit="s", provenance="possession.time_to_intake"),
            # ── possession / opportunity — DERIVABLE ──
            "possession_time_s": _metric(possession_time, unit="s",
                                         provenance="possession"),
            "balls_in_radius_peak": _metric(balls_peak, unit="balls",
                                            provenance="possession"),
            # ── shots (§5.2) — DERIVABLE (accuracy None when unresolvable, honest) ──
            "shot_attempts": _metric(shot_attempts, provenance="possession"),
            "shots_made": _metric(shots_made, provenance="possession"),
            "shot_accuracy": _metric(
                shot_accuracy, available=shot_accuracy is not None,
                needs=None if shot_accuracy is not None
                else "geometry pose or overlay pulse to resolve a shot",
                provenance="possession.shots_resolvable"),
            "miss_rate": _metric(
                miss_rate, available=miss_rate is not None,
                needs=None if miss_rate is not None
                else "geometry pose or overlay pulse to resolve a shot",
                provenance="possession.shots_resolvable"),
            # ── AUTO / phase fuel — DERIVABLE when per-event phase available ──
            "auto_fuel": _metric(
                auto_fuel, available=ts is not None or evt_made > 0,
                needs=None if (ts is not None or evt_made > 0)
                else "phase tagging on shot_success events or overlay timeseries",
                unit="fuel", provenance="shot_success events × phase"),
            "teleop_fuel": _metric(
                teleop_fuel, available=ts is not None or evt_made > 0,
                needs=None if (ts is not None or evt_made > 0)
                else "phase tagging on shot_success events or overlay timeseries",
                unit="fuel", provenance="shot_success events × phase"),
            "endgame_fuel": _metric(
                endgame_fuel, available=ts is not None or evt_made > 0,
                needs=None if (ts is not None or evt_made > 0)
                else "phase tagging on shot_success events or overlay timeseries",
                unit="fuel", provenance="shot_success events × phase (last 0:30)"),
            # ── active-shift efficiency (§5.2 key skill signal) ──
            "active_shift_fuel": _metric(
                active_fuel, available=active_known,
                needs=None if active_known
                else "HUB-light detector for teleop-shift active/inactive alternation",
                unit="fuel", provenance="shot_success × HUB-active state"),
            "wasted_inactive_fuel": _metric(
                wasted_inactive_fuel, available=active_known,
                needs=None if active_known
                else "HUB-light detector for teleop-shift active/inactive alternation",
                unit="fuel", provenance="shot_success × HUB-active state"),

            # ── DETECTOR-GATED (emitted, value=None, available=False) ──
            "sourcing_mix": _metric(
                None, available=False,
                needs="sourcing-location detector (DEPOT vs OUTPOST/human vs floor vs preload)"),
            "auto_mobility": _metric(
                None, available=False,
                needs="zone-geofencing detector (robot crossed starting line in AUTO)"),
            "preload_count": _metric(
                None, available=False,
                needs="pre-match ball-inventory count per robot"),
            "climb_level": _metric(
                None, available=False,
                needs="climb-level detector (bumper height vs LOW/MID/HIGH rung)"),
            "climb_start_time_s": _metric(
                None, available=False, unit="s",
                needs="climb-level detector + climb-onset timing"),
            "time_to_climb_s": _metric(
                None, available=False, unit="s",
                needs="climb-level detector (onset→latched-level duration)"),
            "climb_success": _metric(
                None, available=False,
                needs="climb-level detector (reached a latched level y/n)"),
            "defense_time_s": _metric(
                None, available=False, unit="s",
                needs="zone-geofencing detector (time in opponent ALLIANCE ZONE)"),
            "pins_committed": _metric(
                None, available=False,
                needs="robot-robot proximity/persistence detector (G418 pin >3s)"),
            "pins_drawn": _metric(
                None, available=False,
                needs="robot-robot proximity/persistence detector"),
            "foul_candidates": _metric(
                None, available=False,
                needs="foul-candidate detector (G4xx contact/zone advisory flags)"),
            "fouls_drawn": _metric(
                None, available=False,
                needs="foul-candidate detector (referee-authoritative downstream)"),
            "disables": _metric(
                None, available=False,
                needs="reliability detector (disabled/disconnected/no-show/tip-over)"),
            "endgame_tower_points": _metric(
                None, available=False, unit="points",
                needs="climb-level detector (TOWER points in the 0:30 END GAME window)"),
            "balls_in_radius_mean": _metric(
                None, available=False, unit="balls",
                needs="persisted per-frame balls-in-radius series (only peak retained)"),
        }
        return m

    # ───────────────────────────── §5.1 per-alliance ──────────────────────────
    def _alliance_metrics(self, alliance: str, events: list[dict],
                          robots: dict, ts: Optional[list[dict]]) -> dict:
        cfg = self.cfg
        # fuel: prefer authoritative overlay timeseries, else score-event stream.
        ov_fuel = self._fuel_from_overlay(alliance, ts)
        if ov_fuel is not None:
            fuel = ov_fuel
            fuel_src = "overlay_timeseries"
        else:
            fuel = self._fuel_from_score_events(alliance, events, ts)
            fuel_src = "score_events"

        auto_fuel = fuel["auto"]
        teleop_fuel = fuel["teleop"] + fuel["endgame"]   # endgame is part of teleop period
        total_active_fuel = fuel["total"]
        # The broadcast fuel readout is a 0-100 RP-progress counter; once it saturates the
        # cumulative is a LOWER BOUND, not exact (measured: red 100 vs true 518). Surface that
        # honestly in provenance so a consumer never treats a capped fuel total as the truth.
        fuel_capped = bool(fuel.get("capped", False))
        _cap_note = " [CAPPED: overlay fuel counter saturated at 100 -> LOWER BOUND, not exact; use score/CV]"
        _fsrc = fuel_src + (_cap_note if fuel_capped else "")

        # fuel into inactive HUB (§5.1 waste metric) — needs HUB-light detector to
        # separate active vs inactive scoring; overlay fuel count typically only
        # reflects ACTIVE scoring, so inactive is genuinely unknown here.
        fuel_inactive_known = False
        fuel_inactive = None

        # MATCH points: fuel × 1 + tower points (tower needs climb detector).
        fuel_points = total_active_fuel * cfg.fuel_point
        tower_points = None  # detector-gated

        # match_points_total: we can give the FUEL component; tower is unavailable,
        # so the *complete* total is unavailable. Report the fuel component honestly.
        match_points_available = tower_points is not None
        match_points_total = (fuel_points + (tower_points or 0)) if match_points_available else None

        # foul points (segregated, §6.3) — needs foul detector + referee authority.
        foul_points = None

        # RP flags (§3.2) using config thresholds.
        thr = cfg.rp_thresholds
        energized = total_active_fuel >= thr["energized"]
        supercharged = total_active_fuel >= thr["supercharged"]
        # traversal needs TOWER points (climb detector) -> unavailable.
        traversal_available = tower_points is not None
        traversal = (tower_points >= thr["traversal"]) if traversal_available else None
        # win/tie needs both alliances' totals incl. tower -> unavailable until tower.
        win_tie_available = False

        return {
            "auto_fuel_scored": _metric(auto_fuel, unit="fuel", provenance=fuel_src),
            "teleop_fuel_scored": _metric(teleop_fuel, unit="fuel", provenance=_fsrc),
            "active_fuel_total": _metric(total_active_fuel, unit="fuel", provenance=_fsrc),
            "fuel_counter_capped": _metric(fuel_capped, unit="flag",
                                           provenance="overlay 0-100 RP-counter saturated"),
            "fuel_into_inactive_hub": _metric(
                fuel_inactive, available=fuel_inactive_known,
                needs="HUB-light detector to separate inactive-HUB fuel from the active count",
                unit="fuel"),
            "fuel_points": _metric(fuel_points, unit="points",
                                   provenance=f"{total_active_fuel} fuel × {cfg.fuel_point}"),
            "tower_points": _metric(
                tower_points, available=False, unit="points",
                needs="climb-level detector (TOWER L1/L2/L3 = 10/20/30)"),
            "auto_tower_l1_count": _metric(
                None, available=False,
                needs="climb-level detector (AUTO LEVEL-1 climbs, cap 2/alliance)"),
            "tower_climbs": _metric(
                None, available=False,
                needs="climb-level detector ([{robot, level, phase}] list)"),
            "match_points_total": _metric(
                match_points_total, available=match_points_available,
                needs=None if match_points_available
                else "climb-level detector for TOWER points (fuel component computed; tower missing)",
                unit="points",
                provenance="fuel_points (+ tower_points when available)"),
            "rp_energized": _metric(
                energized, unit="flag",
                provenance=f"active_fuel {total_active_fuel} ≥ {thr['energized']}"),
            "rp_supercharged": _metric(
                supercharged, unit="flag",
                provenance=f"active_fuel {total_active_fuel} ≥ {thr['supercharged']}"),
            "rp_traversal": _metric(
                traversal, available=traversal_available,
                needs=None if traversal_available
                else f"climb-level detector (TOWER points ≥ {thr['traversal']})",
                unit="flag"),
            "rp_win_tie": _metric(
                None, available=win_tie_available,
                needs="complete match_points_total for both alliances (needs TOWER points)",
                unit="flag"),
            "rp_total": _metric(
                None, available=False,
                needs="win/tie + traversal RP (need TOWER points) to sum with energized/supercharged",
                unit="RP"),
            "minor_fouls_received": _metric(
                None, available=False,
                needs="foul-candidate detector (referee-authoritative)"),
            "major_fouls_received": _metric(
                None, available=False,
                needs="foul-candidate detector (referee-authoritative)"),
            "foul_points_credited": _metric(
                foul_points, available=False,
                needs="foul-candidate detector; kept SEGREGATED from earned points (§6.3)",
                unit="points"),
            "cards": _metric(
                None, available=False,
                needs="card/penalty detector (yellow/red per team)"),
        }

    # ───────────────────────── TBA-comparable block ──────────────────────────
    def _tba_comparable(self, per_alliance: dict, ts: Optional[list[dict]]) -> dict:
        """The block the validation layer (eval/tba_verify) checks against TBA /
        overlay truth. JSON-serializable scalars only.

        per-alliance: {auto_fuel, teleop_fuel, total_fuel, estimated_match_points}
        per-match:    {blue_score_est, red_score_est}

        ``estimated_match_points`` is the FUEL component (1 pt/fuel); the TOWER and
        foul components are detector-gated and are surfaced as None so downstream
        knows the estimate is fuel-only, not a fabricated full score."""
        out: dict = {}
        for al in ("blue", "red"):
            a = per_alliance[al]
            auto_f = a["auto_fuel_scored"]["value"]
            tele_f = a["teleop_fuel_scored"]["value"]
            total_f = a["active_fuel_total"]["value"]
            fuel_pts = a["fuel_points"]["value"]
            tower_pts = a["tower_points"]["value"]            # None (gated)
            foul_pts = a["foul_points_credited"]["value"]     # None (gated)
            est_full = None
            if tower_pts is not None:
                est_full = fuel_pts + tower_pts + (foul_pts or 0)
            out[al] = {
                "auto_fuel": auto_f,
                "teleop_fuel": tele_f,
                "total_fuel": total_f,
                "estimated_match_points": fuel_pts,        # fuel-only component
                "estimated_match_points_full": est_full,   # None until TOWER lands
                "fuel_only": tower_pts is None,
            }
        # Expose the overlay's authoritative cumulative SCORE — the uncapped, reliable
        # signal (the live fuel counter's OCR is noisy/can fail; the score is monotonic).
        # Use a robust monotonic clean over the whole series, then take the final (max)
        # value, instead of the raw last sample which may be an OCR spike (e.g. red 105
        # between 315 and 507 on qual59). This is the strongest vision↔TBA-total check.
        # Score is monotonic-increasing with LARGE legit jumps -> use a robust-max
        # (high percentile drops single-frame over-reads), NOT the step-capped walk
        # which stalls on big jumps (red_score: pct-max ~585 vs stalled 399; TBA 613).
        blue_truth = red_truth = None
        if ts:
            blue_truth = _robust_final([s.get("blue_score") for s in ts])
            red_truth = _robust_final([s.get("red_score") for s in ts])
        out["blue_score_est"] = out["blue"]["estimated_match_points"]
        out["red_score_est"] = out["red"]["estimated_match_points"]
        out["blue_score_overlay_truth"] = blue_truth
        out["red_score_overlay_truth"] = red_truth
        return out

    # ─────────────────────── identity-attribution gate ───────────────────────
    # An offensive per-robot stat may ONLY be attributed to a real team number when
    # the identity layer actually RESOLVED that track to a team. The id is resolved
    # when id_method is an authoritative resolver (anchor: a bumper-number/AprilTag
    # anchor; elimination: the last-slot logical deduction) AND id_confidence clears
    # the floor. Anything else (id_method "unknown"/"prior"/None, low confidence, or
    # a null team) is left at attribution_level='alliance': the robot is real and its
    # fuel happened, but we don't know WHICH physical team did it, so crediting it to
    # a synthetic ``track:N`` key would split one team across many ids and fabricate
    # per-team offense. Those samples are EXCLUDED from per-team offensive means.
    RESOLVED_ID_METHODS = frozenset({"anchor", "elimination"})
    MIN_ID_CONFIDENCE = 0.5

    @classmethod
    def _identity_resolved(cls, rm: dict) -> bool:
        team = rm.get("team")
        if team is None:
            return False
        method = rm.get("id_method")
        if method not in cls.RESOLVED_ID_METHODS:
            return False
        conf = rm.get("id_confidence")
        try:
            conf = float(conf) if conf is not None else 0.0
        except (TypeError, ValueError):
            conf = 0.0
        return conf >= cls.MIN_ID_CONFIDENCE

    # ─────────────────────────── §5.3 team aggregates ─────────────────────────
    def team_aggregates(self, match_metrics: list[dict],
                        opr_inputs: Optional[list[dict]] = None) -> dict:
        """Roll many per-match ``aggregate()`` outputs into the §5.3 team table.

        Multi-match aggregates per team_number:
          - avg_fuel_per_match, fuel_variance (consistency, §5.3)
          - climb_rate (% of matches with a successful climb) — available:False
            until the climb detector lands
          - active_shift_conversion (% of own fuel scored while HUB active)
          - matches_played
          - ranking tiebreaker mirror (§5.3 / Table 10-1): ranking_score (avg RP),
            avg_match_points_excl_fouls, avg_auto_fuel, avg_tower_points
        Plus an OPR/DPR/CCWM hook (``opr_dpr_ccwm``) — a clearly-specified linear
        solve interface; the full least-squares solve is optional (stubbed below).

        KEYING FIX (§5.3): per-robot offensive stats are bucketed under a real team
        number ONLY when ``_identity_resolved`` is True (id_method in
        {anchor,elimination} with id_confidence ≥ MIN_ID_CONFIDENCE). Unresolved
        tracks are still surfaced — keyed ``track:N`` and tagged
        ``attribution_level='alliance'`` — but are EXCLUDED from per-team offensive
        means so one physical team never fractures across multiple track ids.
        """
        # gather per-team per-match samples
        per_team: dict[Any, dict] = {}
        for mm in match_metrics:
            for rid, rm in (mm.get("per_robot") or {}).items():
                resolved = self._identity_resolved(rm)
                team = rm.get("team")
                if resolved:
                    key = team
                    attribution = "team"
                else:
                    key = f"track:{rm.get('track_id')}"
                    attribution = "alliance"
                bucket = per_team.setdefault(key, {
                    "team": team if resolved else None,
                    "attribution_level": attribution,
                    "alliance": rm.get("alliance"),
                    "fuel": [], "auto_fuel": [], "active_fuel": [],
                    "wasted_fuel": [], "matches": 0,
                })
                bucket["matches"] += 1
                # OFFENSE is only credited when identity is resolved; otherwise the
                # bucket records the match (so matches_played is honest) but the
                # offensive samples are dropped from the per-team means.
                if not resolved:
                    continue
                fs = rm["fuel_scored"]["value"]
                bucket["fuel"].append(fs if fs is not None else 0)
                af = rm["auto_fuel"]
                bucket["auto_fuel"].append(af["value"] if af["available"] and af["value"] is not None else 0)
                acf = rm["active_shift_fuel"]
                wf = rm["wasted_inactive_fuel"]
                if acf["available"] and acf["value"] is not None:
                    bucket["active_fuel"].append(acf["value"])
                    bucket["wasted_fuel"].append(wf["value"] or 0)

        teams_out: dict = {}
        for key, b in per_team.items():
            attribution = b.get("attribution_level", "team")
            offense_available = attribution == "team"
            fuel = b["fuel"]
            avg_fuel = statistics.fmean(fuel) if fuel else 0.0
            fuel_var = statistics.pvariance(fuel) if len(fuel) >= 1 else 0.0
            avg_auto = statistics.fmean(b["auto_fuel"]) if b["auto_fuel"] else 0.0
            # active-shift conversion: active / (active + wasted), over matches we knew
            act_sum = sum(b["active_fuel"])
            waste_sum = sum(b["wasted_fuel"])
            conv_known = (act_sum + waste_sum) > 0 or bool(b["active_fuel"])
            conversion = (act_sum / (act_sum + waste_sum)) if (act_sum + waste_sum) > 0 else (
                1.0 if act_sum > 0 else None)
            _unattr = ("offense unattributed: track identity unresolved "
                       "(id_method not in {anchor,elimination} or id_confidence < "
                       f"{self.MIN_ID_CONFIDENCE})")
            teams_out[str(key)] = {
                "team": b["team"],
                "attribution_level": attribution,
                "alliance": b.get("alliance"),
                "matches_played": b["matches"],
                "avg_fuel_per_match": _metric(
                    round(avg_fuel, 3) if offense_available else None,
                    available=offense_available, unit="fuel",
                    needs=None if offense_available else _unattr,
                    provenance="mean over matches"),
                "fuel_variance": _metric(
                    round(fuel_var, 3) if offense_available else None,
                    available=offense_available, unit="fuel²",
                    needs=None if offense_available else _unattr,
                    provenance="population variance (consistency)"),
                "avg_auto_fuel": _metric(
                    round(avg_auto, 3) if offense_available else None,
                    available=offense_available, unit="fuel",
                    needs=None if offense_available else _unattr,
                    provenance="ranking tiebreaker #3 (§5.3)"),
                "active_shift_conversion": _metric(
                    None if (conversion is None or not offense_available) else round(conversion, 4),
                    available=offense_available and conversion is not None,
                    needs=(_unattr if not offense_available else
                           (None if conversion is not None
                            else "HUB-light detector to know active vs inactive shift fuel")),
                    unit="fraction"),
                # ── §5.3 ranking tiebreaker mirror — detector-gated parts honest ──
                "ranking_score": _metric(
                    None, available=False,
                    needs="per-match RP totals (need TOWER points for win/tie + traversal)"),
                "avg_match_points_excl_fouls": _metric(
                    None, available=False,
                    needs="complete match_points (TOWER) to average, fouls segregated"),
                "avg_tower_points": _metric(
                    None, available=False,
                    needs="climb-level detector (ranking tiebreaker #4)"),
                "climb_rate": _metric(
                    None, available=False, unit="fraction",
                    needs="climb-level detector (% matches with a successful climb)"),
            }

        return {
            "schema_version": "rebuilt-2026.§5.3",
            "teams": teams_out,
            "opr_dpr_ccwm": self.opr_dpr_ccwm(opr_inputs),
        }

    # ─────────────────────── OPR / DPR / CCWM hook ────────────────────────────
    def opr_dpr_ccwm(self, opr_inputs: Optional[list[dict]] = None) -> dict:
        """Offensive/Defensive Power Rating + Calculated Contribution to Win Margin.

        INTERFACE (the hook). OPR solves the over-determined linear system

            A · x = b

        where each row is one alliance-match: ``A[i, t] = 1`` if team ``t`` played
        on that alliance in match ``i`` (else 0), and ``b[i]`` is that alliance's
        scored points. The least-squares solution ``x = (AᵀA)⁻¹ Aᵀ b`` is each
        team's OPR. DPR uses opponents' scores; CCWM = OPR − DPR.

        ``opr_inputs`` is a list of alliance-match rows:
            {"teams": [t1, t2, t3], "points": <alliance points>,
             "opp_points": <opponent points>}

        We do NOT hardcode the solve to numpy here (keep this layer dependency-light
        and the full solve OPTIONAL). When ``opr_inputs`` is provided AND numpy is
        importable we run the real least-squares; otherwise we return a stub that
        documents the interface and reports ``available: False`` with the reason."""
        if not opr_inputs:
            return _metric(
                None, available=False,
                needs="≥1 alliance-match row {teams, points, opp_points}; "
                      "then this hook runs the AᵀA least-squares solve (numpy optional)")
        try:
            import numpy as np
        except ImportError:
            return _metric(
                None, available=False,
                needs="numpy for the least-squares solve (interface ready; rows supplied)")

        # build the team index
        teams = sorted({t for row in opr_inputs for t in row.get("teams", [])})
        idx = {t: i for i, t in enumerate(teams)}
        n_t = len(teams)
        if n_t == 0:
            return _metric(None, available=False, needs="alliance-match rows had no teams")
        A = np.zeros((len(opr_inputs), n_t))
        b_pts = np.zeros(len(opr_inputs))
        b_opp = np.zeros(len(opr_inputs))
        for i, row in enumerate(opr_inputs):
            for t in row.get("teams", []):
                A[i, idx[t]] = 1.0
            b_pts[i] = row.get("points", 0.0)
            b_opp[i] = row.get("opp_points", 0.0)
        try:
            opr, *_ = np.linalg.lstsq(A, b_pts, rcond=None)
            dpr, *_ = np.linalg.lstsq(A, b_opp, rcond=None)
        except np.linalg.LinAlgError as e:
            return _metric(None, available=False,
                           needs=f"well-conditioned system (solve failed: {e})")
        ccwm = opr - dpr
        return _metric(
            {
                "opr": {t: round(float(opr[idx[t]]), 3) for t in teams},
                "dpr": {t: round(float(dpr[idx[t]]), 3) for t in teams},
                "ccwm": {t: round(float(ccwm[idx[t]]), 3) for t in teams},
            },
            provenance="least-squares AᵀA solve over supplied alliance-match rows")


def _as_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return x
