"""MetricsAggregator tests — pure synthetic match_record dicts, no model/GPU/cv2.

We hand-build finalized match_records (the shape StateFuser produces:
events = list of Event.__dict__, robots = PossessionTracker.stats_dict() with
identity stamped, plus an optional overlay timeseries) and assert:

  * per-robot fuel / intake / shot rollups are correct;
  * RP flags (ENERGIZED / SUPERCHARGED) trip exactly at the §3.2 thresholds;
  * detector-gated metrics report available:False (not a fabricated 0);
  * the TBA-comparable block is present and JSON-serializable;
  * team aggregates over 2 synthetic matches compute avg + variance correctly.
"""
from __future__ import annotations

import json

from frcscout.metrics import (
    MetricsAggregator,
    MetricsConfig,
    REGIONAL_THRESHOLDS,
)


FPS = 60.0


# ─────────────────────────── synthetic record builders ───────────────────────
def _robot_stats(track_id, alliance, team, *, intake_attempts=0, intake_successes=0,
                 shot_attempts=0, shots_made=0, shots_resolvable=0,
                 possession_time_s=0.0, cycle_times=(), time_to_intake=(),
                 balls_peak=0):
    """Mirror RobotStats.to_dict() + apply_identities stamping."""
    n_tti = list(time_to_intake)
    n_cyc = list(cycle_times)
    return {
        "robot_id": track_id,
        "alliance": alliance,
        "team_number": team,
        "intake_attempts": intake_attempts,
        "intake_successes": intake_successes,
        "intake_success_rate": (intake_successes / intake_attempts) if intake_attempts else 0.0,
        "mean_time_to_intake_s": (sum(n_tti) / len(n_tti)) if n_tti else None,
        "time_to_intake_s": n_tti,
        "shot_attempts": shot_attempts,
        "shots_made": shots_made,
        "shots_resolvable": shots_resolvable,
        "shot_accuracy": (shots_made / shots_resolvable) if shots_resolvable else None,
        "miss_rate": (1.0 - shots_made / shots_resolvable) if shots_resolvable else None,
        "shot_successes": shots_made,
        "shot_success_rate": (shots_made / shot_attempts) if shot_attempts else 0.0,
        "held_count": 0,
        "possession_time_s": possession_time_s,
        "balls_in_radius_peak": balls_peak,
        "avg_cycle_time_s": (sum(n_cyc) / len(n_cyc)) if n_cyc else None,
        "cycle_times_s": n_cyc,
        # identity (apply_identities stamps these)
        "team": team,
        "id_method": "anchor",
        "id_confidence": 1.0,
    }


def _shot_success_event(t_s, track_id, alliance):
    return {"t_s": t_s, "frame_idx": int(t_s * FPS), "type": "shot_success",
            "alliance": alliance, "robot_id": track_id, "confidence": 1.0,
            "detail": {"via": "hub"}, "team": None}


def _score_event(t_s, alliance, fuel_delta=1):
    return {"t_s": t_s, "frame_idx": int(t_s * FPS), "type": "score",
            "alliance": alliance, "robot_id": None, "confidence": 1.0,
            "detail": {"fuel_delta": fuel_delta}}


def _overlay_samples(alliance_fuel_curve, *, alliance="blue"):
    """Build a per-frame overlay timeseries from a list of (t_s, phase, cum_fuel)."""
    ts = []
    for t_s, phase, fuel in alliance_fuel_curve:
        ts.append({
            "t_s": t_s, "frame_idx": int(t_s * FPS), "phase": phase,
            f"{alliance}_fuel": fuel, "blue_score": None, "red_score": None,
        })
    return ts


# ────────────────────────────────── tests ────────────────────────────────────
def test_per_robot_fuel_intake_shot_rollup():
    """A crafted record yields correct per-robot fuel/intake/shot rollups."""
    robots = {
        7: _robot_stats(7, "blue", 1234,
                        intake_attempts=5, intake_successes=4,
                        shot_attempts=6, shots_made=3, shots_resolvable=5,
                        possession_time_s=12.5,
                        cycle_times=(4.0, 6.0), time_to_intake=(0.5, 1.5),
                        balls_peak=3),
    }
    # 3 made shots: one in auto (t<20), two in teleop
    events = [
        _shot_success_event(10.0, 7, "blue"),
        _shot_success_event(60.0, 7, "blue"),
        _shot_success_event(90.0, 7, "blue"),
    ]
    rec = {"events": events, "robots": robots, "shots": []}

    out = MetricsAggregator().aggregate(rec)
    r = out["per_robot"]["7"]

    assert r["team"] == 1234
    assert r["fuel_scored"]["value"] == 3
    assert r["intake_attempts"]["value"] == 5
    assert r["intake_successes"]["value"] == 4
    assert abs(r["intake_success_rate"]["value"] - 0.8) < 1e-9
    assert r["shot_attempts"]["value"] == 6
    assert r["shots_made"]["value"] == 3
    # accuracy = made / resolvable = 3/5
    assert abs(r["shot_accuracy"]["value"] - 0.6) < 1e-9
    assert abs(r["miss_rate"]["value"] - 0.4) < 1e-9
    assert abs(r["mean_cycle_time_s"]["value"] - 5.0) < 1e-9
    assert abs(r["mean_time_to_intake_s"]["value"] - 1.0) < 1e-9
    assert r["possession_time_s"]["value"] == 12.5
    assert r["balls_in_radius_peak"]["value"] == 3
    # phase split from event t_s (no overlay): 1 auto, 2 teleop
    assert r["auto_fuel"]["value"] == 1
    assert r["auto_fuel"]["available"] is True
    assert r["teleop_fuel"]["value"] == 2
    # scoring rate = 3 fuel / (160s / 60) = 1.125/min
    assert abs(r["scoring_rate_per_min"]["value"] - (3 / (160 / 60))) < 1e-6


def test_unresolvable_shot_accuracy_is_none_not_zero():
    """Shot accuracy must be None (unknown) when no shot was resolvable — never 0."""
    robots = {3: _robot_stats(3, "red", 9999, shot_attempts=4, shots_made=0,
                              shots_resolvable=0)}
    out = MetricsAggregator().aggregate({"events": [], "robots": robots})
    r = out["per_robot"]["3"]
    assert r["shot_accuracy"]["value"] is None
    assert r["shot_accuracy"]["available"] is False
    assert "needs" in r["shot_accuracy"]
    assert r["miss_rate"]["value"] is None


def test_detector_gated_metrics_report_unavailable():
    """Unavailable metrics report available:False with a needs note, not 0/fake."""
    robots = {1: _robot_stats(1, "blue", 100)}
    out = MetricsAggregator().aggregate({"events": [], "robots": robots})
    r = out["per_robot"]["1"]
    gated = ["sourcing_mix", "climb_level", "climb_start_time_s", "time_to_climb_s",
             "climb_success", "defense_time_s", "pins_committed", "pins_drawn",
             "foul_candidates", "fouls_drawn", "disables", "auto_mobility",
             "preload_count", "endgame_tower_points", "balls_in_radius_mean"]
    for key in gated:
        assert key in r, f"missing gated metric {key}"
        assert r[key]["available"] is False, f"{key} should be unavailable"
        assert r[key]["value"] is None, f"{key} must not fabricate a value"
        assert r[key]["needs"], f"{key} needs a one-line note"

    al = out["per_alliance"]["blue"]
    for key in ["tower_points", "auto_tower_l1_count", "tower_climbs",
                "rp_win_tie", "rp_total", "minor_fouls_received",
                "major_fouls_received", "foul_points_credited", "cards"]:
        assert al[key]["available"] is False
        assert al[key]["value"] is None
        assert al[key]["needs"]


def test_rp_flags_trip_at_thresholds():
    """ENERGIZED/SUPERCHARGED RP flip exactly at the §3.2 Regional thresholds
    (100 / 360), config-driven."""
    cfg = MetricsConfig()  # default Regional
    agg = MetricsAggregator(cfg)
    e = REGIONAL_THRESHOLDS["energized"]      # 100
    s = REGIONAL_THRESHOLDS["supercharged"]   # 360

    def alliance_at(fuel):
        # one score event per fuel, all teleop
        events = [_score_event(50.0 + i * 0.01, "blue") for i in range(fuel)]
        out = agg.aggregate({"events": events, "robots": {}})
        return out["per_alliance"]["blue"]

    just_under = alliance_at(e - 1)
    assert just_under["rp_energized"]["value"] is False
    assert just_under["rp_supercharged"]["value"] is False

    at_energized = alliance_at(e)
    assert at_energized["rp_energized"]["value"] is True
    assert at_energized["rp_supercharged"]["value"] is False

    at_super = alliance_at(s)
    assert at_super["rp_energized"]["value"] is True
    assert at_super["rp_supercharged"]["value"] is True


def test_rp_thresholds_are_config_driven():
    """Champs thresholds change the trip points (not hardcoded)."""
    champs = MetricsConfig.for_level("champs")    # energized 360, super 500
    agg = MetricsAggregator(champs)
    events = [_score_event(50.0 + i * 0.01, "red") for i in range(360)]
    out = agg.aggregate({"events": events, "robots": {}})
    al = out["per_alliance"]["red"]
    assert al["rp_energized"]["value"] is True       # 360 >= 360
    assert al["rp_supercharged"]["value"] is False   # 360 < 500


def test_tba_comparable_block_present_and_serializable():
    """TBA-comparable block exists with the required keys and is JSON-serializable."""
    ts = _overlay_samples([
        (5.0, "auto", 0), (15.0, "auto", 8),
        (60.0, "teleop", 40), (130.0, "teleop", 55),
    ], alliance="blue") + _overlay_samples([
        (5.0, "auto", 0), (15.0, "auto", 5),
        (60.0, "teleop", 30),
    ], alliance="red")
    # add scores to the last samples for overlay truth
    ts[-1]["red_score"] = 30
    ts[3]["blue_score"] = 55
    rec = {"events": [], "robots": {}}

    out = MetricsAggregator().aggregate(rec, overlay_timeseries=ts)
    tba = out["tba_comparable"]

    for al in ("blue", "red"):
        assert set(["auto_fuel", "teleop_fuel", "total_fuel",
                    "estimated_match_points"]).issubset(tba[al])
    assert "blue_score_est" in tba and "red_score_est" in tba

    # blue auto fuel = 8, teleop = 55-8 = 47, total = 55
    assert tba["blue"]["auto_fuel"] == 8
    assert tba["blue"]["total_fuel"] == 55
    assert tba["blue"]["estimated_match_points"] == 55   # 1 pt/fuel
    # fuel-only flag is honest (tower not yet detected)
    assert tba["blue"]["fuel_only"] is True
    assert tba["blue"]["estimated_match_points_full"] is None

    # whole output round-trips through JSON
    s = json.dumps(out)
    assert isinstance(s, str) and len(s) > 0


def test_team_aggregate_avg_and_variance_over_two_matches():
    """Team aggregate over 2 synthetic matches computes avg + variance correctly."""
    agg = MetricsAggregator()

    # team 1234 scores 4 fuel in match A, 8 in match B
    def match_with(track_id, team, n_made, t_base):
        robots = {track_id: _robot_stats(track_id, "blue", team,
                                         shots_made=n_made, shots_resolvable=n_made,
                                         shot_attempts=n_made)}
        events = [_shot_success_event(t_base + i, track_id, "blue") for i in range(n_made)]
        return agg.aggregate({"events": events, "robots": robots})

    m_a = match_with(5, 1234, 4, 50.0)
    m_b = match_with(5, 1234, 8, 50.0)

    team_out = agg.team_aggregates([m_a, m_b])
    t = team_out["teams"]["1234"]

    assert t["matches_played"] == 2
    # avg fuel = (4 + 8)/2 = 6
    assert abs(t["avg_fuel_per_match"]["value"] - 6.0) < 1e-9
    # population variance of [4, 8] = ((4-6)^2 + (8-6)^2)/2 = (4+4)/2 = 4
    assert abs(t["fuel_variance"]["value"] - 4.0) < 1e-9
    # climb_rate gated (no detector)
    assert team_out["teams"]["1234"]["climb_rate"]["available"] is False
    assert team_out["opr_dpr_ccwm"]["available"] is False  # no opr_inputs supplied


def test_opr_hook_runs_when_rows_supplied():
    """OPR/DPR/CCWM hook solves the linear system when alliance-match rows + numpy
    are available; otherwise reports available:False honestly."""
    agg = MetricsAggregator()
    rows = [
        {"teams": [1, 2, 3], "points": 60, "opp_points": 40},
        {"teams": [1, 4, 5], "points": 50, "opp_points": 55},
        {"teams": [2, 4, 6], "points": 45, "opp_points": 50},
        {"teams": [3, 5, 6], "points": 70, "opp_points": 30},
    ]
    res = agg.opr_dpr_ccwm(rows)
    if res["available"]:
        # numpy present: a real solve with opr/dpr/ccwm per team
        assert set(res["value"].keys()) == {"opr", "dpr", "ccwm"}
        assert set(res["value"]["opr"].keys()) == {1, 2, 3, 4, 5, 6}
        # ccwm == opr - dpr
        for t in (1, 2, 3, 4, 5, 6):
            assert abs(res["value"]["ccwm"][t]
                       - (res["value"]["opr"][t] - res["value"]["dpr"][t])) < 1e-6
    else:
        assert "numpy" in res["needs"].lower()

    # no rows -> documented stub, unavailable
    empty = agg.opr_dpr_ccwm(None)
    assert empty["available"] is False
    assert empty["needs"]


def test_alliance_fuel_phase_split_from_overlay():
    """Per-alliance auto/teleop fuel attribution follows the overlay phase tags."""
    ts = _overlay_samples([
        (5.0, "auto", 0), (18.0, "auto", 12),     # 12 fuel in auto
        (40.0, "teleop", 30),                      # +18 teleop
        (125.0, "endgame", 45),                    # +15 endgame (counts as teleop period)
    ], alliance="blue")
    out = MetricsAggregator().aggregate({"events": [], "robots": {}},
                                        overlay_timeseries=ts)
    al = out["per_alliance"]["blue"]
    assert al["auto_fuel_scored"]["value"] == 12
    # teleop period total = teleop(18) + endgame(15) = 33
    assert al["teleop_fuel_scored"]["value"] == 33
    assert al["active_fuel_total"]["value"] == 45
