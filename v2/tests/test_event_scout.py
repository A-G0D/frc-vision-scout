"""Tests for the EVENT-LEVEL scouting product (R7):

  * MetricsAggregator.team_aggregates KEYING FIX — offense is attributed to a real
    team only when identity is resolved; unresolved tracks land in track:N buckets
    tagged attribution_level='alliance' with offense available:False (one physical
    team never splits across track ids).
  * build_event_scout adapter turns the raw run_scout match_*.json shape into the
    aggregate() match_record shape, rolls up the event, feeds OPR from AUTHORITATIVE
    scores (never the fuel estimate), and emits a roster row per real team.
  * make_picklist composite z-score ranking + low_confidence flag + availability
    strip that is never blank/0.
  * the TBA validation step reports winner/score agreement with N and the fuel-only
    caveat, and flags mid-match snapshots instead of scoring them.

All synthetic; no model / cv2 / GPU.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from frcscout.metrics import MetricsAggregator, MetricsConfig

# scripts/ is not a package — load the two modules by path.
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bes = _load("build_event_scout")
mpl = _load("make_picklist")


# ───────────────────────── synthetic raw match builder ───────────────────────

def _raw_robot(track_id, alliance, *, team=None, id_method="unknown",
               id_conf=0.0, shots_made=0, shot_attempts=0, intake_attempts=0,
               intake_successes=0, possession_time_s=0.0):
    """Mirror a raw run_scout per-track dict (scalar keys + a nested metrics block
    that the adapter ignores)."""
    return {
        "team": team, "alliance": alliance, "id_method": id_method,
        "id_confidence": id_conf, "phase": "HOLDING", "held": 0,
        "intake_attempts": intake_attempts, "intake_successes": intake_successes,
        "intake_success_rate": {"value": None, "available": False, "needs": "x"},
        "mean_time_to_intake_s": None, "max_held": 0,
        "possession_time_s": possession_time_s, "balls_shot_at_once": 0,
        "shot_attempts": shot_attempts, "shots_made": shots_made,
        "shot_accuracy": {"value": None, "available": False, "needs": "x"},
        "climb_level": 0,
        "fuel_scored": {"value": None, "available": False, "needs": "x"},
        "metrics": {"jitter_index": {"value": 0.1, "available": True}},
    }


def _raw_match(match_key, *, identified=False):
    """6-robot raw match. identified=True stamps anchor identities on all tracks."""
    blue_teams, red_teams = [101, 102, 103], [201, 202, 203]
    robots = {}
    for i in range(6):
        alliance = "red" if i < 3 else "blue"
        team = None
        method, conf = "unknown", 0.0
        if identified:
            team = (red_teams if i < 3 else blue_teams)[i % 3]
            method, conf = "anchor", 1.0
        robots[str(i)] = _raw_robot(
            i, alliance, team=team, id_method=method, id_conf=conf,
            shots_made=2 * (i + 1), shot_attempts=3 * (i + 1),
            intake_attempts=i + 1, intake_successes=i + 1, possession_time_s=5.0)
    return {
        "match_key": match_key, "fps": 60.0, "robots": robots, "events": [],
        "overlay": {"match_label": match_key, "blue_teams": blue_teams,
                    "red_teams": red_teams, "blue_score": 40, "red_score": 60,
                    "timer_s": 0.0},
        "expected_alliances": {"red": red_teams, "blue": blue_teams},
    }


# ─────────────────────────── keying-fix unit tests ───────────────────────────

def test_unresolved_identity_not_attributed_to_team():
    """All tracks team=null/unknown -> every bucket is track:N, attribution alliance,
    offense available:False. NO synthetic team number is invented."""
    agg = MetricsAggregator()
    raw = _raw_match("q1", identified=False)
    rec = bes._to_match_record(raw)
    mm = agg.aggregate(rec)
    table = agg.team_aggregates([mm])
    teams = table["teams"]
    assert all(k.startswith("track:") for k in teams), teams.keys()
    for row in teams.values():
        assert row["attribution_level"] == "alliance"
        assert row["team"] is None
        assert row["avg_fuel_per_match"]["available"] is False
        assert row["avg_fuel_per_match"]["value"] is None
        assert row["avg_fuel_per_match"]["needs"]


def test_resolved_identity_is_attributed_and_does_not_split():
    """anchor identity + conf 1.0 -> offense attributed to the real team number, and
    the SAME team across two matches collapses to ONE bucket (no track-id split)."""
    agg = MetricsAggregator()
    m1 = agg.aggregate(bes._to_match_record(_raw_match("q1", identified=True)))
    m2 = agg.aggregate(bes._to_match_record(_raw_match("q2", identified=True)))
    table = agg.team_aggregates([m1, m2])
    teams = table["teams"]
    # 6 real team numbers, none synthetic
    assert set(teams) == {"101", "102", "103", "201", "202", "203"}
    for row in teams.values():
        assert row["attribution_level"] == "team"
        assert row["matches_played"] == 2          # collapsed, not split
        assert row["avg_fuel_per_match"]["available"] is True
        assert row["avg_fuel_per_match"]["value"] is not None


def test_low_confidence_is_gated_out():
    agg = MetricsAggregator()
    raw = _raw_match("q1", identified=True)
    # downgrade one track below the confidence floor
    raw["robots"]["0"]["id_confidence"] = 0.1
    table = agg.team_aggregates([agg.aggregate(bes._to_match_record(raw))])
    # track 0 (team 201) falls back to a track bucket
    assert any(k.startswith("track:") for k in table["teams"])
    assert "201" not in table["teams"]


def test_resolved_method_must_be_authoritative():
    agg = MetricsAggregator()
    raw = _raw_match("q1", identified=True)
    raw["robots"]["1"]["id_method"] = "prior"   # not anchor/elimination
    table = agg.team_aggregates([agg.aggregate(bes._to_match_record(raw))])
    assert "202" not in table["teams"]
    assert any(k.startswith("track:") for k in table["teams"])


# ──────────────────────────── adapter / OPR tests ────────────────────────────

def test_adapter_collapses_envelope_fields():
    raw = _raw_match("q1")
    rec = bes._to_match_record(raw)
    r0 = rec["robots"]["0"]
    assert r0["shot_accuracy"] is None          # envelope -> scalar None
    assert r0["shots_made"] == 2
    assert "metrics" not in r0
    assert r0["track_id"] == 0


def test_opr_uses_authoritative_scores_not_fuel(tmp_path):
    sd = tmp_path / "scout_out"
    sd.mkdir()
    for k in ("qual1", "qual2"):
        (sd / f"match_{k}.json").write_text(json.dumps(_raw_match(k)))
    doc = bes.build_event(str(sd))
    opr = doc["opr_dpr_ccwm"]
    assert opr["available"] is True
    # OPR was fed from overlay blue/red_score (40/60), NOT our fuel estimate
    assert opr["value"]["opr"]            # non-empty
    # every authoritative-roster team has a row
    assert {"101", "102", "103", "201", "202", "203"} <= set(doc["teams"])


def test_no_authoritative_score_excludes_match_from_opr(tmp_path):
    sd = tmp_path / "scout_out"
    sd.mkdir()
    raw = _raw_match("qual1")
    raw["overlay"]["blue_score"] = None
    raw["overlay"]["red_score"] = None
    (sd / "match_qual1.json").write_text(json.dumps(raw))
    doc = bes.build_event(str(sd))
    # no usable score row -> OPR has no rows -> available False (no fabrication)
    assert doc["opr_dpr_ccwm"]["available"] is False


def test_csv_cell_never_blank_for_unavailable():
    cell = bes._cell({"value": None, "available": False, "needs": "climb-level detector"})
    assert cell.startswith("n/a")
    assert bes._cell({"value": 12.0, "available": True}) == "12.0"


# ─────────────────────────────── picklist tests ──────────────────────────────

def _event_doc(tmp_path, identified=False):
    sd = tmp_path / "scout_out"
    sd.mkdir()
    for k in ("qual1", "qual2"):
        (sd / f"match_{k}.json").write_text(json.dumps(_raw_match(k, identified=identified)))
    return bes.build_event(str(sd))


def test_picklist_ranks_and_flags_low_confidence(tmp_path):
    doc = _event_doc(tmp_path, identified=False)
    ranked = mpl.build_rankings(doc)
    # only real team numbers are draftable (track:N rows skipped)
    assert all(isinstance(r["team"], int) for r in ranked)
    assert len(ranked) == 6
    assert [r["rank"] for r in ranked] == sorted(r["rank"] for r in ranked)
    # offense unattributed + 2-match OPR -> all low confidence, honestly
    assert all(r["low_confidence"] for r in ranked)
    assert all(r["confidence_reasons"] for r in ranked)


def test_availability_strip_never_blank(tmp_path):
    doc = _event_doc(tmp_path, identified=False)
    ranked = mpl.build_rankings(doc)
    cards = mpl.render_cards(doc, ranked)
    assert "n/a" in cards
    # no metric line should end in a bare value-less colon / blank
    for line in cards.splitlines():
        if line.strip().startswith(("Avg ", "Fuel ", "Active", "Climb", "Ranking")):
            tail = line.split(maxsplit=0)
            assert line.strip()[-1] not in (":",) and len(line.strip()) > 10


def test_picklist_writes_three_files(tmp_path):
    doc = _event_doc(tmp_path, identified=True)
    p = tmp_path / "event_teams.json"
    p.write_text(json.dumps(doc))
    rc = mpl.main(["--event-teams", str(p), "--out-dir", str(tmp_path)])
    assert rc == 0
    for f in ("rankings.csv", "picklist.csv", "team_cards.txt"):
        assert (tmp_path / f).exists() and (tmp_path / f).stat().st_size > 0


def test_zscores_handle_sparse_population():
    # <2 present -> neutral 0 for present, None for absent (no div-by-zero)
    z = mpl._zscores({"a": 5.0, "b": None})
    assert z["a"] == 0.0 and z["b"] is None
    # zero variance -> all 0
    z2 = mpl._zscores({"a": 3.0, "b": 3.0, "c": 3.0})
    assert all(v == 0.0 for v in z2.values())


# ─────────────────────────────── validation test ─────────────────────────────

def test_validate_flags_snapshot_reads(tmp_path):
    sd = tmp_path / "scout_out"
    sd.mkdir()
    # one FINAL read (timer 0, matches TBA winner) + one mid-match SNAPSHOT
    final = _raw_match("qual1")
    final["overlay"].update(blue_score=92, red_score=176, timer_s=0.0)
    snap = _raw_match("qual2")
    snap["overlay"].update(blue_score=10, red_score=5, timer_s=30.0)
    (sd / "match_qual1.json").write_text(json.dumps(final))
    (sd / "match_qual2.json").write_text(json.dumps(snap))

    tba = {"matches": [
        {"key": "evt_qm1", "comp_level": "qm", "match_number": 1,
         "alliances": {"blue": {"team_keys": ["frc101", "frc102", "frc103"], "score": 92},
                       "red": {"team_keys": ["frc201", "frc202", "frc203"], "score": 176}}},
        {"key": "evt_qm2", "comp_level": "qm", "match_number": 2,
         "alliances": {"blue": {"team_keys": ["frc101", "frc102", "frc103"], "score": 88},
                       "red": {"team_keys": ["frc201", "frc202", "frc203"], "score": 70}}},
    ]}
    tp = tmp_path / "tba.json"
    tp.write_text(json.dumps(tba))

    v = bes.validate_vs_tba(str(sd), str(tp), "evt")
    assert v["n_compared"] == 2
    assert v["n_snapshot_reads"] == 1          # qual2 timer != 0
    assert v["n_final_reads"] == 1
    # the one final read agrees on winner (red wins both our 92/176 and TBA 92/176)
    assert v["winner_agreement"] == 1.0
    assert v["score_exact_agreement"] == 1.0
    assert "fuel-only" in v["caveat"].lower() or "fuel" in v["caveat"].lower()


def test_tba_key_mapping():
    assert bes._tba_key_for({"match_key": "qual56"}, "2026ctwat") == "2026ctwat_qm56"
    assert bes._tba_key_for({"match_key": "q9"}, "2026ctwat") == "2026ctwat_qm9"
    assert bes._tba_key_for({"match_key": ""}, "2026ctwat") is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
