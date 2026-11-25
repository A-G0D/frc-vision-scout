"""Attributor tests — locks in the real window-match + softmax-credit behavior.

Regression target: the pre-fix `attribute()` was a TODO stub that returned
`score_events` unchanged (no robot_id ever assigned, no candidates reported,
no config wiring). Every assertion below fails against that stub.
"""
from __future__ import annotations

from frcscout.fuse.attribution import Attributor
from frcscout.io.types import Event


def _score(t_s, alliance="red", frame_idx=0, fuel_delta=1):
    return Event(t_s=t_s, frame_idx=frame_idx, type="score", alliance=alliance,
                 detail={"fuel_delta": fuel_delta})


def _shot(t_s, robot_id, alliance="red", frame_idx=0):
    return Event(t_s=t_s, frame_idx=frame_idx, type="shot_attempt", alliance=alliance,
                 robot_id=robot_id)


def test_single_candidate_gets_credited_with_timing_confidence():
    attributor = Attributor(window_s=3.0, lookahead_s=0.5)
    score = _score(t_s=10.0)
    shot = _shot(t_s=9.0, robot_id=3)  # dt=1.0 inside the 3.0s look-back window

    out = attributor.attribute([score], [shot])

    assert len(out) == 1
    assert out[0].robot_id == 3
    assert 0.0 < out[0].confidence < 1.0
    assert out[0].detail["candidates"] == [{"robot_id": 3, "probability": out[0].confidence}]
    # original score payload passes through untouched
    assert out[0].detail["fuel_delta"] == 1
    assert out[0].t_s == 10.0
    assert out[0].alliance == "red"


def test_exact_timing_match_gets_full_confidence():
    attributor = Attributor(window_s=3.0, lookahead_s=0.5)
    score = _score(t_s=10.0)
    shot = _shot(t_s=10.0, robot_id=5)

    out = attributor.attribute([score], [shot])

    assert out[0].robot_id == 5
    assert out[0].confidence == 1.0


def test_multiple_candidates_never_fabricate_single_winner():
    attributor = Attributor(window_s=3.0, lookahead_s=0.5)
    score = _score(t_s=10.0)
    shots = [_shot(t_s=9.5, robot_id=1), _shot(t_s=9.9, robot_id=2)]

    out = attributor.attribute([score], shots)

    assert len(out) == 1
    ev = out[0]
    assert ev.robot_id is None       # genuinely ambiguous -> no fabricated winner
    assert ev.confidence == 0.0
    candidates = ev.detail["candidates"]
    assert {c["robot_id"] for c in candidates} == {1, 2}
    total_prob = sum(c["probability"] for c in candidates)
    assert abs(total_prob - 1.0) < 1e-9
    # the closer-in-time shot (robot 2, dt=0.1) should carry more credit than
    # the farther one (robot 1, dt=0.5)
    probs = {c["robot_id"]: c["probability"] for c in candidates}
    assert probs[2] > probs[1]


def test_no_candidates_falls_back_to_unattributed_pool():
    attributor = Attributor(window_s=3.0, lookahead_s=0.5)
    score = _score(t_s=10.0)

    out = attributor.attribute([score], [])

    assert out[0].robot_id is None
    assert out[0].confidence == 0.0
    assert out[0].detail["candidates"] == []


def test_candidate_from_other_alliance_is_ignored():
    attributor = Attributor(window_s=3.0, lookahead_s=0.5)
    score = _score(t_s=10.0, alliance="red")
    shot = _shot(t_s=10.0, robot_id=9, alliance="blue")

    out = attributor.attribute([score], [shot])

    assert out[0].robot_id is None
    assert out[0].detail["candidates"] == []


def test_shot_outside_window_is_ignored():
    attributor = Attributor(window_s=3.0, lookahead_s=0.5)
    score = _score(t_s=10.0)
    shot = _shot(t_s=6.0, robot_id=4)  # dt=4.0, past the 3.0s window

    out = attributor.attribute([score], [shot])

    assert out[0].robot_id is None
    assert out[0].detail["candidates"] == []


def test_from_config_reads_score_attribution_section():
    class FakeConfig:
        def get(self, dotted, default=None):
            values = {"score_attribution.window_s": 7.0, "score_attribution.lookahead_s": 1.5}
            return values.get(dotted, default)

    attributor = Attributor.from_config(FakeConfig())

    assert attributor.window_s == 7.0
    assert attributor.lookahead_s == 1.5


def test_coverage_reports_named_fraction_per_alliance():
    attributed = [
        Event(t_s=1.0, frame_idx=0, type="score", alliance="red", robot_id=3,
              confidence=1.0, detail={"fuel_delta": 2}),
        Event(t_s=2.0, frame_idx=0, type="score", alliance="red", robot_id=None,
              confidence=0.0, detail={"fuel_delta": 2, "candidates": []}),
        Event(t_s=3.0, frame_idx=0, type="score", alliance="blue", robot_id=None,
              confidence=0.0, detail={"fuel_delta": 5, "candidates": []}),
    ]

    cov = Attributor.coverage(attributed)

    assert cov["red"] == 0.5   # 2 of 4 red points traced to a named robot
    assert cov["blue"] == 0.0
