"""Tests for live/match_segment.py — segmentation + identification core (no OCR)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from live.match_segment import (  # noqa: E402
    MatchSegmenter, OverlayObs, identify_segment, MatchSegment,
)


def _match_stream(start_idx, comp, num, blue, red, b_final, r_final, n_idle_after=8):
    """Synthesize: a live match (timer 150->0) then a results screen."""
    obs = []
    i = start_idx
    # countdown live bar
    for t in range(150, -1, -10):
        phase = "auto" if t > 135 else ("endgame" if t < 30 else "teleop")
        obs.append(OverlayObs(idx=i, layout="live_bar", timer_s=float(t), phase=phase,
                              blue_score=min(b_final, (150 - t)), red_score=min(r_final, (150 - t)),
                              blue_teams=blue, red_teams=red))
        i += 1
    # results screen (repeat so debounce confirms)
    for _ in range(5):
        obs.append(OverlayObs(idx=i, layout="qual_results", comp_level=comp, match_number=num,
                              blue_score=b_final, red_score=r_final,
                              blue_teams=blue, red_teams=red))
        i += 1
    # idle gap
    for _ in range(n_idle_after):
        obs.append(OverlayObs(idx=i, layout="unknown"))
        i += 1
    return obs, i


def test_single_match_segmented_and_identified():
    seg = MatchSegmenter(event_key="2026ctwat")
    stream, _ = _match_stream(0, "qm", 59, [254, 1678, 118], [33, 1114, 469], 162, 48)
    finals = [f for o in stream if (f := seg.push(o))]
    assert len(finals) == 1
    m = finals[0]
    assert m.match_key == "2026ctwat_qm59"
    assert m.comp_level == "qm" and m.match_number == 59
    assert m.blue_teams == [254, 1678, 118]
    assert m.red_teams == [33, 1114, 469]
    assert m.final_blue_score == 162 and m.final_red_score == 48
    assert m.id_confidence > 0.9


def test_two_matches_no_double_count():
    seg = MatchSegmenter(event_key="2026ctwat")
    s1, nxt = _match_stream(0, "qm", 59, [1, 2, 3], [4, 5, 6], 100, 90)
    s2, _ = _match_stream(nxt, "qm", 60, [7, 8, 9], [10, 11, 12], 77, 88)
    finals = [f for o in (s1 + s2) if (f := seg.push(o))]
    assert [m.match_key for m in finals] == ["2026ctwat_qm59", "2026ctwat_qm60"]
    assert len(seg.completed) == 2


def test_replay_does_not_start_new_match():
    """A brief results->live->results flicker must not spawn a phantom match."""
    seg = MatchSegmenter(event_key="2026ctwat")
    stream, nxt = _match_stream(0, "qm", 5, [1, 2, 3], [4, 5, 6], 50, 60)
    # inject a 2-frame live flicker (below debounce) during the idle tail
    flicker = [OverlayObs(idx=nxt + j, layout="live_bar", timer_s=140.0) for j in range(2)]
    finals = [f for o in (stream + flicker) if (f := seg.push(o))]
    assert len(finals) == 1  # the flicker (2 < confirm=3) starts nothing


def test_flush_closes_open_match():
    seg = MatchSegmenter(event_key="2026ctwat")
    # a match with no trailing results screen (stream cut off)
    obs = [OverlayObs(idx=i, layout="live_bar", timer_s=float(150 - 10 * i),
                      comp_level="qm", match_number=12, blue_teams=[1, 2, 3], red_teams=[4, 5, 6])
           for i in range(6)]
    for o in obs:
        seg.push(o)
    m = seg.flush()
    assert m is not None and m.match_key == "2026ctwat_qm12"


def test_identify_handles_missing_event_key():
    m = MatchSegment(start_idx=0, obs=[
        OverlayObs(idx=0, comp_level="qm", match_number=3, blue_teams=[1, 2, 3]),
    ])
    identify_segment(m, None)
    assert m.match_key is None  # no event key -> no key, but identity still voted
    assert m.comp_level == "qm" and m.match_number == 3
