"""Tests for retroactive identity resolution + elimination closure."""
from __future__ import annotations

from frcscout.fuse.identity import resolve_identities, apply_identities


OVERLAY = {"red": [254, 1114, 1678], "blue": [148, 973, 2056]}


def test_elimination_fills_the_third_red():
    # 2 of 3 red anchored -> the 3rd red track is forced by elimination
    alliances = {1: "red", 2: "red", 3: "red"}
    anchored = {1: 254, 2: 1114}
    res = resolve_identities(alliances, anchored, OVERLAY)
    assert res[1].team == 254 and res[1].method == "anchor"
    assert res[3].team == 1678 and res[3].method == "elimination"
    assert res[3].confidence > 0.5


def test_all_anchored():
    alliances = {4: "blue", 5: "blue", 6: "blue"}
    anchored = {4: 148, 5: 973, 6: 2056}
    res = resolve_identities(alliances, anchored, OVERLAY)
    assert all(res[t].method == "anchor" for t in (4, 5, 6))


def test_ambiguous_left_unknown():
    # nothing anchored, 3 candidates -> cannot disambiguate -> all unknown
    alliances = {7: "red", 8: "red", 9: "red"}
    res = resolve_identities(alliances, {}, OVERLAY)
    assert all(res[t].team is None and res[t].method == "unknown" for t in (7, 8, 9))


def test_over_segmented_alliance_unknown():
    # Re-ID under-merged: 4 red track identities for 3 teams -> can't assign -> unknown
    alliances = {1: "red", 2: "red", 3: "red", 99: "red"}
    anchored = {1: 254, 2: 1114}
    res = resolve_identities(alliances, anchored, OVERLAY)
    # 2 unassigned tracks (3, 99) vs 1 remaining team (1678) -> ambiguous, not forced
    assert res[3].method == "unknown" and res[99].method == "unknown"


def test_retroactive_apply_relabels_early_events():
    # an event recorded at an EARLY frame (before the number was read) gets the team
    # applied from the FINAL resolution -> retroactive.
    alliances = {1: "red", 2: "red", 3: "red"}
    anchored = {1: 254, 2: 1114}            # discovered mid/late match
    res = resolve_identities(alliances, anchored, OVERLAY)
    record = {
        "events": [
            {"type": "shot_success", "frame_idx": 30, "robot_id": 3},   # early, pre-anchor
            {"type": "intake_success", "frame_idx": 900, "robot_id": 1},
        ],
        "robots": {3: {"shot_attempts": 5, "shot_successes": 4}},
    }
    apply_identities(record, res)
    assert record["events"][0]["team"] == 1678          # track 3 -> 1678 by elimination
    assert record["events"][0]["id_method"] == "elimination"
    assert record["events"][1]["team"] == 254
    assert record["robots"][3]["team"] == 1678
