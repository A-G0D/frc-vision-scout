"""Tests for the six-robot team-ID resolver (refine.identify6).

Pure synthetic data — plain ints/floats as OCR reads. No cv2 / numpy / easyocr / GPU.

Covers the four asks (anchor, elimination-fills-third, retroactive relabel, ambiguous->
unknown) plus the resolver's reason-for-existing extras over fuse.identity: noisy
time-series voting, off-roster + wrong-alliance read rejection, two-track team conflict,
confidence-margin ambiguity, noisy-OR confidence, and cascading elimination.
"""
from __future__ import annotations

from frcscout.refine.identify6 import (
    ANCHOR,
    ELIMINATION,
    UNKNOWN,
    BumperRead,
    apply,
    resolve_six,
    resolutions_to_dict,
)


OVERLAY = {"red": [254, 1114, 1678], "blue": [148, 973, 2056]}


# ───────────────────────────── anchors ───────────────────────────────────────
def test_single_confident_read_anchors():
    # one clean, high-confidence read locks the track<->team binding
    alliances = {1: "red"}
    reads = {1: [BumperRead(254, 0.95)]}
    res = resolve_six(alliances, reads, OVERLAY)
    assert res[1].team == 254
    assert res[1].method == ANCHOR
    assert res[1].confidence >= 0.55


def test_repeated_noisy_reads_vote_to_correct_anchor():
    # majority on-roster reads win even with a couple of off-roster / wrong reads mixed in
    alliances = {1: "red"}
    reads = {1: [254, 254, 254, 24, 999, 1114]}  # 24 & 999 are off-roster -> dropped
    res = resolve_six(alliances, reads, OVERLAY)
    assert res[1].team == 254 and res[1].method == ANCHOR
    # the lone on-roster competitor (1114) seen once must not beat triple-254
    assert res[1].votes.get(254, 0) > res[1].votes.get(1114, 0)


def test_off_roster_reads_are_discarded():
    # every read is a number NOT among the overlay's six -> no anchor, no guess
    alliances = {1: "red"}
    reads = {1: [3, 7, 42, 9999]}
    res = resolve_six(alliances, reads, OVERLAY)
    assert res[1].team is None and res[1].method == UNKNOWN
    assert res[1].n_reads == 0


def test_wrong_alliance_read_is_gated_out():
    # a BLUE team number read on a RED track is noise: discarded by the alliance gate
    alliances = {1: "red"}
    reads = {1: [BumperRead(2056, 0.99)]}  # 2056 is blue
    res = resolve_six(alliances, reads, OVERLAY)
    assert res[1].team is None and res[1].method == UNKNOWN


# ──────────────────────── elimination closure ────────────────────────────────
def test_elimination_fills_the_third_red():
    # 2 of 3 red anchored by OCR -> the 3rd red track is forced by elimination
    alliances = {1: "red", 2: "red", 3: "red"}
    reads = {1: [BumperRead(254, 0.95)], 2: [BumperRead(1114, 0.9)]}  # track 3: no reads
    res = resolve_six(alliances, reads, OVERLAY)
    assert res[1].team == 254 and res[1].method == ANCHOR
    assert res[2].team == 1114 and res[2].method == ANCHOR
    assert res[3].team == 1678 and res[3].method == ELIMINATION
    assert res[3].confidence > 0.5


def test_elimination_cascades_to_a_fixpoint():
    # only ONE red anchored, but the other red has reads split between the two FREE teams
    # equally -> it cannot anchor; with 2 unresolved tracks vs 2 free teams nothing is
    # forced. Now anchor a 2nd: that leaves 1 unresolved + 1 free -> elimination fires.
    alliances = {1: "red", 2: "red", 3: "red"}
    reads = {
        1: [BumperRead(254, 0.95)],
        2: [BumperRead(1114, 0.92)],
        # track 3 has only weak/none -> filled purely by elimination
    }
    res = resolve_six(alliances, reads, OVERLAY)
    assert {res[1].team, res[2].team} == {254, 1114}
    assert res[3].team == 1678 and res[3].method == ELIMINATION


def test_full_match_both_alliances_resolve():
    alliances = {1: "red", 2: "red", 3: "red", 4: "blue", 5: "blue", 6: "blue"}
    reads = {
        1: [254, 254], 2: [1114],            # red: 2 anchored, 1678 by elimination
        4: [148], 5: [973], 6: [2056],        # blue: all anchored
    }
    res = resolve_six(alliances, reads, OVERLAY)
    assert res[3].team == 1678 and res[3].method == ELIMINATION
    assert all(res[t].method == ANCHOR for t in (1, 2, 4, 5, 6))
    teams = {res[t].team for t in alliances}
    assert teams == {254, 1114, 1678, 148, 973, 2056}


# ──────────────────────── ambiguity stays unknown ────────────────────────────
def test_ambiguous_no_reads_all_unknown():
    # nothing read, 3 candidate teams -> cannot disambiguate -> all unknown
    alliances = {7: "red", 8: "red", 9: "red"}
    res = resolve_six(alliances, {}, OVERLAY)
    assert all(res[t].team is None and res[t].method == UNKNOWN for t in (7, 8, 9))


def test_split_reads_below_margin_stay_unknown():
    # a track whose reads are genuinely split between two on-roster teams (no clear
    # winner) must NOT guess -> stays unknown
    alliances = {1: "red"}
    reads = {1: [BumperRead(254, 0.8), BumperRead(1114, 0.8)]}
    res = resolve_six(alliances, reads, OVERLAY)
    assert res[1].team is None and res[1].method == UNKNOWN


def test_over_segmented_alliance_unknown():
    # Re-ID under-merge: 4 red track identities for 3 teams. 2 anchored, then 2 unresolved
    # tracks vs 1 free team -> not forced -> both unknown.
    alliances = {1: "red", 2: "red", 3: "red", 99: "red"}
    reads = {1: [BumperRead(254, 0.95)], 2: [BumperRead(1114, 0.95)]}
    res = resolve_six(alliances, reads, OVERLAY)
    assert res[1].method == ANCHOR and res[2].method == ANCHOR
    assert res[3].method == UNKNOWN and res[99].method == UNKNOWN


def test_two_tracks_claim_same_team_stronger_wins():
    # both red tracks 1 & 2 read 254; track 1's evidence is stronger -> it keeps 254,
    # track 2 is released. With 254 taken and 1114/1678 free, track 2 has no reads for
    # them and a 3rd unresolved track exists, so track 2 stays unknown (not mis-anchored).
    alliances = {1: "red", 2: "red", 3: "red"}
    reads = {
        1: [254, 254, 254],          # strong
        2: [BumperRead(254, 0.6)],   # weaker, same team
    }
    res = resolve_six(alliances, reads, OVERLAY)
    assert res[1].team == 254 and res[1].method == ANCHOR
    assert res[2].team != 254
    # 2 unresolved (2,3) vs 2 free (1114,1678) -> ambiguous -> both unknown
    assert res[2].method == UNKNOWN and res[3].method == UNKNOWN


def test_conflict_release_then_elimination_recovers():
    # tracks 1 & 2 both read 254 (1 stronger). Track 3 anchors 1114. That leaves track 2
    # as the ONLY unresolved red and 1678 the ONLY free team -> elimination recovers it.
    alliances = {1: "red", 2: "red", 3: "red"}
    reads = {
        1: [254, 254, 254],
        2: [BumperRead(254, 0.6)],
        3: [BumperRead(1114, 0.95)],
    }
    res = resolve_six(alliances, reads, OVERLAY)
    assert res[1].team == 254 and res[1].method == ANCHOR
    assert res[3].team == 1114 and res[3].method == ANCHOR
    assert res[2].team == 1678 and res[2].method == ELIMINATION


# ──────────────────────── confidence behaviour ───────────────────────────────
def test_noisy_or_confidence_grows_with_agreeing_reads():
    # two mediocre reads of the same team should give higher confidence than one
    one = resolve_six({1: "red"}, {1: [BumperRead(254, 0.5)]}, OVERLAY)
    two = resolve_six({1: "red"}, {1: [BumperRead(254, 0.5), BumperRead(254, 0.5)]}, OVERLAY)
    # one 0.5 read fails the min-conf lock; two combine via noisy-OR to 0.75 and lock
    assert one[1].method == UNKNOWN
    assert two[1].method == ANCHOR and two[1].confidence > 0.7


def test_alliance_none_uses_full_roster():
    # a track with unknown alliance can still anchor against the full six-team roster
    alliances = {1: None}
    res = resolve_six(alliances, {1: [BumperRead(2056, 0.95)]}, OVERLAY)
    assert res[1].team == 2056 and res[1].method == ANCHOR


def test_mapping_and_pair_read_forms_accepted():
    # reads given as {team: weight} and as (team, conf) tuples both work
    r_map = resolve_six({1: "red"}, {1: {254: 0.95}}, OVERLAY)
    r_pair = resolve_six({1: "red"}, {1: [(254, 0.95)]}, OVERLAY)
    assert r_map[1].team == 254 and r_pair[1].team == 254


# ──────────────────────── retroactive relabel ────────────────────────────────
def test_retroactive_apply_relabels_early_events():
    # an event at an EARLY frame (before any number was read for track 3) gets the team
    # from the FINAL elimination resolution -> retroactive.
    alliances = {1: "red", 2: "red", 3: "red"}
    reads = {1: [BumperRead(254, 0.95)], 2: [BumperRead(1114, 0.95)]}
    res = resolve_six(alliances, reads, OVERLAY)
    record = {
        "events": [
            {"type": "shot", "frame_idx": 30, "robot_id": 3},     # early, pre-anything
            {"type": "pickup", "frame_idx": 900, "robot_id": 1},
        ],
        "robots": {3: {"shot_attempts": 5}},
    }
    apply(record, res)
    assert record["events"][0]["team"] == 1678                    # track 3 by elimination
    assert record["events"][0]["id_method"] == ELIMINATION
    assert record["events"][1]["team"] == 254
    assert record["robots"][3]["team"] == 1678


def test_resolutions_to_dict_shape():
    res = resolve_six({1: "red"}, {1: [BumperRead(254, 0.95)]}, OVERLAY)
    d = resolutions_to_dict(res)
    assert set(d[1].keys()) >= {"team", "method", "confidence"}
    assert d[1]["team"] == 254 and d[1]["method"] == ANCHOR
