"""Structural-validity checks — encode the exact v1 failure modes as failing cases so a
regression can never silently ship again."""
from __future__ import annotations

from frcscout.eval.tba_verify import _is_impossible


def test_two_team_alliance_is_impossible():
    assert _is_impossible({"comp_level": "qm", "blue_teams": [254, 1114], "red_teams": [9, 8, 7]})


def test_team_on_both_alliances_is_impossible():
    # v1 LBR5M13: team 3 on both sides
    assert _is_impossible({"comp_level": "sf", "blue_teams": [3, 6, 228], "red_teams": [3, 4909, 4572]})


def test_rank_column_misread_is_impossible():
    # v1 playoff: single/double-digit "teams" were rank-column reads
    assert _is_impossible({"comp_level": "f", "blue_teams": [4909, 4572, 6], "red_teams": [28, 324, 6328]})


def test_valid_qual_passes():
    assert not _is_impossible({"comp_level": "qm",
                               "blue_teams": [10266, 558, 2712],
                               "red_teams": [4909, 7907, 10133]})
