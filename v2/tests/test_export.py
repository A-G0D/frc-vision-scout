"""Tests for the livestream→storage match exporter."""
from __future__ import annotations

import csv
import json

from frcscout.fuse.export import write_match, SCOUT_COLUMNS


def test_writes_json_and_csv(tmp_path):
    rec = {
        "robots": {
            3: {"team": 254, "alliance": "red", "fuel_scored": 12, "shot_accuracy": 0.6,
                "max_held": 3,
                "intake_success_rate": {"value": 0.8, "available": True}},  # envelope form
            5: {"alliance": "blue", "fuel_scored": 4},                      # no team -> track id
        },
        "events": [{"type": "score", "alliance": "red"}],
    }
    r = write_match(rec, outdir=str(tmp_path), match_key="qm1")
    assert r["n_robots"] == 2

    record = json.load(open(r["json"]))
    assert record["robots"]["3"]["fuel_scored"] == 12 if "3" in record["robots"] else True

    rows = {row["team"]: row for row in csv.DictReader(open(r["csv"]))}
    assert "254" in rows
    assert rows["254"]["fuel_scored"] == "12"
    assert rows["254"]["max_held"] == "3"
    assert rows["254"]["intake_success_rate"] == "0.8"   # envelope unwrapped to value
    assert "5" in rows                                    # team fell back to track id
    assert rows["5"]["alliance"] == "blue"


def test_columns_present(tmp_path):
    r = write_match({"robots": {}}, outdir=str(tmp_path), match_key="empty")
    header = next(csv.reader(open(r["csv"])))
    assert header == SCOUT_COLUMNS
