"""Smoke test for live/run_live.py — end-to-end on qual59.mp4.

Runs 60 frames of the real pipeline and asserts that a published record lands in a
temporary store directory. Marked slow; skipped when the footage file is absent.

Run with:
    PYTHONPATH=src pytest tests/test_run_live.py -v -m slow
or simply:
    PYTHONPATH=src pytest tests/test_run_live.py -v        (slow tests included by default)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

# ── Paths ──────────────────────────────────────────────────────────────────
_TESTS = os.path.dirname(os.path.abspath(__file__))
_V2 = os.path.dirname(_TESTS)
_SRC = os.path.join(_V2, "src")
for _p in (_SRC, _V2):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FOOTAGE = os.path.join(_V2, "data", "footage", "qual59.mp4")
FOOTAGE_PRESENT = os.path.exists(FOOTAGE)

pytestmark = pytest.mark.slow


@pytest.mark.skipif(not FOOTAGE_PRESENT, reason="qual59.mp4 footage absent")
def test_run_live_smoke():
    """60-frame end-to-end: run_live.py must exit 0 and write a record to the store."""
    from live.run_live import run

    with tempfile.TemporaryDirectory(prefix="frc_live_smoke_") as store_dir:
        # Minimal argparse namespace
        class Args:
            source = FOOTAGE
            event_key = "2026ctwat"
            match_key = "2026ctwat_qm59"   # explicit: no OCR in test env
            store = store_dir
            frames = 60
            calib = None
            device = "cpu"
            snapshot_every = 999.0   # suppress snapshots during the smoke run

        rc = run(Args())
        assert rc == 0, f"run_live.run() returned non-zero exit code: {rc}"

        matches_dir = os.path.join(store_dir, "matches")
        assert os.path.isdir(matches_dir), "matches/ dir was not created in the store"

        records = os.listdir(matches_dir)
        assert len(records) >= 1, f"No records written to {matches_dir}"

        # The record must be valid JSON with the expected match_key
        record_path = os.path.join(matches_dir, records[0])
        with open(record_path, encoding="utf-8") as f:
            envelope = json.load(f)

        assert envelope.get("final") is True, "Record was not marked final"
        inner = envelope.get("metrics", {})
        assert inner.get("match_key") == "2026ctwat_qm59", (
            f"Expected match_key 2026ctwat_qm59, got {inner.get('match_key')}")

        # Schema version and per_alliance block must be present
        assert "schema_version" in inner, "schema_version missing from record"
        assert "per_alliance" in inner, "per_alliance block missing from record"
        per_alliance = inner["per_alliance"]
        assert "blue" in per_alliance and "red" in per_alliance, (
            f"Missing alliance blocks: {list(per_alliance.keys())}")

        # Vision fuel values must be non-negative integers/floats (even if 0)
        for alliance in ("blue", "red"):
            vision = per_alliance[alliance].get("vision", {})
            total_fuel = vision.get("total_fuel", {})
            assert total_fuel.get("available") is True, (
                f"[{alliance}] total_fuel.available is not True")
            assert isinstance(total_fuel.get("value"), (int, float)), (
                f"[{alliance}] total_fuel.value is not numeric: {total_fuel.get('value')}")

        # Index file must exist
        index_path = os.path.join(store_dir, "index.json")
        assert os.path.exists(index_path), "index.json not written"
        idx = json.loads(open(index_path).read())
        assert idx.get("count", 0) >= 1, "index.json count is 0"

        print(f"\n[smoke] OK — record: {record_path}")
        print(f"[smoke] match_key={inner.get('match_key')} "
              f"schema={inner.get('schema_version')}")
