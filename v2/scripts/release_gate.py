"""Production release gate (RUBRIC_v2 D6) — fails LOUDLY on any regression.

One command that asserts the live pipeline is shippable:
  1. unit suite is green (subprocess pytest),
  2. end-to-end run_live on a short ACTIVE qual59 window publishes a record,
  3. the published record is HONEST (every unavailable metric carries {available:false,
     needs}; no fabricated 0 for an unknown) and TBA-joined,
  4. the verified live metric holds: published BLUE auto-fuel == TBA hubScore.autoCount
     within tolerance (the broadcast-vs-TBA check that proves the chain end to end).

Exit code 0 = PASS (shippable); non-zero = FAIL with the specific failing gate.
Usage: python scripts/release_gate.py [--skip-suite] [--skip-e2e]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
fails: list[str] = []
def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


def _walk_honesty(node, path="") -> list[str]:
    """Find fabrication: a metric envelope with available:false but no needs, or a
    bare value that is non-None while available:false (a fabricated unknown)."""
    bad: list[str] = []
    if isinstance(node, dict):
        if "available" in node and "value" in node:   # a metric envelope
            if node["available"] is False:
                if not node.get("needs"):
                    bad.append(f"{path}: available:false without needs")
                if node.get("value") not in (None, [], {}):
                    bad.append(f"{path}: available:false but value={node['value']!r} (fabricated?)")
        for k, v in node.items():
            bad += _walk_honesty(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            bad += _walk_honesty(v, f"{path}[{i}]")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-suite", action="store_true")
    ap.add_argument("--skip-e2e", action="store_true")
    args = ap.parse_args()
    print("=== FRC Vision Scout — RELEASE GATE ===")

    # ── Gate 1: unit suite ────────────────────────────────────────────────
    if not args.skip_suite:
        print("\n[1] Unit suite (pytest -q -m 'not slow')")
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "not slow"],
                           cwd=str(ROOT), capture_output=True, text=True, timeout=900)
        tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]
        check("unit_suite_green", r.returncode == 0, tail[0][:120])
    else:
        print("\n[1] unit suite SKIPPED")

    # ── Gate 2-4: end-to-end run_live on a short active window ─────────────
    store = ROOT / "data/_gate_store"
    if not args.skip_e2e:
        print("\n[2] End-to-end run_live (short active window)")
        footage = ROOT / "data/footage/qual59.mp4"
        if not footage.exists():
            check("footage_present", False, str(footage))
        else:
            env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
            cmd = [sys.executable, "live/run_live.py", "--source", str(footage),
                   "--event-key", "2026ctwat", "--match-key", "2026ctwat_qm59",
                   "--start", "2600", "--frames", "250", "--store", str(store)]
            r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900)
            check("run_live_exit0", r.returncode == 0,
                  (r.stderr.strip().splitlines()[-1:] or [""])[0][:120])
            rec_path = store / "matches" / "2026ctwat_qm59.json"
            check("record_published", rec_path.exists(), str(rec_path))
            if rec_path.exists():
                rec = json.loads(rec_path.read_text())
                # The publisher wraps the aggregator record under "metrics":
                # {match_key, updated_at, final, metrics: {per_alliance, provenance, ...}}
                agg = rec.get("metrics", rec)
                pa = agg.get("per_alliance", {})
                check("record_has_alliances", "blue" in pa and "red" in pa)
                check("record_tba_joined",
                      agg.get("provenance", {}).get("tba_available") is True
                      or any("tba" in pa.get(al, {}) for al in ("blue", "red")))
                # honesty audit
                bad = _walk_honesty(agg)
                check("no_fabrication", not bad,
                      f"{len(bad)} issue(s): {bad[0] if bad else ''}")
                # verified live metric: BLUE auto-fuel vs TBA
                try:
                    cmp_blue = pa["blue"]["comparison"]["fuel_auto"]["comparison"]  # noqa: F841
                    v, t = cmp_blue.get("vision_est"), cmp_blue.get("tba_actual")
                    ok = (v is not None and t is not None and abs(v - t) <= max(3, 0.1 * t))
                    check("verified_auto_fuel_matches_tba", ok, f"vision={v} tba={t}")
                except Exception as e:
                    check("verified_auto_fuel_matches_tba", False, f"missing comparison: {e}")
    else:
        print("\n[2] end-to-end SKIPPED")

    print("\n=== GATE RESULT ===")
    if fails:
        print(f"FAIL — {len(fails)} gate(s) failed: {', '.join(fails)}")
        return 1
    print("PASS — pipeline is shippable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
