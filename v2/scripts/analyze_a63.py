"""A6.3 measurement: read run_scout A63_DUMP JSONL files and report, per clip, whether any
robot field_xy_m lands outside [-0.5, length+0.5] x [-0.5, width+0.5] (raw vs clamped) and
the max center-field delta introduced by the clamp. Pure measurement — no grading vs model."""
from __future__ import annotations

import json
import sys

from frcscout.geometry.field import FieldModel
from frcscout.io.config import Config

TOL = 0.5
fm = FieldModel(Config.load("configs/deploy.yaml").get("field_config"))
L, W = fm.length_m, fm.width_m
LO_X, HI_X, LO_Y, HI_Y = -TOL, L + TOL, -TOL, W + TOL


def outside(p):
    return not (LO_X <= p[0] <= HI_X and LO_Y <= p[1] <= HI_Y)


for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    tag = [r for r in rows if r["geo"] == "tag"]
    px = [r for r in rows if r["geo"] == "px"]
    raw_out = [r for r in tag if outside(r["raw"])]
    clamp_out = [r for r in tag if outside(r["clamped"])]
    # center-field = raw already in-bounds; clamp must not move it (delta ~0).
    deltas = [max(abs(r["raw"][0] - r["clamped"][0]), abs(r["raw"][1] - r["clamped"][1]))
              for r in tag if not outside(r["raw"])]
    max_delta = max(deltas) if deltas else 0.0
    raw_max_x = max((r["raw"][0] for r in tag), default=0.0)
    print(f"\n=== {path} ===")
    print(f"  field bounds L={L:.3f} W={W:.3f}  bar X[{LO_X},{HI_X:.3f}] Y[{LO_Y},{HI_Y:.3f}]")
    print(f"  tag-projected robot samples: {len(tag)}   (px-fallback samples: {len(px)})")
    print(f"  RAW outside bar:     {len(raw_out)}  (max raw x = {raw_max_x:.2f} m)")
    print(f"  CLAMPED outside bar: {len(clamp_out)}   <-- must be 0")
    print(f"  center-field max clamp delta: {max_delta:.4f} m   <-- must be <= 0.1")
    if raw_out[:3]:
        print("  sample raw->clamped fixes:")
        for r in raw_out[:3]:
            print(f"    f{r['f']} rid{r['rid']}: "
                  f"({r['raw'][0]:.2f},{r['raw'][1]:.2f}) -> "
                  f"({r['clamped'][0]:.2f},{r['clamped'][1]:.2f})")
    px_out = [r for r in px if outside(r["raw"])]
    if px:
        print(f"  [honesty] px-fallback samples outside bar (NOT clamped, geo='px'): "
              f"{len(px_out)}/{len(px)}")
    verdict = "PASS" if (len(clamp_out) == 0 and max_delta <= 0.1) else "FAIL"
    print(f"  VERDICT: {verdict}")
