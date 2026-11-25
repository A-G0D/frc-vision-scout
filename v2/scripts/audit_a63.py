"""A6.3 auditor: measure off-field robot field_xy_m counts + center-field deltas
from the A63_DUMP JSONL files. Pure measurement; no model calls."""
import json, sys, os
sys.path.insert(0, "src")
from frcscout.geometry.field import FieldModel

field = FieldModel("configs/field_2026_rebuilt.yaml")
L, W = field.length_m, field.width_m
TOL = 0.5
xlo, xhi = -TOL, L + TOL
ylo, yhi = -TOL, W + TOL
print(f"field bounds: L={L} W={W} -> x[{xlo},{xhi}] y[{ylo},{yhi}]")

for tag, path in [("qual59", "a63_qual59"), ("match_1080", "a63_m1080"), ("final1", "a63_final1")]:
    p = f"data/scout_out/{path}.jsonl"
    if not os.path.exists(p):
        print(f"{tag}: MISSING {p}")
        continue
    rows = []
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    frames = sorted(set(r["f"] for r in rows))
    nframes = len(frames)
    # off-field on CLAMPED output
    off = [r for r in rows if not (xlo <= r["clamped"][0] <= xhi and ylo <= r["clamped"][1] <= yhi)]
    # raw off-field (how many WOULD be off without clamp)
    raw_off = [r for r in rows if not (xlo <= r["raw"][0] <= xhi and ylo <= r["raw"][1] <= yhi)]
    # center-field: raw already in-bounds -> delta clamp vs raw must be ~0
    max_delta = 0.0
    n_inb = 0
    for r in rows:
        rx, ry = r["raw"]
        if xlo <= rx <= xhi and ylo <= ry <= yhi:
            n_inb += 1
            d = max(abs(r["clamped"][0] - rx), abs(r["clamped"][1] - ry))
            max_delta = max(max_delta, d)
    print(f"\n{tag}: rows={len(rows)} frames_covered={nframes} (span {frames[0]}..{frames[-1]})")
    print(f"  raw off-field:     {len(raw_off)}")
    print(f"  CLAMPED off-field: {len(off)}   <-- must be 0")
    print(f"  in-bounds rows:    {n_inb}  max clamp delta on them: {max_delta:.6f} m  <-- must be <=0.1")
    if off:
        print("  SAMPLE off-field clamped:", off[:3])
