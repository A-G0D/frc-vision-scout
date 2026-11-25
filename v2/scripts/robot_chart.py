"""Per-robot velocity chart (+ optional ball-shots overlay) from anchor tracks.

Consumes the bumper-bar anchor tracks JSON produced by scripts/robot_anchor.py and
renders, per the user's request, each robot's VELOCITY over time on one chart. If a
shot-events JSON is provided (or auto-found), it overlays, per robot, how many balls
were shot from that robot's general area — markers on the timeline + a per-robot tally.

Velocity is reported in px/s (measured directly) and an APPROX m/s using a stated,
clearly-labeled field scale (no homography here — px/s is the honest primary unit).
Shots are NEVER fabricated: if no shot-event file exists, the overlay panel says so.

Usage:
  python scripts/robot_chart.py [tracks.json] [--shots shots.json] [--out chart.png]
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# Rough field scale for an APPROX m/s axis: the 2026 field is ~16.5 m wide and spans
# roughly the inner ~1700 px of this broadcast's field band. Labeled approximate.
M_PER_PX = 16.5 / 1700.0


def speeds(hist, fps):
    """Return (t_seconds, speed_px_s) from a [[frame,x,y],...] history."""
    h = sorted(hist)
    t, v = [], []
    for i in range(1, len(h)):
        f0, x0, y0 = h[i - 1]
        f1, x1, y1 = h[i]
        df = f1 - f0
        if df <= 0:
            continue
        d = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        v.append(d / (df / fps))
        t.append(f1 / fps)
    return np.array(t), np.array(v)


def smooth(v, k=5):
    if len(v) < k:
        return v
    return np.convolve(v, np.ones(k) / k, mode="same")


def total_distance_px(hist):
    h = sorted(hist)
    return float(sum(((h[i][1] - h[i-1][1]) ** 2 + (h[i][2] - h[i-1][2]) ** 2) ** 0.5
                     for i in range(1, len(h))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", nargs="?",
                    default=str(ROOT / "data/scout_out/anchor/qual59_tracks.json"))
    ap.add_argument("--shots", default=None,
                    help="optional shot-events json: [{frame, x, alliance}] or {shots:[...]}")
    ap.add_argument("--out", default=str(ROOT / "data/scout_out/anchor/robot_chart.png"))
    args = ap.parse_args()

    data = json.loads(Path(args.tracks).read_text())
    fps = data.get("fps", 60.0)
    tracks = data["tracks"]
    col = {"blue": "#2d7dff", "red": "#ff3b3b"}

    # optional shots
    shots = None
    shots_path = args.shots
    if shots_path is None:
        auto = Path(args.tracks).with_name("shots.json")
        if auto.exists():
            shots_path = str(auto)
    if shots_path and Path(shots_path).exists():
        sj = json.loads(Path(shots_path).read_text())
        shots = sj.get("shots", sj) if isinstance(sj, (dict, list)) else None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9),
                                   gridspec_kw={"height_ratios": [3, 2]})
    fig.suptitle("FRC Vision Scout — per-robot velocity"
                 + ("  +  shots-from-area" if shots else ""),
                 fontsize=15, fontweight="bold")

    # ---- panel 1: velocity over time ----
    order = sorted(tracks, key=lambda r: -total_distance_px(tracks[r]["hist"]))
    for rid in order:
        tr = tracks[rid]
        if len(tr["hist"]) < 4:
            continue
        t, v = speeds(tr["hist"], fps)
        if len(t) == 0:
            continue
        ax1.plot(t, smooth(v), color=col.get(tr["alliance"], "gray"),
                 lw=1.8, alpha=0.85, label=f"{rid} ({tr['alliance']})")
    ax1.set_ylabel("speed (px/s)")
    ax1.set_xlabel("match time (s)")
    secax = ax1.secondary_yaxis(
        "right", functions=(lambda p: p * M_PER_PX, lambda m: m / M_PER_PX))
    secax.set_ylabel("approx speed (m/s)  [rough scale, no homography]")
    ax1.grid(alpha=0.25)
    ax1.legend(ncol=3, fontsize=9, loc="upper right")

    # ---- panel 2: distance traveled + shots-from-area ----
    rids = [r for r in order if len(tracks[r]["hist"]) >= 2]
    dist_m = [total_distance_px(tracks[r]["hist"]) * M_PER_PX for r in rids]
    colors = [col.get(tracks[r]["alliance"], "gray") for r in rids]
    bars = ax2.bar(rids, dist_m, color=colors, alpha=0.8)
    ax2.set_ylabel("distance traveled (approx m)")
    ax2.grid(alpha=0.25, axis="y")

    if shots:
        # attribute each shot to the nearest robot at the shot's frame (by x).
        tally = {r: 0 for r in rids}
        for s in shots:
            sf, sx = s.get("frame"), s.get("x")
            if sf is None or sx is None:
                continue
            best, bd = None, 1e9
            for r in rids:
                h = sorted(tracks[r]["hist"])
                near = min(h, key=lambda e: abs(e[0] - sf))
                if abs(near[0] - sf) > fps:        # within 1s
                    continue
                d = abs(near[1] - sx)
                if d < bd:
                    bd, best = d, r
            if best is not None:
                tally[best] += 1
        for bar, r in zip(bars, rids):
            if tally[r]:
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"⌖{tally[r]} shots", ha="center", va="bottom",
                         fontsize=10, fontweight="bold")
        ax2.set_title(f"bars = path length · ⌖N = balls shot from that robot's area "
                      f"({len(shots)} shots attributed)", fontsize=10)
    else:
        ax2.set_title("bars = path length · shots-from-area overlay: "
                      "NO shot-event file (run_live emits this stream) — not fabricated",
                      fontsize=10, color="#a33")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")
    print(f"robots charted: {len(rids)}  "
          f"({sum(1 for r in rids if tracks[r]['alliance']=='red')}R/"
          f"{sum(1 for r in rids if tracks[r]['alliance']=='blue')}B)")
    for r in rids:
        t, v = speeds(tracks[r]["hist"], fps)
        if len(v):
            print(f"  {r:3s} {tracks[r]['alliance']:4s} "
                  f"peak={v.max():.0f}px/s mean={v.mean():.0f}px/s "
                  f"dist={total_distance_px(tracks[r]['hist'])*M_PER_PX:.1f}m")


if __name__ == "__main__":
    main()
