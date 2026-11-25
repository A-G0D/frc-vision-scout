"""
run_track_verify.py - Run RobotTracker over the wide segment, produce annotated
frames, and report stability statistics. Called from the agent's self-verify pass.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import cv2
import numpy as np
from sim import detect as _detect
from sim.track import RobotTracker

VIDEO = "C:/Users/agodavarthy/Claude/projects/frc-vision-scout/data/sim/match_full.mp4"
OUT_DIR = "C:/Users/agodavarthy/Claude/projects/frc-vision-scout/track_verify"
WIDE_END = 4170   # ~139s at 30fps = the static wide-camera segment

os.makedirs(OUT_DIR, exist_ok=True)

# ---- fixed color palette for IDs 0-11 ----
ID_COLORS = [
    (0, 200, 255),   # 0 cyan
    (0, 100, 255),   # 1 orange
    (200, 0, 255),   # 2 magenta
    (0, 255, 100),   # 3 lime
    (255, 180, 0),   # 4 sky-blue
    (100, 0, 255),   # 5 purple
    (0, 255, 255),   # 6 yellow
    (0, 255, 0),     # 7 green
    (255, 0, 180),   # 8 pink
    (180, 255, 0),   # 9 chartreuse
    (255, 80, 80),   # 10 salmon
    (80, 80, 255),   # 11 periwinkle
]

def color_for(tid):
    return ID_COLORS[tid % len(ID_COLORS)]

print("Building background over wide segment (0..139s)...")
bg = _detect.build_background(VIDEO, 0.0, 139.0)
print("Background built, shape:", bg.shape)

cap = cv2.VideoCapture(VIDEO)

rt = RobotTracker(
    max_age=45,           # ~1.5s coasting budget for occlusion
    min_hits=4,           # confirm after 4 consecutive detections
    max_dist=80.0,        # gate: 80px max predicted-to-detected centroid distance
    iou_weight=0.5,
    max_tracks=6,
    reacquire_frames=450, # 15s graveyard window for re-ID after long gaps
    reacquire_dist=130.0, # px radius for graveyard re-match
)

# Save frames at multiple times across the wide segment
SAVE_FRAMES = sorted({
    150, 300, 450, 600, 900, 1200, 1500, 1800,
    2100, 2400, 2700, 3000, 3300, 3600, 3900, 4100
})
save_set = set(SAVE_FRAMES)

trails = {}     # id -> deque of last N centroids
TRAIL_LEN = 35

# Stats
track_life = {}  # id -> [first_frame, last_frame, hits, alliance_votes]
confirmed_per_frame = []
detection_per_frame = []

fi = 0
while fi < WIDE_END:
    ok, fr = cap.read()
    if not ok:
        break

    dets = _detect.detect_robots(fr, bg)
    tracked = rt.update(fi, dets)

    detection_per_frame.append(len(dets))
    confirmed_per_frame.append(len(tracked))

    # Update trails and stats
    for t in tracked:
        tid = t["id"]
        cx, cy = int(t["centroid"][0]), int(t["centroid"][1])
        if tid not in trails:
            trails[tid] = []
        trails[tid].append((cx, cy))
        if len(trails[tid]) > TRAIL_LEN:
            trails[tid].pop(0)

        if tid not in track_life:
            track_life[tid] = [fi, fi, 1, {"blue": 0, "red": 0}]
        else:
            track_life[tid][1] = fi
            track_life[tid][2] += 1
        track_life[tid][3][t["alliance"]] += 1

    if fi in save_set:
        h, w = fr.shape[:2]
        ann = fr.copy()

        # Draw alliance-colored background region faintly to orient viewer
        for t in tracked:
            x, y, bw, bh = t["bbox"]
            tid = t["id"]
            al = t["alliance"]
            # Alliance-tinted fill
            color = (255, 80, 80) if al == "blue" else (80, 80, 255)
            overlay = ann.copy()
            cv2.rectangle(overlay, (x, y), (x+bw, y+bh), color, -1)
            ann = cv2.addWeighted(overlay, 0.15, ann, 0.85, 0)

        for t in tracked:
            x, y, bw, bh = t["bbox"]
            tid = t["id"]
            col = color_for(tid)
            al_ch = "B" if t["alliance"] == "blue" else "R"
            lbl = f"{tid}:{al_ch}"

            # Draw trail
            pts = trails.get(tid, [])
            for k in range(1, len(pts)):
                alpha = k / len(pts)
                tc = tuple(int(c * alpha) for c in col)
                cv2.line(ann, pts[k-1], pts[k], tc, 2)

            # Bold box
            cv2.rectangle(ann, (x, y), (x+bw, y+bh), col, 2)
            # Label background
            (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(ann, (x, max(0, y-lh-4)), (x+lw+2, y), col, -1)
            cv2.putText(ann, lbl, (x+1, max(lh, y-2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Centroid dot
            cx_, cy_ = int(t["centroid"][0]), int(t["centroid"][1])
            cv2.circle(ann, (cx_, cy_), 3, col, -1)

        # Status bar
        t_s = fi / 30.0
        nb = sum(1 for t in tracked if t["alliance"] == "blue")
        nr = sum(1 for t in tracked if t["alliance"] == "red")
        status = f"f{fi} t={t_s:.1f}s  tracked:{len(tracked)} (B:{nb} R:{nr})  raw_dets:{len(dets)}"
        cv2.rectangle(ann, (0, h-20), (w, h), (0, 0, 0), -1)
        cv2.putText(ann, status, (4, h-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        # Scale up 2.5x so labels are readable
        big = cv2.resize(ann, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_NEAREST)
        fname = os.path.join(OUT_DIR, f"f{fi:05d}.png")
        cv2.imwrite(fname, big)

    fi += 1

cap.release()

# ---- Report ----
cpf = np.array(confirmed_per_frame)
dpf = np.array(detection_per_frame)

print(f"\n=== RobotTracker Report ({WIDE_END} frames, ~139s) ===")
print(f"Total unique IDs ever spawned: {len(track_life)}")
print(f"Alliance-vote flips after confirm: {rt.id_switches}")
print(f"\nConfirmed tracks/frame: mean={cpf.mean():.2f}  median={np.median(cpf):.0f}  "
      f"min={cpf.min()}  max={cpf.max()}")
print(f"Raw detections/frame:   mean={dpf.mean():.2f}  median={np.median(dpf):.0f}")
print(f"\nFrames with >=5 confirmed tracks: {(cpf >= 5).sum()} "
      f"({100*(cpf>=5).mean():.1f}%)")
print(f"Frames with >=4 confirmed tracks: {(cpf >= 4).sum()} "
      f"({100*(cpf>=4).mean():.1f}%)")
print(f"Frames with >=3 confirmed tracks: {(cpf >= 3).sum()} "
      f"({100*(cpf>=3).mean():.1f}%)")

print(f"\nLong-lived tracks (span >= 300 frames / ~10s):")
long_lived = {}
for tid, (f0, f1, hits, avotes) in sorted(track_life.items()):
    span = f1 - f0
    if span >= 300:
        long_lived[tid] = (f0, f1, hits, avotes)
        al = "blue" if avotes["blue"] >= avotes["red"] else "red"
        print(f"  id {tid:3d}: frames {f0:4d}-{f1:4d}  span={span:4d}  "
              f"hits={hits:4d}  alliance={al}  "
              f"vote_pct={100*max(avotes['blue'],avotes['red'])/max(1,hits):.0f}%")

print(f"\nTotal long-lived tracks (span>=300): {len(long_lived)}")
print(f"ID switches (alliance vote flips after confirm): {rt.id_switches}")
print(f"\nAnnotated frames saved to: {OUT_DIR}")
