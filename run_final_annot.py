"""Final quality annotation of key shot moments."""
import src.sim.track_ball as tb
import src.sim.detect as det
import cv2
import numpy as np
import os

video = 'data/sim/match_full.mp4'
out_dir = 'ball_track_final'
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video)
tracker = tb.make_ball_tracker()

rng = np.random.default_rng(42)
colors = {}

def color_for(i):
    if i not in colors:
        colors[i] = tuple(int(v) for v in rng.integers(80, 230, 3))
    return colors[i]

# Key frames: actual shot event moments + a few quiet frames for context
key_frames = {96, 134, 197, 209, 295, 413, 597, 1414, 1515, 1521, 1580, 1610, 1625, 2228, 2418}
seen_shots = set()

fi = 0
while fi < 4166:
    ok, frame = cap.read()
    if not ok:
        break
    dets = det.detect_balls(frame)
    tracks = tracker.update(fi, dets)

    for t in tracker._tracks:
        if t.shot and t.id not in seen_shots:
            seen_shots.add(t.id)

    if fi in key_frames:
        ann = frame.copy()
        # Draw all non-shot ball tracks
        for d in tracks:
            if d['shot']:
                continue
            cx, cy = d['centroid']
            col = color_for(d['id'])
            cv2.circle(ann, (cx, cy), max(3, d['r']), col, 1)
            vx, vy = d['vel']
            sp = abs(vx) + abs(vy)
            if sp > 1.5:
                ex = int(cx + vx * 3)
                ey = int(cy + vy * 3)
                cv2.arrowedLine(ann, (cx, cy), (ex, ey), col, 1, tipLength=0.35)

        # Draw shot trajectories prominently
        for s in tracker.live_shots():
            path = s['path']
            npath = len(path)
            for k in range(1, npath):
                alpha = k / max(1, npath)
                c = (0, int(80 * (1 - alpha)), int(200 + 55 * alpha))
                cv2.line(ann, path[k - 1], path[k], c, 2)
            cx, cy = s['centroid']
            sid = s['id']
            cv2.circle(ann, (cx, cy), max(5, s['r'] + 3), (0, 0, 255), 2)
            label = 'SHOT ' + str(sid)
            cv2.putText(ann, label, (cx + 6, max(8, cy - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)

        n_shots_total = len(seen_shots)
        n_live = len(tracker.live_shots())
        status = ('f' + str(fi) + ' t=' + str(round(fi / 30.0, 1)) + 's'
                  + ' balls:' + str(len(dets))
                  + ' tracks:' + str(len(tracks))
                  + ' shots_total:' + str(n_shots_total)
                  + ' live:' + str(n_live))
        cv2.putText(ann, status, (4, 354),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
        big = cv2.resize(ann, None, fx=2.0, fy=2.0,
                         interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(out_dir, 'f%05d.png' % fi), big)

    fi += 1

cap.release()
print('Saved annotated frames to', out_dir)
print('Total shots flagged:', len(seen_shots))
