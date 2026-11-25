"""Bumper-bar anchor + alliance dedup -> clean <=6 robots -> track -> velocity.

Pipeline (per the user's design):
  1. YOLO robot boxes (recall is fine; ~6/frame on qual59).
  2. Each box -> bumper-bar ANCHOR: centroid-x of red/blue bumper pixels in the
     lower band, y = box bottom (ground contact). Alliance from the bumper colour.
  3. DEDUP the pile-ups: greedy by box area; drop a box whose anchor is within
     DEDUP_PX of an already-kept SAME-alliance anchor. A red bar and a blue bar
     close together both survive (different alliance) -> two robots, not one.
  4. Cap to 3 red + 3 blue (the 6-robot structural prior).
  5. Track anchors per alliance (nearest within a gate) -> per-robot velocity.

Outputs an annotated mp4 (anchors + ids + velocity vectors) and prints a per-robot
trajectory summary. A separate script turns the trajectory JSON into the chart.
"""
import sys
import json
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from frcscout.detect.robot_net import RobotNet  # noqa: E402

ROBOT_POLY = [(0.02, 0.25), (0.98, 0.25), (1.00, 0.64), (0.00, 0.64)]


def in_field(cx, cy, W, H):
    poly = np.array([(x * W, y * H) for x, y in ROBOT_POLY], np.float32)
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0


def bumper_anchor(frame, xyxy):
    """Return (anchor_x, anchor_y, alliance) using the bumper band, or None."""
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    al = RobotNet.bumper_alliance(frame, xyxy)
    crop = frame[y1:y2, x1:x2]
    H, W = crop.shape[:2]
    band = crop[int(H * 0.45):, :]            # lower bumper band
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(band, cv2.COLOR_BGR2LAB)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    a = lab[..., 1].astype(np.int16) - 128
    b = lab[..., 2].astype(np.int16) - 128
    colored = (s > 55) & (v > 45)
    if al == "red":
        m = colored & (((h <= 12) | (h >= 168)) | ((a >= 18) & (b > -15)))
    elif al == "blue":
        m = colored & (((h >= 95) & (h <= 135)) | (b <= -18))
    else:
        return None
    ys, xs = np.nonzero(m)
    if xs.size < 10:
        ax = (x1 + x2) / 2.0                   # fall back to box center-x
    else:
        ax = x1 + float(np.median(xs))
    ay = float(y2)                             # ground contact = box bottom
    return ax, ay, al


def dedup(anchors, dedup_px=70):
    """Greedy: keep biggest-box anchors; drop same-alliance neighbours within px."""
    anchors = sorted(anchors, key=lambda d: d["area"], reverse=True)
    kept = []
    for d in anchors:
        ok = True
        for k in kept:
            if k["alliance"] == d["alliance"]:
                if abs(k["x"] - d["x"]) < dedup_px and abs(k["y"] - d["y"]) < dedup_px:
                    ok = False
                    break
        if ok:
            kept.append(d)
    # cap 3 per alliance
    out = []
    for al in ("red", "blue"):
        out += [d for d in kept if d["alliance"] == al][:3]
    return out


class Tracks:
    """Per-alliance nearest-neighbour tracker (<=3 ids each)."""
    def __init__(self, gate=90):
        self.gate = gate
        self.next_id = {"red": 0, "blue": 0}
        self.tr = {}   # id -> dict(x,y,vx,vy,alliance,miss)

    def update(self, anchors, t):
        for al in ("red", "blue"):
            dets = [d for d in anchors if d["alliance"] == al]
            ids = [i for i, v in self.tr.items() if v["alliance"] == al]
            used = set()
            for i in ids:
                tv = self.tr[i]
                best, bd = None, self.gate
                for j, d in enumerate(dets):
                    if j in used:
                        continue
                    dist = ((tv["x"] - d["x"]) ** 2 + (tv["y"] - d["y"]) ** 2) ** 0.5
                    if dist < bd:
                        bd, best = dist, j
                if best is not None:
                    d = dets[best]
                    used.add(best)
                    a = 0.5
                    tv["vx"] = a * (d["x"] - tv["x"]) + (1 - a) * tv["vx"]
                    tv["vy"] = a * (d["y"] - tv["y"]) + (1 - a) * tv["vy"]
                    tv["x"], tv["y"], tv["miss"] = d["x"], d["y"], 0
                    tv["hist"].append((t, d["x"], d["y"]))
                else:
                    tv["miss"] += 1
            # births for unused dets, up to 3 alive per alliance
            alive = sum(1 for i in self.tr if self.tr[i]["alliance"] == al
                        and self.tr[i]["miss"] < 12)
            for j, d in enumerate(dets):
                if j in used:
                    continue
                if alive >= 3:
                    break
                nid = f"{al[0].upper()}{self.next_id[al]}"
                self.next_id[al] += 1
                self.tr[nid] = {"x": d["x"], "y": d["y"], "vx": 0.0, "vy": 0.0,
                                "alliance": al, "miss": 0, "hist": [(t, d["x"], d["y"])]}
                alive += 1
        # expire
        for i in list(self.tr):
            if self.tr[i]["miss"] > 30:
                del self.tr[i]

    def live(self):
        return {i: v for i, v in self.tr.items() if v["miss"] < 12}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "qual59"
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    from ultralytics import YOLO
    model = YOLO(str(ROOT / "weights/ball_openvino_model"), task="detect")
    cap = cv2.VideoCapture(str(ROOT / f"data/footage/{name}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    outdir = ROOT / "data/scout_out/anchor"
    outdir.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(outdir / f"{name}_anchor.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    tr = Tracks()
    col = {"red": (0, 0, 255), "blue": (255, 0, 0)}
    nlive = []
    all_hist = {}
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, imgsz=960, conf=0.12, device="cpu", verbose=False)[0]
        xy = r.boxes.xyxy.cpu().numpy()
        cl = r.boxes.cls.cpu().numpy().astype(int)
        anchors = []
        for b, c in zip(xy, cl):
            if c != 1:
                continue
            x1, y1, x2, y2 = b
            cx, cy = (x1 + x2) / 2, y2
            if not in_field(cx, cy, W, H):
                continue
            res = bumper_anchor(frame, b)
            if res is None:
                continue
            ax, ay, al = res
            anchors.append({"x": ax, "y": ay, "alliance": al,
                            "area": float((x2 - x1) * (y2 - y1))})
        kept = dedup(anchors)
        tr.update(kept, i)
        live = tr.live()
        nlive.append(len(live))
        vis = frame.copy()
        for rid, tv in live.items():
            c = col[tv["alliance"]]
            p = (int(tv["x"]), int(tv["y"]))
            cv2.circle(vis, p, 9, c, -1)
            cv2.circle(vis, p, 9, (255, 255, 255), 2)
            cv2.arrowedLine(vis, p, (int(p[0] + tv["vx"] * 4), int(p[1] + tv["vy"] * 4)),
                            (0, 255, 255), 2, tipLength=0.3)
            cv2.putText(vis, rid, (p[0] + 10, p[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        cv2.putText(vis, f"f{start+i}  robots={len(live)} (R{sum(1 for v in live.values() if v['alliance']=='red')}"
                    f"/B{sum(1 for v in live.values() if v['alliance']=='blue')})",
                    (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        vw.write(vis)
    for rid, tv in tr.tr.items():
        all_hist[rid] = {"alliance": tv["alliance"], "hist": tv["hist"]}
    vw.release()
    cap.release()
    (outdir / f"{name}_tracks.json").write_text(json.dumps(
        {"fps": fps, "start": start, "W": W, "H": H, "tracks": all_hist}))
    nlive = np.array(nlive)
    print(f"{name}: frames={len(nlive)}  live robots/frame "
          f"mean={nlive.mean():.1f} median={np.median(nlive):.0f} "
          f"min={nlive.min()} max={nlive.max()}")
    print(f"  total ids: {len(tr.tr)}  -> {outdir / f'{name}_anchor.mp4'}")
    print(f"  tracks json -> {outdir / f'{name}_tracks.json'}")


if __name__ == "__main__":
    main()
