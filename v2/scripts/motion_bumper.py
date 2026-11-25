"""Motion-gated bumper-bar robot detection on a CONSECUTIVE window of qual59.

Discriminator (per user): only robots and balls move. So:
  moving (MOG2 foreground)  AND  red/blue bumper colour (LAB+HSV)  -> robot bumper.
Static red/blue (HUBs, banners, scoring tables) is removed by the motion gate;
moving yellow balls are removed by the colour gate. Each surviving blob's
bottom-centre is the ground anchor. Goal: a clean ~3 red + ~3 blue.

Usage: python scripts/motion_bumper.py [start_frame] [n_frames]
"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data/footage/qual59.mp4"
ROBOT_POLY = [(0.02, 0.25), (0.98, 0.25), (1.00, 0.64), (0.00, 0.64)]


def field_mask(h, w):
    poly = np.array([(int(x * w), int(y * h)) for x, y in ROBOT_POLY], np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [poly], 255)
    return m


def color_masks(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    a = lab[..., 1].astype(np.int16) - 128
    b = lab[..., 2].astype(np.int16) - 128
    colored = (s > 55) & (v > 45)
    red = colored & (((h <= 12) | (h >= 168)) | ((a >= 18) & (b > -15)))
    blue = colored & (((h >= 95) & (h <= 135)) | (b <= -18))
    return red.astype(np.uint8) * 255, blue.astype(np.uint8) * 255


def detect(mask, frame_hw, min_area_frac=0.0004, max_area_frac=0.05):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 7))  # wide: bumper bars
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    fh, fw = frame_hw
    out = []
    for lbl in range(1, num):
        x, y, w, hgt, area = stats[lbl]
        af = (w * hgt) / float(fh * fw)
        if not (min_area_frac <= af <= max_area_frac):
            continue
        out.append({"xyxy": (x, y, x + w, y + hgt), "area": int(area),
                    "anchor": (int(x + w / 2), int(y + hgt))})
    out.sort(key=lambda d: d["area"], reverse=True)
    return out


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    mog = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=30, detectShadows=False)
    outdir = ROOT / "data/scout_out/motion_bumper"
    outdir.mkdir(parents=True, exist_ok=True)
    fm = None
    counts = []
    saved = 0
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        if fm is None:
            H, W = frame.shape[:2]
            fm = field_mask(H, W)
        fg = mog.apply(frame)                       # 0/255 moving foreground
        fg = (fg > 200).astype(np.uint8) * 255      # drop shadow greys
        fg = cv2.bitwise_and(fg, fm)
        # dilate motion a little so a bumper that's only partly "moving" is kept
        fg = cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        red, blue = color_masks(frame)
        red = cv2.bitwise_and(cv2.bitwise_and(red, fm), fg)
        blue = cv2.bitwise_and(cv2.bitwise_and(blue, fm), fg)
        if i < 60:                                  # let MOG2 learn the background
            continue
        rb = detect(red, (H, W))
        bb = detect(blue, (H, W))
        counts.append((len(rb), len(bb)))
        if i % 20 == 0 and saved < 6:               # render a few
            vis = frame.copy()
            cv2.polylines(vis, [np.array([(int(x*W), int(y*H)) for x, y in ROBOT_POLY], np.int32)],
                          True, (0, 255, 0), 1)
            for d in rb[:5]:
                x1, y1, x2, y2 = d["xyxy"]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.circle(vis, d["anchor"], 7, (0, 0, 255), -1)
            for d in bb[:5]:
                x1, y1, x2, y2 = d["xyxy"]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.circle(vis, d["anchor"], 7, (255, 0, 0), -1)
            cv2.putText(vis, f"f{start+i} red={len(rb)} blue={len(bb)}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imwrite(str(outdir / f"mb_{start+i:06d}.png"), vis)
            saved += 1
    cap.release()
    if counts:
        rs = [c[0] for c in counts]
        bs = [c[1] for c in counts]
        print(f"frames analysed: {len(counts)}  (start={start})")
        print(f"red blobs  : mean={np.mean(rs):.1f} median={np.median(rs):.0f} "
              f"min={min(rs)} max={max(rs)}")
        print(f"blue blobs : mean={np.mean(bs):.1f} median={np.median(bs):.0f} "
              f"min={min(bs)} max={max(bs)}")
        print(f"renders -> {outdir}")


if __name__ == "__main__":
    main()
