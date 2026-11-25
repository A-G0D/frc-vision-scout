"""Validate bumper-bar robot detection on real qual59 frames.

The hypothesis (per the user): in this tilted-sideline view each robot's strongest,
most-separable feature is its red/blue bumper BAR. Segment red blobs (expect ~3) and
blue blobs (expect ~3), take each blob's bottom-centre as the ground anchor. This
sidesteps the YOLO box merge/split that maps 3 robots to 1 box (or 1 robot to 2).

Renders: bars (filled), anchor dots, and per-frame red/blue counts. Also tries the
wash-resistant LAB+HSV mask vs the proposer's stock HSV so we can SEE which separates
6 robots better — measured, not claimed.
"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from frcscout.refine.classical_robot import ClassicalRobotProposer  # noqa: E402

# field band where robots live (image-space ROBOT_POLY from run_scout)
ROBOT_POLY = [(0.02, 0.25), (0.98, 0.25), (1.00, 0.64), (0.00, 0.64)]


def field_mask(h, w):
    poly = np.array([(int(x * w), int(y * h)) for x, y in ROBOT_POLY], np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [poly], 255)
    return m


def lab_hsv_masks(frame, fm):
    """Wash-resistant red/blue masks (mirror of the new bumper_alliance metric)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    a = lab[..., 1].astype(np.int16) - 128
    b = lab[..., 2].astype(np.int16) - 128
    colored = (s > 60) & (v > 45)
    red = colored & (((h <= 12) | (h >= 168)) | ((a >= 20) & (b > -15)))
    blue = colored & (((h >= 95) & (h <= 135)) | (b <= -18))
    red = (red.astype(np.uint8) * 255) & fm
    blue = (blue.astype(np.uint8) * 255) & fm
    return red, blue


def blobs(mask, frame_hw, min_area_frac=0.0008, max_area_frac=0.15):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    num, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
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
    frames = sys.argv[1:] or [
        str(ROOT / f"matrix_org/robot_diag/f0{n:02d}.png") for n in (5, 10, 20, 30, 40, 50)
    ]
    prop = ClassicalRobotProposer()
    outdir = ROOT / "data/scout_out/bumperbar"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"{'frame':10s} {'HSV(r/b)':10s} {'LAB(r/b)':10s} top3-areas-red  top3-areas-blue")
    for fp in frames:
        frame = cv2.imread(fp)
        if frame is None:
            continue
        H, W = frame.shape[:2]
        fm = field_mask(H, W)
        # stock proposer (HSV)
        props = prop.propose(frame, field_mask=fm)
        hsv_r = sum(1 for p in props if p["alliance"] == "red")
        hsv_b = sum(1 for p in props if p["alliance"] == "blue")
        # LAB
        rm, bm = lab_hsv_masks(frame, fm)
        rb = blobs(rm, (H, W))
        bb = blobs(bm, (H, W))
        vis = frame.copy()
        cv2.polylines(vis, [np.array([(int(x * W), int(y * H)) for x, y in ROBOT_POLY], np.int32)],
                      True, (0, 255, 0), 1)
        for d in rb[:4]:
            x1, y1, x2, y2 = d["xyxy"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(vis, d["anchor"], 6, (0, 0, 255), -1)
        for d in bb[:4]:
            x1, y1, x2, y2 = d["xyxy"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(vis, d["anchor"], 6, (255, 0, 0), -1)
        cv2.putText(vis, f"LAB red={len(rb)} blue={len(bb)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imwrite(str(outdir / f"bb_{Path(fp).stem}.png"), vis)
        ra = [d["area"] for d in rb[:3]]
        ba = [d["area"] for d in bb[:3]]
        print(f"{Path(fp).stem:10s} {f'{hsv_r}/{hsv_b}':10s} "
              f"{f'{len(rb)}/{len(bb)}':10s} {str(ra):16s} {ba}")


if __name__ == "__main__":
    main()
