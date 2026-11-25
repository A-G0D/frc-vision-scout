"""Before/after harness for RobotNet.bumper_alliance.

Runs the robot detector on real qual59 frames, then for every robot box prints
OLD vs NEW alliance + the raw red/blue/n numbers, and renders boxes coloured by
the NEW classifier so the result can be eyeballed. No ground-truth labels exist,
so the judgement is visual + abstention-rate — NOT a fabricated accuracy number.
"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from frcscout.detect.robot_net import RobotNet  # noqa: E402


def old_alliance(frame, xyxy):
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    crop = frame[(y1 + y2) // 2:y2, x1:x2]
    if crop.size == 0:
        return None, 0, 0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat = (s > 90) & (v > 60)
    red = int((sat & ((h <= 10) | (h >= 170))).sum())
    blue = int((sat & (h >= 100) & (h <= 130)).sum())
    if max(red, blue) < 0.02 * crop.shape[0] * crop.shape[1]:
        return None, red, blue
    return ("red" if red >= blue else "blue"), red, blue


def main():
    from ultralytics import YOLO
    model = YOLO(str(ROOT / "weights/ball_openvino_model"), task="detect")
    frames = sys.argv[1:] or [
        str(ROOT / f"matrix_org/robot_diag/f0{n:02d}.png") for n in (10, 20, 30, 40)
    ]
    outdir = ROOT / "data/scout_out/alliance_cmp"
    outdir.mkdir(parents=True, exist_ok=True)
    col = {"red": (0, 0, 255), "blue": (255, 0, 0), None: (0, 255, 255)}
    tot = {"agree": 0, "flip": 0, "new_abstain": 0, "old_abstain": 0, "both_abstain": 0, "boxes": 0}

    for fp in frames:
        frame = cv2.imread(fp)
        if frame is None:
            print(f"!! cannot read {fp}")
            continue
        r = model.predict(frame, imgsz=960, conf=0.12, device="cpu", verbose=False)[0]
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        vis = frame.copy()
        print(f"\n=== {Path(fp).name} ===")
        for b, c in zip(xyxy, cls):
            if c != 1:
                continue
            tot["boxes"] += 1
            new = RobotNet.bumper_alliance(frame, b)
            old, ored, oblue = old_alliance(frame, b)
            if new is None and old is None:
                tot["both_abstain"] += 1
            elif new is None:
                tot["new_abstain"] += 1
            elif old is None:
                tot["old_abstain"] += 1
            elif new == old:
                tot["agree"] += 1
            else:
                tot["flip"] += 1
            x1, y1, x2, y2 = (int(v) for v in b)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col[new], 3)
            cv2.putText(vis, f"{(new or '?')[:1].upper()}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col[new], 2)
            print(f"  box=({x1:4d},{y1:4d},{x2:4d},{y2:4d}) "
                  f"OLD={str(old):4s}(r{ored} b{oblue})  NEW={str(new)}")
        outp = outdir / f"cmp_{Path(fp).stem}.png"
        cv2.imwrite(str(outp), vis)
        print(f"  -> {outp}")
    print(f"\n=== TOTALS over {tot['boxes']} boxes ===")
    for k, v in tot.items():
        if k != "boxes":
            print(f"  {k:14s} {v}")


if __name__ == "__main__":
    main()
