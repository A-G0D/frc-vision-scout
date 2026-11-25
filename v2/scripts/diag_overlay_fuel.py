"""Diagnostic: is the broadcast fuel counter capped at 100, and is the SCORE cumulative?

Reads the HUD across auto + teleop frames of qual59 and prints raw blue/red fuel and
score. Confirms (or refutes) the hypothesis that fuel is a 0-100 RP-progress display
(so it saturates and cannot yield match-cumulative teleop fuel), while the score is the
true uncapped cumulative signal. Measure, don't assume.
"""
import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from hud_read import HudReader  # noqa: E402

VIDEO = str(ROOT / "data/footage/qual59.mp4")
# frames spanning auto (~2400-3300) -> teleop (~3300-13000) -> late (~14000)
SAMPLE_FRAMES = [2600, 3000, 3300, 4000, 5000, 6500, 8000, 9500, 11000, 12500, 14000]


def main():
    hud = HudReader()
    cap = cv2.VideoCapture(VIDEO)
    print(f"{'frame':>6} {'blue_fuel':>9} {'red_fuel':>8} {'blue_score':>10} "
          f"{'red_score':>9} {'timer':>6} {'phase':>8}")
    for f in SAMPLE_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            print(f"{f:>6}  (read failed)")
            continue
        st = hud.read(frame, f)
        print(f"{f:>6} {str(st.blue_fuel):>9} {str(st.red_fuel):>8} "
              f"{str(st.blue_score):>10} {str(st.red_score):>9} "
              f"{str(getattr(st,'timer_s',None)):>6} {str(getattr(st,'phase',None)):>8}")
    cap.release()
    print("\nTBA truth (qm59): BLUE total=92 (auto 26/teleop 66), RED total=613 (auto 95/teleop 518)")
    print("Hypothesis: fuel saturates ~100 (RP display); score is the uncapped cumulative signal.")


if __name__ == "__main__":
    main()
