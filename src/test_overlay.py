"""
test_overlay.py — validate the Claude overlay reader against real downloaded frames.

GROUND TRUTH below was read by eye from the actual 360p crops in frames/overlay_test360/
(extracted from the downloaded NE Waterbury stream ngvsHrmoW8U). When ANTHROPIC_API_KEY is
set, this runs ocr_overlay.read_bar_image on each real crop and checks Claude's structured
output against that truth, field by field.

Run:  python src/test_overlay.py
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import ocr_overlay

CROPS = Path(__file__).resolve().parent.parent / "frames" / "overlay_test360"

# Ground truth read by eye from the 360p overlay bars (see frames/overlay_test360/).
GROUND_TRUTH = {
    "qual_013000_bar.png": {
        "match_label_contains": "56 of 64",
        "blue_teams": [8889, 6933, 6328],
        "red_teams": [7907, 7127, 10266],
        "is_playoff": False,
    },
    "play_053000_bar.png": {
        "match_label_contains": "Match 3",
        "is_playoff": True,         # bar shows ALLIANCE 7 / ALLIANCE 2, not team numbers
        "blue_score": 30,
        "red_score": 163,
    },
}


def _check(name: str, truth: dict, got: dict) -> list[str]:
    fails = []
    for key, want in truth.items():
        if key == "match_label_contains":
            label = (got.get("match_label") or "")
            if want.lower() not in label.lower():
                fails.append(f"  match_label: expected to contain '{want}', got '{label}'")
        else:
            if got.get(key) != want:
                fails.append(f"  {key}: expected {want}, got {got.get(key)}")
    return fails


def main() -> int:
    if not CROPS.exists() or not any(CROPS.glob("*.png")):
        print(f"No test crops in {CROPS}. Generate them from the 360p video first.")
        return 1

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — cannot run the live Claude call.\n")
        print("The wiring is built and the test frames are ready. To run the live test:")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print("  python src/test_overlay.py\n")
        print("Expected results (ground truth read by eye from the real 360p frames):")
        for name, truth in GROUND_TRUTH.items():
            print(f"  {name}: {truth}")
        return 2

    total_fails = 0
    for name, truth in GROUND_TRUTH.items():
        path = CROPS / name
        if not path.exists():
            print(f"SKIP {name} (missing)"); continue
        got = asdict(ocr_overlay.read_bar_image(path))
        fails = _check(name, truth, got)
        if fails:
            total_fails += len(fails)
            print(f"FAIL {name}")
            print(f"     got: {got}")
            print("\n".join(fails))
        else:
            print(f"PASS {name}  ->  label={got['match_label']!r} "
                  f"blue={got['blue_teams']} red={got['red_teams']} "
                  f"score={got['blue_score']}-{got['red_score']} playoff={got['is_playoff']}")
    print(f"\n{'ALL PASS' if total_fails == 0 else f'{total_fails} field mismatch(es)'}")
    return 0 if total_fails == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
