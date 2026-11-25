"""
ocr_overlay.py — read the broadcast scoreboard overlay with Claude vision.

Replaces the tesseract/easyocr stub. Instead of cropping each field into a separate
sub-ROI and OCRing digits (brittle on stylized broadcast graphics), we crop the whole
overlay bar, upscale it, and hand it to Claude as one image — Claude parses the entire
bar (match label, both alliances' team numbers, the timer, the live fuel counts) in a
single call and returns schema-validated JSON.

Why vision instead of OCR: the overlay digits are small and stylized; classic OCR needs
clean high-contrast glyphs. A vision model reads them in context (it knows "Qualification
24 of 86" is a match label and "1991 4905 5687" are the blue alliance) and degrades
gracefully — returning null for a field it can't read rather than emitting garbage.

Resolution matters: at 144p the in-match bar digits are unreadable even to a human; use
240p+ (ideally 360p) source video for the in-match overlay. Full-screen graphics
(alliance selection) are legible lower.

Robustness: read several frames across a match segment and vote per field (team numbers
are static for the whole match), so a single bad frame can't corrupt the result.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from overlay_detect import crop_norm, OVERLAY_ROIS
import claude_client

# JSON Schema the model must fill. nullable fields let Claude abstain on unreadable parts.
OVERLAY_SCHEMA = {
    "type": "object",
    "properties": {
        "match_label": {"type": ["string", "null"],
                        "description": "e.g. 'Qualification 24 of 86', 'Playoff', 'Final 2'. Null if not visible."},
        "blue_teams": {"type": "array", "items": {"type": "integer"},
                       "description": "Blue alliance FRC team numbers, left side, in display order. Empty if unreadable."},
        "red_teams": {"type": "array", "items": {"type": "integer"},
                      "description": "Red alliance FRC team numbers, right side, in display order. Empty if unreadable."},
        "blue_score": {"type": ["integer", "null"], "description": "Blue score / fuel count (the left X in 'X / NNN'). Null if unreadable."},
        "red_score": {"type": ["integer", "null"], "description": "Red score / fuel count (the right X). Null if unreadable."},
        "timer": {"type": ["string", "null"], "description": "Center match timer, e.g. '0:20'. Null if not visible."},
        "is_playoff": {"type": "boolean", "description": "True if the bar shows alliance numbers (e.g. 'ALLIANCE 7') instead of three team numbers per side."},
    },
    "required": ["match_label", "blue_teams", "red_teams", "blue_score", "red_score", "timer", "is_playoff"],
    "additionalProperties": False,
}

INSTRUCTION = (
    "This is the top scoreboard overlay from a FIRST Robotics Competition (FRC) broadcast. "
    "Blue alliance is on the LEFT, red alliance on the RIGHT, with a match timer in the center "
    "and a 'X / NNN' fuel/score box on each far side. Read the visible values and return JSON. "
    "Qualification matches show three FRC team numbers (typically 1-5 digit) per alliance; playoff "
    "matches show an 'ALLIANCE N' label instead — in that case set is_playoff=true and leave the "
    "team arrays empty. Only report digits you can actually read; use null / empty arrays for "
    "anything blurry or absent. Do not guess."
)


@dataclass
class OverlayRead:
    match_label: str | None
    blue_teams: list[int]
    red_teams: list[int]
    blue_score: int | None
    red_score: int | None
    timer: str | None
    is_playoff: bool


def _bar_crop(frame: np.ndarray, upscale: int = 6) -> np.ndarray:
    """Crop the overlay bar (top of frame) and upscale for legibility."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = OVERLAY_ROIS["bar"]
    bar = frame[int(y0 * h):max(int(y1 * h), int(0.13 * h)), int(x0 * w):int(x1 * w)]
    return cv2.resize(bar, (bar.shape[1] * upscale, bar.shape[0] * upscale),
                      interpolation=cv2.INTER_LANCZOS4)


def read_bar_image(image_path: str | Path) -> OverlayRead:
    """Read a single pre-cropped overlay-bar image (already on disk) via Claude vision."""
    data = claude_client.read_image_json([image_path], INSTRUCTION, OVERLAY_SCHEMA)
    return OverlayRead(**data)


def read_segment(video_path: str | Path, t_start: float, t_end: float,
                 n_samples: int = 3, tmp_dir: str | Path = "frames/_ocr_tmp") -> OverlayRead:
    """
    Sample `n_samples` frames across a match segment, OCR each overlay bar with Claude,
    and vote per field. Returns the consensus OverlayRead.
    """
    tmp = Path(tmp_dir); tmp.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    times = np.linspace(t_start + 5, max(t_start + 6, t_end - 5), n_samples)

    reads: list[OverlayRead] = []
    for i, t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        crop_path = tmp / f"bar_{int(t)}.png"
        cv2.imwrite(str(crop_path), _bar_crop(frame))
        try:
            reads.append(read_bar_image(crop_path))
        except Exception as e:
            print(f"[ocr_overlay] frame @{t:.0f}s failed: {e}")
    cap.release()
    return _vote(reads)


def _vote(reads: list[OverlayRead]) -> OverlayRead:
    """Per-field mode across sampled reads; team arrays voted as whole tuples."""
    if not reads:
        return OverlayRead(None, [], [], None, None, None, False)

    def mode(vals):
        vals = [v for v in vals if v is not None]
        return Counter(vals).most_common(1)[0][0] if vals else None

    def mode_list(lists):
        keyed = [tuple(x) for x in lists if x]
        return list(Counter(keyed).most_common(1)[0][0]) if keyed else []

    return OverlayRead(
        match_label=mode([r.match_label for r in reads]),
        blue_teams=mode_list([r.blue_teams for r in reads]),
        red_teams=mode_list([r.red_teams for r in reads]),
        blue_score=mode([r.blue_score for r in reads]),
        red_score=mode([r.red_score for r in reads]),
        timer=mode([r.timer for r in reads]),
        is_playoff=bool(mode([r.is_playoff for r in reads])),
    )


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) >= 2:
        # read a single pre-made bar image
        print(json.dumps(asdict(read_bar_image(sys.argv[1])), indent=2))
    else:
        print("usage: python ocr_overlay.py <bar_image.png>")
