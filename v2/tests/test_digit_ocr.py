import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bench_overlay_ocr as b  # noqa: E402
from frcscout.detect import digit_ocr  # noqa: E402
from frcscout.detect.overlay_ocr import _LIVE_BAR_ROIS as R, _crop  # noqa: E402


def test_clean_frames_read_back():
    rng = random.Random(1)
    ok_s = ok_t = n = 30
    ok_s = ok_t = 0
    for _ in range(n):
        f, score, timer, _font = b.make_frame(rng, 0)
        ok_s += digit_ocr.read_int(_crop(f, R["blue_score"])) == score
        ok_t += digit_ocr.read_timer(_crop(f, R["timer"])) == timer
    assert ok_s / n >= 0.95 and ok_t / n >= 0.9


def test_empty_crop_is_none():
    assert digit_ocr.read_int(np.zeros((0, 0, 3), np.uint8)) is None
    assert digit_ocr.read_timer(np.zeros((20, 40, 3), np.uint8)) is None


def test_overlay_reader_template_mode():
    from frcscout.detect.overlay_ocr import OverlayReader
    rng = random.Random(2)
    reader = OverlayReader({"ocr": "template"})
    ok = 0
    for i in range(20):
        f, score, timer, _font = b.make_frame(rng, 0)
        st = reader.read(f, i)
        ok += (st.blue_score, st.timer_s) == (score, timer)
    assert ok / 20 >= 0.9
