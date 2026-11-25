"""tests/test_flight.py — pure-logic unit tests for the classical FlightDetector.

NO model, NO GPU. cv2 + numpy only (both in the CI image). All frames are synthetic
BGR arrays; MOG2 is primed with a static background, then a moving blob is introduced.

Focus of this suite (the R4 shots-lead change):
  A. Fuel-COLOR gate — a moving VIVID-YELLOW disc is kept; a moving NON-yellow disc
     (a "person" / banner of equal size+shape) is rejected.
  B. Circularity + max_candidates gates still work.
  C. arc_region restricts candidates to the band above the ground-field.
  D. API contract: signature detect(frame, frame_idx) -> list[Detection]; cv2 guarded.
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from frcscout.detect.flight import FlightDetector  # noqa: E402
from frcscout.io.types import Detection            # noqa: E402


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------
class _NoField:
    """field stub: no homography -> field_mask returns None (broadcast path)."""
    def field_mask(self, hw):
        return None


def _blank(h=200, w=200, bgr=(40, 40, 40)) -> np.ndarray:
    """A static, low-contrast background frame."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = bgr
    return img


# BGR for a VIVID yellow fuel ball (HSV ~ H30 high S/V). In BGR that is (0, 255, 255).
_FUEL_BGR = (0, 255, 255)
# A non-fuel moving object of identical size/shape (e.g. a person/banner) — magenta.
_NONFUEL_BGR = (255, 0, 255)


def _disc(img, cx, cy, r, bgr) -> np.ndarray:
    out = img.copy()
    cv2.circle(out, (int(cx), int(cy)), int(r), bgr, -1)
    return out


def _prime_and_detect(det, bg, moving_frames):
    """Prime MOG2 on the static bg, then feed moving frames; return last det list."""
    for _ in range(30):
        det.detect(bg, 0)
    out = []
    for k, fr in enumerate(moving_frames, start=1):
        out = det.detect(fr, k)
    return out


def _cfg(**over):
    base = {
        "min_area": 6,
        "max_area": 4000,
        "min_circularity": 0.0,
        "max_candidates": 0,
        "fuel_color_gate": True,
    }
    base.update(over)
    return base


# ===========================================================================
# A. Fuel-color gate
# ===========================================================================
class TestFuelColorGate:
    def test_moving_yellow_disc_detected(self):
        det = FlightDetector(_cfg(), _NoField())
        bg = _blank()
        # Disc moves across frames -> MOG2 foreground.
        frames = [_disc(bg, 60 + 8 * k, 100, 9, _FUEL_BGR) for k in range(3)]
        dets = _prime_and_detect(det, bg, frames)
        assert len(dets) >= 1, "a moving vivid-yellow fuel disc must be detected"
        assert all(isinstance(d, Detection) and d.cls == "ball" for d in dets)

    def test_moving_nonyellow_disc_rejected(self):
        """A moving magenta disc (same size/shape) is NOT fuel-colored -> dropped."""
        det = FlightDetector(_cfg(), _NoField())
        bg = _blank()
        frames = [_disc(bg, 60 + 8 * k, 100, 9, _NONFUEL_BGR) for k in range(3)]
        dets = _prime_and_detect(det, bg, frames)
        assert dets == [], "a moving NON-fuel-colored blob must be gated out"

    def test_gate_off_keeps_nonyellow(self):
        """With the color gate disabled, the magenta blob survives (proves the gate,
        not some other filter, is what removed it)."""
        det = FlightDetector(_cfg(fuel_color_gate=False), _NoField())
        bg = _blank()
        frames = [_disc(bg, 60 + 8 * k, 100, 9, _NONFUEL_BGR) for k in range(3)]
        dets = _prime_and_detect(det, bg, frames)
        assert len(dets) >= 1

    def test_static_yellow_not_detected(self):
        """Fuel-colored but STATIONARY (e.g. the resting pile) -> no MOG2 motion ->
        not a flight candidate."""
        det = FlightDetector(_cfg(), _NoField())
        bg = _disc(_blank(), 100, 100, 12, _FUEL_BGR)  # pile present in background
        frames = [bg, bg, bg]                          # never moves
        dets = _prime_and_detect(det, bg, frames)
        assert dets == []

    def test_greyscale_frame_skips_color_gate(self):
        """A single-channel (synthetic) frame has no color -> _fuel_mask returns None ->
        the detector must not crash and should fall back to motion-only."""
        det = FlightDetector(_cfg(), _NoField())
        bg = np.zeros((200, 200), dtype=np.uint8) + 40
        frames = []
        for k in range(3):
            f = bg.copy()
            cv2.circle(f, (60 + 8 * k, 100), 9, 255, -1)
            frames.append(f)
        dets = _prime_and_detect(det, bg, frames)
        assert len(dets) >= 1  # motion-only path still finds the moving blob


# ===========================================================================
# B. Circularity + max_candidates
# ===========================================================================
class TestShapeAndCap:
    def test_circularity_rejects_streak(self):
        """A long thin yellow streak (motion-blur / banner edge) fails circularity."""
        det = FlightDetector(_cfg(min_circularity=0.6), _NoField())
        bg = _blank()
        frames = []
        for k in range(3):
            f = bg.copy()
            # thin rectangle (low circularity), shifted each frame
            cv2.rectangle(f, (40 + 6 * k, 98), (140 + 6 * k, 104), _FUEL_BGR, -1)
            frames.append(f)
        dets = _prime_and_detect(det, bg, frames)
        assert dets == [], "an elongated streak must fail the circularity gate"

    def test_circularity_keeps_disc(self):
        det = FlightDetector(_cfg(min_circularity=0.6), _NoField())
        bg = _blank()
        frames = [_disc(bg, 60 + 8 * k, 100, 9, _FUEL_BGR) for k in range(3)]
        dets = _prime_and_detect(det, bg, frames)
        assert len(dets) >= 1

    def test_max_candidates_caps_output(self):
        det = FlightDetector(_cfg(max_candidates=2), _NoField())
        bg = _blank()
        # five separated moving yellow discs
        frames = []
        for k in range(3):
            f = bg.copy()
            for j, cx in enumerate((30, 70, 110, 150, 185)):
                cv2.circle(f, (cx + 3 * k, 60 + 20 * (j % 2)), 8, _FUEL_BGR, -1)
            frames.append(f)
        dets = _prime_and_detect(det, bg, frames)
        assert len(dets) <= 2


# ===========================================================================
# B2. min_radius_px_blob gate (sub-ball motion speckle floor)
# ===========================================================================
class TestRadiusFloor:
    def test_tiny_blob_rejected_by_radius_floor(self):
        """A moving 2px-radius yellow speckle passes area+circularity but is below the
        radius floor -> dropped. A real ~9px ball is kept."""
        det = FlightDetector(_cfg(min_area=2, min_radius_px_blob=4.0), _NoField())
        bg = _blank()
        tiny = [_disc(bg, 60 + 6 * k, 100, 2, _FUEL_BGR) for k in range(3)]
        assert _prime_and_detect(det, bg, tiny) == [], "tiny speckle must fail the radius floor"

    def test_real_ball_passes_radius_floor(self):
        det = FlightDetector(_cfg(min_area=2, min_radius_px_blob=4.0), _NoField())
        bg = _blank()
        ball = [_disc(bg, 60 + 8 * k, 100, 9, _FUEL_BGR) for k in range(3)]
        assert len(_prime_and_detect(det, bg, ball)) >= 1

    def test_radius_floor_off_by_default(self):
        det = FlightDetector(_cfg(), _NoField())
        assert det.min_radius_px == 0.0


# ===========================================================================
# B3. Tuned fuel band — defaults catch a slightly dim/low-hue airborne ball
# ===========================================================================
class TestTunedBand:
    def test_default_band_catches_dim_lowhue_ball(self):
        """A moving ball at H~16, V~100 (a fast/shadowed shot, below the OLD 18/110 floors)
        is caught with the tuned module defaults (h_lo=16, val_min=95). Color gate from cfg
        defaults (not overridden here)."""
        det = FlightDetector({"min_area": 6}, _NoField())  # cfg without color overrides
        assert det.fuel_h_lo == 16 and det.fuel_val_min == 95
        bg = _blank()
        # HSV (16, 200, 105) -> BGR via cv2 so the mask sees exactly that hue/val.
        hsv = np.zeros((1, 1, 3), np.uint8); hsv[0, 0] = (16, 200, 105)
        bgr = tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
        frames = [_disc(bg, 60 + 8 * k, 100, 9, bgr) for k in range(3)]
        assert len(_prime_and_detect(det, bg, frames)) >= 1

    def test_pale_person_still_rejected(self):
        """A moving PALE blob (low saturation ~70, like crowd clothing) is still gated out by
        the high sat floor — the FP killer must survive the tuning."""
        det = FlightDetector({"min_area": 6}, _NoField())
        bg = _blank()
        hsv = np.zeros((1, 1, 3), np.uint8); hsv[0, 0] = (25, 70, 180)  # pale, S=70
        bgr = tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
        frames = [_disc(bg, 60 + 8 * k, 100, 9, bgr) for k in range(3)]
        assert _prime_and_detect(det, bg, frames) == [], "pale (low-sat) motion must be rejected"


# ===========================================================================
# C. arc_region restriction
# ===========================================================================
class TestArcRegion:
    def test_ground_ball_below_arc_band_excluded(self):
        """A moving yellow ball low in the frame (rolling on carpet) is outside the arc
        band [y 0.0..0.4] and must be excluded; a high ball is kept."""
        arc = [0.0, 0.0, 1.0, 0.4]  # top 40% only
        det = FlightDetector(_cfg(arc_region=arc), _NoField())
        bg = _blank(h=200, w=200)
        # ball low in frame (y=160 -> 0.8 of height) -> below arc band
        frames_low = [_disc(bg, 60 + 8 * k, 160, 9, _FUEL_BGR) for k in range(3)]
        dets_low = _prime_and_detect(det, bg, frames_low)
        assert dets_low == [], "a ground-level ball is outside the arc band"

    def test_high_ball_in_arc_band_kept(self):
        arc = [0.0, 0.0, 1.0, 0.4]
        det = FlightDetector(_cfg(arc_region=arc), _NoField())
        bg = _blank(h=200, w=200)
        frames_high = [_disc(bg, 60 + 8 * k, 50, 9, _FUEL_BGR) for k in range(3)]  # y=0.25
        dets_high = _prime_and_detect(det, bg, frames_high)
        assert len(dets_high) >= 1


# ===========================================================================
# D. API contract
# ===========================================================================
class TestApiContract:
    def test_signature_and_return_type(self):
        det = FlightDetector(_cfg(), _NoField())
        bg = _blank()
        out = det.detect(bg, 0)
        assert isinstance(out, list)

    def test_none_cfg_defaults(self):
        """cfg=None must not crash (color gate defaults ON)."""
        det = FlightDetector(None, _NoField())
        assert det.fuel_color_gate is True
        assert det.min_area == 4 and det.max_area == 600

    def test_empty_frame_no_motion(self):
        det = FlightDetector(_cfg(), _NoField())
        bg = _blank()
        assert det.detect(bg, 0) == [] or isinstance(det.detect(bg, 1), list)
