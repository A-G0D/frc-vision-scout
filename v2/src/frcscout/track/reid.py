"""Re-Identification / persistence manager for the robot tracker.

The ByteTrack Kalman tracker keeps an ID alive only while a track is hit (or
coasts) inside ``max_age``. In a single broadcast camera the six robots in
FRC REBUILT (2026) routinely disappear for *longer* than that: they drive
behind one another, behind the two dark HUB boxes, or off-screen on a camera
cut / pan. When such a robot re-emerges a pure IoU/Kalman tracker mints a brand
new ID — an ID switch that ruins per-robot scouting metrics.

``ReIDManager`` adds three layers of persistence ON TOP of the Kalman tracker,
ordered from cheap-and-soft to expensive-and-permanent:

  (a) Appearance signature — per track, a dependency-light descriptor:
        * alliance        : 'red' | 'blue'  (HARD gate — bumper color is
                            unambiguous at this zoom; a blue track never
                            re-acquires a red detection).
        * bumper hue hist : a normalized coarse Hue histogram of the vivid
                            pixels in the bumper band of the box (reuses the
                            same HSV red/blue idea as RobotNet.bumper_alliance,
                            but keeps the full hue profile to separate two
                            robots of the *same* alliance).
        * size            : last good (w, h); apparent robot size is ~constant
                            on the flat carpet.
      No torch, no learned embedding — just OpenCV + numpy, and even cv2 is
      lazily/optionally used so tests run with frame=None.

  (b) Re-bind — when a birth (an unmatched high-conf detection) would normally
      mint a new ID, first consult the pool of RECENTLY-LOST tracks. If a lost
      track of the same alliance is appearance-similar AND lands within a
      plausible field/pixel radius of where it should have re-emerged (radius
      grows with the gap, in SECONDS via the real fps), REASSIGN that lost
      track's ID instead of allocating a new one.

  (c) Bumper-number anchor — an OPTIONAL hook ``read_bumper_number(crop)`` (a
      lazy, guarded easyocr call) that, when it returns a team number, binds
      ``track_id <-> team_number`` PERMANENTLY in a registry. Once a track is
      anchored, its identity survives an arbitrarily long gap: a later re-bind
      can be confirmed by re-reading the same number, and downstream can map
      any anchored track straight to a real team#.

Inferential / honest caveats
----------------------------
* The appearance descriptor is deliberately weak (a 12-bin hue histogram).
  It reliably separates the two alliances and helps within an alliance, but
  two same-alliance robots with near-identical bumpers are separated mostly by
  the position gate. The bumper-number anchor (c) is the only *certain* cue;
  everything else is probabilistic and should be validated on real footage.
* The OCR hook is best-effort: FRC bumper numbers are small and motion-blurred
  in a wide broadcast shot, so expect anchors only on a minority of frames.
  That is fine — one good read permanently anchors the track.

Stubbing for tests
------------------
* ``embed_crop`` and ``read_bumper_number`` are instance attributes you can
  replace, and ``ReIDManager`` accepts ``embed_fn`` / ``ocr_fn`` ctor kwargs.
  With the defaults and ``frame=None`` the manager runs pure-python (no cv2,
  no easyocr), so the unit tests need neither.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from frcscout.io.config import seconds_to_frames
from frcscout.io.types import Alliance

# Histogram geometry (kept in sync with the v1 prototype's defaults).
_HUE_BINS = 12
_BUMPER_LOWER_FRAC = 0.5     # bumpers sit on the lower half of the box
_SAT_FLOOR = 90
_VAL_FLOOR = 60


# ─────────────────────────────── descriptors ─────────────────────────────────
@dataclass
class Appearance:
    """A cheap appearance + geometry signature for one robot track."""
    alliance: Optional[Alliance] = None
    hue_hist: Optional[np.ndarray] = None       # length _HUE_BINS, sums to 1 (or None)
    size: tuple[float, float] = (0.0, 0.0)      # (w, h) px

    def has_hist(self) -> bool:
        return self.hue_hist is not None and float(np.sum(self.hue_hist)) > 0.0


@dataclass
class LostTrack:
    """A culled track parked for possible re-binding."""
    tid: int
    appearance: Appearance
    last_centroid_px: tuple[float, float]
    vel_px_per_frame: tuple[float, float]
    last_field_xy: Optional[tuple[float, float]]
    last_frame: int
    team_number: Optional[int] = None           # set if it was anchored


# ───────────────────────────── default embedders ─────────────────────────────
def _default_embed_crop(crop: "np.ndarray") -> Optional[np.ndarray]:
    """Normalized coarse Hue histogram of the vivid bumper pixels of a crop.

    ``crop`` is the BGR robot box image (already sliced). Returns a float32
    vector of length ``_HUE_BINS`` summing to 1, or None when cv2 is missing /
    the crop has no vivid pixels. Guarded import keeps the dependency optional.
    """
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    try:
        import cv2
    except Exception:
        return None
    h = crop.shape[0]
    band = crop[int(_BUMPER_LOWER_FRAC * h):, :]
    if band.size == 0:
        return None
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    vivid = (s >= _SAT_FLOOR) & (v >= _VAL_FLOOR)
    if int(vivid.sum()) < 3:
        return None
    hue = hsv[..., 0][vivid].astype(np.float32)     # 0..179
    hist, _ = np.histogram(hue, bins=_HUE_BINS, range=(0.0, 180.0))
    hist = hist.astype(np.float32)
    tot = float(hist.sum())
    return hist / tot if tot > 0 else None


def _default_read_bumper_number(crop: "np.ndarray") -> Optional[int]:
    """Best-effort bumper team-number OCR (lazy, guarded easyocr).

    Returns an int team number or None. Import + reader construction are lazy
    and cached on the function object so the heavy easyocr model loads at most
    once and only if this hook is actually exercised. Any failure -> None.
    """
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    try:
        import re
        reader = getattr(_default_read_bumper_number, "_reader", None)
        if reader is None:
            import easyocr  # guarded: optional heavy dep
            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            _default_read_bumper_number._reader = reader  # type: ignore[attr-defined]
        results = reader.readtext(crop, allowlist="0123456789", detail=0)
        for token in results:
            digits = re.sub(r"\D", "", str(token))
            if digits:
                n = int(digits)
                if 1 <= n <= 99999:           # plausible FRC team number
                    return n
    except Exception:
        return None
    return None


# ────────────────────────────── similarity ───────────────────────────────────
def hist_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    """Histogram-intersection similarity in [0,1]; None if either is empty."""
    if a is None or b is None:
        return None
    sa, sb = float(np.sum(a)), float(np.sum(b))
    if sa <= 0.0 or sb <= 0.0:
        return None
    return float(np.minimum(a, b).sum())


def appearance_similarity(a: Appearance, b: Appearance) -> float:
    """Blended appearance similarity in [0,1] (1 = identical).

    Alliance mismatch -> 0.0 (hard separation). With no usable histogram the
    score degrades to a neutral 0.5 so motion/position can still drive matching.
    """
    if a.alliance is not None and b.alliance is not None and a.alliance != b.alliance:
        return 0.0
    sim = hist_similarity(a.hue_hist, b.hue_hist)
    app = sim if sim is not None else 0.5
    # size agreement (mild)
    wa, ha = a.size
    wb, hb = b.size
    if max(wa, wb) > 0 and max(ha, hb) > 0:
        sw = abs(wa - wb) / max(1.0, wa, wb)
        sh = abs(ha - hb) / max(1.0, ha, hb)
        size_sim = max(0.0, 1.0 - 0.5 * (sw + sh))
    else:
        size_sim = 0.5
    return float(0.75 * app + 0.25 * size_sim)


# ─────────────────────────────── manager ─────────────────────────────────────
class ReIDManager:
    """Persistent-identity manager consulted by RobotTracker on births/culls.

    Parameters (cfg dict, all timing in SECONDS — converted with the real fps):
      reid.lost_ttl_s        how long a culled track stays re-bindable (def 4.0)
      reid.base_gate_px      re-emergence radius at a 1-frame gap (def 40.0)
      reid.per_frame_gate_px radius growth per missed frame       (def 9.0)
      reid.max_gate_px       cap on the pixel gate                (def 400.0)
      reid.field_gate_m      re-emergence radius in field meters  (def 3.0)
      reid.per_s_field_gate_m field-radius growth per second lost (def 2.0)
      reid.min_appearance    min appearance similarity to accept  (def 0.30)
      reid.anchor_min_app    relaxed sim when team# already matches(def 0.0)
    """

    def __init__(self, cfg: Optional[dict] = None, fps: float = 60.0,
                 embed_fn: Optional[Callable] = None,
                 ocr_fn: Optional[Callable] = None):
        self.cfg = (cfg or {})
        self.fps = float(fps)

        self.lost_ttl_frames = seconds_to_frames(
            float(self.cfg.get("lost_ttl_s", 4.0)), self.fps)
        self.base_gate_px = float(self.cfg.get("base_gate_px", 40.0))
        self.per_frame_gate_px = float(self.cfg.get("per_frame_gate_px", 9.0))
        self.max_gate_px = float(self.cfg.get("max_gate_px", 400.0))
        self.field_gate_m = float(self.cfg.get("field_gate_m", 3.0))
        self.per_s_field_gate_m = float(self.cfg.get("per_s_field_gate_m", 2.0))
        self.min_appearance = float(self.cfg.get("min_appearance", 0.30))
        self.anchor_min_app = float(self.cfg.get("anchor_min_app", 0.0))

        # swappable hooks (stub in tests)
        self.embed_crop = embed_fn or _default_embed_crop
        self.read_bumper_number = ocr_fn or _default_read_bumper_number

        # state
        self._lost: list[LostTrack] = []
        # PERMANENT id<->team registry (survives any gap, survives re-binds)
        self.team_of_track: dict[int, int] = {}
        self.track_of_team: dict[int, int] = {}

    # ───────────────────────── appearance building ───────────────────────────
    def build_appearance(self, alliance: Optional[Alliance],
                         xyxy: tuple[float, float, float, float],
                         frame=None) -> Appearance:
        """Construct an Appearance from a box + (optional) frame.

        With ``frame=None`` (tests / no decode) only alliance + size are filled;
        matching then leans on position. The hue histogram comes from the
        swappable ``embed_crop`` hook.
        """
        x1, y1, x2, y2 = xyxy
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        hist = None
        if frame is not None:
            crop = self._crop(frame, xyxy)
            if crop is not None:
                hist = self.embed_crop(crop)
        return Appearance(alliance=alliance, hue_hist=hist, size=(w, h))

    @staticmethod
    def _crop(frame, xyxy):
        try:
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            crop = frame[y1:y2, x1:x2]
            return crop if getattr(crop, "size", 0) else None
        except Exception:
            return None

    # ────────────────────────── anchor (team#) ───────────────────────────────
    def try_anchor(self, tid: int,
                   xyxy: tuple[float, float, float, float],
                   frame=None, crop=None) -> Optional[int]:
        """Attempt to bind ``tid`` to a team number via bumper OCR.

        Returns the team number if (already) anchored or freshly read, else
        None. Idempotent: a track already in the registry returns its number
        without re-reading. A team number already owned by another track is NOT
        stolen (the first anchor wins) unless it's the same track.
        """
        if tid in self.team_of_track:
            return self.team_of_track[tid]
        if crop is None and frame is not None:
            crop = self._crop(frame, xyxy)
        if crop is None:
            return None
        num = self.read_bumper_number(crop)
        if num is None:
            return None
        return self.anchor(tid, num)

    def anchor(self, tid: int, team_number: int) -> Optional[int]:
        """Directly bind tid<->team_number (also used by tests). First-writer
        wins for a given team number across different track ids."""
        existing = self.track_of_team.get(team_number)
        if existing is not None and existing != tid:
            # team already owned by a live/other track — don't double-bind it.
            return self.team_of_track.get(tid)
        self.team_of_track[tid] = team_number
        self.track_of_team[team_number] = tid
        return team_number

    def team_for(self, tid: int) -> Optional[int]:
        return self.team_of_track.get(tid)

    # ─────────────────────────── lost-track pool ─────────────────────────────
    def note_lost(self, tid: int, appearance: Appearance,
                  last_centroid_px: tuple[float, float],
                  vel_px_per_frame: tuple[float, float],
                  last_field_xy: Optional[tuple[float, float]],
                  last_frame: int) -> None:
        """Park a just-culled track so a future birth can re-bind to it."""
        # replace any stale entry for the same tid
        self._lost = [lt for lt in self._lost if lt.tid != tid]
        self._lost.append(LostTrack(
            tid=tid, appearance=appearance,
            last_centroid_px=last_centroid_px,
            vel_px_per_frame=vel_px_per_frame,
            last_field_xy=last_field_xy,
            last_frame=last_frame,
            team_number=self.team_of_track.get(tid),
        ))

    def expire(self, frame_idx: int) -> None:
        """Drop lost tracks past their TTL — UNLESS anchored to a team number,
        which we keep indefinitely so identity survives any gap."""
        self._lost = [
            lt for lt in self._lost
            if lt.team_number is not None
            or (frame_idx - lt.last_frame) <= self.lost_ttl_frames
        ]

    def lost_ids(self) -> list[int]:
        return [lt.tid for lt in self._lost]

    # ───────────────────────────── re-bind ───────────────────────────────────
    def _predicted_px(self, lt: LostTrack, frame_idx: int) -> tuple[float, float]:
        gap = max(1, frame_idx - lt.last_frame)
        cx, cy = lt.last_centroid_px
        vx, vy = lt.vel_px_per_frame
        return (cx + vx * gap, cy + vy * gap)

    def rebind(self, alliance: Optional[Alliance],
               xyxy: tuple[float, float, float, float],
               centroid_px: tuple[float, float],
               frame_idx: int,
               frame=None,
               field_xy: Optional[tuple[float, float]] = None,
               appearance: Optional[Appearance] = None) -> Optional[int]:
        """Decide whether a new detection should REUSE a lost track's ID.

        Returns the reused track id (and consumes that lost entry) or None if
        the detection should mint a fresh id. Scoring:
          * HARD alliance gate.
          * position gate: field-meters when both sides have field coords,
            else predicted-pixel distance; the radius grows with the gap (in
            seconds via the real fps).
          * appearance similarity must clear ``min_appearance`` (relaxed when an
            anchored team number is re-confirmed by OCR).
        Among gated candidates the highest appearance-similarity wins; position
        breaks ties.
        """
        if not self._lost:
            return None
        if appearance is None:
            appearance = self.build_appearance(alliance, xyxy, frame)

        gap = max(1, frame_idx - 0)  # placeholder; per-candidate below
        best: Optional[LostTrack] = None
        best_key = (-1.0, math.inf)  # (appearance_sim, -dist) maximize app then min dist

        for lt in self._lost:
            if (alliance is not None and lt.appearance.alliance is not None
                    and alliance != lt.appearance.alliance):
                continue
            cand_gap = max(1, frame_idx - lt.last_frame)

            # position gate
            if field_xy is not None and lt.last_field_xy is not None:
                d = math.hypot(field_xy[0] - lt.last_field_xy[0],
                               field_xy[1] - lt.last_field_xy[1])
                gate = self.field_gate_m + self.per_s_field_gate_m * (cand_gap / self.fps)
            else:
                px, py = self._predicted_px(lt, frame_idx)
                d = math.hypot(centroid_px[0] - px, centroid_px[1] - py)
                gate = min(self.max_gate_px,
                           self.base_gate_px + self.per_frame_gate_px * cand_gap)
            if d > gate:
                continue

            sim = appearance_similarity(appearance, lt.appearance)

            # appearance acceptance — relax if an anchored team can be reconfirmed
            min_app = self.min_appearance
            if lt.team_number is not None:
                # re-read the number; a confirmed match relaxes the appearance bar.
                num = None
                if frame is not None:
                    crop = self._crop(frame, xyxy)
                    if crop is not None:
                        num = self.read_bumper_number(crop)
                if num == lt.team_number:
                    min_app = self.anchor_min_app
            if sim < min_app:
                continue

            key = (sim, -d)
            if key > best_key:
                best_key = key
                best = lt

        if best is None:
            return None
        # consume the lost entry and return its id (anchor stays in the registry)
        self._lost = [lt for lt in self._lost if lt.tid != best.tid]
        return best.tid
