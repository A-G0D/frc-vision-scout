"""Temporal detection recovery + box smoothing for the track stream.

This refine layer sits *after* tracking (it consumes :class:`Track` objects keyed
by their stable ``tid``) and does two cheap, tracker-agnostic jobs that make the
60Hz output steadier for the downstream geometry / event layers:

1. **Gap fill (recovery).** When an object that was seen in the last few frames is
   *missing this frame* — a one-off detector miss, a brief occlusion, or a frame
   the detector simply skipped on the low cadence — we synthesize a placeholder
   Track by linearly extrapolating its last known box along its estimated
   velocity. The fill is emitted for up to ``max_gap`` frames; after that the id
   is considered genuinely gone and dropped. Synthesized tracks are flagged so
   consumers can tell them apart from real measurements (see "Marking fills").

2. **EMA smoothing.** Detector boxes jitter frame-to-frame even on a static
   object. A per-id exponential moving average on the four box coordinates damps
   that jitter, trading a small, bounded lag for a large drop in variance.

Why a separate layer (the trackers already Kalman-coast)
--------------------------------------------------------
The ByteTrack / Kalman trackers do coast through gaps, but (a) not every track
source is a Kalman tracker, (b) a tracker may *cull* a track before the consumer
would like it gone, and (c) this layer additionally **smooths** the boxes, which
the trackers do not. Keeping recovery + smoothing here makes both behaviours
reusable and unit-testable in isolation, with no model and no GPU.

Marking fills (we must not edit ``Track``'s schema)
---------------------------------------------------
:class:`Track` has no "synthetic" field and ``io/types.py`` is read-only, so a
filled frame is marked *structurally*, exactly the way a coasting Kalman track is:
its ``hits`` count does **not** increase on a filled frame (no new measurement),
while ``age`` keeps advancing. In addition, the recovery buffer exposes the set
of ids it filled on the latest step via :attr:`TemporalRecovery.filled_tids`, and
:meth:`TemporalRecovery.step` returns a richer :class:`RecoveryResult` for callers
that want the present/filled split explicitly. The plain :meth:`update` mirrors
the tracker API (``update(frame_idx, tracks) -> list[Track]``) for drop-in use.

Units & timing
--------------
All timing is configured in SECONDS and converted with the capture's real fps via
``seconds_to_frames`` — never a hardcoded frame count (the v1 bug). Velocity is
estimated internally in **pixels per frame** from the observed centroid history;
when we re-emit it on a Track's ``vel_mps`` field we scale by fps to px/s, the
same convention the trackers use (the geometry layer later converts px/s -> m/s).

Purity
------
Everything here is pure Python and parses/imports/tests with **no numpy, cv2,
scipy or easyocr present**. numpy is imported lazily and guarded purely as a
potential speed-up; the logic never requires it.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from frcscout.io.config import seconds_to_frames
from frcscout.io.types import Track

# --- guarded heavy import --------------------------------------------------
# The module must PARSE and its pure-logic tests must run on a bare interpreter.
# numpy is optional and only ever used as a micro-optimisation; all maths below
# is plain float arithmetic so nothing here depends on it.
try:  # pragma: no cover - exercised both with and without numpy installed
    import numpy as _np  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover
    _np = None  # type: ignore

XYXY = Tuple[float, float, float, float]
Vec2 = Tuple[float, float]


# ===========================================================================
# EMA box smoother (independent, reusable, pure logic)
# ===========================================================================

class BoxSmoother:
    """Per-key exponential-moving-average smoother for ``xyxy`` boxes.

    For each key (typically a track id) it keeps the last smoothed box and blends
    every new measurement in with weight ``alpha``::

        s <- alpha * measurement + (1 - alpha) * s_prev

    ``alpha`` in ``(0, 1]``:

    * ``alpha = 1.0`` is a pass-through (no smoothing, no lag).
    * smaller ``alpha`` -> smoother output, more lag on genuine motion.

    The very first measurement for a key is adopted verbatim (there is nothing to
    blend with), so the smoother never injects an artificial start transient.

    The class owns no notion of frames or time; it is a plain signal filter, which
    keeps it trivially unit-testable and reusable outside the recovery buffer.
    """

    def __init__(self, alpha: float = 0.5):
        a = float(alpha)
        if not (0.0 < a <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha!r}")
        self.alpha = a
        self._state: "OrderedDict[object, XYXY]" = OrderedDict()

    def smooth(self, key: object, xyxy: XYXY) -> XYXY:
        """Return the smoothed box for ``key`` given a new measurement.

        Updates and returns the running average. Pass-through on the first sample
        for a key and whenever ``alpha == 1.0``.
        """
        x1, y1, x2, y2 = (float(xyxy[0]), float(xyxy[1]),
                          float(xyxy[2]), float(xyxy[3]))
        prev = self._state.get(key)
        if prev is None or self.alpha >= 1.0:
            out: XYXY = (x1, y1, x2, y2)
        else:
            a = self.alpha
            b = 1.0 - a
            out = (a * x1 + b * prev[0],
                   a * y1 + b * prev[1],
                   a * x2 + b * prev[2],
                   a * y2 + b * prev[3])
        self._state[key] = out
        return out

    def peek(self, key: object) -> Optional[XYXY]:
        """Last smoothed box for ``key`` without updating, or ``None``."""
        return self._state.get(key)

    def reset(self, key: Optional[object] = None) -> None:
        """Forget one key's state, or all keys when ``key`` is ``None``."""
        if key is None:
            self._state.clear()
        else:
            self._state.pop(key, None)


# ===========================================================================
# Pure extrapolation helper
# ===========================================================================

def extrapolate_box(xyxy: XYXY, vel_px_per_frame: Vec2, n_frames: float) -> XYXY:
    """Translate a box by ``vel * n_frames`` (constant-velocity, size preserved).

    The whole box is shifted by the same displacement, so its width/height are
    kept — the standard short-horizon assumption for a coasting detection (object
    size changes far more slowly than position). ``n_frames`` may be fractional.
    """
    dx = float(vel_px_per_frame[0]) * float(n_frames)
    dy = float(vel_px_per_frame[1]) * float(n_frames)
    return (xyxy[0] + dx, xyxy[1] + dy, xyxy[2] + dx, xyxy[3] + dy)


def _centroid(xyxy: XYXY) -> Vec2:
    return ((xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0)


# ===========================================================================
# Per-id history / motion state
# ===========================================================================

class _Hist:
    """Rolling motion state for one track id inside the recovery buffer.

    Tracks just enough to extrapolate: the last *observed* (real) box and frame,
    an EMA of per-frame centroid velocity (px/frame), and how many consecutive
    frames we have been filling. The template Track preserves identity-ish fields
    (cls / alliance / is_flight / field_xy_m) on synthesized frames.
    """

    __slots__ = (
        "tid", "last_box", "last_centroid", "last_obs_frame",
        "vel", "_has_vel", "fill_count", "template",
        "age", "hits",
    )

    def __init__(self, track: Track, frame_idx: int):
        self.tid = track.tid
        self.last_box: XYXY = tuple(float(v) for v in track.xyxy)  # type: ignore[assignment]
        self.last_centroid: Vec2 = _centroid(self.last_box)
        self.last_obs_frame = frame_idx
        self.vel: Vec2 = (0.0, 0.0)
        self._has_vel = False
        self.fill_count = 0
        self.template = track
        self.age = int(track.age)
        self.hits = int(track.hits)

    def observe(self, track: Track, frame_idx: int, vel_alpha: float) -> None:
        """Fold in a real observation: refresh box/template and update velocity.

        Velocity is the per-frame centroid delta, EMA-smoothed. The *first* delta
        seeds the EMA directly (no blend against the zero seed), so on constant
        velocity the estimate is exact after just two observations instead of
        easing in from zero — which keeps gap-fill accurate with little warm-up.
        """
        box: XYXY = tuple(float(v) for v in track.xyxy)  # type: ignore[assignment]
        c = _centroid(box)
        df = frame_idx - self.last_obs_frame
        if df > 0:
            step = ((c[0] - self.last_centroid[0]) / df,
                    (c[1] - self.last_centroid[1]) / df)
            if not self._has_vel:
                self.vel = step
                self._has_vel = True
            else:
                a = vel_alpha
                self.vel = (a * step[0] + (1.0 - a) * self.vel[0],
                            a * step[1] + (1.0 - a) * self.vel[1])
        self.last_box = box
        self.last_centroid = c
        self.last_obs_frame = frame_idx
        self.fill_count = 0
        self.template = track
        self.age = int(track.age)
        self.hits = int(track.hits)

    def gap(self, frame_idx: int) -> int:
        """Frames since the last real observation (0 == observed this frame)."""
        return frame_idx - self.last_obs_frame


# ===========================================================================
# Result record
# ===========================================================================

@dataclass(frozen=True)
class RecoveryResult:
    """Outcome of one :meth:`TemporalRecovery.step`.

    Attributes
    ----------
    tracks:
        Emitted tracks for this frame — observed (optionally smoothed) tracks
        followed by any synthesized gap-fill tracks. This is exactly what
        :meth:`TemporalRecovery.update` returns.
    present_tids:
        Ids that had a real observation this frame.
    filled_tids:
        Ids that were synthesized (extrapolated) this frame because they were
        missing but seen within ``max_gap`` frames.
    """
    tracks: List[Track]
    present_tids: Tuple[int, ...]
    filled_tids: Tuple[int, ...]


# ===========================================================================
# The recovery buffer
# ===========================================================================

class TemporalRecovery:
    """Velocity-extrapolation gap fill + optional EMA smoothing over a track stream.

    Call :meth:`update` (or :meth:`step`) once per frame with the tracks the
    upstream produced for that frame. Ids seen recently but missing this frame are
    re-emitted as extrapolated placeholders for up to ``max_gap`` frames.

    Construction mirrors the tracker layers: ``TemporalRecovery(cfg_dict, fps=...)``.

    Config (all optional)
    ----------------------
    max_gap_s : float
        Recovery window in **seconds** (preferred, fps-correct). If absent,
        ``max_gap_frames`` (a legacy count authored at 30fps) is rescaled to the
        real fps, identically to the trackers' ``max_age`` handling. Default 6
        legacy frames (= 0.2s, i.e. 12 frames @60fps).
    smooth : bool
        Enable EMA smoothing of observed boxes (default ``True``).
    smooth_alpha : float
        EMA weight for the box smoother, ``(0, 1]`` (default 0.5).
    vel_alpha : float
        EMA weight for the velocity estimate used during extrapolation
        (default 0.5).
    max_extrapolation_px : float | None
        Optional cap on how far (pixels) a single fill may be translated from the
        last observed box; guards against a wild velocity estimate launching a
        box across the frame during a long gap. ``None`` disables the cap.
    """

    def __init__(self, cfg: Optional[dict] = None, fps: float = 60.0):
        self.cfg = cfg or {}
        self.fps = float(fps)

        self.max_gap = self._resolve_max_gap()

        self.smooth_enabled = bool(self.cfg.get("smooth", True))
        self.smooth_alpha = float(self.cfg.get("smooth_alpha", 0.5))
        self.vel_alpha = _clamp01(float(self.cfg.get("vel_alpha", 0.5)))
        cap = self.cfg.get("max_extrapolation_px", None)
        self.max_extrapolation_px: Optional[float] = (
            None if cap is None else float(cap))

        self._smoother = BoxSmoother(self.smooth_alpha) if self.smooth_enabled else None
        self._hist: "OrderedDict[int, _Hist]" = OrderedDict()

        # introspection of the latest step
        self.filled_tids: Tuple[int, ...] = ()
        self.present_tids: Tuple[int, ...] = ()

    # -- config ------------------------------------------------------------
    def _resolve_max_gap(self) -> int:
        """Gap window in FRAMES, derived from SECONDS via the real fps.

        Prefers an explicit ``max_gap_s``. For backward-compat with a legacy
        ``max_gap_frames`` (authored against 30fps) we interpret it as
        30ths-of-a-second and re-derive at the real fps, exactly as the trackers
        do for ``max_age`` — so the same config means the same wall-clock window
        at any capture rate.
        """
        if "max_gap_s" in self.cfg:
            return seconds_to_frames(float(self.cfg["max_gap_s"]), self.fps)
        legacy_frames = float(self.cfg.get("max_gap_frames", 6))
        return seconds_to_frames(legacy_frames / 30.0, self.fps)

    # -- main step ---------------------------------------------------------
    def update(self, frame_idx: int, tracks: Sequence[Track]) -> List[Track]:
        """Tracker-compatible entry point: returns the emitted track list.

        Equivalent to ``self.step(frame_idx, tracks).tracks``.
        """
        return self.step(frame_idx, tracks).tracks

    def step(self, frame_idx: int, tracks: Sequence[Track]) -> RecoveryResult:
        """Process one frame and return observed + filled tracks with the split.

        Order of operations:
          1. ingest every observed track (update history + velocity, smooth box),
          2. for ids missing this frame but within ``max_gap``, synthesize a fill,
          3. drop ids whose gap now exceeds ``max_gap``.
        """
        present_ids: List[int] = []
        out: List[Track] = []
        seen_this_frame: set[int] = set()

        # 1) observed tracks
        for tr in tracks:
            tid = tr.tid
            seen_this_frame.add(tid)
            present_ids.append(tid)

            h = self._hist.get(tid)
            if h is None:
                h = _Hist(tr, frame_idx)
                self._hist[tid] = h
            else:
                h.observe(tr, frame_idx, self.vel_alpha)
                self._hist.move_to_end(tid)

            emit = tr
            if self._smoother is not None:
                sm_box = self._smoother.smooth(tid, tr.xyxy)
                emit = replace(tr, xyxy=sm_box, centroid_px=_centroid(sm_box))
                # keep history's notion of the last *observed* box raw (set above
                # in _Hist via the un-smoothed track) so extrapolation rides the
                # true measurement, not the lagged smoothed value.
            out.append(emit)

        # 2) gap fill for recently-seen-but-missing ids
        filled_ids: List[int] = []
        for tid, h in self._hist.items():
            if tid in seen_this_frame:
                continue
            gap = h.gap(frame_idx)
            if 1 <= gap <= self.max_gap:
                filled = self._synthesize(h, frame_idx, gap)
                if filled is not None:
                    out.append(filled)
                    filled_ids.append(tid)

        # 3) cull ids that have now exceeded the recovery window
        self._cull(frame_idx)

        self.present_tids = tuple(present_ids)
        self.filled_tids = tuple(filled_ids)
        return RecoveryResult(
            tracks=out,
            present_tids=self.present_tids,
            filled_tids=self.filled_tids,
        )

    # -- helpers -----------------------------------------------------------
    def _synthesize(self, h: _Hist, frame_idx: int, gap: int) -> Optional[Track]:
        """Build an extrapolated placeholder Track for a missing id.

        Position = last observed box translated by ``vel * gap`` (optionally
        distance-capped). ``hits`` is held at its last value (no new measurement)
        while ``age`` advances by the gap, so the fill is structurally identical
        to a coasting track and never inflates the hit count.
        """
        box = extrapolate_box(h.last_box, h.vel, gap)
        box = self._cap_extrapolation(h.last_box, box)
        h.fill_count = gap

        vel_px_s = (h.vel[0] * self.fps, h.vel[1] * self.fps)
        return replace(
            h.template,
            xyxy=box,
            centroid_px=_centroid(box),
            vel_mps=vel_px_s,            # px/s; geometry layer converts to m/s
            age=h.age + gap,
            hits=h.hits,                 # unchanged: a fill is not a measurement
        )

    def _cap_extrapolation(self, last_box: XYXY, box: XYXY) -> XYXY:
        """Clamp the fill's centroid displacement to ``max_extrapolation_px``.

        Scales the whole translation down uniformly if the centroid moved farther
        than the cap, preserving box size and direction. No-op when uncapped.
        """
        cap = self.max_extrapolation_px
        if cap is None:
            return box
        c0 = _centroid(last_box)
        c1 = _centroid(box)
        dx = c1[0] - c0[0]
        dy = c1[1] - c0[1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist <= cap or dist == 0.0:
            return box
        s = cap / dist
        return extrapolate_box(last_box, (dx * s, dy * s), 1.0)

    def _cull(self, frame_idx: int) -> None:
        """Forget ids whose gap exceeds ``max_gap`` (also clears smoother state)."""
        dead = [tid for tid, h in self._hist.items()
                if h.gap(frame_idx) > self.max_gap]
        for tid in dead:
            del self._hist[tid]
            if self._smoother is not None:
                self._smoother.reset(tid)

    # -- introspection -----------------------------------------------------
    def tracked_ids(self) -> Tuple[int, ...]:
        """Ids currently retained in the buffer (observed or within the gap)."""
        return tuple(self._hist.keys())

    def velocity_of(self, tid: int) -> Optional[Vec2]:
        """Current estimated velocity (px/frame) for an id, or ``None``."""
        h = self._hist.get(tid)
        return None if h is None else h.vel


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


__all__ = [
    "BoxSmoother",
    "TemporalRecovery",
    "RecoveryResult",
    "extrapolate_box",
]
