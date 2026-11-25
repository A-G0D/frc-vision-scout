"""Robot tracker — ByteTrack-style IoU/Kalman association over the NPU detections.

Robots are large, few, and move smoothly, so association is easy and IDs are stable.
Runs every frame; between NPU detection refreshes it predicts forward (so a 30Hz
detection cadence still yields 60Hz track output). Alliance is assigned once via the
bumper-color vote and then held (majority over the track's life).

Implementation notes
---------------------
* State is a constant-velocity Kalman filter on the box (cx, cy, w, h) plus their
  velocities — 8-d state, 4-d measurement. This is the classic SORT/ByteTrack model.
* Association is two-stage by IoU:
    1. high-confidence detections vs. all active (recently-seen) tracks,
    2. the leftover low-confidence detections vs. the tracks still unmatched after (1).
  ByteTrack's key idea: low-conf boxes are usually real objects under occlusion, so
  they rescue tracks rather than spawning ID switches.
* Birth/death: an unmatched high-conf detection starts a new track; a track that goes
  `max_age` frames without any hit is culled. `max_age` is derived from SECONDS via the
  capture's real fps (never a hardcoded frame count) — see `_resolve_max_age`.
* Output is emitted every frame from the (predicted) Kalman state, so 30Hz detections
  still yield 60Hz tracks.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from frcscout.io.config import seconds_to_frames
from frcscout.io.types import Detection, Track
from frcscout.detect.robot_net import RobotNet
from frcscout.track.reid import ReIDManager
from frcscout.track.latch_backend import LatchBackend


# ───────────────────────────── geometry helpers ──────────────────────────────
def _iou(a: tuple[float, float, float, float],
         b: tuple[float, float, float, float]) -> float:
    """IoU of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _xyxy_to_z(xyxy) -> np.ndarray:
    """xyxy -> measurement (cx, cy, w, h)."""
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], dtype=float)


def _z_to_xyxy(z) -> tuple[float, float, float, float]:
    cx, cy, w, h = z
    w = max(1.0, float(w))
    h = max(1.0, float(h))
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


# ───────────────────────────── kalman track ──────────────────────────────────
class _KFTrack:
    """Constant-velocity Kalman track on (cx, cy, w, h) for a single robot."""

    def __init__(self, tid: int, det: Detection, frame_idx: int, dt: float = 1.0):
        self.tid = tid
        self.cls = det.cls
        self.dt = dt

        # State: [cx, cy, w, h, vx, vy, vw, vh]
        z = _xyxy_to_z(det.xyxy)
        self.x = np.zeros(8, dtype=float)
        self.x[:4] = z

        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = dt
        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)

        self.P = np.eye(8) * 10.0
        self.P[4:, 4:] *= 100.0          # large initial velocity uncertainty
        self.Q = np.eye(8)
        self.Q[4:, 4:] *= 0.01           # velocity drifts slowly
        self.R = np.eye(4) * 1.0         # measurement noise (pixels)

        self.age = 0                     # frames since birth
        self.hits = 1                    # successful measurement updates
        self.time_since_update = 0       # frames since last hit
        self.last_frame = frame_idx
        self.alliance_votes: Counter = Counter()
        self.alliance = None
        self.field_xy: tuple[float, float] | None = None   # last field-projected pos (m)
        self.appearance = None                              # ReID Appearance, if built

    # -- motion --
    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        # keep size positive
        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])

    def update(self, det: Detection, frame_idx: int) -> None:
        z = _xyxy_to_z(det.xyxy)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.hits += 1
        self.time_since_update = 0
        self.last_frame = frame_idx

    # -- views --
    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return _z_to_xyxy(self.x[:4])

    @property
    def centroid(self) -> tuple[float, float]:
        return (float(self.x[0]), float(self.x[1]))

    @property
    def vel_px(self) -> tuple[float, float]:
        return (float(self.x[4]), float(self.x[5]))


class RobotTracker:
    """ByteTrack-style robot tracker. Stable, monotonic IDs."""

    def __init__(self, cfg: dict, fps: float, reid: "ReIDManager | None" = None,
                 latch: "LatchBackend | None" = None):
        self.cfg = cfg or {}
        self.fps = float(fps)
        self.dt = 1.0  # state advances one frame per update() call

        self.max_age = self._resolve_max_age()
        self.iou_match = float(self.cfg.get("iou_match", 0.4))
        # high-conf threshold splits the two association stages
        self.high_thresh = float(self.cfg.get("track_high_thresh", 0.5))
        # detections below this are ignored entirely (don't even rescue)
        self.low_thresh = float(self.cfg.get("track_low_thresh", 0.1))
        # a new track is confirmed for output after this many hits (1 = immediate)
        self.min_hits = int(self.cfg.get("min_hits", 1))

        # Persistence / Re-ID across gaps LONGER than the Kalman max_age. Optional:
        # a None manager keeps the original ByteTrack behaviour exactly (so the
        # existing tests are unaffected unless re-ID is explicitly enabled).
        if reid is None and self.cfg.get("reid_enabled", False):
            reid = ReIDManager(self.cfg.get("reid", {}), fps=self.fps)
        self.reid = reid

        # Optional latch-tracker backend: visually rescues a track on frames where
        # the detector missed it, so it never has to survive the gap on blind
        # Kalman coasting alone. None (the default) means plain ByteTrack — no
        # import of latch-tracker happens unless the caller opts in.
        if latch is None and self.cfg.get("latch_enabled", False):
            latch = LatchBackend()
        self.latch = latch

        self._tracks: list[_KFTrack] = []
        self._next_id = 0

    def _resolve_max_age(self) -> int:
        """Derive max_age in FRAMES from a SECONDS value using the real fps.

        Prefer an explicit seconds key. For backward-compat with deploy.yaml's legacy
        `max_age_frames` (authored against 30fps), interpret it as 30ths-of-a-second and
        re-derive against the real fps so timing is correct at any capture rate.
        """
        if "max_age_s" in self.cfg:
            return seconds_to_frames(float(self.cfg["max_age_s"]), self.fps)
        legacy_frames = float(self.cfg.get("max_age_frames", 30))
        return seconds_to_frames(legacy_frames / 30.0, self.fps)

    # ───────────────────────────── association ───────────────────────────────
    def _associate(self, tracks: list[_KFTrack], dets: list[Detection]):
        """Greedy IoU association above self.iou_match. Returns
        (matches[(track_idx, det_idx)], unmatched_track_idx, unmatched_det_idx)."""
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))

        iou = np.zeros((len(tracks), len(dets)), dtype=float)
        for ti, t in enumerate(tracks):
            for di, d in enumerate(dets):
                iou[ti, di] = _iou(t.xyxy, d.xyxy)

        matches: list[tuple[int, int]] = []
        used_t: set[int] = set()
        used_d: set[int] = set()
        # greedy: repeatedly take the globally-best remaining pair
        order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
        for ti, di in order:
            ti, di = int(ti), int(di)
            if ti in used_t or di in used_d:
                continue
            if iou[ti, di] < self.iou_match:
                break
            matches.append((ti, di))
            used_t.add(ti)
            used_d.add(di)

        unmatched_t = [i for i in range(len(tracks)) if i not in used_t]
        unmatched_d = [i for i in range(len(dets)) if i not in used_d]
        return matches, unmatched_t, unmatched_d

    # ───────────────────────────── main step ─────────────────────────────────
    def update(self, frame_idx: int, dets: list[Detection], frame=None) -> list[Track]:
        """One tracking step. Predict all tracks forward, two-stage IoU-associate the
        detections, update / birth / cull, then emit live Tracks from the state."""
        # 1) predict every existing track forward one frame
        for t in self._tracks:
            t.predict()

        # 2) split detections by confidence
        high = [d for d in dets if d.conf >= self.high_thresh]
        low = [d for d in dets if self.low_thresh <= d.conf < self.high_thresh]

        track_idx_all = list(range(len(self._tracks)))

        # ── stage 1: high-conf dets vs. all tracks ──
        m1, ut1, ud1 = self._associate(
            [self._tracks[i] for i in track_idx_all], high)
        for ti_local, di in m1:
            self._tracks[track_idx_all[ti_local]].update(high[di], frame_idx)

        # tracks still unmatched after stage 1
        remaining_track_idx = [track_idx_all[i] for i in ut1]

        # ── stage 2: leftover low-conf dets rescue the still-unmatched tracks ──
        m2, ut2, _ud2 = self._associate(
            [self._tracks[i] for i in remaining_track_idx], low)
        for ti_local, di in m2:
            self._tracks[remaining_track_idx[ti_local]].update(low[di], frame_idx)

        # 3) birth: each *unmatched high-conf* detection starts a new track. With a
        #    ReIDManager, first try to REBIND to a recently-lost track (same alliance,
        #    appearance-similar, plausible re-emergence position) instead of minting a
        #    fresh ID. This recovers identity across gaps longer than max_age.
        for di in ud1:
            self._birth(high[di], frame_idx, frame)

        # 3b) latch rescue: for every still-unmatched track, ask the optional visual
        #     backend to look at THIS frame directly. A hit resets time_since_update
        #     to 0 exactly as a real detection would, so the track never has to
        #     survive the gap on blind Kalman extrapolation alone. Matched tracks
        #     still get stepped (with matched_this_frame=True) so the visual
        #     tracker's own motion state stays warm for the next gap.
        if self.latch is not None:
            for t in self._tracks:
                matched_now = t.time_since_update == 0
                rescue_xyxy = self.latch.step(t.tid, frame, t.xyxy, matched_now)
                if rescue_xyxy is not None:
                    det = Detection(cls=t.cls, xyxy=rescue_xyxy, conf=0.0,
                                    frame_idx=frame_idx)
                    t.update(det, frame_idx)

        # 4) death: cull tracks unseen for longer than max_age. Culled tracks are
        #    handed to the ReIDManager so a later birth can re-bind their ID.
        survivors: list[_KFTrack] = []
        for t in self._tracks:
            if t.time_since_update <= self.max_age:
                survivors.append(t)
            else:
                if self.reid is not None:
                    self._note_lost(t, frame_idx)
                if self.latch is not None:
                    self.latch.drop(t.tid)
        self._tracks = survivors
        if self.reid is not None:
            self.reid.expire(frame_idx)

        # 5) alliance vote on matched/visible tracks (those updated this frame)
        if frame is not None:
            for t in self._tracks:
                if t.time_since_update == 0:
                    al = RobotNet.bumper_alliance(frame, t.xyxy)
                    if al:
                        t.alliance_votes[al] += 1
                        t.alliance = t.alliance_votes.most_common(1)[0][0]
                    if self.reid is not None:
                        # refresh the appearance descriptor used for re-binding
                        t.appearance = self.reid.build_appearance(
                            t.alliance, t.xyxy, frame=frame)
                        # opportunistically anchor a team number (permanent id<->team)
                        self.reid.try_anchor(t.tid, t.xyxy, frame=frame)

        # 6) emit live tracks
        return [self._to_track(t) for t in self._tracks if self._is_live(t)]

    # ───────────────────────────── birth / re-bind ───────────────────────────
    def _birth(self, det: Detection, frame_idx: int, frame=None) -> None:
        """Spawn a track for an unmatched high-conf detection, re-binding a
        lost-track ID via the ReIDManager when one credibly matches."""
        alliance = RobotNet.bumper_alliance(frame, det.xyxy) if frame is not None else None
        if self.reid is not None:
            cx, cy = _xyxy_to_z(det.xyxy)[:2]
            tid = self.reid.rebind(
                alliance=alliance, xyxy=det.xyxy, centroid_px=(float(cx), float(cy)),
                frame_idx=frame_idx, frame=frame)
            if tid is not None:
                kf = _KFTrack(tid, det, frame_idx, dt=self.dt)
                if alliance:
                    kf.alliance = alliance
                    kf.alliance_votes[alliance] += 1
                self._tracks.append(kf)
                return
        self._spawn(det, frame_idx)

    def _note_lost(self, t: _KFTrack, frame_idx: int) -> None:
        if self.reid is None:
            return
        app = self.reid.build_appearance(t.alliance, t.xyxy, frame=None)
        # carry the live histogram if one was ever built for this id (kept on the KF)
        if getattr(t, "appearance", None) is not None and t.appearance.has_hist():
            app.hue_hist = t.appearance.hue_hist
        self.reid.note_lost(
            tid=t.tid, appearance=app,
            last_centroid_px=t.centroid, vel_px_per_frame=t.vel_px,
            last_field_xy=t.field_xy, last_frame=t.last_frame)

    def _spawn(self, det: Detection, frame_idx: int) -> None:
        self._tracks.append(_KFTrack(self._next_id, det, frame_idx, dt=self.dt))
        self._next_id += 1

    def _is_live(self, t: _KFTrack) -> bool:
        """Confirmed tracks are reported every frame (even while coasting); brand-new
        tracks must reach min_hits before being shown."""
        if t.hits >= self.min_hits:
            return True
        # not yet confirmed: only show if currently seen
        return t.time_since_update == 0

    def _to_track(self, t: _KFTrack) -> Track:
        vx, vy = t.vel_px
        return Track(
            tid=t.tid,
            cls=t.cls,
            xyxy=t.xyxy,
            centroid_px=t.centroid,
            vel_mps=(vx * self.fps, vy * self.fps),  # px/frame -> px/s (geometry refines)
            alliance=t.alliance,
            age=t.age,
            hits=t.hits,
            is_flight=False,
        )
