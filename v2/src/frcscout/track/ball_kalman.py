"""Flight-ball tracker — constant-velocity Kalman tracks for airborne balls, 60Hz.

Only the few in-flight balls are tracked here (FlightDetector proposals). Each track's
trajectory feeds the ballistics solver for 3D arc reconstruction. Speed gating is done
in FIELD m/s (post-homography) when geometry is available, so it's camera-invariant —
unlike v1's px/frame gates. When no homography is available yet (raw pixels) we fall
back to a px/frame speed so the tracker is still useful end-to-end.

Implementation notes
---------------------
* State per ball: constant-velocity Kalman on the centroid (cx, cy, vx, vy).
* Association: predict each track forward, then a SPEED-AWARE gate — the match radius
  grows with the track's current speed (a fast ball moves far between frames) — and a
  greedy nearest-neighbour assignment within that gate.
* Lifecycle: birth on an unmatched detection, coast (predict-only) for up to
  `max_coast` frames through occlusion, cull after that.
* `is_flight`: set True once the field-frame speed sustains above
  `events.shot.min_speed_mps` while the ball has travelled at least
  `events.shot.min_travel_m`. With no homography the same logic runs on a px/frame
  fallback (min_speed_px / a px-travel proxy).
"""
from __future__ import annotations

import math

import numpy as np

from frcscout.io.config import seconds_to_frames  # noqa: F401  (kept for cadence parity)
from frcscout.io.types import Detection, Track


def _center(xyxy) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class _BallKF:
    """Constant-velocity Kalman on a ball centroid (cx, cy, vx, vy)."""

    def __init__(self, tid: int, det: Detection, frame_idx: int, dt: float = 1.0):
        self.tid = tid
        self.cls = det.cls
        self.dt = dt

        cx, cy = _center(det.xyxy)
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=float)
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
        self.P = np.eye(4) * 10.0
        self.P[2:, 2:] *= 100.0
        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 1.0

        # keep the last detected box size so we can re-emit a plausible box
        x1, y1, x2, y2 = det.xyxy
        self.w = x2 - x1
        self.h = y2 - y1

        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.last_frame = frame_idx

        # flight-confirmation bookkeeping (field frame when available, else px)
        self.field_xy = None                     # last known field position (m)
        self.travel_m = 0.0                       # cumulative field travel
        self.travel_px = 0.0                      # cumulative pixel travel (fallback)
        self.sustained_frames = 0                 # consecutive frames over speed gate
        self.is_flight = False

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1

    def update(self, det: Detection, frame_idx: int) -> None:
        z = np.array(_center(det.xyxy), dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        x1, y1, x2, y2 = det.xyxy
        self.w = x2 - x1
        self.h = y2 - y1
        self.hits += 1
        self.time_since_update = 0
        self.last_frame = frame_idx

    @property
    def centroid(self) -> tuple[float, float]:
        return (float(self.x[0]), float(self.x[1]))

    @property
    def speed_px_per_frame(self) -> float:
        return float(math.hypot(self.x[2], self.x[3]))

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        cx, cy = self.centroid
        return (cx - self.w / 2.0, cy - self.h / 2.0,
                cx + self.w / 2.0, cy + self.h / 2.0)


class FlightTracker:
    def __init__(self, cfg: dict, fps: float):
        self.cfg = cfg or {}
        self.fps = float(fps)
        self.dt = 1.0
        self.max_coast = int(self.cfg.get("max_coast_frames", 6))
        self.vel_alpha = float(self.cfg.get("vel_alpha", 0.8))

        # base association gate (pixels) before the speed term widens it
        self.base_gate_px = float(self.cfg.get("gate_px", 40.0))
        # how many px of slack per px/frame of speed
        self.gate_speed_k = float(self.cfg.get("gate_speed_k", 2.0))

        # flight confirmation thresholds (field frame)
        self.shot_speed_mps = float(self.cfg.get("min_speed_mps", 2.0))
        self.shot_min_travel_m = float(self.cfg.get("min_travel_m", 1.0))
        # px fallbacks (used only when no field_xy_m on the tracks)
        self.shot_speed_px = float(self.cfg.get("min_speed_px", 6.0))
        self.shot_min_travel_px = float(self.cfg.get("min_travel_px", 60.0))
        # min consecutive frames over the speed gate to call it sustained
        self.flight_min_frames = int(self.cfg.get("flight_min_frames", 2))

        self._tracks: list[_BallKF] = []
        self._next_id = 0

    # ───────────────────────────── association ───────────────────────────────
    def _gate_radius(self, t: _BallKF) -> float:
        return self.base_gate_px + self.gate_speed_k * t.speed_px_per_frame

    def _associate(self, dets: list[Detection]):
        """Speed-aware greedy nearest-neighbour. Returns
        (matches[(track_idx, det_idx)], unmatched_track_idx, unmatched_det_idx)."""
        if not self._tracks or not dets:
            return [], list(range(len(self._tracks))), list(range(len(dets)))

        det_c = [np.array(_center(d.xyxy)) for d in dets]
        pairs: list[tuple[float, int, int]] = []  # (dist, track_idx, det_idx)
        for ti, t in enumerate(self._tracks):
            pc = np.array(t.centroid)
            gate = self._gate_radius(t)
            for di, dc in enumerate(det_c):
                dist = float(np.linalg.norm(pc - dc))
                if dist <= gate:
                    pairs.append((dist, ti, di))

        pairs.sort(key=lambda p: p[0])  # nearest first
        used_t: set[int] = set()
        used_d: set[int] = set()
        matches: list[tuple[int, int]] = []
        for dist, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            matches.append((ti, di))
            used_t.add(ti)
            used_d.add(di)

        unmatched_t = [i for i in range(len(self._tracks)) if i not in used_t]
        unmatched_d = [i for i in range(len(dets)) if i not in used_d]
        return matches, unmatched_t, unmatched_d

    # ───────────────────────────── main step ─────────────────────────────────
    def update(self, frame_idx: int, dets: list[Detection]) -> list[Track]:
        # 1) predict
        for t in self._tracks:
            t.predict()

        # 2) speed-aware greedy NN association
        matches, _unmatched_t, unmatched_d = self._associate(dets)
        for ti, di in matches:
            self._tracks[ti].update(dets[di], frame_idx)

        # 3) birth from unmatched detections
        for di in unmatched_d:
            self._tracks.append(_BallKF(self._next_id, dets[di], frame_idx, dt=self.dt))
            self._next_id += 1

        # 4) cull tracks that have coasted too long
        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_coast]

        # 5) emit + flight confirmation
        out: list[Track] = []
        for t in self._tracks:
            trk = self._to_track(t)
            self._update_flight_flag(t, trk)
            trk.is_flight = t.is_flight
            out.append(trk)
        return out

    # ──────────────────────── flight confirmation ────────────────────────────
    def _update_flight_flag(self, t: _BallKF, trk: Track) -> None:
        """Mark is_flight once field-frame speed sustains above the shot threshold over
        a minimum travel. Uses field_xy_m when the geometry layer has populated it,
        else a px/frame fallback so the flag still works on raw detections."""
        cur_field = trk.field_xy_m
        if cur_field is not None:
            # accumulate field travel
            if t.field_xy is not None:
                step = math.hypot(cur_field[0] - t.field_xy[0],
                                  cur_field[1] - t.field_xy[1])
                t.travel_m += step
                speed_mps = step * self.fps  # per-frame step -> per-second
            else:
                speed_mps = 0.0
            t.field_xy = cur_field
            over = speed_mps >= self.shot_speed_mps
            t.sustained_frames = t.sustained_frames + 1 if over else 0
            if (t.sustained_frames >= self.flight_min_frames
                    and t.travel_m >= self.shot_min_travel_m):
                t.is_flight = True
        else:
            # px/frame fallback
            speed_px = t.speed_px_per_frame
            t.travel_px += speed_px
            over = speed_px >= self.shot_speed_px
            t.sustained_frames = t.sustained_frames + 1 if over else 0
            if (t.sustained_frames >= self.flight_min_frames
                    and t.travel_px >= self.shot_min_travel_px):
                t.is_flight = True

    def _to_track(self, t: _BallKF) -> Track:
        vx, vy = float(t.x[2]), float(t.x[3])
        return Track(
            tid=t.tid,
            cls=t.cls,
            xyxy=t.xyxy,
            centroid_px=t.centroid,
            vel_mps=(vx * self.fps, vy * self.fps),  # px/s; geometry converts to m/s
            age=t.age,
            hits=t.hits,
            is_flight=t.is_flight,
        )
