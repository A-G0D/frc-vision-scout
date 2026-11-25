"""Camera-pose recovery from AprilTags — the foundation of the geolocator layer.

For a fixed broadcast camera this solves ONCE and caches; it only re-solves when a
pan/zoom is detected (tag reprojection error spikes). Two modes:

    - homography (default): map image px -> field ground plane (z=0) meters. Exact for
      anything ON the carpet (robots, resting balls). Needs >=4 tag corners.
    - full PnP: recover camera rotation+translation from known tag 3D positions. Needed
      for airborne-ball ballistics (we need the camera ray, not just the ground plane).

Uses pupil-apriltags for detection. Tag 3D positions come from the real WPILib field map
(configs/apriltags_2026.json) loaded by geometry/tagmap.py; falls back to the YAML field
config geometry if the JSON map is unavailable.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from frcscout.io.types import CameraPose

# Tag-local corner order matching pupil_apriltags `.corners` (bottom-left, bottom-right,
# top-right, top-left when looking at the tag), expressed in the tag's own frame with the
# tag lying in its local x-y plane (z out of the tag face). Multiplied by half the edge.
_TAG_LOCAL_UNIT = np.array(
    [
        [-1.0, -1.0, 0.0],   # corner 0
        [+1.0, -1.0, 0.0],   # corner 1
        [+1.0, +1.0, 0.0],   # corner 2
        [-1.0, +1.0, 0.0],   # corner 3
    ],
    dtype=np.float64,
)

# Default path to the WPILib AprilTag field map JSON.
_DEFAULT_TAG_MAP_PATH = "configs/apriltags_2026.json"


def _rpy_to_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Field-frame rotation from yaw(Z)-pitch(Y)-roll(X) degrees. The tag's local +z is
    its outward normal; tags on the field face inward along this rotated normal."""
    r, p, y = (math.radians(a) for a in (roll_deg, pitch_deg, yaw_deg))
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _homography_from_pnp(rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray) -> np.ndarray | None:
    """Derive the image->ground-plane (z=0) homography from a solved PnP pose.

    The ground plane in field coordinates is z=0. We map every image point to the z=0
    plane via the camera projection: p_cam = R * p_field + t; pixel = K * p_cam / z_cam.
    Inverting the per-column logic:  H = K * [r1 | r2 | t]  where r1,r2 are the first
    two columns of R (the x-y plane of the field coordinate system).
    Returns a 3x3 float64 ndarray, or None on numerical failure.
    """
    import cv2
    R_mat, _ = cv2.Rodrigues(rvec)          # 3x3 rotation matrix
    r1 = R_mat[:, 0:1]                       # first column (field x-axis in cam frame)
    r2 = R_mat[:, 1:2]                       # second column (field y-axis in cam frame)
    t  = tvec.reshape(3, 1)
    # H maps [X_field, Y_field, 1] -> [u*w, v*w, w] with z_field=0
    col = np.hstack([r1, r2, t])             # 3x3: maps [X_field, Y_field, 1] -> pixel
    H_g2i = K @ col                          # ground -> image (pixel = K * [r1|r2|t] * [X Y 1]^T)
    # CameraPose.H is the IMAGE -> ground homography (see io/types.py); the reprojection
    # check, FieldModel, and the engine field-grid all assume that direction. K*[r1|r2|t]
    # is the OPPOSITE mapping (ground -> image), so invert it before returning.
    try:
        H = np.linalg.inv(H_g2i)
    except np.linalg.LinAlgError:
        return None
    if abs(H[2, 2]) < 1e-12:
        return None
    H = H / H[2, 2]                          # normalise so H[2,2] == 1
    return H.astype(np.float64)


class PoseEstimator:
    def __init__(self, field, redetect_every_s: float = 5.0):
        self.field = field
        self._detector = None
        self.redetect_every_s = redetect_every_s

        # --- Load the real WPILib tag map (quaternion-based, true field meters). ---
        # Prefer a path in the field config; fall back to the default JSON path.
        map_path = None
        try:
            map_path = getattr(field, "cfg", None) and field.cfg.get("apriltag_map_path")
        except Exception:
            pass
        if not map_path:
            map_path = _DEFAULT_TAG_MAP_PATH

        self._tagmap: dict = {}           # {tag_id: {'xyz':..., 'quat':...}}
        self._tag_size_m: float = float(getattr(field, "tag_size_m", 0.1651))
        self._tagmap_loaded: bool = False
        try:
            from frcscout.geometry.tagmap import load_tag_map
            if os.path.exists(map_path):
                loaded = load_tag_map(map_path)
                self._tagmap = loaded.get("tags", {})
                self._tagmap_loaded = bool(self._tagmap)
        except Exception:
            pass  # tagmap unavailable — fall back to YAML geometry

        # Build 3D field-frame corners from the real tag map (numpy arrays, 4x3).
        # Keyed by integer tag id. Used for PnP.
        self._tag_corners_map: dict[int, np.ndarray] = self._build_map_corners()

        # Precompute field-frame 3D corners from the YAML field config (fallback).
        self._tag_corners_field: dict[int, np.ndarray] = self._build_tag_corners()

    # ------------------------------------------------------------------ #
    @property
    def tag_count(self) -> int:
        """Number of tags loaded from the real WPILib map (0 if map unavailable)."""
        return len(self._tagmap)

    # ------------------------------------------------------------------ #
    def _build_map_corners(self) -> dict[int, np.ndarray]:
        """Build 4x3 corner arrays (field meters) from the JSON tag map using tagmap.tag_corners_3d."""
        if not self._tagmap:
            return {}
        try:
            from frcscout.geometry.tagmap import tag_corners_3d
        except Exception:
            return {}
        out: dict[int, np.ndarray] = {}
        for tid, spec in self._tagmap.items():
            corners = tag_corners_3d(spec, size_m=self._tag_size_m)
            out[int(tid)] = np.array(corners, dtype=np.float64)   # 4x3
        return out

    def _build_tag_corners(self) -> dict[int, np.ndarray]:
        """Build 4x3 corner arrays from the YAML field config (RPY-based fallback)."""
        out: dict[int, np.ndarray] = {}
        size = float(self.field.tag_size_m)
        for tid, spec in (self.field.tags or {}).items():
            xyz = spec.get("xyz")
            rpy = spec.get("rpy", [0.0, 0.0, 0.0])
            if xyz is None:
                continue
            # cfg comment documents rpy as [yaw, pitch, roll] deg.
            yaw, pitch, roll = (list(rpy) + [0.0, 0.0, 0.0])[:3]
            rot = _rpy_to_matrix(roll, pitch, yaw)
            half = size / 2.0
            local = _TAG_LOCAL_UNIT * half
            corners = (local @ rot.T) + np.asarray(xyz, dtype=np.float64)
            out[int(tid)] = corners
        return out

    def _ensure(self):
        if self._detector is None:
            from pupil_apriltags import Detector
            self._detector = Detector(families=self.field.tag_family,
                                      nthreads=2, quad_decimate=1.0)

    def maybe_update(self, frame, frame_idx: int, prev: CameraPose | None) -> CameraPose | None:
        """Recover pose if we don't have one yet, or periodically to catch a camera move.
        Cheap: tag detection on a downscaled gray frame, then solvePnP/homography."""
        period = max(1, int(round(self.redetect_every_s * 60.0)))  # ~frames between re-checks @60fps
        if prev and prev.valid and (frame_idx % period != 0):       # cache; re-check ~every 5s@60
            return prev
        return self.estimate(frame, frame_idx) or prev

    def estimate(self, frame, frame_idx: int, K: np.ndarray | None = None,
                 region: tuple[float, float] | None = None,
                 max_reproj_px: float = 3.0) -> CameraPose | None:
        """Detect tags, build image<->field correspondences, solve PnP + homography.

        Steps:
            1. gray = cvtColor(frame); tags = detector.detect(gray)
            2. collect (image_corners, field_3d_corners) from the real tag map (JSON-based)
               for each detected tag; fall back to YAML-based corners if tag not in map
            3. if >=4 correspondences: always attempt solvePnP using the 3D field corners
               (the real WPILib positions in field meters)
            4. derive image->ground-plane homography H from the PnP pose via the z=0
               plane formulation: H = K * [r1 | r2 | t]; fall back to findHomography
               if PnP failed or K is unknown
            5. reproj_err = mean reprojection of tag corners; valid if < 2px and n_tags>=2
        Returns CameraPose or None if too few tags this frame (keep the cached pose)."""
        import cv2

        if frame is None:
            return None

        self._ensure()
        gray = frame
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.ascontiguousarray(gray, dtype=np.uint8)

        tags = self._detector.detect(gray)
        h_img, w_img = gray.shape[:2]

        # STACKED BROADCAST: the composite holds 2+ synchronized camera panes, so the SAME
        # physical tag is detected in each at different image positions. Mixing detections
        # from different cameras into ONE PnP is geometrically impossible (it caused 200px+
        # reproj). Restrict to one pane via `region` (a (y0,y1) fraction band) and DEDUP by
        # id, keeping the largest (nearest) view of each tag.
        ry0 = int((region[0] if region else 0.0) * h_img)
        ry1 = int((region[1] if region else 1.0) * h_img)
        best_by_id: dict = {}
        for det in tags:
            tid = int(det.tag_id)
            if tid not in self._tag_corners_map and tid not in self._tag_corners_field:
                continue
            c = np.asarray(det.corners, dtype=np.float32)
            ctr = getattr(det, "center", None)         # pupil dets have .center; be robust
            cy_det = float(ctr[1]) if ctr is not None else float(c[:, 1].mean())
            if not (ry0 <= cy_det < ry1):
                continue
            area = abs(cv2.contourArea(c))
            if tid not in best_by_id or area > best_by_id[tid][0]:
                best_by_id[tid] = (area, det)

        img_pts: list[list[float]] = []
        ground_pts: list[list[float]] = []
        obj_pts: list[list[float]] = []
        for _area, det in best_by_id.values():
            tid = int(det.tag_id)
            corners3d = self._tag_corners_map.get(tid)
            if corners3d is None:
                corners3d = self._tag_corners_field.get(tid)
            if corners3d is None:
                continue
            det_corners = np.asarray(det.corners, dtype=np.float64)   # 4x2 image px
            # natural 1:1 corner pairing — tag_corners_3d now returns BL,TL,TR,BR to match
            # pupil_apriltags' detected-corner order (validated 0.7px on real frames).
            for k in range(4):
                img_pts.append([det_corners[k][0], det_corners[k][1]])
                ground_pts.append([float(corners3d[k][0]), float(corners3d[k][1])])
                obj_pts.append([float(corners3d[k][0]), float(corners3d[k][1]), float(corners3d[k][2])])
        n_tags = len(best_by_id)

        if n_tags < 2 or len(img_pts) < 8:
            return None  # need >=2 distinct tags (8 pts) for a stable single-camera solve

        img_arr    = np.asarray(img_pts,    dtype=np.float64)
        ground_arr = np.asarray(ground_pts, dtype=np.float64)
        obj_arr    = np.asarray(obj_pts,    dtype=np.float64).reshape(-1, 1, 3)
        img_pnp    = img_arr.reshape(-1, 1, 2)

        # ------------------------------------------------------------------ #
        # Solve PnP. Broadcast cameras pan/ZOOM, so the focal length changes per frame:
        # search a range of focal lengths and keep the lowest-reproj solve (an explicit K,
        # if supplied, is the sole candidate). Principal point = image centre.
        # ------------------------------------------------------------------ #
        if K is not None:
            K_in = np.asarray(K, dtype=np.float64)
            f_cands = [float(K_in[0, 0])]
            cx_p, cy_p = float(K_in[0, 2]), float(K_in[1, 2])
        else:
            f_cands = [700.0, 900.0, 1100.0, 1350.0, 1650.0, 2000.0, 2450.0,
                       3000.0, 3700.0, 4600.0, 5800.0]
            cx_p, cy_p = w_img / 2.0, h_img / 2.0

        best = None   # (reproj_err, rvec_list, tvec_list, K_est)
        for f in f_cands:
            K_f = np.array([[f, 0.0, cx_p], [0.0, f, cy_p], [0.0, 0.0, 1.0]], dtype=np.float64)
            ok, rv, tv = cv2.solvePnP(obj_arr, img_pnp, K_f, None, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            proj, _ = cv2.projectPoints(obj_arr, rv, tv, K_f, None)
            err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_arr, axis=1)))
            if best is None or err < best[0]:
                best = (err, [float(x) for x in rv.ravel()],
                        [float(x) for x in tv.ravel()], K_f)

        if best is None:
            return None
        reproj_err, rvec, tvec, K_est = best

        # image->ground (z=0) homography from the solved pose; planar fallback if it fails.
        H = _homography_from_pnp(
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1), K_est)
        if H is None:
            H_hom, _mask = cv2.findHomography(img_arr, ground_arr, method=cv2.RANSAC,
                                              ransacReprojThreshold=3.0)
            if H_hom is None:
                return None
            H = H_hom

        valid = (n_tags >= 2) and (reproj_err < max_reproj_px)
        return CameraPose(
            H=[[float(c) for c in row] for row in H],
            rvec=rvec,
            tvec=tvec,
            reproj_err_px=reproj_err,
            n_tags=n_tags,
            valid=valid,
        )

    @staticmethod
    def _apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Apply 3x3 homography to Nx2 points -> Nx2."""
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        hom = np.hstack([pts, ones]) @ H.T
        return hom[:, :2] / hom[:, 2:3]
