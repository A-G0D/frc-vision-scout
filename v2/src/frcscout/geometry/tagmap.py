"""Official WPILib 2026 REBUILT AprilTag field map loader (configs/apriltags_2026.json).

Real tag center XYZ + orientation in FIELD METERS (NWU, origin at the blue-alliance wall).
Provides the 3D tag-corner correspondences that geometry/apriltag.py's PnP/homography needs to
localize each camera view to field coordinates — the calibration anchor for BOTH single-view
geolocation and the multi-view triangulation. Pure-python (no numpy/cv2), so it parses and
runs anywhere.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_PATH = "configs/apriltags_2026.json"


def _quat_to_R(w: float, x: float, y: float, z: float):
    """Unit quaternion (W,X,Y,Z) -> 3x3 rotation matrix (row-major lists). No numpy."""
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ]


def load_tag_map(path: str | Path = DEFAULT_PATH) -> dict:
    """Parse the WPILib field JSON -> {'tags': {id: {'xyz':(x,y,z), 'quat':(w,x,y,z)}},
    'field': {'length':m, 'width':m}}."""
    obj = json.load(open(path))
    tags: dict[int, dict] = {}
    for t in obj.get("tags", []):
        tid = int(t["ID"])
        tr = t["pose"]["translation"]
        q = t["pose"]["rotation"]["quaternion"]
        tags[tid] = {
            "xyz": (float(tr["x"]), float(tr["y"]), float(tr["z"])),
            "quat": (float(q["W"]), float(q["X"]), float(q["Y"]), float(q["Z"])),
        }
    f = obj.get("field", {}) or {}
    return {"tags": tags,
            "field": {"length": float(f.get("length", 0.0)), "width": float(f.get("width", 0.0))}}


def tag_corners_3d(tag: dict, size_m: float = 0.1651) -> list[tuple[float, float, float]]:
    """4 corner 3D points (field meters) for a tag given its {'xyz','quat'} and edge length.
    Tag-local frame: x-right, y-up, z-out of the tag face. Corner order is
    **BL, TL, TR, BR** to pair 1:1 with pupil_apriltags' detected-corner output — validated
    by min-reproj grid search on real broadcast frames (0.7px). (An earlier BL,BR,TR,TL
    order gave 200px+ reproj because corners 1 and 3 were swapped vs the detector.)"""
    R = _quat_to_R(*tag["quat"])
    cx, cy, cz = tag["xyz"]
    h = size_m / 2.0
    local = [(-h, -h, 0.0), (-h, h, 0.0), (h, h, 0.0), (h, -h, 0.0)]
    out = []
    for lx, ly, lz in local:
        out.append((
            cx + R[0][0] * lx + R[0][1] * ly + R[0][2] * lz,
            cy + R[1][0] * lx + R[1][1] * ly + R[1][2] * lz,
            cz + R[2][0] * lx + R[2][1] * ly + R[2][2] * lz,
        ))
    return out
