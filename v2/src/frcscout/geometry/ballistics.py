"""Single-camera 3D ball-arc reconstruction — the honest version.

A single view has no depth: an image point is a ray, not a 3D position. We recover the
3D arc by CONSTRAINING the trajectory to projectile physics and fitting it to the
observed 2D track:

    - launch point (x0,y0,z0): the shooter robot's known field XY (ground), z0 ~ launch height
    - dynamics: x(t)=x0+vx t,  y(t)=y0+vy t,  z(t)=z0+vz t - 0.5 g t^2   (g=9.81)
    - observation: each tracked pixel (u_i, v_i) must equal the camera projection P(x,y,z)(t_i)

We solve for (vx, vy, vz, z0) by least-squares minimizing reprojection error of the
parabola through the observed pixels, given the camera matrix P (from full PnP pose).
Output is an arc WITH a covariance / error estimate — never claimed as metric truth.

If two broadcast angles are ever available, triangulate instead (exact). Until then this
is an estimate, and the API returns its own uncertainty so downstream code can weight it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.81


@dataclass
class ArcFit:
    launch_xyz_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]
    apex_height_m: float
    reproj_err_px: float
    confidence: float          # 0..1 from reproj error + #observations + geometry conditioning
    impact_xy_m: tuple[float, float] | None    # where it crosses rim height / ground


def _traj_3d(theta: np.ndarray, t: np.ndarray, launch_xy_m) -> np.ndarray:
    """3D positions (N x 3) under the projectile model anchored at (launch_xy, z0)."""
    vx, vy, vz, z0 = theta
    x0, y0 = launch_xy_m
    x = x0 + vx * t
    y = y0 + vy * t
    z = z0 + vz * t - 0.5 * G * t * t
    return np.stack([x, y, z], axis=1)


def _project(P: np.ndarray, pts3d: np.ndarray) -> np.ndarray:
    """Project N x 3 world points through P (3x4) -> N x 2 pixels. NaN where behind/at z=0."""
    n = pts3d.shape[0]
    hom = np.hstack([pts3d, np.ones((n, 1), dtype=np.float64)])  # N x 4
    proj = hom @ P.T                                             # N x 3
    w = proj[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        px = proj[:, :2] / w
    return px


def fit_arc(observed_px: list[tuple[float, float]],
            times_s: list[float],
            launch_xy_m: tuple[float, float],
            P: np.ndarray,
            launch_z_guess_m: float = 0.6,
            rim_height_m: float | None = None) -> ArcFit | None:
    """Constrained ballistic fit. Needs the full camera projection matrix P (3x4) from PnP.

    Steps:
        1. param theta = [vx, vy, vz, z0]
        2. predict 3D point at each t_i from projectile model anchored at (launch_xy, z0)
        3. project via P -> predicted pixels; residual = observed - predicted
        4. scipy.optimize.least_squares(residuals, theta0)
        5. confidence from final reproj err, n points, and Jacobian conditioning
        6. impact = solve z(t)=rim_height or 0 for downstream HUB-entry test
    Returns None if <4 observations or P is unavailable (degenerate)."""
    if observed_px is None or len(observed_px) < 4 or P is None:
        return None

    from scipy.optimize import least_squares

    obs = np.asarray(observed_px, dtype=np.float64)        # N x 2
    t = np.asarray(times_s, dtype=np.float64)              # N
    P = np.asarray(P, dtype=np.float64)
    if obs.shape[0] != t.shape[0] or P.shape != (3, 4):
        return None
    n = obs.shape[0]

    # Re-zero time so launch is t=0 (anchor + numerical conditioning).
    t = t - t.min()

    def residuals(theta: np.ndarray) -> np.ndarray:
        pts = _traj_3d(theta, t, launch_xy_m)
        pred = _project(P, pts)
        res = (pred - obs).ravel()
        # Guard against points projected behind the camera (NaN/inf) — penalize heavily.
        res = np.where(np.isfinite(res), res, 1e6)
        return res

    # Initial guess: modest upward lob; horizontal velocity unknown -> 0.
    theta0 = np.array([0.0, 0.0, 4.0, float(launch_z_guess_m)], dtype=np.float64)

    try:
        sol = least_squares(residuals, theta0, method="lm", max_nfev=2000)
    except Exception:
        return None

    theta = sol.x
    vx, vy, vz, z0 = (float(v) for v in theta)

    final_res = residuals(theta)
    # per-point RMS reprojection error (px)
    reproj_err = float(np.sqrt(np.mean(final_res ** 2)))

    # ---- apex height ---------------------------------------------------- #
    # z(t) = z0 + vz t - 0.5 g t^2 ; apex at t* = vz/g (if vz>0)
    if vz > 0:
        t_apex = vz / G
        apex_h = z0 + vz * t_apex - 0.5 * G * t_apex ** 2
    else:
        apex_h = z0
    apex_h = float(apex_h)

    # ---- impact point: solve z(t)=target for t>0, take field XY there ----- #
    target_z = float(rim_height_m) if rim_height_m is not None else 0.0
    impact_xy = _solve_impact(vz, z0, target_z, vx, vy, launch_xy_m, t.max())

    # ---- confidence ----------------------------------------------------- #
    confidence = _confidence(reproj_err, n, sol)

    return ArcFit(
        launch_xyz_m=(float(launch_xy_m[0]), float(launch_xy_m[1]), z0),
        velocity_mps=(vx, vy, vz),
        apex_height_m=apex_h,
        reproj_err_px=reproj_err,
        confidence=confidence,
        impact_xy_m=impact_xy,
    )


def _solve_impact(vz: float, z0: float, target_z: float, vx: float, vy: float,
                  launch_xy_m, t_obs_max: float) -> tuple[float, float] | None:
    """Largest positive t with z(t)=target_z, then field XY at that t. None if never reached."""
    # 0.5 g t^2 - vz t + (target_z - z0) = 0  -> a t^2 + b t + c = 0
    a = 0.5 * G
    b = -vz
    c = target_z - z0
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sq = np.sqrt(disc)
    roots = [(-b + sq) / (2 * a), (-b - sq) / (2 * a)]
    pos = [r for r in roots if r > 1e-6]
    if not pos:
        return None
    t_imp = max(pos)  # the descending crossing
    x = launch_xy_m[0] + vx * t_imp
    y = launch_xy_m[1] + vy * t_imp
    return (float(x), float(y))


def _confidence(reproj_err: float, n: int, sol) -> float:
    """0..1 from (a) reprojection error, (b) #observations, (c) Jacobian conditioning.
    A clean fit on many well-conditioned points -> ~1; a noisy/short/degenerate one -> ~0."""
    # (a) reprojection: 2px -> good, falls off beyond.
    err_term = 1.0 / (1.0 + (reproj_err / 2.0) ** 2)

    # (b) observation count: saturates around ~12 points.
    n_term = min(1.0, max(0.0, (n - 4) / 8.0)) * 0.5 + 0.5 if n >= 4 else 0.0

    # (c) Jacobian conditioning: a degenerate (near-rank-deficient) fit is untrustworthy.
    cond_term = 0.5
    try:
        jac = np.asarray(sol.jac, dtype=np.float64)
        if jac.size and jac.shape[0] >= jac.shape[1]:
            sv = np.linalg.svd(jac, compute_uv=False)
            if sv[0] > 0 and sv[-1] > 0:
                cond = sv[0] / sv[-1]
                cond_term = 1.0 / (1.0 + np.log10(max(1.0, cond)) / 4.0)
    except Exception:
        cond_term = 0.5

    conf = float(err_term * n_term * cond_term)
    return max(0.0, min(1.0, conf))
