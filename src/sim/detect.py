"""
detect.py — per-frame ROBOT + BALL detection for the single static wide camera
of this FRC REBUILT (2026) playoff match (Upper Bracket R4 M11, 640x360).

Because the camera is static, the strongest robot cue is MOTION against a median
background, combined with bumper COLOR and the carpet trapezoid. The static HUB
boxes vanish in the background subtraction and are additionally masked out, so
they are never reported as robots.

Pipeline
--------
build_background(video, t0, t1)
    Median of frames sampled over a stable wide-camera window -> clean BGR plate
    in which moving robots/balls/people are averaged away but the HUBs, carpet,
    tape and back wall survive.

detect_robots(frame, bg)
    robot_pixel = MOTION(|frame-bg|) AND CARPET AND (RED or BLUE bumper color)
                  AND NOT inside a HUB region.
    Connected blobs are filtered to robot-sized, reasonably compact shapes and
    labelled by which bumper-color mask they came from. Aim: the ~6 robots with
    few false positives.

detect_balls(frame)
    Carpet-masked yellow blobs tuned for HIGH RECALL (single-ball + cluster
    split), few crowd / HUB-logo false positives. (No motion gate — balls are
    often still.)

Interface (exact):
    build_background(video_path, t0=0.0, t1=None) -> bg (BGR ndarray)
    detect_robots(frame, bg) -> [{'bbox':(x,y,w,h),'centroid':(cx,cy),
                                  'alliance':'blue'|'red'}]
    detect_balls(frame) -> [{'centroid':(cx,cy), 'r':int}]

Classical CV only (OpenCV + numpy), deterministic.
"""

from __future__ import annotations

import cv2
import numpy as np

# --- field geometry: prefer the shared field.py, else inline a carpet poly ----
try:
    from . import field as _field  # type: ignore
except Exception:  # pragma: no cover - fallback for direct execution
    try:
        import field as _field  # type: ignore
    except Exception:
        _field = None

# Inline fallback carpet trapezoid (TL, TR, BR, BL) if field.py is unavailable.
_FALLBACK_POLY_NORM = [
    (0.075, 0.315), (0.925, 0.315), (1.000, 0.550), (0.000, 0.550),
]
# Inline fallback HUB regions (x, y, w, h) normalized + alliance.
_FALLBACK_HUBS_NORM = [
    (0.120, 0.320, 0.120, 0.165, "blue"),
    (0.710, 0.320, 0.120, 0.165, "red"),
]


# ---------------------------------------------------------------------------
# Geometry helpers (cached masks)
# ---------------------------------------------------------------------------
_CARPET_CACHE: dict = {}
_HUBMASK_CACHE: dict = {}


# We use our own measured carpet trapezoid (top at 0.325 sits just below the tan
# back wall to keep crowd motion out; bottom at 0.545 sits at the front barrier
# so refs / player stations / the reflected lower region are excluded). field.py
# is consulted only as a fallback — it is tuned by a sibling module and its
# exact edges drift, so for detection robustness we pin our own.
_CARPET_POLY_NORM = [
    (0.075, 0.325), (0.925, 0.325), (1.000, 0.545), (0.000, 0.545),
]


def _carpet_mask(w: int, h: int) -> np.ndarray:
    key = (w, h)
    m = _CARPET_CACHE.get(key)
    if m is None:
        poly = np.array([(int(x * w), int(y * h))
                         for x, y in _CARPET_POLY_NORM], np.int32)
        m = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(m, poly, 255)
        _CARPET_CACHE[key] = m
    return m


def _hub_regions(w: int, h: int) -> list:
    if _field is not None:
        return _field.hub_regions(w, h)
    out = []
    for nx, ny, nw, nh, al in _FALLBACK_HUBS_NORM:
        out.append({"bbox": (int(nx * w), int(ny * h), int(nw * w), int(nh * h)),
                    "alliance": al})
    return out


# Measured exclusion zones (normalized x0,y0,x1,y1) for the static dark-navy HUB
# box BODIES. These boxes carry alliance trim and stand against the moving crowd,
# producing color+motion noise that is NOT a robot. We exclude only the box
# bodies (NOT the open carpet in front of the ramps) so robots scoring near a HUB
# are still detected. Calibrated directly from this match's frames (independent
# of field.py, which a sibling module tunes for a different purpose).
_EXCLUDE_NORM = [
    (0.165, 0.300, 0.305, 0.475),   # blue HUB dark box + white target panels (L)
    (0.680, 0.300, 0.820, 0.485),   # red HUB dark box + white target panels (R)
    (0.000, 0.300, 0.045, 0.545),   # far-left edge banner / blue player wall
    (0.955, 0.300, 1.000, 0.545),   # far-right edge banner / red player wall
]


def _hub_mask(w: int, h: int) -> np.ndarray:
    """Mask (255 = inside a HUB/ramp exclusion zone) to suppress false robots."""
    key = (w, h)
    m = _HUBMASK_CACHE.get(key)
    if m is None:
        m = np.zeros((h, w), np.uint8)
        for x0, y0, x1, y1 in _EXCLUDE_NORM:
            cv2.rectangle(m, (int(x0 * w), int(y0 * h)),
                          (int(x1 * w), int(y1 * h)), 255, -1)
        _HUBMASK_CACHE[key] = m
    return m


# ---------------------------------------------------------------------------
# HSV gates (OpenCV H is 0..179)
# ---------------------------------------------------------------------------
# Bumper colors: saturated red / blue. High S+V isolate vivid bumpers and reject
# the dark navy HUB bodies, muted tape and shadowed crowd.
_BLUE_LO = np.array([100, 110, 80])
_BLUE_HI = np.array([128, 255, 255])
_RED1_LO = np.array([0, 110, 80]); _RED1_HI = np.array([10, 255, 255])
_RED2_LO = np.array([168, 110, 80]); _RED2_HI = np.array([179, 255, 255])

# Yellow FUEL foam balls.
_YEL_LO = np.array([18, 80, 90], np.uint8)
_YEL_HI = np.array([40, 255, 255], np.uint8)

_K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------
def build_background(video_path: str, t0: float = 0.0, t1=None,
                     max_samples: int = 40) -> np.ndarray:
    """Median-background plate (BGR) over a stable wide-camera window [t0, t1].

    Samples up to ``max_samples`` evenly spaced frames in the window and takes
    the per-pixel median. Robots/balls/people average away; HUBs/carpet survive.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        f0 = int(round(t0 * fps))
        f1 = total if t1 is None else min(total, int(round(t1 * fps)))
        f0 = max(0, min(f0, total - 1))
        f1 = max(f0 + 1, min(f1, total))
        n = min(max_samples, f1 - f0)
        idxs = np.linspace(f0, f1 - 1, n).astype(int)
        frames = []
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, fr = cap.read()
            if ok:
                frames.append(fr)
        if not frames:
            raise RuntimeError("No frames read for background.")
        bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
        return bg
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Robots
# ---------------------------------------------------------------------------
# At this wide zoom a robot is a small (~18-40 px) compact object whose lower
# band is a saturated red/blue bumper. The bumper alone is only a few pixels
# tall, so we detect robots as MOTION blobs (the whole robot body moves against
# the static plate) that CONTAIN enough saturated bumper color, then classify
# the alliance by which color dominates inside the blob. Balls also move but are
# smaller and yellow; people/refs at the edges lack bumper color or fall outside
# the carpet.  A pure strong-color pass is unioned in as a fallback for robots
# that happen to be momentarily still.
_R_MIN_AREA = 50        # min motion-blob area for a robot body
_R_MAX_AREA = 2600      # larger = merged robots / clutter
_R_MIN_DIM = 6
_R_MAX_W = 60
_R_MAX_H = 60
_R_MIN_FILL = 0.26      # contour area / bbox area (robots are roughly solid)
_R_MAX_ASPECT = 2.8
_R_MIN_BUMPER_PX = 6    # min bumper-color px inside a blob to call it a robot
_R_DOMINANCE = 1.15     # color must beat the other by this ratio to assign
_MOTION_THRESH = 20     # |frame-bg| gray threshold


def _bumper_masks(hsv: np.ndarray):
    blue = cv2.inRange(hsv, _BLUE_LO, _BLUE_HI)
    red = cv2.inRange(hsv, _RED1_LO, _RED1_HI) | cv2.inRange(hsv, _RED2_LO, _RED2_HI)
    return blue, red


def _classify_blob(x, y, w, h, blue_m, red_m):
    """Return ('blue'|'red'|None, nblue, nred) for the bumper color inside a bbox.
    Bumpers sit on the lower ~60% of the robot, so weight that band."""
    by0 = y + int(0.30 * h)
    bsub = blue_m[by0:y + h, x:x + w]
    rsub = red_m[by0:y + h, x:x + w]
    nb = int(cv2.countNonZero(bsub))
    nr = int(cv2.countNonZero(rsub))
    if max(nb, nr) < _R_MIN_BUMPER_PX:
        return None, nb, nr
    if nb >= nr * _R_DOMINANCE:
        return "blue", nb, nr
    if nr >= nb * _R_DOMINANCE:
        return "red", nb, nr
    # ambiguous → pick the larger
    return ("blue" if nb >= nr else "red"), nb, nr


def detect_robots(frame: np.ndarray, bg: np.ndarray) -> list:
    """Detect the alliance robots in one frame against the median background.

    Returns a list of {'bbox':(x,y,w,h),'centroid':(cx,cy),'alliance':'blue'|'red'}.
    """
    h, w = frame.shape[:2]
    carpet = _carpet_mask(w, h)
    hubmask = _hub_mask(w, h)
    valid = cv2.bitwise_and(carpet, cv2.bitwise_not(hubmask))

    blur = cv2.GaussianBlur(frame, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    blue_m, red_m = _bumper_masks(hsv)
    blue_v = cv2.bitwise_and(blue_m, valid)
    red_v = cv2.bitwise_and(red_m, valid)

    # Motion mask: robots move vs. the static plate. Drop pure-yellow (ball)
    # pixels so rolling balls don't masquerade as robot motion.
    if bg is not None and bg.shape[:2] == frame.shape[:2]:
        diff = cv2.absdiff(frame, bg)
        gdiff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, motion = cv2.threshold(gdiff, _MOTION_THRESH, 255, cv2.THRESH_BINARY)
        yellow = cv2.inRange(hsv, _YEL_LO, _YEL_HI)
        motion = cv2.bitwise_and(motion, cv2.bitwise_not(yellow))
        motion = cv2.bitwise_and(motion, valid)
        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, _K3, iterations=1)
        have_motion = True
    else:
        motion = valid.copy()
        have_motion = False
    motion_dil = cv2.dilate(motion, _K5, iterations=1)

    # Candidate seeds: strong bumper-color blobs (a robot's vivid bumper is the
    # most reliable cue) consolidated into bodies, UNION the motion blobs. Each
    # seed is then required to (a) carry enough bumper color of one alliance and
    # (b) overlap MOTION — so static colored HUB ramps/banners (color but no
    # motion) are rejected, and so are colorless moving people/refs.
    color_seed = cv2.bitwise_or(blue_v, red_v)
    color_seed = cv2.morphologyEx(color_seed, cv2.MORPH_CLOSE, _K5, iterations=2)
    bodies = cv2.bitwise_or(color_seed, motion)
    bodies = cv2.morphologyEx(bodies, cv2.MORPH_CLOSE, _K5, iterations=1)
    bodies = cv2.bitwise_and(bodies, valid)

    vch = hsv[..., 2]
    cnts, _ = cv2.findContours(bodies, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < _R_MIN_AREA or a > _R_MAX_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < _R_MIN_DIM or bh < _R_MIN_DIM or bw > _R_MAX_W or bh > _R_MAX_H:
            continue
        if a / float(bw * bh) < _R_MIN_FILL:
            continue
        if max(bw / bh, bh / bw) > _R_MAX_ASPECT:
            continue
        # Require motion overlap (rejects static ramps/banners).
        if have_motion:
            msub = motion_dil[y:y + bh, x:x + bw]
            if cv2.countNonZero(msub) < max(6, int(0.10 * a)):
                continue
        # Darkness guard: the navy HUB box faces are overwhelmingly dark; a real
        # robot has a brighter top/body. Reject blobs that are almost all dark.
        vpatch = vch[y:y + bh, x:x + bw]
        if float((vpatch < 55).mean()) > 0.80:
            continue
        alliance, nb, nr = _classify_blob(x, y, bw, bh, blue_v, red_v)
        if alliance is None:
            continue
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        if carpet[int(min(h - 1, max(0, cy))), int(min(w - 1, max(0, cx)))] == 0:
            continue
        out.append({"bbox": (x, y, bw, bh), "centroid": (float(cx), float(cy)),
                    "alliance": alliance})
    return out


# ---------------------------------------------------------------------------
# Balls (high recall, tighter spatial guard)
# ---------------------------------------------------------------------------
_B_MIN_AREA = 14        # raised slightly: very small blobs (<4px r) are noise
_B_SINGLE_MAX = 600
_B_CLUSTER_MAX = 24000
_B_MIN_CIRC = 0.30      # raised from 0.22 → tighter circularity rejects oblong refs
_B_MIN_CIRC_LARGE = 0.18  # looser for larger blobs (multi-ball clusters)
_B_BALL_AREA = 170      # px^2 per ball inside a cluster
_B_NMS = 0.55
# Max radius for a single ball (in pixels at 640x360). A foam ball is ~18px r max.
_B_MAX_R_SINGLE = 22
# Inward erosion (px) of the carpet mask before ball detection — keeps balls
# clearly on the carpet and away from the crowd/barrier/ref edges.
_B_CARPET_ERODE_PX = 6

# Ball-specific carpet mask cache (uses eroded carpet polygon)
_BALL_CARPET_CACHE: dict = {}


def _ball_carpet_mask(w: int, h: int) -> np.ndarray:
    """Eroded carpet mask for ball detection — excludes a thin border at carpet
    edges to suppress crowd/barrier/ref false positives that bleed past the
    trapezoid boundary. Uses field.CARPET_POLY_NORM when available, otherwise
    falls back to detect.py's own _CARPET_POLY_NORM."""
    key = (w, h)
    m = _BALL_CARPET_CACHE.get(key)
    if m is None:
        # Build base carpet mask (re-use _carpet_mask which is already cached)
        base = _carpet_mask(w, h).copy()
        # Erode inward by _B_CARPET_ERODE_PX to form a safety margin at every edge
        if _B_CARPET_ERODE_PX > 0:
            ker = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (_B_CARPET_ERODE_PX * 2 + 1, _B_CARPET_ERODE_PX * 2 + 1))
            base = cv2.erode(base, ker, iterations=1)
        _BALL_CARPET_CACHE[key] = base
        m = base
    return m


def _split_cluster(blob_mask: np.ndarray) -> list:
    dist = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return []
    d8 = (dist * 255 / dist.max()).astype(np.uint8)
    ball_r = max(3, int(np.sqrt(_B_BALL_AREA / np.pi)))
    ksz = ball_r * 2 + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    dil = cv2.dilate(d8, ker)
    floor = max(12, int(d8.mean() * 0.5))
    peaks = ((d8 >= dil - 3) & (d8 > floor)).astype(np.uint8) * 255
    n, _lab, _st, cents = cv2.connectedComponentsWithStats(peaks)
    res = []
    for i in range(1, n):
        res.append((int(cents[i][0]), int(cents[i][1]), ball_r))
    if not res:
        M = cv2.moments(blob_mask)
        if M["m00"] > 0:
            res.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]),
                        max(3, ball_r)))
    return res


def _nms_balls(dets: list) -> list:
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: -d[2])
    sup = [False] * len(dets)
    keep = []
    for i, (xi, yi, ri) in enumerate(dets):
        if sup[i]:
            continue
        keep.append(dets[i])
        for j in range(i + 1, len(dets)):
            if sup[j]:
                continue
            xj, yj, rj = dets[j]
            if np.hypot(xi - xj, yi - yj) < _B_NMS * (ri + rj):
                sup[j] = True
    return keep


def detect_balls(frame: np.ndarray) -> list:
    """Detect yellow FUEL balls on the carpet. High recall, low crowd/ref FP.

    Changes vs. original:
    - Uses an eroded carpet mask (_ball_carpet_mask) so blobs touching the
      crowd/barrier/referee edges are rejected before they are even processed.
    - Single-blob circularity threshold raised (0.30) to reject oblong ref
      jerseys and carpet-edge artifacts.
    - Size guard: single-blob radius must be <= _B_MAX_R_SINGLE (22 px) —
      large oblong crowd blobs that somehow pass circularity are dropped.
    - Final centroid check also uses the eroded mask (tighter than before).

    Returns a list of {'centroid':(cx,cy), 'r':int}.
    """
    h, w = frame.shape[:2]
    # Use eroded carpet mask for ball detection to guard against edge FPs
    ball_carpet = _ball_carpet_mask(w, h)
    # Keep the original (un-eroded) carpet only for the robot sibling to use
    # via _carpet_mask. We do NOT touch that function.

    blur = cv2.GaussianBlur(frame, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    raw = cv2.inRange(hsv, _YEL_LO, _YEL_HI)
    # Apply the tighter ball carpet mask immediately — cuts crowd/ref regions
    raw = cv2.bitwise_and(raw, ball_carpet)

    mask = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, _K3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _K3, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < _B_MIN_AREA or a > _B_CLUSTER_MAX:
            continue
        if a <= _B_SINGLE_MAX:
            per = cv2.arcLength(c, True)
            circ = (4 * np.pi * a / (per * per)) if per > 0 else 0.0
            # Tighter circularity for single blobs rejects oblong ref/crowd hits
            if circ < _B_MIN_CIRC:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            # Size guard: reject blobs that are too large to be a single ball
            if r > _B_MAX_R_SINGLE:
                continue
            dets.append((int(x), int(y), max(2, int(r))))
        else:
            # Large clusters: use looser circularity (merged balls look oblong)
            per = cv2.arcLength(c, True)
            circ = (4 * np.pi * a / (per * per)) if per > 0 else 0.0
            if circ < _B_MIN_CIRC_LARGE:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            bm = np.zeros((bh, bw), np.uint8)
            cv2.drawContours(bm, [c - np.array([bx, by])], -1, 255, -1)
            for cx, cy, r in _split_cluster(bm):
                dets.append((cx + bx, cy + by, r))

    balls = _nms_balls(dets)
    out = []
    for x, y, r in balls:
        # Final check: centroid must lie inside the eroded carpet mask
        if ball_carpet[int(min(h - 1, max(0, y))), int(min(w - 1, max(0, x)))]:
            out.append({"centroid": (int(x), int(y)), "r": int(r)})
    return out


# ---------------------------------------------------------------------------
# CLI self-check: annotate real frames
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os

    video = (sys.argv[1] if len(sys.argv) > 1 else "data/sim/match_full.mp4")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "detect_annot"
    os.makedirs(out_dir, exist_ok=True)

    print("Building background (wide segment 0..139s)...")
    bg = build_background(video, 0.0, 139.0)
    cv2.imwrite(os.path.join(out_dir, "_bg.png"), bg)

    cap = cv2.VideoCapture(video)
    # 7 evenly-spaced frames across the wide segment (0..139s = ~4170 frames at 30fps)
    test_frames = [300, 700, 1200, 1800, 2400, 3000, 3900]
    for fi in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            continue
        h0, w0 = fr.shape[:2]
        robots = detect_robots(fr, bg)
        balls = detect_balls(fr)
        ann = fr.copy()

        # Draw eroded ball-carpet boundary (green = allowed zone for balls)
        bc = _ball_carpet_mask(w0, h0)
        cnts_bc, _ = cv2.findContours(bc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ann, cnts_bc, -1, (0, 200, 0), 1)

        # Draw detected balls (cyan circle)
        for b in balls:
            cv2.circle(ann, b["centroid"], b["r"] + 1, (0, 255, 255), 1)
            cv2.circle(ann, b["centroid"], 2, (0, 255, 255), -1)

        # Draw detected robots
        for r in robots:
            x, y, bw, bh = r["bbox"]
            col = (255, 0, 0) if r["alliance"] == "blue" else (0, 0, 255)
            cv2.rectangle(ann, (x, y), (x + bw, y + bh), col, 2)

        nb = sum(1 for r in robots if r["alliance"] == "blue")
        nr = sum(1 for r in robots if r["alliance"] == "red")
        cv2.putText(ann, f"R b:{nb} r:{nr}  FUEL:{len(balls)}", (6, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        big = cv2.resize(ann, None, fx=2.0, fy=2.0,
                         interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(out_dir, f"f{fi:05d}.png"), big)
        print(f"f{fi}: robots b{nb}/r{nr}, balls {len(balls)}")
    cap.release()
    print("annotated frames ->", out_dir)
