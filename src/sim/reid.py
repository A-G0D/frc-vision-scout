"""
reid.py -- occlusion / re-identification helpers for RobotTracker.

In the single static wide camera of this FRC REBUILT (2026) playoff match the six
robots (3 blue, 3 red) repeatedly drive behind one another and behind the two
dark-navy HUB boxes. A pure IOU/Kalman tracker drops a track when its detection
disappears for a few frames and then -- when the robot re-emerges -- spawns a NEW
id, causing an ID switch. This module supplies the *appearance + motion* glue a
tracker uses to RE-ATTACH a lost track to the right re-emerging detection, so the
id survives a short occlusion.

A robot's identity at this zoom is captured by a cheap, deterministic signature:

  * alliance         -- 'blue' | 'red'  (HARD gate: a blue track never re-acquires
                        a red detection; bumper color is unambiguous here).
  * bumper-color hue histogram -- normalized Hue histogram of the saturated
                        bumper pixels in the lower band of the box. Distinguishes
                        two robots of the SAME alliance by their exact bumper /
                        team-number-panel coloring + lighting.
  * size             -- (w, h) of the last good bbox; robots keep roughly constant
                        apparent size on the flat carpet.
  * last position    -- centroid + a velocity estimate so we can PREDICT where the
                        robot should re-emerge after N missed frames and gate on it.

`match_lost_track(track, detections, frame=None)` scores every candidate detection
of the SAME alliance against the track's signature and returns the single best
detection (or None if none is credible). Scoring blends:

    gate   : alliance must match; predicted-position distance must be within a
             frames-scaled radius (a robot can't teleport across the field).
    score  : weighted sum of  position closeness  +  size similarity
             +  appearance (histogram) similarity.

The tracker is expected to:
    sig = build_signature(frame, bbox, alliance)         # when a track is healthy
    track = {'id':.., 'alliance':.., 'signature':sig,
             'last_centroid':(cx,cy), 'velocity':(vx,vy), 'lost_frames':k}
    det = match_lost_track(track, detections, frame)     # while a track is lost
and, on a hit, snap the track to `det` and refresh its signature.

Classical CV only (OpenCV + numpy). Deterministic.

Interface (exact):
    match_lost_track(track, detections) -> detection | None
        (an optional `frame` 3rd arg enables appearance matching; without it the
         match falls back to motion+size, which still recovers most gaps.)

Helpers (used by track.py / tests):
    build_signature(frame, bbox, alliance) -> dict
    signature_distance(sig_a, sig_b) -> float in [0,1]
    predict_position(track, frames_ahead=None) -> (cx, cy)
    update_track_signature(track, frame, det, alpha=0.5) -> None  (in place)
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Tunables (calibrated against the real match, see __main__ self-check)
# ---------------------------------------------------------------------------
# Hue histogram of bumper pixels.
_HUE_BINS = 12                     # coarse Hue histogram -> robust to noise
_BUMPER_LOWER_FRAC = 0.30          # bumper sits on the lower ~70% of the box
# Saturation/Value floors to keep only vivid bumper pixels (reject shadow/glare).
_SAT_FLOOR = 80
_VAL_FLOOR = 70

# Re-acquisition gating / scoring.
_BASE_GATE_PX = 26.0               # base re-emergence radius at lost_frames=1
_PER_FRAME_GATE_PX = 9.0           # the gate grows while a track stays lost
_MAX_GATE_PX = 130.0               # but never the whole field
_SIZE_TOL = 0.60                   # |dw|/w etc. beyond this -> size term ~0
# Score weights (sum need not be 1; higher = more influence).
_W_POS = 0.50
_W_SIZE = 0.20
_W_APP = 0.30
# Minimum blended score to accept a re-acquisition at all.
_MIN_ACCEPT = 0.35


# ---------------------------------------------------------------------------
# Signature construction
# ---------------------------------------------------------------------------
def _bumper_hue_hist(frame: np.ndarray, bbox) -> np.ndarray:
    """Normalized Hue histogram of vivid pixels in the bumper band of `bbox`.

    Returns a float32 vector of length _HUE_BINS summing to 1 (or all-zeros if
    the patch has no vivid pixels). Hue is cyclic so this is rotation-tolerant
    to the robot's heading -- two views of the same bumper share a hue profile.
    """
    h_img, w_img = frame.shape[:2]
    x, y, w, h = [int(v) for v in bbox]
    # Clamp bbox to image.
    x0 = max(0, x)
    y0 = max(0, y + int(_BUMPER_LOWER_FRAC * h))
    x1 = min(w_img, x + w)
    y1 = min(h_img, y + h)
    if x1 <= x0 or y1 <= y0:
        return np.zeros(_HUE_BINS, np.float32)
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return np.zeros(_HUE_BINS, np.float32)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]
    val = hsv[..., 2]
    vivid = (sat >= _SAT_FLOOR) & (val >= _VAL_FLOOR)
    if int(vivid.sum()) < 3:
        return np.zeros(_HUE_BINS, np.float32)
    hue = hsv[..., 0][vivid].astype(np.float32)  # 0..179
    hist, _ = np.histogram(hue, bins=_HUE_BINS, range=(0.0, 180.0))
    hist = hist.astype(np.float32)
    s = float(hist.sum())
    if s > 0:
        hist /= s
    return hist


def build_signature(frame: Optional[np.ndarray], bbox, alliance: str) -> dict:
    """Build an appearance+geometry signature for a healthy detection/track.

    `frame` may be None (e.g. test injection); the histogram is then empty and
    matching gracefully degrades to motion+size only.
    """
    x, y, w, h = [int(v) for v in bbox]
    hist = (_bumper_hue_hist(frame, bbox)
            if frame is not None else np.zeros(_HUE_BINS, np.float32))
    return {
        "alliance": alliance,
        "hue_hist": hist,
        "size": (int(w), int(h)),
        "centroid": (float(x + w / 2.0), float(y + h / 2.0)),
    }


# ---------------------------------------------------------------------------
# Distances / similarities
# ---------------------------------------------------------------------------
def _hist_similarity(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Histogram-intersection similarity in [0,1]; None if either is empty."""
    if a is None or b is None:
        return None
    sa, sb = float(a.sum()), float(b.sum())
    if sa <= 0.0 or sb <= 0.0:
        return None
    return float(np.minimum(a, b).sum())  # both normalized -> intersection in [0,1]


def signature_distance(sig_a: dict, sig_b: dict) -> float:
    """A blended appearance+size DISTANCE in [0,1] (0 = identical).

    Alliance mismatch -> 1.0 (maximally far). Used by tests and as a tie-break.
    """
    if sig_a.get("alliance") != sig_b.get("alliance"):
        return 1.0
    sim = _hist_similarity(sig_a.get("hue_hist"), sig_b.get("hue_hist"))
    app_d = (1.0 - sim) if sim is not None else 0.5  # neutral if no appearance
    wa, ha = sig_a.get("size", (1, 1))
    wb, hb = sig_b.get("size", (1, 1))
    sw = abs(wa - wb) / float(max(1, wa, wb))
    sh = abs(ha - hb) / float(max(1, ha, hb))
    size_d = min(1.0, 0.5 * (sw + sh) / _SIZE_TOL)
    return float(0.65 * app_d + 0.35 * size_d)


# ---------------------------------------------------------------------------
# Motion prediction
# ---------------------------------------------------------------------------
def predict_position(track: dict, frames_ahead: Optional[int] = None):
    """Predict where a lost track should re-emerge.

    Uses last_centroid + velocity * (frames since last seen). Falls back to the
    last centroid (or signature centroid) when velocity is unknown.
    """
    cx, cy = track.get("last_centroid") or track.get("signature", {}).get(
        "centroid", (0.0, 0.0))
    vx, vy = track.get("velocity", (0.0, 0.0))
    n = frames_ahead if frames_ahead is not None else int(track.get("lost_frames", 1))
    n = max(1, n)
    return (float(cx) + float(vx) * n, float(cy) + float(vy) * n)


# ---------------------------------------------------------------------------
# Re-acquisition
# ---------------------------------------------------------------------------
def _det_signature(det: dict, frame: Optional[np.ndarray]) -> dict:
    return build_signature(frame, det["bbox"], det["alliance"])


def match_lost_track(track: dict, detections: list,
                     frame: Optional[np.ndarray] = None) -> Optional[dict]:
    """Re-attach a lost `track` to the best matching detection, or return None.

    `track` (dict) must carry at least:
        'alliance'            : 'blue'|'red'
        'signature'           : dict from build_signature (size/hist/centroid)
      and ideally:
        'last_centroid'       : (cx,cy) where it was last seen
        'velocity'            : (vx,vy) px/frame
        'lost_frames'         : how many frames it has been missing (>=1)

    `detections` is a list of detect_robots() dicts:
        {'bbox':(x,y,w,h),'centroid':(cx,cy),'alliance':'blue'|'red'}

    `frame` (optional): the current BGR frame; if given, appearance (bumper hue
    histogram) similarity is used, which is the decisive cue for same-alliance
    robots. Without it, the match uses predicted-position + size only.

    Returns the chosen detection dict (the same object from `detections`), or
    None if no candidate is credible.
    """
    if not detections:
        return None
    alliance = track.get("alliance") or track.get("signature", {}).get("alliance")
    sig = track.get("signature") or {}
    sig_size = sig.get("size", (None, None))
    sig_hist = sig.get("hue_hist")

    lost = max(1, int(track.get("lost_frames", 1)))
    gate = min(_MAX_GATE_PX, _BASE_GATE_PX + _PER_FRAME_GATE_PX * lost)
    px, py = predict_position(track)

    best = None
    best_score = -1.0
    for det in detections:
        # HARD alliance gate -- bumper color is unambiguous in this match.
        if alliance is not None and det.get("alliance") != alliance:
            continue
        cx, cy = det["centroid"]
        dist = float(np.hypot(cx - px, cy - py))
        if dist > gate:
            continue  # can't have teleported there

        # Position term: 1 at the prediction, ->0 at the gate edge.
        pos_term = max(0.0, 1.0 - dist / gate)

        # Size term.
        bw, bh = det["bbox"][2], det["bbox"][3]
        if sig_size[0]:
            sw = abs(bw - sig_size[0]) / float(max(1, bw, sig_size[0]))
            sh = abs(bh - sig_size[1]) / float(max(1, bh, sig_size[1]))
            size_term = max(0.0, 1.0 - 0.5 * (sw + sh) / _SIZE_TOL)
        else:
            size_term = 0.5  # unknown -> neutral

        # Appearance term (only if we have both a frame and a stored histogram).
        if frame is not None and sig_hist is not None and float(np.sum(sig_hist)) > 0:
            det_hist = _bumper_hue_hist(frame, det["bbox"])
            sim = _hist_similarity(sig_hist, det_hist)
            if sim is None:
                app_term = 0.5
                w_app = _W_APP * 0.4  # weak when det has no vivid bumper px
            else:
                app_term = sim
                w_app = _W_APP
        else:
            app_term = 0.5
            w_app = 0.0  # no appearance info -> redistribute to pos/size

        w_pos, w_size = _W_POS, _W_SIZE
        wsum = w_pos + w_size + w_app
        score = (w_pos * pos_term + w_size * size_term + w_app * app_term) / wsum

        if score > best_score:
            best_score = score
            best = det

    if best is None or best_score < _MIN_ACCEPT:
        return None
    return best


def update_track_signature(track: dict, frame: Optional[np.ndarray],
                           det: dict, alpha: float = 0.5) -> None:
    """Refresh a track's signature/motion in place after a confirmed match.

    The hue histogram is EMA-blended (alpha = weight of the new observation) so
    the identity adapts slowly to lighting while resisting single-frame noise.
    Size snaps to the new bbox; velocity is updated from the centroid delta.
    """
    new_sig = build_signature(frame, det["bbox"], det["alliance"])
    old = track.get("signature") or {}
    old_hist = old.get("hue_hist")
    new_hist = new_sig["hue_hist"]
    if (old_hist is not None and float(np.sum(old_hist)) > 0
            and float(np.sum(new_hist)) > 0):
        blended = (1.0 - alpha) * old_hist + alpha * new_hist
        s = float(blended.sum())
        if s > 0:
            blended = (blended / s).astype(np.float32)
        new_sig["hue_hist"] = blended
    elif float(np.sum(new_hist)) == 0 and old_hist is not None:
        new_sig["hue_hist"] = old_hist  # keep the last good appearance

    prev_c = track.get("last_centroid")
    cx, cy = det["centroid"]
    if prev_c is not None:
        lost = max(1, int(track.get("lost_frames", 1)))
        track["velocity"] = ((cx - prev_c[0]) / lost, (cy - prev_c[1]) / lost)
    track["signature"] = new_sig
    track["last_centroid"] = (float(cx), float(cy))
    track["lost_frames"] = 0


# ---------------------------------------------------------------------------
# CLI self-check: inject synthetic gaps on the REAL match and confirm recovery
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys

    # Import sibling detect/field whether run as module or script.
    try:
        from . import detect as _detect  # type: ignore
    except Exception:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import detect as _detect  # type: ignore

    video = sys.argv[1] if len(sys.argv) > 1 else "data/sim/match_full.mp4"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "reid_annot"
    os.makedirs(out_dir, exist_ok=True)

    print("Building background (wide segment 0..139s)...")
    bg = _detect.build_background(video, 0.0, 139.0)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # --- Experiment 1: signature separability ---------------------------------
    # On a frame with several robots, confirm same-id (consecutive frames) is
    # closer than cross-robot, and that same-alliance robots are separable.
    def read(fi):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        return fr if ok else None

    test_fi = 1500
    fr = read(test_fi)
    dets = _detect.detect_robots(fr, bg)
    sigs = [build_signature(fr, d["bbox"], d["alliance"]) for d in dets]
    print(f"\n[frame {test_fi}] {len(dets)} robot dets")
    for i, d in enumerate(dets):
        print(f"  det{i} {d['alliance']:>4} bbox={d['bbox']} "
              f"hist_sum={float(np.sum(sigs[i]['hue_hist'])):.2f}")

    # --- Experiment 2: re-acquisition across an injected gap ------------------
    # Simulate a track that was healthy at frame F, vanished for GAP frames,
    # then we try to re-attach it at frame F+GAP. Count correct recoveries.
    GAP = 8
    starts = list(range(900, 3600, 120))
    recovered = 0
    attempts = 0
    id_switch = 0
    detail = []
    prev = None
    for F in starts:
        f_now = read(F)
        f_then = read(F + GAP)
        if f_now is None or f_then is None:
            continue
        d_now = _detect.detect_robots(f_now, bg)
        d_then = _detect.detect_robots(f_then, bg)
        if len(d_now) < 2 or len(d_then) < 1:
            continue
        # Build a track from each robot at F, blank it, re-acquire at F+GAP.
        for d in d_now:
            sig = build_signature(f_now, d["bbox"], d["alliance"])
            track = {
                "id": 0, "alliance": d["alliance"], "signature": sig,
                "last_centroid": d["centroid"], "velocity": (0.0, 0.0),
                "lost_frames": GAP,
            }
            # ground-truth: nearest same-alliance det at F+GAP within reason
            gts = [(np.hypot(d["centroid"][0] - e["centroid"][0],
                             d["centroid"][1] - e["centroid"][1]), e)
                   for e in d_then if e["alliance"] == d["alliance"]]
            if not gts:
                continue
            gts.sort(key=lambda t: t[0])
            gt_dist, gt = gts[0]
            if gt_dist > 90:   # the robot likely left / merged; skip as unknowable
                continue
            attempts += 1
            chosen = match_lost_track(track, d_then, f_then)
            if chosen is None:
                detail.append((F, d["alliance"], "MISS(None)", gt_dist))
                continue
            ok = (abs(chosen["centroid"][0] - gt["centroid"][0]) < 1e-6 and
                  abs(chosen["centroid"][1] - gt["centroid"][1]) < 1e-6)
            if ok:
                recovered += 1
            else:
                id_switch += 1
                detail.append((F, d["alliance"], "SWITCH",
                               (gt_dist,
                                np.hypot(chosen["centroid"][0] - gt["centroid"][0],
                                         chosen["centroid"][1] - gt["centroid"][1]))))

    print(f"\n[re-acquire, GAP={GAP}f] attempts={attempts} "
          f"recovered={recovered} switches={id_switch} "
          f"none={attempts - recovered - id_switch}")
    if attempts:
        print(f"  recovery rate = {100.0 * recovered / attempts:.1f}%")
    for row in detail[:15]:
        print("   ", row)

    # --- Annotate one re-acquisition for visual proof -------------------------
    F = 1500
    f_now = read(F)
    f_then = read(F + GAP)
    if f_now is not None and f_then is not None:
        d_now = _detect.detect_robots(f_now, bg)
        d_then = _detect.detect_robots(f_then, bg)
        ann = f_then.copy()
        for d in d_then:
            x, y, w, h = d["bbox"]
            col = (255, 0, 0) if d["alliance"] == "blue" else (0, 0, 255)
            cv2.rectangle(ann, (x, y), (x + w, y + h), col, 1)
        for d in d_now:
            sig = build_signature(f_now, d["bbox"], d["alliance"])
            track = {"id": 0, "alliance": d["alliance"], "signature": sig,
                     "last_centroid": d["centroid"], "velocity": (0.0, 0.0),
                     "lost_frames": GAP}
            ch = match_lost_track(track, d_then, f_then)
            ox, oy = int(d["centroid"][0]), int(d["centroid"][1])
            cv2.circle(ann, (ox, oy), 3, (0, 255, 255), -1)  # where it was
            if ch is not None:
                nx, ny = int(ch["centroid"][0]), int(ch["centroid"][1])
                cv2.arrowedLine(ann, (ox, oy), (nx, ny), (0, 255, 0), 1,
                                tipLength=0.3)
        big = cv2.resize(ann, None, fx=2.0, fy=2.0,
                         interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(out_dir, f"reacq_F{F}_gap{GAP}.png"), big)
        print("annotated ->", os.path.join(out_dir, f"reacq_F{F}_gap{GAP}.png"))

    cap.release()
