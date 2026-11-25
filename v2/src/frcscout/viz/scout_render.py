"""Scouting-demo renderer — draw PHASES + scouting METRICS on the broadcast frame.

This is the human-facing centerpiece of the demo video. Where ``Engine._draw`` shows
the raw tracking layer (up to 32 churning ByteTrack ids), this renderer shows the
*refined* scouting story: the SIX stable robots (the structural prior), each labelled
with its stable id, resolved team number, current PHASE and held-ball count, plus a
right-hand panel of the per-robot scouting metrics the user asked for.

Design contract (mirrors the rest of the refine/viz layer)
----------------------------------------------------------
* **Headless-safe.** ``draw_scout_overlay`` returns the annotated ndarray and NEVER
  calls ``cv2.imshow`` — the caller (or the engine's own display path) decides whether
  to show it. So this runs unchanged on a headless CI box / inside tests.
* **Guarded heavy deps.** ``cv2`` and ``numpy`` are imported lazily *inside* the draw
  call, so the module imports on a bare interpreter; only the actual draw needs them
  (the tests ``importorskip`` them).
* **Colour strictly by alliance.** red / blue / gray-when-unknown — never by track id
  (the whole point of the 6-robot prior is that identity is stable, so we colour by the
  thing that matters to a scout: which alliance).
* **NEVER fabricate a metric.** A metric carrying ``available: False`` (the
  ``{value, available, needs}`` envelope from PhaseTracker) is drawn as ``"—"`` / the
  ``na`` token, NOT as ``0`` — a missing field pose must read as *unknown*, not *zero*.

Inputs
------
``draw_scout_overlay(frame, st, six, phase_out, metrics, fps=None)``:
  * ``frame``     a BGR ndarray (H, W, 3) — the decoded broadcast frame.
  * ``st``        the :class:`~frcscout.io.types.FrameState` (overlay, balls, events).
  * ``six``       the SixRobotTracker output for this frame: a list of dicts
                  ``{id, alliance, xyxy, centroid, est, hits, age}`` (<=6). This is the
                  authority for WHERE the boxes are drawn.
  * ``phase_out`` PhaseTracker.update() output: ``{robot_id: {phase, held, label,
                  climb_level, alliance, metrics}}`` — the per-robot PHASE this frame.
  * ``metrics``   per-robot METRICS dict ``{robot_id: metrics}`` (PhaseTracker.metrics())
                  used for the right-side panel. May be the same dict reused each frame.
  * ``fps``       optional float for the HUD fps readout.

Returns the annotated frame (a copy; the input is not mutated).
"""
from __future__ import annotations

from typing import Optional

# ── Heavy deps are guarded so the module PARSES/imports on a bare interpreter. The
#    actual draw needs them; ``draw_scout_overlay`` raises a clear error if they are
#    absent (the tests importorskip them).
try:  # pragma: no cover - exercised both with and without the deps
    import cv2 as _cv2  # type: ignore
except Exception:  # pragma: no cover
    _cv2 = None  # type: ignore

try:  # pragma: no cover
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None  # type: ignore


# ── BGR palette (kept consistent with Engine's conventions). ──────────────────
_RED_BGR = (0, 0, 220)
_BLUE_BGR = (220, 80, 0)
_YELLOW_BGR = (0, 220, 220)
_WHITE_BGR = (255, 255, 255)
_GRAY_BGR = (120, 120, 120)
_GREEN_BGR = (0, 200, 0)
_PANEL_BG = (24, 24, 24)
_PANEL_DIM = (170, 170, 170)
_EST_BGR = (90, 90, 90)        # coasting (est=True) box hint
_BALL_BGR = (255, 0, 255)      # bright magenta ball outline (pops on yellow balls + carpet)

_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX == 0; avoid touching cv2 at import time.

# Tokens drawn for an UNAVAILABLE metric — never "0" (a missing pose is unknown,
# not zero).
_NA = "--"
_PANEL_W = 290                 # right-side metrics-panel width in px


def _alliance_color(alliance: Optional[str]) -> tuple:
    """Colour STRICTLY by alliance: red / blue / gray-when-unknown."""
    if alliance == "red":
        return _RED_BGR
    if alliance == "blue":
        return _BLUE_BGR
    return _GRAY_BGR


def _prefix(robot_id, alliance: Optional[str]) -> str:
    """'R3' / 'B1' / '#7' style id prefix (matches PhaseTracker._prefix)."""
    letter = "R" if alliance == "red" else ("B" if alliance == "blue" else "#")
    return f"{letter}{robot_id}"


def _team_str(rec: Optional[dict]) -> str:
    """'#254' if a team number was resolved onto the phase/six record, else ''."""
    if not rec:
        return ""
    team = rec.get("team")
    return f" #{team}" if team is not None else ""


def _unwrap(metric):
    """Unwrap a ``{value, available, ...}`` envelope -> (value, available). A plain
    scalar is treated as available. Mirrors export._val + the available flag."""
    if isinstance(metric, dict) and "value" in metric:
        return metric.get("value"), bool(metric.get("available", metric.get("value") is not None))
    return metric, metric is not None


def _fmt_metric(metric, ndigits: int = 2) -> str:
    """Render a (possibly enveloped) metric for the panel. Unavailable / None -> the
    NA token (NEVER 0). Numbers are rounded; everything else stringified."""
    value, available = _unwrap(metric)
    if not available or value is None:
        return _NA
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{ndigits}f}"
    return str(value)


def _phase_label(robot_id, rec: Optional[dict], six_alliance: Optional[str]) -> str:
    """The on-box label: stable id + team (if known) + phase label + held count.

    Prefers PhaseTracker's own ``label`` (already 'B1 HOLDING(2)' / 'R5 CLIMB L2'
    style) and splices the team number in after the id token; falls back to a plain
    id+phase when no phase record exists for this robot yet."""
    alliance = (rec.get("alliance") if rec else None) or six_alliance
    prefix = _prefix(robot_id, alliance)
    team = _team_str(rec)
    if rec and rec.get("label"):
        label = str(rec["label"])
        # PhaseTracker's label already starts with the id prefix (e.g. 'B1 HOLDING(2)').
        # Splice the team number right after that first token so colour+id+team+phase
        # all read together: 'B1 #1678 HOLDING(2)'.
        head, _, tail = label.partition(" ")
        return f"{head}{team} {tail}".rstrip() if tail else f"{head}{team}"
    phase = (rec.get("phase") if rec else None) or ""
    return f"{prefix}{team} {phase}".rstrip()


def _put(img, text, org, color, scale=0.5, thick=1):
    _cv2.putText(img, text, org, _FONT, scale, color, thick, _cv2.LINE_AA)


# ──────────────────────────────── the entry point ────────────────────────────
def draw_scout_overlay(frame, st, six, phase_out, metrics, fps: Optional[float] = None):
    """Annotate ``frame`` with the scouting overlay and return the annotated copy.

    See the module docstring for the argument contract. Requires cv2 + numpy at call
    time (guarded import); raises ``RuntimeError`` if they are unavailable. The input
    frame is not mutated.
    """
    if _cv2 is None or _np is None:  # pragma: no cover - covered by the importorskip tests
        raise RuntimeError(
            "draw_scout_overlay requires numpy + cv2 at call time "
            "(the module imports without them, but drawing needs them)."
        )

    out = frame.copy()
    h, w = out.shape[:2]
    phase_out = phase_out or {}
    metrics = metrics or {}
    six = six or []

    # ── 1. Ball inventory — a bright thin outline per detected ball, so a DENSE PILE
    #       visibly reads as "detected" (the headline requirement). Avoids the slow,
    #       faint per-ball full-frame alpha-blend the first version used. ─────────────
    for t in getattr(st, "balls_inventory", None) or []:
        cx, cy = (int(v) for v in t.centroid_px)
        r = max(5, (int(t.xyxy[2]) - int(t.xyxy[0])) // 2)
        _cv2.circle(out, (cx, cy), r, _BALL_BGR, 2)

    # ── 2. The SIX robot boxes (alliance-coloured) + phase labels. ───────────────
    for d in six:
        rid = d.get("id")
        alliance = d.get("alliance")
        color = _alliance_color(alliance)
        x1, y1, x2, y2 = (int(v) for v in d["xyxy"])
        # a coasting (est) track is drawn thinner / dimmer so a predicted box is honest.
        est = bool(d.get("est"))
        _cv2.rectangle(out, (x1, y1), (x2, y2), color, 1 if est else 2)
        rec = phase_out.get(rid)
        label = _phase_label(rid, rec, alliance)
        if est:
            label += " ~"          # mark a predicted (coasting) box
        # label background strip for legibility on a busy broadcast frame
        (tw, th), _ = _cv2.getTextSize(label, _FONT, 0.5, 1)
        ly = max(th + 4, y1 - 4)
        _cv2.rectangle(out, (x1, ly - th - 4), (x1 + tw + 6, ly + 2), _PANEL_BG, -1)
        _put(out, label, (x1 + 3, ly - 2), color if not est else _PANEL_DIM, 0.5, 1)

    # ── 3. Flight balls (SHOTS) — a small marker + label, NO long velocity arrow (the
    #       arrows cluttered the whole frame) and CAPPED so a noisy frame can't fill the
    #       image with lines. The fuel-color gate already removes most non-ball motion.
    for t in (list(getattr(st, "balls_flight", None) or []))[:16]:
        cx, cy = (int(v) for v in t.centroid_px)
        # Prominent SHOT marker — the airborne ball is small + far, so the old 9px ring was
        # invisible at video resolution (the user "could never find when they shoot"). A big
        # black-haloed bright ring + center dot + a labelled tag makes a shot unmissable.
        _cv2.circle(out, (cx, cy), 20, (0, 0, 0), 4)        # black halo for contrast
        _cv2.circle(out, (cx, cy), 20, _YELLOW_BGR, 2)      # bright ring
        _cv2.circle(out, (cx, cy), 3, _YELLOW_BGR, -1)      # center dot
        tx, ty = cx + 25, cy - 6
        _cv2.rectangle(out, (tx - 3, ty - 15), (tx + 58, ty + 5), (0, 0, 0), -1)
        _put(out, "SHOT", (tx, ty), _YELLOW_BGR, 0.6, 2)

    # ── 4. Recent events, bottom-left. ───────────────────────────────────────────
    y_off = h - 12
    for ev in (getattr(st, "events", None) or [])[-5:]:
        color = _alliance_color(getattr(ev, "alliance", None))
        etype = getattr(ev, "type", "")
        label = f"{str(etype).upper()}"
        rid = getattr(ev, "robot_id", None)
        if rid is not None:
            label += f" {_prefix(rid, getattr(ev, 'alliance', None))}"
        _put(out, label, (10, y_off), color, 0.5, 1)
        y_off -= 18

    # ── 5. Right-side per-robot METRICS panel. ───────────────────────────────────
    _draw_metrics_panel(out, six, phase_out, metrics, w, h)

    # ── 6. Field MINIMAP (top-down) — robots/balls at REAL field XY when tag-localized. ──
    _draw_minimap(out, st, w, h)

    # ── 7. Top HUD: match label, alliance scores/fuel, fps. ──────────────────────
    _draw_hud(out, st, fps, w)

    return out


def _draw_minimap(out, st, w: int, h: int, field_len: float = 16.54, field_wid: float = 8.07) -> None:
    """Top-down field minimap (bottom-centre): robots (alliance dots + id) and balls plotted
    at their REAL field XY — drawn ONLY when an AprilTag pose resolved this frame, so it
    visualizes geolocation honestly (no map shown = no real pose that frame)."""
    pose = getattr(st, "pose", None)
    if pose is None or not getattr(pose, "valid", False):
        return
    mw, mh = 320, int(320 * field_wid / field_len)
    x0, y0 = w // 2 - mw // 2, h - mh - 16
    panel = out.copy()
    _cv2.rectangle(panel, (x0, y0), (x0 + mw, y0 + mh), (30, 45, 30), -1)
    _cv2.addWeighted(panel, 0.6, out, 0.4, 0, out)
    _cv2.rectangle(out, (x0, y0), (x0 + mw, y0 + mh), _WHITE_BGR, 1)
    _cv2.line(out, (x0 + mw // 2, y0), (x0 + mw // 2, y0 + mh), (90, 90, 90), 1)
    _put(out, "FIELD top-down (m) - tag-localized", (x0 + 4, y0 - 5), _WHITE_BGR, 0.42, 1)

    def to_px(fx, fy):
        return (int(x0 + (fx / field_len) * mw), int(y0 + (fy / field_wid) * mh))

    for t in getattr(st, "balls_inventory", None) or []:
        xy = getattr(t, "field_xy_m", None)
        if xy and 0 <= xy[0] <= field_len and 0 <= xy[1] <= field_wid:
            _cv2.circle(out, to_px(xy[0], xy[1]), 2, _BALL_BGR, -1)
    for rb in getattr(st, "robots", None) or []:
        if getattr(rb, "geo", "") != "tag":
            continue
        xy = getattr(rb, "field_xy_m", None)
        if not xy or not (0 <= xy[0] <= field_len and 0 <= xy[1] <= field_wid):
            continue
        px = to_px(xy[0], xy[1])
        _cv2.circle(out, px, 5, _alliance_color(rb.alliance), -1)
        _cv2.circle(out, px, 5, _WHITE_BGR, 1)
        _put(out, str(rb.tid), (px[0] + 6, px[1] + 3), _WHITE_BGR, 0.4, 1)


# ───────────────────────────────── panels ────────────────────────────────────
def _draw_metrics_panel(out, six, phase_out, metrics, w: int, h: int) -> None:
    """A translucent right-side column: one block per robot with the headline scouting
    metrics. UNAVAILABLE metrics render as the NA token (never 0)."""
    pw = min(_PANEL_W, max(120, w // 4))
    x0 = w - pw
    # translucent backdrop
    panel = out.copy()
    _cv2.rectangle(panel, (x0, 40), (w, h), _PANEL_BG, -1)
    _cv2.addWeighted(panel, 0.55, out, 0.45, 0, out)

    _put(out, "SCOUTING", (x0 + 8, 60), _WHITE_BGR, 0.55, 1)

    # order robots by alliance then id so red/blue group together
    order = sorted(
        six,
        key=lambda d: (0 if d.get("alliance") == "red" else (1 if d.get("alliance") == "blue" else 2),
                       d.get("id", 0)),
    )
    y = 78
    line_h = 14
    for d in order:
        rid = d.get("id")
        alliance = d.get("alliance")
        color = _alliance_color(alliance)
        rec = phase_out.get(rid) or {}
        m = metrics.get(rid) or {}

        team = rec.get("team")
        team_txt = f"#{team}" if team is not None else _prefix(rid, alliance)
        phase = rec.get("phase") or _NA
        held = rec.get("held")
        held_txt = str(held) if held is not None else _NA

        _put(out, f"{_prefix(rid, alliance)} {team_txt}", (x0 + 8, y), color, 0.5, 1)
        y += line_h
        _put(out, f"  phase {phase}", (x0 + 8, y), _PANEL_DIM, 0.42, 1)
        y += line_h
        _put(out, f"  held {held_txt}  acc {_fmt_metric(m.get('shot_accuracy'))}",
             (x0 + 8, y), _PANEL_DIM, 0.42, 1)
        y += line_h
        _put(out,
             f"  maxHeld {_fmt_metric(m.get('max_held'), 0)}  climbL {_fmt_metric(m.get('climb_level'), 0)}",
             (x0 + 8, y), _PANEL_DIM, 0.42, 1)
        y += line_h + 4
        if y > h - line_h:
            break


def _draw_hud(out, st, fps: Optional[float], w: int) -> None:
    """Top bar: match label (center), blue/red score+fuel (left), fps (right)."""
    ov = getattr(st, "overlay", None)

    bar = out[:36, :].copy()
    _cv2.rectangle(bar, (0, 0), (w, 36), (20, 20, 20), -1)
    _cv2.addWeighted(bar, 0.7, out[:36, :], 0.3, 0, out[:36, :])

    if fps is not None:
        _put(out, f"{fps:.1f} fps", (w - 110, 24), _WHITE_BGR, 0.6, 2)

    if ov is not None:
        label = getattr(ov, "match_label", None)
        if label:
            (tw, _th), _ = _cv2.getTextSize(label, _FONT, 0.6, 2)
            _put(out, label, ((w - tw) // 2, 24), _WHITE_BGR, 0.6, 2)

        blue_txt = "B"
        bscore = getattr(ov, "blue_score", None)
        bfuel = getattr(ov, "blue_fuel", None)
        if bscore is not None:
            blue_txt += f":{bscore}"
        if bfuel is not None:
            blue_txt += f"  F{bfuel}"
        _put(out, blue_txt, (8, 24), _BLUE_BGR, 0.6, 2)

        red_txt = "R"
        rscore = getattr(ov, "red_score", None)
        rfuel = getattr(ov, "red_fuel", None)
        if rscore is not None:
            red_txt += f":{rscore}"
        if rfuel is not None:
            red_txt += f"  F{rfuel}"
        _put(out, red_txt, (200, 24), _RED_BGR, 0.6, 2)


__all__ = ["draw_scout_overlay"]
