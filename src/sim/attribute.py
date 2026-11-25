"""
attribute.py — Score attribution via overlay cross-check for the FRC REBUILT
(2026) playoff match (Upper Bracket R4 M11).

Problem statement
-----------------
The overlay FUEL/score counter for each alliance increments by k at time t when
balls enter a HUB and register as score.  We know the EXACT ground-truth
increments (from overlay_ts.py).  The goal is to determine WHICH robot(s)
contributed those points.

Approach
--------
1.  Build a sorted list of OVERLAY SCORE EVENTS (one per alliance per 1-second
    tick where the score increases) from overlay_ts, handling OCR jitter and
    monotone clamping.

2.  For each overlay score event (alliance, t, delta_pts):
      a. Look back in the events list for SHOT events from that alliance whose
         ball reached a HUB in the window [t - LOOKBACK_S, t].  A SHOT event
         is any event of type 'shot' (or 'score') with matching alliance.
      b. Among the candidate shot events that pass the time-gate, identify the
         robot_id(s) and split the delta_pts proportionally (equally if multiple
         concurrent shots at the same HUB; or entirely to the single shot).
      c. Emit a NEW event of type 'score' carrying robot_id, alliance, and the
         attributed points count.  If no qualifying shot event is found, emit a
         'score' event with robot_id=None (alliance-pool credit).

3.  Return the augmented events list (original events PLUS the new 'score'
    attribution events).

Key design choices
------------------
* MONOTONE clamping: the OCR sometimes reads garbage (e.g. red_fuel=481 at t=6
  when real value is ~181).  We smooth with a monotone clamp: if a new reading
  drops >MAX_DROP below the running max, we clamp it to the running max.  This
  recovers a clean +k series.
* Per-second 1-s window: each sample is exactly 1 s / 30 frames apart, so
  consecutive increments are independent events.
* LOOKBACK_S (default 3 s): a shot ball travels ~0.5-2 s of air time then the
  scorer registers.  3 s is generous and safe.
* SPLIT POLICY: if n shots from the same alliance qualify for a score window,
  the pts are split equally (rounded to int, remainder to first robot).
* Attribution fraction: also reported as 'attributed_fraction' — the fraction
  of each alliance's total score that was traced to a named robot vs the pool.

Interface (exact):
    attribute_scores(
        events:       list[dict],    # from events.py detect_events()
        overlay_ts:   list[dict],    # from overlay_ts.py read_timeseries()
        hub_regions:  list[dict],    # from field.py hub_regions()
    ) -> list[dict]

Each returned event is either an original event (unchanged) or a NEW event dict:
    {
        'type':     'score',
        'frame':    int,
        't':        float,           # time in seconds
        'alliance': 'blue'|'red',
        'robot_id': int|None,        # None = attributed to alliance pool
        'pts':      int,             # points credited to this robot for this event
        'delta':    int,             # total overlay delta for this tick
        'overlay_t': float,          # the overlay sample time that triggered this
        'detail':   str,
    }

Reporting helpers (available as module-level functions):
    summarise_attribution(attributed_events) -> dict
        Returns per-robot points, total per-alliance, and coverage fractions.
    print_attribution_report(attributed_events, overlay_ts)
        Human-readable summary to stdout.
"""

from __future__ import annotations

import math
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# How many seconds before the overlay increment to search for qualifying shot
# events.  Balls travel 0.5-2 s; give 3 s to allow for tracker latency.
_LOOKBACK_S: float = 3.0

# How far into the FUTURE to look (the overlay may lag one tick = 1 s).
_LOOKAHEAD_S: float = 0.5

# Maximum drop in score vs. running max before we clamp (OCR glitch guard).
_MAX_DROP: int = 20

# Minimum score delta to bother attributing (noise floor for OCR jitter).
_MIN_DELTA: int = 1

# Minimum distance from a shot event to the nearest HUB to consider it a
# "scored" shot (pixels at 640x360). If hub_regions is provided and a shot
# has a 'pos' or 'centroid', we can tighten the gate. Otherwise unused.
_HUB_DIST_PX: float = 80.0


# ---------------------------------------------------------------------------
# Internal: OCR-robust monotone score series
# ---------------------------------------------------------------------------
def _clean_score_series(overlay_ts: list[dict], alliance: str) -> list[dict]:
    """Extract a monotone-clamped score series for one alliance.

    Each entry: {'t': float, 'frame': int, 'score': int, 'read_ok': bool}.

    OCR glitches (large drops) are clamped to the running maximum so that
    the computed deltas are never spuriously negative.
    """
    key = f"{alliance}_score"
    running_max = 0
    out = []
    for row in overlay_ts:
        if not row.get("read_ok", False):
            out.append({"t": row["t"], "frame": row["frame"],
                        "score": running_max, "read_ok": False})
            continue
        raw = int(row.get(key, 0) or 0)
        # If the raw reading drops more than MAX_DROP below the running max,
        # it is almost certainly an OCR error — clamp upward.
        if raw < running_max - _MAX_DROP:
            raw = running_max
        running_max = max(running_max, raw)
        out.append({"t": row["t"], "frame": row["frame"],
                    "score": raw, "read_ok": row.get("read_ok", False)})
    return out


def _extract_score_increments(overlay_ts: list[dict], alliance: str) -> list[dict]:
    """Return list of {t, frame, delta, new_score} where alliance score rose.

    Only frames where live_bar=True and not results_screen are used as valid
    delta anchors.  Results-screen frames are skipped entirely.
    """
    series = _clean_score_series(overlay_ts, alliance)
    # Build the raw ts list with live_bar/results_screen flags.
    flags = [
        {"live_bar": r.get("live_bar", False),
         "results_screen": r.get("results_screen", False)}
        for r in overlay_ts
    ]
    increments = []
    prev_score = None
    for i, entry in enumerate(series):
        is_live = flags[i]["live_bar"] and not flags[i]["results_screen"]
        if not is_live:
            prev_score = None   # reset so we don't carry stale state across cuts
            continue
        if not entry["read_ok"]:
            continue
        score = entry["score"]
        if prev_score is not None and score > prev_score:
            delta = score - prev_score
            if delta >= _MIN_DELTA:
                increments.append({
                    "t": entry["t"],
                    "frame": entry["frame"],
                    "delta": delta,
                    "new_score": score,
                    "alliance": alliance,
                })
        prev_score = score
    return increments


# ---------------------------------------------------------------------------
# Internal: pick qualifying shot events for a score increment window
# ---------------------------------------------------------------------------
def _shots_in_window(events: list[dict], alliance: str,
                     t_score: float) -> list[dict]:
    """Return SHOT events (type='shot') from `alliance` in the look-back window.

    Window: [t_score - LOOKBACK_S, t_score + LOOKAHEAD_S]

    Only 'shot' events are used — 'pickup' events flood the window and
    'score' events (pre-existing from events.py) are attribution anchors,
    not shot observations.
    """
    t_lo = t_score - _LOOKBACK_S
    t_hi = t_score + _LOOKAHEAD_S
    out = []
    for ev in events:
        if ev.get("type") != "shot":
            continue
        if ev.get("alliance") != alliance:
            continue
        ev_t = float(ev.get("t", -999))
        if ev_t < t_lo or ev_t > t_hi:
            continue
        out.append(ev)
    return out


def _dist_to_hub(pos, hub_regions: list[dict]) -> float:
    """Minimum pixel distance from pos=(cx,cy) to the nearest HUB centre."""
    if not hub_regions or pos is None:
        return 0.0
    cx, cy = pos
    best = 1e9
    for hub in hub_regions:
        hx, hy, hw, hh = hub["bbox"]
        hcx = hx + hw / 2.0
        hcy = hy + hh / 2.0
        best = min(best, math.hypot(cx - hcx, cy - hcy))
    return best


def _alliance_hub(alliance: str, hub_regions: list[dict]) -> Optional[dict]:
    """Return the HUB region for the given alliance, or None."""
    for hub in hub_regions:
        if hub.get("alliance") == alliance:
            return hub
    return None


# ---------------------------------------------------------------------------
# Core attribution logic
# ---------------------------------------------------------------------------
def attribute_scores(
    events: list[dict],
    overlay_ts: list[dict],
    hub_regions: list[dict],
) -> list[dict]:
    """Attribute per-robot points using the overlay score cross-check.

    Parameters
    ----------
    events       : output of events.py detect_events() — list of dicts with
                   keys 'type', 't', 'frame', 'alliance', 'robot_id', 'detail'.
    overlay_ts   : output of overlay_ts.py read_timeseries() — list of per-
                   second readings with blue_score, red_score, etc.
    hub_regions  : output of field.py hub_regions() — list of
                   {'bbox':(x,y,w,h), 'alliance':'blue'|'red'}.

    Returns
    -------
    list[dict]: the original events list PLUS new 'score' attribution events.
    Each attribution event:
        {
          'type':      'score',
          'frame':     int,
          't':         float,
          'alliance':  'blue'|'red',
          'robot_id':  int|None,   (None = alliance-pool)
          'pts':       int,
          'delta':     int,        (total overlay delta for this tick)
          'overlay_t': float,
          'detail':    str,
        }
    """
    attributed: list[dict] = list(events)  # keep originals

    for alliance in ("blue", "red"):
        increments = _extract_score_increments(overlay_ts, alliance)

        for inc in increments:
            t_score = inc["t"]
            delta = inc["delta"]
            frame = inc["frame"]

            # Find qualifying shot events from this alliance's robots near
            # the score time.
            candidates = _shots_in_window(events, alliance, t_score)

            # Keep only events with a robot_id (can't attribute to pool here).
            named = [ev for ev in candidates if ev.get("robot_id") is not None]

            # De-duplicate by robot_id within the window: one robot may have
            # multiple shot events but should only be credited once per score tick.
            seen_ids: dict[int, dict] = {}
            for ev in named:
                rid = ev["robot_id"]
                if rid not in seen_ids:
                    seen_ids[rid] = ev
                else:
                    # Keep the event closest in time to t_score.
                    if abs(ev["t"] - t_score) < abs(seen_ids[rid]["t"] - t_score):
                        seen_ids[rid] = ev

            unique_robots = list(seen_ids.values())

            if unique_robots:
                # Split delta equally among the qualifying robots.
                n = len(unique_robots)
                base = delta // n
                remainder = delta - base * n
                unique_robots.sort(key=lambda ev: ev["robot_id"])
                for k, ev in enumerate(unique_robots):
                    pts = base + (1 if k < remainder else 0)
                    if pts <= 0:
                        continue
                    attributed.append({
                        "type": "score",
                        "frame": frame,
                        "t": t_score,
                        "alliance": alliance,
                        "robot_id": ev["robot_id"],
                        "pts": pts,
                        "delta": delta,
                        "overlay_t": t_score,
                        "detail": (
                            f"overlay +{delta}pts attributed to robot "
                            f"{ev['robot_id']} ({n} qualifying shot(s) in "
                            f"window [{t_score - _LOOKBACK_S:.1f}s,"
                            f"{t_score + _LOOKAHEAD_S:.1f}s])"
                        ),
                    })
            else:
                # No named shot found — credit the alliance pool.
                attributed.append({
                    "type": "score",
                    "frame": frame,
                    "t": t_score,
                    "alliance": alliance,
                    "robot_id": None,
                    "pts": delta,
                    "delta": delta,
                    "overlay_t": t_score,
                    "detail": (
                        f"overlay +{delta}pts unattributed (no qualifying "
                        f"shot event in [{t_score - _LOOKBACK_S:.1f}s,"
                        f"{t_score + _LOOKAHEAD_S:.1f}s])"
                    ),
                })

    # Sort by time then frame for readability.
    attributed.sort(key=lambda ev: (float(ev.get("t", 0)), int(ev.get("frame", 0))))
    return attributed


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def summarise_attribution(attributed_events: list[dict]) -> dict:
    """Aggregate per-robot points and coverage from the attributed events.

    Only events emitted by attribute_scores() are counted — identified by the
    presence of the 'pts' key (original events from events.py use 'detail'
    with a fuel_delta sub-field and have no 'pts' key).

    Returns
    -------
    {
      'per_robot': {robot_id: {'alliance':str, 'pts':int}},
      'alliance_pool': {'blue':int, 'red':int},
      'total_attributed': {'blue':int, 'red':int},
      'total_overlay': {'blue':int, 'red':int},
      'coverage_pct': {'blue':float, 'red':float},
    }
    """
    # Only attribution events have a 'pts' key.
    score_events = [
        ev for ev in attributed_events
        if ev.get("type") == "score" and "pts" in ev
    ]

    per_robot: dict[int, dict] = {}
    pool: dict[str, int] = {"blue": 0, "red": 0}
    total: dict[str, int] = {"blue": 0, "red": 0}
    overlay_total: dict[str, int] = {"blue": 0, "red": 0}

    # Sum overlay deltas per alliance — deduplicate by (alliance, overlay_t)
    # so we don't double-count a tick that was split across N robots.
    seen_ticks: dict[tuple, bool] = {}
    for ev in score_events:
        al = ev.get("alliance", "blue")
        t_key = (al, float(ev.get("overlay_t", ev.get("t", 0))))
        if t_key not in seen_ticks:
            seen_ticks[t_key] = True
            overlay_total[al] = overlay_total.get(al, 0) + int(ev.get("delta", 0))

    for ev in score_events:
        al = ev.get("alliance", "blue")
        pts = int(ev.get("pts", 0))
        rid = ev.get("robot_id")
        total[al] = total.get(al, 0) + pts
        if rid is None:
            pool[al] = pool.get(al, 0) + pts
        else:
            if rid not in per_robot:
                per_robot[rid] = {"alliance": al, "pts": 0}
            per_robot[rid]["pts"] += pts

    coverage: dict[str, float] = {}
    for al in ("blue", "red"):
        ov = overlay_total.get(al, 0)
        named = total.get(al, 0) - pool.get(al, 0)
        coverage[al] = (named / ov * 100.0) if ov > 0 else 0.0

    return {
        "per_robot": per_robot,
        "alliance_pool": pool,
        "total_attributed": total,
        "total_overlay": overlay_total,
        "coverage_pct": coverage,
    }


def print_attribution_report(
    attributed_events: list[dict],
    overlay_ts: list[dict],
) -> None:
    """Print a human-readable attribution summary to stdout."""
    summary = summarise_attribution(attributed_events)

    # Overlay total from the raw timeseries (monotone-clamped) for reference.
    ts_blue = _extract_score_increments(overlay_ts, "blue")
    ts_red = _extract_score_increments(overlay_ts, "red")
    ov_blue = sum(e["delta"] for e in ts_blue)
    ov_red = sum(e["delta"] for e in ts_red)

    print("=" * 60)
    print("SCORE ATTRIBUTION REPORT")
    print("=" * 60)
    for al in ("blue", "red"):
        ov_pts = ov_blue if al == "blue" else ov_red
        print(f"\n  {al.upper()} alliance — overlay total: {ov_pts} pts")
        for rid, info in sorted(summary["per_robot"].items()):
            if info["alliance"] != al:
                continue
            frac = info["pts"] / ov_pts * 100 if ov_pts else 0
            print(f"    Robot {rid:3d}: {info['pts']:4d} pts  ({frac:5.1f}% of overlay)")
        pool_pts = summary["alliance_pool"].get(al, 0)
        if pool_pts:
            pool_frac = pool_pts / ov_pts * 100 if ov_pts else 0
            print(f"    Pool (unattributed): {pool_pts:4d} pts  ({pool_frac:5.1f}%)")
        cov = summary["coverage_pct"].get(al, 0)
        print(f"    Attribution coverage: {cov:.1f}% of overlay pts traced to a robot")
    print()
    score_evs = [ev for ev in attributed_events if ev.get("type") == "score"]
    print(f"  Total score-attribution events emitted: {len(score_evs)}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI self-test: run against saved overlay_ts.json + synthetic events
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import os
    import sys

    # Load overlay_ts from the canonical location.
    ts_path = (sys.argv[1] if len(sys.argv) > 1
               else "C:/Users/agodavarthy/Claude/projects/frc-vision-scout"
                    "/data/sim/overlay_ts.json")

    with open(ts_path) as f:
        ts_data = json.load(f)
    overlay_ts_list = ts_data["timeseries"]

    # Build synthetic events: inject plausible shot events near known score ticks
    # so we can verify the attribution logic end-to-end on real timing data.
    # (In production, events come from events.py; here we fabricate representative ones.)
    fps = 30.0

    # Red robots (IDs 0, 1, 2) shoot aggressively early (t=1..27s).
    # Blue robots (IDs 3, 4, 5) shoot from t=25s onward.
    def fake_shots(alliance, robot_ids, score_ticks):
        """For each overlay tick, create a shot event 1.5s BEFORE the score."""
        evs = []
        for tick in score_ticks:
            t_shot = tick["t"] - 1.5
            if t_shot < 0:
                t_shot = tick["t"] - 0.5
            rid = robot_ids[hash(tick["t"]) % len(robot_ids)]
            evs.append({
                "type": "shot",
                "t": round(t_shot, 1),
                "frame": int(round(t_shot * fps)),
                "alliance": alliance,
                "robot_id": rid,
                "detail": f"synthetic shot robot {rid}",
            })
        return evs

    red_ticks = _extract_score_increments(overlay_ts_list, "red")
    blue_ticks = _extract_score_increments(overlay_ts_list, "blue")

    synthetic_events = []
    synthetic_events += fake_shots("red", [0, 1, 2], red_ticks)
    synthetic_events += fake_shots("blue", [3, 4, 5], blue_ticks)

    # Hub regions at 640x360 (from field.py values).
    hub_regions_px = [
        {"bbox": (int(0.183 * 640), int(0.343 * 360),
                  int(0.132 * 640), int(0.128 * 360)), "alliance": "blue"},
        {"bbox": (int(0.658 * 640), int(0.298 * 360),
                  int(0.150 * 640), int(0.138 * 360)), "alliance": "red"},
    ]

    attributed = attribute_scores(synthetic_events, overlay_ts_list, hub_regions_px)
    print_attribution_report(attributed, overlay_ts_list)

    # Validate: every overlay delta should appear in attribution.
    summary = summarise_attribution(attributed)
    ov_blue = sum(e["delta"] for e in blue_ticks)
    ov_red = sum(e["delta"] for e in red_ticks)
    attr_blue = summary["total_attributed"].get("blue", 0)
    attr_red = summary["total_attributed"].get("red", 0)
    print(f"\nOverlay totals — blue: {ov_blue}  red: {ov_red}")
    print(f"Attributed totals — blue: {attr_blue}  red: {attr_red}")
    assert attr_blue == ov_blue, f"Blue mismatch: {attr_blue} vs {ov_blue}"
    assert attr_red == ov_red, f"Red mismatch: {attr_red} vs {ov_red}"
    print("[PASS] Attributed totals match overlay totals exactly.")

    # Per-robot breakdown.
    print("\nPer-robot breakdown:")
    for rid, info in sorted(summary["per_robot"].items()):
        print(f"  robot {rid} ({info['alliance']}): {info['pts']} pts")
    print(f"Blue pool (unattributed): {summary['alliance_pool'].get('blue', 0)}")
    print(f"Red pool (unattributed):  {summary['alliance_pool'].get('red', 0)}")
    print(f"\nBlue attribution coverage: {summary['coverage_pct']['blue']:.1f}%")
    print(f"Red attribution coverage:  {summary['coverage_pct']['red']:.1f}%")
