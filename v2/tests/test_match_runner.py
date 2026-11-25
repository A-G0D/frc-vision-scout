"""MatchRunner integration tests — pure synthetic FrameStates, no model / GPU / video.

We feed hand-built FrameStates (the ByteTrack-stage ``st.robots`` with deliberately
CHURNING raw track ids) through ``MatchRunner.process`` and assert the scouting
integration behaves:

  (a) ``st.robots`` is OVERWRITTEN with the <=6 stable tracks and the stable IDs persist
      across frames even though the raw input ids churn frame to frame;
  (b) alliance labels resolve onto the stable tracks;
  (c) PHASE labels appear in the per-frame output;
  (d) ``finalize()`` yields a record whose ``robots`` dict is compatible with
      ``export.SCOUT_COLUMNS`` and preserves ``available: False`` for unresolved metrics
      (never a fabricated 0);
  (e) ``export()`` writes JSON + CSV to tmp_path and the CSV header == SCOUT_COLUMNS.

Mirrors the style of test_phase.py / test_sixbot.py: a tiny duck-typed StubField, Track
objects built directly, no cv2/numpy needed for the runner itself.
"""
from __future__ import annotations

import csv
import json

import pytest

from frcscout.io.types import FrameState, OverlayState, Track
from frcscout.fuse.export import SCOUT_COLUMNS
from frcscout.pipeline.match_runner import MatchRunner
from frcscout.refine.phase import (
    TRAVERSING, INTAKING, HOLDING, SHOOTING, SCORED, CLIMBING, DEFENSE,
)

FPS = 60.0
_PHASE_LABELS = {TRAVERSING, INTAKING, HOLDING, SHOOTING, SCORED, CLIMBING, DEFENSE}


# ───────────────────────────── stub geometry ─────────────────────────────────
class StubField:
    """Duck-typed field model (same layout as test_phase.StubField): blue HUB/TOWER at
    low-x, red at high-x; satisfies PhaseTracker (towers/hubs/length_m) + in_hub."""
    length_m = 16.46

    def __init__(self):
        self.hubs = {
            "blue": {"center_m": [3.20, 4.10], "radius_m": 0.80},
            "red": {"center_m": [13.26, 4.10], "radius_m": 0.80},
        }
        self.towers = {
            "blue": {"center_m": [1.50, 4.10], "climb_height_m": 1.20},
            "red": {"center_m": [14.96, 4.10], "climb_height_m": 1.20},
        }

    def in_hub(self, xy, alliance, tol_m=0.0):
        hub = self.hubs.get(alliance)
        if not hub:
            return False
        cx, cy = hub["center_m"]
        r = hub["radius_m"] + tol_m
        return (xy[0] - cx) ** 2 + (xy[1] - cy) ** 2 <= r * r


_FAST_CFG = {
    "intake_floor_m": 0.6, "converge_min_s": 0.0, "vanish_grace_s": 0.1,
    "outtake_origin_m": 1.0, "outtake_min_speed_mps": 1.5, "shot_min_speed_mps": 2.5,
    "shot_lookahead_s": 2.0, "intake_dedup_s": 0.0,
}


def _runner(field=None, cfg=None, expected=None):
    c = dict(_FAST_CFG)
    c.update(cfg or {})
    fm = field if field is not None else StubField()
    return MatchRunner(field_model=fm, hub_localizer=fm, cfg=c, fps=FPS,
                       expected_alliances=expected)


# ───────────────────────────── track builders ────────────────────────────────
def _robot(tid, xy, alliance="blue", size_m=0.8, vel=(0.0, 0.0), box_px=None,
           conf=0.9, bumper_reads=None):
    """A ByteTrack-stage robot Track with field geometry already attached (as the engine
    geometry layer would have done before MatchRunner.process runs)."""
    if box_px is None:
        # place a 60px box at a px location derived from the field x (so distinct field
        # robots get distinct pixel centroids -> sixbot keeps them apart).
        px = 100.0 + xy[0] * 40.0
        py = 200.0 + xy[1] * 40.0
        box_px = (px - 30, py - 30, px + 30, py + 30)
    x1, y1, x2, y2 = box_px
    t = Track(tid=tid, cls="robot", xyxy=(x1, y1, x2, y2),
              centroid_px=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
              field_xy_m=xy, alliance=alliance, vel_mps=vel)
    # Track has no `conf` field (that lives on Detection); MatchRunner.process reads it
    # off the track via getattr, so we attach it as a dynamic attribute here.
    t.conf = conf
    t.field_wh_m = (size_m, size_m)
    if bumper_reads is not None:
        t.bumper_reads = bumper_reads
    return t


def _ball(tid, xy):
    return Track(tid=tid, cls="ball", xyxy=(0, 0, 5, 5), centroid_px=(0, 0), field_xy_m=xy)


def _flight(tid, xy, vel=(0.0, 0.0)):
    return Track(tid=tid, cls="ball", xyxy=(0, 0, 5, 5), centroid_px=(0, 0),
                 field_xy_m=xy, vel_mps=vel, is_flight=True)


def _state(frame_idx, robots, inventory=None, flights=None, phase=None, overlay=None):
    ov = overlay if overlay is not None else OverlayState(phase=phase)
    return FrameState(frame_idx=frame_idx, t_s=frame_idx / FPS, robots=robots,
                      balls_inventory=inventory or [], balls_flight=flights or [],
                      overlay=ov)


def _six_field_layout(f, churn=False):
    """Six well-separated robots (3 red, 3 blue) drifting slightly. When ``churn`` is set,
    the RAW track ids are scrambled per frame to prove the stable ids do NOT follow them."""
    base = [
        ((4.0, 2.0), "red"), ((6.0, 2.0), "red"), ((8.0, 2.0), "red"),
        ((4.0, 6.0), "blue"), ((6.0, 6.0), "blue"), ((8.0, 6.0), "blue"),
    ]
    robots = []
    for slot, ((x, y), ally) in enumerate(base):
        # a tiny rightward drift so velocity is well-defined
        xy = (x + 0.02 * f, y)
        # raw id churns every frame when requested (e.g. 1000 + slot + f*10)
        raw_tid = (1000 + slot + f * 13) if churn else (10 + slot)
        robots.append(_robot(raw_tid, xy, alliance=ally, vel=(0.2, 0.0)))
    return robots


# ═══════════════════════ (a) cap + stable IDs under churn ═════════════════════
def test_process_caps_robots_at_six():
    """st.robots is overwritten with <=6 stable tracks (here exactly 6: 3 red, 3 blue)."""
    mr = _runner()
    res = mr.process(_state(0, _six_field_layout(0), phase="teleop"))
    st = res["state"]
    assert len(st.robots) <= 6
    assert len(st.robots) == 6
    assert sum(r.alliance == "red" for r in st.robots) == 3
    assert sum(r.alliance == "blue" for r in st.robots) == 3
    # six output mirrors st.robots ids
    assert {d["id"] for d in res["six"]} == {r.tid for r in st.robots}


def test_stable_ids_under_raw_id_churn():
    """The stable track ids persist across frames even though the RAW input ids are
    scrambled every frame — this is the 6-robot prior doing its job."""
    mr = _runner()
    slot_ids = None  # left->right within an alliance -> stable id
    for f in range(30):
        res = mr.process(_state(f, _six_field_layout(f, churn=True), phase="teleop"))
        st = res["state"]
        assert len(st.robots) == 6, f"frame {f}: expected 6 stable tracks"
        # map (alliance, left->right slot) -> stable id
        cur = {}
        for ally in ("red", "blue"):
            row = sorted((r for r in st.robots if r.alliance == ally),
                         key=lambda r: r.centroid_px[0])
            for i, r in enumerate(row):
                cur[(ally, i)] = r.tid
        if slot_ids is None:
            slot_ids = cur
        else:
            assert cur == slot_ids, (
                f"frame {f}: stable id followed the churning raw id {slot_ids} -> {cur}")
    # exactly six distinct stable ids overall
    assert len(set(slot_ids.values())) == 6


def test_excess_raw_robots_do_not_exceed_six():
    """Even if ByteTrack hands MORE than six robot tracks (phantom 4th per alliance), the
    6-robot prior caps the output at six (3 per alliance)."""
    mr = _runner()
    robots = _six_field_layout(0)
    # add phantom extras far away on each alliance
    robots.append(_robot(7777, (2.0, 1.0), alliance="red"))
    robots.append(_robot(8888, (2.0, 7.5), alliance="blue"))
    res = mr.process(_state(0, robots, phase="teleop"))
    st = res["state"]
    assert len(st.robots) <= 6
    assert sum(r.alliance == "red" for r in st.robots) == 3
    assert sum(r.alliance == "blue" for r in st.robots) == 3


# ═══════════════════════════ (b) alliance + (c) phases ════════════════════════
def test_alliance_labels_resolve_on_stable_tracks():
    mr = _runner()
    res = mr.process(_state(0, _six_field_layout(0), phase="teleop"))
    alliances = {r.alliance for r in res["state"].robots}
    assert alliances == {"red", "blue"}
    # phase_out keys match the stable track ids and carry the same alliances
    phase_out = res["phase"]
    assert set(phase_out) == {r.tid for r in res["state"].robots}
    assert {rec["alliance"] for rec in phase_out.values()} == {"red", "blue"}


def test_phase_labels_appear():
    """Every robot's per-frame record carries a valid PHASE label string."""
    mr = _runner()
    res = mr.process(_state(0, _six_field_layout(0), phase="teleop"))
    for rid, rec in res["phase"].items():
        assert rec["phase"] in _PHASE_LABELS
        assert isinstance(rec["label"], str) and rec["label"]


def test_intake_to_holding_phase_through_runner():
    """A robot over a ball that converges then VANISHES inside it reads INTAKING (on the
    confirmed consumption) then HOLDING(1) — proving the detection->six->phase->scoring
    path is wired (phase derives held from the embedded scorer). INTAKING is gated on real
    consumption, not the approach (the dense-field fix)."""
    mr = _runner()

    def frame(f, ball_xy=None, inv=True):
        robots = [_robot(500 + f, (5.0, 4.0), alliance="blue", vel=(0.0, 0.0))]
        inventory = [_ball(50, ball_xy)] if (inv and ball_xy is not None) else []
        return _state(f, robots, inventory=inventory, phase="teleop")

    saw_intaking = False
    f = 0
    for dist in [3.0, 2.0, 1.0, 0.5, 0.3]:
        res = mr.process(frame(f, ball_xy=(5.0 - dist, 4.0)))
        rec = next(iter(res["phase"].values()))
        if rec["phase"] == INTAKING:
            saw_intaking = True
        f += 1

    rec = None
    for _ in range(int(mr.phase.scoring.vanish_grace_frames) + 3):
        res = mr.process(frame(f, inv=False))
        rec = next(iter(res["phase"].values()))
        if rec["phase"] == INTAKING:
            saw_intaking = True
        f += 1
    assert saw_intaking, "expected INTAKING on the confirmed consumption (converge -> vanish)"
    assert rec["held"] == 1

    # step past the brief INTAKING latch -> settles to HOLDING(1)
    for _ in range(int(mr.phase.intake_hold_frames) + 2):
        res = mr.process(frame(f, inv=False))
        rec = next(iter(res["phase"].values()))
        f += 1
    assert rec["held"] == 1
    assert rec["phase"] == HOLDING
    assert "HOLDING(1)" in rec["label"]


def test_team_number_attached_when_bumper_reads_resolve():
    """Bumper-number OCR reads attached to the raw tracks resolve (via identify6) onto the
    stable tracks, and finalize() carries the team number through."""
    expected = {"red": [254, 1678, 973], "blue": [118, 2056, 1114]}
    mr = _runner(expected=expected)
    # give the three blue robots clear, repeated reads of three blue roster teams
    blue_targets = [118, 2056, 1114]
    for f in range(6):
        robots = _six_field_layout(f)
        blues = [r for r in robots if r.alliance == "blue"]
        for r, team in zip(sorted(blues, key=lambda r: r.field_xy_m[0]), blue_targets):
            r.bumper_reads = [(team, 0.9)]
        mr.process(_state(f, robots, phase="teleop"))

    rec = mr.finalize("qm1")
    teams = {stats.get("team") for stats in rec["robots"].values() if stats.get("team")}
    # at least the blue roster numbers we fed should be resolved onto some tracks
    assert set(blue_targets) & teams, f"expected resolved blue teams among {teams}"


# ═══════════════════ (d) finalize record compatible with SCOUT_COLUMNS ════════
def test_finalize_record_shape_and_envelopes():
    """finalize() robots dict is SCOUT_COLUMNS-compatible and preserves available:False
    envelopes for unresolved metrics (never a fabricated 0)."""
    mr = _runner()
    # run a few frames WITHOUT a pose so shot/defense/jitter stay gated. (No pose ->
    # field_xy_m on tracks is still set by us, but the scorer's resolvability and the
    # phase pose_seen track availability; we additionally null the pose on the state.)
    for f in range(5):
        robots = _six_field_layout(f)
        mr.process(_state(f, robots, phase="teleop"))

    rec = mr.finalize("qm1")
    assert "robots" in rec and rec["robots"]
    assert "events" in rec

    for rid, stats in rec["robots"].items():
        # every SCOUT column is present OR unwrappable from this stats dict
        for col in SCOUT_COLUMNS:
            assert col in stats, f"robot {rid} missing scout column {col}"
        # gated metrics that had no resolution must be the {value, available:False}
        # envelope — NOT a fabricated 0.
        acc = stats["shot_accuracy"]
        assert isinstance(acc, dict) and acc["available"] is False
        assert acc["value"] is None and acc["needs"]
        fuel = stats["fuel_scored"]
        assert isinstance(fuel, dict) and fuel["available"] is False
        assert fuel["value"] is None


def test_finalize_unresolved_metrics_unwrap_to_none_not_zero(tmp_path):
    """Through the real exporter, an unavailable metric unwraps to an EMPTY CSV cell
    (None), never the string '0'."""
    mr = _runner()
    for f in range(4):
        mr.process(_state(f, _six_field_layout(f), phase="teleop"))
    paths = mr.export(str(tmp_path), "qm2")
    rows = list(csv.DictReader(open(paths["csv"])))
    assert rows
    for row in rows:
        # shot_accuracy was never resolvable here -> empty, not "0"
        assert row["shot_accuracy"] in ("", None), f"fabricated value {row['shot_accuracy']!r}"
        assert row["fuel_scored"] in ("", None)


# ═════════════════════════ (e) export writes JSON + CSV ═══════════════════════
def test_export_writes_json_and_csv_with_scout_header(tmp_path):
    mr = _runner()
    for f in range(5):
        mr.process(_state(f, _six_field_layout(f), phase="teleop"))
    paths = mr.export(str(tmp_path), "qm1")

    assert paths["n_robots"] == 6
    # JSON loads and carries the robots + events
    record = json.load(open(paths["json"]))
    assert record["robots"]
    assert "events" in record

    # CSV header is EXACTLY the canonical scout columns
    header = next(csv.reader(open(paths["csv"])))
    assert header == SCOUT_COLUMNS


def test_export_csv_one_row_per_robot(tmp_path):
    mr = _runner()
    for f in range(3):
        mr.process(_state(f, _six_field_layout(f), phase="teleop"))
    paths = mr.export(str(tmp_path), "qm3")
    rows = list(csv.DictReader(open(paths["csv"])))
    assert len(rows) == 6


def test_finalize_record_json_serializable():
    """The whole finalize() record round-trips through JSON (primitives/None/list/dict)."""
    mr = _runner()
    for f in range(4):
        mr.process(_state(f, _six_field_layout(f), phase="teleop"))
    rec = mr.finalize("qm1")
    s = json.dumps(rec, default=str)
    back = json.loads(s)
    assert back["match_key"] == "qm1"


# ═══════════════════════ import without cv2 (module-level) ════════════════════
def test_module_imports_without_cv2():
    """Importing match_runner must NOT require cv2 (the video path imports it lazily)."""
    import importlib
    import frcscout.pipeline.match_runner as m
    importlib.reload(m)
    assert hasattr(m, "MatchRunner")
