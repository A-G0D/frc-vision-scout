# FRC Vision Scout — Engine Framework & Preliminary Model

A structured architecture for a video-driven engine that watches FRC (REBUILT 2026) event
streams and produces, automatically, the data a room full of human scouts/scorekeepers
would: which teams played each match, what happened, and the per-robot metrics.

This document is **grounded in real data** — the structure below comes from actually
downloading and frame-analyzing the official *NE FIRST Waterbury Day 2* stream
(`ngvsHrmoW8U`, 9.5h). See `FEASIBILITY.md` for the evidence and `REBUILT-2026-game-model.md`
(in `../frc-vision-scout/` sibling doc) for the rules/scoring the engine must reproduce.

---

## 0. Design principle: read the broadcast first, watch the field second

The single most important finding: **the broadcast overlay already encodes the match
identity and the live score.** So the engine is built as two cooperating layers, simplest
first:

```
                         ┌─────────────────────────────────────────────┐
   YouTube stream  ──►   │  LAYER A — READ THE BROADCAST (overlay OCR)  │  ──►  schedule,
   (9h, low-res)         │  match id · 6 teams · live FUEL · timer/phase │       scores, clock
                         └─────────────────────────────────────────────┘
                                              │  (anchors space + time + identity)
                                              ▼
                         ┌─────────────────────────────────────────────┐
                         │  LAYER B — WATCH THE FIELD (physical CV)      │  ──►  per-robot
                         │  FUEL · robots · climbs · zones via AprilTags │       scouting metrics
                         └─────────────────────────────────────────────┘
```

Layer A is cheap, mostly non-ML, and **fully satisfies the original goal** (extract every
match + competitors). Layer B is the hard, high-value CV work and *depends on* Layer A for
match boundaries, team identity, and a score signal to validate against.

---

## 1. System architecture (modules)

```
src/
  youtube_extractor.py   [BUILT]  yt-dlp+ffmpeg: meta, full low-res download, frame sampling
  match_list.py          [BUILT]  ground-truth schedule from TBA (key) or Statbotics (open)
  overlay_detect.py      [BUILT]  match-vs-filler via overlay color signature (no OCR)
  scan_timeline.py       [BUILT]  whole-stream → match SEGMENTS + ROI crops + schedule align
  ── next ──
  ocr_overlay.py         [STUB ]  read team #s / match label / FUEL count from overlay ROIs
  field_register.py      [PLAN ]  AprilTag detect → camera↔field homography (Layer B anchor)
  detect_fuel.py         [PLAN ]  FUEL ball detection + HUB-entry events
  detect_robots.py       [PLAN ]  robot detection, bumper-color alliance, team-# association
  detect_climb.py        [PLAN ]  TOWER climb level (L1/L2/L3) per robot
  score_engine.py        [PLAN ]  apply REBUILT rules → reconstruct score (mirrors FMS)
  aggregate.py           [PLAN ]  emit per-match / per-robot / per-team scouting schema
pipeline.py              [PLAN ]  orchestrates: download → scan → OCR → align → (Layer B)
```

Status legend: **BUILT** = implemented & validated on real frames · **STUB** = interface
defined · **PLAN** = designed below.

---

## 2. The data model (preliminary)

Three nested records. JSON-serializable; these are the contracts every module reads/writes.

### 2.1 `EventVideo`
```jsonc
{
  "video_id": "ngvsHrmoW8U",
  "channel": "New England FIRST",
  "event_key": "2026ctwat",          // links to ground-truth schedule (TBA/Statbotics)
  "duration_s": 34298,
  "local_path": "data/ngvsHrmoW8U_144p.mp4",
  "scan_fps": 0.5,
  "segments": [ /* MatchSegment[] */ ]
}
```

### 2.2 `MatchSegment`  (Layer A output — this is the original deliverable)
```jsonc
{
  "index": 12,
  "t_start": 4980.0, "t_end": 5140.0,     // seconds into the video
  "kind": "qual",                          // qual | playoff | final | replay | recap
  "match_label": "Qualification 24 of 86", // OCR'd from overlay, or aligned from schedule
  "blue": [1991, 4905, 5687],              // OCR'd, then reconciled with schedule
  "red":  [ 230, 1100, 6328],
  "live_score": { "blue_fuel": 73, "red_fuel": 61 },  // last reliable overlay read
  "phase_track": [ {"t":4980,"phase":"AUTO"}, {"t":5000,"phase":"SHIFT_1"}, ... ],
  "confidence": 0.91,
  "source": "overlay+schedule"             // provenance for every field
}
```

### 2.3 `RobotMatchRecord`  (Layer B output — scouting)
Mirrors the metrics schema in `REBUILT-2026-game-model.md §5.2`:
```jsonc
{
  "match": "2026ctwat_qm24", "team": 4905, "alliance": "blue", "station": 2,
  "auto": { "mobility": true, "fuel": 7, "climb_level": 1 },
  "teleop": { "fuel_scored": 41, "fuel_into_inactive_hub": 6, "climb_level": 3,
              "climb_start_t": 148.2 },
  "cycles": { "count": 19, "avg_cycle_s": 6.8, "miss_rate": 0.12,
              "source_mix": {"depot":0.5,"human":0.3,"floor":0.2} },
  "defense": { "time_in_opp_zone_s": 14, "pins_committed": 1, "fouls_drawn": 2 },
  "active_shift_conversion": 0.87,         // fuel scored while own HUB active / total
  "reliability": { "disabled": false, "tipped": false },
  "provenance": "field-cv"
}
```

---

## 3. Layer A — read the broadcast (mostly built)

### 3.1 The overlay map (measured from real frames, normalized to W×H)
These ROIs live in `overlay_detect.py::OVERLAY_ROIS` and are the single source of geometry.
```
y∈[0,0.10]  ┌────────┬───────────────┬────────┬───────────────┬────────┐
            │blue fuel│  BLUE 3 teams │ TIMER  │  RED 3 teams  │red fuel│
            │ x/100  │  (0.18–0.42)  │(.45–.55)│  (0.58–0.82)  │ x/100  │
            └────────┴───────────────┴────────┴───────────────┴────────┘
   match label "Qualification N of M" sits at the very top center (y∈[0,0.04]).
```
Confirmed on real frames at 144p:
- **Alliance Selection** full-screen team grid → digits cleanly legible.
- **In-match bar** → present/absent cleanly detectable; digits legible at 240p+.

### 3.2 Segmentation (built, validated)
`scan_timeline.py`: sample @0.5 fps → `overlay_detect` per frame → bridge ≤12s gaps → keep
runs ≥30s. On `ngvsHrmoW8U`: **66 segments (34 full matches, 32 short recaps/replays)**.
*Tuning to do:* split over-merged segments (long-run median ≈425s spans queue+match+score-
hold); classify `recap/replay` vs real match by duration + presence of the `match_label`
animation; detect playoff overlays (alliance numbers, not team numbers — observed at 5:30:00).

### 3.3 OCR stage (`ocr_overlay.py`, next to build)
- Input: the per-segment ROI crops `scan_timeline.dump_roi_crops` already produces (6× upscaled).
- Backend: **easyocr** (GPU-optional) or **tesseract** with a digit allow-list
  (`--psm 7 -c tessedit_char_whitelist=0123456789`). Neither is installed yet — this is the
  one remaining dependency for Layer A.
- Robustness: OCR several frames across a segment and take the **mode** per slot (team
  numbers are static for the whole match → temporal voting kills single-frame errors).

### 3.4 Alignment to ground truth (built: `align_to_schedule`)
Streams play matches in schedule order, so the *k*-th detected match == the *k*-th scheduled
match. Two-way reconciliation:
- Schedule → fills teams when OCR is weak (low-res).
- OCR → detects/【repairs misalignment (skipped match, replay inserted, stream gap).
- Disagreement ⇒ lower `confidence`, flag for review. This redundancy is why Layer A is
  reliable even at 144p.

---

## 4. Layer B — watch the field (planned CV)

This is where "robot detection engines later" lands. Build order chosen so each stage is
independently testable and each depends only on earlier ones.

### 4.1 Field registration (`field_register.py`) — the anchor
- Detect **AprilTags 36h11 IDs 1–32** (known field positions, see game model §2.1).
- Solve camera→field homography per shot. The broadcast cuts cameras, so maintain a small
  bank of homographies keyed by detected tag constellation; re-localize on each cut.
- Output: a function mapping pixel→field coordinates, so all later detections become
  semantic ("FUEL entered the RED HUB", "robot in BLUE alliance zone").

### 4.2 Object detection
- **FUEL** (`detect_fuel.py`): 15 cm yellow foam balls. Start with HSV color + Hough/blob;
  upgrade to a small YOLO if needed. Event of interest: a ball crossing into the HUB top
  opening (game model §6.1 FUEL_SCORED). Honor the 3-second scoring tails.
- **Robots** (`detect_robots.py`): detect robots, classify **alliance by bumper color**,
  associate **team number** by (a) bumper-number OCR when readable, else (b) cross-reference
  the Layer-A roster + tracking. 6 robots/match.
- **Climb** (`detect_climb.py`): per robot, bumper-cover height vs LOW/MID/HIGH rung → L1/L2/L3
  (game model §3.1 rules: AUTO=L1 only & ≤2 robots; one TELEOP level per robot).

### 4.3 The atomic event bus
Everything Layer B emits reduces to the 5 events defined in the game model §6.1:
`FUEL_SCORED · CLIMB_STATE · ZONE_TRANSITION · HUB_ACTIVE_CHANGE · CONTACT/FOUL_CANDIDATE`.
Layer A supplies two of these for free: HUB_ACTIVE_CHANGE (overlay light state / shift
schedule) and a cross-check on FUEL_SCORED (overlay `x/100` count).

### 4.4 Score engine (`score_engine.py`)
Pure function: events + phase track → score, applying REBUILT rules (game model §1.1, §3).
**Validation loop:** its reconstructed FUEL count must track the overlay's live `x/100`
read — that's a built-in, free accuracy meter for Layer B. Final score must match the
schedule's final score (from TBA) within tolerance.

---

## 5. Phasing / roadmap

| Phase | Deliverable | Depends on | Status |
|------:|-------------|-----------|--------|
| 0 | Tooling, download, frame sampling | — | ✅ done |
| 1 | Match/filler segmentation over whole stream | 0 | ✅ done (66 segs) |
| 2 | Overlay OCR → match id + 6 teams + live score | 1 + OCR install | ◻ next |
| 3 | Schedule alignment + reconciliation | 2, match_list | ◻ (code ready) |
| 4 | Segmentation tuning (split/classify recaps, playoff overlays) | 1 | ◻ |
| 5 | AprilTag field registration | 0 | ◻ Layer B start |
| 6 | FUEL detection + HUB-entry events | 5 | ◻ |
| 7 | Robot detect + alliance/team association | 5, 3 | ◻ |
| 8 | Climb-level detection | 5,7 | ◻ |
| 9 | Score engine + overlay cross-validation | 6,7,8 | ◻ |
| 10 | Aggregate → per-robot/team scouting schema | 9 | ◻ |

**Phases 2–3 finish the original request** (every match + its competitors, timestamped).
Phases 5–10 are the scouting/scoring engine.

---

## 6. Known risks & open tuning knobs
- **Segmentation over-merge** — bridge/min-duration params need tuning; add `match_label`
  presence as a second gate so queue/score-hold time isn't counted as match time.
- **Resolution vs OCR** — 144p proves structure; production should pull 240–360p (still small).
- **Playoff overlays show alliance numbers, not teams** — handle via schedule (alliance→teams).
- **Camera cuts** in Layer B — require per-shot re-registration; replays can double-count
  events (dedupe by match clock, which resets/animates on replays).
- **External deps** — OCR backend (easyocr/tesseract) + a `TBA_AUTH_KEY`. Statbotics is a
  keyless fallback but was 500-ing during testing.
- **Generalization** — ROI fractions are tuned to the NE FIRST overlay; other broadcasters
  (e.g. main FIRST channel, Twitch mirrors) differ → keep ROIs per-template, auto-pick by
  matching a few known frames.

---

## 7. What exists right now (runnable)
```bash
# metadata
python src/youtube_extractor.py <video_id>
# download one low-res copy (done for ngvsHrmoW8U @144p)
# segment the whole stream into matches:
python src/scan_timeline.py data/ngvsHrmoW8U_144p.mp4   # -> *_segments.json (66 segments)
# match/filler check on any frame:
python src/overlay_detect.py frames/spread/s_*.jpg
# ground-truth schedule (needs TBA key or Statbotics up):
python src/match_list.py 2026ctwat
```
Real artifacts in `data/` and `frames/`: the 144p stream copy, sampled frames, the upscaled
overlay strip (`frames/spread/zoom_013000_top.png`), contact sheets, and the 66-segment JSON.
