# FRC Vision Scout v2 — Results

**Watches an FRC REBUILT (2026) broadcast — a local file OR a live YouTube URL — and writes,
per match, a scouting JSON + CSV and an annotated demo video. Runs on CPU (no CUDA) via
OpenVINO. Every metric is measured; anything not derivable from the footage is marked
`available:false` with a `needs` note — never faked.**

Built/validated on real footage from **2026ctwat** (NE District Waterbury). All numbers below
are measured on this machine (Intel Core Ultra 7 255U, CPU/OpenVINO, ~3.8 fps synchronous).

---

## Run it

```bash
# from v2/ ; the package is run with PYTHONPATH=src
# (1) a downloaded match file:
PYTHONPATH=src python scripts/run_scout.py --sync \
    --source data/footage/qual56.mp4 --start 2000 --frames 600 \
    --out-mp4 data/scout_out/qual56_demo.mp4 \
    --out-dir data/scout_out --match qual56

# (2) directly from a live / VOD YouTube URL (resolved via yt-dlp, then streamed):
PYTHONPATH=src python scripts/run_scout.py --sync \
    --source "https://www.youtube.com/watch?v=<LIVE_ID>" \
    --out-dir data/scout_out --match live   # add --out-mp4 to also render

# dashboard from the exported match record:
python scripts/make_scout_dashboard.py
```

`--sync` is the deterministic per-frame path (recommended for files/VODs). The async 60fps
`Engine` (`pipeline/engine.py`) is the real-time live path; on a fast file source it races,
so `--sync` is what the demos use. For a live stream, force a 720p60 format — team OCR is
unreadable below ~360p.

## Outputs (in `data/scout_out/`)

| file | what |
|---|---|
| `qual56_demo.mp4` | annotated video: 6 alliance-colored robot boxes + stable IDs + phase labels, magenta-outlined balls, real HUD (teams/score/timer), right-side per-robot metrics panel |
| `match_qual56.json` | full per-robot record: phase, held, every metric in the `{value,available,needs}` envelope, per-robot phase timeline, the read overlay (teams/scores/fuel) |
| `scouting_qual56.csv` | one row per robot, the 18 scout columns |
| `scout_dashboard.png` | 9-panel results dashboard (phase mix, possession, intake, ball-count timeline, 6-robot establishment, ID-stability fix, jitter, max-held, measured-vs-gated honesty panel) |

## What's MEASURED (qual56, frames 2000–2600)

- **6-robot tracking, stable identity.** Exactly 3 red + 3 blue held every frame; IDs stay
  fixed (0–5) and **coast** through full occlusions (raw detections drop to 0 → still 6 tracked).
  Distinct IDs over 600 frames: vanilla ByteTrack 32 → 6-robot prior 6.
- **Dense balls — never zero.** The ball model (YOLO11s@960) reads **up to 74 balls/frame**
  in a pile (5→30→57→74 as it accumulates).
- **Phases:** TRAVERSING / INTAKING / HOLDING (possession), per robot, per frame.
- **Per-robot metrics:** intake attempts, mean time-to-intake (1.4–2.6 s), max balls held
  (1–2), possession time (0–8.6 s), jitter index.
- **HUD OCR (real):** teams **B 8889/6933/6328  R 7907/7127/10266** (conf 1.0, voted over
  frames), match label, timer (counts down), blue score. These teams **match the
  TBA-verified qm56 rosters** (v1 validated 16/16 quals = 100% vs TBA at 2026ctwat) → the
  broadcast OCR is cross-validated against the known-correct result.

## Generality (qual59, 2nd match, same broadcast)

Tracking + ball detection generalize: 6 robots held, **117–135 balls/frame** detected, real
scores read (B 19 / R 58). It also exposed honest limitations (see below).

## Honest limitations (not bugs — physics / footage)

- **Per-robot scoring (`fuel_scored`, `shot_accuracy`, `scoring_rate`) = n/a.** Resolving a
  *made shot to a specific robot* needs a hub pose or an unambiguous shot; the classical
  flight detector is unreliable on a *moving broadcast camera* (suppressed in render). The
  **alliance** fuel/score IS read from the HUD and is authoritative.
- **No field pose on broadcast.** Field AprilTags aren't visible in the broadcast, so
  positions are **image-approximate** (a linear scale, good enough for proximity-based
  intake which is scale-invariant) and **climb** is disabled (the only monocular signal is a
  bbox-growth heuristic that a robot driving toward camera can fool — so we never claim a
  climb we can't verify). The AprilTag PnP path is fixed + tested (5/5) and resolves on any
  footage that *does* show the tags (e.g. a calibrated field cam).
- **Intake over-fires in ball-flooded endgames.** When 100+ balls blanket the field, every
  robot is always within intake radius of *some* ball, so proximity can't distinguish
  "surrounded" from "consuming." Normal play is clean; the end-of-match dump over-credits
  INTAKING.
- **Team→robot attribution is alliance-level**, not per-robot (the 6 teams are known from the
  HUD, but mapping which track is which team needs bumper-number OCR, unreliable on broadcast).
- **Team OCR robustness varies by match/window** (qual56 perfect; qual59's window misread).

## Architecture

`detect (OpenVINO) → track (ByteTrack) → 6-robot structural prior (sixbot) → identity →
phase state machine (embeds the scoring-event tracker) → metrics → export`. The robot
detector reuses the **multi-class ball model's `robot` class @960** (the single-class robot
nets detect ~0 on downscaled broadcast). Centerpiece: `pipeline/match_runner.py` +
`viz/scout_render.py`; HUD reader `scripts/hud_read.py`.

## Tests

`PYTHONPATH=src python -m pytest` → **543 passed**, 1 skipped. (compileall clean.)
