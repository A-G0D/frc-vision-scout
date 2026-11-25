# FRC Vision Scout v2

A rebuild of the scout engine around one principle and one constraint:

- **Principle:** the broadcast **overlay is ground truth for the count/score**; computer
  vision is for **space** (who, where, which arc, which HUB). That split is what makes
  60fps-on-CPU feasible — you never run the expensive ball detector at 60Hz.
- **Constraint:** trained on **CUDA in the cloud (scriptable, no browser)**, deployed on the
  **Intel Core Ultra 7 (CPU / Arc iGPU / NPU) at 60fps, no CUDA, no internet** at contest.

This tree started as a **trainable skeleton** with per-module algorithm bodies marked `TODO`.
Those bodies are now filled in (see Status below) — wiring, configs, eval gates, cloud-training
app, and the detection/tracking/geometry/fuse algorithms are all implemented and covered by
the test suite. Point it at footage, and it trains.

> v1 lives one directory up and is untouched. Read `../PROJECT_SUMMARY.md` and the teardown
> for *why* every design choice here is the way it is.

---

## Why each design choice (vs v1)

| v1 problem | v2 fix | Where |
|---|---|---|
| 90.7% mAP was teacher-agreement + frame leakage | mAP vs **human** labels; **match-level** split | `eval/`, `scripts/split_by_match.py` |
| 31fps called "real-time" on 60fps footage | explicit **60fps target + p99<16.6ms gate** | `pipeline/scheduler.py`, `eval/gates.py` |
| "100% vs TBA" hid broken playoffs | per-level acc + **structural-validity errors** | `eval/tba_verify.py`, `tests/test_tba_validity.py` |
| robots: 0 labels, dead class shipped | single-class robot net + **bumper-color** alliance | `detect/robot_net.py`, `configs/train_robot.yaml` |
| attribution self-test was tautological | **probabilistic** credit + reported coverage | `fuse/attribution.py` |
| fps=30 hardcoded vs 60fps footage | all timing in **seconds→frames(real fps)** | `io/config.py`, `configs/events.yaml` |
| geometry designed, never built | AprilTag PnP/homography + ballistic arc fit | `geometry/` |
| 7.5GB un-ignored in a shared repo | exhaustive `.gitignore`, weights in cloud | `.gitignore` |
| Colab browser cells | **Modal** CLI app, no browser | `train/modal_app.py` |

---

## File map

```
configs/            all knobs; timing in seconds/meters, never frame counts
  deploy.yaml         device map + per-layer cadence (the 60fps recipe)
  field_2026_rebuilt.yaml  AprilTag/HUB/TOWER geometry (meters)
  events.yaml         pickup/shot/score thresholds (seconds, meters)
  train_ball.yaml / train_robot.yaml   two specialist students
  sources.yaml        footage manifest + match-level split
  review.yaml         who corrects labels (claude_read | vision_mcp | human)
  eval.yaml           held-out test paths + gate inputs

src/frcscout/
  io/         types.py (dataclasses), config.py (loader + seconds_to_frames)
  pipeline/   engine.py (orchestrator), scheduler.py (async cadence), decode.py (QSV)
  detect/     robot_net (NPU), ball_net (iGPU, 5Hz), flight (CPU 60Hz), overlay_ocr (2Hz)
  track/      bytetrack (robots), ball_kalman (flight)
  geometry/   apriltag (pose), field (img<->field), ballistics (single-cam 3D arc)
  fuse/       events, attribution (probabilistic), state
  eval/       gates (release gate), metrics, tba_verify

train/        modal_app.py (cloud GPU), train_student.py, export_openvino.py
scripts/      fetch_footage, fetch_tba, extract_frames, label_review, split_by_match
tests/        scheduler-never-blocks, structural-validity (v1 regressions as tests)
.github/workflows/gate.yml   CPU CI: lint + tests + gate
```

---

## The pipeline — chosen path: **Kaggle** (free T4, no credit card)

Training (autolabel → train → export) happens in one Kaggle notebook run. Two flavors:

**A. Browser (zero setup beyond the account):**
```
make footage                 # Claude: yt-dlp the clips (no browser)
# you: kaggle.com -> New Dataset -> upload data/footage/  (drag + drop)
# you: open kaggle/frc_train_kaggle.ipynb -> GPU T4x2 + Internet On -> Run All
# you: download frc_model_openvino.zip from the Output tab
unzip frc_model_openvino.zip -d weights/
```

**B. API-driven (Claude does it all; you only make the account + token once):**
```
make footage                 # Claude: download clips
make kaggle-up               # Claude: upload footage as a Kaggle Dataset (API)
make kaggle-push             # Claude: push + run the notebook headless (GPU+internet)
make kaggle-status           # Claude: poll until done
make pull                    # Claude: download the model zip -> weights/
```

Then, on the laptop / locally:
```
make tba                     # Claude: fetch TBA ground truth (no browser)
make eval                    # metrics vs HUMAN labels + overlay + TBA
make gate                    # CI gate: nonzero exit if any threshold fails
make bench SOURCE=clip.mp4   # laptop: p99 frame time vs 16.6ms
make run   SOURCE=<url>       # laptop: live 60fps engine
```

> Modal remains as an alternative (`make train-modal` / `export-modal`) if you ever want
> fully hands-off and don't mind the small credit usage.

---

## How Claude drives this without you in a browser

Almost everything is a CLI or API call I run via the Bash tool. The split:

**I do automatically (after one-time auth):**
- Footage download (`yt-dlp`), frame extraction, dataset assembly — Bash.
- TBA ground truth — `requests` HTML-scrape path (no key) or API if `TBA_AUTH_KEY` set.
- Cloud GPU train/export — Kaggle API from Bash (`kaggle datasets create`, `kernels push`
  with GPU+internet, `kernels output`). No Colab cells, no editor clicks.
- Pull weights — `modal volume get`. Eval + gates — local CPU, Bash.
- **Labeling** — in `claude_read` mode I open the review composites with the Read tool
  (I have native vision), compare to the teacher draft, and write corrected YOLO labels.
  This is the bottleneck I take off your hands. Dev-time only — no API in the deployed engine.
- Repo/CI — push and watch the gate run via the GitHub MCP.

**Only you can do (one-time, ~10 min, then never again):**
- Create a Kaggle account, **phone-verify it** (unlocks free GPU + internet), and download
  an API token (`kaggle.json`) to `~/.kaggle/`. After that I drive every run via the API.
  (Or skip the token and just click *Run All* in the browser — path A above.)
- Provide/approve the target-laptop `make bench` result (the fps gate needs the real Arc/NPU).

---

## MCPs / tools needed to automate it

You barely need new MCPs — the ones already connected plus one CLI cover ~95%:

| Capability | What it uses | Status |
|---|---|---|
| Local scaffolding, dataset assembly | **filesystem MCP** | ✅ connected |
| Repo, push, CI runs, PR review | **GitHub MCP** + Actions | ✅ connected |
| TBA data, FRC field/AprilTag layout, docs | **WebFetch / WebSearch** | ✅ available |
| Cloud GPU train/export | **`kaggle` CLI** (Bash) — datasets + kernels API | free, no card; needs one-time account + `kaggle.json` — **not an MCP, just a CLI I drive** |
| Labeling at scale (optional) | a **vision MCP** for batch frame correction | optional; the Read-tool path needs nothing |
| Weight/dataset hosting (optional) | HF Hub, or an **S3/GCS MCP** | optional; Modal Volumes already cover handoff |

Honest takeaway: there's **no "Modal MCP"** and you don't need one — Modal's CLI is fully
scriptable, so I run it through Bash like any other command. The only genuinely manual,
un-automatable step is minting the cloud token once. Everything downstream is mine to run.

---

## Status

**All six layers implemented and tested** (823 unit tests, full suite green, whole tree compiles):
geometry (AprilTag pose/homography + single-cam ballistic arc fit), tracking (Kalman ByteTrack
robots + flight balls), detection (tiled NMS + per-layout overlay OCR), eval (mAP-vs-human, IDF1,
TBA incl. playoffs, release gate), training (autolabel → active-learning-until-plateau → distill,
QAT export), and the live async 60fps engine (stream ingest + reconnect + render).

**Dataset:** 6 distinct 1080p60 matches from the NE Waterbury VOD — train ×4, val ×1,
`final1` held out as the honest test match (never uploaded, never trained).

**Held-out generalization gate (`scripts/eval_held_out.py`, `tests/test_held_out_gate.py`):**
until now `final1` had only ever been touched by the A6.3 geometry audit — the full
`detect -> track -> 6-robot prior -> phase state machine` pipeline had never actually been
run against it, so "v2 generalizes to unseen footage" was an unverified claim. This script
runs the SAME `ScoutPipeline` used for the qual56/qual59 demos on `final1` and reports
robot/ball detection health against the validated `qual56` baseline over the identical
window. Measured on frames 2000-2600 (`--start 2000 --frames 600`, `intel:cpu`):

| metric | qual56 (val split, baseline) | final1 (test split, held-out) |
|---|---|---|
| six_any_frac (>=1 robot tracked) | 0.76 | 1.00 |
| ball_nonzero_frac (>=1 ball detected) | 0.99 | 1.00 |
| max robots simultaneously tracked | 4 | 6 |
| six_full_frac (exactly 6 tracked) | 0.00 | 0.87 |

The held-out match does not degrade relative to the baseline (gate: PASS) — it in fact
tracks more cleanly than qual56 does under the CURRENT config on this window (qual56's
max is 4 robots here, not the 6 documented from an earlier config/window in the section
above; a real, measured discrepancy worth re-checking, not swept under the rug). Run it
yourself: `PYTHONPATH=src python scripts/eval_held_out.py --baseline qual56 --held-out
final1 --start 2000 --frames 600`. The pytest version (`test_held_out_gate.py`, marked
`slow`) runs the same check on a smaller 90-frame window for CI speed.

**Known deferrals (documented, not bugs):**
- Flight-tracker speed gate uses the pixel fallback (fine for a fixed broadcast camera); the
  field-frame refinement needs pose threaded into `FlightTracker.update`.
- `_LIVE_BAR_ROIS` + the `field_2026_rebuilt.yaml` tag/HUB coordinates are structural
  placeholders — re-measure on real footage / the official field drawings before trusting
  metric geometry output.
- GPU run is gated on the Kaggle account being phone-verified (unlocks free GPU + internet).
