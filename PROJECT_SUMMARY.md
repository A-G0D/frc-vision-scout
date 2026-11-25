# FRC Vision Scout — Complete Project Summary

A video-analysis engine that watches **FIRST Robotics Competition REBUILT (2026)** match footage
and automatically detects, tracks, and counts game elements (FUEL balls, robots) — replacing human
scorers/scouts. This document is the full story of the project: the goal, every pivot, the final
architecture, results, hard-won gotchas, current state, and how to run/extend it.

---

## 1. The Goal (and how it evolved)

**Original ask:** research the current FRC game (REBUILT 2026), then build a system that watches
official livestreams/VODs and auto-extracts every match's competitors + scores, checked against
The Blue Alliance (TBA).

**Evolution of intent over the project:**
1. Extract match data (teams, scores) from broadcast overlays → ✅ done (100% vs TBA on quals).
2. Detect/track on-field objects (balls, robots) and simulate scoring → partially done.
3. **Hard constraint added:** *no Anthropic inference API* — classical CV + local models only.
4. **Revolution demanded:** classical CV (HSV) ball detection was too weak → switch to deep models.
5. **Quality bar raised:** "level 1 → level 10/20, industrial grade," ~99% accuracy, "INSANE results."
6. **Speed bar:** a 9-hour video parsed in ~15 min; ultimately **live, real-time** inference.
7. **Final hard requirement:** **LIVE inference is non-negotiable** for contest day, running on the
   user's own hardware (Intel laptop, no CUDA), with as much one-time precompute/training as needed.

**Final shape of the product:** a custom-trained ball/robot detector that runs **live on the user's
Intel Core Ultra 7 laptop (CPU / Arc iGPU / NPU) via OpenVINO — no CUDA, no internet, no cloud** —
produced by a one-time GPU training pipeline on free cloud (Colab/Kaggle).

---

## 2. The Game: REBUILT (2026)

- **FUEL** = yellow foam balls, the primary scoring object (what we detect/count).
- **HUBs** = scoring goals; **TOWERs** = climb structures.
- Phases: **AUTO** then **TELEOP**; **3 robots per alliance** (red/blue); Ranking Points.
- Live overlay shows match id, blue teams (left) / red teams (right), timer, and a live `X/100`
  FUEL count (ENERGIZED ranking point).
- Game rules/metrics captured in `REBUILT-2026-game-model.md`.

---

## 3. Architecture — the journey through three designs

### 3a. Layer A / Layer B (original)
- **Layer A "read the broadcast":** OCR the scoreboard overlay for match data. Cheap, mostly
  non-ML. → satisfied the "extract every match + competitors" ask.
- **Layer B "watch the field":** AprilTag field registration → object detection → score engine.
  The hard CV; the scouting granularity.

### 3b. The deep-learning pivot (balls)
Classical HSV ball detection hit a ceiling (~56 balls/frame, crowd leakage, missed airborne/far).
Switched to **YOLO11x** (COCO "sports ball" class 32) at high resolution + **SAHI** sliced
inference. This dramatically improved recall but is **slow** (offline-only on CPU).

### 3c. The final architecture: teacher → student → CPU deploy (THE design)
The key insight that makes "high recall **and** live" possible:

```
  ┌─ ONE-TIME PRECOMPUTE (free cloud GPU: Colab/Kaggle T4) ──────────────────┐
  │                                                                          │
  │  match clip ──► AUTO-LABEL (the "teacher")                               │
  │                  • YOLO11x + SAHI tiling  → every FUEL ball              │
  │                  • YOLO-World ("robot")   → robots                       │
  │                  • PLAY-column gate, no human labeling                   │
  │                        │                                                 │
  │                        ▼                                                 │
  │                 TRAIN the "student" (YOLO11s, 120 epochs, heavy aug)     │
  │                        │                                                 │
  │                        ▼                                                 │
  │                 EXPORT  OpenVINO INT8  (+ONNX, +TensorRT backup)         │
  └────────────────────────┬─────────────────────────────────────────────────┘
                           ▼
  ┌─ LIVE (user's Intel laptop — NO CUDA / NO internet) ─────────────────────┐
  │   OpenVINO INT8 student  +  ByteTrack  →  31 fps on the Arc iGPU          │
  └──────────────────────────────────────────────────────────────────────────┘
```

- The **slow SAHI teacher runs once**, only to *label* frames.
- The **fast student learns that recall** and reproduces it in a single pass → live.
- A generic model can't be both fast and high-recall; a *trained* one can. This is the whole trick.

---

## 4. Results (measured)

| Metric | Result | Notes |
|---|---|---|
| **Ball detection mAP@50** | **0.907 (90.7%)** | custom YOLO11s, validated on held-out frames |
| Ball detection mAP@50-95 | 0.74 | |
| Precision / Recall (ball) | 0.91 / 0.86 | across 6,083 ball instances |
| **Live speed (Arc iGPU)** | **31.1 fps** @ imgsz 768 | OpenVINO INT8, real-time (broadcast = 30fps) |
| Match teams + scores vs TBA | 100% | proven on 16/16 quals (event 2026ctwat) |
| Frame classification (live/info/filler) | 100% | on labeled set |
| Model size | **9.8 MB** (OpenVINO INT8) | runs on CPU/iGPU/NPU |
| Deploy hardware | Intel Core Ultra 7 255U | Arc iGPU + NPU, no CUDA |

**Proven end-to-end:** trained on Colab T4 → exported OpenVINO INT8 → ran live on the user's laptop
at 31 fps producing an annotated video, fully offline. The core deliverable works.

### Known gaps (honest)
- **Robots not detected** in the trained model — YOLO-World produced **0 robot labels**, so the
  student never learned the class. Needs a fix (better robot auto-labeling or bumper-color prior).
- **Small dataset** — the training clip was ~4 min → only ~71 val frames. INT8 calibration warned
  ">300 images recommended." More footage (more matches) will lift accuracy further.
- **360p is a hard floor** — balls are ~5 px and undetectable at 360p; 720p/1080p is required.
- **Per-robot point attribution** remains fundamentally ambiguous (concurrent scoring); not solved.

---

## 5. Key Files

### Deployment (runs on the user's machine)
- `src/live_openvino.py` — **the live runner.** Loads the OpenVINO model, runs ByteTrack tracking,
  shows/saves annotated video, prints fps. Supports `--device intel:cpu|gpu|npu`, `--benchmark`
  (times all three devices), `--save out.mp4` (headless), `--source <file|streamURL>`.
  - Live window: `python src/live_openvino.py models/frc_model_openvino_model --device intel:gpu --source match_720.mp4`
  - Save video: add `--save out.mp4` (needed because the installed OpenCV is headless/no-GUI)
  - Benchmark:  `python src/live_openvino.py models/frc_model_openvino_model --benchmark --source match_720.mp4`
- `models/frc_model_openvino_model/` — the deployed model (`best.xml` + `best.bin`), unzipped from
  the Colab/Kaggle output `frc_model_openvino.zip`.

### Training notebooks (one-time, free cloud GPU)
- `colab/frc_train.ipynb` — 8-cell pipeline for **Google Colab**: setup → upload clip → auto-label
  → train (auto-resumes on disconnect) → export OpenVINO INT8 → optional demo render → package.
- `kaggle/frc_train_kaggle.ipynb` — same pipeline adapted for **Kaggle** (no Drive; reads clip from
  `/kaggle/input/**`, writes to `/kaggle/working`). Use when Colab GPU limit is hit.
- `colab/frc_colab.ipynb` — baseline (no training): high-res YOLO11x + SAHI + ByteTrack on the GPU.
- `colab/README.md` — click-by-click Colab instructions.

### Earlier engines (classical / superseded, still in repo)
- `src/detect_fuel.py` — HSV classical ball detector (iteration 6, carpet trapezoid + perspective).
- `src/detect_fuel_yolo.py` — YOLO11x ball detector w/ PLAY-column gate (the bridge to deep models).
- `src/results_extractor.py`, `src/scan_results.py`, `src/verify_tba.py` — match-data extraction +
  TBA verification (100% on quals).
- `src/frame_classify.py`, `src/detect_robots.py`, `src/detect_structures.py`, `pipeline_fast.py`
  — the original overlay/segmentation stack (9h video in ~12 min).
- `src/sim/` — round-1 tracking + game-state simulation (rough; per-robot attribution unsolved).

### Docs
- `ARCHITECTURE.md` — the cutting-edge pipeline design + honest per-metric accuracy targets.
- `UPGRADE_PLAN.md` — 30 upgrades across 6 categories (balls, robots, tracking, perf, geometry, MLOps).
- `REBUILT-2026-game-model.md`, `FEASIBILITY.md`, `FRAMEWORK.md` — game model, feasibility, framework.
- `PROJECT_SUMMARY.md` — this file.

### Data
- `match_720.mp4` (repo root, ~80 MB) — the **good training clip**: 1280×720@60, ~4 min playoff
  window (Upper Bracket R4 M11) from event 2026ctwat.
- `match_clean.mp4` (~66 MB) — re-encoded 720p30 baseline-h264 version (extra-compatible).
- `data/demo20_h264.mp4` — the **live-on-iGPU demo** (20s, 31 fps, ball tracking).
- `data/match_1080p*.mp4` — (none valid; 1080p section downloads kept failing, see gotchas).
- `data/tba_waterbury2026.json` — TBA ground truth (nested under "matches" key).

---

## 6. Hard-Won Gotchas (the stuff that cost hours — read before repeating)

### Downloading footage (yt-dlp)
- **Clipping a section from a long (9.5h) VOD:** use the **HLS format `-f 300`** (720p60, audio
  muxed) or `-f 301` (1080p60) **+ `--js-runtimes node` + `--force-ipv4` + `--download-sections
  "*HH:MM:SS-HH:MM:SS"`**. HLS downloads only the ~40 fragments for the window — fast & reliable.
  - ✅ Working command:
    `yt-dlp -f 300 --js-runtimes node --force-ipv4 --download-sections "*6:58:30-7:02:30" -o match_720.mp4 "<url>"`
- **DON'T use DASH formats** (`299`/`298`+`140`) with `--download-sections` — ffmpeg must seek 7h
  into one 12 GB remote file and **hangs at "Destination"** (froze for an hour). 
- **`--js-runtimes node` is mandatory** — without it, formats show "not available" / extraction fails.
- **`--force-ipv4`** — googlevideo CDN failed IPv6 DNS ("name does not resolve"); IPv4 fixed it.
- Source video: `https://www.youtube.com/watch?v=ngvsHrmoW8U` (NE Waterbury 2026 Day 2, 9.5h).
  Playoff match is at ~6h59m (`25150s`).

### Resolution
- **360p gives only ~13 balls/frame** — that's the resolution ceiling, NOT engine failure. Balls
  are ~5 px. **Use 720p+.** This was misdiagnosed as "bad detection" for a while.

### Colab
- **Colab's datacenter IP is bot-gated by YouTube** ("Sign in to confirm you're not a bot") on ALL
  player clients — you **cannot download YouTube on Colab**. Download locally (residential IP works)
  and **upload the clip** instead.
- **`os.replace()` fails Drive↔local** ("Invalid cross-device link") — use **`shutil.move()`**.
- **Install:** drop deprecated **`openvino-dev`** (breaks pip resolver → `ModuleNotFoundError:
  ultralytics`); install `ultralytics openvino nncf supervision sahi`. Don't hide errors with
  `>/dev/null`.
- **Free GPU limit** resets in ~12-24h; sessions disconnect when idle.
- **Training survives disconnects** because checkpoints write to Drive every epoch → re-run the
  train cell, it `resume=True`s from `last.pt`.

### Kaggle (free GPU alternative)
- Requires **phone-verified account** to unlock GPU + internet.
- No Google Drive — upload the clip as a **Dataset**, attached at `/kaggle/input/**`.
- **Accelerator is set in the EDITOR** (right panel → Session options → Accelerator → GPU T4 x2),
  not on the notebook's view page. Also flip **Internet → On**.
- `/kaggle/working` is **wiped between sessions** → finish in one run or **Save Version**.
- **GitHub Codespaces has NO GPU** (CPU-only) — not an option for training.

### OpenCV / OpenVINO (local deploy)
- **`cv2.get(CAP_PROP_FRAME_COUNT)` is unreliable** — returns **0** for some mp4 encodings even when
  the video reads fine. Test readability with `cap.read()`, not the frame count.
- The installed OpenCV is a **headless build** (no GUI) → `cv2.imshow`/`destroyAllWindows` throw.
  For video output use **`--save`** (no window). Live window needs GUI-enabled `opencv-python`.
- OpenVINO sees devices **`['CPU','GPU','NPU']`** on the Core Ultra 7; ultralytics device strings are
  **`intel:cpu` / `intel:gpu` / `intel:npu`**. The Arc iGPU (`intel:gpu`) gave 31 fps @768.

### Misc
- **Wrong-file trap:** a 262-byte file is a **failed download stub**. `match_720**p**.mp4` (with "p")
  was a dud; `match_720.mp4` (no "p") is the real 80 MB file. Always check size.
- Environment was often very slow (Python/cv2 startup 1-2 min) — batch work into single processes.

---

## 7. Current State

✅ **Working & delivered:**
- Custom ball detector, **90.7% mAP**, trained from this footage with zero manual labeling.
- **Live on the user's laptop at 31 fps** (Arc iGPU, OpenVINO INT8, no CUDA/internet).
- Annotated demo video produced locally (`data/demo20_h264.mp4`).
- Match-data extraction 100% vs TBA; frame classification 100%.
- Reproducible training notebooks for both Colab and Kaggle (auto-resume, crash-safe).

🟡 **In progress / open:**
- **Robots not detected** (YOLO-World labeled none) — top fix.
- More footage to push accuracy past ~90% (current dataset is one ~4-min clip).
- Optional: teacher (YOLO11x+SAHI) "supreme model" reference video — needs a GPU session (Colab
  reset or Kaggle); cell provided.
- Geometry/3D (AprilTag homography, ball-arc reconstruction) — designed, not built.
- Per-robot point attribution — fundamentally hard, not solved.

---

## 8. How To Use It

### Run the model live / make an annotated video (on the Intel laptop)
```bash
pip install ultralytics openvino supervision opencv-python    # one time
# pick the fastest chip:
python src/live_openvino.py models/frc_model_openvino_model --benchmark --source match_720.mp4
# save an annotated video (headless-safe):
python src/live_openvino.py models/frc_model_openvino_model --device intel:gpu --source match_720.mp4 --save out.mp4
# contest day: point --source at the live stream URL
```

### Get a fresh clip from a long YouTube VOD
```bash
yt-dlp -f 300 --js-runtimes node --force-ipv4 \
  --download-sections "*START-END" -o clip.mp4 "<youtube_url>"   # times as HH:MM:SS
```

### Retrain / improve the model (free GPU)
1. Upload a 720p+ clip (Colab: file picker; Kaggle: as a Dataset).
2. Run the notebook top-to-bottom (`colab/frc_train.ipynb` or `kaggle/frc_train_kaggle.ipynb`).
3. Download `frc_model_openvino.zip`, unzip into `models/`, run `live_openvino.py`.

---

## 9. Recommended Next Steps (in priority order)

1. **Fix robot detection** — the YOLO-World prompt yielded 0 labels. Options: tune the prompt /
   confidence, add a bumper-color (red/blue) detector as a second teacher, or hand-label a few
   robot frames. Re-train; robots are 6-per-match and large, so this should land easily.
2. **Add footage** — auto-label 2-3 more 720p+ matches to grow the dataset past ~300+ frames; lifts
   mAP and INT8 calibration quality.
3. **Active-learning round** — auto-mine frames the student misses (airborne, dense piles), correct,
   retrain → pushes mAP toward the high-90s.
4. **Lock the contest config** — benchmark CPU vs iGPU vs NPU, pick imgsz (640 for headroom at 60fps
   vs 768 for recall), and validate live-stream ingestion end-to-end.
5. **Geometry/scoring** (optional, ambitious) — AprilTag homography → field coordinates → 3D ball
   arcs → "did it score" calls, cross-checked against TBA.

---

## 10. Constraints to Always Respect

- **No Anthropic inference API** — classical CV + local models only (training uses open YOLO models,
  not API calls).
- **Live deployment must run on the user's Intel laptop** — no CUDA, ideally no internet at contest.
- Training is a one-time precompute on **free** cloud GPU (Colab/Kaggle T4); no paid compute assumed.
- Use **playoff** footage for demos (qualifier overlays were noted as unreliable).
