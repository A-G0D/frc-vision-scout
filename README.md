# FRC Vision Scout

## What it is
A computer-vision engine that watches FIRST Robotics Competition (FRC REBUILT 2026)
livestreams and tries to auto-extract match and scouting data: match segmentation,
team/score reading from the broadcast overlay, and robot/ball detection on the field.
Two implementations live in this repo: `v1` (root `src/`) and `v2` (`v2/`, a from-scratch
rewrite). They are separate lineages, not a shared pipeline.

## Why it exists
Manual FRC scouting and scorekeeping is slow and error-prone. The goal is to read the
broadcast overlay (teams, score, FUEL count) and track robots/balls on the field well
enough to replace manual scouting, running live on the author's own laptop with no cloud
inference and no CUDA (Intel iGPU/NPU target).

## State: prototype, not production ready
Neither v1 nor v2 is a working end-to-end product yet.

- **v1** proved feasibility on one real event stream: 90.7% mAP ball detector, 31fps live
  on an Intel iGPU, and 100% match-ID agreement against The Blue Alliance on 16/16
  qualification matches from that one stream. But robot detection never worked (YOLO-World
  produced 0 robot labels), and per-robot scoring attribution (which robot scored) was never
  solved.
- **v2** is a ground-up rewrite: real configs, eval gates, a cloud-training app, and a test
  suite that collects and runs cleanly. But per v2/README.md, the per-module algorithm
  bodies are documented as TODO scaffolding, and the field/AprilTag geometry coordinates in
  `v2/configs/field_2026_rebuilt.yaml` are explicitly placeholders, not measured against the
  real field.
- All validated results come from one historical event (2026ctwat, NE Waterbury). Nothing
  has been run against a live or unseen stream.
- Root video assets referenced in some docs (e.g. `match_720.mp4`) are not present in this
  working tree, so any "quick start" that names them cannot be reproduced as written.

## How to run it
**v1**, from repo root (requires `yt-dlp`, `ffmpeg`, `opencv-python`, `numpy`):
```bash
python src/scan_timeline.py data/ngvsHrmoW8U_144p.mp4
```
This needs the source video downloaded first (`src/youtube_extractor.py`); the file is not
included in this repo (see Known gaps). OCR-based overlay reading additionally needs
`easyocr` installed and is only partly built.

**v2**, from `v2/` (requires the deps in `v2/pyproject.toml`, notably `torch`, `ultralytics`,
`openvino`, `pytest`):
```bash
cd v2
pip install -e .
PYTHONPATH=src python scripts/run_scout.py --sync --source data/footage/qual56.mp4 \
  --out-dir data/scout_out --match qual56
```
This also needs source footage placed at the given path; none ships in this repo.

Tests (v2, from `v2/`):
```bash
python -m pytest -q
```
Result as of this remediation: all collected tests pass, 1 skipped, 0 failed. (The exact
test count line was not capturable from this shell session due to output truncation on a
long pytest run; the audit's own record states 543 passed / 1 skipped on a prior run, and
this remediation's re-run showed 0 failures and 1 skip, consistent with that.)

## Layout
- `src/`: v1: broadcast-overlay match extraction (`scan_timeline.py`, `overlay_detect.py`,
  `match_list.py`, `verify_tba.py`) plus classical/YOLO ball and robot detection
  (`detect_fuel*.py`, `detect_robots.py`, `pipeline*.py`) and the OpenVINO live deploy path
  (`live_openvino.py`).
- `v2/src/frcscout/`: v2 package: `io/` (config, types), `pipeline/` (orchestrator,
  scheduler), `detect/`, `track/`, `geometry/` (AprilTag pose, field mapping), `fuse/`
  (event/attribution logic), `eval/` (gates, metrics, TBA verification).
- `v2/configs/`: all tunables (timing in seconds/meters, not frame counts); includes the
  field geometry file that is still a placeholder.
- `v2/tests/`: the v2 test suite.
- `colab/`, `kaggle/`: cloud training notebooks (teacher-student distillation to OpenVINO).
- `PROJECT_SUMMARY.md`, `ARCHITECTURE.md`, `UPGRADE_PLAN.md`: v1 results and gap writeups.
- `v2/README.md`, `v2/RESULTS.md`, `v2/AUTONOMOUS.md`: v2 design rationale and results.

## Known gaps
- Robot detection never worked in v1 (0 labels from YOLO-World); v2's proposed fix
  (bumper-color / robot_net) has real code but untested field geometry underneath it.
- Per-robot scoring attribution is explicitly unsolved in both v1 and v2 (v2/RESULTS.md
  lists it as n/a; attribution is alliance-level, not per-robot).
- v2's per-module algorithm bodies are documented as TODO scaffolding despite a passing
  test suite — passing tests do not mean the detection/geometry logic is finished.
- AprilTag field pose is not visible on broadcast footage, so real-world geometry is
  image-approximate only.
- No validation against a live stream or a different contest event; everything is from one
  historical event (2026ctwat). Within that event, `v2/scripts/eval_held_out.py` now runs
  the full detect->track->phase pipeline against `final1` (the match `configs/sources.yaml`
  holds out, never trained on) and measures it against the validated `qual56` baseline —
  see `v2/README.md` "Held-out generalization gate" for the numbers. That is a held-out
  MATCH check, not a held-out EVENT/venue/camera check, so it does not close this gap on
  its own.
- Root video assets some docs reference are missing from this working tree.

## Score and timer reader (v2, CPU only)

`v2/src/frcscout/detect/digit_ocr.py` reads the score and match-timer boxes of the overlay with numpy and OpenCV only: Otsu threshold, connected components, template match against Hershey-font digits. The existing easyocr path in `overlay_ocr.py` needs easyocr or tesseract, neither of which installs in the CPU-only setup this was measured on.

```
cd v2
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt   # once
.venv/bin/python -m pytest -q
.venv/bin/python scripts/bench_overlay_ocr.py 200 --out results/overlay_ocr.json
```

```
frames 200   fps 88
score exact  1.00 (clean 1.00, degraded 1.00)
timer exact  0.96 (clean 1.00, degraded 0.91, held-out font 0.97)
```

The bench draws 640x360 frames with a blue-score box and a timer box at the `_LIVE_BAR_ROIS` positions, in three Hershey fonts, with brightness jitter, noise, blur and JPEG. Level 0 is clean; level 1 is degraded. Triplex is not in the template set. Command above, machine `macOS arm64, cv2 5.0.0`, one core under `nice`.

### Results

| Metric | Ours (synthetic) | Market standard | Goal |
|---|---|---|---|
| Score exact match | 1.00 on synthetic Hershey frames only; the templates use the same fonts, so this is not a real-footage result (first template version: 0.88) | No published digit-accuracy figure for broadcast scoreboards found | 1.5x fewer errors than a published figure, once one exists |
| Timer exact match | 0.96 (first version: 0.46; synthetic, same caveat) | same | same |
| Frames/s, two crops | about 88 (full OverlayReader.read per frame, cv2 5.1.0) | none found | not set |

Sources checked for a standard: [TBA webhook docs](https://thebluealliance.com/apidocs/webhooks) describe match_score pushes but give no latency number; [TBA tech talk](https://blog.thebluealliance.com/2017/11/10/tech-talk-efficiently-querying-the-tba-api/) shows a 61 s cache header as an example; the Tesseract/EasyOCR/PaddleOCR comparisons found ([example](https://dev.to/felipe_anyformat/best-python-ocr-libraries-179)) are small self-run tests on documents, not scoreboards. None is comparable, so no goal multiple is claimed.

### Limits

- Synthetic data only. Templates and test frames both come from Hershey fonts, so the numbers are optimistic. No real broadcast frame is in the repo and none was committed.
- Degraded timer accuracy is 0.91. Typical misses: a "1" merging with the colon halo is read as "2", and 5/6 confusion.
- The ROI coordinates are placeholders for the real REBUILT overlay; `OverlayReader({"ocr": "template"})` uses the new reader for score and timer only (fuel, phase and teams still need easyocr and are left unset), and the bench measures through it.
- v1 and v2 are still not an end-to-end product (see above).
- `metrics.aggregator` (MatchRecordBuilder, referenced by `v2/live/run_live.py` and the build log) was real but never committed: an old generated `.gitignore` line excluded `v2/metrics/`. No source survives in this clone, so its test was deleted and `run_live.py`'s aggregator path is broken until it is rewritten.
- Tests need Python 3.10 or newer; the Quickstart above is the exact command.

## License

MIT, see `LICENSE`.
