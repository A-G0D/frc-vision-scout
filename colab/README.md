# Run the FRC vision pipeline on a free Colab GPU

This is the GPU engine. Free Colab gives you a T4 — enough to run high-resolution detection +
SAHI + ByteTrack tracking at full frame rate, which fixes the recall and frame-rate problems
the CPU machine can't.

## What you do (≈4 clicks, ~10–15 min)
1. Go to **colab.research.google.com** → `File` → **Upload notebook** → pick
   `projects/frc-vision-scout/colab/frc_colab.ipynb` from this repo.
2. `Runtime` → `Change runtime type` → **T4 GPU** → Save.
3. `Runtime` → **Run all**.
4. It pulls the match clip from YouTube, runs GPU tracking, and **auto-downloads**
   `tracked_h264.mp4` (the annotated tracked video) + `sahi_frame.png` (max-recall sample).

Then drop `tracked_h264.mp4` (or just tell me the ball counts it printed) back into this chat and
I'll tune the next pass.

## What this first notebook delivers (no training yet)
- **GPU real-time**: every frame processed, output at the source frame rate (vs 5 fps on CPU).
- **High-res recall**: 720p input + imgsz 1920 → far/airborne balls become resolvable.
- **SAHI** sliced inference cell → maximum small-object recall on a sample frame.
- **ByteTrack**: persistent ball track IDs (uses low-confidence detections to bridge gaps).

## What comes next (I'll add cells once this baseline works)
- **Auto-label + custom-train** a FUEL (and robot) YOLO on this footage → the accuracy ceiling-raiser.
- **Robots** via YOLO-World / the custom model.
- **Eval harness**: mAP / MOTA / scoring-vs-TBA numbers so "level-10" is measured, not claimed.

## Honest notes
- I can't click your screen or log into your Google account (no screen-control tool is available
  to me), so you do the ~4 clicks above; everything else is automated in the notebook.
- Free Colab has session/time limits; this notebook is sized to finish well within them.
- If a cell errors, paste the error here — I edit the notebook and you re-upload/run.
