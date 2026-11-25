# FRC Vision Scout — Cutting-Edge Pipeline Architecture

Target: maximal accuracy + real-time speed on a **free Colab T4**. Every stage is the current
SOTA-practical choice for that hardware. Honest accuracy targets are stated per metric — not a
blanket "99%."

## Hardware map
- **Colab T4 (16GB)** = the GPU engine: auto-labeling, training, TensorRT inference, tracking.
- **This CPU box / GitHub Codespaces** = orchestration, eval, geometry, the repo. (Free
  Codespaces is CPU-only — no GPU — so it does *not* accelerate models; it's a dev/CI box.)
- **Google Drive** = the bridge between stages (datasets, weights, outputs persist across
  Colab sessions so you can resume).

## The data engine: auto-label → train → distill (this is what unlocks high accuracy)
A generic COCO "sports ball" model is the ceiling we're stuck at. The leap is a **model trained
on THIS ball/field**, created without manual labeling:

1. **Teacher ensemble auto-labeling.** Run heavy models over sampled frames to produce
   pseudo-labels:
   - FUEL: **YOLO11x + SAHI** (sliced tiling) at high res → catches small/airborne/dense balls.
   - Robots: **YOLO-World / Grounding-DINO** (open-vocab, prompt "robot") + bumper-color prior.
   - **Ensemble agreement filter**: keep boxes multiple teachers/scales agree on → clean labels.
2. **Train a specialist student.** Fine-tune **YOLO11** (or **RT-DETRv2**, transformer SOTA
   real-time) on the pseudo-labels with strong augmentation (mosaic, copy-paste, blur for
   motion). Student specialized on FUEL/robots ≫ generic COCO.
3. **(To reach ~99%) one human-correction round.** Auto-labels cap the student at the teacher's
   quality (~88-93%). A single fast review pass (correct the worst frames, surfaced by
   active-learning) lifts mAP toward 97-99%. Built as a near-zero-effort review notebook.
4. **Active-learning loop.** Auto-mine frames where the model is uncertain or detectors disagree
   (airborne, dense piles, occlusion), correct just those, retrain. Targeted gains.

## Inference + tracking (accuracy)
- **SAHI sliced inference** at deploy for max small-object recall (airborne/far/tiny).
- **BoT-SORT + ReID** (appearance embeddings) for persistent IDs through occlusion/cuts —
  fixes "12 IDs for 6 robots." ByteTrack as the fast fallback.
- **Track interpolation / physics ball model** — parabolic motion for airborne arcs; fill
  missed frames (also boosts effective frame rate).
- **Temporal voting** for counts/identities (a detection missed in 1 frame is recovered).

## Speed (real-time on T4)
- **TensorRT FP16 export** of the student → 2-5× throughput on T4.
- **Distill to YOLO11s/m** for real-time; keep YOLO11x/RT-DETR as the offline-max + teacher.
- **Detect-every-Nth + tracker fill** → smooth 30fps output at a fraction of inference cost.
- **Batched, half-precision, stream decode.** Target: **30+ fps at 720p on T4** with the
  distilled+TensorRT model.

## Geometry & scoring (the "game simulation")
- **AprilTag (36h11) field homography** → top-down field coordinates (positions in meters).
- **3D ball-arc reconstruction** from 2D track + physics → real "did it enter the HUB" calls.
- **Scoring zones in field coords**; score reconstruction **cross-checked against the overlay
  counter + TBA** (already exact on quals).

## Honest accuracy targets (measured, per metric)
| Metric | Realistic target | Ceiling note |
|---|---|---|
| Frame class (live/info/filler) | ~99-100% | already 100% on labeled set |
| Match teams + scores vs TBA | 100% | proven on 16/16 quals |
| FUEL detection mAP (visible) | 90-95% auto, **97-99% with 1 correction round** | dense-pile occlusion is a hard cap on *count* |
| Robot detection mAP | 90-97% custom-trained | |
| Robot track ID persistence | high (static cam + ReID) | camera cuts are the stressor |
| Per-robot point attribution | measured agreement only | **fundamentally ambiguous** — no model fixes concurrent scoring |
| Throughput | 30+ fps @720p (TensorRT/distilled) | T4-bound |

## Notebooks (staged, Drive-checkpointed so you can resume)
- `colab/frc_colab.ipynb` — **baseline** (no training): GPU + high-res YOLO + SAHI + ByteTrack.
- `colab/frc_train.ipynb` — **the engine**: auto-label → train student → TensorRT export →
  BoT-SORT inference → eval. (This doc's stages 1-3 + inference + speed.)
- *next:* `frc_geometry.ipynb` (AprilTag homography + 3D arcs), `frc_active_learning.ipynb`
  (review/correct/retrain loop to push mAP to ~99%).
