# FRC Vision Scout — Industrial-Grade Upgrade Plan

Goal: take the pipeline from a level-1/2 CPU demo (generic COCO model, single-pass inference,
hand-rolled IOU tracker, pixel ROIs) to a level-10/20 professional system: a **custom-trained
detector**, **SOTA tracking**, **GPU real-time throughput**, **true field geometry**, and a
**measured evaluation loop**.

Each category lists 5 upgrades: **what · why it's a level-up · how · dependency.**

---

## A. Ball Detection (the #1 gap — ~50% floor / ~20% air recall today)

1. **Custom-trained FUEL detector** — *the single biggest lever.* Fine-tune YOLO11 on this
   exact foam ball instead of COCO "sports ball." A model that has seen thousands of FUEL
   instances (on carpet, airborne, clustered, blurred) will hit 90%+ where the generic class
   plateaus. *How:* auto-label frames with the current model → human/Claude-in-the-loop
   correction → fine-tune. *Dep:* GPU + labeled data.
2. **SAHI sliced inference** — tile each frame into overlapping crops, detect per tile at full
   resolution, merge with NMS. This is *the* technique for small objects in large frames and
   directly fixes far/tiny and airborne balls a single downscaled pass misses. *How:* `sahi`
   package over the YOLO model. *Dep:* GPU for speed (else slow but works).
3. **Motion-fused airborne recovery** — airborne balls are fast, blurred, low-confidence for a
   static detector. Add an optical-flow / frame-difference channel; fuse fast small moving
   blobs with detections so flying balls aren't dropped. *Dep:* none (classical + detector).
4. **Temporal interpolation / track-back** — a ball missed in one frame is recovered from its
   track (forward/backward prediction). Lifts effective recall and smooths counts. *Dep:* the
   new tracker (Cat C).
5. **Regime-specific tuning + image enhancement** — separate confidence/size gates for tiny vs
   large balls; motion-deblur/sharpen pass for airborne blur; test-time augmentation + multi-
   scale. *Dep:* none.

## B. Robot Detection, ID & State

1. **Custom-trained robot detector** — COCO has no "robot" class; train one on red/blue
   bumpered robots so they stop being confused with HUBs/crowd. *Dep:* GPU + labels.
2. **Team-number identity** — bumper-number OCR (cropped + super-resolved) to tag tracks with
   the actual team number, not just alliance color.
3. **YOLO-World bootstrap labeling** — open-vocab ("robot") to auto-seed the robot training set
   before the custom model exists. *Dep:* CLIP weights.
4. **Orientation / intake-direction estimation** — know which way a robot faces (keypoint or
   oriented-box model) so pickup/shot events can be attributed correctly.
5. **Per-robot state classifier** — has-ball / shooting / climbing / disabled, for real
   scouting metrics. *Dep:* GPU + labels.

## C. Multi-Object Tracking

1. **SOTA tracker (ByteTrack / BoT-SORT / OC-SORT)** — replace the hand-rolled IOU matcher.
   ByteTrack uses *low-confidence* detections to bridge occlusion — exactly our failure mode.
   *How:* built into ultralytics (`model.track(..., tracker="bytetrack.yaml")`).
2. **Appearance Re-ID embeddings** — a small embedding per track so IDs survive occlusion and
   camera cuts (fixes the "12 IDs for 6 robots" problem). *Dep:* GPU helps.
3. **Physics-based ball tracking** — constant-acceleration/parabolic motion model for airborne
   arcs; predict landing and trajectory; distinguishes a shot from a roll.
4. **Track lifecycle + gap interpolation** — confirm-before-spawn, coast-through-occlusion,
   interpolate missed frames (also boosts output frame rate, Cat D).
5. **Object-type-aware association** — separate gating/motion models for balls vs robots vs
   HUBs; class-consistent IDs.

## D. Performance & Real-Time (the low-frame-rate complaint)

1. **GPU / CUDA inference** — *the core fix.* CPU is the entire reason for 5fps and ~10s/frame.
   A mid GPU runs YOLO11x at 30–100+ fps. *Dep:* CUDA GPU.
2. **TensorRT / ONNX export + FP16/INT8** — 2–5× more throughput by compiling the model for the
   target hardware. *Dep:* GPU.
3. **Async batched pipeline** — decode, infer, track on separate workers; batch frames to the
   GPU; ffmpeg hardware decode. Saturates the GPU instead of serializing.
4. **Detect-every-Nth + tracker fill** — infer on keyframes, let the tracker interpolate the
   rest → smooth 30fps output at a fraction of the inference cost.
5. **Model-tier switch** — yolo11s/m for real-time, yolo11x (or the custom model) for offline
   max-accuracy; one flag. Render output at full **30 fps**, not 5.

## E. Field Geometry & 3D

1. **AprilTag field calibration** — detect the field's 36h11 tags → homography to top-down
   field coordinates, so positions/scoring live in meters, not pixels, and survive camera moves.
2. **Camera intrinsics + undistortion** — calibrate lens distortion for accurate geometry.
3. **3D ball-trajectory reconstruction** — recover arc height & landing point from the 2D track
   + a physics prior; enables real "did it go in the HUB" judgments.
4. **HUB/zones in field coordinates** — define scoring volumes in field space (robust to camera
   pan/zoom) instead of fixed pixel boxes.
5. **Multi-feed fusion** — combine the wide cam with any other available angle to beat occlusion
   and triangulate 3D. *Dep:* extra feeds.

## F. Data, Training & Evaluation (MLOps — what makes it *stay* good)

1. **Labeling pipeline** — auto-label with current detectors → fast review/correction UI →
   export YOLO-format dataset. The flywheel for every custom model above.
2. **Versioned dataset + splits** — train/val/test, held-out matches, reproducible.
3. **Training + HPO pipeline** — fine-tune scripts, augmentation, hyperparameter search,
   checkpoints. *Dep:* GPU.
4. **Real evaluation harness** — detection mAP / precision / recall vs labeled frames; tracking
   MOTA / IDF1; scoring vs TBA. Replaces "looks better" with numbers. This is how we *prove*
   level-10.
5. **Active-learning loop** — auto-mine the frames where the model fails (airborne, dense piles,
   occlusion), label those, retrain. Targeted recall gains instead of random labeling.

---

## The hard dependency: a GPU + a labeled dataset

Three of the highest-leverage upgrades (custom FUEL detector, custom robot detector, real-time
throughput) **require a CUDA GPU**. This machine is CPU-only — it's why everything is slow and
why we're stuck on a generic model. I can build *all* the software here (SAHI wrapper, tracker
integration, labeling + training + eval pipelines, geometry) and it will run on CPU for
correctness; but **real-time speed and the custom-model accuracy jump are gated on GPU access**.

## Recommended sequence (highest leverage first)
1. **D1 GPU** + **C1 ByteTrack** — instantly fixes frame rate and tracking with no training.
2. **A2 SAHI** + **A3 motion-fusion** — big ball-recall jump before any training.
3. **F1 labeling** → **A1 custom FUEL model** + **B1 custom robot model** — the accuracy ceiling-raiser.
4. **F4 eval harness** — lock in measured mAP / MOTA / scoring-vs-TBA so "level-10" is provable.
5. **E AprilTag geometry** + **C3 physics trajectories** — true 3D scoring & shot attribution.
