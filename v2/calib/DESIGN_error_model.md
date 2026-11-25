# TBA-Trained Error-Correction Model — Design Document

## Problem Statement

The vision pipeline produces per-component score estimates (fuel points, tower points,
foul points, total score) for each alliance per match.  The Blue Alliance (TBA) provides
ground-truth values post-match.  Systematic vision bias — e.g., consistently
under-counting fuel — is learnable from paired (estimate, truth) samples and should be
corrected before downstream scouting decisions are made.

---

## 1. Data Collection

### 1.1 Pairing strategy

For each match where both a vision estimate and a TBA result are available, we record
one row per (component, alliance):

```
match_key     : str            # e.g. "2026miket_qm12"
comp_level    : str            # "qm" | "sf" | "f"
alliance      : str            # "blue" | "red"
component     : str            # see §1.2
vision_est    : float          # raw pipeline output
tba_actual    : float          # TBA ground truth for that component
residual      : float          # tba_actual - vision_est  (positive = vision underestimates)
timestamp_utc : str | None     # ISO-8601; helps detect seasonal drift
event_key     : str            # e.g. "2026miket"
```

### 1.2 Components tracked

Only components that TBA records with authoritative fidelity are usable:

| Component key         | TBA field(s)                        | Notes                              |
|-----------------------|-------------------------------------|------------------------------------|
| `total_score`         | `blue_score` / `red_score`          | Most reliable; always present      |
| `fuel_total`          | `blue_fuel` / `red_fuel`            | TBA stores total fuel count        |
| `foul_points`         | `blue_foul_count` × 5/15 etc.       | May need breakdownV2 API endpoint  |
| `tower_points`        | `score_breakdown.towerPoints`       | breakdownV2 only; may be null      |

**Not collected** (TBA has no authoritative record): auto/teleop fuel split, shot
accuracy, cycle time, intake rates, per-robot anything.  Do not fabricate ground-truth
for these.

### 1.3 TBA lookup location

TBA lookup is performed in `calib/collect.py` (not the live scout path) using the
existing TBA client already called by `eval/tba_verify.py`.  The collection step is
offline — run it once per event after TBA publishes final scores.

Data is written to `calib/data/paired_<event_key>.jsonl` (one JSON object per line,
schema from §1.1).  A merged view `calib/data/all_pairs.jsonl` is built by
`calib/collect.py aggregate`.

### 1.4 Storage format

JSONL (newline-delimited JSON) for streaming append and easy version control diffs.
One record = one (match, alliance, component) triple.  File is append-safe: re-running
the collector for the same event upserts by `(match_key, alliance, component)`.

---

## 2. The Model

### 2.1 Baseline: per-component affine fit

For each component `c`, fit:

```
tba_actual_c = a_c * vision_est_c + b_c
```

by ordinary least squares (OLS).  `a_c` is the scale factor; `b_c` is a constant
offset.  This is interpretable: if the pipeline systematically misses 30% of fuel,
`a_c ≈ 1.43` and `b_c ≈ 0`.  If the pipeline always adds a phantom 10 points,
`b_c ≈ -10`.

OLS is solved in closed form via `numpy.linalg.lstsq` — no optimization, no
hyperparameters.

**Robust variant** (preferred when outliers are present): use Huber regression or
iteratively re-weighted least squares (IRLS) with Huber weights, so that a match with
a catastrophic vision failure (robot disabled, camera occlusion) doesn't pull the
intercept.  Implementation: `sklearn.linear_model.HuberRegressor` with
`epsilon=1.35` (standard 95%-efficiency setting).

### 2.2 Why not overfit on few matches?

A per-component affine model has exactly 2 parameters (`a`, `b`).  With as few as
8 matched pairs the fit is already over-determined.  The fit generalizes because:

1. The bias is *systematic* (same camera, same field layout, same game) — not
   match-specific noise.
2. We fit only 2 degrees of freedom per component regardless of how many matches we
   have.
3. We guard with a minimum-data threshold (see §3.3): below `MIN_MATCHES_PER_COMPONENT`
   we return the identity calibration (`a=1, b=0`) rather than a noisy fit.
4. We do NOT use match-level features (robot counts, opponent strength) in the baseline
   model.  Adding features risks overfitting on the training set and makes the model
   event-specific rather than transferable.

### 2.3 Extension path: feature-augmented per-component regression

When ≥30 paired matches are available, the model can optionally condition on:

- `phase_ratio` = fraction of match in auto (proxy for score-composition): already
  computable from the overlay timeseries.
- `crowd_density_est` (future): if the blob detector returns a crowd-occlusion score,
  use it as a regressor to adjust the fuel undercount under heavy occlusion.
- `comp_level` as a categorical (playoffs may have different score distributions).

Use ridge regression (L2 penalty) in this regime to prevent the additional features
from overfitting.  Always start the pipeline using the baseline affine — the
feature-augmented model is opt-in via `ErrorCorrectionModel(mode="ridge")`.

---

## 3. Training and Evaluation

### 3.1 Train/holdout split

Split **by match** (not by row), using the last 20% of matches (sorted by
`match_key` lexicographically, which is approximately chronological within an event).
Never split randomly across alliances of the same match — that would leak information.

```
holdout = last ceil(0.2 * n_matches) unique match keys
train   = all other match keys
```

### 3.2 Metrics reported

For each component, report before/after calibration on the holdout set:

| Metric                  | Definition                                         |
|-------------------------|----------------------------------------------------|
| `mae_raw`               | Mean absolute error of raw vision estimate vs TBA  |
| `mae_cal`               | MAE after applying calibration                     |
| `mae_reduction_pct`     | (mae_raw - mae_cal) / mae_raw * 100                |
| `rmse_raw` / `rmse_cal` | Root mean squared error, same split                |
| `bias_raw`              | mean(vision - tba) — positive = overestimate       |
| `bias_cal`              | Should be near zero after calibration              |

A calibration is considered useful only if `mae_reduction_pct > 5%` on the holdout
set AND the holdout set has at least `MIN_HOLDOUT` pairs.

### 3.3 Minimum-data thresholds

| Threshold constant         | Value | Effect if not met                      |
|----------------------------|-------|-----------------------------------------|
| `MIN_MATCHES_TOTAL`        | 8     | Refuse to fit; return identity calib    |
| `MIN_MATCHES_PER_COMPONENT`| 6     | That component gets identity calib only |
| `MIN_HOLDOUT`              | 3     | Skip holdout eval; report "insufficient"|

These are conservative: they ensure the 2-parameter fit has at least 3× overdetermination.

---

## 4. Live Application

### 4.1 Apply calibration

Given a new vision estimate `x` for component `c`:

```python
calibrated = a_c * x + b_c
```

where `(a_c, b_c)` are loaded from the most recently saved calibration artifact.

### 4.2 Keep both raw and calibrated

The output dict always carries both values:

```json
{
  "total_score_est_raw": 142.0,
  "total_score_est_cal": 198.4,
  "total_score_cal_ci_lo": 181.2,
  "total_score_cal_ci_hi": 215.6,
  "cal_component": "total_score",
  "cal_version": "2026miket-v1"
}
```

Never overwrite or discard the raw estimate — downstream analysts may want to diagnose
the pipeline, not just consume the corrected output.

### 4.3 Residual-based confidence interval

After calibration is fit, compute the standard deviation of residuals on the
**training** set: `sigma_c = std(tba_actual - calibrated_train)`.

For a new prediction, attach a symmetric ±1.96σ interval (95% Gaussian approximation):

```
ci_lo = calibrated - 1.96 * sigma_c
ci_hi = calibrated + 1.96 * sigma_c
```

This is an *empirical* CI — it reflects observed pipeline scatter, not formal
Bayesian uncertainty.  It is honest: if only 8 training pairs were used, `sigma_c` is
noisy and the CI is wide.

---

## 5. Honest Failure Modes

| Failure mode                            | Detection                              | Mitigation                               |
|-----------------------------------------|----------------------------------------|------------------------------------------|
| Fewer than MIN_MATCHES_TOTAL pairs      | Count on load                          | Return identity (a=1, b=0); log warning  |
| Component always zero in vision         | `std(vision_est) == 0`                 | Skip fit for that component              |
| TBA API returns null breakdown          | `tba_actual is None`                   | Drop row; log; may hit min-data guard    |
| Calibration inverts ordering (a_c < 0) | Check sign post-fit                    | Clamp `a_c = max(a_c, 0.1)`; warn loudly|
| Seasonal/game-patch drift               | Compare residuals on newest 25% of data| Expose `--event-window` to restrict fit  |
| Different scoring game vs training game | `event_key` prefix mismatch            | Require same year prefix; warn on mismatch|
| Overly wide CI from small sigma sample  | CI width > 2× component range          | Report CI as "unreliable" in output dict |

### 5.1 What this model cannot do

- It cannot correct for *random* per-match vision failures (occlusion, arena lighting
  changes) — only systematic bias.
- It cannot estimate tower points or foul points until the vision pipeline produces
  them as outputs; calibrating a zero series is meaningless.
- It does not produce per-team calibration — only per-component across all matches.
  Per-team correction would require many more samples per team than a single event provides.

---

## 6. File Map

```
calib/
  DESIGN_error_model.md     # this document
  error_model.py            # importable ErrorCorrectionModel class
  collect.py                # (TODO) offline TBA pairing + JSONL writer
  data/
    paired_<event_key>.jsonl
    all_pairs.jsonl
  artifacts/
    calibration_<event_key>_<version>.json   # serialized (a, b, sigma) per component
```
