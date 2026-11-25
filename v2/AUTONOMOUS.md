# AUTONOMOUS PRODUCTION BUILD — Master Control File

> Single source of truth for the unattended overnight build. Any fresh session
> (incl. the 4h cron restart) MUST read this top-to-bottom, then continue from
> the **NEXT-ACTIONS QUEUE**. Append every work cycle to `autobuild_log.md`.
> Authorized by the user 2026-06-26: *"FULLY functional task when I wake up.
> Don't stop until I tell you to. Keep finding issues and getting other models to
> work on different parts. No more questions. No waiting."*

## NORTH-STAR DELIVERABLE
A **production script that ingests a YouTube livestream URL and LIVE-updates an
online repository with dozens-to-hundreds of VERIFIED per-match FRC metrics**,
with a **TBA-trained error-correction model** that auto-scales raw vision
estimates to match real alliance score breakdowns.

End-to-end flow:
```
YT livestream URL
  → live frame ingest (yt-dlp/streamlink + ffmpeg, real-time, auto-reconnect)
  → per-frame vision (ScoutPipeline: 6 robots w/ bumper-bar anchor+alliance+phase,
       balls spread+pile+airborne shots, HUB activity, overlay OCR)
  → match segmentation (detect match start/end from overlay timer + score reset)
  → match identification (overlay event/match OCR → TBA lookup)
  → metric aggregation (the CATALOG: dozens-hundreds of community-standard metrics)
  → ERROR CORRECTION (compare vision breakdown vs TBA actual → per-breakdown scale
       factors → trained calibration model → apply live)
  → LIVE PUBLISH (incremental git push / hosted JSON+dashboard, updated every cycle)
```

## HARD RULES (never violate)
1. **No fabrication.** Every metric is measured; gated metrics carry {value, available, needs}.
2. **Verify everything, live.** Cross-check against TBA (`2026ctwat` + auto-located matches).
   Adversarially verify each component with a DIFFERENT model + non-leading prompt.
3. **Generalize.** Hold on ≥3 of the 6 clips incl. held-out `final1`.
4. **Don't break what works.** Run `pytest` before declaring done (~543+ tests must stay green).
5. **Repo hygiene.** `~/Claude` git HANGS — do NOT git commit there; use file backups.
   Set `BOSSMAN_NO_AUTOPUSH=1` on spawned children. Rotate any pasted secrets; never commit them.

## CURRENT STATE (updated each cycle — see autobuild_log.md for detail)
- **Vision core (DONE):** ScoutPipeline (bit-identical refactor), balls (crowd-clean + pile area-est
  + airborne SHOT), phases (intake/hold over-fire fixed), HUB funnels aligned, edge-pose clamp.
- **Robot detection (NEW, this session):** bumper-bar **anchor + alliance dedup** collapses YOLO
  pile-ups (max box-IoU 0.69) into a clean **6 tracked IDs** (qual59 200f: mean 4.5 live/frame,
  max 6). Scripts: `scripts/robot_anchor.py`, validated. Camera is STATIC on qual59+match_1080.
- **Alliance color (IMPROVED):** `RobotNet.bumper_alliance` rewritten — bumper-RING sample + LAB a*/b*
  + abstain-on-ambiguity. Measured 168 boxes: fixed the blue-HUB→red error, blue/red now track field side.
- **Kaggle retrain (RUNNING, do not stop):** `frc-robotrain-v2`, epoch ~28/32 (10.5h-capped),
  ETA tonight. NOTE: 2024 Crescendo data = generic base, NOT this broadcast domain; treat as a
  base only. The user clarified the view is a **tilted SIDELINE** angle (not overhead) — robots
  ARE detectable; the problem was instance-separation, now addressed by the bumper-bar anchor.
- **Orchestrator:** `matrix_org/orchestrator/{run.mjs,skills.json}` (CLI fan-out: claude -p + codex exec).
- **Rubric:** `matrix_org/RUBRIC.md` (offline analyzer, ≈52/100 baseline). **Extended for production
  in `matrix_org/RUBRIC_v2.md` (this build).**

## PRODUCTION GAP (what's missing for the north-star) — drives the queue
- [ ] Robust LIVE YT-livestream ingest (current `live_app.py` is file-oriented; needs true live + reconnect).
- [ ] Match segmentation (auto match start/end) + identification (which match on TBA).
- [ ] METRIC CATALOG — research + implement the dozens-hundreds of community metrics (TBA/Statbotics/OPR…).
- [ ] Metric aggregator: vision events → the full catalog per match.
- [ ] TBA error-correction MODEL: collect (vision, TBA) pairs → per-breakdown scaling → trained calibrator → live apply.
- [ ] Live repo publisher (incremental, online, verified).
- [ ] Production verification harness: every metric verified vs an oracle, release gate, no-fabrication audit.

## DONE + VERIFIED (with tests)
- D2 segmenter `live/match_segment.py` (5 tests) · D4 error model `calib/error_model.py` (7 tests) ·
  D5 publisher `publish/repo_writer.py` (self-test) · catalog `metrics/CATALOG.md` (166 metrics).
- **Ground truth** `metrics/SCHEMA_VERIFIED.md` (real TBA schema from `data/tba/2026ctwat_matches.json`).
- TBA client `src/frcscout/integrations/tba_client.py` + live ingest `live/ingest.py` (69 tests pass).
- `metrics/SCORING_2026.md` (manual-cited, wave-2). pytest config fixed → **FULL SUITE GREEN**.

## ✅ FUNCTIONAL DELIVERABLE (validated 2026-06-27)
`live/run_live.py` runs the WHOLE chain end-to-end and PUBLISHES a verified record:
file/URL → ScoutPipeline (HUD reader fixed to load from any cwd) → OverlayObs → MatchSegmenter
(engages: IN_MATCH, real timer) → MetricsAggregator (vision fuel + honest gated tower/foul) ⨝ TBA
actuals (TBAClient offline from data/tba) → error_model calibration → LivePublisher (matches/*.json +
index + dashboard). PROOF: on qual59 the published BLUE fuel_auto vision=26 == TBA hubScore.autoCount=26
(0.0% error). No fabrication. CLI: `PYTHONPATH=src python live/run_live.py --source <file|yturl>
--event-key 2026ctwat --match-key 2026ctwat_qm59 [--start N --frames M] --store <dir>`.
A full-match run (store data/live_store_full) is in progress for the COMPLETE record.

## MEASURED ACCURACY (qual59 vs TBA, 2026-06-27 — the verified-metric deliverable working)
- SCORE vs TBA totalPoints (headline, via aggregator `score_verification`): **BLUE 92=92 (0.0%)**,
  **RED 593 vs 613 (3.26%)**. Both <3.3% — broadcast-read score verified against TBA.
- FUEL: **BLUE exact** (auto 26=26, teleop 66=66, total 92=92 after the robust-monotonic noise fix);
  **RED capped at 100** (overlay RP-progress counter saturates; true 518 unrepresentable — hard limit,
  not bias; needs CV ball-count or results-screen). Saved series: data/live_store_full2/overlay_timeseries.json.

## SCORE: RUBRIC_v2 ≈ 84/150 (was ≈62). D-block 30/50 (D1 5, D2 5, D3 8, D4 5, D5 4, D6 3).
Release gate PASSES end-to-end; D6 done. Full scorecard + evidence in autobuild_log.md cycle 7.

## ROBOT DETECTION (the user's hot-button) — measured state
Ground truth: data/robot_gt/qual59_diag_labels.json (24 frames wired; 50 labeled). Eval: scripts/eval_robots.py.
Baseline raw net F1 0.25/P0.30. **Integrated `src/frcscout/detect/robot_filter.py` (in ScoutPipeline):
F1 0.44, precision 0.61, alliance 0.70** — render-verified (tight boxes on real robots, person-FP gone).
Codex fleet built it (net_smart F1 0.51/P0.90 was qual59-overfit; shipped the generalizable subset).
RUNNING: codex static-structure-mask agent (data/scout_out/codex/staticmask.json) to push P 0.61->~0.9
generalizably (mask fixed red/blue structures built FROM the video). Crowd-SHOT FPs: diagnosed (is_flight
latches + coasts + arc_region overlaps stands), NOT threshold-tunable — needs SHOT ground-truth + structural fix.
Perf: gated 2.2->11.9 fps (sync_performance flag).

## NEXT-ACTIONS QUEUE (highest value first — pop from top)
0. **Consume static-mask agent** → if precision up + suite green + generalizable, integrate the mask-build
   into ScoutPipeline init (build from first N frames of the source). Re-render qual59 + match_1080.
1. **Results-screen final-score capture**: at match end the broadcast shows the qual_results screen with
   the FINAL score (OverlayReader already classifies `qual_results` layout). Capture it in run_live so the
   final score closes the red 593→613 gap (live-bar goes unreadable in the last ~20pts). Highest-value
   accuracy win remaining.
2. **score_verification regression test** (lock the headline metric: build a record from synthetic vision
   + a small real TBA match fixture; assert overlay_score-vs-TBA-total comparison present + correct).
3. **CV ball-counting** for true cumulative fuel (genuinely hard CV; overlay fuel caps at 100 — this is the
   only path to red teleop fuel independent of the score). Scope a first version; measure vs TBA hubScore.
4. **A-C analyzer re-measure** (the 100-pt block is stale at ~54; re-run the offline analyzer evals to
   refresh A1-A7/B/C with current numbers, esp. after the bumper-bar anchor + alliance-color + phase fixes).
5. Strengthen `tests/test_run_live.py` (passes even with 0 scoring — assert the comparison/score block on a
   longer/active window via --start).
6. Fold robot bumper-bar anchor into ScoutPipeline; use the robot KF for velocity (chart had jitter spikes).
7. Pull Kaggle frc-robotrain-v2 when done; measure vs current; adopt only if better (don't stop it early).
8. Real-time perf: ~2 fps on CPU (HUD/easyocr-bound). For true-live, cadence the HUD read + try intel:gpu.
9. Toward ship-ready (≥130/150): a real live-YT-URL ingest proof + multi-match VOD segmentation + real
   calibration pairs (needs results-screen final + more matches' final-vision).

## REMEMBER (verified facts — do not re-fabricate)
- Vision emits FUEL ONLY → calibrate vision fuel → TBA `hubScore.totalCount` (NEVER total→total).
- Real TBA keys live in `SCHEMA_VERIFIED.md`. AUTO=20s. Tower L1/L2/L3=10/20/30. Foul 5/15.
- Cross-model verify caught 2 fabrications already (fake TBA SHA, invented RP thresholds/cycle distances).
  KEEP adversarially verifying every agent claim with a different model; run pytest before "done".

## ORCHESTRATION PROTOCOL
- Use the **Workflow tool** for reliable Claude fan-out (opus for hard/verify, sonnet for build/research),
  pipeline by default, adversarial-verify each finding with a 2nd independent prompt.
- Use the **CLI orchestrator / codex** for cross-model (GPT) second opinions on design + review.
- After each component: (a) run its tests, (b) have a different model adversarially review it, (c) log.
- Re-score against `RUBRIC_v2.md` each cycle; target the lowest dimension.
