# REBUILT (2026) — Game Model for a Vision-Based Scoring & Scouting Engine

> **Purpose:** Complete, machine-oriented breakdown of the 2026 FRC game *REBUILT presented by Haas* — every way to earn points, every penalty, and every metric worth tracking — framed as the data model and detection targets for a video pipeline that watches matches and scores/scouts them automatically (replacing human scorers/scouts).
>
> **Source of truth:** Official *2026 FIRST Robotics Competition Game Manual*, Version **TU22** (`2026_REBUILT_GameManual.pdf`, 166 pp.). All numbers below are pulled from that manual. When the engine and a referee disagree, the referee/FMS is authoritative — this doc models the *intended* rules.
>
> **Scope note:** This is the game-understanding + framework layer. Robot/object detection models come later. This doc defines *what* must be detected and *how it maps to score*, so the perception layer has concrete targets.

---

## 1. Match structure (the state machine)

A match is **2:40 total**: 20s autonomous + 2:20 teleop. The engine must track a **match clock** and a **phase/shift state machine**, because *scoring eligibility changes by shift* (the HUB activation mechanic).

| Phase | Sub-period | Duration | Timer (counts down) | Both HUBs active? |
|-------|-----------|----------|---------------------|-------------------|
| **AUTO** | Autonomous | 20 s | 0:20 → 0:00 | ✅ Both active |
| **TELEOP** | TRANSITION SHIFT | 10 s | 2:20 → 2:10 | ✅ Both active |
| TELEOP | SHIFT 1 | 25 s | 2:10 → 1:45 | ⚠️ One active only |
| TELEOP | SHIFT 2 | 25 s | 1:45 → 1:20 | ⚠️ One active (alternates) |
| TELEOP | SHIFT 3 | 25 s | 1:20 → 0:55 | ⚠️ One active (alternates) |
| TELEOP | SHIFT 4 | 25 s | 0:55 → 0:30 | ⚠️ One active (alternates) |
| TELEOP | END GAME | 30 s | 0:30 → 0:00 | ✅ Both active |

**3-second scoring tails** the engine must honor (don't cut scoring at the buzzer):
- FUEL scored in the HUB keeps counting **up to 3 s after AUTO ends** and **up to 3 s after TELEOP ends**.
- FUEL also keeps counting **3 s after any HUB deactivates** (processing time through the sensor array).
- AUTO TOWER points assessed *after* AUTO 0:00.
- TELEOP TOWER points assessed **3 s after** TELEOP 0:00, or once all robots come to rest, whichever first.

### 1.1 The HUB activation mechanic (critical — drives scoring eligibility)
- Each alliance has **one HUB**. FUEL only scores points when put into an **active** HUB. FUEL into an **inactive** HUB = **0 points** (but is still physically "scored" / consumed).
- During AUTO, TRANSITION, and END GAME → **both HUBs active**.
- During SHIFT 1–4 → **exactly one** HUB active at a time, alternating each shift.
- **Who's inactive first?** The alliance that scored **more FUEL in AUTO** has their HUB set **inactive for SHIFT 1** (penalty for winning auto — flips the field). Tie in AUTO → FMS picks randomly.
- After SHIFT 1, active/inactive **alternates** each shift.
- **Engine implication:** to score correctly the engine must (a) count AUTO fuel per alliance, (b) derive the shift activation schedule from that, OR (c) read HUB light state directly (DMX light bars: alliance-color solid = active; pulsing w/ white chase = deactivation warning 3s out; off = inactive). Reading the lights is the more robust signal — see §6.

---

## 2. Field elements (detection landmarks)

Field is ~317.7 in × 651.2 in carpet. Per alliance: 1 OUTPOST, 1 HUB, 1 TOWER, 1 DEPOT. Shared: 2 DEPOTS total (1/alliance), 4 BUMPS, 4 TRENCHES, a CENTER LINE bisecting the NEUTRAL ZONE.

| Element | What it is | Why the engine cares |
|---------|-----------|----------------------|
| **FUEL** | 5.91 in (15 cm) foam ball, 0.45–0.50 lb | The *only* scoring element. Primary detection target. |
| **HUB** | 47×47 in tower, hexagonal **41.7 in opening at 72 in height**; sensor array inside | Scoring target for FUEL. Has **DMX light bars** showing active/inactive. |
| **TOWER** | 49×45×78 in climbing structure w/ 3 RUNGS: LOW @27 in, MID @45 in, HIGH @63 in | Endgame climb scoring; 3 levels. |
| **DEPOT** | Floor corral on ALLIANCE WALL | FUEL source/refill point. |
| **OUTPOST** | Human-player feed station; CHUTE (holds ~25 FUEL) + CORRAL | FUEL enters field here; robot↔human handoff. |
| **BUMP** | 73 in wide ramps either side of HUB | Field traversal; defines NEUTRAL ZONE edge. |
| **TRENCH** | Structure robots drive *under* (22.25 in clearance) | Traversal chokepoint. |
| **Zones** | ALLIANCE ZONE, NEUTRAL ZONE (center), CENTER LINE, ROBOT/HUMAN STARTING LINEs | Geofencing for rules (e.g. scoring zone, AUTO line crossing). |

### 2.1 AprilTags — the engine's coordinate anchors
- **32 unique tags**, family **36h11, IDs 1–32**, each **8.125 in (20.64 cm)** on a 10.5 in panel.
- Placement: HUB faces (IDs 2,3,4,5,8,9,10,11,18,19,20,21,24,25,26,27 — centers @44.25 in), TOWER WALL (15,16,31,32 @21.75 in), OUTPOST (13,14,29,30 @21.75 in), TRENCH (1,6,7,12,17,22,23,28 @35 in).
- **Use:** these give the vision system absolute field pose / camera localization for free. Map a detected tag ID → known field location → reproject everything else into field coordinates. This is the backbone for turning pixel detections into "FUEL entered red HUB" semantic events.

---

## 3. Scoring — every way to earn points

### 3.1 MATCH points (decide who wins the match)

| Action | Condition | AUTO value | TELEOP value |
|--------|-----------|-----------:|-------------:|
| **FUEL** | Scored in an **active** HUB (through top opening + sensor array) | **1** | **1** |
| FUEL | Scored in an **inactive** HUB | 0 | 0 |
| **TOWER — LEVEL 1** | Robot off the carpet/TOWER BASE (not touching ground) | **10** | **10** |
| **TOWER — LEVEL 2** | BUMPER covers entirely **above LOW RUNG** | — | **20** |
| **TOWER — LEVEL 3** | BUMPER covers entirely **above MID RUNG** | — | **30** |

**TOWER scoring rules the engine must encode:**
- **AUTO**: only **LEVEL 1** can be earned, and **max 2 robots** per alliance may earn AUTO LEVEL 1.
- **TELEOP**: each robot may earn points for **only one LEVEL** (the highest it qualifies for).
- A robot that climbed in AUTO **can still earn additional TOWER points** in TELEOP.
- To qualify, the robot must be **contacting at least one RUNG and/or UPRIGHT** of its own TOWER, and may *only* additionally contact: the TOWER WALL, support structure, FUEL, and/or another ROBOT.
- TOWER points are **human-judged** in real FRC — meaning they're the hardest for the engine and the highest-value to get right.

### 3.2 Ranking Points (RP) — decide qualification ranking, not match win
Awarded per qualification match. A win/tie plus up to 3 bonus RPs:

| RP | How earned | RP value |
|----|-----------|---------:|
| **Win** | More MATCH points than opponent | **3** |
| **Tie** | Equal MATCH points | **1** |
| **ENERGIZED RP** | Active-HUB FUEL total ≥ lower threshold | **1** |
| **SUPERCHARGED RP** | Active-HUB FUEL total ≥ higher threshold | **1** |
| **TRAVERSAL RP** | Total TOWER points in the match ≥ threshold | **1** |

**Bonus RP thresholds (Table 6-5):**

| Bonus RP | Regional / District | District Champs | FIRST Champs |
|----------|--------------------:|----------------:|-------------:|
| ENERGIZED (FUEL) | 100 | 240 | 360 |
| SUPERCHARGED (FUEL) | 360 | 360 | 500 |
| TRAVERSAL (TOWER) | 50 | 50 | 50 |

> Champs thresholds may rise via Team Updates — keep these **config-driven**, not hardcoded.

**Engine note:** ENERGIZED/SUPERCHARGED count only **active-HUB** FUEL (same pool the FUEL match points use). TRAVERSAL counts TOWER *points* (so one LEVEL-3 climb = 30 toward the 50 threshold).

---

## 4. Penalties — every foul & card

### 4.1 Penalty currency (Table 6-6)

| Penalty | Effect |
|---------|--------|
| **MINOR FOUL** | **+5** MATCH points to the **opponent** |
| **MAJOR FOUL** | **+15** MATCH points to the **opponent** |
| **YELLOW CARD** | Warning; 2nd yellow in a tournament phase → RED |
| **RED CARD** | Team DISQUALIFIED for that match (0 pts) |
| **DISABLED** | Robot deactivated for rest of match |
| **DISQUALIFIED** | 0 match pts + 0 RP (quals); 0 alliance pts (playoffs) |
| **VERBAL WARNING** | Spoken warning, often escalates on repeat |
| **Ineligible for RP** | Alliance loses a specified RP for that match |

Foul phrasing patterns: `per instance`, `per additional SCORING ELEMENT`, `MINOR then MAJOR if REPEATED`, and time-accrual (`MAJOR FOUL, +1 MAJOR per 3 s uncorrected`). Duration words: **MOMENTARY** ≈ <3 s, **CONTINUOUS** ≈ >10 s, **REPEATED** = >1× in a match.

### 4.2 In-match robot/gameplay fouls (G4xx — the video-detectable ones)
These are the penalties an automated engine could plausibly flag (the rest are human/pit/admin):

| Rule | Short name | Violation |
|------|-----------|-----------|
| **G401** | Behind the lines (AUTO) — drive team in staged areas | MINOR FOUL |
| **G402** | Let the robot do its thing (no AUTO interaction) | MINOR FOUL + YELLOW |
| **G403** | Limited AUTO opponent interaction (don't fully cross CENTER LINE) | MAJOR FOUL + MAJOR per contact |
| **G404** | Robots: use SCORING ELEMENTS as directed | MAJOR FOUL |
| **G405** | Keep SCORING ELEMENTS in bounds (no intentional ejection out) | MINOR FOUL; MAJOR if REPEATED |
| **G406** | Don't abuse SCORING ELEMENTS (no damage) | VERBAL; MAJOR if REPEATED |
| **G407** | Only score while in your ALLIANCE ZONE | MAJOR FOUL |
| **G408** | Don't catch FUEL released by the HUB | MINOR; MAJOR+VERBAL if strategic; +YELLOW if repeated strategic |
| **G409** | Robots must be safe | DISABLED + corrective action |
| **G410** | Keep BUMPERS low (no extension into carpet/BUMPS/TOWER) | MINOR FOUL |
| **G411** | Don't damage the field | VERBAL; DISABLED if more damage likely |
| **G412** | Watch your field interaction | MAJOR; +YELLOW if REPEATED/>MOMENTARY |
| **G413** | Expansion limits (extension envelope) | MAJOR FOUL + corrective action |
| **G414** | Don't climb on each other (no fully supporting another robot) | Supported robot ineligible for TOWER points |
| **G415** | Stay out of other robots (bumpers fully outside own zone) | MAJOR + YELLOW; worse if opponent can't drive |
| **G416** | This isn't combat robotics (no intentional/reckless damage) | MAJOR + YELLOW; escalates |
| **G417** | Don't tip or entangle | MAJOR + YELLOW; MAJOR + RED if CONTINUOUS / opponent disabled |
| **G418** | 3-count on PINS (no pin >3 s) | MINOR + MAJOR per 3 s uncorrected |
| **G419** | Don't collude to shut down gameplay | MAJOR + MAJOR per 3 s uncorrected |
| **G420** | TOWER protection (don't contact opponent robot on/at their tower) | MAJOR; opponent awarded points if off ground |
| **G421** | No wandering (drive team in designated area) | MINOR FOUL |
| **G422** | Hands off the controls (only DRIVERS operate) | MAJOR; RED if >MOMENTARY |
| **G424** | Humans: use SCORING ELEMENTS as directed | MAJOR FOUL |
| **G425** | SCORING ELEMENT delivery (FUEL enters only via HUMAN PLAYER/legal path) | MAJOR FOUL |
| **G426** | Drive coaches: don't touch SCORING ELEMENTS | MINOR FOUL |
| **G427** | OUTPOST storage limit (off-field FUEL only in CHUTE/CORRAL) | MINOR; MAJOR if CONTINUOUS |

> Full G1xx/G2xx/G3xx (safety, conduct, pre-match), all R-rules (robot construction/electrical/pneumatics), and T-rules (tournament/alliance-selection) exist but are **not in-match video events** — out of scope for the scoring engine, in scope only if you later build a rules-compliance auditor. Notable strategic ones: **G206** (no colluding for RPs → ineligible for bonus RPs), **G209** (leaving a part on field → RED), **G210** (harm-others strategy → MAJOR).

---

## 5. Metrics to track (the scouting schema)

Two buckets: **(A) official score reconstruction** (what the engine must compute to mirror FMS) and **(B) scouting analytics** (what makes the data valuable beyond the scoreboard).

### 5.1 Per-match, per-alliance (official reconstruction)
- `auto_fuel_scored` (active HUB) — also the tiebreaker for HUB activation order
- `teleop_fuel_scored` (active HUB)
- `fuel_into_inactive_hub` (0 pts but useful waste metric)
- `tower_climbs`: list of `{robot, level (1/2/3), phase (auto/teleop)}`
- `auto_tower_l1_count` (cap 2)
- `match_points_total`, `rp_total`, and each RP flag: `win/tie`, `energized`, `supercharged`, `traversal`
- `minor_fouls_received`, `major_fouls_received`, `foul_points_credited`
- `cards`: yellow/red per team

### 5.2 Per-robot, per-match (scouting gold)
- **Fuel cycle metrics:** fuel scored, scoring rate (fuel/min), avg cycle time (pickup→score), miss rate (shots not entering HUB)
- **Sourcing mix:** % from DEPOT vs OUTPOST/human vs floor/center vs preload
- **Active-shift efficiency:** fuel scored *while own HUB active* vs *wasted into inactive HUB* (a key skill signal — good teams hold fuel during inactive shifts)
- **AUTO performance:** preload count, auto fuel, auto mobility (left starting line?), auto LEVEL-1 climb (y/n)
- **Climb:** best level achieved, climb start time, climb success/fail, time-to-climb
- **Defense:** time spent in opponent zone, pins drawn/committed, fouls drawn/committed
- **Reliability:** disabled/disconnected events, no-shows, tip-overs
- **Endgame contribution:** fuel + tower points in the 0:30 END GAME window

### 5.3 Derived team aggregates (rankings mirror — Table 10-1 tiebreakers)
Engine should replicate the official ranking sort so its standings match FMS:
1. **Ranking Score** (avg RP)
2. **Avg ALLIANCE MATCH points** *excluding* foul points
3. **Avg FUEL scored in AUTO**
4. **Avg TOWER points**
5. Random (FMS)

Plus scouting aggregates: OPR/DPR/CCWM-style contributions, consistency (variance), climb-rate %, avg fuel/match, active-shift conversion %.

---

## 6. Detection-engine framework (how the above maps to perception)

The pipeline turns video → semantic events → score. Suggested layering:

```
[video frames]
   → AprilTag detection (36h11) ........ camera↔field pose, identify HUB/TOWER/OUTPOST/TRENCH
   → field registration ................ build homography → field coordinates
   → object detection .................. FUEL (balls), ROBOTS (bumper color=alliance, team # OCR)
   → state estimation .................. match clock + phase/shift FSM + HUB active state (lights)
   → event detection ................... FUEL-into-HUB, climb-level transitions, zone crossings
   → scoring/rules engine .............. apply §3 values + §1.1 eligibility + §4 fouls
   → metrics aggregator ................ emit §5 schema (per-match, per-robot, per-team)
```

### 6.1 The handful of "atomic events" everything reduces to
1. **FUEL_SCORED**`{alliance, hub, t, hub_active?}` — ball passes the HUB top opening. *Hardest sensor:* the real HUB uses an internal sensor array; from video you detect a ball crossing into the 41.7 in opening at 72 in height. Honor the 3-s tail.
2. **CLIMB_STATE**`{robot, level, t}` — robot transitions to LEVEL 1/2/3 (bumper-cover height vs LOW/MID rung). Latch highest level per robot per phase; enforce AUTO=L1-only & ≤2.
3. **ZONE_TRANSITION**`{robot, zone, t}` — for AUTO line crossing (G403), scoring-in-zone (G407), wandering/defense metrics.
4. **HUB_ACTIVE_CHANGE**`{hub, active, t}` — ideally read the DMX light bars directly; fallback = derive from AUTO fuel count + shift schedule.
5. **CONTACT/PIN/FOUL candidates**`{robotA, robotB, type, t}` — robot-robot proximity & persistence for G415–G420; flag as *candidate* fouls for human confirmation (don't auto-assess — referees are authoritative and never review video).

### 6.2 Phasing recommendation (build order)
1. **Clock + phase FSM** (no vision needed; sync to the audience-display timer or audio cues¹).
2. **AprilTag localization + HUB lights** → gives you eligibility for free.
3. **FUEL scoring detection** → reconstructs the bulk of points + ENERGIZED/SUPERCHARGED RP.
4. **Climb level detection** → TOWER points + TRAVERSAL RP (high value, human-judged in reality so accuracy matters).
5. **Robot ID (bumper color + team-number OCR)** → attributes events to teams = scouting.
6. **Foul candidates** → last, advisory-only.

¹ Audio cues (Table 5-4): "Cavalry Charge" @ AUTO start, Buzzer @ AUTO end, "3 Bells" @ teleop/transition (2:20), "Power Up" pops at shift changes (2:10/1:45/1:20/0:55), "Steam Whistle" @ END GAME (0:30), Buzzer @ end, Foghorn @ stop. Useful as a cheap secondary clock sync.

### 6.3 Key gotchas the engine must not get wrong
- **Inactive-HUB fuel = 0 points** but still happens — track it separately, never add to score.
- **3-second scoring tails** at AUTO/TELEOP end and after HUB deactivation.
- **AUTO fuel count determines shift order** — get AUTO right or all of TELEOP eligibility is wrong.
- **AUTO LEVEL-1 climb caps at 2 robots/alliance**; AUTO is L1-only.
- **One TELEOP level per robot** (count the highest, not the sum).
- **Foul points are excluded** from the 2nd ranking tiebreaker — keep foul points segregated from "earned" points.
- Don't auto-assess fouls as official; referees never consult video (per manual §6.7).

---

## 7. Quick-reference cheat sheet

```
FUEL (active HUB)      = 1 pt   (AUTO or TELEOP)
TOWER L1              = 10 pt   (AUTO: ≤2 robots; or TELEOP)
TOWER L2             = 20 pt   (TELEOP only)
TOWER L3             = 30 pt   (TELEOP only)
WIN = 3 RP | TIE = 1 RP
ENERGIZED RP   : active-HUB FUEL ≥ 100 / 240 / 360
SUPERCHARGED RP: active-HUB FUEL ≥ 360 / 360 / 500
TRAVERSAL RP   : TOWER points ≥ 50
MINOR FOUL = +5 to opponent | MAJOR FOUL = +15 to opponent
Match = 20s AUTO + 2:20 TELEOP (Transition 10s, Shifts 1-4 ×25s, Endgame 30s)
HUB active: both in AUTO/Transition/Endgame; one-at-a-time in Shifts 1-4
```

---

*Generated from the official 2026 REBUILT Game Manual (TU22). Manual PDF: `C:\Users\agodavarthy\Claude\2026_REBUILT_GameManual.pdf`. Online: https://firstfrc.blob.core.windows.net/frc2026/Manual/2026GameManual.pdf*
