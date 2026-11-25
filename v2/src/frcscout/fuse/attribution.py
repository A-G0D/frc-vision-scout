"""Probabilistic per-robot score attribution — the honest replacement for v1.

v1 shipped a tautological self-test (synthetic shots 1.5s before each known tick, random
robot assignment, assert sums match). This version assigns CREDIT WITH CONFIDENCE and
never fabricates an integer split.

For each overlay score delta (authoritative), find shot events from that alliance inside
[score_t - window_s, score_t + lookahead_s] (configs/events.yaml: score_attribution). Then:

    - 1 qualifying robot  -> credit it, confidence from timing tightness (1.0 at dt=0,
                             decaying to 0 at the edge of the window/lookahead)
    - N qualifying robots -> distribute a softmax over timing-tightness across
                             candidates and report it in detail["candidates"]; the
                             event's own robot_id stays None (genuinely ambiguous — never
                             fabricate a single winner) so downstream pool accounting
                             stays correct
    - 0 qualifying robots -> alliance-pool credit, confidence 0 (unattributed)

The original event's alliance/t_s/frame_idx/detail (pts/fuel_delta) always pass through
unchanged — only robot_id/confidence/detail["candidates"] are added — so summing
detail["fuel_delta"] over the output still reproduces the true overlay total.

Coverage (% of overlay points traced to a named robot) is REPORTED, not maximized — a low
number is an honest signal, not a bug to hide.
"""
from __future__ import annotations

import math
from typing import Optional

from frcscout.io.types import Event


class Attributor:
    def __init__(self, window_s: float = 3.0, lookahead_s: float = 0.5):
        self.window_s = float(window_s)
        self.lookahead_s = float(lookahead_s)

    @classmethod
    def from_config(cls, cfg) -> "Attributor":
        """Build from the ``score_attribution`` section of configs/events.yaml."""
        return cls(
            window_s=cfg.get("score_attribution.window_s", 3.0),
            lookahead_s=cfg.get("score_attribution.lookahead_s", 0.5),
        )

    def attribute(self, score_events: list[Event], shot_events: list[Event]) -> list[Event]:
        """Window-match each score delta to candidate shot events of the same alliance
        and emit one Event per score delta carrying the attribution (see module
        docstring for the 0/1/N-candidate policy)."""
        out = []
        for score in score_events:
            candidates = [
                s for s in shot_events
                if s.alliance == score.alliance
                and (score.t_s - self.window_s) <= s.t_s <= (score.t_s + self.lookahead_s)
            ]
            if not candidates:
                out.append(self._emit(score, robot_id=None, confidence=0.0, candidates=[]))
                continue

            weights = [self._timing_score(score.t_s, c.t_s) for c in candidates]
            if len(candidates) == 1:
                robot_id = candidates[0].robot_id
                confidence = weights[0]
                report = [{"robot_id": robot_id, "probability": confidence}]
                out.append(self._emit(score, robot_id=robot_id, confidence=confidence,
                                       candidates=report))
            else:
                probs = self._softmax(weights)
                report = [{"robot_id": c.robot_id, "probability": p}
                          for c, p in zip(candidates, probs)]
                # genuinely ambiguous: never fabricate a single winning robot_id.
                out.append(self._emit(score, robot_id=None, confidence=0.0,
                                       candidates=report))
        return out

    def _timing_score(self, score_t_s: float, shot_t_s: float) -> float:
        """1.0 at dt=0, decaying linearly to 0.0 at the edge of the matched window
        (looking back window_s) or lookahead (looking forward)."""
        dt = score_t_s - shot_t_s
        span = self.window_s if dt >= 0 else self.lookahead_s
        return max(0.0, 1.0 - abs(dt) / max(span, 1e-9))

    @staticmethod
    def _softmax(weights: list[float]) -> list[float]:
        m = max(weights)
        exps = [math.exp(w - m) for w in weights]
        total = sum(exps)
        return [e / total for e in exps]

    @staticmethod
    def _emit(score: Event, robot_id: Optional[int], confidence: float,
              candidates: list[dict]) -> Event:
        detail = dict(score.detail)
        detail["candidates"] = candidates
        return Event(t_s=score.t_s, frame_idx=score.frame_idx, type=score.type,
                     alliance=score.alliance, robot_id=robot_id, confidence=confidence,
                     detail=detail)

    @staticmethod
    def coverage(attributed: list[Event]) -> dict:
        """Per-alliance fraction of overlay points traced to a named robot (report-only)."""
        named = {"blue": 0, "red": 0}
        total = {"blue": 0, "red": 0}
        for e in attributed:
            if e.type != "score" or e.alliance is None:
                continue
            pts = e.detail.get("pts", e.detail.get("fuel_delta", 0))
            total[e.alliance] += pts
            if e.robot_id is not None:
                named[e.alliance] += pts * e.confidence
        return {al: (named[al] / total[al] if total[al] else 0.0) for al in ("blue", "red")}
