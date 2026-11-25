"""
match_list.py — pull the ground-truth match schedule for an FRC event.

This is the "answer key" the video pipeline is validated against: for each match we know
exactly which 3 red + 3 blue teams competed, and the final score. The vision pipeline
detects match-preview graphics in the stream and reads the team numbers; we then align
those detections to this schedule to (a) confirm correctness and (b) timestamp each match
within the multi-hour video.

Two backends:
  - The Blue Alliance (TBA) API v3  -> requires a free auth key (env TBA_AUTH_KEY).
      The user asked for TBA specifically; this is the primary path.
  - Statbotics API v3               -> open, no key. Fallback so we're never blocked.

An FRC "event key" looks like 2026<district/region code><event>, e.g. 2026ctwat.
Find it from the event's TBA URL: thebluealliance.com/event/<KEY>.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field

import requests

TBA_BASE = "https://www.thebluealliance.com/api/v3"
STATBOTICS_BASE = "https://api.statbotics.io/v3"


@dataclass
class Match:
    key: str               # e.g. "2026ctwat_qm12"
    comp_level: str        # qm / sf / f
    set_number: int
    match_number: int
    red: list[int]         # 3 team numbers
    blue: list[int]        # 3 team numbers
    red_score: int | None = None
    blue_score: int | None = None
    winner: str | None = None   # "red" / "blue" / "tie"

    @property
    def label(self) -> str:
        if self.comp_level == "qm":
            return f"Qual {self.match_number}"
        if self.comp_level == "sf":
            return f"Playoff SF{self.set_number}-{self.match_number}"
        if self.comp_level == "f":
            return f"Final {self.match_number}"
        return self.key

    @property
    def teams(self) -> list[int]:
        return self.red + self.blue


# ----------------------------------------------------------------------------- TBA
def _tba_headers() -> dict:
    key = os.environ.get("TBA_AUTH_KEY", "").strip()
    if not key:
        raise RuntimeError("TBA_AUTH_KEY not set; use statbotics backend instead")
    return {"X-TBA-Auth-Key": key}


def from_tba(event_key: str, timeout: int = 60) -> list[Match]:
    r = requests.get(f"{TBA_BASE}/event/{event_key}/matches",
                     headers=_tba_headers(), timeout=timeout)
    r.raise_for_status()
    out = []
    for m in r.json():
        red = [int(t[3:]) for t in m["alliances"]["red"]["team_keys"]]
        blue = [int(t[3:]) for t in m["alliances"]["blue"]["team_keys"]]
        rs = m["alliances"]["red"].get("score")
        bs = m["alliances"]["blue"].get("score")
        out.append(Match(
            key=m["key"], comp_level=m["comp_level"],
            set_number=m["set_number"], match_number=m["match_number"],
            red=red, blue=blue,
            red_score=rs if rs is not None and rs >= 0 else None,
            blue_score=bs if bs is not None and bs >= 0 else None,
            winner=(m.get("winning_alliance") or None),
        ))
    return _sorted(out)


# ----------------------------------------------------------------- Statbotics (open)
def from_statbotics(event_key: str, timeout: int = 60) -> list[Match]:
    r = requests.get(f"{STATBOTICS_BASE}/matches",
                     params={"event": event_key, "limit": 1000}, timeout=timeout)
    r.raise_for_status()
    out = []
    for m in r.json():
        a = m["alliances"]
        red = [a["red"]["team_keys"][i] for i in range(3)] if "team_keys" in a["red"] \
              else [m["red_1"], m["red_2"], m["red_3"]]
        blue = [a["blue"]["team_keys"][i] for i in range(3)] if "team_keys" in a["blue"] \
               else [m["blue_1"], m["blue_2"], m["blue_3"]]
        out.append(Match(
            key=m["key"], comp_level=m.get("comp_level", "qm"),
            set_number=int(m.get("set_number", 1)),
            match_number=int(m.get("match_number", 0)),
            red=[int(x) for x in red], blue=[int(x) for x in blue],
            red_score=m.get("result", {}).get("red_score"),
            blue_score=m.get("result", {}).get("blue_score"),
            winner=m.get("result", {}).get("winner"),
        ))
    return _sorted(out)


def _sorted(matches: list[Match]) -> list[Match]:
    order = {"qm": 0, "ef": 1, "qf": 2, "sf": 3, "f": 4}
    return sorted(matches, key=lambda m: (order.get(m.comp_level, 9),
                                          m.set_number, m.match_number))


def get_matches(event_key: str, prefer: str = "auto") -> list[Match]:
    """prefer: 'tba' | 'statbotics' | 'auto' (tba if key present else statbotics)."""
    if prefer == "tba" or (prefer == "auto" and os.environ.get("TBA_AUTH_KEY")):
        try:
            return from_tba(event_key)
        except Exception as e:
            if prefer == "tba":
                raise
            print(f"[match_list] TBA failed ({e}); falling back to statbotics")
    return from_statbotics(event_key)


if __name__ == "__main__":
    import json, sys
    ek = sys.argv[1] if len(sys.argv) > 1 else "2026ctwat"
    ms = get_matches(ek)
    print(f"{len(ms)} matches for {ek}")
    for m in ms[:5]:
        print(f"  {m.label:16} red={m.red} blue={m.blue} score={m.red_score}-{m.blue_score}")
    json.dump([asdict(m) for m in ms],
              open(f"data/{ek}_matches.json", "w"), indent=2)
