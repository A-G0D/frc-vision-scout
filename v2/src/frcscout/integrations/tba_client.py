"""TBA API v3 client — typed, cached, defensive.

Provides live access to The Blue Alliance match data for:
    - verifying vision-pipeline estimates against official scores
    - collecting (vision_est, tba_actual) calibration pairs for error_model.py

Design constraints
------------------
* Auth via X-TBA-Auth-Key header from env var TBA_AUTH_KEY.  Never hardcoded.
  If the env var is absent the client enters *offline mode*: all requests are
  served from the on-disk cache only; a live API call raises TBAAuthError.
* On-disk response cache under data/tba_cache/ (JSON files, keyed by URL path).
  The cache is read-through: a hit returns the cached response; a miss fetches,
  stores, then returns.
* score_breakdown fields are read DEFENSIVELY because the exact 2026 TBA JSON
  key names are TENTATIVE (see metrics/TBA_2026_breakdown.md).  extract_breakdown
  tries every candidate name, records which ones it actually found, and never
  raises on missing keys — it just reports None + surfaces which keys were absent.

Public API
----------
    TBAClient(cache_dir, offline)           constructor
    client.get_match(match_key)             -> dict (raw TBA match response)
    client.get_event_matches(event_key)     -> list[dict]
    client.extract_breakdown_components(m)  -> BreakdownComponents

    TBAAuthError                            raised when auth key absent + live needed
    TBACacheError                           raised when cache read fails unexpectedly
    BreakdownComponents                     typed dataclass returned by extract_breakdown
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TBA_BASE = "https://www.thebluealliance.com/api/v3"

#: Default on-disk cache location, relative to this file's v2 root.
#: file = v2/src/frcscout/integrations/tba_client.py -> parents[3] == v2 (NOT [4]=repo root).
_DEFAULT_CACHE_DIR = (
    Path(__file__).resolve().parents[3] / "data" / "tba_cache"
)

# ---------------------------------------------------------------------------
# 2026 score_breakdown candidate field names
#
# The exact TBA JSON key names for 2026 REBUILT are TENTATIVE/INFERRED
# (see metrics/TBA_2026_breakdown.md).  We list every plausible candidate so
# the client can probe them and report which were actually found.  Once a live
# 2026 match is fetched the real names can be confirmed and locked here.
#
# Layout: each tuple is (component_label, [candidate_key_names_in_priority_order])
# ---------------------------------------------------------------------------

_BREAKDOWN_CANDIDATES: Dict[str, List[str]] = {
    # Grand total
    "total_score": ["totalPoints", "total_points", "totalScore", "total_score"],

    # Fuel / hub scoring — nested under hubScore or flat
    "fuel_auto_points": [
        "hubScore.autoPoints",
        "autoFuelPoints",
        "auto_fuel_points",
        "autoHubPoints",
        "hubScore.auto_points",
    ],
    "fuel_teleop_points": [
        "hubScore.teleopPoints",
        "teleopFuelPoints",
        "teleop_fuel_points",
        "teleopHubPoints",
        "hubScore.teleop_points",
    ],
    "fuel_total_points": [
        "hubScore.totalPoints",
        "fuelPoints",
        "fuel_points",
        "hubTotalPoints",
        "hub_total_points",
    ],
    "fuel_auto_count": [
        "hubScore.autoCount",
        "autoFuelCount",
        "auto_fuel_count",
    ],
    "fuel_teleop_count": [
        "hubScore.teleopCount",
        "teleopFuelCount",
        "teleop_fuel_count",
    ],
    "fuel_total_count": [
        "hubScore.totalCount",
        "fuelCount",
        "fuel_count",
        "totalFuelCount",
    ],

    # Tower climbing
    "tower_auto_points": [
        "autoTowerPoints",
        "auto_tower_points",
        "autoClimbPoints",
        "auto_climb_points",
    ],
    "tower_endgame_points": [
        "endGameTowerPoints",
        "endgame_tower_points",
        "endGameClimbPoints",
        "endgame_climb_points",
    ],
    "tower_total_points": [
        "totalTowerPoints",
        "total_tower_points",
        "climbPoints",
        "climb_points",
    ],

    # Fouls (received by this alliance, i.e. from opponent fouls)
    "foul_points": ["foulPoints", "foul_points"],
    "minor_foul_count": ["minorFoulCount", "minor_foul_count"],
    "major_foul_count": ["majorFoulCount", "major_foul_count"],

    # Adjustments
    "adjust_points": ["adjustPoints", "adjust_points"],
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TBAAuthError(RuntimeError):
    """Raised when a live API call is needed but TBA_AUTH_KEY is not set."""


class TBACacheError(OSError):
    """Raised when reading a cache file fails in an unexpected way."""


# ---------------------------------------------------------------------------
# BreakdownComponents — typed return value for extract_breakdown_components
# ---------------------------------------------------------------------------

@dataclass
class AllianceBreakdown:
    """Extracted components for one alliance in one match.

    Every numeric field may be None if TBA did not include it (or if the
    key name was not found among the candidates we tried).
    """
    alliance: str                           # "blue" or "red"

    total_score: Optional[int] = None

    fuel_auto_points: Optional[int] = None
    fuel_teleop_points: Optional[int] = None
    fuel_total_points: Optional[int] = None
    fuel_auto_count: Optional[int] = None
    fuel_teleop_count: Optional[int] = None
    fuel_total_count: Optional[int] = None

    tower_auto_points: Optional[int] = None
    tower_endgame_points: Optional[int] = None
    tower_total_points: Optional[int] = None

    foul_points: Optional[int] = None
    minor_foul_count: Optional[int] = None
    major_foul_count: Optional[int] = None
    adjust_points: Optional[int] = None

    # Audit trail — populated by extract_breakdown_components
    keys_found: Dict[str, str] = field(default_factory=dict)
    """Maps component_label -> actual JSON key that was found."""
    keys_missing: List[str] = field(default_factory=list)
    """Component labels for which no candidate key matched."""
    raw_breakdown: Optional[Dict[str, Any]] = None
    """The raw alliance score_breakdown dict, for future key discovery."""


@dataclass
class BreakdownComponents:
    """Breakdown for both alliances in one match."""
    match_key: str
    blue: AllianceBreakdown
    red: AllianceBreakdown
    # True only if score_breakdown was present in the TBA response
    breakdown_available: bool = False
    # Schema status: the 2026 field names ARE confirmed against real event data
    # (data/tba/2026ctwat_matches.json, 79 matches) — see metrics/SCHEMA_VERIFIED.md.
    # The candidate-key probing remains as defensive fallback, but the priority-0
    # candidates (hubScore.totalCount, totalTowerPoints, foulPoints, …) are the verified keys.
    schema_note: str = (
        "score_breakdown field names verified against real 2026ctwat event data "
        "(see metrics/SCHEMA_VERIFIED.md); priority-0 candidate keys are the confirmed names."
    )


# ---------------------------------------------------------------------------
# TBAClient
# ---------------------------------------------------------------------------

class TBAClient:
    """Thin, cached client for TBA API v3.

    Parameters
    ----------
    cache_dir : Path or str, optional
        Directory for on-disk JSON cache.  Created on first use.
        Defaults to <v2_root>/data/tba_cache/.
    offline : bool, optional
        If True, never attempt live requests; only serve from cache.
        Automatically True when TBA_AUTH_KEY is not set in the environment.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        offline: Optional[bool] = None,
    ) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._auth_key: Optional[str] = os.environ.get("TBA_AUTH_KEY")

        if offline is None:
            self._offline = self._auth_key is None
        else:
            self._offline = offline

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_match(self, match_key: str) -> dict:
        """Fetch a single match by its TBA match key (e.g. '2026miket_qm12').

        Returns the raw TBA match dict.  Raises TBAAuthError if offline and
        no cached response exists.
        """
        path = f"/match/{match_key}"
        return self._fetch(path)

    def get_event_matches(self, event_key: str) -> List[dict]:
        """Fetch all matches for an event (e.g. '2026miket').

        Returns a list of raw TBA match dicts.
        """
        path = f"/event/{event_key}/matches"
        result = self._fetch(path)
        if isinstance(result, list):
            return result
        # Some cache hits might be wrapped in a dict — unwrap defensively
        if isinstance(result, dict) and "matches" in result:
            return result["matches"]
        return []

    def extract_breakdown_components(
        self, match: dict
    ) -> BreakdownComponents:
        """Extract the score components our error model uses from a TBA match dict.

        Reads score_breakdown defensively: probes every candidate key name,
        records which ones were found vs absent, and never raises on a missing
        key.  Both alliances are processed and returned as BreakdownComponents.

        Parameters
        ----------
        match : dict
            A raw TBA match dict (as returned by get_match or get_event_matches).

        Returns
        -------
        BreakdownComponents
            Contains AllianceBreakdown for blue and red, plus audit fields
            (keys_found, keys_missing, raw_breakdown) that allow locking in
            the real 2026 field names once confirmed.
        """
        key = match.get("key", "unknown")
        sb = match.get("score_breakdown")

        available = isinstance(sb, dict)

        blue_bd = self._extract_alliance("blue", sb if available else None, match)
        red_bd = self._extract_alliance("red", sb if available else None, match)

        return BreakdownComponents(
            match_key=key,
            blue=blue_bd,
            red=red_bd,
            breakdown_available=available,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch(self, api_path: str) -> Any:
        """Return JSON for api_path, hitting cache first, then live TBA."""
        cache_file = self._cache_path(api_path)

        # Cache hit
        if cache_file.exists():
            try:
                with open(cache_file, encoding="utf-8") as fh:
                    return json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                raise TBACacheError(
                    f"Failed to read cache file {cache_file}: {exc}"
                ) from exc

        # Offline — no cache hit
        if self._offline:
            if self._auth_key is None:
                raise TBAAuthError(
                    "TBA_AUTH_KEY environment variable is not set. "
                    "Set it to your TBA API key, or pre-populate the cache "
                    f"at {self._cache_dir} to use offline mode. "
                    f"Requested path: {api_path}"
                )
            raise TBAAuthError(
                f"Offline mode is active and no cached response exists for: {api_path}"
            )

        # Live fetch
        data = self._live_fetch(api_path)

        # Write to cache
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

        return data

    def _live_fetch(self, api_path: str) -> Any:
        """Execute a real HTTP request to TBA API v3."""
        if not self._auth_key:
            raise TBAAuthError(
                "TBA_AUTH_KEY environment variable is not set.  "
                "Cannot make live API requests without an auth key."
            )
        url = _TBA_BASE + api_path
        req = urllib.request.Request(
            url,
            headers={
                "X-TBA-Auth-Key": self._auth_key,
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"TBA API returned HTTP {exc.code} for {url}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Network error fetching {url}: {exc.reason}"
            ) from exc
        return json.loads(body)

    def _cache_path(self, api_path: str) -> Path:
        """Map an API path to a deterministic cache file path."""
        # Use a hash suffix so long paths don't exceed OS filename limits
        slug = api_path.lstrip("/").replace("/", "_")
        if len(slug) > 80:
            slug = slug[:60] + "_" + hashlib.md5(api_path.encode()).hexdigest()[:12]
        return self._cache_dir / f"{slug}.json"

    @staticmethod
    def _extract_alliance(
        alliance: str,
        score_breakdown: Optional[dict],
        match: dict,
    ) -> AllianceBreakdown:
        """Extract one alliance's breakdown components from score_breakdown."""
        bd = AllianceBreakdown(alliance=alliance)

        # The top-level match dict carries totalPoints as blue_score / red_score
        bd.total_score = match.get(f"{alliance}_score")

        if score_breakdown is None:
            bd.keys_missing = list(_BREAKDOWN_CANDIDATES.keys())
            return bd

        # Alliance sub-dict (TBA stores per-alliance under 'blue'/'red')
        alliance_sb: Optional[dict] = score_breakdown.get(alliance)
        if alliance_sb is None:
            bd.raw_breakdown = score_breakdown
            bd.keys_missing = list(_BREAKDOWN_CANDIDATES.keys())
            return bd

        bd.raw_breakdown = dict(alliance_sb)  # shallow copy for audit

        # hubScore may be a nested dict — flatten into a flat lookup with dotted keys
        flat = _flatten_breakdown(alliance_sb)

        for component, candidates in _BREAKDOWN_CANDIDATES.items():
            if component == "total_score":
                # Prefer the flat match-level score; also try breakdown total
                if bd.total_score is None:
                    val, found_key = _probe(flat, candidates)
                    if found_key:
                        bd.total_score = val
                        bd.keys_found[component] = found_key
                    else:
                        bd.keys_missing.append(component)
                # If we already have it from match root, mark found with sentinel
                elif bd.total_score is not None:
                    bd.keys_found[component] = f"<match.{alliance}_score>"
                continue

            val, found_key = _probe(flat, candidates)
            if found_key is not None:
                setattr(bd, component, val)
                bd.keys_found[component] = found_key
            else:
                bd.keys_missing.append(component)

        return bd


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _flatten_breakdown(d: dict, prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested breakdown dict using dot-notation keys.

    e.g. {"hubScore": {"autoPoints": 12}} becomes {"hubScore.autoPoints": 12}
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        full_key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten_breakdown(v, full_key))
        else:
            out[full_key] = v
    # Also include top-level non-dotted keys for flat candidate names
    if not prefix:
        for k, v in d.items():
            if not isinstance(v, dict):
                out[k] = v
    return out


def _probe(
    flat: Dict[str, Any], candidates: List[str]
) -> Tuple[Optional[Any], Optional[str]]:
    """Return the first candidate key that is present in flat, plus the key name.

    Returns (value, key_name) or (None, None) if none match.
    """
    for cand in candidates:
        if cand in flat:
            return flat[cand], cand
    return None, None
