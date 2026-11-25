"""Scouting METRICS aggregation layer (§5 of the REBUILT-2026 game model).

Consumes a *finalized* ``match_record`` (events + per-robot possession stats +
an optional per-frame overlay timeseries + resolved identities) and produces the
FULL §5 scouting schema: per-robot analytics, per-alliance official score
reconstruction, the multi-match team aggregates, and the TBA-comparable block the
validation layer checks downstream.

Honesty contract
----------------
Every metric in §5 is emitted. Metrics derivable from what the engine already
produces (events / possession / overlay) carry a real value with
``available: True``. Metrics that need detectors we do NOT have yet (climb level,
zone/defense geofencing, sourcing location, fouls, disables) are STILL emitted —
with ``value: None``, ``available: False`` and a one-line ``needs:`` note. We
never fabricate a 0 for an unknown.
"""
from __future__ import annotations

from frcscout.metrics.scout import (
    MetricsAggregator,
    MetricsConfig,
    Metric,
    REGIONAL_THRESHOLDS,
    DISTRICT_CHAMPS_THRESHOLDS,
    FIRST_CHAMPS_THRESHOLDS,
)

__all__ = [
    "MetricsAggregator",
    "MetricsConfig",
    "Metric",
    "REGIONAL_THRESHOLDS",
    "DISTRICT_CHAMPS_THRESHOLDS",
    "FIRST_CHAMPS_THRESHOLDS",
]
