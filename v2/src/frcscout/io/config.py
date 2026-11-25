"""Tiny YAML config loader with dotted-path access and seconds->frames helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """dict with attribute + dotted access: cfg.get('detectors.robot.imgsz')."""

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path) as f:
            return cls(yaml.safe_load(f))

    def get(self, dotted: str, default: Any = None) -> Any:  # type: ignore[override]
        node: Any = self
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node


def seconds_to_frames(seconds: float, fps: float) -> int:
    """The conversion v1 got wrong everywhere by hardcoding fps=30. Always use the
    capture's REAL fps, passed in from the decode layer."""
    return max(1, round(seconds * fps))
