"""Visualization layer — scouting-demo render on top of the validated engine output.

The single public entry point is :func:`draw_scout_overlay` (in ``scout_render``),
which annotates a frame with the 6-robot prior, per-robot PHASE labels, ball flight /
inventory, recent events, and a right-side scouting METRICS panel. It is headless-safe
(returns the annotated ndarray; never calls ``cv2.imshow``) and the heavy deps
(numpy / cv2) are imported lazily inside the draw call, so importing this package never
requires them.
"""
from __future__ import annotations

__all__ = ["draw_scout_overlay"]


def __getattr__(name: str):
    # Lazy re-export so ``from frcscout.viz import draw_scout_overlay`` works without
    # importing scout_render (and its guarded cv2/numpy touch) at package import.
    if name == "draw_scout_overlay":
        from frcscout.viz.scout_render import draw_scout_overlay
        return draw_scout_overlay
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
