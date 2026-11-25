"""Shared dataclasses passed between layers. Frozen where possible for thread-safety
(detectors run in worker threads; the consumer reads snapshots)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Alliance = Literal["red", "blue"]
EventType = Literal["pickup", "shot", "score", "climb"]


@dataclass(frozen=True)
class Detection:
    """Raw detector output, image pixels."""
    cls: str                      # "ball" | "robot"
    xyxy: tuple[float, float, float, float]
    conf: float
    frame_idx: int


@dataclass
class Track:
    """A tracked object with identity and motion state."""
    tid: int
    cls: str
    xyxy: tuple[float, float, float, float]
    centroid_px: tuple[float, float]
    field_xy_m: Optional[tuple[float, float]] = None   # filled by geometry layer
    vel_mps: tuple[float, float] = (0.0, 0.0)
    alliance: Optional[Alliance] = None                # robots only (bumper color)
    age: int = 0
    hits: int = 1
    is_flight: bool = False                            # ball in the air


@dataclass
class CameraPose:
    """Recovered camera geometry. `H` maps image px -> field-plane meters (z=0)."""
    H: list[list[float]]                  # 3x3 homography (image -> ground plane)
    rvec: Optional[list[float]] = None    # full extrinsics (if PnP solved)
    tvec: Optional[list[float]] = None
    reproj_err_px: float = 0.0
    n_tags: int = 0
    valid: bool = False


@dataclass
class OverlayState:
    """Authoritative match data read from the broadcast overlay (ground truth)."""
    match_label: Optional[str] = None
    comp_level: Optional[str] = None
    match_number: Optional[int] = None
    blue_teams: list[int] = field(default_factory=list)
    red_teams: list[int] = field(default_factory=list)
    blue_score: Optional[int] = None
    red_score: Optional[int] = None
    blue_fuel: Optional[int] = None
    red_fuel: Optional[int] = None
    phase: Optional[str] = None           # "auto" | "teleop" | "endgame"
    timer_s: Optional[float] = None


@dataclass
class Event:
    t_s: float
    frame_idx: int
    type: EventType
    alliance: Optional[Alliance] = None
    robot_id: Optional[int] = None
    confidence: float = 1.0               # probabilistic attribution -> NOT always 1.0
    detail: dict = field(default_factory=dict)


@dataclass
class FrameState:
    """Everything the engine knows at one frame — the 60Hz output record."""
    frame_idx: int
    t_s: float
    robots: list[Track] = field(default_factory=list)
    balls_flight: list[Track] = field(default_factory=list)
    balls_inventory: list[Track] = field(default_factory=list)
    overlay: OverlayState = field(default_factory=OverlayState)
    pose: Optional[CameraPose] = None
    events: list[Event] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)    # per-stage latency for the budget gate
