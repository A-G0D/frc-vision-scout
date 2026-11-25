"""frcscout — FRC REBUILT (2026) vision scout, v2.

Train on CUDA (cloud, scriptable — no browser). Deploy on Intel CPU/iGPU/NPU at 60fps.

Layers (each runs at its own cadence on its own compute engine):
    decode        QSV media engine    60Hz
    detect.robot  NPU                 60Hz   (6 large objects, cheap)
    detect.ball_inventory  iGPU        5Hz   (static carpet litter, high-res tiled)
    detect.ball_flight     CPU        60Hz   (the few airborne balls, classical)
    detect.overlay         CPU         2Hz   (authoritative match/score/fuel/phase)
    track/geometry/fuse    CPU        60Hz   (cheap; this is the hot path)

Design principle: the overlay is ground truth for the COUNT; CV is for SPACE
(who/where/which-arc/which-HUB). That split is what makes 60fps-on-CPU feasible.
"""
__version__ = "2.0.0"
