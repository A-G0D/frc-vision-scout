"""Multi-view pane segmentation — split the STACKED broadcast composite into its individual
synchronized camera views.

The 2026 REBUILT broadcast stacks a main elevated wide view over a low opposite-angle view,
separated by a dark divider bar (the catwalk/structure). Splitting the composite gives 2–3
views of the SAME instant; each can then be AprilTag-calibrated (tagmap + apriltag.py) to field
coordinates and the views triangulated → recover occluded balls + true 3D positions.

Detection: find wide dark/low-variance horizontal bands (dividers); the segments between them
are panes. Falls back to config fractions, then to a single full-frame pane. numpy guarded.
"""
from __future__ import annotations


def find_pane_ranges(frame, min_pane_frac: float = 0.18, dark_thresh: float = 0.16):
    """Return [(y0, y1), ...] row-ranges of the stacked panes by detecting divider bands.
    A divider row is dark AND low-variance (a uniform bar). Pure-numpy."""
    import numpy as np
    g = frame.mean(axis=2) if getattr(frame, "ndim", 2) == 3 else frame
    H = g.shape[0]
    rowmean = g.mean(axis=1) / 255.0
    rowvar = g.var(axis=1)
    med = float(np.median(rowvar)) or 1.0
    divider = (rowmean < dark_thresh) | (rowvar < med * 0.25)
    ranges = []
    y = 0
    while y < H:
        if divider[y]:
            y += 1
            continue
        y0 = y
        while y < H and not divider[y]:
            y += 1
        if (y - y0) >= min_pane_frac * H:
            ranges.append((y0, y))
    return ranges or [(0, H)]


def segment_panes(frame, fracs=None):
    """Split into panes. `fracs` = optional [(y0f, y1f), ...] in fractions of H (use when the
    layout is known/fixed); else auto-detect dividers. Returns
    [{'name','y0','y1','crop','offset':(x0,y0),'size':(w,h)}]."""
    H, W = frame.shape[:2]
    if fracs:
        ranges = [(int(a * H), int(b * H)) for a, b in fracs]
    else:
        ranges = find_pane_ranges(frame)
    panes = []
    for i, (y0, y1) in enumerate(ranges):
        panes.append({
            "name": f"pane{i}",
            "y0": int(y0), "y1": int(y1),
            "crop": frame[int(y0):int(y1)],
            "offset": (0, int(y0)),        # add to a pane-local detection to get composite coords
            "size": (W, int(y1 - y0)),
        })
    return panes


def remap_to_composite(box_xyxy, offset):
    """Map a pane-local xyxy box back to full-composite coordinates."""
    ox, oy = offset
    x1, y1, x2, y2 = box_xyxy
    return (x1 + ox, y1 + oy, x2 + ox, y2 + oy)
