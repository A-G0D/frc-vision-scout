"""Tests for refine/tta.py — test-time augmentation.

Two layers of tests:

1. PURE-LOGIC (no numpy/cv2 needed): the coordinate inversion lives in
   ``_View.invert_box`` and ``build_views``. These are exercised directly with
   hand-placed boxes, so flip/scale inversion exactness is verified on a bare
   interpreter. This is the load-bearing math.

2. END-TO-END (skipped without numpy): ``tta_detect``/``detect_fuse`` must build
   a real transformed frame per view, which inherently needs an array backend.
   These use real ndarrays + a MOCK detector that reads the handed frame's
   ``.shape`` and returns hand-placed boxes, proving the views are sized right
   AND that inverted boxes land on exactly-known source pixels.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from frcscout.io.types import Detection
from frcscout.refine import tta
from frcscout.refine.tta import build_views, detect_fuse, tta_detect

np = tta._np
needs_np = pytest.mark.skipif(np is None, reason="numpy not installed")


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class RecordingDetector:
    """Mock detect(frame) -> [Detection].

    Returns hand-placed boxes (interpreted as being in the handed frame's own
    coordinates) and records every frame shape it was called with, so tests can
    confirm which views were actually built and at what size.
    """

    def __init__(self, boxes_by_call=None, const_boxes=None):
        self.boxes_by_call = list(boxes_by_call) if boxes_by_call else None
        self.const_boxes = const_boxes or []
        self.seen_shapes = []
        self._i = 0

    def __call__(self, frame):
        self.seen_shapes.append(tuple(frame.shape))
        if self.boxes_by_call is not None:
            boxes = self.boxes_by_call[self._i]
            self._i += 1
        else:
            boxes = self.const_boxes
        return [
            Detection(cls=c, xyxy=tuple(map(float, b)), conf=cf, frame_idx=0)
            for (c, b, cf) in boxes
        ]


def _zeros(h, w):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _approx_box(b, expected, tol=1e-7):
    assert len(b) == 4
    for got, exp in zip(b, expected):
        assert got == pytest.approx(exp, abs=tol), f"{tuple(b)} != {tuple(expected)}"


def _one_view(W, H, **cfg):
    return build_views(W, H, include_identity=False, **cfg)[0]


# ===========================================================================
# PURE LOGIC: build_views (no numpy required)
# ===========================================================================

def test_identity_view_is_first_and_native_size():
    views = build_views(1920, 1080, scales=(1.0,), flips=(False,))
    assert views[0].name == "orig"
    assert (views[0].out_w, views[0].out_h) == (1920, 1080)
    assert views[0].flip is False


def test_views_dedup_by_realized_size_and_flip():
    # scale 1.0 upright duplicates the identity view -> collapsed.
    views = build_views(100, 50, scales=(1.0,), flips=(False, True), include_identity=True)
    keys = {(v.out_w, v.out_h, v.flip) for v in views}
    assert keys == {(100, 50, False), (100, 50, True)}
    assert len(views) == 2


def test_multiscale_view_count_and_sizes():
    views = build_views(200, 100, scales=(1.0, 1.5), flips=(False, True),
                        include_identity=True)
    # identity + {1.0,1.5} x {F,T}; the 1.0/F dup of identity is removed -> 4.
    assert len(views) == 4
    sizes = {(v.out_w, v.out_h) for v in views}
    assert (200, 100) in sizes
    assert (300, 150) in sizes  # 1.5x


def test_view_records_source_size():
    v = _one_view(640, 480, scales=(2.0,), flips=(False,))
    assert (v.src_w, v.src_h) == (640, 480)
    assert (v.out_w, v.out_h) == (1280, 960)


# ===========================================================================
# PURE LOGIC: invert_box exactness — flip
# ===========================================================================

def test_hflip_inversion_exact():
    """A box on the LEFT of the flipped frame inverts to the RIGHT in source."""
    v = _one_view(100, 60, scales=(1.0,), flips=(True,))
    # Mirror across width W=100: x1'=W-x2, x2'=W-x1; y unchanged.
    _approx_box(v.invert_box((10, 20, 30, 40)), (70.0, 20.0, 90.0, 40.0))


def test_identity_view_does_not_move_boxes():
    v = _one_view(800, 600, scales=(1.0,), flips=(False,))
    _approx_box(v.invert_box((100, 100, 200, 250)), (100.0, 100.0, 200.0, 250.0))


def test_hflip_is_involution():
    v = _one_view(256, 128, scales=(1.0,), flips=(True,))
    orig = (12.0, 34.0, 56.0, 78.0)
    _approx_box(v.invert_box(v.invert_box(orig)), orig)


def test_hflip_touching_edges():
    v = _one_view(100, 50, scales=(1.0,), flips=(True,))
    # full-width box stays full-width; right edge box -> left edge box.
    _approx_box(v.invert_box((0, 0, 100, 50)), (0.0, 0.0, 100.0, 50.0))
    _approx_box(v.invert_box((90, 0, 100, 50)), (0.0, 0.0, 10.0, 50.0))


# ===========================================================================
# PURE LOGIC: invert_box exactness — scale
# ===========================================================================

def test_upscale_inversion_exact():
    """Box detected in a 2x frame divides by 2 back to source."""
    v = _one_view(100, 80, scales=(2.0,), flips=(False,))  # view 200x160
    _approx_box(v.invert_box((40, 60, 80, 100)), (20.0, 30.0, 40.0, 50.0))


def test_downscale_inversion_exact():
    v = _one_view(100, 100, scales=(0.5,), flips=(False,))  # view 50x50
    _approx_box(v.invert_box((5, 5, 25, 25)), (10.0, 10.0, 50.0, 50.0))


def test_inversion_uses_realized_integer_size_not_float_scale():
    """When src*scale isn't integral, inversion must use the ROUNDED view size.

    src_w=101, scale=1.5 -> round(151.5)=152. A full-width box [0,152] must
    invert to exactly [0,101], not to 152 * 101/151.5.

    NB: round() is banker's rounding (half-to-even), so round(76.5)=76. The
    inversion divides by the REALIZED out size (76), which is exactly why it
    stays exact regardless of how the round broke the tie.
    """
    v = _one_view(101, 51, scales=(1.5,), flips=(False,))
    assert (v.out_w, v.out_h) == (152, 76)  # round(151.5)=152, round(76.5)=76 (even)
    _approx_box(v.invert_box((0.0, 0.0, float(v.out_w), float(v.out_h))),
                (0.0, 0.0, 101.0, 51.0))


# ===========================================================================
# PURE LOGIC: combined flip + scale
# ===========================================================================

def test_flip_and_scale_inversion_exact():
    """Forward = resize then flip; inverse = un-flip (in resized width) then /scale."""
    v = _one_view(100, 50, scales=(2.0,), flips=(True,))  # view 200x100, flipped
    # un-flip in out_w=200: x1'=200-50=150, x2'=200-10=190; then /2 -> 75,95.
    _approx_box(v.invert_box((10, 20, 50, 60)), (75.0, 10.0, 95.0, 30.0))


def test_round_trip_arbitrary_box_through_view():
    """invert_box is the exact inverse of the forward corner map for many configs."""
    W, H = 320, 240
    configs = [
        dict(scales=(1.0,), flips=(False,)),
        dict(scales=(1.0,), flips=(True,)),
        dict(scales=(1.5,), flips=(False,)),
        dict(scales=(0.75,), flips=(True,)),
        dict(scales=(2.0,), flips=(True,)),
    ]
    boxes_src = [(0, 0, 10, 10), (33, 44, 100, 200), (5.5, 9.25, 311.0, 233.0)]
    for cfg in configs:
        v = _one_view(W, H, **cfg)
        rx = v.out_w / W
        ry = v.out_h / H
        for (x1, y1, x2, y2) in boxes_src:
            fx1, fy1, fx2, fy2 = x1 * rx, y1 * ry, x2 * rx, y2 * ry
            if v.flip:
                fx1, fx2 = v.out_w - fx2, v.out_w - fx1
            _approx_box(v.invert_box((fx1, fy1, fx2, fy2)), (x1, y1, x2, y2))


def test_invert_box_reorders_corners():
    """Even if a detector hands x2<x1, the inverted box has x1<=x2, y1<=y2."""
    v = _one_view(100, 100, scales=(1.0,), flips=(False,))
    out = v.invert_box((30, 40, 10, 20))
    assert out[0] <= out[2] and out[1] <= out[3]
    _approx_box(out, (10.0, 20.0, 30.0, 40.0))


# ===========================================================================
# PURE LOGIC: IoU + NMS primitives
# ===========================================================================

def test_iou_identical_disjoint_and_partial():
    assert tta._iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert tta._iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # a=[0,0,10,10], b=[5,0,15,10] -> inter 50, union 150 -> 1/3.
    assert tta._iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_nms_keeps_highest_conf_in_cluster():
    d = [
        Detection("ball", (0, 0, 10, 10), 0.5, 0),
        Detection("ball", (1, 1, 11, 11), 0.9, 0),   # overlaps -> wins
        Detection("ball", (100, 100, 110, 110), 0.4, 0),
    ]
    kept = tta._greedy_nms(d, iou_threshold=0.5, class_aware=True)
    assert len(kept) == 2
    assert kept[0].conf == 0.9


def test_nms_class_aware_vs_agnostic():
    d = [
        Detection("ball", (0, 0, 10, 10), 0.9, 0),
        Detection("robot", (0, 0, 10, 10), 0.8, 0),
    ]
    assert len(tta._greedy_nms(d, 0.5, class_aware=True)) == 2
    assert len(tta._greedy_nms(d, 0.5, class_aware=False)) == 1


def test_nms_empty():
    assert tta._greedy_nms([], 0.5, True) == []


# ===========================================================================
# PURE LOGIC: _frame_hw tolerates tuple/ndarray-like
# ===========================================================================

def test_frame_hw_from_tuple_and_shape():
    assert tta._frame_hw((60, 100)) == (60, 100)
    assert tta._frame_hw((60, 100, 3)) == (60, 100)

    class HasShape:
        shape = (480, 640, 3)

    assert tta._frame_hw(HasShape()) == (480, 640)


def test_frame_hw_rejects_unsupported():
    with pytest.raises(TypeError):
        tta._frame_hw(object())


# ===========================================================================
# END-TO-END (real numpy frames + mock detector)
# ===========================================================================

@needs_np
def test_identity_only_passes_frame_through():
    det = RecordingDetector(const_boxes=[("robot", (100, 100, 200, 250), 0.5)])
    out = tta_detect(_zeros(600, 800), det, scales=(1.0,), flips=(False,),
                     include_identity=True)
    assert len(out) == 1
    _approx_box(out[0].xyxy, (100.0, 100.0, 200.0, 250.0))
    assert out[0].cls == "robot" and out[0].conf == 0.5
    assert det.seen_shapes == [(600, 800, 3)]


@needs_np
def test_e2e_hflip_frame_and_inversion_exact():
    """Real flipped ndarray: detector sees a 100-wide frame, box inverts exactly."""
    det = RecordingDetector(const_boxes=[("ball", (10, 20, 30, 40), 0.9)])
    out = tta_detect(_zeros(60, 100), det, scales=(1.0,), flips=(True,),
                     include_identity=False)
    assert det.seen_shapes[0][1] == 100  # un-resized width
    _approx_box(out[0].xyxy, (70.0, 20.0, 90.0, 40.0))


@needs_np
def test_e2e_upscale_builds_bigger_frame_and_inverts():
    det = RecordingDetector(const_boxes=[("ball", (40, 60, 80, 100), 0.7)])
    out = tta_detect(_zeros(80, 100), det, scales=(2.0,), flips=(False,),
                     include_identity=False)
    assert det.seen_shapes[0] == (160, 200, 3)  # 2x
    _approx_box(out[0].xyxy, (20.0, 30.0, 40.0, 50.0))


@needs_np
def test_e2e_flip_and_scale_together():
    det = RecordingDetector(const_boxes=[("ball", (10, 20, 50, 60), 0.8)])
    out = tta_detect(_zeros(50, 100), det, scales=(2.0,), flips=(True,),
                     include_identity=False)
    assert det.seen_shapes[0] == (100, 200, 3)
    _approx_box(out[0].xyxy, (75.0, 10.0, 95.0, 30.0))


@needs_np
def test_e2e_detector_called_once_per_distinct_view():
    det = RecordingDetector(const_boxes=[])
    tta_detect(_zeros(100, 200), det, scales=(1.0, 2.0), flips=(False, True),
               include_identity=True)
    shapes = set(det.seen_shapes)
    assert (100, 200, 3) in shapes      # native
    assert (200, 400, 3) in shapes      # 2x
    assert len(det.seen_shapes) == 4    # orig + (1.0/T, 2.0/F, 2.0/T); 1.0/F dups orig


# ===========================================================================
# END-TO-END: detect_fuse (union -> NMS)
# ===========================================================================

@needs_np
def test_fuse_merges_duplicate_across_views_keeping_highest_conf():
    """Same object upright and flipped lands on the same source box -> 1 kept."""
    # Flipped frame width 100: box [40,20,60,40] inverts to [40,20,60,40] (same).
    det = RecordingDetector(boxes_by_call=[
        [("ball", (40, 20, 60, 40), 0.6)],   # orig view
        [("ball", (40, 20, 60, 40), 0.9)],   # hflip view -> same source box
    ])
    fused = detect_fuse(_zeros(60, 100), det, scales=(1.0,), flips=(False, True),
                        include_identity=True, iou_threshold=0.5)
    assert len(fused) == 1
    assert fused[0].conf == 0.9
    _approx_box(fused[0].xyxy, (40.0, 20.0, 60.0, 40.0))


@needs_np
def test_fuse_keeps_distinct_objects():
    det = RecordingDetector(boxes_by_call=[
        [("ball", (5, 5, 15, 15), 0.8), ("ball", (80, 40, 95, 55), 0.7)],
        [],
    ])
    fused = detect_fuse(_zeros(60, 100), det, scales=(1.0,), flips=(False, True),
                        include_identity=True, iou_threshold=0.5)
    assert len(fused) == 2


@needs_np
def test_fuse_class_aware_keeps_ball_and_robot_on_same_box():
    det = RecordingDetector(boxes_by_call=[
        [("ball", (10, 10, 50, 50), 0.9), ("robot", (10, 10, 50, 50), 0.8)],
        [],
    ])
    fused = detect_fuse(_zeros(60, 100), det, scales=(1.0,), flips=(False, True),
                        include_identity=True, iou_threshold=0.5, class_aware=True)
    assert {d.cls for d in fused} == {"ball", "robot"}


@needs_np
def test_fuse_class_agnostic_merges_regardless_of_class():
    det = RecordingDetector(boxes_by_call=[
        [("ball", (10, 10, 50, 50), 0.9), ("robot", (10, 10, 50, 50), 0.8)],
        [],
    ])
    fused = detect_fuse(_zeros(60, 100), det, scales=(1.0,), flips=(False, True),
                        include_identity=True, iou_threshold=0.5, class_aware=False)
    assert len(fused) == 1
    assert fused[0].cls == "ball"  # higher conf wins


@needs_np
def test_tta_detect_returns_raw_union_without_nms():
    det = RecordingDetector(boxes_by_call=[
        [("ball", (40, 20, 60, 40), 0.6)],
        [("ball", (40, 20, 60, 40), 0.9)],  # inverts onto the same source box
    ])
    union = tta_detect(_zeros(60, 100), det, scales=(1.0,), flips=(False, True),
                       include_identity=True)
    assert len(union) == 2  # union, no suppression


@needs_np
def test_empty_detector_yields_empty():
    det = RecordingDetector(const_boxes=[])
    assert tta_detect(_zeros(60, 100), det) == []
    assert detect_fuse(_zeros(60, 100), det) == []
