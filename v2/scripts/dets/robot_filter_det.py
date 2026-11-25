"""Eval shim: score src/frcscout/detect/robot_filter.filter_robot_dets vs ground truth.
Run: PYTHONPATH=src python scripts/eval_robots.py --detector-module scripts/dets/robot_filter_det.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/ -> eval_robots helpers
from eval_robots import _net_boxes, _in_poly                    # noqa: E402
from frcscout.detect.robot_filter import filter_robot_dets       # noqa: E402


def detect(frame):
    H, W = frame.shape[:2]
    raw = [b for b in _net_boxes(frame) if _in_poly(b, W, H)]
    return filter_robot_dets(frame, raw, W, H)
