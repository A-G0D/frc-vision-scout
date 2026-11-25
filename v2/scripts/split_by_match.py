"""Match-level train/val/test split — the fix for v1's frame-level leakage.

v1 sampled one 4-min clip at 2fps and split RANDOMLY, so val frames sat 0.5s from their
train neighbors (near-duplicates) and "90.7% mAP" was partly memorization. Here, whole
MATCHES go entirely to train, val, or test. The test match is never seen until the gate.

Writes dataset_ball/ and dataset_robot/ with data.yaml pointing at the splits.
"""
from __future__ import annotations

import argparse

from frcscout.io.config import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sources.yaml")
    a = ap.parse_args()
    cfg = Config.load(a.config)
    matches = cfg.get("matches", [])
    test = [m for m in matches if m.get("split") == "test"]
    val = [m for m in matches if m.get("split") == "val"]
    train = [m for m in matches if m.get("split", "train") == "train"]
    assert test, "Designate at least one match as split: test — it is never trained on."
    print(f"train={len(train)} val={len(val)} test={len(test)} matches (no frame leakage)")
    # TODO: assemble images/labels per split from reviewed labels; write data.yaml x2.


if __name__ == "__main__":
    main()
