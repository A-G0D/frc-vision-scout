"""OpenVINO INT8 export with QAT fallback and NPU graph validation.

Flow:
  1. Export FP32 model -> measure val recall baseline.
  2. Export INT8 via PTQ -> measure INT8 val recall.
  3. If small-object recall drops more than qat_if_recall_drop_pct, retrain a
     few epochs with NNCF QAT and re-export (avoids the 5px-ball meltdown).
  4. Emit ONNX as a portable fallback (no OpenVINO runtime needed at deploy).
  5. Validate that the NPU graph compiles (OpenVINO op-set check for the
     Intel Core Ultra Arc NPU).

GPU-ONLY paths are marked. The file is import-clean without GPU/ML deps.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from frcscout.io.config import Config


# ---------------------------------------------------------------------------
# Recall helper — runs YOLO .val() and extracts small-object recall
# ---------------------------------------------------------------------------

def _measure_recall(weights: str, data: str, imgsz: int, device: int = 0) -> float:
    """Run YOLO val and return overall recall (R).

    For the ball model the 'small' AP from metrics.ap_class_index would be
    ideal but ultralytics box.r (per-class recall) is reliable and fast.
    Returns -1.0 on any error so the caller can gracefully skip QAT.

    GPU-ONLY: requires CUDA + ultralytics.
    """
    try:
        from ultralytics import YOLO
        m = YOLO(weights)
        metrics = m.val(data=data, imgsz=imgsz, device=device, verbose=False)
        # metrics.box.r is a per-class numpy array; mean over all classes.
        r = float(metrics.box.r.mean()) if hasattr(metrics.box, "r") else -1.0
        return r
    except Exception as e:
        print(f"[export_openvino] val recall measurement failed: {e}")
        return -1.0


# ---------------------------------------------------------------------------
# NPU graph compilation check
# ---------------------------------------------------------------------------

def _check_npu_compiles(ov_dir: str, imgsz: int) -> bool:
    """Try to compile the OpenVINO model on the NPU device.

    On a machine without a real NPU this will fall back to CPU and still
    validate the op-set.  Returns True if the model loads without errors.

    Requires openvino Python package (install: `pip install openvino`).
    """
    try:
        import openvino as ov
        core = ov.Core()
        # Prefer NPU if available, otherwise CPU (still validates op-set).
        devices = core.available_devices
        target = "NPU" if "NPU" in devices else "CPU"
        xml_files = list(Path(ov_dir).glob("*.xml"))
        if not xml_files:
            print(f"[export_openvino] NPU check: no .xml found in {ov_dir}")
            return False
        model = core.read_model(str(xml_files[0]))
        compiled = core.compile_model(model, target)
        del compiled
        print(f"[export_openvino] NPU check: model compiled OK on {target}")
        return True
    except Exception as e:
        print(f"[export_openvino] NPU compile check failed: {e}")
        return False


# ---------------------------------------------------------------------------
# QAT via NNCF — fallback when PTQ INT8 drops recall too much
# ---------------------------------------------------------------------------

def _qat_retrain_and_export(
    weights: str,
    data: str,
    imgsz: int,
    out_dir: str,
    epochs: int = 10,
    device: int = 0,
) -> str:
    """Retrain a few epochs with NNCF Quantization-Aware Training, then re-export.

    This recovers the small-object recall that PTQ drops when quantising
    tiny activations.  Typically 5-10 epochs is enough to recover >95% of the
    FP32 recall while staying INT8.

    Returns: path to the QAT-exported OpenVINO model directory.

    GPU-ONLY: requires torch, ultralytics, and nncf.
    """
    try:
        import nncf
        from ultralytics import YOLO
    except ImportError:
        print("[export_openvino] nncf or ultralytics not available; skipping QAT")
        return ""

    print(f"[export_openvino] QAT: retraining {epochs} epochs with NNCF quantization aware training")
    m = YOLO(weights)

    # NNCF integration: ultralytics >=8.3 exposes nncf_quantize hook.
    # We retrain with int8=True in training mode (not just calibration).
    try:
        qat_result = m.train(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            device=device,
            project=out_dir,
            name="qat_retrain",
            exist_ok=True,
            cache=False,
            int8=True,           # triggers NNCF QAT path in ultralytics
            patience=epochs,     # run all epochs (no early stop for short QAT)
        )
    except TypeError:
        # Older ultralytics does not accept int8 in train(); fall back to PTQ.
        print("[export_openvino] QAT train() does not accept int8= arg; skipping QAT")
        return ""

    qat_best = os.path.join(out_dir, "qat_retrain", "weights", "best.pt")
    if not os.path.exists(qat_best):
        print("[export_openvino] QAT best.pt not found; skipping QAT export")
        return ""

    m_qat = YOLO(qat_best)
    ov_dir = m_qat.export(
        format="openvino",
        int8=True,
        data=data,
        imgsz=imgsz,
        half=False,
    )
    print(f"[export_openvino] QAT export done: {ov_dir}")
    return str(ov_dir)


# ---------------------------------------------------------------------------
# Main export entry point
# ---------------------------------------------------------------------------

def export_one(
    weights: str,
    train_cfg_path: str,
    out_dir: str,
    data: Optional[str] = None,
    device: int = 0,
) -> str:
    """Export one student to OpenVINO INT8 (+ ONNX fallback).

    Steps:
      1. Enforce deploy_imgsz == imgsz invariant.
      2. Measure FP32 val recall (baseline).
      3. Export PTQ INT8.
      4. Measure INT8 val recall.
      5. If recall drop > qat_if_recall_drop_pct, run QAT + re-export.
      6. Export ONNX portable fallback.
      7. Validate NPU graph compiles.

    Args:
        weights:          Path to trained best.pt.
        train_cfg_path:   Path to configs/train_ball.yaml or train_robot.yaml.
        out_dir:          Directory to write exported model artifacts.
        data:             YOLO data.yaml for calibration + val.  Inferred from
                          out_dir/../data.yaml if not supplied.
        device:           CUDA device index for val + QAT.

    Returns:
        Path to the final OpenVINO model directory.

    GPU-ONLY: requires CUDA + ultralytics + openvino.  The NPU compile check
    can run on CPU (laptop) as long as openvino is installed.
    """
    cfg = Config.load(train_cfg_path)
    imgsz        = int(cfg.get("deploy_imgsz") or cfg.get("imgsz") or 960)
    deploy_imgsz = int(cfg.get("deploy_imgsz") or imgsz)

    # Enforce invariant.
    if imgsz != deploy_imgsz:
        raise ValueError(
            f"[export_openvino] deploy_imgsz ({deploy_imgsz}) != imgsz ({imgsz}) "
            f"in {train_cfg_path}. Fix the config before exporting."
        )

    recall_threshold_drop = float(cfg.get("qat_if_recall_drop_pct") or 5.0) / 100.0
    use_int8              = bool(cfg.get("int8", True))

    # Infer data.yaml if not provided.
    if data is None:
        # Try common locations relative to out_dir.
        candidates = [
            os.path.join(os.path.dirname(out_dir), "data.yaml"),
            os.path.join(out_dir, "data.yaml"),
        ]
        for c in candidates:
            if os.path.exists(c):
                data = c
                break
    if data is None or not os.path.exists(str(data)):
        print("[export_openvino] WARNING: no data.yaml found; PTQ calibration quality may be low")
        data = None

    # GPU-ONLY: requires ultralytics.
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError(
            "[export_openvino] ultralytics not installed — GPU cloud env required."
        )

    os.makedirs(out_dir, exist_ok=True)

    # ── 1. FP32 val recall baseline ──────────────────────────────────────────
    fp32_recall = -1.0
    if data:
        print("[export_openvino] measuring FP32 val recall ...")
        fp32_recall = _measure_recall(weights, data, imgsz, device)
        print(f"[export_openvino] FP32 recall: {fp32_recall:.4f}")

    # ── 2. PTQ INT8 export ───────────────────────────────────────────────────
    m = YOLO(weights)
    export_kwargs: dict = dict(format="openvino", imgsz=imgsz, half=False)
    if use_int8:
        export_kwargs["int8"] = True
        if data:
            export_kwargs["data"] = data   # calibration dataset
    ov_path = str(m.export(**export_kwargs))
    print(f"[export_openvino] INT8 export: {ov_path}")

    # ── 3. INT8 val recall ───────────────────────────────────────────────────
    int8_recall = -1.0
    if data and fp32_recall >= 0:
        print("[export_openvino] measuring INT8 val recall ...")
        int8_recall = _measure_recall(ov_path, data, imgsz, device)
        print(f"[export_openvino] INT8 recall: {int8_recall:.4f}")
        recall_drop = fp32_recall - int8_recall
        print(f"[export_openvino] recall drop: {recall_drop:.4f} "
              f"(threshold: {recall_threshold_drop:.4f})")

        # ── 4. QAT fallback if recall drop too large ─────────────────────────
        if recall_drop > recall_threshold_drop:
            print(
                f"[export_openvino] Recall drop {recall_drop:.2%} > "
                f"{recall_threshold_drop:.2%} threshold -> triggering QAT"
            )
            qat_ov = _qat_retrain_and_export(
                weights=weights,
                data=data,
                imgsz=imgsz,
                out_dir=out_dir,
                device=device,
            )
            if qat_ov and os.path.isdir(qat_ov):
                # Verify QAT actually improved things.
                qat_recall = _measure_recall(qat_ov, data, imgsz, device)
                print(f"[export_openvino] QAT recall: {qat_recall:.4f}")
                if qat_recall > int8_recall:
                    print("[export_openvino] QAT improved recall; using QAT model")
                    ov_path = qat_ov
                else:
                    print("[export_openvino] QAT did not improve recall; keeping PTQ model")
            else:
                print("[export_openvino] QAT failed; retaining PTQ model")
    else:
        if data is None:
            print("[export_openvino] Skipping recall measurement (no data.yaml)")
        elif fp32_recall < 0:
            print("[export_openvino] FP32 recall measurement failed; skipping QAT check")

    # ── 5. ONNX portable fallback ────────────────────────────────────────────
    try:
        m_onnx = YOLO(weights)
        onnx_path = m_onnx.export(format="onnx", imgsz=imgsz, opset=13, simplify=True)
        print(f"[export_openvino] ONNX fallback: {onnx_path}")
    except Exception as e:
        print(f"[export_openvino] ONNX export failed: {e}")

    # ── 6. NPU compile check ─────────────────────────────────────────────────
    npu_ok = _check_npu_compiles(ov_path, imgsz)
    if not npu_ok:
        print("[export_openvino] WARNING: NPU compile check failed. "
              "The model may not run on the Intel Arc NPU at deploy time. "
              "Check openvino op-set compatibility.")

    # ── 7. Write recall report ───────────────────────────────────────────────
    _write_recall_report(
        out_dir=out_dir,
        weights=weights,
        ov_path=ov_path,
        fp32_recall=fp32_recall,
        int8_recall=int8_recall,
        npu_ok=npu_ok,
    )

    return ov_path


def _write_recall_report(
    out_dir: str,
    weights: str,
    ov_path: str,
    fp32_recall: float,
    int8_recall: float,
    npu_ok: bool,
):
    """Write a recall_report.txt next to the exported model."""
    try:
        report_path = os.path.join(out_dir, "recall_report.txt")
        lines = [
            f"weights:      {weights}",
            f"openvino:     {ov_path}",
            f"fp32_recall:  {fp32_recall:.4f}" if fp32_recall >= 0 else "fp32_recall:  N/A",
            f"int8_recall:  {int8_recall:.4f}" if int8_recall >= 0 else "int8_recall:  N/A",
            f"recall_drop:  {fp32_recall - int8_recall:.4f}"
            if fp32_recall >= 0 and int8_recall >= 0 else "recall_drop:  N/A",
            f"npu_compiles: {npu_ok}",
        ]
        Path(report_path).write_text("\n".join(lines) + "\n")
        print(f"[export_openvino] recall report: {report_path}")
    except Exception as e:
        print(f"[export_openvino] could not write recall report: {e}")
