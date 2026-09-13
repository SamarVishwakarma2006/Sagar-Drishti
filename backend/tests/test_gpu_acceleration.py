"""
Sagar-Drishti — GPU Hardware Acceleration & DirectML Test Suite
Tests AMD Radeon / DirectML execution provider and hardware acceleration configs.
"""

import pytest
import numpy as np
from app.services.gpu_accelerator import (
    detect_system_gpus,
    DirectMLManager,
    HardwareAccelerationManager,
    gpu_manager,
)


def test_detect_system_gpus():
    """Verifies GPU detection retrieves device metadata."""
    devices = detect_system_gpus()
    assert len(devices) > 0
    primary = devices[0]
    assert primary.name != ""
    assert primary.vendor in ("AMD", "Intel", "NVIDIA", "Generic")


def test_directml_availability():
    """Verifies DirectMLExecutionProvider is present in ONNX Runtime."""
    dml_mgr = DirectMLManager()
    assert dml_mgr.is_available is True
    assert "DmlExecutionProvider" in dml_mgr.available_providers


def test_directml_priority_order():
    """Verifies priority provider ordering has DmlExecutionProvider first."""
    dml_mgr = DirectMLManager()
    providers = dml_mgr.get_providers_priority(prefer_gpu=True)
    assert providers[0] == "DmlExecutionProvider"


def test_directml_execution_benchmark():
    """Verifies DirectML executes tensor operations accurately on the AMD GPU."""
    dml_mgr = DirectMLManager()
    bench = dml_mgr.run_benchmark(dim=256, iterations=5)
    assert bench["status"] == "SUCCESS"
    assert bench["directml_available"] is True
    assert "DmlExecutionProvider" in bench["active_providers"]
    assert bench["numerical_verification_passed"] is True
    assert bench["avg_latency_ms"] > 0.0
    assert bench["throughput_gflops"] > 0.0


def test_hardware_acceleration_manager():
    """Verifies complete hardware acceleration manager summary and tree configurations."""
    mgr = HardwareAccelerationManager()
    summary = mgr.get_summary()

    assert "platform" in summary
    assert "primary_gpu" in summary
    assert summary["directml_available"] is True

    tree_cfg = summary["tree_acceleration"]
    assert "xgboost" in tree_cfg
    assert "lightgbm" in tree_cfg
    assert "tensor_inference" in tree_cfg

    assert tree_cfg["xgboost"]["tree_method"] == "hist"
    assert tree_cfg["tensor_inference"]["provider"] == "DmlExecutionProvider"
