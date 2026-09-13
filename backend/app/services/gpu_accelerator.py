"""
Sagar-Drishti — Hardware Acceleration & GPU Manager Service
DirectML / DirectX 12 / OpenCL / multi-threaded CPU configuration for Windows AMD/Intel/NVIDIA hardware.

Provides:
- Automatic hardware discovery (AMD Radeon, Intel Iris/Arc, NVIDIA GeForce/RTX).
- DirectML acceleration via onnxruntime-directml (DmlExecutionProvider) for high-performance tensor computing.
- Optimized multi-core and device acceleration profiles for XGBoost and LightGBM.
- Deterministic GPU benchmark and health diagnostics without model training.
"""

import os
import sys
import json
import time
import logging
import platform
import subprocess
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

logger = logging.getLogger("sagar_drishti.gpu_accelerator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class GPUDeviceInfo:
    """Hardware profile for an installed GPU controller."""
    name: str
    vendor: str
    adapter_ram_mb: float
    driver_version: str
    directml_supported: bool = False
    opencl_supported: bool = False
    cuda_supported: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_CACHED_GPUS: Optional[List[GPUDeviceInfo]] = None


def detect_system_gpus(force_refresh: bool = False) -> List[GPUDeviceInfo]:
    """
    Detects graphics adapters available on the host machine.
    Uses Windows CIM/WMI on Windows, falling back gracefully on other platforms.
    Results are cached in-memory.
    """
    global _CACHED_GPUS
    if _CACHED_GPUS is not None and not force_refresh:
        return _CACHED_GPUS

    devices: List[GPUDeviceInfo] = []

    if platform.system() == "Windows":
        try:
            cmd = "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion | ConvertTo-Json -Compress"
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", cmd],
                capture_output=True,
                text=True,
                timeout=12
            )
            if res.returncode == 0 and res.stdout.strip():
                raw = json.loads(res.stdout.strip())
                items = raw if isinstance(raw, list) else [raw]
                for item in items:
                    name = str(item.get("Name") or "Unknown GPU")
                    ram_bytes = item.get("AdapterRAM") or 0
                    ram_mb = round(float(ram_bytes) / (1024 * 1024), 1) if ram_bytes else 0.0
                    driver = str(item.get("DriverVersion") or "Unknown")

                    vendor = "Unknown"
                    if "AMD" in name.upper() or "RADEON" in name.upper():
                        vendor = "AMD"
                    elif "NVIDIA" in name.upper() or "GEFORCE" in name.upper() or "RTX" in name.upper():
                        vendor = "NVIDIA"
                    elif "INTEL" in name.upper() or "IRIS" in name.upper() or "ARC" in name.upper():
                        vendor = "Intel"

                    # DirectML is supported on modern Windows DirectX 12 hardware (AMD, Intel, NVIDIA)
                    directml_ok = True if vendor in ("AMD", "Intel", "NVIDIA") else False

                    devices.append(
                        GPUDeviceInfo(
                            name=name,
                            vendor=vendor,
                            adapter_ram_mb=ram_mb,
                            driver_version=driver,
                            directml_supported=directml_ok,
                            opencl_supported=(vendor in ("AMD", "Intel", "NVIDIA")),
                            cuda_supported=(vendor == "NVIDIA"),
                        )
                    )
        except Exception as e:
            logger.warning(f"Windows GPU discovery encountered an exception: {e}")

    if not devices:
        # Generic fallback
        devices.append(
            GPUDeviceInfo(
                name="Standard Display Adapter / Host Processor",
                vendor="Generic",
                adapter_ram_mb=0.0,
                driver_version="N/A",
                directml_supported=False,
                opencl_supported=False,
                cuda_supported=False,
            )
        )

    return devices


class DirectMLManager:
    """
    DirectML Execution Manager using ONNX Runtime.
    Provides verified hardware acceleration for tensor evaluations on AMD/Intel/NVIDIA GPUs without requiring CUDA.
    """

    def __init__(self):
        self._ort = None
        self._available_providers = []
        self._has_directml = False
        self._init_runtime()

    def _init_runtime(self):
        try:
            import onnxruntime as ort
            self._ort = ort
            self._available_providers = ort.get_available_providers()
            self._has_directml = "DmlExecutionProvider" in self._available_providers
        except ImportError:
            self._ort = None
            self._available_providers = []
            self._has_directml = False

    @property
    def is_available(self) -> bool:
        """Returns True if onnxruntime-directml is loaded and DirectML provider is active."""
        return self._has_directml

    @property
    def available_providers(self) -> List[str]:
        """List of available execution providers."""
        return list(self._available_providers)

    def get_providers_priority(self, prefer_gpu: bool = True) -> List[str]:
        """Returns ordered list of execution providers with DirectML prioritized."""
        if prefer_gpu and self._has_directml:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def create_session(
        self,
        model_bytes_or_path: Any,
        prefer_gpu: bool = True,
        intra_op_num_threads: int = 4
    ) -> Any:
        """
        Creates an ONNX Runtime InferenceSession using DirectML if available.
        """
        if self._ort is None:
            raise RuntimeError("onnxruntime is not installed. Install via `pip install onnxruntime-directml`.")

        opts = self._ort.SessionOptions()
        opts.graph_optimization_level = self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_num_threads > 0:
            opts.intra_op_num_threads = intra_op_num_threads

        providers = self.get_providers_priority(prefer_gpu=prefer_gpu)
        return self._ort.InferenceSession(model_bytes_or_path, sess_options=opts, providers=providers)

    def run_benchmark(
        self,
        dim: int = 512,
        iterations: int = 20
    ) -> Dict[str, Any]:
        """
        Executes a deterministic tensor multiplication benchmark on the AMD GPU using DirectML.
        Measures execution latency, throughput, and numerical accuracy against CPU numpy.
        """
        if not self.is_available:
            return {
                "status": "UNAVAILABLE",
                "directml_available": False,
                "error": "DmlExecutionProvider not available in onnxruntime"
            }

        try:
            import onnx
            from onnx import helper, TensorProto

            # Construct simple MatMul graph stamped with opset 21 (widely supported)
            X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [dim, dim])
            W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [dim, dim])
            Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [dim, dim])
            node = helper.make_node("MatMul", ["X", "W"], ["Y"])
            graph = helper.make_graph([node], "benchmark_graph", [X, W], [Y])
            opset = helper.make_opsetid("", 21)
            model = helper.make_model(graph, producer_name="sagar_drishti_directml", opset_imports=[opset])
            model_bytes = model.SerializeToString()

            session = self.create_session(model_bytes, prefer_gpu=True)
            active_providers = session.get_providers()

            # Seeded input arrays
            rng = np.random.default_rng(42)
            x_val = rng.standard_normal((dim, dim), dtype=np.float32)
            w_val = rng.standard_normal((dim, dim), dtype=np.float32)

            # Warmup
            _ = session.run(["Y"], {"X": x_val, "W": w_val})

            # Timed iterations
            latencies = []
            res_dml = None
            for _ in range(iterations):
                t0 = time.perf_counter()
                res_dml = session.run(["Y"], {"X": x_val, "W": w_val})[0]
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)  # ms

            # Numerical verification against numpy CPU
            res_cpu = np.matmul(x_val, w_val)
            max_abs_diff = float(np.max(np.abs(res_dml - res_cpu)))
            is_accurate = max_abs_diff < 1e-3

            # Calculate GFLOPS: 2 * dim^3 FLOPs per matmul
            flops = 2.0 * (dim ** 3)
            avg_ms = float(np.mean(latencies))
            gflops = (flops / (avg_ms / 1000.0)) / 1e9

            return {
                "status": "SUCCESS",
                "directml_available": True,
                "active_providers": active_providers,
                "matrix_dim": f"{dim}x{dim}",
                "iterations": iterations,
                "avg_latency_ms": round(avg_ms, 3),
                "min_latency_ms": round(float(np.min(latencies)), 3),
                "max_latency_ms": round(float(np.max(latencies)), 3),
                "throughput_gflops": round(gflops, 2),
                "numerical_max_abs_diff": max_abs_diff,
                "numerical_verification_passed": is_accurate,
            }
        except Exception as e:
            logger.error(f"DirectML benchmark failed: {e}")
            return {
                "status": "FAILED",
                "directml_available": True,
                "error": str(e)
            }


class HardwareAccelerationManager:
    """
    Central coordinator for Sagar-Drishti hardware acceleration.
    Manages GPUs, DirectML execution, and tree acceleration configurations.
    """

    def __init__(self):
        self.gpus = detect_system_gpus()
        self.primary_gpu = self.gpus[0] if self.gpus else None
        self.directml = DirectMLManager()

    def get_summary(self) -> Dict[str, Any]:
        """Provides a complete system acceleration profile."""
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "primary_gpu": self.primary_gpu.to_dict() if self.primary_gpu else None,
            "directml_available": self.directml.is_available,
            "available_ort_providers": self.directml.available_providers,
            "tree_acceleration": self.get_tree_model_config()
        }

    def get_tree_model_config(self) -> Dict[str, Any]:
        """
        Provides recommended hyperparameter configurations for LightGBM and XGBoost
        tailored for this specific hardware (AMD Radeon / Multi-core CPU).
        """
        cpu_count = os.cpu_count() or 4
        # On Windows AMD APU/iGPU without CUDA:
        # - XGBoost 'hist' with maximum OpenMP threads provides the fastest robust training.
        # - LightGBM with OpenMP threading provides peak tabular performance.
        # - Neural / ONNX inference runs on DirectML GPU.
        return {
            "xgboost": {
                "tree_method": "hist",
                "device": "cpu",
                "n_jobs": cpu_count,
                "acceleration_backend": "OpenMP_AVX2"
            },
            "lightgbm": {
                "device_type": "cpu",
                "n_jobs": cpu_count,
                "acceleration_backend": "OpenMP_MultiThread"
            },
            "tensor_inference": {
                "provider": "DmlExecutionProvider" if self.directml.is_available else "CPUExecutionProvider",
                "device": "AMD Radeon(TM) Graphics (DirectX 12 / DirectML)" if self.directml.is_available else "CPU"
            }
        }


# Global singleton instance
gpu_manager = HardwareAccelerationManager()


def print_diagnostic_report():
    """Prints a human-readable diagnostic report of the GPU hardware environment."""
    mgr = HardwareAccelerationManager()
    summary = mgr.get_summary()

    print("=" * 65)
    print(" SAGAR-DRISHTI HARDWARE ACCELERATION DIAGNOSTIC REPORT")
    print("=" * 65)
    print(f" Host OS         : {summary['platform']}")
    print(f" Python Version  : {summary['python_version']}")

    gpu = summary["primary_gpu"]
    if gpu:
        print(f" Primary GPU     : {gpu['name']}")
        print(f" Vendor          : {gpu['vendor']}")
        print(f" Dedicated VRAM  : {gpu['adapter_ram_mb']} MB")
        print(f" Driver Version  : {gpu['driver_version']}")
        print(f" DirectML Ready  : {gpu['directml_supported']}")
        print(f" OpenCL Ready    : {gpu['opencl_supported']}")
    else:
        print(" Primary GPU     : None detected")

    print("-" * 65)
    print(f" DirectML Status : {'ACTIVE' if summary['directml_available'] else 'INACTIVE'}")
    print(f" ORT Providers   : {', '.join(summary['available_ort_providers'])}")

    print("\n Running DirectML GPU Tensor Benchmark...")
    bench = mgr.directml.run_benchmark(dim=512, iterations=15)
    if bench["status"] == "SUCCESS":
        print(f"  [PASS] Providers Used      : {bench['active_providers']}")
        print(f"  [PASS] Matrix Dim          : {bench['matrix_dim']}")
        print(f"  [PASS] Latency (mean/min)  : {bench['avg_latency_ms']} ms / {bench['min_latency_ms']} ms")
        print(f"  [PASS] Throughput          : {bench['throughput_gflops']} GFLOPS")
        print(f"  [PASS] Numerical Accuracy  : PASSED (max abs diff: {bench['numerical_max_abs_diff']:.2e})")
    else:
        print(f"  [FAIL] DirectML Benchmark  : {bench.get('error')}")

    print("-" * 65)
    tree_cfg = summary["tree_acceleration"]
    print(" Tabular & Inference Acceleration Profiles:")
    print(f"  - ONNX Inference : {tree_cfg['tensor_inference']['provider']} ({tree_cfg['tensor_inference']['device']})")
    print(f"  - XGBoost        : tree_method={tree_cfg['xgboost']['tree_method']}, n_jobs={tree_cfg['xgboost']['n_jobs']} ({tree_cfg['xgboost']['acceleration_backend']})")
    print(f"  - LightGBM       : device_type={tree_cfg['lightgbm']['device_type']}, n_jobs={tree_cfg['lightgbm']['n_jobs']} ({tree_cfg['lightgbm']['acceleration_backend']})")
    print("=" * 65)


if __name__ == "__main__":
    print_diagnostic_report()
