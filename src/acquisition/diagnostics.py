"""Runtime evidence, including a real CUDA compute compatibility probe.

Torch is optional for geometry-only mode and imported only on request. Reporting
CUDA availability alone never labels inference as GPU accelerated.
"""
from __future__ import annotations

import importlib.metadata
import os
import platform
from typing import Any


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _host_memory() -> dict[str, int | None]:
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {"total_bytes": int(mem.total), "available_bytes": int(mem.available)}
    except (ImportError, OSError):
        pass
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        return {"total_bytes": page * os.sysconf("SC_PHYS_PAGES"),
                "available_bytes": page * os.sysconf("SC_AVPHYS_PAGES")}
    except (AttributeError, OSError, ValueError):
        return {"total_bytes": None, "available_bytes": None}


def runtime_diagnostics() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu_count": os.cpu_count(), "host_memory": _host_memory(),
        "versions": {name: _version(name) for name in ("numpy", "opencv-python", "PyQt6", "torch", "torchvision", "anomalib")},
        "torch_available": False, "cuda_available": False, "cuda_compute_verified": False,
        "pytorch_cuda_version": None, "gpu_name": None, "gpu_memory": None,
        "device": "cpu", "inference_device": "not_loaded", "error": None,
    }
    try:
        import torch
    except Exception as exc:
        report["error"] = f"PyTorch unavailable; geometry-only CPU operation is available: {exc}"
        return report
    report["torch_available"] = True
    report["pytorch_cuda_version"] = torch.version.cuda
    try:
        report["cuda_available"] = bool(torch.cuda.is_available())
        if not report["cuda_available"]:
            report["error"] = "CUDA unavailable. Neural inference will require CPU or a compatible CUDA PyTorch installation."
            return report
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        report.update(gpu_name=properties.name,
                      gpu_compute_capability=[properties.major, properties.minor],
                      pytorch_cuda_architectures=list(torch.cuda.get_arch_list()))
        # Blackwell can be advertised by the driver while the installed wheel
        # lacks executable kernels. An actual kernel + synchronize catches that.
        with torch.inference_mode():
            probe = torch.ones((32, 32), device=f"cuda:{index}")
            output = probe @ probe
            torch.cuda.synchronize(index)
            if float(output[0, 0].item()) != 32.0:
                raise RuntimeError("CUDA compute probe returned an unexpected value")
            del output, probe
        free, total = torch.cuda.mem_get_info(index)
        report.update(cuda_compute_verified=True, device=f"cuda:{index}",
                      gpu_memory={"free_bytes": int(free), "total_bytes": int(total),
                                  "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                                  "reserved_bytes": int(torch.cuda.memory_reserved(index))})
    except Exception as exc:
        report["error"] = f"CUDA compute verification failed; use CPU until the installation is corrected: {exc}"
        report["device"] = "cpu"
    return report
