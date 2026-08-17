from __future__ import annotations

import gc
import time

import torch
from loguru import logger


class GarbageCollection:
    def __init__(self, gc_freq: int = 1000):
        assert gc_freq > 0, "gc_freq must be a positive integer"
        self.gc_freq = gc_freq
        gc.disable()
        self.collect("Initial GC collection")

    def run(self, step_count: int):
        if step_count > 1 and step_count % self.gc_freq == 0:
            self.collect("Performing periodic GC collection")

    @staticmethod
    def collect(reason: str, generation: int = 1):
        begin = time.monotonic()
        gc.collect(generation)
        logger.info("[GC] {} took {:.2f} seconds", reason, time.monotonic() - begin)


def get_peak_flops(device_name: str) -> int:
    if "A100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/a100/
        return int(312e12)
    elif "H100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        # NOTE: Specifications are one-half lower without sparsity.
        if "NVL" in device_name:
            return int(835e12)
        elif "PCIe" in device_name:
            return int(756e12)
        else:  # for H100 SXM and other variants
            return int(989e12)
    elif "H200" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h200/
        return int(989e12)
    elif "B200" in device_name:
        # data from https://nvdam.widen.net/s/wwnsxrhm2w/blackwell-datasheet-3384703
        return int(2.25e15)
    elif "MI355X" in device_name:
        # MI355X data from https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html
        return int(2500e12)
    elif "MI300X" in device_name or "MI325X" in device_name:
        # MI300X data from https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
        # MI325X data from https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html
        return int(1300e12)
    elif "MI250X" in device_name:
        # data from https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html (per GCD)
        return int(191.5e12)
    elif "Data Center GPU Max 1550" in device_name:
        # Also known as Ponte Vecchio (PVC).
        # data from https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html
        # Dot Product Accumulate Systolic (DPAS):
        # - Freq: 1300MHz
        # - #ops: 512
        # Full EU mode (i.e. 512 max compute units): 340.8 TFLOPS (BF16)
        # Standard EU mode (i.e. 448 max compute units): 298.2 TFLOPS (BF16)
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6
    elif "l40s" in device_name:
        # data from: "https://resources.nvidia.com/en-us-l40s/l40s-datasheet-28413"
        return int(362e12)

    else:  # for other GPU types, assume A100
        logger.warning(f"Peak flops undefined for: {device_name}, fallback to A100")
        return int(312e12)


def _round_up(x: int, y: int) -> int:
    """Round up x to the nearest multiple of y."""
    x_ceil_div_y = (x + y - 1) // y
    return x_ceil_div_y * y
