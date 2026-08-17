import os
import time
from collections import namedtuple
from datetime import datetime, timezone
from typing import Any

import torch
from loguru import logger
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from torchfeather.components.lr_scheduler import LRSchedulersContainer
from torchfeather.components.optimizer import OptimizersContainer
from torchfeather.config import JobConfig
from torchfeather.distributed import ParallelDims
from torchfeather.tools import utils
from torchfeather.tools.device_utils import device_module, device_type

DeviceMemStats = namedtuple(
    "DeviceMemStats",
    [
        "max_active_gib",
        "max_active_pct",
        "max_reserved_gib",
        "max_reserved_pct",
        "num_alloc_retries",
        "num_ooms",
    ],
)


def _tensor_norm(tensor: torch.Tensor) -> torch.Tensor:
    """L2 norm of a tensor, handling DTensors correctly."""
    norm = torch.nn.utils.get_total_norm([tensor], 2.0)
    if isinstance(norm, DTensor):
        norm = norm.full_tensor()
    return norm


@torch.no_grad()
def collect_parameter_norm_metrics(
    model_parts: list[torch.nn.Module],
    pp_mesh: DeviceMesh | None = None,
) -> dict[str, float]:
    """Compute per-parameter gradient and parameter norms for logging."""
    metrics: dict[str, float] = {}

    for model in model_parts:
        for name, param in model.named_parameters():
            metrics[f"param_norms/{name}"] = _tensor_norm(param.detach()).item()
            if param.grad is not None:
                metrics[f"param_grad_norms/{name}"] = _tensor_norm(
                    param.grad.detach()
                ).item()

    # When pipeline parallelism is enabled, each rank only holds a subset of the model's parameters. Gather metrics from all PP ranks so the logging rank can report norms for every parameter.
    if pp_mesh is not None:
        gathered: list[dict[str, float] | None] = [None] * pp_mesh.size()
        torch.distributed.all_gather_object(
            gathered, metrics, group=pp_mesh.get_group()
        )
        merged: dict[str, float] = {}
        for d in gathered:
            if d is not None:
                merged.update(d)
        metrics = merged

    return metrics


class DeviceMemoryMonitor:
    def __init__(self, device: str = f"{device_type}:0"):
        self.device = torch.device(device)  # device object
        self.device_name = device_module.get_device_name(self.device)
        self.device_capacity = device_module.get_device_properties(
            self.device
        ).total_memory
        self.device_capacity_gib = self._to_gib(self.device_capacity)

        device_module.reset_peak_memory_stats()
        device_module.empty_cache()

    def _to_gib(self, memory_in_bytes):
        # NOTE: GiB (gibibyte) is 1024, vs GB is 1000
        _gib_in_bytes = 1024 * 1024 * 1024
        memory_in_gib = memory_in_bytes / _gib_in_bytes
        return memory_in_gib

    def _to_pct(self, memory):
        return 100 * memory / self.device_capacity

    def get_peak_stats(self):
        device_info = device_module.memory_stats(self.device)

        max_active = device_info.get("active_bytes.all.peak", -1)
        max_active_gib = self._to_gib(max_active)
        max_active_pct = self._to_pct(max_active)

        max_reserved = device_info.get("reserved_bytes.all.peak", -1)
        max_reserved_gib = self._to_gib(max_reserved)
        max_reserved_pct = self._to_pct(max_reserved)

        num_retries = device_info.get("num_alloc_retries", -1)
        num_ooms = device_info.get("num_ooms", -1)

        if num_retries > 0:
            logger.warning(
                f"{num_retries} {device_type.upper()} memory allocation retries."
            )
        if num_ooms > 0:
            logger.warning(f"{num_ooms} {device_type.upper()} OOM errors thrown.")

        return DeviceMemStats(
            max_active_gib,
            max_active_pct,
            max_reserved_gib,
            max_reserved_pct,
            num_retries,
            num_ooms,
        )

    def reset_peak_stats(self):
        device_module.reset_peak_memory_stats()


def build_device_memory_monitor():
    device_memory_monitor = DeviceMemoryMonitor(device_type)
    logger.info(
        f"{device_type.upper()} capacity: {device_memory_monitor.device_name} "
        f"with {device_memory_monitor.device_capacity_gib:.2f}GiB memory"
    )
    return device_memory_monitor


class BaseLogger:
    def log(self, metrics: dict[str, Any], step: int) -> None:
        pass

    def close(self) -> None:
        pass


class WandBLogger(BaseLogger):
    def __init__(self, log_dir: str, job_config: JobConfig, tag: str | None = None):
        # Import wandb here to avoid startup import
        import wandb

        self.wandb = wandb
        self.tag = tag

        os.makedirs(log_dir, exist_ok=True)

        self.wandb.init(
            entity=os.getenv("WANDB_TEAM", None),
            project=os.getenv("WANDB_PROJECT", "torchfeather"),
            name=os.getenv("WANDB_RUN_NAME", None),
            dir=log_dir,
            config=job_config.to_dict(),
        )
        logger.info("WandB logging enabled")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        wandb_metrics = {
            (k if self.tag is None else f"{self.tag}/{k}"): v
            for k, v in metrics.items()
        }
        self.wandb.log(wandb_metrics, step=step)

    def close(self) -> None:
        if self.wandb.run is not None:
            self.wandb.finish()


def _get_metrics_rank(
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> int:
    # Early return for non-pipeline-parallel configurations
    if not parallel_dims.pp_enabled:
        return 0

    # V Block Schedules return loss on rank 0
    if job_config.parallelism.pipeline_parallel_schedule == "ZBVZeroBubble":
        return 0

    # Calculate first rank of the last pipeline stage
    world_size = parallel_dims.world_size
    pp_size = parallel_dims.pp
    return (world_size // pp_size) * (pp_size - 1)


def _build_metric_logger(
    job_config: JobConfig, parallel_dims: ParallelDims, tag: str | None = None
) -> BaseLogger:
    """
    Build an appropriate metric logger based on configuration.
    """
    metrics_config = job_config.metrics

    # Log initial config state
    logger.debug("Building logger")

    should_log = True
    if not metrics_config.save_for_all_ranks:
        metrics_rank = _get_metrics_rank(parallel_dims, job_config)
        should_log = torch.distributed.get_rank() == metrics_rank

    logger.debug(f"Logging decision: should_log={should_log}")

    if not should_log:
        logger.debug("Returning BaseLogger due to should_log=False")
        return BaseLogger()

    # Setup logging directory
    dump_dir = job_config.job.dump_folder
    base_log_dir = os.path.join(
        dump_dir, metrics_config.save_folder, datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M")
    )

    if metrics_config.save_for_all_ranks:
        base_log_dir = os.path.join(
            base_log_dir, f"rank_{torch.distributed.get_rank()}"
        )

    wandb_logger = WandBLogger(base_log_dir, job_config, tag)
    return wandb_logger


class MetricsProcessor:
    logger: BaseLogger
    parallel_dims: ParallelDims
    job_config: JobConfig
    device_memory_monitor: DeviceMemoryMonitor

    gpu_peak_flops: int
    ntokens_since_last_log: int
    data_loading_times: list[float]
    time_last_log: float

    num_flops_per_token: int
    optimizers: OptimizersContainer | None
    lr_schedulers: LRSchedulersContainer | None
    model_parts: list[torch.nn.Module] | None

    def __init__(
        self,
        job_config: JobConfig,
        parallel_dims: ParallelDims,
        tag: str | None = None,
    ):
        self.logger = _build_metric_logger(job_config, parallel_dims, tag)
        self.parallel_dims = parallel_dims
        self.job_config = job_config
        self.device_memory_monitor = build_device_memory_monitor()

        self.gpu_peak_flops = utils.get_peak_flops(
            self.device_memory_monitor.device_name
        )
        self.ntokens_since_last_log = 0
        self.data_loading_times = []
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()

        # These variables have to be set later as they depend on other components or model.
        self.num_flops_per_token = -1
        self.optimizers = None
        self.lr_schedulers = None
        self.model_parts = None

    def should_log(self, step: int) -> bool:
        return step == 1 or step % self.job_config.metrics.log_freq == 0

    def log(
        self,
        step: int,
        global_avg_loss: float,
        global_max_loss: float,
        grad_norm: float,
        extra_metrics: dict[str, Any] | None = None,
    ):
        assert self.num_flops_per_token > 0, "num_flops_per_token must be set"

        time_delta = time.perf_counter() - self.time_last_log

        # tokens per second per device, abbreviated as tps
        tps = self.ntokens_since_last_log / (
            time_delta * self.parallel_dims.non_data_parallel_size
        )
        # model FLOPS utilization
        # For its definition and calculation, please refer to the PaLM paper:
        # https://arxiv.org/abs/2204.02311
        mfu = 100 * self.num_flops_per_token * tps / self.gpu_peak_flops
        tflops = self.num_flops_per_token * tps / 1e12

        time_end_to_end = time_delta / self.job_config.metrics.log_freq
        time_data_loading = sum(self.data_loading_times) / len(self.data_loading_times)
        time_data_loading_pct = 100 * sum(self.data_loading_times) / time_delta

        device_mem_stats = self.device_memory_monitor.get_peak_stats()

        metrics = {
            "loss_metrics/global_avg_loss": global_avg_loss,
            "loss_metrics/global_max_loss": global_max_loss,
            "grad_norm": grad_norm,
            "throughput(tps)": tps,
            "tflops": tflops,
            "mfu(%)": mfu,
            "time_metrics/end_to_end(s)": time_end_to_end,
            "time_metrics/data_loading(s)": time_data_loading,
            "time_metrics/data_loading(%)": time_data_loading_pct,
            "memory/max_active(GiB)": device_mem_stats.max_active_gib,
            "memory/max_active(%)": device_mem_stats.max_active_pct,
            "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
            "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
            "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
            "memory/num_ooms": device_mem_stats.num_ooms,
        }

        if extra_metrics:
            metrics.update(extra_metrics)

        self.logger.log(metrics, step)

        logger.info(
            f"step: {step:2}  "
            f"loss: {global_avg_loss:7.4f}  "
            f"grad_norm: {grad_norm:7.4f}  "
            f"memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)  "
            f"tps: {round(tps):,}  "
            f"tflops: {tflops:,.2f}  "
            f"mfu: {mfu:.2f}%"
        )

        self.ntokens_since_last_log = 0
        self.data_loading_times.clear()
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()

    def close(self):
        self.logger.close()
