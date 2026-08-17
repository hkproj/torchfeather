import contextlib
import math
import os
from collections.abc import Callable, Iterable
from typing import cast

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from loguru import logger
from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.experimental._attention import (
    _cp_options,
    set_rotate_method,
)
from torch.distributed.tensor.parallel import loss_parallel

from torchfeather.config import TORCH_DTYPE_MAP
from torchfeather.distributed.parallel_dims import ParallelDims
from torchfeather.tools.device_utils import device_module


def _dist_reduce(
    x: torch.Tensor,
    reduceOp: str,
    mesh: DeviceMesh,
    extra_pg: dist.ProcessGroup | None,
) -> int | float:
    """Perform distributed reduction on a tensor.

    Args:
        x (torch.Tensor): Input tensor.
        reduceOp (str): Reduce operation to perform.
        mesh (DeviceMesh): Device mesh to use for reduction.
        extra_pg (dist.ProcessGroup, optional): Extra process group to use for reduction.
            Defaults to None. If provided, this all_reduce will be called for the extra process group, and then the result will be all_reduced for the mesh.
    """
    if isinstance(x, DTensor):
        # functional collectives do not support DTensor inputs
        x = x.full_tensor()

    if extra_pg is not None:
        x = funcol.all_reduce(x, reduceOp=reduceOp, group=extra_pg)

    assert x.numel() == 1  # required by `.item()`
    return funcol.all_reduce(x, reduceOp=reduceOp, group=mesh).item()


def dist_max(
    x: torch.Tensor,
    mesh: DeviceMesh,
    extra_pg: dist.ProcessGroup | None = None,
) -> int | float:
    return _dist_reduce(
        x, reduceOp=c10d.ReduceOp.MAX.name, mesh=mesh, extra_pg=extra_pg
    )


def dist_sum(
    x: torch.Tensor,
    mesh: DeviceMesh,
    extra_pg: dist.ProcessGroup | None = None,
) -> int | float:
    return _dist_reduce(
        x, reduceOp=c10d.ReduceOp.SUM.name, mesh=mesh, extra_pg=extra_pg
    )


def dist_mean(
    x: torch.Tensor,
    mesh: DeviceMesh,
    extra_pg: dist.ProcessGroup | None = None,
) -> float:
    return _dist_reduce(
        x, reduceOp=c10d.ReduceOp.AVG.name, mesh=mesh, extra_pg=extra_pg
    )


def set_determinism(
    parallel_dims: ParallelDims | None,
    device: torch.device,
    seed: int | None = None,
    deterministic: bool = False,
    distinct_seed_mesh_dims: list[str] | None = None,
) -> None:
    """
    Set the same DTensor manual seed for all dimensions in world mesh, but only different seeds across dimensions denoted by `distinct_seed_mesh_dims`.
    An example use case is pipeline parallelism, where we want to have the same seed across SPMD groups, but different seeds across PP groups.
    The reason is that in PP ranks you want - for example - the dropout masks to be different.
    This doesn't happen with the FSDP case because each FSDP rank has its own RNG state and it affects all layers inside of it (two layers in the same rank will naturally have different dropout masks).

    Multiple dims can be specified (e.g., ["pp", "ep"]) to ensure unique seeds across both PP and EP ranks. Uniqueness is guaranteed via multi-dimensional array indexing.

    Currently, does not set seeds for the CUDA RNG since TorchFeather always uses DTensor for SPMD parallelisms, and DTensor manages its own RNG tracker, but we could extend to support both if needed.
    """
    if distinct_seed_mesh_dims is None:
        distinct_seed_mesh_dims = ["pp"]

    if deterministic:
        logger.info("Deterministic algorithm enabled (expect perf degradation).")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # env var for deterministic CuBLAS
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # to ensure we can control which ranks have same or different seeds, all ranks agree on a starting seed.
    # If user provides one, we use this. Otherwise rank 0 rolls the dice and everyone else uses that.
    if seed is None:
        seed_tensor = torch.get_rng_state()[:8].to(device)
        torch.distributed.broadcast(seed_tensor, src=0)
        seed = cast(int, seed_tensor.to("cpu").view(torch.uint64).item())

    if parallel_dims is not None and c10d.get_world_size() > 1:
        distinct_meshes = [
            mesh
            for dim in distinct_seed_mesh_dims
            if (mesh := parallel_dims.get_optional_mesh(dim)) is not None
        ]

        if distinct_meshes:
            seed_offset = 0
            cumulative_size = 1
            for distinct_mesh in distinct_meshes:
                local_rank = distinct_mesh.get_local_rank()
                seed_offset += local_rank * cumulative_size
                cumulative_size *= distinct_mesh.size()

            seed += seed_offset
            seed %= 2**64

            logger.debug(
                f"Distinct dims {distinct_seed_mesh_dims}, Global rank {c10d.get_rank()} using seed: {seed}"
            )
    else:
        logger.debug(f"Global Rank {c10d.get_rank()} using seed: {seed}")

    # The native RNGs and python RNG may not be important, except for the 1-D PP case, but we seed them for consistency.
    torch.manual_seed(seed)
    # PYTHONHASHSEED can be a decimal number in the range [0, 2**32 - 1]
    os.environ["PYTHONHASHSEED"] = str(seed % 2**32)


def create_context_parallel_ctx(
    cp_mesh: DeviceMesh,
    cp_buffers: list[torch.Tensor],
    cp_seq_dims: list[int],
    cp_no_restore_buffers: set[torch.Tensor],
    cp_rotate_method: str,
    cp_load_balance: bool = True,
):
    set_rotate_method(cp_rotate_method)
    _cp_options.enable_load_balance = cp_load_balance
    return context_parallel(
        cp_mesh,
        buffers=cp_buffers,
        buffer_seq_dims=cp_seq_dims,
        no_restore_buffers=cp_no_restore_buffers,
    )


def get_train_context(
    enable_loss_parallel: bool, enable_compiled_autograd: bool
) -> Callable[..., contextlib.AbstractContextManager]:
    @contextlib.contextmanager
    def context(cp_context: contextlib.AbstractContextManager | None = None):
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(loss_parallel())

            if enable_compiled_autograd:
                stack.enter_context(
                    torch._dynamo.utils.maybe_enable_compiled_autograd(True)
                )

            if cp_context:
                stack.enter_context(cp_context)

            yield

    return context


def maybe_enable_amp(
    parallel_dims: ParallelDims, mixed_precision_param: str, device_type: str
) -> contextlib.AbstractContextManager:
    if parallel_dims.fsdp_enabled:
        # FSDP handles mixed precision internally
        logger.info("Mixed precision training is handled by fully_shard")
        return contextlib.nullcontext()
    else:
        if parallel_dims.tp_enabled or parallel_dims.pp_enabled:
            logger.warning(
                "Mixed precision training with TP or PP is only supported when FSDP/HSDP/CP is enabled."
            )
            logger.info("Mixed precision training is disabled")
            return contextlib.nullcontext()
        else:
            # the following code will only be executed for DDP or single-device training
            logger.info("Mixed precision training is handled by AMP")
            return torch.autocast(
                device_type,
                dtype=TORCH_DTYPE_MAP[mixed_precision_param],
            )


def set_pg_timeouts(timeout, parallel_dims: ParallelDims):
    """
    Sets the timeout for all PGs in the provided mesh, and the default (world) group.

    Note: synchronizes via a barrier, before changing the timeouts. This is important, because otherwise you may face a race where the slow rank has not reached the timeout reduction point yet due to slow operations permitted under the old timeout value, but other faster ranks may start issuing collectives under the new shorter timeout and then immediately timeout.
    """
    logger.info(
        f"Synchronizing and adjusting timeout for all ProcessGroups to {timeout}"
    )
    # Ensure that all the ranks have reached the point of setting the new timeout- otherwise, some ranks may issue collectives with the new/shorter timeout and those may time out, before other ranks have finished with initialization done under the old/slow timeout.
    torch.distributed.barrier(device_ids=[device_module.current_device()])
    device_module.synchronize()

    groups: list[dist.ProcessGroup | None] = [
        mesh.get_group()
        for mesh in parallel_dims.get_all_one_dimensional_meshes().values()
    ]

    # None represents the 'default' PG, not part of the mesh
    groups.append(None)
    for group in groups:
        torch.distributed.distributed_c10d._set_pg_timeout(timeout, group)


@torch.no_grad()
def clip_grad_norm_(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh: DeviceMesh | None = None,
    ep_enabled: bool = False,
) -> torch.Tensor:
    """
    Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/TorchFeather/issues/596 for details.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
        pp_mesh: Pipeline Parallel device mesh. If not None, will reduce gradient norm across PP stages.
        ep_dense_params_mesh_ndim: Mesh ndim of the dense params when EP is used. If EP is not used,
            set it to ``None``.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).

    """
    if ep_enabled:
        return _clip_grad_norm_with_ep(
            parameters,
            max_norm,
            norm_type,
            error_if_nonfinite,
            foreach,
            pp_mesh,
        )

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(
        grads, norm_type, error_if_nonfinite, foreach
    )

    # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
    # We can simply reduce the DTensor to get the total norm in this tensor's process group and then convert it to a local tensor.
    # NOTE: It has two purposes:
    #       1. to make sure the total norm is computed correctly when PP is used (see below)
    #       2. to return a reduced total_norm tensor whose .item() would return the correct value
    if isinstance(total_norm, DTensor):
        # Will reach here if any non-PP parallelism is used.
        # If only using PP, total_norm will be a local tensor.
        total_norm = total_norm.full_tensor()

    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type  # ty:ignore[unsupported-operator]
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type  # ty:ignore[unsupported-operator]

    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


@torch.no_grad()
def _clip_grad_norm_with_ep(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float,
    error_if_nonfinite: bool,
    foreach: bool | None,
    pp_mesh: DeviceMesh | None,
) -> torch.Tensor:
    ep_params = []
    non_ep_params = []
    ep_grads = []
    non_ep_grads = []

    for p in parameters:
        if p.grad is None:
            continue
        assert isinstance(p, DTensor) and isinstance(p.grad, DTensor)
        if (
            p.device_mesh.mesh_dim_names is not None
            and "ep" in p.device_mesh.mesh_dim_names
        ):
            ep_params.append(p)
            ep_grads.append(p.grad)
        else:
            non_ep_params.append(p)
            non_ep_grads.append(p.grad)
    ep_grads_total_norm = torch.nn.utils.get_total_norm(
        ep_grads, norm_type, error_if_nonfinite, foreach
    )
    # ep_grads may be an empty list, in which case get_total_norm returns tensor(0.), a non-DTensor
    # This can occur in PP + EP setups where certain PP ranks only own non-EP layers, for instance.
    if isinstance(ep_grads_total_norm, DTensor):
        ep_grads_total_norm = ep_grads_total_norm.full_tensor()

    non_ep_grads_total_norm = torch.nn.utils.get_total_norm(
        non_ep_grads, norm_type, error_if_nonfinite, foreach
    )
    # non_ep_grads may be an empty list, in which case get_total_norm returns tensor(0.), a non-DTensor
    # This can occur in PP + EP setups where certain PP ranks only own non-EP layers, for instance.
    if isinstance(non_ep_grads_total_norm, DTensor):
        non_ep_grads_total_norm = non_ep_grads_total_norm.full_tensor()

    if math.isinf(norm_type):
        total_norm = torch.maximum(ep_grads_total_norm, non_ep_grads_total_norm)
    else:
        total_norm = ep_grads_total_norm**norm_type + non_ep_grads_total_norm**norm_type
        total_norm **= 1.0 / norm_type  # ty:ignore[unsupported-operator]

    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type  # ty:ignore[unsupported-operator]
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type  # ty:ignore[unsupported-operator]

    torch.nn.utils.clip_grads_with_norm_(ep_params, max_norm, total_norm, foreach)
    torch.nn.utils.clip_grads_with_norm_(non_ep_params, max_norm, total_norm, foreach)

    return total_norm
