from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import (
    DTensor,
    Partial,
    Replicate,
    Shard,
    distribute_tensor,
)
from torch.profiler import ProfilerActivity, profile, record_function

BATCH_SIZE = 2
SEQUENCE_LENGTH = 4
HIDDEN_SIZE = 8
INTERMEDIATE_SIZE = 16
EXPECTED_WORLD_SIZE = 2
SEED = 0
EPSILON = 1e-6
TRACE_DIR = Path("traces")


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    variance = x.square().mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + epsilon) * weight


def swiglu(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    normalized = rms_norm(x, norm_weight, EPSILON)
    gate = F.linear(normalized, gate_weight)
    up = F.linear(normalized, up_weight)
    return x + F.linear(F.silu(gate) * up, down_weight)


def make_full_tensors() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    x = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        HIDDEN_SIZE,
        generator=generator,
    )
    norm_weight = torch.randn(HIDDEN_SIZE, generator=generator)
    gate_weight = torch.randn(
        INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
        generator=generator,
    )
    up_weight = torch.randn(
        INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
        generator=generator,
    )
    down_weight = torch.randn(
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        generator=generator,
    )
    return x, norm_weight, gate_weight, up_weight, down_weight


def clone_as_leaf(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().requires_grad_()


def require_gradient(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.grad is None:
        raise RuntimeError(f"missing gradient for {name}")
    return tensor.grad.detach()


def full_gradient(tensor: DTensor, name: str) -> torch.Tensor:
    if tensor.grad is None:
        raise RuntimeError(f"missing DTensor gradient for {name}")
    gradient = tensor.grad
    if not isinstance(gradient, DTensor):
        raise RuntimeError(f"expected a DTensor gradient for {name}")
    return gradient.full_tensor().detach()


def describe(rank: int, name: str, tensor: DTensor) -> None:
    print(
        f"[rank {rank}] {name:14} "
        f"global={tuple(tensor.shape)!s:12} "
        f"local={tuple(tensor.to_local().shape)!s:12} "
        f"placements={tensor.placements}",
        flush=True,
    )


def main() -> None:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "launch with: uv run torchrun --standalone "
            "--nproc-per-node=2 tp_swiglu.py"
        )

    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != EXPECTED_WORLD_SIZE:
            raise RuntimeError(
                f"this example expects {EXPECTED_WORLD_SIZE} ranks, got {world_size}"
            )
        if INTERMEDIATE_SIZE % world_size != 0:
            raise RuntimeError("INTERMEDIATE_SIZE must be divisible by world size")
        if SEQUENCE_LENGTH % world_size != 0:
            raise RuntimeError("SEQUENCE_LENGTH must be divisible by world size")

        mesh = init_device_mesh(
            "cpu",
            (world_size,),
            mesh_dim_names=("tp",),
        )

        full_x, full_norm, full_gate, full_up, full_down = make_full_tensors()

        ref_x = clone_as_leaf(full_x)
        ref_norm = clone_as_leaf(full_norm)
        ref_gate = clone_as_leaf(full_gate)
        ref_up = clone_as_leaf(full_up)
        ref_down = clone_as_leaf(full_down)
        ref_output = swiglu(ref_x, ref_norm, ref_gate, ref_up, ref_down)
        ref_loss = ref_output.square().mean()
        ref_loss.backward()

        x = distribute_tensor(clone_as_leaf(full_x), mesh, [Shard(1)])
        norm_weight = distribute_tensor(
            clone_as_leaf(full_norm), mesh, [Replicate()]
        )
        gate_weight = distribute_tensor(clone_as_leaf(full_gate), mesh, [Shard(0)])
        up_weight = distribute_tensor(clone_as_leaf(full_up), mesh, [Shard(0)])
        down_weight = distribute_tensor(clone_as_leaf(full_down), mesh, [Shard(1)])

        if rank == 0:
            TRACE_DIR.mkdir(exist_ok=True)
        dist.barrier()
        trace_path = TRACE_DIR / f"rank-{rank}.json"

        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as profiler:
            with record_function("forward/rms_norm"):
                normalized = rms_norm(x, norm_weight, EPSILON)

            with record_function("forward/sequence_to_replicate"):
                replicated = normalized.redistribute(mesh, [Replicate()])  # ty:ignore[unresolved-attribute]

            with record_function("forward/gate_up_projection"):
                gate = F.linear(replicated, gate_weight)
                up = F.linear(replicated, up_weight)

            with record_function("forward/swiglu_activation"):
                hidden = F.silu(gate) * up

            with record_function("forward/down_projection"):
                down_partial = F.linear(hidden, down_weight)

            if not any(
                isinstance(placement, Partial)
                for placement in down_partial.placements  # ty:ignore[unresolved-attribute]
            ):
                raise RuntimeError(
                    f"expected a Partial output, got {down_partial.placements}"  # ty:ignore[unresolved-attribute]
                )

            with record_function("forward/partial_to_sequence"):
                down_sequence = down_partial.redistribute(mesh, [Shard(1)])  # ty:ignore[unresolved-attribute]

            with record_function("forward/residual"):
                output = x + down_sequence

            with record_function("forward/global_loss"):
                output_replicated = output.redistribute(mesh, [Replicate()])
                loss = output_replicated.square().mean()

            with record_function("backward"):
                loss.backward()

        profiler.export_chrome_trace(str(trace_path))

        for name, tensor in (
            ("x", x),
            ("norm_weight", norm_weight),
            ("normalized", normalized),
            ("replicated", replicated),
            ("gate_weight", gate_weight),
            ("up_weight", up_weight),
            ("gate", gate),
            ("up", up),
            ("hidden", hidden),
            ("down_weight", down_weight),
            ("down_partial", down_partial),
            ("down_sequence", down_sequence),
            ("output", output),
            ("output_full", output_replicated),
        ):
            describe(rank, name, tensor)  # ty:ignore[invalid-argument-type]
        print(f"[rank {rank}] trace: {trace_path}", flush=True)

        torch.testing.assert_close(
            output_replicated.full_tensor(),
            ref_output.detach(),
        )
        torch.testing.assert_close(loss.full_tensor(), ref_loss.detach())
        torch.testing.assert_close(
            full_gradient(x, "x"),
            require_gradient(ref_x, "reference x"),
            atol=1e-4,
            rtol=5e-4,
        )
        torch.testing.assert_close(
            full_gradient(norm_weight, "norm_weight"),
            require_gradient(ref_norm, "reference norm_weight"),
        )
        torch.testing.assert_close(
            full_gradient(gate_weight, "gate_weight"),
            require_gradient(ref_gate, "reference gate_weight"),
        )
        torch.testing.assert_close(
            full_gradient(up_weight, "up_weight"),
            require_gradient(ref_up, "reference up_weight"),
        )
        torch.testing.assert_close(
            full_gradient(down_weight, "down_weight"),
            require_gradient(ref_down, "reference down_weight"),
        )

        dist.barrier()
        if rank == 0:
            print(
                "PASS: DTensor output and gradients match the ordinary tensors",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
