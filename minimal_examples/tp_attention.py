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
NUM_HEADS = 4
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
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


def split_heads(x: torch.Tensor) -> torch.Tensor:
    return x.view(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        NUM_HEADS,
        HEAD_DIM,
    ).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).reshape(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        HIDDEN_SIZE,
    )


def attention(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    output_weight: torch.Tensor,
) -> torch.Tensor:
    normalized = rms_norm(x, norm_weight, EPSILON)
    q = split_heads(F.linear(normalized, q_weight))
    k = split_heads(F.linear(normalized, k_weight))
    v = split_heads(F.linear(normalized, v_weight))
    context = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return x + F.linear(merge_heads(context), output_weight)


def local_attention(q: DTensor, k: DTensor, v: DTensor) -> DTensor:
    local_context = F.scaled_dot_product_attention(
        q.to_local(),
        k.to_local(),
        v.to_local(),
        is_causal=True,
    )
    return DTensor.from_local(
        local_context,
        q.device_mesh,
        [Shard(1)],
        run_check=False,
    )


def make_full_tensors() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    x = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        HIDDEN_SIZE,
        generator=generator,
    )
    norm_weight = torch.randn(HIDDEN_SIZE, generator=generator)
    q_weight = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, generator=generator)
    k_weight = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, generator=generator)
    v_weight = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, generator=generator)
    output_weight = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, generator=generator)
    return x, norm_weight, q_weight, k_weight, v_weight, output_weight


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
        f"[rank {rank}] {name:20} "
        f"global={tuple(tensor.shape)!s:14} "
        f"local={tuple(tensor.to_local().shape)!s:14} "
        f"placements={tensor.placements}",
        flush=True,
    )


def main() -> None:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "launch with: uv run torchrun --standalone "
            "--nproc-per-node=2 tp_attention.py"
        )

    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != EXPECTED_WORLD_SIZE:
            raise RuntimeError(
                f"this example expects {EXPECTED_WORLD_SIZE} ranks, got {world_size}"
            )
        if HIDDEN_SIZE % NUM_HEADS != 0:
            raise RuntimeError("HIDDEN_SIZE must be divisible by NUM_HEADS")
        if NUM_HEADS % world_size != 0:
            raise RuntimeError("NUM_HEADS must be divisible by world size")
        if SEQUENCE_LENGTH % world_size != 0:
            raise RuntimeError("SEQUENCE_LENGTH must be divisible by world size")

        mesh = init_device_mesh(
            "cpu",
            (world_size,),
            mesh_dim_names=("tp",),
        )

        (
            full_x,
            full_norm,
            full_q,
            full_k,
            full_v,
            full_output,
        ) = make_full_tensors()

        ref_x = clone_as_leaf(full_x)
        ref_norm = clone_as_leaf(full_norm)
        ref_q = clone_as_leaf(full_q)
        ref_k = clone_as_leaf(full_k)
        ref_v = clone_as_leaf(full_v)
        ref_output_weight = clone_as_leaf(full_output)
        ref_result = attention(
            ref_x,
            ref_norm,
            ref_q,
            ref_k,
            ref_v,
            ref_output_weight,
        )
        ref_loss = ref_result.square().mean()
        ref_loss.backward()

        x = distribute_tensor(clone_as_leaf(full_x), mesh, [Shard(1)])
        norm_weight = distribute_tensor(
            clone_as_leaf(full_norm), mesh, [Replicate()]
        )
        q_weight = distribute_tensor(clone_as_leaf(full_q), mesh, [Shard(0)])
        k_weight = distribute_tensor(clone_as_leaf(full_k), mesh, [Shard(0)])
        v_weight = distribute_tensor(clone_as_leaf(full_v), mesh, [Shard(0)])
        output_weight = distribute_tensor(
            clone_as_leaf(full_output), mesh, [Shard(1)]
        )

        if rank == 0:
            TRACE_DIR.mkdir(exist_ok=True)
        dist.barrier()
        trace_path = TRACE_DIR / f"tp-attention-rank-{rank}.json"

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

            with record_function("forward/qkv_projections"):
                q = F.linear(replicated, q_weight)
                k = F.linear(replicated, k_weight)
                v = F.linear(replicated, v_weight)

            with record_function("forward/split_heads"):
                q_heads = split_heads(q)
                k_heads = split_heads(k)
                v_heads = split_heads(v)

            if q_heads.placements != (Shard(1),):  # ty:ignore[unresolved-attribute]
                raise RuntimeError(
                    f"expected Q heads sharded on heads, got {q_heads.placements}"  # ty:ignore[unresolved-attribute]
                )

            with record_function("forward/local_attention"):
                context_heads = local_attention(q_heads, k_heads, v_heads)  # ty:ignore[invalid-argument-type]

            with record_function("forward/merge_heads"):
                context = merge_heads(context_heads)

            with record_function("forward/output_projection"):
                projected_partial = F.linear(context, output_weight)

            if not any(
                isinstance(placement, Partial)
                for placement in projected_partial.placements  # ty:ignore[unresolved-attribute]
            ):
                raise RuntimeError(
                    f"expected a Partial output, got {projected_partial.placements}"  # ty:ignore[unresolved-attribute]
                )

            with record_function("forward/partial_to_sequence"):
                projected_sequence = projected_partial.redistribute(  # ty:ignore[unresolved-attribute]
                    mesh, [Shard(1)]
                )

            with record_function("forward/residual"):
                result = x + projected_sequence

            with record_function("forward/global_loss"):
                result_replicated = result.redistribute(mesh, [Replicate()])
                loss = result_replicated.square().mean()

            with record_function("backward"):
                loss.backward()

        profiler.export_chrome_trace(str(trace_path))

        for name, tensor in (
            ("x", x),
            ("norm_weight", norm_weight),
            ("normalized", normalized),
            ("replicated", replicated),
            ("q_weight", q_weight),
            ("k_weight", k_weight),
            ("v_weight", v_weight),
            ("q", q),
            ("k", k),
            ("v", v),
            ("q_heads", q_heads),
            ("k_heads", k_heads),
            ("v_heads", v_heads),
            ("context_heads", context_heads),
            ("context", context),
            ("output_weight", output_weight),
            ("projected_partial", projected_partial),
            ("projected_sequence", projected_sequence),
            ("result", result),
            ("result_full", result_replicated),
        ):
            describe(rank, name, tensor)  # ty:ignore[invalid-argument-type]
        print(f"[rank {rank}] trace: {trace_path}", flush=True)

        assert_close = {
            "atol": 1e-5,
            "rtol": 1e-4,
        }
        torch.testing.assert_close(
            result_replicated.full_tensor(),
            ref_result.detach(),
            **assert_close,  # ty:ignore[invalid-argument-type]
        )
        torch.testing.assert_close(
            loss.full_tensor(),
            ref_loss.detach(),
            **assert_close,  # ty:ignore[invalid-argument-type]
        )
        for tensor, reference, name in (
            (x, ref_x, "x"),
            (norm_weight, ref_norm, "norm_weight"),
            (q_weight, ref_q, "q_weight"),
            (k_weight, ref_k, "k_weight"),
            (v_weight, ref_v, "v_weight"),
            (output_weight, ref_output_weight, "output_weight"),
        ):
            torch.testing.assert_close(
                full_gradient(tensor, name),
                require_gradient(reference, f"reference {name}"),
                **assert_close,  # ty:ignore[invalid-argument-type]
            )

        dist.barrier()
        if rank == 0:
            print(
                "PASS: tensor-parallel attention output and gradients "
                "match the ordinary tensors",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
