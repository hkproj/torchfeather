from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

BATCH_SIZE = 1
NUM_HEADS = 2
SEQUENCE_LENGTH = 8
HEAD_DIM = 4
EXPECTED_WORLD_SIZE = 4
SEED = 0
TRACE_DIR = Path("traces")


def validate_local_tensors(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    if not dist.is_initialized():
        raise RuntimeError("ring attention requires an initialized process group")
    if q.ndim != 4:
        raise RuntimeError(f"expected rank-4 Q, got shape {tuple(q.shape)}")
    if q.shape != k.shape or q.shape != v.shape:
        raise RuntimeError(
            f"Q, K, and V shapes must match, got "
            f"{tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}"
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise RuntimeError("Q, K, and V dtypes must match")


def rotate_tensors(
    tensors: tuple[torch.Tensor, ...],
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, ...]:
    next_rank = (rank + 1) % world_size
    previous_rank = (rank - 1) % world_size
    received = tuple(torch.empty_like(tensor) for tensor in tensors)
    operations: list[dist.P2POp] = []
    for tag, (send, recv) in enumerate(zip(tensors, received, strict=True)):
        operations.append(dist.P2POp(dist.isend, send, next_rank, tag=tag))
        operations.append(dist.P2POp(dist.irecv, recv, previous_rank, tag=tag))
    requests = dist.batch_isend_irecv(operations)
    for request in requests:
        request.wait()
    return received


def apply_local_causal_mask(scores: torch.Tensor) -> torch.Tensor:
    local_sequence_length = scores.shape[-1]
    mask = torch.ones(
        local_sequence_length,
        local_sequence_length,
        dtype=torch.bool,
        device=scores.device,
    ).tril()
    return scores.masked_fill(~mask, float("-inf"))


def ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    validate_local_tensors(q, k, v)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    scale = q.shape[-1] ** -0.5

    running_max = torch.full(
        q.shape[:-1],
        float("-inf"),
        dtype=q.dtype,
        device=q.device,
    )
    running_sum = torch.zeros_like(running_max)
    running_values = torch.zeros_like(q)
    current_k = k
    current_v = v

    for step in range(world_size):
        owner = (rank - step) % world_size
        if owner <= rank:
            with record_function(f"step_{step}/owner_{owner}/compute"):
                scores = (
                    torch.matmul(
                        q,
                        current_k.transpose(-2, -1),
                    )
                    * scale
                )
                if owner == rank:
                    scores = apply_local_causal_mask(scores)
                block_max = scores.amax(dim=-1)
                new_max = torch.maximum(running_max, block_max)
                previous_scale = torch.exp(running_max - new_max)
                probabilities = torch.exp(scores - new_max.unsqueeze(-1))
                running_sum = previous_scale * running_sum + probabilities.sum(dim=-1)
                running_values = previous_scale.unsqueeze(
                    -1
                ) * running_values + torch.matmul(probabilities, current_v)
                running_max = new_max
        else:
            with record_function(f"step_{step}/owner_{owner}/causal_skip"):
                pass

        with record_function(f"step_{step}/rotate_kv"):
            current_k, current_v = rotate_tensors(
                (current_k, current_v),
                rank,
                world_size,
            )

    return running_values / running_sum.unsqueeze(-1)


def make_full_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    shape = (BATCH_SIZE, NUM_HEADS, SEQUENCE_LENGTH, HEAD_DIM)
    q = torch.randn(shape, generator=generator)
    k = torch.randn(shape, generator=generator)
    v = torch.randn(shape, generator=generator)
    return q, k, v


def format_ring_schedule(rank: int, world_size: int) -> str:
    lines = [f"[rank {rank}] ring steps:"]
    for step in range(world_size):
        owner = (rank - step) % world_size
        action = "compute" if owner <= rank else "causal skip"
        lines.append(f"  step {step}: KV owner {owner}, {action}")
    return "\n".join(lines)


def main() -> None:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "launch with: uv run torchrun --standalone --nproc-per-node=4 "
            "./minimal_examples/ring_attention.py"
        )

    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != EXPECTED_WORLD_SIZE:
            raise RuntimeError(
                f"this example expects {EXPECTED_WORLD_SIZE} ranks, got {world_size}"
            )
        if SEQUENCE_LENGTH % world_size != 0:
            raise RuntimeError("SEQUENCE_LENGTH must be divisible by world size")

        local_sequence_length = SEQUENCE_LENGTH // world_size
        sequence_start = rank * local_sequence_length
        sequence_end = sequence_start + local_sequence_length

        full_q, full_k, full_v = make_full_tensors()
        reference_output = F.scaled_dot_product_attention(
            full_q,
            full_k,
            full_v,
            is_causal=True,
        )

        q = full_q[:, :, sequence_start:sequence_end].contiguous()
        k = full_k[:, :, sequence_start:sequence_end].contiguous()
        v = full_v[:, :, sequence_start:sequence_end].contiguous()

        if rank == 0:
            TRACE_DIR.mkdir(exist_ok=True)
        dist.barrier()
        trace_path = TRACE_DIR / f"ring-attention-rank-{rank}.json"

        print(format_ring_schedule(rank, world_size), flush=True)
        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            acc_events=True,
        ) as profiler:
            with record_function("ring_attention"):
                output = ring_attention(q, k, v)

        profiler.export_chrome_trace(str(trace_path))
        reference_slice = reference_output[:, :, sequence_start:sequence_end]
        torch.testing.assert_close(
            output,
            reference_slice,
            atol=1e-5,
            rtol=1e-4,
        )

        print(f"[rank {rank}] trace: {trace_path}", flush=True)
        dist.barrier()
        if rank == 0:
            print(
                "PASS: ring attention output matches full attention",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
