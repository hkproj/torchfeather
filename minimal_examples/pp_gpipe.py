from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.profiler import ProfilerActivity, profile, record_function

LOCAL_BATCH_SIZE = 8
MICRO_BATCH_SIZE = 2
HIDDEN_SIZE = 8
PP_SIZE = 2
DP_REPLICATE_SIZE = 4
EXPECTED_WORLD_SIZE = PP_SIZE * DP_REPLICATE_SIZE
MODEL_SEED = 0
DATA_SEED = 100
FORWARD_TAG_BASE = 100
BACKWARD_TAG_BASE = 200
TRACE_DIR = Path("traces")


class ActionKind(Enum):
    IRECV_FORWARD = auto()
    WAIT_RECV_FORWARD = auto()
    FORWARD = auto()
    ISEND_FORWARD = auto()
    WAIT_SEND_FORWARD = auto()
    IRECV_BACKWARD = auto()
    WAIT_RECV_BACKWARD = auto()
    BACKWARD = auto()
    ISEND_BACKWARD = auto()
    WAIT_SEND_BACKWARD = auto()
    ALL_REDUCE_GRADS = auto()


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    microbatch: int | None = None


class RepeatedDataLoader:
    def __init__(self, dp_coordinate: int) -> None:
        generator = torch.Generator(device="cpu").manual_seed(
            DATA_SEED + dp_coordinate
        )
        self.inputs = torch.randn(
            LOCAL_BATCH_SIZE,
            HIDDEN_SIZE,
            generator=generator,
        )
        self.targets = torch.randn(
            LOCAL_BATCH_SIZE,
            HIDDEN_SIZE,
            generator=generator,
        )

    def __iter__(self) -> RepeatedDataLoader:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs.clone(), self.targets.clone()


class MLPBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(x))


def make_block(pp_coordinate: int) -> MLPBlock:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(MODEL_SEED + pp_coordinate)
        return MLPBlock()


def build_stage_zero_schedule(num_microbatches: int) -> list[list[Action]]:
    schedule = [[] for _ in range(2 * num_microbatches + 3)]

    for microbatch in range(num_microbatches):
        actions = [Action(ActionKind.FORWARD, microbatch)]
        if microbatch > 0:
            actions.append(Action(ActionKind.WAIT_SEND_FORWARD, microbatch - 1))
        actions.append(Action(ActionKind.ISEND_FORWARD, microbatch))
        schedule[microbatch].extend(actions)
    schedule[num_microbatches].append(
        Action(ActionKind.WAIT_SEND_FORWARD, num_microbatches - 1)
    )

    first_backward = num_microbatches + 1
    schedule[first_backward].append(
        Action(ActionKind.IRECV_BACKWARD, num_microbatches - 1)
    )
    for offset, microbatch in enumerate(reversed(range(num_microbatches))):
        timestamp = first_backward + 1 + offset
        schedule[timestamp].append(
            Action(ActionKind.WAIT_RECV_BACKWARD, microbatch)
        )
        if microbatch > 0:
            schedule[timestamp].append(
                Action(ActionKind.IRECV_BACKWARD, microbatch - 1)
            )
        schedule[timestamp].append(Action(ActionKind.BACKWARD, microbatch))

    schedule[-1].append(Action(ActionKind.ALL_REDUCE_GRADS))
    return schedule


def build_stage_one_schedule(num_microbatches: int) -> list[list[Action]]:
    schedule = [[] for _ in range(2 * num_microbatches + 3)]
    schedule[0].append(Action(ActionKind.IRECV_FORWARD, 0))

    for microbatch in range(num_microbatches):
        timestamp = microbatch + 1
        schedule[timestamp].append(
            Action(ActionKind.WAIT_RECV_FORWARD, microbatch)
        )
        if microbatch + 1 < num_microbatches:
            schedule[timestamp].append(
                Action(ActionKind.IRECV_FORWARD, microbatch + 1)
            )
        schedule[timestamp].append(Action(ActionKind.FORWARD, microbatch))

    first_backward = num_microbatches + 1
    for offset, microbatch in enumerate(reversed(range(num_microbatches))):
        timestamp = first_backward + offset
        schedule[timestamp].append(Action(ActionKind.BACKWARD, microbatch))
        if offset > 0:
            schedule[timestamp].append(
                Action(ActionKind.WAIT_SEND_BACKWARD, microbatch + 1)
            )
        schedule[timestamp].append(Action(ActionKind.ISEND_BACKWARD, microbatch))
    schedule[-2].append(Action(ActionKind.WAIT_SEND_BACKWARD, 0))
    schedule[-1].append(Action(ActionKind.ALL_REDUCE_GRADS))
    return schedule


def build_schedule(
    pp_coordinate: int,
    num_microbatches: int,
) -> list[list[Action]]:
    if num_microbatches < 1:
        raise ValueError("num_microbatches must be positive")
    if pp_coordinate == 0:
        return build_stage_zero_schedule(num_microbatches)
    if pp_coordinate == 1:
        return build_stage_one_schedule(num_microbatches)
    raise ValueError(f"expected pipeline coordinate 0 or 1, got {pp_coordinate}")


def format_schedule(rank: int, schedule: list[list[Action]]) -> str:
    lines = [f"[rank {rank}] schedule:"]
    for timestamp, actions in enumerate(schedule):
        descriptions = [
            action.kind.name
            if action.microbatch is None
            else f"{action.kind.name}({action.microbatch})"
            for action in actions
        ]
        lines.append(f"  t={timestamp:02d}: {', '.join(descriptions)}")
    return "\n".join(lines)


def require_work(work: dist.Work | None, name: str) -> dist.Work:
    if work is None:
        raise RuntimeError(f"{name} did not return a work handle")
    return work


def require_gradient(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.grad is None:
        raise RuntimeError(f"missing gradient for {name}")
    return tensor.grad


def average_gradients(module: nn.Module, group: dist.ProcessGroup) -> None:
    for parameter in module.parameters():
        gradient = require_gradient(parameter, "parameter")
        dist.all_reduce(gradient, group=group)
        gradient.div_(DP_REPLICATE_SIZE)


def peer_rank(mesh: DeviceMesh, pp_coordinate: int, dp_coordinate: int) -> int:
    return int(mesh.mesh[1 - pp_coordinate, dp_coordinate].item())


def run_schedule(
    pp_coordinate: int,
    peer: int,
    dp_group: dist.ProcessGroup,
    block: MLPBlock,
    schedule: list[list[Action]],
    input_microbatches: tuple[torch.Tensor, ...],
    target_microbatches: tuple[torch.Tensor, ...],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    num_microbatches = len(input_microbatches)
    forward_outputs: dict[int, torch.Tensor] = {}
    received_activations: dict[int, torch.Tensor] = {}
    losses: dict[int, torch.Tensor] = {}

    forward_recv_buffers: dict[int, torch.Tensor] = {}
    forward_recv_requests: dict[int, dist.Work] = {}
    forward_send_buffers: dict[int, torch.Tensor] = {}
    forward_send_requests: dict[int, dist.Work] = {}
    backward_recv_buffers: dict[int, torch.Tensor] = {}
    backward_recv_requests: dict[int, dist.Work] = {}
    backward_send_buffers: dict[int, torch.Tensor] = {}
    backward_send_requests: dict[int, dist.Work] = {}

    for timestamp, actions in enumerate(schedule):
        for action in actions:
            microbatch = action.microbatch
            label = action.kind.name.lower()
            if microbatch is not None:
                label = f"{label}/microbatch_{microbatch}"
            with record_function(f"timestamp_{timestamp}/{label}"):
                if action.kind is ActionKind.IRECV_FORWARD:
                    assert microbatch is not None, action
                    buffer = torch.empty(MICRO_BATCH_SIZE, HIDDEN_SIZE)
                    forward_recv_buffers[microbatch] = buffer
                    forward_recv_requests[microbatch] = require_work(
                        dist.irecv(
                            buffer,
                            src=peer,
                            tag=FORWARD_TAG_BASE + microbatch,
                        ),
                        "forward irecv",
                    )
                elif action.kind is ActionKind.WAIT_RECV_FORWARD:
                    assert microbatch is not None, action
                    forward_recv_requests.pop(microbatch).wait()
                elif action.kind is ActionKind.FORWARD:
                    assert microbatch is not None, action
                    if pp_coordinate == 0:
                        forward_outputs[microbatch] = block(
                            input_microbatches[microbatch]
                        )
                    else:
                        activation = forward_recv_buffers[microbatch]
                        activation.requires_grad_()
                        received_activations[microbatch] = activation
                        output = block(activation)
                        forward_outputs[microbatch] = output
                        losses[microbatch] = F.mse_loss(
                            output,
                            target_microbatches[microbatch],
                        ) / num_microbatches
                elif action.kind is ActionKind.ISEND_FORWARD:
                    assert microbatch is not None, action
                    buffer = forward_outputs[microbatch].detach()
                    forward_send_buffers[microbatch] = buffer
                    forward_send_requests[microbatch] = require_work(
                        dist.isend(
                            buffer,
                            dst=peer,
                            tag=FORWARD_TAG_BASE + microbatch,
                        ),
                        "forward isend",
                    )
                elif action.kind is ActionKind.WAIT_SEND_FORWARD:
                    assert microbatch is not None, action
                    forward_send_requests.pop(microbatch).wait()
                    del forward_send_buffers[microbatch]
                elif action.kind is ActionKind.IRECV_BACKWARD:
                    assert microbatch is not None, action
                    buffer = torch.empty(MICRO_BATCH_SIZE, HIDDEN_SIZE)
                    backward_recv_buffers[microbatch] = buffer
                    backward_recv_requests[microbatch] = require_work(
                        dist.irecv(
                            buffer,
                            src=peer,
                            tag=BACKWARD_TAG_BASE + microbatch,
                        ),
                        "backward irecv",
                    )
                elif action.kind is ActionKind.WAIT_RECV_BACKWARD:
                    assert microbatch is not None, action
                    backward_recv_requests.pop(microbatch).wait()
                elif action.kind is ActionKind.BACKWARD:
                    assert microbatch is not None, action
                    if pp_coordinate == 0:
                        forward_outputs[microbatch].backward(
                            backward_recv_buffers[microbatch]
                        )
                    else:
                        losses[microbatch].backward()
                        backward_send_buffers[microbatch] = require_gradient(
                            received_activations[microbatch],
                            f"activation {microbatch}",
                        ).detach()
                elif action.kind is ActionKind.ISEND_BACKWARD:
                    assert microbatch is not None, action
                    backward_send_requests[microbatch] = require_work(
                        dist.isend(
                            backward_send_buffers[microbatch],
                            dst=peer,
                            tag=BACKWARD_TAG_BASE + microbatch,
                        ),
                        "backward isend",
                    )
                elif action.kind is ActionKind.WAIT_SEND_BACKWARD:
                    assert microbatch is not None, action
                    backward_send_requests.pop(microbatch).wait()
                    del backward_send_buffers[microbatch]
                elif action.kind is ActionKind.ALL_REDUCE_GRADS:
                    average_gradients(block, dp_group)
                else:
                    raise AssertionError(action)

    return forward_outputs, losses


def run_reference(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    pp_coordinate: int,
    dp_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, MLPBlock]:
    first_block = make_block(0)
    second_block = make_block(1)
    output = second_block(first_block(inputs))
    loss = F.mse_loss(output, targets)
    loss.backward()
    selected_block = first_block if pp_coordinate == 0 else second_block
    average_gradients(selected_block, dp_group)
    return output.detach(), loss.detach(), selected_block


def assert_parameter_gradients_match(
    block: MLPBlock,
    reference_block: MLPBlock,
) -> None:
    for (name, parameter), (reference_name, reference_parameter) in zip(
        block.named_parameters(),
        reference_block.named_parameters(),
        strict=True,
    ):
        assert name == reference_name, (name, reference_name)
        torch.testing.assert_close(
            require_gradient(parameter, name),
            require_gradient(reference_parameter, reference_name),
        )


def main() -> None:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "launch with: uv run torchrun --standalone "
            "--nproc-per-node=8 ./minimal_examples/pp_gpipe.py"
        )
    if LOCAL_BATCH_SIZE % MICRO_BATCH_SIZE != 0:
        raise RuntimeError(
            "LOCAL_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE"
        )

    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != EXPECTED_WORLD_SIZE:
            raise RuntimeError(
                f"this example expects {EXPECTED_WORLD_SIZE} ranks, "
                f"got {world_size}"
            )

        mesh = init_device_mesh(
            "cpu",
            (PP_SIZE, DP_REPLICATE_SIZE),
            mesh_dim_names=("pp", "dp_replicate"),
        )
        coordinate = mesh.get_coordinate()
        if coordinate is None:
            raise RuntimeError(f"rank {rank} is not part of the device mesh")
        pp_coordinate, dp_coordinate = coordinate
        pipeline_peer = peer_rank(mesh, pp_coordinate, dp_coordinate)
        dp_group = mesh.get_group("dp_replicate")

        inputs, targets = next(iter(RepeatedDataLoader(dp_coordinate)))
        input_microbatches = inputs.split(MICRO_BATCH_SIZE)
        target_microbatches = targets.split(MICRO_BATCH_SIZE)
        num_microbatches = LOCAL_BATCH_SIZE // MICRO_BATCH_SIZE
        assert len(input_microbatches) == num_microbatches, (
            len(input_microbatches),
            num_microbatches,
        )
        assert len(target_microbatches) == num_microbatches, (
            len(target_microbatches),
            num_microbatches,
        )

        reference_output, reference_loss, reference_block = run_reference(
            inputs,
            targets,
            pp_coordinate,
            dp_group,
        )
        block = make_block(pp_coordinate)
        schedule = build_schedule(pp_coordinate, num_microbatches)
        print(
            f"[rank {rank}] coordinate={(pp_coordinate, dp_coordinate)} "
            f"peer={pipeline_peer}\n{format_schedule(rank, schedule)}",
            flush=True,
        )

        if rank == 0:
            TRACE_DIR.mkdir(exist_ok=True)
        dist.barrier()
        trace_path = TRACE_DIR / f"pp-gpipe-rank-{rank}.json"

        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as profiler:
            forward_outputs, losses = run_schedule(
                pp_coordinate,
                pipeline_peer,
                dp_group,
                block,
                schedule,
                input_microbatches,
                target_microbatches,
            )

        profiler.export_chrome_trace(str(trace_path))
        assert_parameter_gradients_match(block, reference_block)

        if pp_coordinate == PP_SIZE - 1:
            pipeline_output = torch.cat(
                [forward_outputs[i].detach() for i in range(num_microbatches)]
            )
            pipeline_loss = torch.stack(
                [losses[i].detach() for i in range(num_microbatches)]
            ).sum()
            torch.testing.assert_close(pipeline_output, reference_output)
            torch.testing.assert_close(pipeline_loss, reference_loss)

        print(f"[rank {rank}] trace: {trace_path}", flush=True)
        dist.barrier()
        if rank == 0:
            print(
                "PASS: GPipe outputs and averaged gradients match the reference",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
