from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import (
    DTensor,
    Partial,
    Replicate,
    Shard,
    distribute_module,
    distribute_tensor,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    PrepareModuleInputOutput,
    RowwiseParallel,
    parallelize_module,
)

from torchfeather.distributed import NoParallel
from torchfeather.model.moe.utils import _permute, _unpermute


class TensorParallel(ParallelStyle):
    def _prepare_input_fn(self, _mod, inputs, device_mesh):
        routed_input, num_tokens_per_expert = inputs
        # This round-trip is a forward no-op. The expert computation uses local tensors,
        # so DTensor cannot infer that each TP rank produces only a partial input gradient.
        # Mark it Partial so backward reduces it to Replicate across the TP ranks.
        routed_input = DTensor.from_local(
            routed_input, device_mesh, (Replicate(),)
        ).to_local(grad_placements=(Partial(),))

        return routed_input, num_tokens_per_expert

    def _partition_fn(self, _name, module, device_mesh):
        # NOTE: they are DTensors but will be converted into local tensors when using GroupedExperts.forward

        # w1 shape = (experts, out_dim, in_dim)
        module.register_parameter(
            "w1", nn.Parameter(distribute_tensor(module.w1, device_mesh, [Shard(1)]))
        )  # Column-wise sharding

        # w2 shape = (experts, in_dim, out_dim)
        module.register_parameter(
            "w2",
            nn.Parameter(distribute_tensor(module.w2, device_mesh, [Shard(2)])),
        )  # Row-wise sharding

        # w3 shape = (experts, out_dim, in_dim)
        module.register_parameter(
            "w3",
            nn.Parameter(distribute_tensor(module.w3, device_mesh, [Shard(1)])),
        )  # Column-wise sharding

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        input_fn: Any = self._prepare_input_fn
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
            input_fn,
        )


class ExpertParallel(ParallelStyle):
    def __init__(self):
        super().__init__()
        self.input_splits = None
        self.output_splits = None
        self.input_shape = None
        self.permuted_indices = None

    def _token_dispatch(self, mod, inputs, device_mesh):
        routed_input, num_tokens_per_expert = inputs

        # Example: DP-shard = 4, TP = 2, EP = 4, ETP disabled, top_k = 4, 16 tokens per data parallel coordinate, and 12 experts.
        # world_size = DP-shard * TP = 4 * 2 = 8. The dense TP groups are [0,1], [2,3], [4,5], and [6,7].
        # EFSDP = DP-shard * TP / EP = 2, so the sparse EP groups are [0,1,2,3] and [4,5,6,7]. Each EP rank owns three experts.
        #
        # Consider EP group [0,1,2,3]. TP group [0,1] processes one 16-token input A, while TP group [2,3] processes another 16-token input B.
        # ReordererSequenceParallel gives rank 0 A0-A7, rank 1 A8-A15, rank 2 B0-B7, and rank 3 B8-B15.
        # Thus each routed_input is a plain local tensor with shape (16 // 2 * 4, dim) = (32, dim), not all routes from either full input.
        # Each token label below stands for its full hidden-state row of shape (dim,), not a token index.
        #
        # rank 0 routed_input, grouped by global expert:
        #     E0: A0,A5       E1: A0,A1,A6       E2: A1,A2,A7
        #     E3: A2,A3       E4: A0,A3,A4       E5: A1,A4,A5
        #     E6: A2,A5,A6    E7: A0,A3,A6,A7    E8: A1,A4,A7
        #     E9: A2,A5      E10: A3,A6          E11: A4,A7
        # rank 0 num_tokens_per_expert: [2, 3, 3, 2, 3, 3, 3, 4, 3, 2, 2, 2]
        #
        # rank 1 routed_input, grouped by global expert:
        #     E0: A11,A14     E1: A12,A15        E2: A8,A13
        #     E3: A8,A9,A14   E4: A9,A10,A15     E5: A10,A11
        #     E6: A8,A11,A12  E7: A9,A12,A13     E8: A10,A13,A14
        #     E9: A8,A11,A14,A15    E10: A9,A12,A15    E11: A10,A13
        # rank 1 num_tokens_per_expert: [2, 2, 2, 3, 3, 2, 3, 3, 3, 4, 3, 2]
        #
        # rank 2 routed_input, grouped by global expert:
        #     E0: B1,B4,B7    E1: B2,B5          E2: B3,B6
        #     E3: B4,B7       E4: B0,B5          E5: B0,B1,B6
        #     E6: B1,B2,B7    E7: B2,B3          E8: B0,B3,B4
        #     E9: B1,B4,B5   E10: B2,B5,B6       E11: B0,B3,B6,B7
        # rank 2 num_tokens_per_expert: [3, 2, 2, 2, 2, 3, 3, 2, 3, 3, 3, 4]
        #
        # rank 3 routed_input, grouped by global expert:
        #     E0: B10,B13,B14     E1: B8,B11,B14,B15    E2: B9,B12,B15
        #     E3: B10,B13         E4: B11,B14            E5: B12,B15
        #     E6: B8,B13          E7: B8,B9,B14          E8: B9,B10,B15
        #     E9: B10,B11        E10: B8,B11,B12        E11: B9,B12,B13
        # rank 3 num_tokens_per_expert: [3, 4, 3, 2, 2, 2, 2, 3, 3, 2, 3, 3]
        #
        # Every local histogram sums to 32. EP group [4,5,6,7] performs the same calculation independently for its two TP/data inputs.
        ep_degree = device_mesh.shape[0]
        # num_tokens_per_expert.shape = (num_experts,) = (12,)
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree

        # generate the input splits and output splits for all-to-all
        with torch.no_grad():
            # returns a AsyncCollectiveTensor that must be awaited

            # The equal-split all-to-all sends each source rank's three-count expert-owner chunks to the corresponding EP rank.
            # The result remains source-major and has shape (ep_degree * num_local_experts,) = (12,):
            #
            # rank 0 num_tokens_per_expert_group:
            #     [2,3,3 | 2,2,2 | 3,2,2 | 3,4,3]  # E0-E2 counts from sources 0, 1, 2, and 3
            # rank 1 num_tokens_per_expert_group:
            #     [2,3,3 | 3,3,2 | 2,2,3 | 2,2,2]  # E3-E5 counts from sources 0, 1, 2, and 3
            # rank 2 num_tokens_per_expert_group:
            #     [3,4,3 | 3,3,3 | 3,2,3 | 2,3,3]  # E6-E8 counts from sources 0, 1, 2, and 3
            # rank 3 num_tokens_per_expert_group:
            #     [2,2,2 | 4,3,2 | 3,3,4 | 2,3,3]  # E9-E11 counts from sources 0, 1, 2, and 3
            num_tokens_per_expert_group = all_to_all_single(
                num_tokens_per_expert,
                None,
                None,
                group=device_mesh.get_group(),
            )
            # Need to wait explicitly because it is used by a triton kernel later which doesn't realize that AsyncCollectiveTensor needs unwrapping
            num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
                num_tokens_per_expert_group  # ty:ignore[invalid-argument-type]
            )

            # input_splits has shape (ep_degree,). Entry d is the number of
            # local routed_input rows sent to destination EP rank d:
            # rank 0: [2, 3, 3, 2, 3, 3, 3, 4, 3, 2, 2, 2] => view[ep_degree,-1] => [2, 3, 3 | 2, 3, 3 | 3, 4, 3 | 2, 2, 2] => sum[dim=1] => [8, 8, 10, 6]
            # This means that rank 0 will send 8 tokens to the rank that possesses the 1st group of experts
            # This means that rank 0 will send 8 tokens to the rank that possesses the 2nd group of experts
            # This means that rank 0 will send 10 tokens to the rank that possesses the 3rd group of experts
            # This means that rank 0 will send 6 tokens to the rank that possesses the 4th group of experts
            # The sum 8 + 8 + 10 + 6 = (local_batch_size / tp_degree) * top_k = (16 / 2) * 4 = 8 * 4 = 32
            # rank 1: [6, 8, 9, 9] => sum = 32
            # rank 2: [7, 7, 8, 10] => sum = 32
            # rank 3: [10, 6, 8, 8] => sum = 32
            input_splits = (
                num_tokens_per_expert.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=True)
            )
            # NOTE: this would incur a device-to-host sync
            # output_splits also has shape (ep_degree,). Entry s is the number of routed rows received from source EP rank s:
            # rank 0: [2, 3, 3, 2, 2, 2, 3, 2, 2, 3, 4, 3] => view[ep_degree,-1] => [2,3,3 | 2,2,2 | 3,2,2 | 3,4,3] => sum[dim=1] => [8, 6, 7, 10]
            # This means rank 0 will receive 8 tokens from Rank 0, because it possesses the 1st group of experts
            # This means rank 0 will receive 6 tokens from Rank 1, because it possesses the 1st group of experts
            # This means rank 0 will receive 7 tokens from Rank 2, because it possesses the 1st group of experts
            # This means rank 0 will receive 10 tokens from Rank 3, because it possesses the 1st group of experts
            # The sum 8 + 6 + 7 + 10 = total number of tokens Rank 0 will receive across all EP ranks = 31 tokens
            # rank 1: [8, 8, 7, 6] => sum = 29
            # rank 2: [10, 9, 8, 8] => sum = 35
            # rank 3: [6, 9, 10, 8] => sum = 33
            # NOTE: as you can see, each rank receives a (potentially) different number of tokens, that's why load balancing is important
            output_splits = (
                num_tokens_per_expert_group.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=False)
            )
            self.input_splits = input_splits.tolist()
            self.output_splits = output_splits.tolist()

        # perform all-to-all
        routed_input = all_to_all_single_autograd(
            routed_input,  # (batch_size * seq_len // tp_degree * top_k, dim)
            self.output_splits,  # (ep_degree,) <-- what each rank needs to receive from other ranks
            self.input_splits,  # (ep_degree,) <-- what each rank needs to send to other ranks
            device_mesh.get_group(),
        )

        # In the example, ranks 0-3 receive 31, 29, 35, and 33 rows respectively, so their routed_input shapes are
        # (31, dim), (29, dim), (35, dim), and (33, dim). The rows remain source-major, matching num_tokens_per_expert_group.

        # NOTE: We are not using fixed-capacity padded all-to-all, that's why we need to do two all-to-all, the first one to figure out how many tokens each rank will receive, the second one to actually send/receive them.

        # NOTE: After this all-to-all, the routed input is put on proper EP rank.
        # However, the num_tokens_per_expert_group is not of the final target format
        # [#tokens for local expert 0, #tokens for local expert 1, ...]
        # Rather, it is of the format
        # [#tokens for every local expert from source EP rank 0, #tokens for every local expert from source EP rank 1, ...]
        # We need to perform another shuffle to get the correct layout, via the _permute function below, which also does padding to make sure the number of tokens each expert gets locally is a multiple of TOKEN_GROUP_ALIGN_SIZE_M.
        # Note that this will create side effects when wrapping the for-loop implementation of GroupedExperts, as it does not need padding.
        # Before padding, the example's local expert counts are
        # rank 0 num_tokens_per_expert_group = [2,3,3 | 2,2,2 | 3,2,2 | 3,4,3]
        # what we want is:  [ 2,  3,  3] +
        #                   [ 2,  2,  2] +
        #                   [ 3,  2,  2] +
        #                   [ 3,  4,  3] =
        #                   [10, 11, 10] <--- how many tokens to feed expert 0, expert 1 and expert 2 respectively

        # rank 0: [10, 11, 10], rank 1: [9, 10, 10], rank 2: [11, 12, 12], and rank 3: [11, 11, 11].

        (
            self.input_shape,
            routed_input,
            self.permuted_indices,
            num_tokens_per_expert_group,
        ) = _permute(
            routed_input,  # rank 0 shape: (31, dim) because it received 31 tokens in total
            num_tokens_per_expert_group,
            ep_degree,
            num_local_experts,
        )

        return routed_input, num_tokens_per_expert_group

    @staticmethod
    def _partition_fn(_name, mod, device_mesh):
        # shard on the expert dimension
        for name, param in mod.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            mod.register_parameter(name, dist_param)

    # performing all-to-all combine on the output
    def _token_combine(self, _mod, routed_output, device_mesh):
        routed_output = _unpermute(
            routed_output, self.input_shape, self.permuted_indices
        )

        routed_output = all_to_all_single_autograd(
            routed_output,
            self.input_splits,
            self.output_splits,
            device_mesh.get_group(),
        )
        return routed_output

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        input_fn: Any = self._token_dispatch
        output_fn: Any = self._token_combine
        return distribute_module(
            module,
            device_mesh,
            partition_fn=ExpertParallel._partition_fn,
            input_fn=input_fn,
            output_fn=output_fn,
        )


class ExpertTensorParallel(ExpertParallel):

    def __init__(self, ep_etp_mesh: DeviceMesh):
        super().__init__()
        assert ep_etp_mesh.mesh_dim_names == ("ep", "etp"), (
            ep_etp_mesh.mesh_dim_names,
        )
        self.ep_etp_mesh = ep_etp_mesh
        self.ep_mesh = ep_etp_mesh["ep"]
        self.etp_mesh = ep_etp_mesh["etp"]

    def _token_dispatch(self, mod, inputs, device_mesh):
        routed_input, num_tokens_per_expert = inputs
        # NOTE: This no-op marking of the tensor as Partial is due to the same reason as in `TensorParallel`.
        routed_input = DTensor.from_local(
            routed_input, self.etp_mesh, (Replicate(),)
        ).to_local(grad_placements=(Partial(),))
        return super()._token_dispatch(
            mod, (routed_input, num_tokens_per_expert), self.ep_mesh
        )

    @staticmethod
    def _partition_fn(_name, mod, device_mesh):
        for name, param in mod.named_parameters(recurse=False):
            # colwise TP: shard out_dim (dim 1) for w1/w3;
            # rowwise TP: shard out_dim (dim 2) for w2
            _tp_shard_dims = {"w1": 1, "w2": 2, "w3": 1}
            shard_dim = _tp_shard_dims.get(name)
            placements = (
                (Shard(0), Shard(shard_dim))
                if shard_dim is not None
                else (Shard(0), Replicate())
            )
            mod.register_parameter(
                name,
                nn.Parameter(
                    distribute_tensor(
                        param,
                        device_mesh,
                        placements,
                    )
                ),
            )

    def _token_combine(self, _mod, routed_output, device_mesh):
        return super()._token_combine(
            _mod,
            routed_output,
            self.ep_mesh,
        )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        assert device_mesh.mesh_dim_names == ("ep",), (device_mesh.mesh_dim_names,)
        input_fn: Any = self._token_dispatch
        output_fn: Any = self._token_combine
        return distribute_module(
            module,
            self.ep_etp_mesh,
            partition_fn=ExpertTensorParallel._partition_fn,
            input_fn=input_fn,
            output_fn=output_fn,
        )


class ReordererSequenceParallel(ParallelStyle):
    def __init__(self):
        super().__init__()

    def _prepare_input_fn(self, _mod, inputs, device_mesh):
        # shape (batch_size*seq_len, top_k)
        top_scores, selected_experts_indices = inputs
        num_tokens, _ = top_scores.shape

        def _split_along_first_dim(x: torch.Tensor) -> torch.Tensor:
            assert x.is_contiguous()
            if num_tokens % device_mesh.size() != 0:
                raise ValueError(
                    "Uneven split of tokens of is not supported yet. Requires TP degree dividing batch size * seq len."
                )
            local_num_tokens = num_tokens // device_mesh.size()
            local_rank = device_mesh.get_local_rank()
            offset = local_rank * local_num_tokens
            output = x[offset : offset + local_num_tokens]

            return output

        top_scores = _split_along_first_dim(top_scores)
        selected_experts_indices = _split_along_first_dim(selected_experts_indices)

        # shape (batch_size * seq_len // tp_degree, top_k)
        return top_scores, selected_experts_indices

    def _prepare_output_fn(self, mod, outputs, device_mesh):
        # shape (batch_size * seq_len * top_k // tp_degree)
        top_scores, token_indices_experts_sorted, num_tokens_per_expert = outputs

        assert hasattr(mod, "top_k")
        # This is `tokens_per_rank` due to the shape of `top_scores` (defined above)
        tokens_per_rank = top_scores.shape[0] // mod.top_k

        local_rank = device_mesh.get_local_rank()
        # Transform from local indices to global indices.
        # You can verify that each rank has local indices (in token_indices_experts_sorted) of the tokens assigned to it by checking the output of TokenReorderer when its input is split by the _prepare_input_fn function above.
        token_indices_experts_sorted += tokens_per_rank * local_rank

        return top_scores, token_indices_experts_sorted, num_tokens_per_expert

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        # NOTE: the device_mesh is the TP mesh
        input_fn: Any = self._prepare_input_fn
        output_fn: Any = self._prepare_output_fn
        return distribute_module(
            module,
            device_mesh,
            partition_fn=None,
            input_fn=input_fn,
            output_fn=output_fn,
        )


def apply_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
    etp_enabled: bool,
    ep_etp_mesh: DeviceMesh | None = None,
):
    assert ep_mesh is not None or tp_mesh is not None
    if ep_mesh is not None and etp_enabled:
        assert ep_etp_mesh is not None

    layers = cast(nn.ModuleDict, model.layers)

    for transformer_block in layers.values():
        if not transformer_block.moe_enabled:
            continue

        if tp_mesh is not None:
            moe = cast(Any, transformer_block).moe

            # When score_before_experts=False, the router gradient depends on the expert output, which is a TP-partial sum (different on each TP rank).
            # Without correction the gate weights silently diverge across TP ranks, degrading loss for every TP config.
            # Marking the local gate-output gradient as Partial lets DTensor reduce the per-rank contributions when reconciling them with replicated tensors:
            #   sum_k(grad * partial_k) = grad * full_output
            gate_grad_placements: tuple[Partial] | None = (
                (Partial(),) if not moe.score_before_experts else None
            )

            moe_layer_plan: dict[str, ParallelStyle] = {
                # input / output sharding on the seqlen dim, because we want to align it with expectations for Sequence Parallel
                # all-gather for input, reduce-scatter for output
                "moe": PrepareModuleInputOutput(
                    input_layouts=(Shard(1),),  # sharded because coming from SP region
                    desired_input_layouts=(
                        Replicate(),
                    ),  # we use full sequence in TP region
                    use_local_input=True,  # module receives a plain local tensor (.to_local()) after the all-gather, not a DTensor
                    output_layouts=(
                        Partial(),
                    ),  # partial because at the exit of a TP region you still need to do all-reduce
                    desired_output_layouts=(
                        Shard(1),
                    ),  # sharded because going to another SP region
                ),
                "moe.router.gate": NoParallel(
                    output_grad_placements=gate_grad_placements,
                ),
            }
            if ep_mesh is not None and not etp_enabled:
                # If TP is borrowed for EP, then split the tokens across TP ranks so that the reorderer, the all-to-all comms, and routed experts computation are effectively running Sequence Parallel (split along the folded bs*slen dim)
                moe_layer_plan.update({"moe.reorderer": ReordererSequenceParallel()})
            shared_experts = moe.shared_experts
            if shared_experts is not None:
                # input Replicate, output Partial
                moe_layer_plan.update(
                    {
                        "moe.shared_experts.w1": ColwiseParallel(),
                        "moe.shared_experts.w2": RowwiseParallel(
                            output_layouts=Partial()
                        ),
                        "moe.shared_experts.w3": ColwiseParallel(),
                    }
                )
            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=moe_layer_plan,
            )

        experts_mesh: DeviceMesh | None = None
        experts_plan: ParallelStyle | None = None
        if ep_mesh is None:
            # TP only. Every rank holds all experts, but each expert's weights are sliced (col/row-wise) across TP ranks.
            experts_mesh = tp_mesh
            # input Replicate, output Partial
            experts_plan = TensorParallel()
        elif tp_mesh is None or not etp_enabled:
            # EP only or EP+TP, but no ETP. Experts are split across EP ranks; each expert's weights stay whole (no col/row slicing).
            # `ReordererSequenceParallel` is applied in this case
            experts_mesh = ep_mesh
            # input / output sharding on the batch / tokens dim
            experts_plan = ExpertParallel()
        else:
            # ETP. Experts are split across EP ranks, and each expert's weights are further sliced (col/row-wise) across the TP ranks in the same EP group.
            experts_mesh = ep_mesh
            assert ep_etp_mesh is not None
            experts_plan = ExpertTensorParallel(ep_etp_mesh=ep_etp_mesh)

        moe_module = cast(Any, transformer_block).moe
        assert experts_plan is not None and experts_mesh is not None
        parallelize_module(moe_module.experts, experts_mesh, experts_plan)
