import functools
from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar, cast, overload

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

from torchfeather.config import Optimizer as OptimizerConfig
from torchfeather.distributed import ParallelDims

__all__ = [
    "OptimizersContainer",
    "build_optimizers",
    "build_optimizers_with_moe_load_balancing",
]


T = TypeVar("T", bound=Optimizer)


def _group_params_by_mesh(
    params: list[nn.Parameter],
) -> list[dict[str, list[nn.Parameter]]] | list[nn.Parameter]:
    """Group parameters by DTensor device mesh.

    Fused/foreach optimizer ops batch parameters together, but DTensor dispatch requires all operands to be on the same mesh. When EP is enabled, expert params live on a different mesh than regular FSDP params.
    Splitting into separate param groups ensures each group is processed independently.
    """
    mesh_groups: dict[tuple[str, ...] | None, list[nn.Parameter]] = {}
    for p in params:
        if isinstance(p.data, DTensor):
            key = p.data.device_mesh.mesh_dim_names
        else:
            key = None
        mesh_groups.setdefault(key, []).append(p)
    if len(mesh_groups) <= 1:
        return params
    return [{"params": group} for group in mesh_groups.values()]


class OptimizersContainer(Optimizer, Stateful, Generic[T]):
    optimizers: list[T]
    model_parts: list[nn.Module]

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
    ) -> None:
        all_params = []
        self.optimizers = []
        self.model_parts = model_parts
        for model in self.model_parts:
            params = [p for p in model.parameters() if p.requires_grad]
            # Group params by DTensor mesh into separate param groups so that fused/foreach ops don't batch params from different meshes (e.g. EP expert params vs regular FSDP params).
            param_groups = _group_params_by_mesh(params)
            self.optimizers.append(optimizer_cls(param_groups, **optimizer_kwargs))
            all_params.extend(params)
        self._validate_length(len(self.model_parts))
        self._post_init(all_params, optimizer_kwargs)

    def __iter__(self) -> Iterator[T]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    @overload
    def step(self, closure: None = ...) -> None: ...
    @overload
    def step(self, closure: Callable[[], float]) -> float: ...
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        ret: float | None = None
        for optimizer in self.optimizers:
            ret = optimizer.step(closure)
        return ret

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {
            k: v
            for sd in map(func, self.model_parts, self.optimizers)
            for k, v in sd.items()
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model_parts, self.optimizers))

    def _validate_length(self, expected_length: int) -> None:
        assert expected_length == len(self.optimizers), (
            "Must pass one optimizer per model part."
        )

    def _post_init(
        self, all_params: list[nn.Parameter], optimizer_kwargs: dict[str, Any]
    ) -> None:
        # We need to call Optimizer.__init__() to initialize some necessary optimizer functionality such as hooks.
        Optimizer.__init__(self, all_params, optimizer_kwargs)


def build_optimizers(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
) -> OptimizersContainer:
    name = optimizer_config.name
    lr = optimizer_config.lr
    beta1 = optimizer_config.beta1
    beta2 = optimizer_config.beta2
    eps = optimizer_config.eps
    weight_decay = optimizer_config.weight_decay

    optim_implementation = optimizer_config.implementation
    assert optim_implementation in ["fused", "foreach", "for-loop"]

    fused = optim_implementation == "fused"
    foreach = optim_implementation == "foreach"

    optimizer_kwargs = {
        "lr": lr,
        "betas": (beta1, beta2),
        "eps": eps,
        "weight_decay": weight_decay,
        "fused": fused,
        "foreach": foreach,
    }

    optimizer_classes = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
    }
    if name not in optimizer_classes:
        raise NotImplementedError(f"Optimizer {name} not added.")
    optimizer_cls = optimizer_classes[name]

    return OptimizersContainer(model_parts, optimizer_cls, optimizer_kwargs)


def build_optimizers_with_moe_load_balancing(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
) -> OptimizersContainer:
    optimizers = build_optimizers(
        model_parts=model_parts,
        optimizer_config=optimizer_config,
        parallel_dims=parallel_dims,
    )

    def _should_register_moe_balancing_hook(model_parts: list[nn.Module]) -> bool:
        for model_part in model_parts:
            layers = cast(nn.ModuleDict, model_part.layers)
            for transformer_block in layers.values():
                if transformer_block.moe_enabled:
                    # Assumption: load_balance_coeff is set universally on all moe blocks.
                    moe = cast(Any, transformer_block).moe
                    return bool(moe.load_balance_coeff)
        return False

    # for MoE auxiliary-loss-free load balancing
    def _is_recomputation_enabled(module):
        return getattr(module, "checkpoint_impl", None) is CheckpointImpl.NO_REENTRANT

    def _update_expert_bias(
        model_parts: list[nn.Module],
        parallel_dims: ParallelDims,
    ):
        dp_cp_mesh = parallel_dims.get_optional_mesh("loss")
        tokens_per_expert_list = []
        for model_part in model_parts:
            layers = cast(nn.ModuleDict, model_part.layers)
            for transformer_block in layers.values():
                if not transformer_block.moe_enabled:
                    continue
                moe = cast(Any, transformer_block).moe
                if moe.load_balance_coeff is None:
                    return
                tokens_per_expert = moe.tokens_per_expert
                if _is_recomputation_enabled(transformer_block):
                    # This does not affect to expert choice, but affects the experts usage metrics.
                    # We divide by 2 to correct for this double-counting due to recomputation
                    tokens_per_expert = tokens_per_expert // 2
                tokens_per_expert_list.append(tokens_per_expert)

        tokens_per_expert_by_layer = torch.vstack(tokens_per_expert_list)

        if dp_cp_mesh is not None:
            # Perform single all-reduce to get global statistics across all processes
            pg = dp_cp_mesh.get_group()
            torch.distributed.all_reduce(
                tokens_per_expert_by_layer, group=pg, op=torch.distributed.ReduceOp.SUM
            )

        moe_layer_idx = 0
        with torch.no_grad():
            for model_part in model_parts:
                layers = cast(nn.ModuleDict, model_part.layers)
                for transformer_block in layers.values():
                    if not transformer_block.moe_enabled:
                        continue
                    moe = cast(Any, transformer_block).moe

                    tokens_per_expert = tokens_per_expert_by_layer[
                        moe_layer_idx
                    ].float()
                    moe_layer_idx += 1

                    # update the expert bias
                    # this is not exactly the same as https://arxiv.org/pdf/2408.15664 proposed
                    expert_bias_delta = moe.load_balance_coeff * torch.sign(
                        tokens_per_expert.mean() - tokens_per_expert
                    )
                    expert_bias_delta = expert_bias_delta - expert_bias_delta.mean()
                    moe.expert_bias.add_(expert_bias_delta)
                    moe.tokens_per_expert.zero_()

    if _should_register_moe_balancing_hook(model_parts):
        optimizers.register_step_pre_hook(
            lambda *_args, **_kwargs: _update_expert_bias(
                model_parts, parallel_dims=parallel_dims
            )
        )

    return optimizers
