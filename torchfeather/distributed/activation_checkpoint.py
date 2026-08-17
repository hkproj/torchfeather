from collections import defaultdict
from typing import cast

import torch
from loguru import logger
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

from torchfeather.config.job_config import ActivationCheckpoint as ACConfig

_layer_sac_count = 0


def _apply_layer_sac(module: nn.Module, ac_config: ACConfig) -> nn.Module:
    global _layer_sac_count
    _layer_sac_count += 1
    ac_freq = int(ac_config.selective_ac_option)
    if not ac_freq or _layer_sac_count % ac_freq == 0:
        return ptd_checkpoint_wrapper(module, preserve_rng_state=False)
    else:
        return module


def _apply_op_sac(
    module: nn.Module,
    ac_config: ACConfig,
    *,
    base_fqn: str | None = None,
    op_sac_save_list: set[torch._ops.OpOverload],
) -> nn.Module:
    from torch.utils.checkpoint import (
        CheckpointPolicy,
        create_selective_checkpoint_contexts,
    )

    mm_recompute_shapes = set()
    if len(ac_config.per_op_sac_force_recompute_mm_shapes_by_fqns) > 0:
        for module_fqn, submod in module.named_modules():
            fqn = module_fqn
            if base_fqn is not None:
                fqn = f"{base_fqn}.{module_fqn}"
            if not any(
                filter_fqn in fqn
                for filter_fqn in ac_config.per_op_sac_force_recompute_mm_shapes_by_fqns
            ):
                continue
            if not isinstance(submod, nn.Linear):
                raise TypeError(
                    "per_op_sac_force_recompute_mm_shapes_by_fqns expected to match "
                    f"a nn.Linear, but got: {submod}"
                )
            out_f, in_f = submod.weight.shape
            mm_recompute_shapes.add((in_f, out_f))
        logger.debug(
            f"Selective op AC force recomputing mms with rhs shapes {mm_recompute_shapes}"
        )

    mm_ops = (torch.ops.aten.mm.default, torch.ops.aten.linear.default)

    def _get_custom_policy(meta):
        def _custom_policy(ctx, func, *args, **kwargs):
            if (
                func == torch.ops.aten._to_copy.default
                and "cuda" in str(args[0].device)
                and "device" in kwargs
                and str(kwargs["device"]) == "cpu"
            ):
                return CheckpointPolicy.MUST_SAVE

            mode = "recompute" if ctx.is_recompute else "forward"
            mm_count_key = f"{mode}_mm_count"
            if func in mm_ops:
                weight_shape = args[1].shape
                if func == torch.ops.aten.linear.default:
                    weight_shape = torch.Size((weight_shape[1], weight_shape[0]))
                if weight_shape in mm_recompute_shapes:
                    return CheckpointPolicy.PREFER_RECOMPUTE
                meta[mm_count_key] += 1
            # Saves output of all compute ops, except every second mm
            to_save = func in op_sac_save_list and not (
                func in mm_ops and meta[mm_count_key] % 2 == 0
            )
            return (
                CheckpointPolicy.MUST_SAVE
                if to_save
                else CheckpointPolicy.PREFER_RECOMPUTE
            )

        return _custom_policy

    def selective_checkpointing_context_fn():
        meta = defaultdict(int)
        return create_selective_checkpoint_contexts(_get_custom_policy(meta))

    return ptd_checkpoint_wrapper(
        module,
        context_fn=selective_checkpointing_context_fn,
        preserve_rng_state=False,
    )


def _apply_full_ac(module: nn.Module, ac_config: ACConfig) -> nn.Module:
    return ptd_checkpoint_wrapper(module, preserve_rng_state=False)


def _apply_ac_to_transformer_block(
    module: nn.Module,
    ac_config: ACConfig,
    *,
    base_fqn: str | None = None,
    model_compile_enabled: bool = False,
    op_sac_save_list: set[torch._ops.OpOverload] | None = None,
) -> nn.Module:
    valid_ac_modes = ("full", "selective")
    if ac_config.mode not in valid_ac_modes:
        raise ValueError(
            f"Invalid AC mode: {ac_config.mode}. Valid modes: {valid_ac_modes}"
        )

    if ac_config.mode == "full":
        return _apply_full_ac(module, ac_config)

    assert ac_config.mode == "selective", f"{ac_config.mode}"
    use_op_sac = ac_config.selective_ac_option == "op"
    use_layer_sac = ac_config.selective_ac_option.isdigit()
    if not use_op_sac and not use_layer_sac:
        raise ValueError(
            f"Invalid selective AC option: {ac_config.selective_ac_option}. "
            f"Valid options: 'op' or a positive int representing layer frequency"
        )

    if use_op_sac:
        op_sac_save_list = op_sac_save_list or set()
        return _apply_op_sac(
            module, ac_config, base_fqn=base_fqn, op_sac_save_list=op_sac_save_list
        )

    return _apply_layer_sac(module, ac_config)


def apply_ac(
    model: nn.Module,
    ac_config: ACConfig,
    *,
    model_compile_enabled: bool = False,
    op_sac_save_list: set[torch._ops.OpOverload] | None = None,
    base_folder: str = "",
) -> None:
    # Disable dynamo LRU cache to avoid assertion failures when PP sends multiple microbatches through the same SAC-wrapped layer.
    # See https://github.com/pytorch/pytorch/issues/166926
    torch._C._dynamo.eval_frame._set_lru_cache(False)  # ty:ignore[unresolved-attribute]

    layers = cast(nn.ModuleDict, model.layers)
    for layer_id, transformer_block in layers.named_children():
        transformer_block = _apply_ac_to_transformer_block(
            transformer_block,
            ac_config,
            base_fqn=f"layers.{layer_id}",
            model_compile_enabled=model_compile_enabled,
            op_sac_save_list=op_sac_save_list,
        )
        layers.register_module(layer_id, transformer_block)

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")
