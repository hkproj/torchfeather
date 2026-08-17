from typing import Any, cast

import torch
from loguru import logger
from torch import nn
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed._composable.replicate import replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Shard


def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    enable_compile: bool,
    enable_compiled_autograd: bool,
):
    if enable_compile:
        if enable_compiled_autograd:
            torch._dynamo.config.optimize_ddp = (
                "python_reducer_without_compiled_forward"
            )
        else:
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)

    logger.info("Applied DDP to the model")


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    dp_replicate_efsdp_mesh: DeviceMesh | None = None,
):
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}

    match reshard_after_forward_policy:
        case "always":
            reshard_after_forward = True
        case "never":
            reshard_after_forward = False
        case "default":
            # For PP, by default do not reshard after forward to avoid per-microbatch all-gathers, which can be expensive and non-overlapped
            reshard_after_forward = not pp_enabled
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

    tok_embeddings: Any = model.tok_embeddings
    layers = cast(nn.ModuleDict, model.layers)
    norm: Any = model.norm
    output: Any = model.output

    if tok_embeddings is not None:
        fully_shard(
            tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    for transformer_block in layers.values():
        # NOTE: When EP is enabled, In an MoE layer, we use the following FSDP wrapping
        # - the router and the shared experts are sharded together with the TransformerBlock
        # - the routed experts are sharded with the dp_replicate_efsdp_mesh
        if transformer_block.moe_enabled and ep_degree > 1:
            fsdp_mod_ep_config = fsdp_config.copy()
            fsdp_mod_ep_config["mesh"] = dp_replicate_efsdp_mesh

            # NOTE: EP first shards the expert dimension, leaving each EP rank with num_experts / ep_degree local experts.
            # If the EFSDP shard degree exceeds that local expert count, FSDP's default Shard(0) leaves some EFSDP ranks with empty or very small shards.
            # Shard dim 1 instead so every EFSDP rank partitions the hidden dimension.
            # The dp_replicate axis does not count because it replicates parameters rather than sharding them.
            _experts_shard_placement_fn = None
            assert dp_replicate_efsdp_mesh is not None
            assert hasattr(transformer_block, "moe")
            moe = cast(Any, transformer_block).moe
            efsdp_degree = dp_replicate_efsdp_mesh["efsdp"].size()
            if efsdp_degree * ep_degree > moe.experts.num_experts:

                def _shard_on_dim1(_param: torch.Tensor) -> Shard:
                    return Shard(1)

                _experts_shard_placement_fn = _shard_on_dim1

            fully_shard(
                moe.experts,
                **fsdp_mod_ep_config,
                reshard_after_forward=reshard_after_forward,
                shard_placement_fn=_experts_shard_placement_fn,
            )

        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # As an optimization, do not reshard_after_forward the last layers by default since FSDP would prefetch them immediately after the forward pass
    if norm is not None and output is not None:
        fully_shard(
            [norm, output],
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )

    fully_shard(model, **fsdp_config)
    disable_fsdp_gradient_division(model)

    # NOTE: set up explicit prefetching when EP is enabled, as D2H syncs in EP could interfere with implicit prefetching in FSDP
    if ep_degree == 1:
        return

    # forward
    transformer_blocks: list[Any] = list(layers.values())
    next_transformer_blocks = transformer_blocks[1:] + [None]

    if tok_embeddings is not None and len(layers) > 0:
        tok_embeddings.set_modules_to_forward_prefetch([transformer_blocks[0]])  # ty:ignore[call-non-callable, unresolved-attribute]

    for transformer_block, next_transformer_block in zip(
        transformer_blocks, next_transformer_blocks
    ):
        if next_transformer_block is not None:
            if next_transformer_block.moe_enabled:
                transformer_block.set_modules_to_forward_prefetch(
                    [next_transformer_block, next_transformer_block.moe.experts]
                )
            else:
                transformer_block.set_modules_to_forward_prefetch(
                    [next_transformer_block]
                )
        elif norm is not None and output is not None:
            transformer_block.set_modules_to_forward_prefetch([norm, output])

    # backward
    reversed_transformer_blocks: list[Any] = list(reversed(list(layers.values())))
    prev_transformer_blocks = reversed_transformer_blocks[1:] + [None]

    if norm is not None and output is not None and len(layers) > 0:
        output.set_modules_to_backward_prefetch([reversed_transformer_blocks[0]])  # ty:ignore[call-non-callable, unresolved-attribute]

    for transformer_block, prev_transformer_block in zip(
        reversed_transformer_blocks, prev_transformer_blocks
    ):
        if prev_transformer_block is not None:
            if prev_transformer_block.moe_enabled:
                transformer_block.set_modules_to_backward_prefetch(
                    [prev_transformer_block, prev_transformer_block.moe.experts]
                )
            else:
                transformer_block.set_modules_to_backward_prefetch(
                    [prev_transformer_block]
                )
        elif tok_embeddings is not None:
            transformer_block.set_modules_to_backward_prefetch([tok_embeddings])


def disable_fsdp_gradient_division(model: nn.Module) -> None:
    """Disable FSDP's automatic gradient division for all FSDP modules.

    Sets gradient_divide_factor=1.0 so gradient scaling can be handled manually (e.g., by normalizing loss with global token count).
    """
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(1.0)
