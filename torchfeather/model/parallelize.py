import torch
from loguru import logger
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

from torchfeather.config import TORCH_DTYPE_MAP, JobConfig
from torchfeather.config.job_config import Compile as CompileConfig
from torchfeather.distributed import NoParallel, ParallelDims
from torchfeather.distributed.activation_checkpoint import apply_ac
from torchfeather.distributed.expert_parallel import apply_moe_ep_tp
from torchfeather.distributed.model_parallel import apply_ddp, apply_fsdp


# for selective op activation checkpointing
def _build_op_sac_save_list() -> set:
    """Build the set of ops whose outputs should be saved during selective AC."""
    try:
        from torch._functorch.partitioners import get_default_op_list

        save_ops = {op.default for op in get_default_op_list().compute_intensive_ops}  # ty:ignore[unresolved-attribute]
    except (ImportError, AttributeError):
        save_ops = {
            torch.ops.aten.mm.default,
            torch.ops.aten._scaled_dot_product_efficient_attention.default,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
        }

    extra_ops = [
        torch.ops.aten.linear.default,
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        torch.ops._c10d_functional.all_to_all_single.default,
        torch.ops.aten.max.default,
    ]
    for op in extra_ops:
        if op is not None:
            save_ops.add(op)

    # FlexAttention higher-order op (may not exist in all PyTorch versions)
    if hasattr(torch._higher_order_ops, "flex_attention"):
        save_ops.add(torch._higher_order_ops.flex_attention)  # ty:ignore[invalid-argument-type]

    return save_ops


_op_sac_save_list = _build_op_sac_save_list()


# Adapted from llama4/infra/parallelize.py
def parallelize_deepseekv3(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    # Currently we use local tensors along with DTensors for some computations. When you shard a DTensor, it can be ragged (e.g. you can have a size 10 dimension be split across 4 ranks, with the last rank having 1 element). But if we use local tensors, then we should keep track of this ourselves. Since we don't track any metadata for this, we just enforce it.
    assert job_config.training.seq_len % parallel_dims.seq_len_divisor == 0, f"""
        Sequence length {job_config.training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    if parallel_dims.tp_enabled:
        tp = parallel_dims.tp
        n_heads = model.model_args.n_heads  # ty:ignore[unresolved-attribute]
        if n_heads % tp != 0:  # ty:ignore[unsupported-operator]
            raise ValueError(
                f"tensor_parallel_degree ({tp}) must divide n_heads ({n_heads})."
            )

        apply_non_moe_tp(
            model,
            parallel_dims.get_mesh("tp"),
            loss_parallel=not job_config.parallelism.disable_loss_parallel,
        )

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        ep_etp_mesh = (
            parallel_dims.get_mesh(["ep", "etp"])
            if parallel_dims.ep_enabled and parallel_dims.etp_enabled
            else None
        )
        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            ep_etp_mesh=ep_etp_mesh,
            etp_enabled=parallel_dims.etp_enabled,
        )

    model_compile_enabled = (
        job_config.compile.enable and "model" in job_config.compile.components
    )

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(
            model,
            job_config.activation_checkpoint,
            model_compile_enabled=model_compile_enabled,
            op_sac_save_list=_op_sac_save_list,
            base_folder=job_config.job.dump_folder,
        )

    if model_compile_enabled:
        apply_compile(model, job_config.compile)

    dp_mesh: DeviceMesh | None = None
    if parallel_dims.fsdp_enabled or parallel_dims.ep_enabled:
        # apply FSDP or HSDP, potentially with Context Parallel
        if parallel_dims.dp_replicate_enabled:
            dp_mesh = parallel_dims.get_mesh(["dp_replicate", "fsdp"])
        else:
            dp_mesh = parallel_dims.get_mesh("fsdp")

        # the mesh for FSDP sharding of MoE params in the EP region
        if parallel_dims.ep_enabled:
            if parallel_dims.dp_replicate_enabled:
                dp_replicate_efsdp_mesh = parallel_dims.get_mesh(
                    ["dp_replicate", "efsdp"]
                )
            else:
                dp_replicate_efsdp_mesh = parallel_dims.get_mesh("efsdp")
        else:
            dp_replicate_efsdp_mesh = None

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
            ep_degree=parallel_dims.ep,
            dp_replicate_efsdp_mesh=dp_replicate_efsdp_mesh,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to the model")

    elif parallel_dims.dp_replicate_enabled:
        if parallel_dims.world_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        dp_mesh = parallel_dims.world_mesh
        apply_ddp(
            model,
            dp_mesh,
            enable_compile=model_compile_enabled,
            enable_compiled_autograd=job_config.parallelism.enable_compiled_autograd,
        )

    return model


def apply_compile(model: nn.Module, compile_config: CompileConfig):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    # NOTE: This flag is needed for torch.compile to avoid graph breaking on dynamic shapes in token-choice MoE but it is experimental.
    # torch._dynamo.config.capture_scalar_outputs = True
    for layer_id, transformer_block in model.layers.named_children():  # ty:ignore[unresolved-attribute]
        fullgraph = True
        if transformer_block.moe_enabled:
            fullgraph = False
        transformer_block = torch.compile(
            transformer_block,
            backend=compile_config.backend,
            fullgraph=fullgraph,
        )
        model.layers.register_module(layer_id, transformer_block)  # ty:ignore[unresolved-attribute, invalid-argument-type]

    logger.info("Compiling each TransformerBlock with torch.compile")


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
):
    # parallelize_module just calls the _apply method of the plan for each module that matches the pattern specified for the plan.

    # Each "ParallelStyle" just exposed a _apply method that:
    # - Shards the parameter of the module to which it is applied (if necessary)
    # - Registers a pre_forward_hook to make sure the input confirms to the expected input layout (the argument IS NOT the desired input layout, but the real layout)
    # - Registers a forward_hook (which is called after the fwd method) to make sure the output confirms to the expected output layout (the argument IS the desired output layout)

    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),  # the input tokens ID live on each rank.
                output_layouts=Shard(
                    1
                ),  # the output is sharded on the sequence dimension, which is what the norm expects
            ),
            "norm": SequenceParallel(),
            # loss_parallel requires the logits to be split on the vocabulary dimension
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    for transformer_block in model.layers.values():  # ty:ignore[call-non-callable]
        layer_plan: dict[str, ParallelStyle] = {
            "attention_norm": SequenceParallel(),
            "attention": PrepareModuleInput(
                input_layouts=(Shard(1), Replicate()),
                desired_input_layouts=(Replicate(), Replicate()),
            ),
            "attention.wkv_a": NoParallel(
                use_local_output=False
            ),  # No need to shard because it's not per-head
            "attention.wkv_b": ColwiseParallel(
                use_local_output=False
            ),  # this is per-head, so we can shard
            "attention.kv_norm": NoParallel(use_local_output=False),
            "attention.inner_attention": PrepareModuleInput(
                input_layouts=(Shard(1), Shard(1), Shard(1)),  # split Q,K,V per head
                desired_input_layouts=(Shard(1), Shard(1), Shard(1)),
                use_local_output=False,
            ),
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
        }

        if transformer_block.attention.q_lora_rank == 0:
            layer_plan.update(
                {
                    "attention.wq": ColwiseParallel(
                        use_local_output=False
                    ),  # This is only used when q_lora_rank==0
                }
            )
        else:
            layer_plan.update(
                {
                    "attention.wq_a": NoParallel(
                        use_local_output=False
                    ),  # Same reasoning as above, no need to shard since it's not per-head
                    "attention.wq_b": ColwiseParallel(
                        use_local_output=False
                    ),  # Can shard, because it's per head
                    "attention.q_norm": NoParallel(use_local_output=False),
                }
            )

        if not transformer_block.moe_enabled:
            layer_plan.update(
                {
                    # The input of this comes from `ffn_norm`, which is SP.
                    # We are entering a TP region after a SP region, so we must have the entire input as required by TP.
                    "feed_forward": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "feed_forward.w1": ColwiseParallel(),
                    "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
                    "feed_forward.w3": ColwiseParallel(),
                }
            )

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    logger.info("Applied Tensor Parallelism to the model")
