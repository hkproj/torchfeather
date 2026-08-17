from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from torchfeather.model.model_args import DeepSeekV3ModelArgs


@dataclass
class Job:
    dump_folder: str = "./torchfeather/outputs"


@dataclass
class Profiling:
    enable_profiling: bool = False
    save_traces_folder: str = "profile_traces"
    profile_freq: int = 10
    profiler_active: int = 1
    profiler_warmup: int = 3
    enable_memory_snapshot: bool = False
    save_memory_snapshot_folder: str = "memory_snapshot"


@dataclass
class Metrics:
    log_freq: int = 10
    save_folder: str = "metrics"
    save_for_all_ranks: bool = False


@dataclass
class Model:
    hf_assets_path: str = ""
    args: DeepSeekV3ModelArgs = field(default_factory=DeepSeekV3ModelArgs)


@dataclass
class Optimizer:
    name: str = "AdamW"
    lr: float = 8e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    implementation: Literal["for-loop", "foreach", "fused"] = "fused"


@dataclass
class LRScheduler:
    warmup_steps: int = 200

    decay_ratio: float | None = None
    """
    Controls the proportion of the training steps allocated to the learning rate decay phase.
    If `None`, the learning rate will begin decaying immediately after the warmup period.
    Otherwise, the learning rate will remain stable after the warmup period and
    only start decaying during the last `decay_ratio` portion of the total training steps.
    This is known as the Warmup-Stable-Decay (WSD) schedule, as described in https://arxiv.org/abs/2404.06395.
    """

    decay_type: Literal["linear", "sqrt", "cosine"] = "linear"
    """
    Learning rate decay type to use during training:
    - 'linear': linearly decays learning rate from initial to final value
    - 'sqrt': decays learning rate following a 1 minus square root curve
    - 'cosine': smoothly decays learning rate following a cosine curve
    """

    min_lr_factor: float = 0.0
    """
    Min lr ratio for lr scheduler.
    If provided, the range of decay factor is scaled from 1 to `min_lr_factor`
    to ensure the learning rate does not drop below `optimizer.lr * lr_scheduler.min_lr_factor`.
    """


@dataclass
class Training:
    dataset: str = "fineweb"

    dataset_path: str | None = None

    local_batch_size: int = 8

    global_batch_size: int = -1

    seq_len: int = 2048

    max_norm: float | int = 1.0

    steps: int = 10000

    dtype: Literal["bfloat16", "float32"] = "float32"
    """
    torch dtype for training. In contrast to mixed precision training, setting training_dtype=bfloat16 will
    put all parameters, gradients, and optimizer states in bfloat16, without an extra copy of fp32 weights.
    In the case of full bf16 training, RoPE calculations and logits will still be in fp32.
    """

    mixed_precision_param: Literal["bfloat16", "float32"] = "bfloat16"
    """
    torch dtype to use for parameters when applying mixed precision via fully_shard or torch.autocast.
    This feature takes effect via fully_shard when data_parallel_shard_degree > 1 or
    context_parallel_degree > 1; it takes effect via torch.autocast when data_replicate_degree >= 1
    and no other parallelism is enabled, i.e. under DDP or single-device training.
    """

    mixed_precision_reduce: Literal["float32"] = "float32"
    """
    torch dtype to use for reductions when applying mixed precision via FSDP.
    This feature only takes effect when data_parallel_shard_degree > 1
    """

    gc_freq: int = 50

    seed: int | None = None

    deterministic: bool = False


@dataclass
class Parallelism:
    data_parallel_replicate_degree: int = 1
    """
    The `data_parallel_replicate_degree` argument specifies the degree of
    data parallelism for weight replication. When this value is greater
    than 1, weights will be replicated across `data_parallel_replicate_degree`
    ranks. If `data_parallel_shard_degree` is also greater than 1, the parallelism
    method used is HSDP (Hybrid Sharded Data Parallelism). Otherwise, the
    parallelism method used is DDP (Distributed Data Parallelism).
    1 means disabled.
    """

    enable_compiled_autograd: bool = False
    """Enable CompiledAutograd to compile the backward."""

    data_parallel_shard_degree: int = -1
    """
    The `data_parallel_shard_degree` argument specifies the degree of data
    parallelism for weight sharding. When this value is greater than 1, weights
    will be sharded across `data_parallel_shard_degree` ranks. If
    `data_parallel_replicate_degree` is also greater than 1, the parallelism
    method used is HSDP (Hybrid Sharded Data Parallelism). Otherwise, the
    parallelism method used is FSDP (Fully Sharded Data Parallelism).
    -1 means leftover ranks will be used (After DP_REPLICATE/SP/PP). Note that
    only `data_parallel_shard_degree` can be negative. 1 means disabled.
    """

    fsdp_reshard_after_forward: Literal["default", "always", "never"] = "default"
    """
    `reshard_after_forward` specifies the policy for applying `reshard_after_forward`
    within an FSDP setup. `reshard_after_forward` controls parameter behavior after forward,
    trading off memory and communication. See torch's `fully_shard` API for more documentation
    on `reshard_after_forward`.

    The supported policies include "default", "always" and "never":

    - "default" applies default resharding behavior, implementing "smart defaults" for known optimal
      scenarios.
    - "always" will enable `reshard_after_forward` for all forward passes.
    - "never" will disable `reshard_after_forward` for all forward passes.
    """

    tensor_parallel_degree: int = 1
    """Tensor Parallelism degree. 1 means disabled."""

    disable_loss_parallel: bool = False
    """Whether to apply loss parallel when sequence parallel is enabled"""

    pipeline_parallel_degree: int = 1
    """
    Pipeline Parallelism degree, or number of ranks. 1 means disabled.
    If using looped schedules, this still specifies the number of physical ranks, not the number
    of stages. Stages per rank are inferred from split points degree, and schedule.
    """

    module_fqns_per_model_part: list[list[str]] | None = None
    """
    Specify a list of lists containing the FQNs (Fully Qualified Names) of modules for each model chunk.
    Each inner list represents one model chunk and contains the module names that belong to that chunk.
    e.g. [['tok_embeddings', 'layers.0'], ['layers.1', 'layers.2'], ['layers.3', 'layers.4']]
    will create 3 chunks: the first containing tok_embeddings and layers.0,
    the second containing layers.1 and layers.2, and the third containing layers.3 and layers.4.
    This provides more explicit control over which modules belong to each chunk compared to split points.
    """

    pipeline_parallel_first_stage_less_layers: int = 1
    """
    The number of layers to reduce in the first stage of pipeline parallelism. This is because
    the first stage has the extra overhead of the embedding layer, which is not present in the other stages.
    """

    pipeline_parallel_last_stage_less_layers: int = 1
    """
    The number of layers to reduce in the last stage of pipeline parallelism. This is because
    the last stage has the extra overhead of the output layer, which is not present in the other stages.
    """

    pipeline_parallel_layers_per_stage: int | None = None
    """
    The number of layers per (virtual) pipeline stage.
    """

    pipeline_parallel_schedule: str = "1F1B"
    """
    Specify the Pipeline Parallel schedule to use. The supported schedules are:
    https://github.com/pytorch/pytorch/blob/de4c2a3b4e89d96334dc678d1c3f2ae51a6630a0/torch/distributed/pipelining/schedules.py#L2161.
    The schedule must be compatible with the split points and stages_per_rank.
    Looped schedules (e.g. Interleaved1F1B) require specifying pipeline_parallel_degree = number of ranks,
    and split_points = number of stages - 1
    """

    pipeline_parallel_microbatch_size: int = 1
    """
    The size of each pipeline parallel microbatch (default 1).
    This value is used to compute the total number of microbatches by dividing local_batch_size with
    pipeline_parallel_microbatch_size.
    The global training batch size must be evenly divisible by pipeline_parallel_microbatch_size.
    """

    context_parallel_degree: int = 1
    """Context parallelism degree. 1 means disabled."""

    context_parallel_rotate_method: Literal["allgather", "alltoall"] = "allgather"
    """
    The collective to use in context parallel SDPA for kv shards exchange.
    - 'allgather' means to all-gather all kv shards on ranks after the first sub-SDPA computation,
    - 'alltoall' means to all-to-all shuffle the kv shards.
    The default value is 'allgather'.
    """

    context_parallel_load_balance: bool = True
    """
    Whether to enable round-robin load balancing for context parallel with causal attention.
    Disable this if you encounter illegal memory access errors in the ring attention backward.
    """

    expert_parallel_degree: int = 1
    """
    Expert parallelism degree. 1 means disabled. No effect for non-MoE models.

    Currently, it is supported with the following constraints:

    - when etp = tp:

      - cp <= ep <= dp_shard * cp
      - ep % cp == 0
      - dp_shard * cp % ep == 0

    - when etp = 1:

      - cp * tp <= ep <= dp_shard * cp * tp
      - ep % (cp * tp) == 0
      - dp_shard * cp * tp % ep == 0

    Note that this is still an experimental feature. Some constraints will be
    relaxed soon when we have more flexible DeviceMesh support.
    """

    expert_tensor_parallel_degree: int = 1
    """
    Expert tensor parallelism degree. 1 means disabled. No effect for non-MoE models, or when ep = 1.
    With this option, the tensor parallel degree on routed experts can be different from that on other params.
    Currently, we only support either
    - [partial dp -> ep] etp = tp
    - [partial dp + all tp -> ep] etp = 1
    Note that this is still an experimental feature.
    """


@dataclass
class Checkpoint:
    enable: bool = False

    folder: str = "checkpoint"

    interval: int = 500

    initial_load_path: str | None = None

    async_mode: Literal["disabled", "async"] = "disabled"

    keep_latest_k: int = 10

    load_step: int = -1

    enable_first_step_checkpoint: bool = False

    def __post_init__(self) -> None:
        if self.keep_latest_k == 1:
            raise ValueError(
                "The last checkpoint may still be in the process of being written, so keep_latest_k must be at least 2."
            )


@dataclass
class ActivationCheckpoint:
    mode: Literal["selective", "full", "none"] = "selective"

    selective_ac_option: str = "2"
    """
    Selective activation checkpointing options ['int', 'op'].
    'int' (e.g., 2) for every nth layer, or 'op' for op level ac.
    """

    per_op_sac_force_recompute_mm_shapes_by_fqns: list[str] = field(
        default_factory=lambda: ["moe.router.gate"]
    )
    """
    When per-op selective ac is used, this list of fully qualified names is used
    to determine which mm shapes to force recompute, rather than being considered
    by rest of the sac policy, e.g save every other mm. Only nn.Linear modules are
    supported today.

    Note: this config applies to mms not limited to those matching the specified
    fqns, e.g. if "moe.router.gate", corresponding to Linear(in, out), is specified,
    ANY mm with shape matching (*, in) x (in, out) will be force recomputed.
    """


@dataclass
class Compile:
    enable: bool = False

    components: list[Literal["model", "loss"]] = field(
        default_factory=lambda: ["model", "loss"]
    )
    backend: str = "inductor"


@dataclass
class Comm:
    init_timeout_seconds: int = 300
    """Timeout for communication operations, during initialization and first train step."""

    train_timeout_seconds: int = 100
    """
    Timeout for communication operations after the first train step --
    usually a tighter bound than during initialization.
    """


@dataclass
class JobConfig:
    """
    Default container for training configuration.
    """

    job: Job = field(default_factory=Job)
    profiling: Profiling = field(default_factory=Profiling)
    metrics: Metrics = field(default_factory=Metrics)
    model: Model = field(default_factory=Model)
    optimizer: Optimizer = field(default_factory=Optimizer)
    lr_scheduler: LRScheduler = field(default_factory=LRScheduler)
    training: Training = field(default_factory=Training)
    parallelism: Parallelism = field(default_factory=Parallelism)
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    activation_checkpoint: ActivationCheckpoint = field(
        default_factory=ActivationCheckpoint
    )
    compile: Compile = field(default_factory=Compile)
    comm: Comm = field(default_factory=Comm)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
