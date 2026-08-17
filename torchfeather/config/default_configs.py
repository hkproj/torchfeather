from torchfeather.config.job_config import JobConfig
from torchfeather.model.model_args import DeepSeekV3ModelArgs
from torchfeather.model.moe.moe import MoEArgs


def get_deepseek_v3_model_args() -> DeepSeekV3ModelArgs:
    return DeepSeekV3ModelArgs(
        vocab_size=102400,
        dim=2048,
        inter_dim=10944,
        moe_inter_dim=1408,
        n_layers=27,
        n_dense_layers=1,
        n_heads=16,
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=2,
            top_k=6,
            score_func="softmax",
            route_norm=False,
            score_before_experts=False,
        ),
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mscale=0.70,
    )


def get_deepseek_v3_base_config() -> JobConfig:
    config = JobConfig()

    config.job.dump_folder = "./outputs"

    config.profiling.enable_profiling = False
    config.profiling.save_traces_folder = "profile_trace"
    config.profiling.profile_freq = 10
    config.profiling.enable_memory_snapshot = False
    config.profiling.save_memory_snapshot_folder = "memory_snapshot"

    config.metrics.log_freq = 1
    config.metrics.save_folder = "metrics"

    config.model.hf_assets_path = "./assets/hf/deepseek-moe-16b-base"
    config.model.args = get_deepseek_v3_model_args()

    config.optimizer.name = "AdamW"
    config.optimizer.lr = 2.2e-4
    config.optimizer.eps = 1e-8

    config.lr_scheduler.warmup_steps = 50
    config.lr_scheduler.decay_ratio = 0.8
    config.lr_scheduler.decay_type = "cosine"
    config.lr_scheduler.min_lr_factor = 0.1

    config.training.local_batch_size = 5
    config.training.seq_len = 4096
    config.training.max_norm = 1.0
    config.training.steps = 500
    config.training.dataset = "fineweb"

    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.fsdp_reshard_after_forward = "default"
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 1
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.expert_parallel_degree = 1
    config.parallelism.expert_tensor_parallel_degree = 1

    config.checkpoint.enable = True
    config.checkpoint.folder = "checkpoint"
    config.checkpoint.interval = 100
    config.checkpoint.async_mode = "async"

    config.activation_checkpoint.mode = "selective"
    config.activation_checkpoint.selective_ac_option = "op"

    config.compile.enable = True
    config.compile.components = ["model", "loss"]

    return config


def get_deepseek_v3_pp_tp_config() -> JobConfig:
    config = get_deepseek_v3_base_config()
    config.model.args.n_layers = 6
    config.training.local_batch_size = 4

    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    return config


def get_deepseek_v3_hsdp_ep_config() -> JobConfig:
    config = get_deepseek_v3_base_config()
    config.model.args.n_layers = 6
    config.training.local_batch_size = 1

    config.parallelism.data_parallel_replicate_degree = 2
    config.parallelism.data_parallel_shard_degree = 8
    config.parallelism.expert_parallel_degree = 8
    return config


def get_deepseek_v3_fsdp_cp_config() -> JobConfig:
    config = get_deepseek_v3_base_config()
    config.model.args.n_layers = 6
    config.training.local_batch_size = 2

    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 8
    config.parallelism.context_parallel_degree = 2
    return config


def get_deepseek_v3_fsdp_tp_config() -> JobConfig:
    config = get_deepseek_v3_base_config()
    # Reduce number of layers and batch size to fit in memory with DDP
    config.model.args.n_layers = 6
    config.training.local_batch_size = 8

    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 8
    return config


def get_deepseek_v3_hsdp_config() -> JobConfig:
    config = get_deepseek_v3_base_config()
    config.model.args.n_layers = 6
    config.training.local_batch_size = 1
    config.parallelism.data_parallel_replicate_degree = 2
    config.parallelism.data_parallel_shard_degree = 8
    return config


def get_deepseek_v3_ddp_config() -> JobConfig:
    config = get_deepseek_v3_base_config()

    # Reduce number of layers and batch size to fit in memory with DDP
    config.model.args.n_layers = 6
    config.training.local_batch_size = 1

    config.parallelism.data_parallel_replicate_degree = 16
    config.parallelism.data_parallel_shard_degree = 1
    return config


def get_deepseek_v3_fsdp_config() -> JobConfig:
    config = get_deepseek_v3_base_config()
    config.model.args.n_layers = 6
    config.training.local_batch_size = 1
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 16
    return config


def get_deepseek_v3_fsdp_ep_tp_config() -> JobConfig:
    config = get_deepseek_v3_base_config()
    config.model.args.n_layers = 6
    config.training.local_batch_size = 2

    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 8
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    config.parallelism.expert_tensor_parallel_degree = 1
    return config


def get_deepseek_v3_fsdp_ep_etp_config() -> JobConfig:
    config = get_deepseek_v3_base_config()
    config.model.args.n_layers = 6
    config.training.local_batch_size = 2

    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 8
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    config.parallelism.expert_tensor_parallel_degree = 2
    return config


config_map = {
    "hsdp": get_deepseek_v3_hsdp_config,
    "ddp": get_deepseek_v3_ddp_config,
    "fsdp": get_deepseek_v3_fsdp_config,
    "pp_tp": get_deepseek_v3_pp_tp_config,
    "fsdp_tp": get_deepseek_v3_fsdp_tp_config,
    "fsdp_cp": get_deepseek_v3_fsdp_cp_config,
    "hsdp_ep": get_deepseek_v3_hsdp_ep_config,
    "fsdp_ep_tp": get_deepseek_v3_fsdp_ep_tp_config,
    "fsdp_ep_etp": get_deepseek_v3_fsdp_ep_etp_config,
}


def get_config(name: str) -> JobConfig:
    return config_map[name]()
