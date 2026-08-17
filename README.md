# TorchFeather

TorchFeather is a distributed training framework derived from [TorchTitan](https://github.com/pytorch/torchtitan).

## Setup

Install the dependencies:

```bash
uv sync --all-extras
```

Download the tokenizer from [DeepSeek-MoE 16B Base](https://huggingface.co/deepseek-ai/deepseek-moe-16b-base):

```bash
uv run python scripts/download_hf_assets.py \
  --repo_id deepseek-ai/deepseek-moe-16b-base \
  --assets tokenizer
```

This stores it at `assets/hf/deepseek-moe-16b-base`, the path used by the default configurations.

## Run

Set the cluster-specific paths and partition in `trainer.slurm`, then launch a training job:

```bash
./launch.sh multinode fsdp
```
