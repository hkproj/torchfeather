#!/bin/bash

#SBATCH --chdir=/link-to-your-data-folder/torchfeather/
#SBATCH --cpus-per-task=16
#SBATCH --error=/link-to-your-data-folder/torchfeather/outputs/%x/%j.err
#SBATCH --gpus-per-node=8
#SBATCH --mem-per-gpu=128G
#SBATCH --output=/link-to-your-data-folder/outputs/%x/%j.out
#SBATCH --partition=name-of-your-partition

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

export TORCHFEATHER_CONFIG="$SLURM_JOB_NAME"

mkdir -p outputs/$TORCHFEATHER_CONFIG

srun uv run torchrun --nnodes $SLURM_JOB_NUM_NODES --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" -m torchfeather.train "$@"
