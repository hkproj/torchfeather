#!/bin/bash

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <singlenode|multinode|NUM_NODES> <config_name> [extra args...]"
    exit 1
fi

case $1 in
    singlenode) NODES=1 ;;
    multinode)  NODES=2 ;;
    *[!0-9]*)   echo "Error: first argument must be singlenode, multinode, or a number"; exit 1 ;;
    *)          NODES=$1 ;;
esac

CONFIG=$2
shift 2

if [ "$CONFIG" = "all" ]; then
    CONFIGS=$(grep -oP '^\s+"\K[^"]+(?=":)' torchfeather/config/default_configs.py)
    echo "Will schedule configs: $CONFIGS"
    echo "This will delete the outputs/ folder."
    read -p "Proceed? [y/N] " confirm
    if [ "$confirm" != "y" ]; then
        echo "Aborted."
        exit 0
    fi
    rm -rf outputs
    for cfg in $CONFIGS; do
        mkdir -p outputs/$cfg
        export WANDB_RUN_NAME="$cfg"
        sbatch --job-name="$cfg" --nodes=$NODES trainer.slurm "$@"
    done
else
    mkdir -p outputs/$CONFIG
    export WANDB_RUN_NAME="$CONFIG"
    sbatch --job-name="$CONFIG" --nodes=$NODES trainer.slurm "$@"
fi
