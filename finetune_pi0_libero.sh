#!/bin/bash
#SBATCH --job-name=pi0-finetune
#SBATCH --output=logs/pi0-finetune-%j.out
#SBATCH --error=logs/pi0-finetune-%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate openpi
cd ~/Downloads/ICL/openpi

# Compute normalization statistics (run once)
uv run scripts/compute_norm_stats.py --config-name pi0_libero_low_mem_finetune --max-frames 300 #pi0_libero_low_mem_finetune is for lora


# IF DATASET IS FOUND LOCALLY
# HF_DATASETS_OFFLINE=1 HF_HOME=/l/users/malak.mansour/Datasets/libero/lerobot uv run scripts/compute_norm_stats.py --config-name pi0_libero_low_mem_finetune

# Set environment variables to force offline mode and use local dataset
# export HF_DATASETS_OFFLINE=1
# export HF_HOME=/l/users/malak.mansour/Datasets/libero/lerobot  # Path to your datasets
# export TRANSFORMERS_OFFLINE=1 
# export HF_HUB_OFFLINE=1  # This is important!
# export HF_HUB_DISABLE_TELEMETRY=1
# export HF_HUB_DISABLE_PROGRESS_BARS=1
# export HF_HUB_DISABLE_SYMLINKS_WARNING=1

# unset HF_DATASETS_OFFLINE
# unset HF_HOME
# unset TRANSFORMERS_OFFLINE
# unset HF_HUB_OFFLINE
# unset HF_HUB_DISABLE_TELEMETRY
# unset HF_HUB_DISABLE_PROGRESS_BARS
# unset HF_HUB_DISABLE_SYMLINKS_WARNING

# # Compute normalization statistics with limited frames
# uv run scripts/compute_norm_stats.py --config-name pi0_libero_low_mem_finetune --max-frames 100


# Fine-tune π₀ on LIBERO
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune --exp-name=pi0_libero_run1 --overwrite


# launch with: sbatch finetune_pi0_libero.sh
