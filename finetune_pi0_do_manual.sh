#!/bin/bash
#SBATCH --job-name=pi0-finetune
#SBATCH --output=logs/pi0-finetune-%j.out
#SBATCH --error=logs/pi0-finetune-%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos        
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00


source ~/.bashrc
conda activate openpi
cd ~/Downloads/ICL/openpi


# Compute normalization statistics (run once)
uv run scripts/compute_norm_stats.py --config-name pi0_do_manual_delta --max-frames 300 
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_do_manual_delta --exp-name=pi0_do_manual_run1 --overwrite


uv run scripts/compute_norm_stats.py --config-name pi0_do_manual_teleop --max-frames 300 
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_do_manual_teleop --exp-name=pi0_do_manual_run1 --overwrite

uv run scripts/compute_norm_stats.py --config-name pi0_do_manual_next --max-frames 300 
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_do_manual_next --exp-name=pi0_do_manual_run1 --overwrite

# launch with: sbatch finetune_pi0_do_manual.sh
