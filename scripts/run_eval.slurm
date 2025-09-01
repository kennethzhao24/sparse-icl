#!/bin/bash
#SBATCH --partition=gpuA100x4
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=4   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --gpus-per-node=1
#SBATCH --account=bekz-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH --time=02:00:00

echo "job is starting on `hostname`"

## Initialize Conda
module load cuda/12.4.0 # switch to CUDA 12.4
module load anaconda3_gpu
conda env list
source activate tts


CUDA_VISIBLE_DEVICES=0 python3 eval_vllm.py --model google/gemma-3-1b-it --task gsm8k --dtype bfloat16 --seed 42 --batch_size 32 --max_tokens 1024 --exp_name a100_gsm8k_bf16_seq_1k