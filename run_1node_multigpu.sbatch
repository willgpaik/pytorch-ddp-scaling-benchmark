#!/bin/bash
#SBATCH --job-name=bench_1node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=00:45:00
#SBATCH --output=logs/bench_1node_%j.out
#SBATCH --error=logs/bench_1node_%j.err
# Adjust --gres=gpu:N to match your node.
# --ntasks=1 is correct: torchrun spawns the workers itself.

set -euo pipefail
mkdir -p logs results

module purge
# Adjust to your cluster's available CUDA version.
# B200/RTX Pro 6000 need CUDA 12.8+. Older GPUs need less.
module load cuda/13.1   # or whatever your cluster provides
module load miniforge3
# source activate <env>
# or conda activate <env>

export NCCL_DEBUG=WARN

echo "=== Job info ==="
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : $(hostname)"
echo "GPUs      : ${SLURM_GPUS_ON_NODE:-?}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python -c "import torch; print('torch:', torch.__version__); print('arch_list:', torch.cuda.get_arch_list())"
echo "================"

SCALING="${SCALING:-weak}"
PER_GPU_BS="${PER_GPU_BS:-256}"
GLOBAL_BS="${GLOBAL_BS:-2048}"
MODEL="${MODEL:-resnet50}"

NPROC=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
TAG="1node_${NPROC}gpu_${SCALING}_${MODEL}_${SLURM_JOB_ID}"
OUT="results/${TAG}.json"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=${NPROC} \
    benchmark_ddp.py \
        --scaling "${SCALING}" \
        --batch-size-per-gpu "${PER_GPU_BS}" \
        --global-batch-size "${GLOBAL_BS}" \
        --model "${MODEL}" \
        --precision bf16 \
        --warmup-sec 60 \
        --bench-sec 600 \
        --tag "${TAG}" \
        --output-json "${OUT}"
