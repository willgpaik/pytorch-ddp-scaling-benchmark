#!/bin/bash
#SBATCH --job-name=bench_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=logs/bench_1gpu_%j.out
#SBATCH --error=logs/bench_1gpu_%j.err
# Time budget: 60s warmup + 600s bench + ~60s startup/teardown = 45 minutes.
# Adjust for your cluster:
#   --partition=<your-partition>
#   --constraint=<gpu-type>
#   --account=<your-account>

set -euo pipefail
mkdir -p logs results

module purge
# Adjust to your cluster's available CUDA version.
# B200/RTX Pro 6000 require CUDA 12.8+; older GPUs work with earlier versions.
module load cuda/13.1   # or whatever your cluster provides
module load miniforge3
# source activate <env>
# or conda activate <env>

export NCCL_DEBUG=WARN

echo "=== Job info ==="
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : $(hostname)"
echo "GPUs      : ${SLURM_GPUS_ON_NODE:-?}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import torch; print('torch:', torch.__version__); print('arch_list:', torch.cuda.get_arch_list())"
echo "================"

# Configuration via env vars; override at submit time:
#   sbatch --export=ALL,SCALING=strong,GLOBAL_BS=2048,PRECISION=fp16 run_1gpu.sbatch
SCALING="${SCALING:-weak}"
PER_GPU_BS="${PER_GPU_BS:-256}"
GLOBAL_BS="${GLOBAL_BS:-512}"
MODEL="${MODEL:-resnet152}"
PRECISION="${PRECISION:-bf16}"

TAG="1gpu_${SCALING}_${MODEL}_${PRECISION}_$(hostname)_${SLURM_JOB_ID}"
OUT="results/${TAG}.json"

torchrun \
    --standalone \
    --nproc_per_node=1 \
    benchmark_ddp.py \
        --scaling "${SCALING}" \
        --batch-size-per-gpu "${PER_GPU_BS}" \
        --global-batch-size "${GLOBAL_BS}" \
        --model "${MODEL}" \
        --precision "${PRECISION}" \
        --warmup-sec 60 \
        --bench-sec 600 \
        --tag "${TAG}" \
        --output-json "${OUT}"
