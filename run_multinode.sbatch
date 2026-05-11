#!/bin/bash
#SBATCH --job-name=bench_multinode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=00:50:00
#SBATCH --output=logs/bench_multinode_%j.out
#SBATCH --error=logs/bench_multinode_%j.err
# Pattern: one Slurm task per node, srun launches torchrun on each node,
# torchrun fans out to nproc_per_node workers locally and rendezvous on rank0.

set -euo pipefail
mkdir -p logs results

module purge
# Adjust to your cluster's available CUDA version.
# B200/RTX Pro 6000 need CUDA 12.8+. Older GPUs need less.
module load cuda/13.1   # or whatever your cluster provides
module load miniforge3
# source activate <env>
# or conda activate <env>

# === Rendezvous setup ===
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
RDZV_PORT=29500

echo "=== Multi-node setup ==="
echo "Job ID         : ${SLURM_JOB_ID}"
echo "Nodes          : ${SLURM_NNODES}"
echo "Node list      : ${SLURM_JOB_NODELIST}"
echo "Head node      : ${head_node} (${head_node_ip})"
echo "Rendezvous port: ${RDZV_PORT}"
echo "GPUs/node      : ${SLURM_GPUS_ON_NODE:-?}"
echo "========================"

# NCCL: start with INFO to confirm it picks the right interface, drop to WARN
# once you've verified.
export NCCL_DEBUG=INFO
# Tune if NCCL picks the wrong interface:
# export NCCL_SOCKET_IFNAME=ib0
# export NCCL_IB_HCA=mlx5

SCALING="${SCALING:-weak}"
PER_GPU_BS="${PER_GPU_BS:-256}"
GLOBAL_BS="${GLOBAL_BS:-2048}"
MODEL="${MODEL:-resnet50}"

NPROC=${SLURM_GPUS_ON_NODE:-8}
WORLD=$((SLURM_NNODES * NPROC))
TAG="${SLURM_NNODES}node_${NPROC}gpu_${WORLD}rank_${SCALING}_${MODEL}_${SLURM_JOB_ID}"
OUT="results/${TAG}.json"

# Note: variables that should expand on each node (\${SLURM_NODEID}) are
# escaped; variables that should expand here (port, host, tag) are not.
srun --kill-on-bad-exit=1 bash -c "
    echo \"[\$(hostname)] launching torchrun, node_rank=\${SLURM_NODEID}\"
    torchrun \
        --nnodes=${SLURM_NNODES} \
        --nproc_per_node=${NPROC} \
        --node_rank=\${SLURM_NODEID} \
        --rdzv_id=${SLURM_JOB_ID} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${head_node_ip}:${RDZV_PORT} \
        benchmark_ddp.py \
            --scaling ${SCALING} \
            --batch-size-per-gpu ${PER_GPU_BS} \
            --global-batch-size ${GLOBAL_BS} \
            --model ${MODEL} \
            --precision bf16 \
            --warmup-sec 60 \
            --bench-sec 600 \
            --tag '${TAG}' \
            --output-json '${OUT}'
"
