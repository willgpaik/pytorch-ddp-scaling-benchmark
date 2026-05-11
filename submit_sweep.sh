#!/bin/bash
# Submit a matrix of jobs covering both scaling modes and all node counts.
# Edit the arrays to match what you want to sweep.
#
# Usage:
#   ./submit_sweep.sh
#
# Each job writes results/<tag>.json. Aggregate later with analyze_results.py.

set -euo pipefail

MODEL="resnet50"
PER_GPU_BS=128        # used in weak scaling
GLOBAL_BS_LIST=(1024 2048)   # used in strong scaling

# === 1 GPU baseline (always submit; weak == strong here) ===
sbatch --export=ALL,SCALING=weak,PER_GPU_BS=${PER_GPU_BS},MODEL=${MODEL} \
    run_1gpu.sbatch

# === Weak scaling: keep per-GPU batch fixed, scale up resources ===
sbatch --export=ALL,SCALING=weak,PER_GPU_BS=${PER_GPU_BS},MODEL=${MODEL} \
    run_1node_multigpu.sbatch
sbatch --export=ALL,SCALING=weak,PER_GPU_BS=${PER_GPU_BS},MODEL=${MODEL} \
    run_multinode.sbatch

# === Strong scaling: fix global batch, scale up resources ===
# Submit one job per global batch size at each scale.
for gbs in "${GLOBAL_BS_LIST[@]}"; do
    sbatch --export=ALL,SCALING=strong,GLOBAL_BS=${gbs},MODEL=${MODEL} \
        run_1gpu.sbatch
    sbatch --export=ALL,SCALING=strong,GLOBAL_BS=${gbs},MODEL=${MODEL} \
        run_1node_multigpu.sbatch
    sbatch --export=ALL,SCALING=strong,GLOBAL_BS=${gbs},MODEL=${MODEL} \
        run_multinode.sbatch
done

echo "Submitted. Check with: squeue -u \$USER"
