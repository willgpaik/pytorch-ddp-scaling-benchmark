#!/bin/bash
# Submit a full benchmark matrix covering both scaling modes, all GPU counts,
# multiple models, and both fp16/bf16 precisions.
#
# Usage:
#   ./submit_sweep.sh
#
# Each job writes results/<tag>.json. Aggregate with analyze_results.py.
#
# Before running, set per-GPU batch sizes that fit your GPU's VRAM.
# Use find_max_bs.py to determine safe values:
#   python find_max_bs.py --model resnet152 --precision fp16
#   python find_max_bs.py --model vit_b_16  --precision fp16

set -euo pipefail

# === CONFIG ===
# Set PER_GPU_BS per model based on find_max_bs.py output for your GPU.
# These defaults are calibrated for A100 80GB.
# For V100 16GB: resnet152=128, vit_b_16=32
# For H100 80GB: resnet152=1024, vit_b_16=512
MODELS=("resnet152" "vit_b_16")
PRECISIONS=("fp16" "bf16")

declare -A PER_GPU_BS_MAP=(
    ["resnet152"]=512
    ["vit_b_16"]=1024
)

# Strong scaling global batch sizes.
# Must fit in 1-GPU VRAM: per_gpu = GLOBAL_BS / 1 = GLOBAL_BS at 1 GPU.
# Set to <= PER_GPU_BS_MAP values above.
declare -A GLOBAL_BS_MAP=(
    ["resnet152"]="256 512"
    ["vit_b_16"]="512 1024"
)
# === END CONFIG ===

for MODEL in "${MODELS[@]}"; do
    BS="${PER_GPU_BS_MAP[$MODEL]}"

    for PRECISION in "${PRECISIONS[@]}"; do
        # V100 does not support bf16 natively -- skip bf16 on V100
        # (add: if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" == *"V100"* ]] && [ "${PRECISION}" = "bf16" ]; then continue; fi)

        # Weak scaling: 1 GPU baseline
        sbatch --export=ALL,SCALING=weak,PER_GPU_BS=${BS},MODEL=${MODEL},PRECISION=${PRECISION} \
            run_1gpu.sbatch

        # Weak scaling: all GPUs on one node
        sbatch --export=ALL,SCALING=weak,PER_GPU_BS=${BS},MODEL=${MODEL},PRECISION=${PRECISION} \
            run_1node_multigpu.sbatch

        # Strong scaling
        for GBS in ${GLOBAL_BS_MAP[$MODEL]}; do
            sbatch --export=ALL,SCALING=strong,PER_GPU_BS=${BS},GLOBAL_BS=${GBS},MODEL=${MODEL},PRECISION=${PRECISION} \
                run_1gpu.sbatch
            sbatch --export=ALL,SCALING=strong,PER_GPU_BS=${BS},GLOBAL_BS=${GBS},MODEL=${MODEL},PRECISION=${PRECISION} \
                run_1node_multigpu.sbatch
        done
    done
done

echo "Submitted. Check with: squeue -u \$USER"
