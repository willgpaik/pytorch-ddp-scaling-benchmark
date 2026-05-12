# PyTorch DDP Scaling Benchmark

A reproducible benchmark for measuring PyTorch Distributed Data Parallel (DDP) scaling on NVIDIA GPUs. Reports weak scaling, strong scaling, speedup, parallel efficiency, and GPU activity across single-GPU, single-node multi-GPU, and multi-node multi-GPU configurations.

Designed for HPC environments using Slurm. Uses synthetic on-device data to isolate GPU compute and NCCL communication from data loading.

## Supported hardware

Tested on NVIDIA B200 (sm_100) and RTX Pro 6000 Blackwell (sm_120). Should work on any NVIDIA GPU supported by a recent PyTorch release, including A100 (sm_80), H100/H200 (sm_90), and newer architectures. The code itself is architecture-agnostic; only the PyTorch and CUDA versions need to match your hardware.

## Files

- `benchmark_ddp.py` — main script, launched via torchrun
- `run_1gpu.sbatch`, `run_1node_multigpu.sbatch`, `run_multinode.sbatch` — Slurm wrappers
- `submit_sweep.sh` — submit a full matrix of jobs
- `analyze_results.py` — aggregate JSON outputs into a table with speedup/efficiency
- `visualize_results.py` — generate PNG plots from results
- `PROGRAM_RUNDOWN.md` — full explanation of what the benchmark does and why

## Setup

You need a Python environment with PyTorch built for your hardware.

### Step 1: choose your PyTorch wheel

Visit https://pytorch.org/get-started/locally/ and pick the wheel matching your CUDA toolkit version. Newer GPUs need newer CUDA: B200 and RTX Pro 6000 Blackwell require CUDA 12.8+ (PyTorch 2.7+); older Hopper and Ampere cards work with earlier versions.

Example install (adjust the CUDA suffix as needed):

```bash
# Create environment (using conda/miniforge as an example; venv works too)
conda create --name pytorch-bench python=3.12 -y
source activate pytorch-bench # or conda activate pytorch-bench

# Install PyTorch for CUDA 13.0 (works for Blackwell and newer)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Or for CUDA 12.4 (works for Hopper, Ampere, Ada Lovelace)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Step 2: verify your install includes your GPU's compute capability

```bash
python -c "import torch; print(torch.cuda.get_arch_list())"
```

Match against your hardware:
- A100 → `sm_80`
- H100 / H200 → `sm_90`
- B200 → `sm_100`
- RTX Pro 6000 Blackwell → `sm_120`

If your compute capability is missing, PyTorch will fail at runtime with "no kernel image is available for execution on the device." Switch to a newer PyTorch wheel or use the NVIDIA NGC PyTorch container.

### Step 3: edit the sbatch scripts

Each sbatch script has a placeholder for environment activation near the top:

```bash
# source /path/to/your/venv/bin/activate
```

Replace with your activation command, e.g.:

```bash
module load miniforge3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pytorch-bench
```

You may also need to adjust:
- `--partition`, `--account`, `--constraint` for your Slurm setup
- `module load cuda/...` for your cluster's module names
- `--gres=gpu:N` to match GPUs per node

## Running

Single job:

```bash
sbatch run_1gpu.sbatch
sbatch run_1node_multigpu.sbatch
sbatch run_multinode.sbatch
```

Override defaults at submit time:

```bash
sbatch --export=ALL,SCALING=strong,GLOBAL_BS=4096 run_1node_multigpu.sbatch
```

Full sweep (1 GPU + 1 node + multi-node, both scaling modes, multiple global batch sizes):

```bash
./submit_sweep.sh
```

## Scaling modes

The fundamental question changes between modes.

**Weak scaling** (`SCALING=weak`, default): per-GPU batch fixed. Global batch grows with world size. Answers: "can each GPU stay fully utilized as we add more?" Throughput/GPU should be roughly constant. Falling throughput/GPU = communication overhead eating into compute.

**Strong scaling** (`SCALING=strong`): global batch fixed. Per-GPU batch shrinks with world size. Answers: "how much faster does the same problem run with more resources?" Speedup should approach N for N GPUs. Speedup well below N = communication or insufficient parallelism.

For a publishable scaling story, run BOTH:
- Weak scaling shows the system is healthy
- Strong scaling shows real-world parallel efficiency
- Strong scaling at multiple global batch sizes (e.g. 1024 and 2048) shows the tradeoff between parallel efficiency and per-step computational intensity

## Time budget

Defaults are 60s warmup + 600s measurement = ~11 minutes wall time per job, plus startup/teardown. Slurm `--time=00:45:00` is set with margin.

The benchmark uses a probe phase to measure step time, then computes a matching step count so all ranks execute the same number of steps. This is required for DDP correctness (rank-divergent step counts deadlock at the NCCL allreduce). The total measured time approximates `bench_sec` within ~5%.

If you enable `--compile`, the probe phase absorbs the torch.compile JIT cost. 60s should be enough for ResNet-50 with compile; for transformers or larger models you may need 120s+.

## Workload size knobs

Defaults in `submit_sweep.sh`:

- `MODEL=resnet50` — also supports `resnet101`, `resnet152`. Larger models stress the GPU and interconnect more; useful for distinguishing hardware tiers.
- `PER_GPU_BS=128` — conservative default that fits most cards. Bump higher on GPUs with more memory (e.g. 256+ on B200/H200/H100 80GB+) to exercise the full memory subsystem.
- `GLOBAL_BS_LIST=(1024 2048)` — global batch sizes for strong scaling. Submitting at multiple values reveals the tradeoff between parallel efficiency and per-step computational intensity.

> For ResNet-152 on ≤96GB cards, reduce to `PER_GPU_BS=64` and `GLOBAL_BS_LIST=(512 1024)`. ResNet-152's activation memory at the 1-GPU strong-scaling baseline is roughly 2.5x ResNet-50 at the same batch.

> **General OOM guidance**: if you hit OOM on the 1-GPU strong-scaling baseline, halve `GLOBAL_BS_LIST` values until it fits. If you hit OOM on multi-GPU runs, halve `PER_GPU_BS` instead. Aim for 1-GPU peak memory below 85% of GPU capacity for clean scaling numbers, since memory pressure inflates baseline step time and can produce misleading "superlinear" efficiency.

To change defaults for your environment, edit the variables at the top of `submit_sweep.sh`:

```bash
MODEL="resnet50"
PER_GPU_BS=128
GLOBAL_BS_LIST=(1024 2048)
```

Or override per-job via `sbatch --export=ALL,SCALING=...,PER_GPU_BS=...` without editing the sbatch files.

## Analyzing results

```bash
python analyze_results.py results/
```

Prints a table of per-config throughput plus speedup/efficiency for strong scaling, and flags suspicious results (high straggler ratio, high step-time variance, high memory spread across ranks, low GPU activity).

## Visualizing results

```bash
pip install matplotlib   # one-time
python visualize_results.py results/ --device-label "your-gpu-name"
```

Generates PNG plots in `plots/`: a 2x2 overview, individual plots for each metric (throughput, speedup, efficiency), and a GPU activity bar chart. The individual PNGs are sized for slide decks.

## Multi-node troubleshooting

If `run_multinode.sbatch` hangs at NCCL init:

1. Check `NCCL_DEBUG=INFO` output for the interface NCCL chose. If it picked the management network, set `NCCL_SOCKET_IFNAME` to the high-speed NIC (typically `ib0` for IB or something starting with `enp` for RoCE).
2. Confirm port 29500 is reachable between nodes.
3. If allreduce time is suspiciously high (per-GPU throughput drops sharply going from 1-node to 2-node), NCCL is likely on the slow network.

The `analyze_results.py` script flags step-time variance and straggler ratio per run. A straggler ratio above 1.10 means at least one rank is much slower than the others, usually a sign of NCCL or network issues, less often a faulty GPU.

If a job fails with a CUDA initialization error on a specific node (`Unable to determine device handle`), the node has a GPU/driver issue unrelated to this code. Exclude it with `sbatch --exclude=<node>` and report to your cluster admin.

## What this benchmark does NOT measure

- Data loading throughput (uses synthetic data on-device)
- Model accuracy or convergence (random labels, no validation)
- Mixed precision overhead beyond BF16 (no FP8/Transformer Engine)
- Optimizer overhead (SGD; Adam/AdamW have higher memory/compute)

If you need any of these, this is the wrong benchmark. For pure GPU and NCCL throughput characterization, this is the right one.

## Sample results

Results from our validation runs on B200 (16 GPUs across 2 nodes):

- ResNet-50 weak scaling: per-GPU throughput stays flat to 16 GPUs
- ResNet-50 strong scaling: near-100% efficiency at all tested global batches (1024, 2048)
- ResNet-152 strong scaling at global batch 1024: ~100% efficiency
- ResNet-152 strong scaling at global batch 512: drops to ~70% at 16 GPUs

The crossover where communication overhead becomes visible depends on model size and per-GPU batch. See `PROGRAM_RUNDOWN.md` for the underlying mechanics.

Generate your own with `visualize_results.py`.

## License

MIT License. See LICENSE file.

## Citation

If you use this benchmark in academic work, please cite:

```bibtex
@software{paik_pytorch_ddp_benchmark_2026,
  author = {Paik, Ghanghoon},
  title = {PyTorch DDP Scaling Benchmark},
  url = {https://github.com/willgpaik/pytorch-ddp-scaling-benchmark},
  year = {2026}
}
```
