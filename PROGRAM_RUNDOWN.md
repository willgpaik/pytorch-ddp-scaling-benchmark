# Program rundown: PyTorch DDP Scaling Benchmark

This document explains what the benchmark does, why each piece is the way it is, and what the numbers mean. It is written assuming familiarity with HPC and Slurm but not deep ML knowledge.

The benchmark was developed and validated on NVIDIA B200 and RTX Pro 6000 Blackwell. Most of the discussion below uses those as concrete reference points, but the design is architecture-agnostic and works on any modern NVIDIA GPU. A section near the end discusses what to expect on different hardware.

## What problem are we solving?

We want to characterize two new GPU clusters (B200-based and RTX Pro 6000 Blackwell-based) for distributed deep-learning workloads. Specifically:

1. How fast can a single GPU train a representative model?
2. How well does that performance scale when we add more GPUs in the same node? (NVLink / PCIe interconnect)
3. How well does it scale across multiple nodes? (network interconnect)
4. Where are the bottlenecks (compute, memory, communication)?

The answers inform capacity planning, user training material, and grant proposal performance claims.

## Why this design instead of training a real model?

Training real models to convergence on real datasets is what users *do*, but it is the wrong tool for *characterizing hardware*. Real training mixes GPU compute, data loading from storage, model accuracy, optimizer state, learning rate schedules, and many other things. If we tried to use real training as a benchmark, we would not know whether a slow run was caused by the GPU, the network, the storage filesystem, or a model convergence quirk.

A benchmark isolates one variable. This benchmark isolates GPU compute and inter-GPU communication by removing everything else:

- **Synthetic data generated on the GPU** removes the storage and data loader from the picture
- **Random labels** mean we cannot measure accuracy, but accuracy is irrelevant to throughput
- **Fixed model and fixed precision** mean numbers are comparable across hardware
- **Time-based measurement** means all runs have similar statistical confidence

If users want to know "how long will my actual training job take," they need a different test. This benchmark answers "how fast is the hardware itself."

## The model: ResNet-50 (with options for larger)

ResNet-50 is a 50-layer convolutional neural network for image classification. We do not care about the classification — we care that:

- It is the most widely benchmarked model in the world. There are published numbers for every GPU since 2016, so our results are comparable to NVIDIA's, MLPerf's, and other clusters'.
- It exercises matrix multiplication, convolution, and batch normalization — the three primitives that dominate ML workloads.
- It is well-understood at all batch sizes.

The script also supports ResNet-101 and ResNet-152 via `MODEL=resnet152`. These are larger versions of the same architecture (more layers, more parameters) and put more compute and communication pressure on the GPU. If ResNet-50 underutilizes very capable GPUs (very possible — B200 in particular is enormous), running ResNet-152 gives a more honest picture of peak capability and reveals where communication becomes a bottleneck.

## The framework: PyTorch DDP

PyTorch is the dominant ML framework. DDP stands for Distributed Data Parallel. It is the simplest and most common approach to multi-GPU training.

The idea: each GPU holds a complete copy of the model. Each GPU processes a different chunk of the input batch. After computing gradients (the "how should the model change to improve" signal), all GPUs exchange their gradients and average them, so they all stay in sync. This averaging step is called allreduce and is the dominant communication pattern.

Why DDP and not something fancier (FSDP, tensor parallelism, pipeline parallelism)? Because ResNet-50/101/152 fits comfortably on any modern GPU, so model-parallel approaches are unnecessary. DDP is what the vast majority of users run. Fancier parallelism is for huge models (LLMs).

Allreduce uses NCCL (NVIDIA Collective Communications Library) under the hood. NCCL automatically picks the fastest available network: NVLink within a node, InfiniBand or RoCE between nodes, plain TCP as a last resort.

## The launcher: torchrun

`torchrun` is PyTorch's standard launcher for distributed jobs. We use it the same way for all three scenarios; only the arguments change:

- 1 GPU: `torchrun --nproc_per_node=1`
- 1 node, N GPUs: `torchrun --nproc_per_node=N`
- M nodes, N GPUs each: `torchrun --nnodes=M --nproc_per_node=N --rdzv_endpoint=...`

In the multi-node case, one process on each node (launched by `srun`) calls `torchrun`, which then spawns N worker processes locally and they all rendezvous through a coordination service running on the first node.

## Scaling modes: weak vs strong

This is the central conceptual point and the one most worth understanding deeply.

### Weak scaling

Each GPU processes the same amount of work. The total problem grows as we add GPUs.

- 1 GPU at batch 128: total batch = 128
- 8 GPUs at batch 128 each: total batch = 1024
- 16 GPUs at batch 128 each: total batch = 2048

The question this answers: "Can each GPU stay busy as we add more GPUs?" If yes, throughput per GPU stays roughly constant and total throughput grows linearly. If communication overhead eats into compute time, per-GPU throughput drops.

This is what users do in practice. When you get more GPUs, you usually want to train on more data or with bigger batches.

### Strong scaling

The total problem is fixed. Each additional GPU gets a smaller piece of it.

- 1 GPU processes batch 2048: per-GPU batch = 2048
- 8 GPUs share batch 2048: per-GPU batch = 256
- 16 GPUs share batch 2048: per-GPU batch = 128

The question this answers: "How much faster does the *same* problem run with more resources?" If perfect, 8 GPUs make the problem 8x faster. In practice, communication overhead and reduced per-GPU work make the speedup less than ideal.

This is the classic HPC speedup curve. It is the most damning view of communication overhead.

### Why we run both

Weak scaling tells you the system is healthy — each GPU is staying busy. Strong scaling tells you what real parallel efficiency looks like. They are complementary.

A common failure mode: weak scaling looks great (each GPU at 95% throughput across all configurations), but strong scaling is terrible (16 GPUs only 4x faster than 1 GPU). This means the system can absorb work but cannot make individual problems faster. That is a real and important conclusion for grant proposals.

## Time-based measurement

The benchmark runs for a wall-clock target (default 10 minutes of measurement after 1 minute of warmup) instead of a fixed number of steps. Reasons:

1. Different configurations have different step times. A 1-GPU run does fewer steps than a 16-GPU strong-scaling run in the same time. Time-based runs give all of them similar statistical confidence.
2. Step-counted runs on a fast GPU finish in seconds, dominated by startup noise.
3. 10 minutes is long enough that any cluster-wide noise (other users on shared infrastructure, network jitter, power management) averages out.

**Implementation detail**: all ranks must execute exactly the same number of steps to avoid DDP deadlock (a rank that exits the loop early blocks at the next barrier while the staying ranks block at the next allreduce). The benchmark uses a short probe phase to measure per-step time, then has all ranks agree (via NCCL all-reduce with MAX) on a step count that satisfies the wall-time target. The total measured time approximates `bench_sec` within roughly 5%.

## Warmup matters

The first ~30 seconds of any GPU workload is unrepresentative. Multiple things are happening:

- **cuDNN autotuning**: cuDNN is NVIDIA's deep learning library. It picks the fastest implementation of each operation by trying several and timing them. This happens lazily on first invocation.
- **NCCL initialization**: First-time allreduce establishes connections, allocates buffers, picks algorithms.
- **GPU memory allocator warmup**: PyTorch's memory pool stabilizes.
- **`torch.compile` JIT** (if enabled): up to 30 seconds of graph capture and code generation.

We exclude all of this from the measurement. The benchmark literally throws away the warmup steps' timing data.

## Numerical precision: BF16

The benchmark uses BFloat16 (BF16) by default, not the older FP32. Reasons:

- BF16 is what real users run. Almost no one trains in FP32 anymore on modern hardware.
- BF16 uses tensor cores efficiently on Hopper, Blackwell, and Ampere.
- BF16 has the same numerical range as FP32 (just less precision), so we do not need gradient scaling tricks.
- B200 also supports FP8 via NVIDIA's Transformer Engine, which would give better numbers, but adds complexity and is workload-specific. We did not include it because not all target GPUs support FP8 equally and the comparison would be unfair.

## What the script measures and reports

For each benchmark run, the script writes a JSON file with:

- **Configuration**: world size, batch sizes, model, precision
- **Timing**: mean step time, p50/p95/p99 step time, standard deviation
- **Throughput**: images per second per GPU and globally
- **Memory**: peak GPU memory across all ranks (min/mean/max/spread)
- **Per-rank stats**: same metrics for every rank, so we can spot stragglers
- **GPU sampling**: nvidia-smi polled once per second during the bench window — activity %, power draw, SM clock, memory used
- **Environment**: PyTorch version, CUDA version, GPU model

## What "GPU activity %" actually means

This is important and easy to misread. We sample `nvidia-smi`'s `utilization.gpu` field every second. NVIDIA documents this as "Percent of time over the past sample period during which one or more kernels was executing on the GPU."

This is NOT:

- SM occupancy (% of the SMs on the GPU that are doing work)
- FLOP utilization (% of peak FLOPs being delivered)
- Memory bandwidth utilization

A kernel running on 1 SM out of 132 (B200) reads as 100%. A kernel using all SMs at full speed also reads as 100%.

What it IS useful for: detecting *under*-utilization. If activity is 50%, the GPU is sitting idle half the time and the bottleneck is elsewhere (CPU, communication, memory). High activity does not prove peak performance, but low activity proves the GPU is not the bottleneck.

For real SM and FLOP utilization, NVIDIA provides tools like DCGM and Nsight Compute. We did not require those because they add cluster-wide dependencies and the activity % metric is sufficient for our purposes.

## File-by-file walkthrough

### `benchmark_ddp.py`

The main benchmark. Single-file Python that handles all three scenarios. Launched by torchrun. Reads CLI flags for scaling mode, batch size, model, etc. Writes a JSON to `results/`.

Key sections in the file:
- `parse_args`: command-line flags with sensible defaults
- `setup_dist`: initializes the PyTorch distributed group from environment variables that torchrun sets
- `NvidiaSmiSampler`: background thread that polls nvidia-smi during the bench window
- `resolve_batch_size`: computes per-GPU batch from scaling mode
- `benchmark`: the actual measurement function with probe-based warmup, bench, stats gathering
- `main`: entry point with proper cleanup

### `run_1gpu.sbatch`, `run_1node_multigpu.sbatch`, `run_multinode.sbatch`

Slurm submission scripts. They are intentionally simple wrappers — they set up the environment, configure NCCL, then call torchrun. The multi-node script is the only one with non-trivial logic: it discovers the head node, resolves its IP, and uses srun to launch torchrun on each node with the rendezvous endpoint.

Configuration is via environment variables (SCALING, PER_GPU_BS, GLOBAL_BS, MODEL) so the same sbatch file can run different experiments without editing.

### `submit_sweep.sh`

Convenience driver that submits jobs covering the full matrix: 1 GPU + 1 node + multi-node, both scaling modes, multiple global batch sizes for strong scaling. Edit the variables at the top to change the sweep.

### `analyze_results.py`

Reads all JSONs in `results/`, prints two tables (weak scaling and strong scaling), computes speedup and parallel efficiency for strong scaling, and flags suspicious results: high straggler ratio, high step-time variance, high memory spread between ranks, or low GPU activity.

### `visualize_results.py`

Generates PNG plots from the JSONs:
- Throughput vs world size (weak scaling)
- Throughput vs world size (strong scaling)
- Speedup vs world size with ideal-linear reference (strong scaling)
- Parallel efficiency vs world size (strong scaling)
- GPU activity per configuration (bar chart)

Outputs both a 2x2 overview PNG and individual PNGs suitable for slide decks.

## How to read the plots

### Throughput plots (log-log scale)

Both axes log scale. Lines should be roughly straight. For weak scaling, the global throughput line should be straight at a 1:1 slope (doubling GPUs doubles throughput). The per-GPU line should be roughly flat.

For strong scaling, the global throughput line should be straight at a 1:1 slope IF parallel efficiency is 100%. Less-than-1:1 slope means falling efficiency.

### Speedup plot (log-log scale)

Speedup vs world size. The dashed black line is "ideal linear" — perfect speedup of N for N GPUs. Real curves bend below the line as world size grows. The gap between your curve and the dashed line is the parallel inefficiency.

### Efficiency plot (linear y-axis)

Speedup divided by world size, expressed as a fraction. 1.0 = ideal. The curves drop with scale. Industry rule of thumb: above 0.8 (80%) is acceptable, above 0.9 is great, below 0.5 means something is seriously wrong.

You may occasionally see efficiency slightly above 1.0 (superlinear speedup). This is a real phenomenon, not a bug. The most common cause is cuDNN selecting different kernels for the smaller per-GPU batch in strong scaling configurations. Smaller batches can hit more arithmetic-efficient implementations or fit better in cache.

### GPU activity bar chart

One bar per configuration. Bars below 80% (orange dotted line) suggest the GPU was not the bottleneck for that run. Possible causes: small per-GPU batch size in strong scaling, communication overhead saturating, CPU-side bottleneck.

## Common patterns to expect on B200 / RTX Pro 6000

If everything is healthy, you should see:

- Weak scaling: per-GPU throughput nearly constant, global throughput growing linearly
- Strong scaling: speedup near-linear up to single-node max, with possibly a small kink at the node boundary where multi-node communication kicks in
- GPU activity: 90%+ at large per-GPU batches, possibly dropping at small per-GPU batches in strong scaling
- Memory spread: under 5% across ranks
- Straggler ratio: under 1.05

Red flags:

- Per-GPU throughput dropping sharply in weak scaling: communication is bottlenecked
- Big drop in efficiency at the multi-node boundary: NCCL may be on the slow network (check NCCL_DEBUG=INFO output)
- One rank consistently slower than others: faulty GPU or CPU contention
- GPU activity below 80% even at large batches: workload too small for the GPU (try ResNet-152)

## What to expect on other architectures

The benchmark is architecture-agnostic and produces the same kinds of plots and metrics regardless of hardware. What changes is the absolute numbers.

**On A100 (sm_80)**: ResNet-50 throughput will be roughly 30-40% of B200 numbers per GPU. Scaling behavior is similar in shape because NVLink and DDP semantics are unchanged. Memory is smaller (40GB or 80GB depending on variant), so you may need to reduce `PER_GPU_BS` for the strong-scaling 1-GPU baseline.

**On H100/H200 (sm_90)**: Roughly 60-80% of B200 per-GPU throughput depending on workload. Memory is 80GB (H100) or 141GB (H200). Scaling behavior should match B200 closely since the interconnect (NVLink 4) is the same generation.

**On older consumer GPUs (RTX 30/40 series)**: Will work but per-GPU throughput will be much lower (factor of 5-10x slower than datacenter cards) and there is no NVLink in most consumer cards, so single-node multi-GPU will use PCIe and show worse scaling. Multi-node scaling will be very poor unless you have a high-speed network.

**General rule**: as long as the scaling *curves* show sensible shapes (weak scaling flat, strong scaling near-linear within a node), the benchmark is doing its job. Absolute numbers will differ by hardware tier.

## What this benchmark does NOT tell you

- **End-to-end training time**: We do not include data loading, validation, checkpointing, or convergence dynamics. A real training job will be slower than the throughput here suggests, because data loaders are usually a 5-15% overhead.
- **Inference performance**: Different operation patterns. Use a different benchmark.
- **LLM training**: ResNet is a CNN; LLMs are transformers with very different communication patterns (attention all-to-all). Use Megatron-LM or similar for LLM characterization.
- **Energy efficiency over a full training run**: We capture instantaneous power draw, not energy-to-accuracy.

## Caveats and uncertainty

A few things in this benchmark and documentation are estimates rather than measured facts:

- The expected throughput ratios across A100/H100/H200/B200 in the section above are rough estimates based on published vendor and MLPerf data, not measured by this benchmark. Your actual numbers may differ by 20-30%.
- Default batch sizes (`PER_GPU_BS=128`, `GLOBAL_BS_LIST=(1024 2048)`) are sized for ResNet-50. Larger models need smaller batches; see the README for ResNet-152 recommendations.
- Memory pressure on the 1-GPU strong-scaling baseline can produce apparent superlinear speedup. If 1-GPU peak memory exceeds ~90% of GPU capacity, the baseline step time is inflated by allocator pressure, and reported strong-scaling efficiency may exceed 100% as an artifact. Choose global batch sizes that keep 1-GPU memory under 80% of capacity for clean scaling numbers.

When publishing numbers, label them with the exact configuration used. There is no single "ResNet-50 on B200" number; there is "ResNet-50, BF16, batch 128, channels-last, no compile, on B200" and so on.
