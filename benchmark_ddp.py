#!/usr/bin/env python3
"""
DDP benchmark: ResNet on synthetic data, time-based measurement.

Scaling modes:
  --scaling weak   : per-GPU batch is fixed; global batch grows with world size.
                     Shows: "can each GPU stay busy as we add more?"
  --scaling strong : global batch is fixed; per-GPU batch shrinks with world size.
                     Shows: "how much faster does the same problem run?"

Measurement is time-based, not step-based, so all configurations get
statistically comparable samples regardless of per-step speed.

Launched via torchrun. Same script handles 1 GPU, 1 node multi-GPU, and
multi-node multi-GPU.
"""

import argparse
import json
import os
import socket
import statistics
import subprocess
import threading
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models


def parse_args():
    p = argparse.ArgumentParser()

    # Scaling mode and batch sizing
    p.add_argument("--scaling", choices=["weak", "strong"], default="weak",
                   help="weak: fixed per-GPU batch. strong: fixed global batch.")
    p.add_argument("--batch-size-per-gpu", type=int, default=256,
                   help="Per-GPU batch (used in weak scaling)")
    p.add_argument("--global-batch-size", type=int, default=2048,
                   help="Global batch (used in strong scaling)")

    # Workload
    p.add_argument("--model", default="resnet50",
                   choices=["resnet50", "resnet101", "resnet152", "vit_b_16"],
                   help="Larger models stress big GPUs more honestly. "
                        "vit_b_16 is communication-heavy vs resnet compute-heavy.")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-classes", type=int, default=1000)

    # Time-based measurement
    p.add_argument("--warmup-sec", type=float, default=60.0,
                   help="Wall seconds of warmup. Long enough to absorb "
                        "torch.compile JIT and cuDNN autotune.")
    p.add_argument("--bench-sec", type=float, default=600.0,
                   help="Wall seconds of measured steady-state run")
    p.add_argument("--min-warmup-steps", type=int, default=20,
                   help="Minimum warmup steps even if warmup-sec elapses fast")
    p.add_argument("--min-bench-steps", type=int, default=50,
                   help="Minimum measured steps even if bench-sec elapses fast")

    # Numerics & perf knobs
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--channels-last", action="store_true", default=True)
    p.add_argument("--no-channels-last", dest="channels_last", action="store_false")
    p.add_argument("--cudnn-benchmark", action="store_true", default=True,
                   help="Enable cudnn.benchmark for autotuning")
    p.add_argument("--compile", action="store_true",
                   help="Use torch.compile (adds ~30s to warmup, often 10-20% faster)")
    p.add_argument("--smi-sample-sec", type=float, default=1.0,
                   help="nvidia-smi sampling interval. 0 to disable sampling.")

    # Output
    p.add_argument("--output-json", type=str, default=None,
                   help="Path for rank0 to write results JSON")
    p.add_argument("--tag", type=str, default="run", help="Label for this run")
    return p.parse_args()


def setup_dist():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world_size


class NvidiaSmiSampler:
    """Background thread that polls nvidia-smi and records GPU activity stats.

    Important caveats about what these numbers mean:
      - utilization.gpu is "% of time at least one kernel was active during
        the sample interval." NOT SM occupancy. NOT FLOP utilization.
        A kernel using 1 SM out of 132 reads as 100%.
      - utilization.memory is "% of time the memory controller was active",
        also not memory bandwidth utilization.
      - power.draw and clocks.sm are useful sanity checks: a GPU at peak
        power and base clocks is likely actually busy; one at idle power
        but reporting 100% util is suspect.

    For real SM occupancy you'd need DCGM or Nsight Compute. This sampler
    is a sanity check, not a profiling tool.

    Samples ALL visible GPUs on the local node. Each rank's local_rank
    selects the relevant device from the result.
    """

    def __init__(self, local_rank, interval_sec=1.0):
        self.local_rank = local_rank
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread = None
        self._samples = []  # list of (timestamp, util, mem_util, mem_mb, power_w, sm_clock_mhz)
        self._available = self._check_smi_available()

    @staticmethod
    def _check_smi_available():
        try:
            subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, check=True, timeout=5,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError,
                subprocess.TimeoutExpired):
            return False

    def _sample_once(self):
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 f"--id={self.local_rank}",
                 "--query-gpu=utilization.gpu,utilization.memory,memory.used,"
                 "power.draw,clocks.sm",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            # Some fields can be "[N/A]" depending on driver/GPU; convert defensively
            def _to_float(s):
                try:
                    return float(s)
                except (ValueError, TypeError):
                    return float("nan")
            return tuple(_to_float(p) for p in parts)
        except Exception:
            return None

    def _run(self):
        while not self._stop.is_set():
            sample = self._sample_once()
            if sample is not None:
                self._samples.append((time.perf_counter(),) + sample)
            self._stop.wait(self.interval_sec)

    def start(self):
        if not self._available:
            return False
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5)

    def summary(self):
        """Return min/mean/max for each metric, or None if no samples."""
        if not self._samples:
            return None
        # samples = [(t, util, mem_util, mem_mb, power_w, sm_clock), ...]
        utils = [s[1] for s in self._samples]
        mem_utils = [s[2] for s in self._samples]
        mem_mb = [s[3] for s in self._samples]
        power = [s[4] for s in self._samples]
        sm_clock = [s[5] for s in self._samples]

        def _stats(xs):
            xs = [x for x in xs if x == x]  # drop NaN
            if not xs:
                return {"min": None, "mean": None, "max": None}
            return {"min": min(xs), "mean": sum(xs) / len(xs), "max": max(xs)}

        return {
            "n_samples": len(self._samples),
            "interval_sec": self.interval_sec,
            "gpu_activity_pct": _stats(utils),
            "mem_controller_pct": _stats(mem_utils),
            "mem_used_mb": _stats(mem_mb),
            "power_w": _stats(power),
            "sm_clock_mhz": _stats(sm_clock),
        }


def resolve_batch_size(args, world_size, rank):
    """Compute per-GPU batch from scaling mode."""
    if args.scaling == "weak":
        per_gpu = args.batch_size_per_gpu
        global_bs = per_gpu * world_size
    else:  # strong
        if args.global_batch_size % world_size != 0:
            raise ValueError(
                f"global-batch-size {args.global_batch_size} not divisible "
                f"by world_size {world_size}"
            )
        per_gpu = args.global_batch_size // world_size
        global_bs = args.global_batch_size
    if rank == 0:
        print(f"[rank0] scaling={args.scaling} per_gpu={per_gpu} "
              f"global={global_bs}", flush=True)
    return per_gpu, global_bs


def get_amp_dtype(precision):
    return {"fp32": torch.float32, "bf16": torch.bfloat16,
            "fp16": torch.float16}[precision]


def build_model(name, num_classes):
    return getattr(models, name)(num_classes=num_classes)


def is_vit(model_name):
    return model_name.startswith("vit")


def make_synthetic_batch(batch_size, image_size, num_classes, device,
                         channels_last):
    images = torch.randn(batch_size, 3, image_size, image_size,
                         device=device, dtype=torch.float32)
    if channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    return images, labels


def benchmark(args, rank, local_rank, world_size):
    device = torch.device(f"cuda:{local_rank}")
    amp_dtype = get_amp_dtype(args.precision)
    use_amp = args.precision != "fp32"

    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    per_gpu_batch, global_batch = resolve_batch_size(args, world_size, rank)

    model = build_model(args.model, args.num_classes)
    # ViT does not benefit from channels_last (not a conv-first architecture)
    use_channels_last = args.channels_last and not is_vit(args.model)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)

    if args.compile:
        model = torch.compile(model)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if args.precision == "fp16" else None

    images, labels = make_synthetic_batch(
        per_gpu_batch, args.image_size, args.num_classes,
        device, use_channels_last,
    )

    def step():
        optimizer.zero_grad(set_to_none=True)
        amp_ctx = (torch.amp.autocast("cuda", dtype=amp_dtype)
                   if use_amp else nullcontext())
        with amp_ctx:
            output = model(images)
            loss = criterion(output, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    # === Warmup ===
    # IMPORTANT: all ranks MUST do the same number of warmup steps. If different
    # ranks exit at different step counts, the leaving ranks block at the next
    # barrier while the staying ranks block at the next allreduce, causing
    # a deadlock that's only resolved by the NCCL watchdog timeout.
    #
    # Strategy: do a small fixed number of probe steps (also handles cuDNN/
    # compile warmup), measure step time, then have all ranks agree on a
    # step count that satisfies the warmup_sec target.
    if rank == 0:
        print(f"[rank0] Warmup: target {args.warmup_sec}s, "
              f"min {args.min_warmup_steps} steps", flush=True)

    warmup_start = time.perf_counter()

    # Phase 1: probe steps to absorb cuDNN/compile JIT and measure step time.
    # Generous count because torch.compile graph capture happens here.
    probe_steps = 10
    for _ in range(probe_steps):
        step()
    torch.cuda.synchronize()
    dist.barrier()

    # Phase 2: time a few clean steps to estimate steady-state step duration.
    t0 = time.perf_counter()
    timed_probe = 5
    for _ in range(timed_probe):
        step()
    torch.cuda.synchronize()
    measured_step_sec = max((time.perf_counter() - t0) / timed_probe, 1e-6)

    # Phase 3: compute remaining steps needed, agree across ranks.
    elapsed_so_far = time.perf_counter() - warmup_start
    remaining_sec = max(0.0, args.warmup_sec - elapsed_so_far)
    extra_steps_local = max(
        0,
        max(args.min_warmup_steps - (probe_steps + timed_probe),
            int(remaining_sec / measured_step_sec) + 1),
    )
    # Take the max across ranks so the slowest rank still hits its target.
    extra_t = torch.tensor([extra_steps_local], device=device,
                           dtype=torch.long)
    dist.all_reduce(extra_t, op=dist.ReduceOp.MAX)
    extra_steps = int(extra_t.item())

    for _ in range(extra_steps):
        step()
    torch.cuda.synchronize()
    dist.barrier()

    warmup_steps = probe_steps + timed_probe + extra_steps
    warmup_elapsed = time.perf_counter() - warmup_start

    if rank == 0:
        print(f"[rank0] Warmup done: {warmup_steps} steps in "
              f"{warmup_elapsed:.1f}s", flush=True)

    torch.cuda.reset_peak_memory_stats(device)

    # === Measurement ===
    if rank == 0:
        print(f"[rank0] Benchmark: target {args.bench_sec}s, "
              f"min {args.min_bench_steps} steps", flush=True)

    # Each rank samples its own local GPU. Disable with --smi-sample-sec 0.
    sampler = None
    if args.smi_sample_sec > 0:
        sampler = NvidiaSmiSampler(local_rank, interval_sec=args.smi_sample_sec)
        if not sampler.start():
            if rank == 0:
                print("[rank0] nvidia-smi unavailable; skipping GPU sampling",
                      flush=True)
            sampler = None

    step_times = []
    bench_start = time.perf_counter()

    # Same rule as warmup: every rank must do the SAME number of steps.
    # We estimate target_steps from the measured warmup step time and have
    # all ranks agree on the max.
    bench_target_steps_local = max(
        args.min_bench_steps,
        int(args.bench_sec / measured_step_sec) + 1,
    )
    bench_target_t = torch.tensor([bench_target_steps_local], device=device,
                                  dtype=torch.long)
    dist.all_reduce(bench_target_t, op=dist.ReduceOp.MAX)
    bench_target_steps = int(bench_target_t.item())

    if rank == 0:
        print(f"[rank0] Bench target: {bench_target_steps} steps "
              f"(estimated {bench_target_steps * measured_step_sec:.0f}s)",
              flush=True)

    for _ in range(bench_target_steps):
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_times.append(t1 - t0)

    if sampler is not None:
        sampler.stop()

    dist.barrier()
    bench_elapsed = time.perf_counter() - bench_start
    n_steps = len(step_times)

    smi_summary = sampler.summary() if sampler is not None else None

    # Stats
    mean_step = statistics.mean(step_times)
    stdev_step = statistics.stdev(step_times) if n_steps > 1 else 0.0
    sorted_t = sorted(step_times)
    p50 = sorted_t[n_steps // 2]
    p95 = sorted_t[int(0.95 * n_steps)]
    p99 = sorted_t[int(0.99 * n_steps)]

    images_per_sec_local = per_gpu_batch / mean_step
    images_per_sec_global = images_per_sec_local * world_size

    peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    # Gather per-rank
    stats_local = torch.tensor(
        [mean_step, stdev_step, p50, p95, p99,
         images_per_sec_local, peak_mem_gb, float(n_steps)],
        device=device,
    )
    stats_all = [torch.zeros_like(stats_local) for _ in range(world_size)]
    dist.all_gather(stats_all, stats_local)

    # Gather smi summaries (nested dicts) via object collective.
    # Use gather_object so only rank0 needs to allocate the full list.
    smi_all = [None] * world_size if rank == 0 else None
    dist.gather_object(smi_summary, smi_all if rank == 0 else None, dst=0)

    if rank == 0:
        per_rank = [s.cpu().tolist() for s in stats_all]
        slowest_rank = max(range(world_size),
                           key=lambda i: per_rank[i][0])
        fastest_rank = min(range(world_size),
                           key=lambda i: per_rank[i][0])
        straggler_ratio = (per_rank[slowest_rank][0]
                           / per_rank[fastest_rank][0])

        # Memory aggregation across ranks. In DDP with synthetic data and
        # identical model replicas, ranks should be within ~1% of each other.
        # Larger spread suggests something rank-specific (e.g. NCCL buffer
        # allocator differences, async checkpointing, logging).
        mem_per_rank = [s[6] for s in per_rank]
        mem_stats = {
            "min_gb": min(mem_per_rank),
            "mean_gb": sum(mem_per_rank) / len(mem_per_rank),
            "max_gb": max(mem_per_rank),
            "spread_pct": (
                (max(mem_per_rank) - min(mem_per_rank))
                / max(mem_per_rank) * 100
                if max(mem_per_rank) > 0 else 0.0
            ),
        }

        # Topology snapshot (rank 0 only, best-effort)
        topo_raw = None
        try:
            topo_raw = subprocess.check_output(
                ["nvidia-smi", "topo", "--matrix"], text=True, timeout=10
            ).strip()
        except Exception:
            pass

        results = {
            "tag": args.tag,
            "scaling_mode": args.scaling,
            "world_size": world_size,
            "nodes": int(os.environ.get("SLURM_NNODES", "1")),
            "gpus_per_node": int(os.environ.get("SLURM_GPUS_ON_NODE",
                                  str(torch.cuda.device_count()))),
            "model": args.model,
            "batch_size_per_gpu": per_gpu_batch,
            "global_batch_size": global_batch,
            "image_size": args.image_size,
            "precision": args.precision,
            "channels_last": use_channels_last,
            "compile": args.compile,
            "warmup_sec_target": args.warmup_sec,
            "warmup_sec_actual": warmup_elapsed,
            "warmup_steps": warmup_steps,
            "bench_sec_target": args.bench_sec,
            "bench_sec_actual": bench_elapsed,
            "bench_steps": n_steps,
            "mean_step_sec": mean_step,
            "stdev_step_sec": stdev_step,
            "p50_step_sec": p50,
            "p95_step_sec": p95,
            "p99_step_sec": p99,
            "images_per_sec_per_gpu": images_per_sec_local,
            "images_per_sec_global": images_per_sec_global,
            "peak_mem_gb": mem_stats,
            "straggler_ratio": straggler_ratio,
            "slowest_rank": slowest_rank,
            "fastest_rank": fastest_rank,
            "gpu_sampling_rank0": smi_all[0] if smi_all else None,
            "per_rank": [
                {"rank": i, "mean_step_sec": s[0], "stdev_step_sec": s[1],
                 "p50_step_sec": s[2], "p95_step_sec": s[3],
                 "p99_step_sec": s[4], "images_per_sec": s[5],
                 "peak_mem_gb": s[6], "n_steps": int(s[7]),
                 "gpu_sampling": (smi_all[i] if smi_all else None)}
                for i, s in enumerate(per_rank)
            ],
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
            "nccl_topology": topo_raw,
        }

        print("\n" + "=" * 64, flush=True)
        print(f"RESULTS [{args.tag}]", flush=True)
        print("=" * 64, flush=True)
        print(f"Scaling mode         : {args.scaling}", flush=True)
        print(f"World size / nodes   : {world_size} / "
              f"{results['nodes']}", flush=True)
        print(f"Device               : {results['device_name']}", flush=True)
        print(f"Compute capability   : {results['device_capability']}", flush=True)
        print(f"Model                : {args.model}", flush=True)
        print(f"Per-GPU / global bs  : {per_gpu_batch} / {global_batch}", flush=True)
        print(f"Precision            : {args.precision}"
              f"{' +compile' if args.compile else ''}", flush=True)
        print(f"Steps measured       : {n_steps} in {bench_elapsed:.1f}s", flush=True)
        print(f"Mean step (s)        : {mean_step:.4f} "
              f"(stdev {stdev_step:.4f})", flush=True)
        print(f"p50 / p95 / p99 (s)  : {p50:.4f} / {p95:.4f} / "
              f"{p99:.4f}", flush=True)
        print(f"Images/sec per GPU   : {images_per_sec_local:.1f}", flush=True)
        print(f"Images/sec global    : {images_per_sec_global:.1f}", flush=True)
        print(f"Peak mem (GB)        : "
              f"min={mem_stats['min_gb']:.2f} "
              f"mean={mem_stats['mean_gb']:.2f} "
              f"max={mem_stats['max_gb']:.2f} "
              f"spread={mem_stats['spread_pct']:.1f}%", flush=True)
        print(f"Straggler ratio      : {straggler_ratio:.3f} "
              f"(slowest/fastest rank)", flush=True)

        # GPU activity summary from rank 0's local GPU.
        # NOTE: utilization.gpu is "% time at least one kernel ran",
        # NOT SM occupancy. See NvidiaSmiSampler docstring.
        rank0_smi = smi_all[0] if smi_all else None
        if rank0_smi is not None:
            act = rank0_smi["gpu_activity_pct"]
            pwr = rank0_smi["power_w"]
            clk = rank0_smi["sm_clock_mhz"]
            print(f"GPU activity rank0   : "
                  f"min={act['min']:.0f}% mean={act['mean']:.0f}% "
                  f"max={act['max']:.0f}% "
                  f"({rank0_smi['n_samples']} samples)", flush=True)
            if pwr["mean"] is not None:
                print(f"Power rank0 (W)      : "
                      f"mean={pwr['mean']:.0f} max={pwr['max']:.0f}",
                      flush=True)
            if clk["mean"] is not None:
                print(f"SM clock rank0 (MHz) : "
                      f"mean={clk['mean']:.0f} max={clk['max']:.0f}",
                      flush=True)
        print("=" * 64, flush=True)

        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[rank0] Wrote {args.output_json}", flush=True)


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_dist()

    hostname = socket.gethostname()
    print(f"[rank {rank}] host={hostname} local_rank={local_rank} "
          f"world_size={world_size} "
          f"device={torch.cuda.get_device_name(local_rank)}", flush=True)

    try:
        benchmark(args, rank, local_rank, world_size)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

