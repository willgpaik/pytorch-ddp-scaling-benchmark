#!/usr/bin/env python3
"""
Aggregate result JSON files into a table. Computes scaling efficiency
where possible.

Usage:
    python analyze_results.py results/
"""

import argparse
import glob
import json
import os
from collections import defaultdict


def load_results(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            try:
                rows.append(json.load(f))
            except json.JSONDecodeError as e:
                print(f"Skipping {path}: {e}")
    return rows


def fmt(x, p=1):
    if x is None:
        return "-"
    return f"{x:.{p}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    args = ap.parse_args()

    rows = load_results(args.results_dir)
    if not rows:
        print("No results found.")
        return

    # Group by (scaling, model, global_batch for strong / per_gpu for weak)
    weak = [r for r in rows if r["scaling_mode"] == "weak"]
    strong = [r for r in rows if r["scaling_mode"] == "strong"]

    print("\n=== WEAK SCALING ===")
    print("Per-GPU batch fixed. Each row is a different scale.")
    print(f"{'World':>6} {'Nodes':>5} {'Per-GPU BS':>10} {'Global BS':>9} "
          f"{'Img/s/GPU':>10} {'Img/s global':>13} {'Step ms':>8} "
          f"{'Activity%':>9} {'Mem GB':>7} {'Straggler':>9}")
    for r in sorted(weak, key=lambda x: x["world_size"]):
        smi = r.get("gpu_sampling_rank0")
        act = (smi["gpu_activity_pct"]["mean"]
               if smi and smi.get("gpu_activity_pct") else None)
        mem = r.get("peak_mem_gb")
        mem_max = mem["max_gb"] if isinstance(mem, dict) else mem
        print(f"{r['world_size']:>6} {r['nodes']:>5} "
              f"{r['batch_size_per_gpu']:>10} {r['global_batch_size']:>9} "
              f"{fmt(r['images_per_sec_per_gpu']):>10} "
              f"{fmt(r['images_per_sec_global']):>13} "
              f"{fmt(r['mean_step_sec']*1000, 2):>8} "
              f"{fmt(act, 0):>9} "
              f"{fmt(mem_max, 1):>7} "
              f"{fmt(r['straggler_ratio'], 3):>9}")

    print("\n=== STRONG SCALING ===")
    # Group strong-scaling rows by global batch size for proper speedup calc
    strong_by_gbs = defaultdict(list)
    for r in strong:
        strong_by_gbs[r["global_batch_size"]].append(r)

    for gbs in sorted(strong_by_gbs):
        runs = sorted(strong_by_gbs[gbs], key=lambda x: x["world_size"])
        baseline = next((r for r in runs if r["world_size"] == 1), None)
        baseline_t = baseline["mean_step_sec"] if baseline else None

        print(f"\n  Global batch = {gbs}")
        print(f"  {'World':>6} {'Nodes':>5} {'Per-GPU BS':>10} "
              f"{'Step ms':>8} {'Img/s global':>13} {'Speedup':>8} "
              f"{'Efficiency':>10} {'Activity%':>9}")
        for r in runs:
            ws = r["world_size"]
            step_ms = r["mean_step_sec"] * 1000
            speedup = (baseline_t / r["mean_step_sec"]
                       if baseline_t else None)
            efficiency = speedup / ws if speedup else None
            smi = r.get("gpu_sampling_rank0")
            act = (smi["gpu_activity_pct"]["mean"]
                   if smi and smi.get("gpu_activity_pct") else None)
            print(f"  {ws:>6} {r['nodes']:>5} {r['batch_size_per_gpu']:>10} "
                  f"{fmt(step_ms, 2):>8} "
                  f"{fmt(r['images_per_sec_global']):>13} "
                  f"{fmt(speedup, 2):>8} "
                  f"{fmt(efficiency*100 if efficiency else None, 1):>9}% "
                  f"{fmt(act, 0):>9}")

    # Sanity warnings
    print("\n=== SANITY CHECKS ===")
    for r in rows:
        warnings = []
        if r["straggler_ratio"] > 1.10:
            warnings.append(
                f"straggler ratio {r['straggler_ratio']:.2f} "
                f"(rank {r['slowest_rank']} much slower than "
                f"rank {r['fastest_rank']})"
            )
        if r["bench_steps"] < 100:
            warnings.append(f"only {r['bench_steps']} steps measured")
        if r.get("stdev_step_sec", 0) > 0.1 * r["mean_step_sec"]:
            warnings.append(
                f"step time stdev {r['stdev_step_sec']*1000:.2f}ms is "
                f">10% of mean ({r['mean_step_sec']*1000:.2f}ms); "
                f"high variance"
            )
        # Memory spread across ranks. Should be tiny in healthy DDP.
        mem = r.get("peak_mem_gb")
        if isinstance(mem, dict) and mem.get("spread_pct", 0) > 5:
            warnings.append(
                f"memory spread across ranks: {mem['spread_pct']:.1f}% "
                f"(min {mem['min_gb']:.2f} / max {mem['max_gb']:.2f} GB); "
                f"unusual for DDP"
            )
        # GPU activity. Coarse metric (% time any kernel was running),
        # but very low values indicate the GPU is starved.
        smi = r.get("gpu_sampling_rank0")
        if smi and smi.get("gpu_activity_pct", {}).get("mean") is not None:
            act_mean = smi["gpu_activity_pct"]["mean"]
            if act_mean < 80:
                warnings.append(
                    f"low GPU activity on rank0: mean {act_mean:.0f}%; "
                    f"workload may not be saturating the device "
                    f"(consider larger batch or larger model)"
                )
        if warnings:
            print(f"  [{r['tag']}]")
            for w in warnings:
                print(f"    - {w}")


if __name__ == "__main__":
    main()
