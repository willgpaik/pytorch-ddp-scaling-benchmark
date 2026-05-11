#!/usr/bin/env python3
"""
Visualize benchmark results from results/*.json.

Produces four plots:
  1. Throughput vs world size (weak scaling)
  2. Throughput vs world size (strong scaling, one line per global batch)
  3. Speedup vs world size (strong scaling, with ideal-linear reference)
  4. Parallel efficiency vs world size (strong scaling)

Optional 5th plot: GPU activity % per configuration (if smi data is present).

Usage:
    pip install matplotlib    # one-time, only needed for this script
    python visualize_results.py results/
    python visualize_results.py results/ --output-dir plots/
    python visualize_results.py results/ --device-label "B200"

The --device-label is just for plot titles; nothing functional.
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict


def load_results(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(path) as f:
                rows.append(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Skipping {path}: {e}", file=sys.stderr)
    return rows


def get_activity(r):
    """Return mean GPU activity % from rank 0 sampling, or None."""
    smi = r.get("gpu_sampling_rank0")
    if not smi:
        return None
    act = smi.get("gpu_activity_pct", {})
    return act.get("mean")


def get_max_mem(r):
    """Return max peak memory across ranks, in GB."""
    mem = r.get("peak_mem_gb")
    if isinstance(mem, dict):
        return mem.get("max_gb")
    return mem  # backward compat with old format


def plot_weak_throughput(weak_rows, ax, device_label):
    """Weak scaling: throughput per GPU should stay flat. Total grows linearly."""
    if not weak_rows:
        ax.text(0.5, 0.5, "No weak-scaling data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Weak scaling: throughput")
        return

    # Group by model so different models become different lines.
    by_model = defaultdict(list)
    for r in weak_rows:
        by_model[r["model"]].append(r)

    for model, runs in by_model.items():
        runs = sorted(runs, key=lambda x: x["world_size"])
        ws = [r["world_size"] for r in runs]
        per_gpu = [r["images_per_sec_per_gpu"] for r in runs]
        global_tput = [r["images_per_sec_global"] for r in runs]
        ax.plot(ws, per_gpu, "o-",
                label=f"{model} per-GPU", linewidth=2, markersize=8)
        ax.plot(ws, global_tput, "s--",
                label=f"{model} global", linewidth=2, markersize=8, alpha=0.7)

    ax.set_xlabel("World size (number of GPUs)")
    ax.set_ylabel("Throughput (images/sec)")
    title = "Weak scaling: per-GPU batch fixed"
    if device_label:
        title += f" — {device_label}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")


def plot_strong_throughput(strong_rows, ax, device_label):
    """Strong scaling: throughput should grow roughly linearly with world size."""
    if not strong_rows:
        ax.text(0.5, 0.5, "No strong-scaling data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Strong scaling: throughput")
        return

    by_gbs = defaultdict(list)
    for r in strong_rows:
        by_gbs[r["global_batch_size"]].append(r)

    for gbs in sorted(by_gbs):
        runs = sorted(by_gbs[gbs], key=lambda x: x["world_size"])
        ws = [r["world_size"] for r in runs]
        global_tput = [r["images_per_sec_global"] for r in runs]
        ax.plot(ws, global_tput, "o-",
                label=f"global batch={gbs}", linewidth=2, markersize=8)

    ax.set_xlabel("World size (number of GPUs)")
    ax.set_ylabel("Global throughput (images/sec)")
    title = "Strong scaling: global batch fixed"
    if device_label:
        title += f" — {device_label}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")


def plot_speedup(strong_rows, ax, device_label):
    """Strong scaling speedup with ideal-linear reference."""
    if not strong_rows:
        ax.text(0.5, 0.5, "No strong-scaling data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Speedup")
        return

    by_gbs = defaultdict(list)
    for r in strong_rows:
        by_gbs[r["global_batch_size"]].append(r)

    max_ws = 1
    for gbs in sorted(by_gbs):
        runs = sorted(by_gbs[gbs], key=lambda x: x["world_size"])
        baseline = next((r for r in runs if r["world_size"] == 1), None)
        if baseline is None:
            print(f"Warning: no 1-GPU baseline for global batch {gbs}; "
                  f"skipping speedup plot for this batch size",
                  file=sys.stderr)
            continue
        baseline_t = baseline["mean_step_sec"]
        ws = [r["world_size"] for r in runs]
        speedup = [baseline_t / r["mean_step_sec"] for r in runs]
        ax.plot(ws, speedup, "o-",
                label=f"global batch={gbs}", linewidth=2, markersize=8)
        max_ws = max(max_ws, max(ws))

    # Ideal linear speedup reference line
    ideal_x = [2 ** i for i in range(0, max_ws.bit_length())]
    ideal_x = [x for x in ideal_x if x <= max_ws]
    if ideal_x[-1] != max_ws:
        ideal_x.append(max_ws)
    ax.plot(ideal_x, ideal_x, "k--",
            label="ideal linear", linewidth=1.5, alpha=0.5)

    ax.set_xlabel("World size (number of GPUs)")
    ax.set_ylabel("Speedup vs 1 GPU")
    title = "Strong scaling speedup"
    if device_label:
        title += f" — {device_label}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)


def plot_efficiency(strong_rows, ax, device_label):
    """Parallel efficiency = speedup / world_size. 1.0 = ideal."""
    if not strong_rows:
        ax.text(0.5, 0.5, "No strong-scaling data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Efficiency")
        return

    by_gbs = defaultdict(list)
    for r in strong_rows:
        by_gbs[r["global_batch_size"]].append(r)

    for gbs in sorted(by_gbs):
        runs = sorted(by_gbs[gbs], key=lambda x: x["world_size"])
        baseline = next((r for r in runs if r["world_size"] == 1), None)
        if baseline is None:
            continue
        baseline_t = baseline["mean_step_sec"]
        ws = [r["world_size"] for r in runs]
        efficiency = [(baseline_t / r["mean_step_sec"]) / r["world_size"]
                      for r in runs]
        ax.plot(ws, efficiency, "o-",
                label=f"global batch={gbs}", linewidth=2, markersize=8)

    ax.axhline(y=1.0, color="k", linestyle="--", linewidth=1, alpha=0.5,
               label="ideal (100%)")
    ax.axhline(y=0.8, color="orange", linestyle=":", linewidth=1, alpha=0.5,
               label="80% (acceptable)")
    ax.set_xlabel("World size (number of GPUs)")
    ax.set_ylabel("Parallel efficiency (speedup / N)")
    title = "Parallel efficiency"
    if device_label:
        title += f" — {device_label}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.15)


def plot_gpu_activity(rows, ax, device_label):
    """Bar chart of mean GPU activity % per configuration."""
    runs_with_activity = [r for r in rows if get_activity(r) is not None]
    if not runs_with_activity:
        ax.text(0.5, 0.5, "No GPU sampling data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("GPU activity")
        return

    runs_with_activity.sort(key=lambda r: (r["scaling_mode"],
                                           r["world_size"]))
    labels = [
        f"{r['scaling_mode']}\nW={r['world_size']}\n"
        f"bs={r['batch_size_per_gpu']}"
        for r in runs_with_activity
    ]
    activities = [get_activity(r) for r in runs_with_activity]

    bars = ax.bar(range(len(labels)), activities,
                  color=["steelblue" if r["scaling_mode"] == "weak"
                         else "coral" for r in runs_with_activity])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean GPU activity (%)")
    ax.axhline(y=80, color="orange", linestyle=":", linewidth=1, alpha=0.5,
               label="80% threshold")
    title = "GPU activity (rank 0)"
    if device_label:
        title += f" — {device_label}"
    ax.set_title(title)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    # Annotate bars with values
    for bar, val in zip(bars, activities):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", help="Directory containing *.json results")
    ap.add_argument("--output-dir", default="plots",
                    help="Where to save PNG files (default: plots/)")
    ap.add_argument("--device-label", default="",
                    help="Optional label for plot titles, e.g. 'B200'")
    ap.add_argument("--show", action="store_true",
                    help="Show interactive plots (requires display)")
    args = ap.parse_args()

    try:
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")  # headless backend for cluster nodes
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not installed. Install with:",
              file=sys.stderr)
        print("    pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    rows = load_results(args.results_dir)
    if not rows:
        print(f"No results found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    weak = [r for r in rows if r["scaling_mode"] == "weak"]
    strong = [r for r in rows if r["scaling_mode"] == "strong"]

    print(f"Loaded {len(rows)} results: "
          f"{len(weak)} weak, {len(strong)} strong")

    os.makedirs(args.output_dir, exist_ok=True)

    # 2x2 grid of the main four plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_weak_throughput(weak, axes[0, 0], args.device_label)
    plot_strong_throughput(strong, axes[0, 1], args.device_label)
    plot_speedup(strong, axes[1, 0], args.device_label)
    plot_efficiency(strong, axes[1, 1], args.device_label)
    fig.tight_layout()
    overview_path = os.path.join(args.output_dir, "scaling_overview.png")
    fig.savefig(overview_path, dpi=120)
    print(f"Wrote {overview_path}")

    # Each plot also as a standalone PNG for slide decks
    for name, plotter, data in [
        ("weak_throughput", plot_weak_throughput, weak),
        ("strong_throughput", plot_strong_throughput, strong),
        ("speedup", plot_speedup, strong),
        ("efficiency", plot_efficiency, strong),
    ]:
        fig_single, ax = plt.subplots(figsize=(8, 6))
        plotter(data, ax, args.device_label)
        fig_single.tight_layout()
        path = os.path.join(args.output_dir, f"{name}.png")
        fig_single.savefig(path, dpi=120)
        plt.close(fig_single)
        print(f"Wrote {path}")

    # GPU activity bar chart (separate because it has different layout needs)
    fig_act, ax_act = plt.subplots(figsize=(12, 6))
    plot_gpu_activity(rows, ax_act, args.device_label)
    fig_act.tight_layout()
    activity_path = os.path.join(args.output_dir, "gpu_activity.png")
    fig_act.savefig(activity_path, dpi=120)
    plt.close(fig_act)
    print(f"Wrote {activity_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
