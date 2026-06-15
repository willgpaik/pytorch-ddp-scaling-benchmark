#!/usr/bin/env python3
"""
Visualize benchmark results from results/*.json.

Produces four plots:
  1. Weak scaling: per-GPU throughput by (model, precision)
  2. Strong scaling: global throughput by (model, precision, GBS)
  3. Strong scaling speedup with ideal-linear reference
  4. Parallel efficiency

Usage:
    python visualize_results.py results/
    python visualize_results.py results/ --output-dir plots/
    python visualize_results.py results/ --device-label "A100 80GB"
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
    smi = r.get("gpu_sampling_rank0")
    if not smi:
        return None
    return smi.get("gpu_activity_pct", {}).get("mean")


def get_max_mem(r):
    mem = r.get("peak_mem_gb")
    if isinstance(mem, dict):
        return mem.get("max_gb")
    return mem


def make_label(model, prec):
    return f"{model} ({prec})"


def group_weak(weak_rows):
    """Group weak scaling rows by (model, precision)."""
    groups = defaultdict(list)
    for r in weak_rows:
        key = (r["model"], r.get("precision", "fp16"))
        groups[key].append(r)
    return {k: sorted(v, key=lambda x: x["world_size"])
            for k, v in groups.items()}


def group_strong(strong_rows):
    """Group strong scaling rows by (model, precision, global_batch_size)."""
    groups = defaultdict(list)
    for r in strong_rows:
        key = (r["model"], r.get("precision", "fp16"), r["global_batch_size"])
        groups[key].append(r)
    return {k: sorted(v, key=lambda x: x["world_size"])
            for k, v in groups.items()}


def color_cycle(ax, n):
    import matplotlib.pyplot as plt
    return [ax._get_lines.get_next_color() for _ in range(n)]


def plot_weak_throughput(weak_rows, ax, device_label):
    """Per-GPU throughput should stay flat -- drop shows DDP overhead."""
    if not weak_rows:
        ax.text(0.5, 0.5, "No weak-scaling data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Weak scaling: per-GPU throughput")
        return

    groups = group_weak(weak_rows)

    for (model, prec), runs in sorted(groups.items()):
        ws = [r["world_size"] for r in runs]
        per_gpu = [r["images_per_sec_per_gpu"] for r in runs]
        label = make_label(model, prec)
        ax.plot(ws, per_gpu, "o-", label=label, linewidth=2, markersize=8)

    ax.set_xlabel("World size (number of GPUs)")
    ax.set_ylabel("Throughput (images/sec/GPU)")
    title = "Weak scaling: per-GPU batch fixed"
    if device_label:
        title += f" — {device_label}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)


def plot_weak_efficiency(weak_rows, ax, device_label):
    """Efficiency = throughput_N_per_gpu / throughput_1_per_gpu (ideal = 100%)."""
    if not weak_rows:
        ax.text(0.5, 0.5, "No weak-scaling data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Weak scaling: efficiency")
        return

    groups = group_weak(weak_rows)

    for (model, prec), runs in sorted(groups.items()):
        baseline = next((r for r in runs if r["world_size"] == 1), None)
        if baseline is None:
            continue
        base_tput = baseline["images_per_sec_per_gpu"]
        ws = [r["world_size"] for r in runs]
        eff = [r["images_per_sec_per_gpu"] / base_tput * 100 for r in runs]
        label = make_label(model, prec)
        ax.plot(ws, eff, "o-", label=label, linewidth=2, markersize=8)

    ax.axhline(y=100, color="k", linestyle="--", linewidth=1, alpha=0.5,
               label="ideal (100%)")
    ax.axhline(y=90, color="orange", linestyle=":", linewidth=1, alpha=0.5,
               label="90% threshold")
    ax.set_xlabel("World size (number of GPUs)")
    ax.set_ylabel("Scaling efficiency (%)")
    title = "Weak scaling: efficiency"
    if device_label:
        title += f" — {device_label}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_ylim(80, 105)


def plot_strong_throughput(strong_rows, ax, device_label):
    if not strong_rows:
        ax.text(0.5, 0.5, "No strong-scaling data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Strong scaling: throughput")
        return

    groups = group_strong(strong_rows)

    for (model, prec, gbs), runs in sorted(groups.items()):
        ws = [r["world_size"] for r in runs]
        global_tput = [r["images_per_sec_global"] for r in runs]
        label = f"{make_label(model, prec)} GBS={gbs}"
        ax.plot(ws, global_tput, "o-", label=label, linewidth=2, markersize=8)

    ax.set_xlabel("World size (number of GPUs)")
    ax.set_ylabel("Global throughput (images/sec)")
    title = "Strong scaling: global batch fixed"
    if device_label:
        title += f" — {device_label}"
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")


def plot_speedup(strong_rows, ax, device_label):
    if not strong_rows:
        ax.text(0.5, 0.5, "No strong-scaling data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Speedup")
        return

    groups = group_strong(strong_rows)
    max_ws = 1

    for (model, prec, gbs), runs in sorted(groups.items()):
        # baseline must be 1-GPU run for this exact (model, precision, GBS)
        baseline = next((r for r in runs if r["world_size"] == 1), None)
        if baseline is None:
            print(f"Warning: no 1-GPU baseline for {model} {prec} GBS={gbs}",
                  file=sys.stderr)
            continue
        baseline_t = baseline["mean_step_sec"]
        ws = [r["world_size"] for r in runs]
        speedup = [baseline_t / r["mean_step_sec"] for r in runs]
        label = f"{make_label(model, prec)} GBS={gbs}"
        ax.plot(ws, speedup, "o-", label=label, linewidth=2, markersize=8)
        max_ws = max(max_ws, max(ws))

    # Ideal linear reference
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
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)


def plot_efficiency(strong_rows, ax, device_label):
    if not strong_rows:
        ax.text(0.5, 0.5, "No strong-scaling data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Efficiency")
        return

    groups = group_strong(strong_rows)

    for (model, prec, gbs), runs in sorted(groups.items()):
        baseline = next((r for r in runs if r["world_size"] == 1), None)
        if baseline is None:
            continue
        baseline_t = baseline["mean_step_sec"]
        ws = [r["world_size"] for r in runs]
        efficiency = [(baseline_t / r["mean_step_sec"]) / r["world_size"]
                      for r in runs]
        label = f"{make_label(model, prec)} GBS={gbs}"
        ax.plot(ws, efficiency, "o-", label=label, linewidth=2, markersize=8)

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
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.15)


def plot_gpu_activity(rows, ax, device_label):
    runs_with_activity = [r for r in rows if get_activity(r) is not None]
    if not runs_with_activity:
        ax.text(0.5, 0.5, "No GPU sampling data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("GPU activity")
        return

    runs_with_activity.sort(key=lambda r: (r["scaling_mode"],
                                           r.get("model", ""),
                                           r.get("precision", ""),
                                           r["world_size"]))
    labels = [
        f"{r['scaling_mode']}\n{r.get('model','?')}\n"
        f"{r.get('precision','?')} W={r['world_size']}"
        for r in runs_with_activity
    ]
    activities = [get_activity(r) for r in runs_with_activity]

    bars = ax.bar(range(len(labels)), activities,
                  color=["steelblue" if r["scaling_mode"] == "weak"
                         else "coral" for r in runs_with_activity])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
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
    for bar, val in zip(bars, activities):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", help="Directory containing *.json results")
    ap.add_argument("--output-dir", default="plots")
    ap.add_argument("--device-label", default="")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    try:
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    rows = load_results(args.results_dir)
    if not rows:
        print(f"No results found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    weak = [r for r in rows if r["scaling_mode"] == "weak"]
    strong = [r for r in rows if r["scaling_mode"] == "strong"]
    print(f"Loaded {len(rows)} results: {len(weak)} weak, {len(strong)} strong")

    os.makedirs(args.output_dir, exist_ok=True)

    # 2x2: weak throughput, weak efficiency, strong speedup, strong efficiency
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_weak_throughput(weak,   axes[0, 0], args.device_label)
    plot_weak_efficiency(weak,   axes[0, 1], args.device_label)
    plot_speedup(strong,         axes[1, 0], args.device_label)
    plot_efficiency(strong,      axes[1, 1], args.device_label)
    fig.tight_layout()
    overview_path = os.path.join(args.output_dir, "scaling_overview.png")
    fig.savefig(overview_path, dpi=120)
    print(f"Wrote {overview_path}")
    plt.close(fig)

    # Standalone PNGs
    for name, plotter, data in [
        ("weak_throughput",  plot_weak_throughput,  weak),
        ("weak_efficiency",  plot_weak_efficiency,  weak),
        ("strong_throughput", plot_strong_throughput, strong),
        ("speedup",          plot_speedup,           strong),
        ("efficiency",       plot_efficiency,        strong),
    ]:
        fig_s, ax = plt.subplots(figsize=(8, 6))
        plotter(data, ax, args.device_label)
        fig_s.tight_layout()
        path = os.path.join(args.output_dir, f"{name}.png")
        fig_s.savefig(path, dpi=120)
        plt.close(fig_s)
        print(f"Wrote {path}")

    fig_act, ax_act = plt.subplots(figsize=(14, 6))
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
