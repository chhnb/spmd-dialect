"""
Generate timeline visualization for F1 OSHER overhead comparison.
Shows kernel execution bars and gaps (overhead) for each strategy.

Usage: python benchmark/plot_timeline.py
Output: benchmark/results/timeline_comparison.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================================================
# Data from 3060 experiments (F1 OSHER, N=64, fp64)
# Source: 3060_hydro_f1_osher_adaptive_threads_v2.txt
# ============================================================

# Per-step timing (μs)
# Each step has 2 kernels: flux (compute) + transfer (update)
# We simulate ~10 steps of timeline

strategies = {
    "Taichi (Python DSL)": {
        "total_per_step": 149.2,  # measured Sync time for N=64
        "compute_per_kernel": 25.0,  # estimated from Graph result
        "n_kernels": 8,  # Taichi splits into ~8 cu kernels (from nsys)
        "sync_per_step": True,
        "color": "#e74c3c",  # red
        "label_extra": "149.2 μs/step",
    },
    "Warp (Python DSL)": {
        "total_per_step": 128.1,  # Warp measured
        "compute_per_kernel": 25.0,
        "n_kernels": 8,  # Warp also ~8 kernels (from nsys)
        "sync_per_step": True,
        "color": "#e67e22",  # orange
        "label_extra": "128.1 μs/step",
    },
    "CUDA Sync": {
        "total_per_step": 149.2,  # matches Taichi
        "compute_per_kernel": 25.0,
        "n_kernels": 2,  # 2 CUDA kernels: flux + transfer
        "sync_per_step": True,
        "color": "#f39c12",  # yellow-orange
        "label_extra": "149.2 μs/step",
    },
    "Kokkos (C++ Async)": {
        "total_per_step": 57.0,  # Kokkos measured
        "compute_per_kernel": 25.0,
        "n_kernels": 2,
        "sync_per_step": False,  # async, small gap only
        "color": "#3498db",  # blue
        "label_extra": "57.0 μs/step",
    },
    "CUDA Graph": {
        "total_per_step": 69.4,  # Graph measured for N=64
        "compute_per_kernel": 33.0,  # Graph kernel time (slightly higher due to replay overhead)
        "n_kernels": 2,
        "sync_per_step": False,
        "is_graph": True,
        "color": "#2ecc71",  # green
        "label_extra": "69.4 μs/step",
    },
    "CUDA Persistent": {
        "total_per_step": 56.1,  # Persistent measured for N=64
        "compute_per_kernel": 56.1,  # single continuous kernel
        "n_kernels": 1,  # ONE kernel, never exits
        "sync_per_step": False,
        "is_persistent": True,
        "color": "#9b59b6",  # purple
        "label_extra": "56.1 μs/step",
    },
}

N_STEPS = 6  # show 6 steps in timeline
fig, axes = plt.subplots(len(strategies), 1, figsize=(14, 8), sharex=True)
fig.suptitle("F1 Hydro-cal OSHER (N=64, fp64) — RTX 3060\nKernel Execution Timeline: 6 Strategies Compared",
             fontsize=13, fontweight='bold', y=0.98)

for idx, (name, cfg) in enumerate(strategies.items()):
    ax = axes[idx]
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(f"{name}\n({cfg['label_extra']})", fontsize=8, rotation=0,
                  ha='right', va='center', labelpad=100)

    t = 0  # current time position
    total_per_step = cfg["total_per_step"]
    n_k = cfg["n_kernels"]
    is_persistent = cfg.get("is_persistent", False)
    is_graph = cfg.get("is_graph", False)

    if is_persistent:
        # One continuous bar for all steps
        total_time = total_per_step * N_STEPS
        ax.barh(0.5, total_time, height=0.7, left=0,
                color=cfg["color"], edgecolor='black', linewidth=0.5)
        # Show grid_sync markers
        for s in range(1, N_STEPS):
            sync_x = total_per_step * s
            ax.axvline(x=sync_x, color='white', linewidth=1, linestyle='--', alpha=0.5)
        t = total_time
    else:
        for step in range(N_STEPS):
            if is_graph:
                # Graph: kernels back-to-back with tiny replay gap
                for k in range(n_k):
                    kernel_time = cfg["compute_per_kernel"] / n_k
                    ax.barh(0.5, kernel_time, height=0.7, left=t,
                            color=cfg["color"], edgecolor='black', linewidth=0.3)
                    t += kernel_time
                    t += 0.5  # tiny gap for graph replay
                # small gap between steps
                t += 2.0
            elif cfg["sync_per_step"]:
                # Sync mode: kernel, gap, kernel, gap, BIG sync gap
                compute_per_k = cfg["compute_per_kernel"] / n_k
                gap = (total_per_step - cfg["compute_per_kernel"]) / n_k

                for k in range(n_k):
                    # kernel execution
                    ax.barh(0.5, compute_per_k, height=0.7, left=t,
                            color=cfg["color"], edgecolor='black', linewidth=0.3)
                    t += compute_per_k
                    # overhead gap (launch + sync)
                    ax.barh(0.5, gap, height=0.7, left=t,
                            color='#ecf0f1', edgecolor='#bdc3c7', linewidth=0.3,
                            hatch='///', alpha=0.3)
                    t += gap
            else:
                # Async mode: kernels with small launch gaps
                compute_per_k = cfg["compute_per_kernel"] / n_k
                small_gap = (total_per_step - cfg["compute_per_kernel"]) / max(n_k, 1)

                for k in range(n_k):
                    ax.barh(0.5, compute_per_k, height=0.7, left=t,
                            color=cfg["color"], edgecolor='black', linewidth=0.3)
                    t += compute_per_k
                    t += small_gap  # small launch gap

    ax.set_xlim(0, strategies["Taichi (Python DSL)"]["total_per_step"] * N_STEPS * 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

axes[-1].set_xlabel("Time (μs)", fontsize=10)

# Legend
compute_patch = mpatches.Patch(color='#95a5a6', label='GPU Compute')
overhead_patch = mpatches.Patch(facecolor='#ecf0f1', edgecolor='#bdc3c7',
                                 hatch='///', label='Overhead (idle GPU)', alpha=0.3)
fig.legend(handles=[compute_patch, overhead_patch], loc='lower center',
           ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0.15, 0.04, 1, 0.95])
plt.savefig("benchmark/results/timeline_comparison.png", dpi=200, bbox_inches='tight')
plt.savefig("benchmark/results/timeline_comparison.pdf", bbox_inches='tight')
print("Saved: benchmark/results/timeline_comparison.png")
print("Saved: benchmark/results/timeline_comparison.pdf")
