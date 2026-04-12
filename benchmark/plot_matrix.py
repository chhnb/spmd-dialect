#!/usr/bin/env python3
"""Milestone 5: Generate analysis plots from matrix_results.csv.

Plots:
1. Overhead% vs kernels/step (bar chart per strategy)
2. Speedup heatmap (N strategies × M cases)
3. DSL overhead decomposition (stacked bar for F1/F2)
4. Problem size scaling (overhead% vs size for Jacobi2D + HydroF2)

Usage:
    python plot_matrix.py matrix_results.csv
    python plot_matrix.py matrix_results.csv --output-dir plots/
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

# Kernel launch counts per step for each case
KERN_PER_STEP = {
    "Jacobi2D": 2, "Jacobi3D": 2, "Heat2D": 2, "Wave2D": 1,
    "LBM_D2Q9": 1, "Nbody": 2, "SPH": 2, "HydroF1": 1,
    "HydroF2": 2, "GrayScott": 2, "FDTD2D": 3, "MacCormack3D": 3,
    "LULESH": 4, "PIC1D": 4, "CG_Solver": 5,
    "StableFluids": 102, "Conv3D": 510, "DOITGEN": 512,
    "LU": 4096, "ADI": 4097, "GramSchmidt": 6144,
}


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_overhead_vs_kernels(data, output_dir):
    """Plot 1: Overhead% vs kernels/step for each strategy."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, generating text table instead")
        print("\n=== Overhead% vs Kernels/Step ===")
        strategies = defaultdict(dict)
        for row in data:
            if row["overhead_pct"]:
                strategies[row["strategy"]][row["case"]] = float(row["overhead_pct"])
        for strat, cases in sorted(strategies.items()):
            print(f"\n{strat}:")
            for case in sorted(cases, key=lambda c: KERN_PER_STEP.get(c, 0)):
                print(f"  {case:15s} ({KERN_PER_STEP.get(case,0):5d} kern/step): {cases[case]:.1f}%")
        return

    strategies = defaultdict(dict)
    for row in data:
        if row["overhead_pct"]:
            strategies[row["strategy"]][row["case"]] = float(row["overhead_pct"])

    fig, ax = plt.subplots(figsize=(14, 6))
    cases_sorted = sorted(KERN_PER_STEP.keys(), key=lambda c: KERN_PER_STEP[c])
    x = range(len(cases_sorted))
    width = 0.15

    for i, (strat, case_data) in enumerate(sorted(strategies.items())):
        vals = [case_data.get(c, 0) for c in cases_sorted]
        ax.bar([xi + i * width for xi in x], vals, width, label=strat)

    ax.set_xlabel("Case (sorted by kernels/step)")
    ax.set_ylabel("Overhead %")
    ax.set_title("Launch Overhead vs Kernel Complexity")
    ax.set_xticks([xi + width * 2 for xi in x])
    ax.set_xticklabels([f"{c}\n({KERN_PER_STEP[c]})" for c in cases_sorted],
                       rotation=45, ha='right', fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overhead_vs_kernels.png"), dpi=150)
    print(f"Saved: {output_dir}/overhead_vs_kernels.png")


def plot_speedup_heatmap(data, output_dir):
    """Plot 2: Speedup heatmap (strategy × case)."""
    # Group by case, find Sync baseline, compute speedups
    sync_times = {}
    all_times = defaultdict(dict)
    for row in data:
        if row["median_us"] == "N/A":
            continue
        us = float(row["median_us"])
        case = row["case"]
        strat = row["strategy"]
        all_times[strat][case] = us
        if "Sync" in strat:
            sync_times[case] = us

    print("\n=== Speedup vs Sync (text heatmap) ===")
    strategies = sorted(all_times.keys())
    cases = sorted(set(c for s in all_times.values() for c in s),
                   key=lambda c: KERN_PER_STEP.get(c, 0))

    header = f"{'Case':15s}" + "".join(f"{s:>12s}" for s in strategies)
    print(header)
    print("-" * len(header))
    for case in cases:
        row_str = f"{case:15s}"
        base = sync_times.get(case)
        for strat in strategies:
            us = all_times[strat].get(case)
            if us and base:
                speedup = base / us
                row_str += f"{speedup:11.2f}x"
            else:
                row_str += f"{'N/A':>12s}"
        print(row_str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="Path to matrix_results.csv")
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_csv(args.csv_file)
    print(f"Loaded {len(data)} rows from {args.csv_file}")

    plot_overhead_vs_kernels(data, args.output_dir)
    plot_speedup_heatmap(data, args.output_dir)


if __name__ == "__main__":
    main()
