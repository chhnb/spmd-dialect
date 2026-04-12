#!/usr/bin/env python3
"""Milestone 5: Generate analysis plots from matrix_results.csv.

4 plot types:
1. Overhead% vs kernels/step (grouped by strategy)
2. Speedup heatmap (N strategies × M cases)
3. DSL overhead decomposition (stacked bar for cases with full coverage)
4. Problem size scaling (overhead% vs size for select cases)

Usage:
    python plot_matrix.py matrix_results.csv
    python plot_matrix.py matrix_results.csv --output-dir plots/
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

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


def plot1_overhead_vs_kernels(data, output_dir):
    """Plot 1: Overhead% vs kernels/step, one line per strategy."""
    # Group: (case, size, strategy) -> overhead_pct
    grouped = defaultdict(dict)
    for row in data:
        if row.get("overhead_pct") and row["overhead_pct"] != "":
            key = (row["case"], row["problem_size"])
            grouped[key][row["strategy"]] = float(row["overhead_pct"])

    print("\n=== Plot 1: Overhead% vs Kernels/Step ===")
    # Use smallest size per case for the overview
    case_smallest = {}
    for (case, size), strats in grouped.items():
        if case not in case_smallest or size < case_smallest[case][0]:
            case_smallest[case] = (size, strats)

    cases_sorted = sorted(case_smallest.keys(), key=lambda c: KERN_PER_STEP.get(c, 0))
    all_strats = sorted(set(s for _, strats in case_smallest.values() for s in strats))

    header = f"{'Case':15s} {'K/S':>5s}" + "".join(f"{s:>15s}" for s in all_strats)
    print(header)
    print("-" * len(header))
    for case in cases_sorted:
        size, strats = case_smallest[case]
        row = f"{case:15s} {KERN_PER_STEP.get(case,0):5d}"
        for s in all_strats:
            if s in strats:
                row += f"{strats[s]:14.1f}%"
            else:
                row += f"{'—':>15s}"
        print(row)


def plot2_speedup_heatmap(data, output_dir):
    """Plot 2: Speedup vs CUDA Sync for each strategy×case (smallest size)."""
    # Collect: (case, size, strategy) -> median_us
    times = defaultdict(lambda: defaultdict(dict))
    for row in data:
        if row["median_us"] != "N/A" and row["median_us"]:
            times[row["case"]][row["problem_size"]][row["strategy"]] = float(row["median_us"])

    print("\n=== Plot 2: Speedup Heatmap (vs CUDA_Sync, smallest size) ===")
    all_strats = sorted(set(s for c in times.values() for sz in c.values() for s in sz))
    cases_sorted = sorted(times.keys(), key=lambda c: KERN_PER_STEP.get(c, 0))

    header = f"{'Case':15s}" + "".join(f"{s:>15s}" for s in all_strats)
    print(header)
    print("-" * len(header))
    for case in cases_sorted:
        sizes = times[case]
        smallest_size = min(sizes.keys())
        strat_times = sizes[smallest_size]
        sync_us = None
        for k in ("CUDA_Sync Loop", "CUDA_Sync", "CUDA_Sync (4 kernels/step)",
                   "CUDA_Sync (5 kern + host readback)"):
            if k in strat_times:
                sync_us = strat_times[k]
                break
        row_str = f"{case:15s}"
        for s in all_strats:
            if s in strat_times and sync_us:
                speedup = sync_us / strat_times[s]
                row_str += f"{speedup:14.2f}x"
            else:
                row_str += f"{'—':>15s}"
        print(row_str)


def plot3_dsl_decomposition(data, output_dir):
    """Plot 3: DSL overhead decomposition for cases with CUDA + Taichi data."""
    print("\n=== Plot 3: DSL Overhead Decomposition ===")
    print(f"{'Case':15s} {'Size':>8s} {'CUDA_cmp':>10s} {'CUDA_Sync':>10s} {'Taichi':>10s} {'DSL_OH':>10s}")
    print("-" * 65)

    by_case = defaultdict(lambda: defaultdict(dict))
    for row in data:
        if row["median_us"] and row["median_us"] != "N/A":
            by_case[row["case"]][row["problem_size"]][row["strategy"]] = float(row["median_us"])

    for case in sorted(by_case.keys(), key=lambda c: KERN_PER_STEP.get(c, 0)):
        for size in sorted(by_case[case].keys()):
            s = by_case[case][size]
            cuda_sync = None
            for k in s:
                if "Sync" in k and "CUDA" in k:
                    cuda_sync = s[k]
                    break
            taichi = s.get("Taichi")
            if cuda_sync and taichi:
                # Estimate GPU compute from fastest CUDA strategy
                fastest_cuda = min(v for k, v in s.items() if "CUDA" in k and v > 0)
                dsl_overhead = taichi - fastest_cuda
                print(f"{case:15s} {size:>8s} {fastest_cuda:10.1f} {cuda_sync:10.1f} {taichi:10.1f} {dsl_overhead:10.1f}")


def plot4_size_scaling(data, output_dir):
    """Plot 4: How overhead% changes with problem size for select cases."""
    print("\n=== Plot 4: Size Scaling (overhead% vs problem size) ===")

    by_case_size = defaultdict(lambda: defaultdict(dict))
    for row in data:
        if row.get("overhead_pct") and row["overhead_pct"] != "":
            by_case_size[row["case"]][row["problem_size"]][row["strategy"]] = float(row["overhead_pct"])

    for case in sorted(by_case_size.keys(), key=lambda c: KERN_PER_STEP.get(c, 0)):
        sizes = by_case_size[case]
        if len(sizes) < 2:
            continue
        print(f"\n  {case}:")
        all_strats = sorted(set(s for sz in sizes.values() for s in sz))
        header = f"    {'Size':>10s}" + "".join(f"{s:>15s}" for s in all_strats)
        print(header)
        for size in sorted(sizes.keys()):
            row = f"    {size:>10s}"
            for s in all_strats:
                if s in sizes[size]:
                    row += f"{sizes[size][s]:14.1f}%"
                else:
                    row += f"{'—':>15s}"
            print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="Path to matrix_results.csv")
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_csv(args.csv_file)
    print(f"Loaded {len(data)} rows from {args.csv_file}")

    plot1_overhead_vs_kernels(data, args.output_dir)
    plot2_speedup_heatmap(data, args.output_dir)
    plot3_dsl_decomposition(data, args.output_dir)
    plot4_size_scaling(data, args.output_dir)


if __name__ == "__main__":
    main()
