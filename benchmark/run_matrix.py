#!/usr/bin/env python3
"""Unified N×M benchmark matrix runner.

Runs all CUDA 4-strategy benchmarks and collects results into a single CSV.

Usage:
    python run_matrix.py                    # run all cases, default sizes
    python run_matrix.py --cases C1 C8 C9   # run specific cases
    python run_matrix.py --dry-run           # show what would be run
"""
import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent

# Case definitions: (case_id, binary_name, default_args_small, default_args_large)
CASES = {
    "C1":  ("jacobi2d_bench",       [("256", "100", "10"), ("4096", "100", "10")]),
    "C2":  ("jacobi3d_bench",       [("64", "100", "10"), ("256", "100", "10")]),
    "C3":  ("overhead_solutions_a100", []),  # runs all sizes internally
    "C4":  ("wave2d_bench",         [("512", "100", "10"), ("4096", "100", "10")]),
    "C5":  ("lbm2d_bench",          [("512", "256", "100", "10"), ("2048", "1024", "100", "10")]),
    "C6":  ("nbody_bench",          [("4096", "10", "10"), ("32768", "10", "10")]),
    "C7":  ("sph_bench",            [("8192", "10", "10"), ("65536", "10", "10")]),
    "C8":  ("hydro_f1_a100",        [("10", "10"), ("10", "10", str(BENCHMARK_DIR / "F1_hydro_shallow_water" / "data_20w" / "binary/"))]),
    "C9":  ("hydro_osher_a100",     [("899", "10"), ("900", "10", str(BENCHMARK_DIR / "F2_hydro_refactored" / "data_20w" / "binary/"))]),
    "C10": ("overhead_solutions_a100", []),  # same binary as C3
    "C11": ("fdtd2d_bench",         [("512", "100", "10"), ("4096", "100", "10")]),
    "C12": ("maccormack3d_bench",   [("64", "100", "10"), ("128", "100", "10")]),
    "C13": ("lulesh_fusion_a100",   []),  # runs internally
    "C14": ("pic1d_bench",          [("4096", "256", "100", "10"), ("16384", "1024", "100", "10")]),
    "C15": ("cg_fusion_a100",       []),  # runs internally
    "C16": ("stable_fluids_bench",  [("256", "5", "10"), ("1024", "5", "10")]),
    "C17": ("conv3d_bench",         [("128", "1", "10"), ("256", "1", "10")]),
    "C18": ("doitgen_bench",        [("128", "1", "10"), ("256", "1", "10")]),
    "C19": ("lu_bench",             [("512", "10"), ("1024", "10")]),
    "C20": ("adi_bench",            [("256", "3", "10"), ("512", "3", "10")]),
    "C21": ("gramschmidt_bench",    [("256", "10"), ("512", "10")]),
}

CASE_NAMES = {
    "C1": "Jacobi2D", "C2": "Jacobi3D", "C3": "Heat2D", "C4": "Wave2D",
    "C5": "LBM_D2Q9", "C6": "Nbody", "C7": "SPH", "C8": "HydroF1",
    "C9": "HydroF2", "C10": "GrayScott", "C11": "FDTD2D", "C12": "MacCormack3D",
    "C13": "LULESH", "C14": "PIC1D", "C15": "CG_Solver", "C16": "StableFluids",
    "C17": "Conv3D", "C18": "DOITGEN", "C19": "LU", "C20": "ADI",
    "C21": "GramSchmidt",
}


def detect_gpu():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip().split("\n")[0]
    except Exception:
        return "unknown"


def parse_output(output, case_id):
    """Parse benchmark output to extract strategy timings."""
    results = []
    # Match patterns like: [Sync Loop] 100 steps: median=3.516 ms, 35.16 us/step
    # or: [1] Sync (4 kernels/step):  20.1 us/step
    patterns = [
        r'\[([^\]]+)\].*?(\d+\.?\d*)\s*us/step',
        r'\[([^\]]+)\].*?median=(\d+\.?\d*)\s*ms.*?(\d+\.?\d*)\s*us/step',
    ]
    for line in output.split("\n"):
        for pat in patterns:
            m = re.search(pat, line)
            if m:
                strategy = m.group(1).strip()
                if len(m.groups()) == 3:
                    us_per_step = float(m.group(3))
                else:
                    us_per_step = float(m.group(2))
                results.append((strategy, us_per_step))
                break
        # Also match N/A lines
        if "N/A" in line and "[" in line:
            m = re.match(r'\[([^\]]+)\]\s*N/A', line)
            if m:
                results.append((m.group(1).strip(), None))
    return results


def run_case(case_id, binary, args_list, gpu_name, dry_run=False):
    """Run a single case with all its size configurations."""
    binary_path = BENCHMARK_DIR / binary
    if not binary_path.exists():
        print(f"  SKIP {case_id}: binary not found: {binary_path}")
        return []

    all_results = []
    if not args_list:
        # Binary runs all sizes internally
        args_list = [()]

    for args in args_list:
        cmd = [str(binary_path)] + list(args)
        size_str = "x".join(args[:2]) if len(args) >= 2 else "default"

        if dry_run:
            print(f"  {case_id} [{size_str}]: {' '.join(cmd)}")
            continue

        print(f"  {case_id} [{size_str}]...", end="", flush=True)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
                env={**os.environ,
                     "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "")}
            )
            timings = parse_output(result.stdout, case_id)
            for strategy, us in timings:
                all_results.append({
                    "case": CASE_NAMES.get(case_id, case_id),
                    "strategy": strategy,
                    "gpu": gpu_name,
                    "problem_size": size_str,
                    "median_us": f"{us:.2f}" if us is not None else "N/A",
                })
            print(f" {len(timings)} strategies")
        except subprocess.TimeoutExpired:
            print(" TIMEOUT")
        except Exception as e:
            print(f" ERROR: {e}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run N×M benchmark matrix")
    parser.add_argument("--cases", nargs="+", default=None,
                        help=f"Cases to run (default: all). Options: {list(CASES.keys())}")
    parser.add_argument("--output", default="matrix_results.csv",
                        help="Output CSV file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show commands without running")
    args = parser.parse_args()

    gpu_name = detect_gpu()
    print(f"GPU: {gpu_name}")
    print(f"Output: {args.output}")

    case_ids = args.cases or sorted(CASES.keys(), key=lambda x: int(x[1:]))
    all_results = []

    for case_id in case_ids:
        if case_id not in CASES:
            print(f"Unknown case: {case_id}")
            continue
        binary, args_list = CASES[case_id]
        print(f"\n=== {case_id}: {CASE_NAMES.get(case_id, case_id)} ===")
        results = run_case(case_id, binary, args_list, gpu_name, args.dry_run)
        all_results.extend(results)

    if not args.dry_run and all_results:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case", "strategy", "gpu", "problem_size", "median_us"])
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {args.output} ({len(all_results)} rows)")


if __name__ == "__main__":
    main()
