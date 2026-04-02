#!/usr/bin/env python3
"""Run all simulation benchmarks and collect results.

Usage:
    python run_all.py                    # run everything
    python run_all.py --benchmarks A1 B1 E1  # run specific benchmarks
    python run_all.py --quick            # small problem sizes, fewer repeats
"""
import argparse
import os
import subprocess
import sys

PYTHON = os.path.join(os.path.dirname(__file__), "..", "..", "spmd-venv", "bin", "python")
if not os.path.exists(PYTHON):
    PYTHON = sys.executable

BENCHMARKS = {
    "A1": ("A1_jacobi_2d",          "Jacobi 2D 5-point stencil"),
    "A2": ("A2_lbm_d2q9",           "LBM D2Q9 (Lattice Boltzmann)"),
    "A3": ("A3_wave_equation",       "2D Wave Equation"),
    "B1": ("B1_nbody",              "N-body (direct all-pairs)"),
    "B2": ("B2_sph",                "SPH density computation"),
    "C1": ("C1_mpm",                "MPM (Material Point Method)"),
    "D2": ("D2_stable_fluids",      "Stable Fluids (incompressible NS)"),
    "E1": ("E1_reduction",          "Global Reduction (sum)"),
    "F1": ("F1_hydro_shallow_water", "2D Shallow Water (Osher, hydro-cal)"),
}

# D1 Cloth uses Warp's own benchmark harness
# A4 Euler, B3 DEM, C2 PIC: STATUS.md only (not yet implemented)


def run_benchmark(key, bench_dir, desc, extra_args=None):
    run_script = os.path.join(bench_dir, "run.py")
    if not os.path.exists(run_script):
        print(f"  [{key}] {desc}: SKIP (no run.py)")
        return False

    print(f"\n{'='*60}")
    print(f"  [{key}] {desc}")
    print(f"{'='*60}")

    cmd = [PYTHON, run_script]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(["--output", os.path.join(bench_dir, "results.csv")])

    result = subprocess.run(cmd, cwd=bench_dir)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all simulation benchmarks")
    parser.add_argument("--benchmarks", nargs="+", default=None,
                        help=f"Which benchmarks to run. Available: {list(BENCHMARKS.keys())}")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: small sizes, fewer repeats")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    keys = args.benchmarks or list(BENCHMARKS.keys())

    extra = []
    if args.quick:
        extra = ["--warmup", "1", "--repeat", "3"]

    passed = 0
    failed = 0
    skipped = 0

    for key in keys:
        if key not in BENCHMARKS:
            print(f"Unknown benchmark: {key}")
            continue
        dirname, desc = BENCHMARKS[key]
        bench_dir = os.path.join(base, dirname)
        if not os.path.isdir(bench_dir):
            print(f"  [{key}] {desc}: SKIP (dir not found)")
            skipped += 1
            continue
        ok = run_benchmark(key, bench_dir, desc, extra)
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"Done: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"Results in each benchmark's results.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
