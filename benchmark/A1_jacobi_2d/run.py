#!/usr/bin/env python3
"""Benchmark runner for Jacobi 2D 5-point stencil.

Usage:
    python run.py                           # run all available frameworks
    python run.py --frameworks taichi warp  # run specific frameworks
    python run.py --sizes 1024 2048 4096    # custom problem sizes
    python run.py --steps 10                # Jacobi iterations per timed call
"""

import argparse
import importlib
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.harness import BenchmarkResult, run_kernel_benchmark, save_results, print_results


FRAMEWORKS = {
    "numpy":   ("jacobi_numpy",   "cpu",  "NumPy"),
    "taichi_cpu":  ("jacobi_taichi",  "cpu",  "Taichi"),
    "taichi_cuda": ("jacobi_taichi",  "cuda", "Taichi"),
    "warp_cpu":    ("jacobi_warp",    "cpu",  "Warp"),
    "warp_cuda":   ("jacobi_warp",    "cuda", "Warp"),
    "triton_cuda": ("jacobi_triton",  "cuda", "Triton"),
}

DEFAULT_SIZES = [1024, 4096, 8192]
DEFAULT_STEPS = 100  # Jacobi iterations per timed call


def try_import(module_name):
    try:
        return importlib.import_module(module_name)
        return True
    except ImportError as e:
        print(f"  Skipping {module_name}: {e}")
        return None


def run_one(framework_key, N, steps, warmup, repeat):
    module_name, backend, display_name = FRAMEWORKS[framework_key]
    mod = try_import(module_name)
    if mod is None:
        return None

    try:
        if backend == "cpu" and module_name in ("jacobi_numpy",):
            step_fn, sync_fn, _ = mod.run(N, steps=steps)
        else:
            step_fn, sync_fn, _ = mod.run(N, steps=steps, backend=backend)
    except Exception as e:
        print(f"  Error initializing {framework_key}: {e}")
        return None

    times = run_kernel_benchmark(step_fn, sync_fn, warmup=warmup, repeat=repeat)

    result = BenchmarkResult(
        kernel="jacobi_2d_5pt",
        framework=display_name,
        backend=backend,
        problem_size=f"{N}x{N}",
        warmup_runs=warmup,
        timed_runs=repeat,
        times_ms=times,
    )
    print(f"  {result.summary()}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Jacobi 2D Stencil Benchmark")
    parser.add_argument("--frameworks", nargs="+", default=None,
                        help=f"Frameworks to benchmark. Available: {list(FRAMEWORKS.keys())}")
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES,
                        help=f"Problem sizes NxN (default: {DEFAULT_SIZES})")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"Jacobi iterations per timed call (default: {DEFAULT_STEPS})")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--output", type=str, default="results.csv")
    args = parser.parse_args()

    fw_keys = args.frameworks or list(FRAMEWORKS.keys())
    results = []

    print(f"Jacobi 2D Benchmark: sizes={args.sizes}, steps={args.steps}")
    print(f"Frameworks: {fw_keys}")
    print()

    for fw in fw_keys:
        if fw not in FRAMEWORKS:
            print(f"Unknown framework: {fw}")
            continue
        for N in args.sizes:
            print(f"[{fw}] N={N}...")
            r = run_one(fw, N, args.steps, args.warmup, args.repeat)
            if r:
                results.append(r)

    print_results(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
