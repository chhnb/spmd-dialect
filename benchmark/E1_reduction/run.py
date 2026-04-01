#!/usr/bin/env python3
"""Benchmark runner for Global Reduction (sum).

Usage:
    python run.py
    python run.py --frameworks numpy taichi_cuda triton_cuda
    python run.py --sizes 1000000 10000000 100000000
"""

import argparse
import importlib
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.harness import BenchmarkResult, run_kernel_benchmark, save_results, print_results


FRAMEWORKS = {
    "numpy":       ("reduce_numpy",   "cpu",  "NumPy"),
    "taichi_cpu":  ("reduce_taichi",  "cpu",  "Taichi"),
    "taichi_cuda": ("reduce_taichi",  "cuda", "Taichi"),
    "warp_cuda":   ("reduce_warp",    "cuda", "Warp"),
    "triton_cuda": ("reduce_triton",  "cuda", "Triton"),
}

DEFAULT_SIZES = [100_000, 1_000_000, 10_000_000, 100_000_000]


def try_import(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        print(f"  Skipping {module_name}: {e}")
        return None


def run_one(framework_key, N, warmup, repeat):
    module_name, backend, display_name = FRAMEWORKS[framework_key]
    mod = try_import(module_name)
    if mod is None:
        return None

    try:
        step_fn, sync_fn, _ = mod.run(N, backend=backend)
    except Exception as e:
        print(f"  Error initializing {framework_key}: {e}")
        return None

    times = run_kernel_benchmark(step_fn, sync_fn, warmup=warmup, repeat=repeat)

    result = BenchmarkResult(
        kernel="global_reduction_sum",
        framework=display_name,
        backend=backend,
        problem_size=f"{N}",
        warmup_runs=warmup,
        timed_runs=repeat,
        times_ms=times,
    )
    print(f"  {result.summary()}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Global Reduction Benchmark")
    parser.add_argument("--frameworks", nargs="+", default=None)
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--output", type=str, default="results.csv")
    args = parser.parse_args()

    fw_keys = args.frameworks or list(FRAMEWORKS.keys())
    results = []

    print(f"Global Reduction Benchmark: sizes={args.sizes}")
    print(f"Frameworks: {fw_keys}")
    print()

    for fw in fw_keys:
        if fw not in FRAMEWORKS:
            print(f"Unknown framework: {fw}")
            continue
        for N in args.sizes:
            print(f"[{fw}] N={N:,}...")
            r = run_one(fw, N, args.warmup, args.repeat)
            if r:
                results.append(r)

    print_results(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
