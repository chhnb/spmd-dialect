#!/usr/bin/env python3
"""Benchmark runner for F2: Refactored Hydro-Cal (edge-parallel, fp32).

Real hydro-cal mesh (6675 cells), 2 kernels per step (CalculateFlux + UpdateCell).
"""
import argparse
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.harness import (
    BenchmarkResult,
    run_kernel_benchmark,
    save_results,
    print_results,
)

FRAMEWORKS = {
    "taichi_cuda": ("hydro_refactored_taichi", "cuda", "Taichi"),
    "warp_cuda":   ("hydro_refactored_warp",   "cuda", "Warp"),
}

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark F2: Refactored Hydro-Cal (edge-parallel, fp32)"
    )
    parser.add_argument("--frameworks", nargs="+", default=None)
    parser.add_argument("--days", type=int, default=10, help="Number of simulation days")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--output", type=str, default="results.csv")
    args = parser.parse_args()

    fw_keys = args.frameworks or list(FRAMEWORKS.keys())
    results = []

    for fw in fw_keys:
        if fw not in FRAMEWORKS:
            print(f"Unknown framework: {fw}")
            continue
        mod_name, backend, display = FRAMEWORKS[fw]
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as e:
            print(f"  Skip {fw}: {e}")
            continue

        print(f"[{fw}] {args.days} days (6675 cells, fp32)...")
        try:
            step_fn, sync_fn, _ = mod.run(days=args.days, backend=backend)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback; traceback.print_exc()
            continue

        times = run_kernel_benchmark(
            step_fn, sync_fn, warmup=args.warmup, repeat=args.repeat
        )
        r = BenchmarkResult(
            kernel="hydro_refactored",
            framework=display,
            backend=backend,
            problem_size=f"6675cells_{args.days}days",
            warmup_runs=args.warmup,
            timed_runs=args.repeat,
            times_ms=times,
        )
        print(f"  {r.summary()}")
        results.append(r)

    print_results(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
