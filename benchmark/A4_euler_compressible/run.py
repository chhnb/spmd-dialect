#!/usr/bin/env python3
"""Benchmark runner for Compressible Euler 2D."""
import argparse, importlib, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.harness import BenchmarkResult, run_kernel_benchmark, save_results, print_results

FRAMEWORKS = {
    "taichi_cuda": ("euler_taichi", "cuda", "Taichi"),
    "taichi_cpu":  ("euler_taichi", "cpu",  "Taichi"),
}
DEFAULT_SIZES = [128, 256, 512]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frameworks", nargs="+", default=None)
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--output", type=str, default="results.csv")
    args = parser.parse_args()
    fw_keys = args.frameworks or list(FRAMEWORKS.keys())
    results = []
    for fw in fw_keys:
        if fw not in FRAMEWORKS:
            continue
        mn, be, dn = FRAMEWORKS[fw]
        mod = importlib.import_module(mn)
        for n in args.sizes:
            print(f"[{fw}] N={n}...")
            try:
                s, sy, _ = mod.run(n, steps=args.steps, backend=be)
            except Exception as e:
                print(f"  Error: {e}")
                continue
            times = run_kernel_benchmark(s, sy, warmup=args.warmup, repeat=args.repeat)
            r = BenchmarkResult("euler_compressible", dn, be, f"{n}x{n}", args.warmup, args.repeat, times)
            print(f"  {r.summary()}")
            results.append(r)
    print_results(results)
    save_results(results, args.output)

if __name__ == "__main__":
    main()
