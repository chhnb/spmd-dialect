#!/usr/bin/env python3
"""Benchmark runner for SPH density computation."""
import argparse, importlib, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.harness import BenchmarkResult, run_kernel_benchmark, save_results, print_results

FRAMEWORKS = {
    "taichi_cuda": ("sph_taichi",  "cuda", "Taichi"),
    "warp_cuda":   ("sph_warp",    "cuda", "Warp"),
}
DEFAULT_SIZES = [8192, 16384, 32768, 65536]

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
        if fw not in FRAMEWORKS: continue
        mn, be, dn = FRAMEWORKS[fw]
        try: mod = importlib.import_module(mn)
        except ImportError as e: print(f"  Skip {fw}: {e}"); continue
        for N in args.sizes:
            print(f"[{fw}] N={N}...")
            try: s, sy, _ = mod.run(N, steps=args.steps, backend=be)
            except Exception as e: print(f"  Error: {e}"); continue
            times = run_kernel_benchmark(s, sy, warmup=args.warmup, repeat=args.repeat)
            r = BenchmarkResult("sph_density", dn, be, f"{N}", args.warmup, args.repeat, times)
            print(f"  {r.summary()}"); results.append(r)
    print_results(results); save_results(results, args.output)

if __name__ == "__main__": main()
