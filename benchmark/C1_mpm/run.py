#!/usr/bin/env python3
"""Benchmark runner for MPM."""
import argparse, importlib, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.harness import BenchmarkResult, run_kernel_benchmark, save_results, print_results

FRAMEWORKS = {
    "taichi_cuda": ("mpm_taichi", "cuda", "Taichi"),
    "taichi_cpu":  ("mpm_taichi", "cpu",  "Taichi"),
}
# (n_grid, n_particles) pairs
DEFAULT_CONFIGS = [(64, 4096), (128, 8192), (128, 32768), (256, 65536)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frameworks", nargs="+", default=None)
    parser.add_argument("--steps", type=int, default=50, help="substeps per timed call")
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
        for ng, np_ in DEFAULT_CONFIGS:
            label = f"grid{ng}_p{np_}"
            print(f"[{fw}] {label}...")
            try: s, sy, _ = mod.run(n_grid=ng, n_particles=np_, steps=args.steps, backend=be)
            except Exception as e: print(f"  Error: {e}"); continue
            times = run_kernel_benchmark(s, sy, warmup=args.warmup, repeat=args.repeat)
            r = BenchmarkResult("mpm", dn, be, label, args.warmup, args.repeat, times)
            print(f"  {r.summary()}"); results.append(r)
    print_results(results); save_results(results, args.output)

if __name__ == "__main__": main()
