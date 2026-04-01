#!/usr/bin/env python3
"""Benchmark runner for LBM D2Q9."""
import argparse, importlib, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.harness import BenchmarkResult, run_kernel_benchmark, save_results, print_results

FRAMEWORKS = {
    "numpy":       ("lbm_numpy",   "cpu",  "NumPy"),
    "taichi_cpu":  ("lbm_taichi",  "cpu",  "Taichi"),
    "taichi_cuda": ("lbm_taichi",  "cuda", "Taichi"),
}
DEFAULT_SIZES = [(512, 256), (1024, 512), (2048, 1024)]

def main():
    parser = argparse.ArgumentParser(description="LBM D2Q9 Benchmark")
    parser.add_argument("--frameworks", nargs="+", default=None)
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
        module_name, backend, display_name = FRAMEWORKS[fw]
        try:
            mod = importlib.import_module(module_name)
        except ImportError as e:
            print(f"  Skip {fw}: {e}"); continue
        for nx, ny in DEFAULT_SIZES:
            print(f"[{fw}] {nx}x{ny}...")
            try:
                step_fn, sync_fn, _ = mod.run(nx, ny, steps=args.steps, backend=backend)
            except Exception as e:
                print(f"  Error: {e}"); continue
            times = run_kernel_benchmark(step_fn, sync_fn, warmup=args.warmup, repeat=args.repeat)
            r = BenchmarkResult("lbm_d2q9", display_name, backend, f"{nx}x{ny}",
                                args.warmup, args.repeat, times)
            print(f"  {r.summary()}")
            results.append(r)
    print_results(results)
    save_results(results, args.output)

if __name__ == "__main__":
    main()
