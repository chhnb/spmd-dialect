#!/usr/bin/env python3
"""Benchmark runner for 2D Shallow Water Equations (Osher Riemann solver).

Supports two modes:
  - Synthetic dam-break on NxN grid (default, scalable sizes)
  - Real hydro-cal mesh (--real flag, 6675 cells from data/)
"""
import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.harness import (
    BenchmarkResult,
    run_kernel_benchmark,
    save_results,
    print_results,
)

FRAMEWORKS = {
    "numpy":         ("hydro_numpy",    "cpu",  "NumPy"),
    "taichi_cpu":    ("hydro_taichi",   "cpu",  "Taichi"),
    "taichi_cuda":   ("hydro_taichi",   "cuda", "Taichi"),
    "warp_cuda":     ("hydro_warp",     "cuda", "Warp"),
    "triton_cuda":   ("hydro_triton",   "cuda", "Triton"),
    "tilelang_cuda": ("hydro_tilelang", "cuda", "TileLang"),
}

DEFAULT_SIZES = [128, 256, 512]


def run_synthetic(args):
    """Run benchmarks on synthetic dam-break meshes."""
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

        for N in args.sizes:
            cel = N * N
            print(f"[{fw}] N={N} ({cel} cells)...")
            try:
                step_fn, sync_fn, _ = mod.run(N, steps=args.steps, backend=backend)
            except Exception as e:
                print(f"  Error: {e}")
                import traceback; traceback.print_exc()
                continue

            times = run_kernel_benchmark(
                step_fn, sync_fn, warmup=args.warmup, repeat=args.repeat
            )
            r = BenchmarkResult(
                kernel="hydro_swe_osher",
                framework=display,
                backend=backend,
                problem_size=f"{N}x{N}",
                warmup_runs=args.warmup,
                timed_runs=args.repeat,
                times_ms=times,
            )
            print(f"  {r.summary()}")
            results.append(r)

    return results


def run_real_mesh(args):
    """Run benchmarks on real hydro-cal mesh."""
    from mesh_loader import load_hydro_mesh
    mesh_data = load_hydro_mesh(mesh=args.mesh)
    cel = mesh_data['CEL']
    print(f"Loaded mesh: {cel} cells (mesh={args.mesh})")

    fw_keys = args.frameworks or ["numpy"]
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

        if not hasattr(mod, "run_real"):
            print(f"  Skip {fw}: no run_real() function")
            continue

        print(f"[{fw}] real mesh ({cel} cells)...")
        try:
            step_fn, sync_fn, _ = mod.run_real(steps=args.steps, backend=backend, mesh=args.mesh)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback; traceback.print_exc()
            continue

        times = run_kernel_benchmark(
            step_fn, sync_fn, warmup=args.warmup, repeat=args.repeat
        )
        r = BenchmarkResult(
            kernel="hydro_swe_osher_real",
            framework=display,
            backend=backend,
            problem_size=f"{cel}cells",
            warmup_runs=args.warmup,
            timed_runs=args.repeat,
            times_ms=times,
        )
        print(f"  {r.summary()}")
        results.append(r)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark 2D Shallow Water (Osher) — dam break / real mesh"
    )
    parser.add_argument("--frameworks", nargs="+", default=None,
                        help=f"Frameworks to run. Options: {list(FRAMEWORKS.keys())}")
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES,
                        help="Grid side lengths for synthetic mode (NxN cells)")
    parser.add_argument("--steps", type=int, default=10,
                        help="Time steps per timed call")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--output", type=str, default="results.csv")
    parser.add_argument("--real", action="store_true",
                        help="Use real hydro-cal mesh instead of synthetic")
    parser.add_argument("--mesh", type=str, default="default",
                        help="Mesh dataset: 'default' (6675 cells), '20w' (207234 cells)")
    args = parser.parse_args()

    if args.real:
        results = run_real_mesh(args)
    else:
        results = run_synthetic(args)

    print_results(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
