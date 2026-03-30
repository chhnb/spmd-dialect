#!/usr/bin/env python3
"""
run_reduction.py — Correctness and performance test for atomic_sum_kernel.

Reduction: out = sum(A[0..N-1])   (global atomic float add per thread)

Usage:
  bash scripts/gen-ptx.sh test/SPMD/lower-to-gpu-nvptx-reduction.mlir ewise /tmp/reduction.ptx
  python3 harness/run_reduction.py [--ptx /tmp/reduction.ptx] [--perf]
"""

import argparse
import math
import time
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import cuda_driver as cd

# ── Kernel constants ──────────────────────────────────────────────────────────
# Default tile size; overridden at runtime by --tile-size to match the compiled PTX.
_DEFAULT_TILE_SIZE = 256

# ── ABI reference (derived from LLVM IR) ─────────────────────────────────────
#
# define void @atomic_sum_kernel(
#   i64 %0,       param_0  = TILE_SIZE (blockDim limit)
#   i64 %1,       param_1  = N         (array length)
#   ptr %2,       param_2  = A.base_ptr
#   ptr %3,       param_3  = A.aligned_ptr  ← actual data pointer
#   i64 %4,       param_4  = A.offset (0)
#   i64 %5,       param_5  = A.size = N
#   i64 %6,       param_6  = A.stride = 1
#   ptr %7,       param_7  = out.base_ptr
#   ptr %8,       param_8  = out.aligned_ptr  ← scalar output pointer
#   i64 %9)       param_9  = out.offset (0)
#
# Launch: grid=(⌈N/TILE_SIZE⌉, 1, 1), block=(TILE_SIZE, 1, 1)
# Each thread atomically adds A[blockIdx.x * TILE_SIZE + threadIdx.x] to *out.
# Bounds check in kernel: threads with global index ≥ N are skipped.
# Caller must zero-initialize *out before launch.


def run_sum_gpu(fn, A: np.ndarray, tile_size: int) -> float:
    N = A.size
    A_c = np.ascontiguousarray(A, dtype=np.float32)

    A_d   = cd.alloc(A_c.nbytes)
    out_d = cd.alloc(4)          # 4 bytes = one f32 scalar

    cd.memcpy_h2d(A_d, A_c)
    # Zero-initialize the accumulator before atomic adds
    cd.memset(out_d, 0, 4)

    grid = (math.ceil(N / tile_size), 1, 1)
    cd.launch(
        fn,
        grid, (tile_size, 1, 1),
        # param_0: tile_size, param_1: N
        tile_size, N,
        # params 2-6: A memref1d descriptor
        A_d, A_d, 0, N, 1,
        # params 7-9: out rank-0 memref descriptor (base, aligned, offset)
        out_d, out_d, 0,
    )
    cd.synchronize()

    result_np = np.zeros(1, dtype=np.float32)
    cd.memcpy_d2h(result_np, out_d)

    A_d.free()
    out_d.free()
    return float(result_np[0])


def test_correctness(fn, sizes, tile_size: int):
    # RandomState is used here (and in run_host.py) so that all backends generate
    # the same deterministic inputs with the same seed, enabling true differential
    # comparison across CPU serial, OpenMP, and GPU backends.
    rng = np.random.RandomState(0)
    print(f"{'N':>12}  {'gpu_sum':>14}  {'ref_sum':>14}  {'rel_err':>10}  result")
    all_pass = True
    for N in sizes:
        A     = rng.random_sample(N).astype(np.float32)
        ref   = float(np.sum(A))
        gpu   = run_sum_gpu(fn, A, tile_size)

        # Use relative error tolerance for float32 reduction (accumulation error grows with N)
        rel_err = abs(gpu - ref) / max(abs(ref), 1e-6)
        ok = rel_err < 1e-3
        if not ok:
            all_pass = False
        print(f"{N:>12}  {gpu:>14.6f}  {ref:>14.6f}  {rel_err:>10.2e}  {'PASS' if ok else 'FAIL'}")
    return all_pass


def test_performance(fn, sizes, tile_size: int, repeats=10):
    rng = np.random.default_rng(1)
    print(f"\n{'N':>12}  {'cpu_ms':>8}  {'gpu_ms':>8}  {'speedup':>8}")
    for N in sizes:
        A   = rng.random(N, dtype=np.float32)
        A_c = np.ascontiguousarray(A, dtype=np.float32)
        A_d   = cd.alloc(A_c.nbytes)
        out_d = cd.alloc(4)
        cd.memcpy_h2d(A_d, A_c)

        grid = (math.ceil(N / tile_size), 1, 1)

        def launch_once():
            cd.memset(out_d, 0, 4)
            cd.launch(fn, grid, (tile_size, 1, 1),
                      tile_size, N,
                      A_d, A_d, 0, N, 1,
                      out_d, out_d, 0)

        # Warmup
        for _ in range(3):
            launch_once()
        cd.synchronize()

        # CPU
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = np.sum(A)
        t_cpu = (time.perf_counter() - t0) / repeats * 1e3

        # GPU
        t0 = time.perf_counter()
        for _ in range(repeats):
            launch_once()
        cd.synchronize()
        t_gpu = (time.perf_counter() - t0) / repeats * 1e3

        A_d.free()
        out_d.free()
        print(f"{N:>12}  {t_cpu:>8.3f}  {t_gpu:>8.3f}  {t_cpu/t_gpu:>8.1f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptx",   default="/tmp/reduction.ptx")
    ap.add_argument("--sizes", default="1024,65536,1048576")
    ap.add_argument("--perf",  action="store_true")
    ap.add_argument("--perf-sizes", default="",
                    help="Sizes to use for performance timing (default: same as --sizes).")
    ap.add_argument("--tile-size", type=int, default=_DEFAULT_TILE_SIZE,
                    help="Tile size baked into the PTX (blockDim.x). "
                         "Must match the spmd.tile_sizes used when compiling the PTX.")
    args = ap.parse_args()

    tile_size = args.tile_size
    sizes = [int(s) for s in args.sizes.split(",")]

    # Performance sizes default to same as correctness sizes
    perf_sizes_str = args.perf_sizes if args.perf_sizes else args.sizes
    perf_sizes = [int(s) for s in perf_sizes_str.split(",")]

    print(f"Loading PTX: {args.ptx}")
    cd.init()
    mod = cd.load_ptx(args.ptx)
    fn  = cd.get_function(mod, "atomic_sum_kernel")

    print("\n=== Correctness ===")
    ok = test_correctness(fn, sizes, tile_size)

    if args.perf:
        print("\n=== Performance ===")
        test_performance(fn, perf_sizes, tile_size)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
