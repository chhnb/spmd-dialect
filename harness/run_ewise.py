#!/usr/bin/env python3
"""
run_ewise.py — Correctness + basic performance test for ewise_kernel.

What this tests:
  B[i] = A[i] + B_in[i]   (elementwise float32 add)
  Three backends compared:
    cpu_ref  : numpy A + B
    gpu_spmd : SPMD-generated PTX on CUDA device

Usage:
  # 1. generate PTX first (auto-detects SM)
  bash scripts/gen-ptx.sh test/SPMD/lower-to-gpu-nvptx.mlir ewise /tmp/ewise.ptx

  # 2. run harness
  python3 harness/run_ewise.py [--ptx /tmp/ewise.ptx] [--sizes 32,1024,1000000]
"""

import argparse
import math
import time
import sys
import os

import numpy as np

# Insert harness dir so cuda_driver is importable regardless of cwd
sys.path.insert(0, os.path.dirname(__file__))
import cuda_driver as cd

# ── ewise_kernel ABI ──────────────────────────────────────────────────────────
#
# Confirmed from PTX inspection (sm_80 and sm_100 produce identical ABI):
#
#  param_0  : i64  tile_size (= blockDim.x = 32)
#  param_1  : i64  N
#  param_2  : u64  A.base_ptr  (= A.aligned_ptr for simple allocations)
#  param_3  : u64  A.aligned_ptr   ← actual data pointer used
#  param_4  : i64  A.offset  (= 0)
#  param_5  : i64  A.size    (= N)
#  param_6  : i64  A.stride  (= 1)
#  param_7  : u64  B.base_ptr
#  param_8  : u64  B.aligned_ptr
#  param_9  : i64  B.offset
#  param_10 : i64  B.size
#  param_11 : i64  B.stride
#  param_12 : u64  C.base_ptr
#  param_13 : u64  C.aligned_ptr
#  param_14 : i64  C.offset
#  param_15 : i64  C.size
#  param_16 : i64  C.stride
#
# Launch: grid=(ceil(N/32), 1, 1), block=(32, 1, 1)

def memref1d(ptr: cd.DevicePtr, n: int):
    """Return the 5-element memref descriptor for a 1D memref<?xf32>."""
    return (ptr, ptr, 0, n, 1)


def run_ewise_gpu(fn, A_d: cd.DevicePtr, B_d: cd.DevicePtr,
                  C_d: cd.DevicePtr, N: int, tile: int):
    grid = (math.ceil(N / tile), 1, 1)
    cd.launch(
        fn,
        grid, (tile, 1, 1),
        tile, N,
        *memref1d(A_d, N),
        *memref1d(B_d, N),
        *memref1d(C_d, N),
    )


def test_correctness(fn, sizes, tile: int):
    rng = np.random.default_rng(42)
    print(f"{'N':>12}  {'max_err':>10}  result")
    all_pass = True
    for N in sizes:
        A = rng.random(N, dtype=np.float32)
        B = rng.random(N, dtype=np.float32)
        C_ref = A + B   # numpy reference

        A_d = cd.alloc(A.nbytes); cd.memcpy_h2d(A_d, A)
        B_d = cd.alloc(B.nbytes); cd.memcpy_h2d(B_d, B)
        C_d = cd.alloc(A.nbytes)

        run_ewise_gpu(fn, A_d, B_d, C_d, N, tile)
        cd.synchronize()

        C_gpu = np.empty(N, dtype=np.float32)
        cd.memcpy_d2h(C_gpu, C_d)

        A_d.free(); B_d.free(); C_d.free()

        max_err = float(np.max(np.abs(C_gpu - C_ref)))
        ok = max_err < 1e-5
        if not ok:
            all_pass = False
        print(f"{N:>12}  {max_err:>10.2e}  {'PASS' if ok else 'FAIL'}")
    return all_pass


def test_performance(fn, sizes, tile: int, repeats=20):
    rng = np.random.default_rng(0)
    print(f"\n{'N':>12}  {'cpu_ms':>8}  {'gpu_ms':>8}  {'speedup':>8}")
    for N in sizes:
        A = rng.random(N, dtype=np.float32)
        B = rng.random(N, dtype=np.float32)

        A_d = cd.alloc(A.nbytes); cd.memcpy_h2d(A_d, A)
        B_d = cd.alloc(B.nbytes); cd.memcpy_h2d(B_d, B)
        C_d = cd.alloc(A.nbytes)

        # Warmup
        for _ in range(3):
            run_ewise_gpu(fn, A_d, B_d, C_d, N, tile)
        cd.synchronize()

        # CPU timing
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = A + B
        t_cpu = (time.perf_counter() - t0) / repeats * 1e3

        # GPU timing (includes launch overhead, not transfer)
        t0 = time.perf_counter()
        for _ in range(repeats):
            run_ewise_gpu(fn, A_d, B_d, C_d, N, tile)
        cd.synchronize()
        t_gpu = (time.perf_counter() - t0) / repeats * 1e3

        A_d.free(); B_d.free(); C_d.free()
        print(f"{N:>12}  {t_cpu:>8.3f}  {t_gpu:>8.3f}  {t_cpu/t_gpu:>8.1f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptx",   default="/tmp/ewise.ptx")
    ap.add_argument("--sizes", default="32,100,1024,10000,1000000")
    ap.add_argument("--perf",  action="store_true", help="also run perf test")
    ap.add_argument("--perf-sizes", default="100000,1000000,10000000")
    ap.add_argument("--tile-size", type=int, default=32,
                    help="Tile size baked into the PTX (blockDim.x). "
                         "Must match the spmd.tile_sizes used when compiling the PTX.")
    args = ap.parse_args()

    tile       = args.tile_size
    sizes      = [int(x) for x in args.sizes.split(",")]
    perf_sizes = [int(x) for x in args.perf_sizes.split(",")]

    print(f"Loading PTX: {args.ptx}")
    cd.init()
    mod = cd.load_ptx(args.ptx)
    fn  = cd.get_function(mod, "ewise_kernel")

    print("\n=== Correctness ===")
    ok = test_correctness(fn, sizes, tile)

    if args.perf:
        print("\n=== Performance (wall-clock, no data transfer) ===")
        test_performance(fn, perf_sizes, tile)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
