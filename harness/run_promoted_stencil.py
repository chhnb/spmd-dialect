#!/usr/bin/env python3
"""
run_promoted_stencil.py — Correctness test for promoted_stencil_kernel.

Stencil: B[i,j] = A[i,j] + A[i,j+1] + A[i+1,j]   (interior elements only)

Usage:
  bash scripts/gen-ptx.sh test/SPMD/lower-to-gpu-nvptx-promoted.mlir promoted /tmp/stencil.ptx
  python3 harness/run_promoted_stencil.py [--ptx /tmp/stencil.ptx] [--perf]
"""

import argparse
import math
import time
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import cuda_driver as cd

# ── Default kernel constants (match source MLIR spmd.tile_sizes = [32, 8]) ────
# Override at runtime with --tile-row / --tile-col to match the compiled PTX.
_DEFAULT_TILE_ROW = 32
_DEFAULT_TILE_COL = 8

# ── ABI reference (derived from gpu-kernel-outlining MLIR output) ────────────
#
# The outlined kernel always has the form:
#   gpu.launch_func args(TILE_ROW, TILE_COL, A_memref, %c1, B_memref)
#
# COOP_COLS, COOP_THREADS, COMPUTE_THREADS are baked in as PTX immediates
# (they are arith.constant ops inside the kernel body, not function arguments).
# %c1 is the stencil neighbor offset (j+1, i+1); it is a function argument
# because it is defined at function scope in the source MLIR.
#
# When TILE_ROW != TILE_COL (e.g. 32x8, 64x8): 17 params
#   i64 %0  = TILE_ROW
#   i64 %1  = TILE_COL
#   ptr %2  = A.base_ptr,    ptr %3  = A.aligned_ptr
#   i64 %4  = A.offset(0),   i64 %5  = N,  i64 %6  = M
#   i64 %7  = A.stride[0]=M, i64 %8  = A.stride[1]=1
#   i64 %9  = 1  (stencil constant %c1)
#   ptr %10 = B.base_ptr,    ptr %11 = B.aligned_ptr
#   i64 %12 = B.offset(0),   i64 %13 = N,  i64 %14 = M
#   i64 %15 = B.stride[0]=M, i64 %16 = B.stride[1]=1
#
# When TILE_ROW == TILE_COL (e.g. 16x16): 16 params
#   MLIR CSE merges the two equal arith.constant ops into one SSA value, so
#   TILE_ROW and TILE_COL share a single kernel param.  Layout:
#   i64 %0  = TILE_ROW(=TILE_COL)
#   ptr %1  = A.base_ptr,    ptr %2  = A.aligned_ptr
#   i64 %3  = A.offset(0),   i64 %4  = N,  i64 %5  = M
#   i64 %6  = A.stride[0]=M, i64 %7  = A.stride[1]=1
#   i64 %8  = 1  (stencil constant %c1)
#   ptr %9  = B.base_ptr,    ptr %10 = B.aligned_ptr
#   i64 %11 = B.offset(0),   i64 %12 = N,  i64 %13 = M
#   i64 %14 = B.stride[0]=M, i64 %15 = B.stride[1]=1
#
# Launch: grid=(⌈N/TILE_ROW⌉, ⌈M/TILE_COL⌉, 1), block=(COOP_THREADS, 1, 1)
# Shared: static (.shared declared in PTX; pass shared_bytes=0 to cuLaunchKernel)
#
# NOTE: the kernel writes B without a bounds check on the output index, so N
# must be a multiple of TILE_ROW and M a multiple of TILE_COL to avoid OOB writes.


def run_stencil_gpu(fn, A: np.ndarray, tile_row: int, tile_col: int) -> np.ndarray:
    coop_threads = (tile_row + 1) * (tile_col + 1)

    N, M = A.shape
    assert N % tile_row == 0, f"N={N} must be a multiple of tile_row={tile_row}"
    assert M % tile_col == 0, f"M={M} must be a multiple of tile_col={tile_col}"

    A_c  = np.ascontiguousarray(A, dtype=np.float32)
    B_np = np.zeros((N, M), dtype=np.float32)

    A_d = cd.alloc(A_c.nbytes); cd.memcpy_h2d(A_d, A_c)
    B_d = cd.alloc(B_np.nbytes)

    grid = (math.ceil(N / tile_row), math.ceil(M / tile_col), 1)
    # When tile_row == tile_col, MLIR CSE merges the two equal arith.constant
    # ops into one SSA value → 16 params; otherwise 17 params.
    if tile_row == tile_col:
        cd.launch(
            fn,
            grid, (coop_threads, 1, 1),
            tile_row,                      # param_0: tile (shared row=col)
            A_d, A_d, 0, N, M, M, 1,      # params 1-7: A descriptor
            1,                             # param_8: stencil constant
            B_d, B_d, 0, N, M, M, 1,      # params 9-15: B descriptor
            shared_bytes=0,
        )
    else:
        cd.launch(
            fn,
            grid, (coop_threads, 1, 1),
            tile_row, tile_col,            # params 0-1: tile geometry
            A_d, A_d, 0, N, M, M, 1,      # params 2-8: A descriptor
            1,                             # param_9: stencil constant
            B_d, B_d, 0, N, M, M, 1,      # params 10-16: B descriptor
            shared_bytes=0,
        )
    cd.synchronize()

    cd.memcpy_d2h(B_np, B_d)
    A_d.free(); B_d.free()
    return B_np


def cpu_reference(A: np.ndarray) -> np.ndarray:
    """B[i,j] = A[i,j] + A[i,j+1] + A[i+1,j]  for i<N-1, j<M-1 (interior)."""
    N, M = A.shape
    B = np.zeros_like(A)
    B[:N-1, :M-1] = A[:N-1, :M-1] + A[:N-1, 1:M] + A[1:N, :M-1]
    return B


def test_correctness(fn, shapes, tile_row: int, tile_col: int):
    # RandomState is used here (and in run_host.py) so that all backends generate
    # the same deterministic inputs with the same seed, enabling true differential
    # comparison across CPU serial, OpenMP, and GPU backends.
    #
    # A is allocated as (N+1, M+1) to match the CPU harness layout: the extra
    # row/col allows the stencil to read A[i,j+1] and A[i+1,j] at the boundary
    # without OOB access. The GPU kernel receives the A[:N, :M] sub-region; the
    # interior comparison uses only values within that sub-region, so both
    # backends operate on identical data for all compared cells.
    rng = np.random.RandomState(0)
    print(f"{'shape':>14}  {'max_err':>10}  result")
    all_pass = True
    for (N, M) in shapes:
        A_big = rng.random_sample((N + 1, M + 1)).astype(np.float32)
        A     = A_big[:N, :M]   # (N, M) sub-region passed to GPU kernel
        B_ref = cpu_reference(A)
        B_gpu = run_stencil_gpu(fn, A, tile_row, tile_col)

        # Compare interior only (last row/col are zero in both, not computed)
        interior = (slice(0, N-1), slice(0, M-1))
        max_err  = float(np.max(np.abs(B_gpu[interior] - B_ref[interior])))
        ok = max_err < 1e-4
        if not ok:
            all_pass = False
        print(f"({N:4d},{M:4d}){' ':>3}  {max_err:>10.2e}  {'PASS' if ok else 'FAIL'}")
    return all_pass


def test_performance(fn, shapes, tile_row: int, tile_col: int, repeats=10):
    coop_threads = (tile_row + 1) * (tile_col + 1)

    rng = np.random.default_rng(1)
    print(f"\n{'shape':>14}  {'cpu_ms':>8}  {'gpu_ms':>8}  {'speedup':>8}")
    for (N, M) in shapes:
        A   = rng.random((N, M), dtype=np.float32)
        A_c = np.ascontiguousarray(A, dtype=np.float32)
        A_d = cd.alloc(A_c.nbytes); cd.memcpy_h2d(A_d, A_c)
        B_d = cd.alloc(A_c.nbytes)

        grid = (math.ceil(N / tile_row), math.ceil(M / tile_col), 1)

        def launch_once():
            if tile_row == tile_col:
                cd.launch(fn, grid, (coop_threads, 1, 1),
                          tile_row,
                          A_d, A_d, 0, N, M, M, 1,
                          1,
                          B_d, B_d, 0, N, M, M, 1)
            else:
                cd.launch(fn, grid, (coop_threads, 1, 1),
                          tile_row, tile_col,
                          A_d, A_d, 0, N, M, M, 1,
                          1,
                          B_d, B_d, 0, N, M, M, 1)

        # Warmup
        for _ in range(3): launch_once()
        cd.synchronize()

        # CPU
        t0 = time.perf_counter()
        for _ in range(repeats): _ = cpu_reference(A)
        t_cpu = (time.perf_counter() - t0) / repeats * 1e3

        # GPU
        t0 = time.perf_counter()
        for _ in range(repeats): launch_once()
        cd.synchronize()
        t_gpu = (time.perf_counter() - t0) / repeats * 1e3

        A_d.free(); B_d.free()
        print(f"({N:4d},{M:4d}){' ':>3}  {t_cpu:>8.3f}  {t_gpu:>8.3f}  {t_cpu/t_gpu:>8.1f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptx",         default="/tmp/stencil.ptx")
    ap.add_argument("--shapes",      default="64x64,128x128,512x512,1024x1024")
    ap.add_argument("--perf",        action="store_true")
    ap.add_argument("--perf-shapes", default="",
                    help="Shapes to use for performance timing (default: same as --shapes)")
    ap.add_argument("--tile-row", type=int, default=_DEFAULT_TILE_ROW,
                    help="Tile row size baked into the PTX. "
                         "Must match the spmd.tile_sizes[0] used when compiling the PTX.")
    ap.add_argument("--tile-col", type=int, default=_DEFAULT_TILE_COL,
                    help="Tile col size baked into the PTX. "
                         "Must match the spmd.tile_sizes[1] used when compiling the PTX.")
    args = ap.parse_args()

    tile_row = args.tile_row
    tile_col = args.tile_col

    shapes = []
    for s in args.shapes.split(","):
        n, m = map(int, s.split("x"))
        shapes.append((n, m))

    # Performance shapes default to same as correctness shapes
    perf_shapes_str = args.perf_shapes if args.perf_shapes else args.shapes
    perf_shapes = []
    for s in perf_shapes_str.split(","):
        n, m = map(int, s.split("x"))
        perf_shapes.append((n, m))

    print(f"Loading PTX: {args.ptx}")
    cd.init()
    mod = cd.load_ptx(args.ptx)
    fn  = cd.get_function(mod, "promoted_stencil_kernel")

    print("\n=== Correctness ===")
    ok = test_correctness(fn, shapes, tile_row, tile_col)

    if args.perf:
        print("\n=== Performance ===")
        test_performance(fn, perf_shapes, tile_row, tile_col)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
