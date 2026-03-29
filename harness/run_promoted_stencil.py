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

# ── Kernel constants (must match SPMD tile_sizes and halo) ────────────────────
TILE_ROW  = 32
TILE_COL  = 8
# Cooperative copy covers a (TILE_ROW+1) × (TILE_COL+1) halo tile.
COOP_COLS = TILE_COL + 1          # 9  — inner col divisor for cooperative-copy linearisation
COOP_THREADS = (TILE_ROW + 1) * (TILE_COL + 1)  # 297 = blockDim.x
COMPUTE_THREADS = TILE_ROW * TILE_COL            # 256 = compute phase upper bound

# ── ABI reference (derived from LLVM IR, not PTX text) ───────────────────────
#
# define ptx_kernel void @promoted_stencil_kernel(
#   i64 %0,       param_0  = TILE_ROW  (blockIdx.x step)
#   i64 %1,       param_1  = TILE_COL  (blockIdx.y step; compute divisor)
#   i64 %2,       param_2  = COOP_COLS (= TILE_COL+1; coop-copy linearisation)
#   i64 %3,       param_3  = COOP_THREADS (= 297; coop-copy participation bound)
#   ptr %4,       param_4  = A.base_ptr
#   ptr %5,       param_5  = A.aligned_ptr      ← actual data pointer
#   i64 %6,       param_6  = A.offset (0)
#   i64 %7,       param_7  = A.size[0] = N
#   i64 %8,       param_8  = A.size[1] = M
#   i64 %9,       param_9  = A.stride[0] = M    ← row-major stride
#   i64 %10,      param_10 = A.stride[1] = 1
#   i64 %11,      param_11 = 0  (dim-index for row boundary check)
#   i64 %12,      param_12 = 1  (dim-index for col boundary check AND stencil offset)
#   i64 %13,      param_13 = COMPUTE_THREADS (= 256; compute phase bound)
#   ptr %14,      param_14 = B.base_ptr
#   ptr %15,      param_15 = B.aligned_ptr
#   i64 %16,      param_16 = B.offset (0)
#   i64 %17,      param_17 = B.size[0] = N
#   i64 %18,      param_18 = B.size[1] = M
#   i64 %19,      param_19 = B.stride[0] = M
#   i64 %20)      param_20 = B.stride[1] = 1
#
# Launch: grid=(⌈N/TILE_ROW⌉, ⌈M/TILE_COL⌉, 1), block=(297, 1, 1)
# Shared: 1188 B static (.shared declared in PTX; pass shared_bytes=0 to cuLaunchKernel)
#
# NOTE: the kernel writes B without a bounds check on the output index, so N
# must be a multiple of TILE_ROW and M a multiple of TILE_COL to avoid OOB writes.


def run_stencil_gpu(fn, A: np.ndarray) -> np.ndarray:
    N, M = A.shape
    assert N % TILE_ROW == 0, f"N={N} must be a multiple of TILE_ROW={TILE_ROW}"
    assert M % TILE_COL == 0, f"M={M} must be a multiple of TILE_COL={TILE_COL}"

    A_c  = np.ascontiguousarray(A, dtype=np.float32)
    B_np = np.zeros((N, M), dtype=np.float32)

    A_d = cd.alloc(A_c.nbytes); cd.memcpy_h2d(A_d, A_c)
    B_d = cd.alloc(B_np.nbytes)

    grid = (math.ceil(N / TILE_ROW), math.ceil(M / TILE_COL), 1)
    cd.launch(
        fn,
        grid, (COOP_THREADS, 1, 1),
        # params 0-3: tile geometry
        TILE_ROW, TILE_COL, COOP_COLS, COOP_THREADS,
        # params 4-10: A memref2d descriptor
        A_d, A_d, 0, N, M, M, 1,
        # params 11-13: dim indices (constants) + compute thread bound
        0, 1, COMPUTE_THREADS,
        # params 14-20: B memref2d descriptor
        B_d, B_d, 0, N, M, M, 1,
        shared_bytes=0,   # static .shared in PTX handles the 1188 B automatically
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


def test_correctness(fn, shapes):
    rng = np.random.default_rng(0)
    print(f"{'shape':>14}  {'max_err':>10}  result")
    all_pass = True
    for (N, M) in shapes:
        A     = rng.random((N, M), dtype=np.float32)
        B_ref = cpu_reference(A)
        B_gpu = run_stencil_gpu(fn, A)

        # Compare interior only (last row/col are zero in both, not computed)
        interior = (slice(0, N-1), slice(0, M-1))
        max_err  = float(np.max(np.abs(B_gpu[interior] - B_ref[interior])))
        ok = max_err < 1e-4
        if not ok:
            all_pass = False
        print(f"({N:4d},{M:4d}){' ':>3}  {max_err:>10.2e}  {'PASS' if ok else 'FAIL'}")
    return all_pass


def test_performance(fn, shapes, repeats=10):
    rng = np.random.default_rng(1)
    print(f"\n{'shape':>14}  {'cpu_ms':>8}  {'gpu_ms':>8}  {'speedup':>8}")
    for (N, M) in shapes:
        A   = rng.random((N, M), dtype=np.float32)
        A_c = np.ascontiguousarray(A, dtype=np.float32)
        A_d = cd.alloc(A_c.nbytes); cd.memcpy_h2d(A_d, A_c)
        B_d = cd.alloc(A_c.nbytes)

        grid = (math.ceil(N / TILE_ROW), math.ceil(M / TILE_COL), 1)

        def launch_once():
            cd.launch(fn, grid, (COOP_THREADS, 1, 1),
                      TILE_ROW, TILE_COL, COOP_COLS, COOP_THREADS,
                      A_d, A_d, 0, N, M, M, 1,
                      0, 1, COMPUTE_THREADS,
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
    ap.add_argument("--ptx",    default="/tmp/stencil.ptx")
    ap.add_argument("--shapes", default="64x64,128x128,512x512,1024x1024")
    ap.add_argument("--perf",   action="store_true")
    args = ap.parse_args()

    # All shapes must be multiples of TILE_ROW × TILE_COL
    shapes = []
    for s in args.shapes.split(","):
        n, m = map(int, s.split("x"))
        shapes.append((n, m))

    print(f"Loading PTX: {args.ptx}")
    cd.init()
    mod = cd.load_ptx(args.ptx)
    fn  = cd.get_function(mod, "promoted_stencil_kernel")

    print("\n=== Correctness ===")
    ok = test_correctness(fn, shapes)

    if args.perf:
        perf_shapes = [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]
        print("\n=== Performance ===")
        test_performance(fn, perf_shapes)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
