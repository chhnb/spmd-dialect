#!/usr/bin/env python3
"""
run_reduction.py — Correctness and performance test for GPU reduction kernels.

Reduction: out = sum(A[0..N-1])

Atomic-only (default):
  bash scripts/gen-ptx.sh test/SPMD/lower-to-gpu-nvptx-reduction.mlir ewise /tmp/reduction.ptx
  python3 harness/run_reduction.py [--ptx /tmp/reduction.ptx] [--perf]

Hierarchical (shared-memory tree + single global atomic per block):
  bash scripts/gen-ptx.sh test/SPMD/lower-to-gpu-nvptx-hierarchical-reduction.mlir \\
       hierarchical /tmp/reduction_hierarchical.ptx
  python3 harness/run_reduction.py --hierarchical [--ptx /tmp/reduction_hierarchical.ptx]
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


# ── Hierarchical kernel ABI ────────────────────────────────────────────────────
#
# define ptx_kernel void @hierarchical_sum_kernel(
#   i64    param_0 = tile_size  (blockDim.x = compile-time constant)
#   i64    param_1 = N
#   i64    param_2 = 0          (safe-idx clamp constant)
#   ptr    param_3 = A.base_ptr
#   ptr    param_4 = A.aligned_ptr
#   i64    param_5 = A.offset = 0
#   i64    param_6 = A.size = N
#   i64    param_7 = A.stride = 1
#   float  param_8 = 0.0        (reduction identity)
#   i64    param_9 .. param_{9+log2(tile_size)-1} = tree strides (tile_size/2 .. 1)
#   ptr    param_{9+K}   = out.base_ptr
#   ptr    param_{10+K}  = out.aligned_ptr
#   i64    param_{11+K}  = out.offset = 0
# ) where K = log2(tile_size)
#
# Launch: grid=(⌈N/tile_size⌉, 1, 1), block=(tile_size, 1, 1)

def _tree_strides(tile_size: int) -> list:
    """Return tree reduction strides [tile_size/2, .., 1] for given blockDim."""
    strides, s = [], tile_size // 2
    while s >= 1:
        strides.append(s)
        s //= 2
    return strides


def run_hierarchical_gpu(fn, A: np.ndarray, tile_size: int) -> float:
    N = A.size
    A_c = np.ascontiguousarray(A, dtype=np.float32)

    A_d   = cd.alloc(A_c.nbytes)
    out_d = cd.alloc(4)

    cd.memcpy_h2d(A_d, A_c)
    cd.memset(out_d, 0, 4)

    grid    = (math.ceil(N / tile_size), 1, 1)
    strides = _tree_strides(tile_size)

    cd.launch(
        fn,
        grid, (tile_size, 1, 1),
        tile_size, N, 0,          # params 0-2: tile_size, N, c0
        A_d, A_d, 0, N, 1,        # params 3-7: A descriptor
        0.0,                      # param 8: identity (float)
        *strides,                 # params 9..(8+K): tree strides
        out_d, out_d, 0,          # params (9+K)..(11+K): out descriptor
    )
    cd.synchronize()

    result_np = np.zeros(1, dtype=np.float32)
    cd.memcpy_d2h(result_np, out_d)

    A_d.free()
    out_d.free()
    return float(result_np[0])


def _check_result(label, gpu, ref):
    rel_err = abs(gpu - ref) / max(abs(ref), 1e-6)
    ok = rel_err < 1e-3
    print(f"{label:>20}  {gpu:>14.6f}  {ref:>14.6f}  {rel_err:>10.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def test_correctness(fn, sizes, tile_size: int):
    # RandomState is used here (and in run_host.py) so that all backends generate
    # the same deterministic inputs with the same seed, enabling true differential
    # comparison across CPU serial, OpenMP, and GPU backends.
    rng = np.random.RandomState(0)
    print(f"{'N':>20}  {'gpu_sum':>14}  {'ref_sum':>14}  {'rel_err':>10}  result")
    all_pass = True
    for N in sizes:
        A     = rng.random_sample(N).astype(np.float32)
        ref   = float(np.sum(A))
        gpu   = run_sum_gpu(fn, A, tile_size)
        if not _check_result(str(N), gpu, ref):
            all_pass = False
    return all_pass


# AC-6 sizes: exact powers of two, boundary cases, non-multiples of tile_size.
_HIERARCHICAL_SIZES = [1, 32, 33, 255, 256, 257, 1000, 1024, 65536, 65537,
                       1048576, 16777216]


def test_hierarchical_correctness(fn, sizes, tile_size: int):
    """
    Extended correctness suite for the hierarchical kernel (AC-6).

    Covers:
      - Variable sizes (including non-multiples and large arrays)
      - All-zeros input  (expected sum = 0.0)
      - All-ones input   (expected sum = N)
      - 3× multi-launch  (same kernel invoked three times; all must agree)
    """
    rng = np.random.RandomState(0)
    print(f"{'case':>20}  {'gpu_sum':>14}  {'ref_sum':>14}  {'rel_err':>10}  result")
    all_pass = True

    # Variable-size random inputs
    for N in sizes:
        A   = rng.random_sample(N).astype(np.float32)
        ref = float(np.sum(A))
        gpu = run_hierarchical_gpu(fn, A, tile_size)
        if not _check_result(str(N), gpu, ref):
            all_pass = False

    # All-zeros
    for N in [256, 1024]:
        A   = np.zeros(N, dtype=np.float32)
        ref = 0.0
        gpu = run_hierarchical_gpu(fn, A, tile_size)
        if not _check_result(f"zeros-{N}", gpu, ref):
            all_pass = False

    # All-ones
    for N in [256, 1024]:
        A   = np.ones(N, dtype=np.float32)
        ref = float(N)
        gpu = run_hierarchical_gpu(fn, A, tile_size)
        if not _check_result(f"ones-{N}", gpu, ref):
            all_pass = False

    # 3× multi-launch: same array, same kernel, three independent launches.
    A   = rng.random_sample(65536).astype(np.float32)
    ref = float(np.sum(A))
    for rep in range(3):
        gpu = run_hierarchical_gpu(fn, A, tile_size)
        if not _check_result(f"multi-{rep+1}/3", gpu, ref):
            all_pass = False

    return all_pass


def test_hierarchical_negative(fn, tile_size: int) -> bool:
    """
    Negative correctness test for the hierarchical kernel.

    Verifies that the harness correctly detects a wrong result when the
    output accumulator is not zero-initialized before launch.  Runs the
    kernel twice on the same input without clearing the accumulator between
    launches: the second launch accumulates on top of the first, producing
    approximately 2× the true sum.  The relative error vs the true reference
    must exceed the 1e-3 threshold, proving the harness can distinguish
    correct from incorrect output.

    Returns True (test behaved as expected) if the wrong result was detected,
    False if the harness accidentally reported the wrong result as correct.
    """
    N = 1024
    rng = np.random.RandomState(99)
    A   = rng.random_sample(N).astype(np.float32)
    ref = float(np.sum(A))

    A_c   = np.ascontiguousarray(A, dtype=np.float32)
    A_d   = cd.alloc(A_c.nbytes)
    out_d = cd.alloc(4)
    cd.memcpy_h2d(A_d, A_c)

    strides = _tree_strides(tile_size)
    grid    = (math.ceil(N / tile_size), 1, 1)

    def _launch():
        cd.launch(fn, grid, (tile_size, 1, 1),
                  tile_size, N, 0,
                  A_d, A_d, 0, N, 1,
                  0.0, *strides,
                  out_d, out_d, 0)

    # First launch with zero-initialized accumulator (correct result).
    cd.memset(out_d, 0, 4)
    _launch()
    # Second launch WITHOUT re-zeroing: accumulator now holds sum + sum ≈ 2×ref.
    _launch()
    cd.synchronize()

    result_np = np.zeros(1, dtype=np.float32)
    cd.memcpy_d2h(result_np, out_d)

    A_d.free()
    out_d.free()

    gpu     = float(result_np[0])
    rel_err = abs(gpu - ref) / max(abs(ref), 1e-6)
    # The harness correctly detected the wrong result iff rel_err >= 1e-3.
    detected = rel_err >= 1e-3
    print(f"{'negative-test':>20}  {gpu:>14.6f}  {ref:>14.6f}  {rel_err:>10.2e}  "
          f"{'PASS' if detected else 'FAIL'}")
    return detected


def test_performance_hierarchical(fn, sizes, tile_size: int, repeats=10):
    rng = np.random.default_rng(1)
    print(f"\n{'N':>12}  {'cpu_ms':>8}  {'gpu_ms':>8}  {'speedup':>8}")
    for N in sizes:
        A   = rng.random(N, dtype=np.float32)
        A_c = np.ascontiguousarray(A, dtype=np.float32)
        A_d   = cd.alloc(A_c.nbytes)
        out_d = cd.alloc(4)
        cd.memcpy_h2d(A_d, A_c)

        strides = _tree_strides(tile_size)
        grid    = (math.ceil(N / tile_size), 1, 1)

        def launch_once():
            cd.memset(out_d, 0, 4)
            cd.launch(fn, grid, (tile_size, 1, 1),
                      tile_size, N, 0,
                      A_d, A_d, 0, N, 1,
                      0.0,
                      *strides,
                      out_d, out_d, 0)

        for _ in range(3):
            launch_once()
        cd.synchronize()

        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = np.sum(A)
        t_cpu = (time.perf_counter() - t0) / repeats * 1e3

        t0 = time.perf_counter()
        for _ in range(repeats):
            launch_once()
        cd.synchronize()
        t_gpu = (time.perf_counter() - t0) / repeats * 1e3

        A_d.free()
        out_d.free()
        print(f"{N:>12}  {t_cpu:>8.3f}  {t_gpu:>8.3f}  {t_cpu/t_gpu:>8.1f}x")


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
    ap.add_argument("--ptx",   default="")
    ap.add_argument("--hierarchical", action="store_true",
                    help="Test hierarchical kernel (shared-memory tree + single atomic). "
                         "Defaults to /tmp/reduction_hierarchical.ptx.")
    ap.add_argument("--sizes", default="",
                    help="Comma-separated correctness sizes. "
                         "Default: 1024,65536,1048576 (atomic) or AC-6 suite (hierarchical).")
    ap.add_argument("--perf",  action="store_true")
    ap.add_argument("--perf-sizes", default="",
                    help="Sizes to use for performance timing (default: same as --sizes).")
    ap.add_argument("--tile-size", type=int, default=_DEFAULT_TILE_SIZE,
                    help="Tile size baked into the PTX (blockDim.x). "
                         "Must match the spmd.tile_sizes used when compiling the PTX.")
    args = ap.parse_args()

    tile_size = args.tile_size

    if args.hierarchical:
        ptx_path    = args.ptx or "/tmp/reduction_hierarchical.ptx"
        kernel_name = "hierarchical_sum_kernel"
        sizes_str   = args.sizes or ",".join(str(s) for s in _HIERARCHICAL_SIZES)
    else:
        ptx_path    = args.ptx or "/tmp/reduction.ptx"
        kernel_name = "atomic_sum_kernel"
        sizes_str   = args.sizes or "1024,65536,1048576"

    sizes = [int(s) for s in sizes_str.split(",")]
    perf_sizes_str = args.perf_sizes if args.perf_sizes else sizes_str
    perf_sizes = [int(s) for s in perf_sizes_str.split(",")]

    print(f"Loading PTX: {ptx_path}")
    cd.init()
    mod = cd.load_ptx(ptx_path)
    fn  = cd.get_function(mod, kernel_name)

    print("\n=== Correctness ===")
    if args.hierarchical:
        ok = test_hierarchical_correctness(fn, sizes, tile_size)
        print("\n=== Negative test (uninitialized accumulator must FAIL) ===")
        neg_ok = test_hierarchical_negative(fn, tile_size)
        if not neg_ok:
            print("ERROR: negative test did not detect wrong result — harness is broken")
            ok = False
    else:
        ok = test_correctness(fn, sizes, tile_size)

    if args.perf:
        print("\n=== Performance ===")
        if args.hierarchical:
            test_performance_hierarchical(fn, perf_sizes, tile_size)
        else:
            test_performance(fn, perf_sizes, tile_size)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
