#!/usr/bin/env python3
"""
run_host.py — Host (CPU/OpenMP) kernel runner for differential correctness testing.

Loads a compiled MLIR kernel shared library via ctypes and runs it against
the numpy reference for ewise, stencil, or reduction kernels.

The function ABIs match MLIR's default memref-to-LLVM lowering (descriptor
unpacking): each memref<?xf32> becomes (base_ptr, aligned_ptr, offset, size,
stride) and each memref<?x?xf32> becomes (base_ptr, aligned_ptr, offset, size0,
size1, stride0, stride1). Index values become int64.

Usage:
  python3 harness/run_host.py --lib /tmp/ewise_scf.so \\
      --kernel ewise --sizes 1024
  python3 harness/run_host.py --lib /tmp/stencil_scf.so \\
      --kernel stencil --shapes 128x128
  python3 harness/run_host.py --lib /tmp/reduction_scf.so \\
      --kernel reduction --sizes 65536

Output format matches the GPU harnesses (PASS/FAIL lines with numeric metrics).
"""

import argparse
import ctypes
import sys

import numpy as np

# ── ABI helpers ───────────────────────────────────────────────────────────────

c_p = ctypes.c_void_p
c_i = ctypes.c_int64


def _ptr(arr: np.ndarray) -> ctypes.c_void_p:
    """Return a ctypes void pointer to the first element of a numpy array."""
    return arr.ctypes.data_as(ctypes.c_void_p)


def _desc1d(arr: np.ndarray, n: int):
    """
    Unrolled 1D memref<?xf32> descriptor.
    Returns (base, aligned, offset, size, stride) as ctypes values.
    """
    p = _ptr(arr)
    return (p, p, c_i(0), c_i(n), c_i(1))


def _desc2d(arr: np.ndarray, n: int, m: int, stride_row: int):
    """
    Unrolled 2D memref<?x?xf32> descriptor (row-major).
    Returns (base, aligned, offset, size0, size1, stride0, stride1) as ctypes values.
    """
    p = _ptr(arr)
    return (p, p, c_i(0), c_i(n), c_i(m), c_i(stride_row), c_i(1))


def _desc0d(arr: np.ndarray):
    """
    Unrolled rank-0 memref<f32> descriptor.
    Returns (base, aligned, offset) as ctypes values.
    """
    p = _ptr(arr)
    return (p, p, c_i(0))


def _load(lib_path: str, fn_name: str, argtypes: list):
    """Load a function from a shared library with the given ctypes argtypes."""
    lib = ctypes.CDLL(lib_path)
    fn = getattr(lib, fn_name)
    fn.restype = None
    fn.argtypes = argtypes
    return fn


# ── Ewise kernel ──────────────────────────────────────────────────────────────
# Source: test/SPMD/lower-to-gpu-nvptx.mlir → @ewise
# Semantics: C[i] = A[i] + B[i]
#
# LLVM ABI (16 args):
#   void ewise(A×5, B×5, C×5, N)
#   where ×5 = (base_ptr, aligned_ptr, offset, size, stride) for 1D memref
_EWISE_ARGTYPES = (
    [c_p, c_p, c_i, c_i, c_i] +   # A descriptor
    [c_p, c_p, c_i, c_i, c_i] +   # B descriptor
    [c_p, c_p, c_i, c_i, c_i] +   # C descriptor
    [c_i]                           # N (loop bound)
)


def run_ewise(lib_path: str, sizes: list) -> bool:
    fn  = _load(lib_path, "ewise", _EWISE_ARGTYPES)
    rng = np.random.RandomState(42)
    all_pass = True
    print(f"{'N':>12}  {'max_err':>10}  result")
    for N in sizes:
        A = rng.random_sample(N).astype(np.float32)
        B = rng.random_sample(N).astype(np.float32)
        C = np.zeros(N, dtype=np.float32)
        fn(*_desc1d(A, N), *_desc1d(B, N), *_desc1d(C, N), c_i(N))
        ref     = A + B
        max_err = float(np.max(np.abs(C - ref)))
        ok      = max_err < 1e-5
        if not ok:
            all_pass = False
        print(f"{N:>12}  {max_err:>10.2e}  {'PASS' if ok else 'FAIL'}")
    return all_pass


# ── Stencil kernel ────────────────────────────────────────────────────────────
# Source: test/SPMD/differential-stencil.mlir → @stencil_cpu
# Semantics: B[i,j] = A[i,j] + A[i,j+1] + A[i+1,j]
#   A has shape (N+1, M+1); forall runs over [0,N) × [0,M)
#
# LLVM ABI (16 args):
#   void stencil_cpu(A×7, B×7, N, M)
#   where ×7 = (base, aligned, offset, size0, size1, stride0, stride1) for 2D memref
_STENCIL_ARGTYPES = (
    [c_p, c_p, c_i, c_i, c_i, c_i, c_i] +   # A descriptor (N+1, M+1)
    [c_p, c_p, c_i, c_i, c_i, c_i, c_i] +   # B descriptor (N, M)
    [c_i, c_i]                                 # N, M (loop bounds)
)


def run_stencil(lib_path: str, shapes: list) -> bool:
    fn  = _load(lib_path, "stencil_cpu", _STENCIL_ARGTYPES)
    rng = np.random.RandomState(0)
    all_pass = True
    print(f"{'shape':>14}  {'max_err':>10}  result")
    for (N, M) in shapes:
        # A is (N+1, M+1) so the kernel can access A[i,j+1] and A[i+1,j] for all i<N, j<M
        A = rng.random_sample((N + 1, M + 1)).astype(np.float32)
        B = np.zeros((N, M), dtype=np.float32)
        fn(*_desc2d(A, N + 1, M + 1, M + 1),
           *_desc2d(B, N, M, M),
           c_i(N), c_i(M))
        # Numpy reference: same as run_promoted_stencil.py's cpu_reference()
        B_ref = np.zeros((N, M), dtype=np.float32)
        B_ref[:N-1, :M-1] = A[:N-1, :M-1] + A[:N-1, 1:M] + A[1:N, :M-1]
        # Compare interior only (last row/col boundary values differ between host/GPU)
        interior = (slice(0, N - 1), slice(0, M - 1))
        max_err  = float(np.max(np.abs(B[interior] - B_ref[interior])))
        ok       = max_err < 1e-4
        if not ok:
            all_pass = False
        print(f"({N:4d},{M:4d})   {max_err:>10.2e}  {'PASS' if ok else 'FAIL'}")
    return all_pass


# ── Reduction kernel ──────────────────────────────────────────────────────────
# Source: test/SPMD/lower-to-gpu-nvptx-reduction.mlir → @atomic_sum
# Semantics: out += sum(A[0..N-1])   (atomic float add per element)
#   Caller must zero-initialize out before each call.
#
# LLVM ABI (9 args):
#   void atomic_sum(A×5, out×3, N)
#   A×5  = (base, aligned, offset, size, stride)  — 1D memref<?xf32>
#   out×3 = (base, aligned, offset)               — rank-0 memref<f32>
_REDUCTION_ARGTYPES = (
    [c_p, c_p, c_i, c_i, c_i] +   # A descriptor
    [c_p, c_p, c_i] +              # out descriptor (rank-0)
    [c_i]                           # N (loop bound)
)


def run_reduction(lib_path: str, sizes: list) -> bool:
    fn  = _load(lib_path, "atomic_sum", _REDUCTION_ARGTYPES)
    rng = np.random.RandomState(0)
    all_pass = True
    print(f"{'N':>12}  {'cpu_sum':>14}  {'ref_sum':>14}  {'rel_err':>10}  result")
    for N in sizes:
        A   = rng.random_sample(N).astype(np.float32)
        out = np.zeros(1, dtype=np.float32)
        fn(*_desc1d(A, N), *_desc0d(out), c_i(N))
        ref     = float(np.sum(A))
        rel_err = abs(float(out[0]) - ref) / max(abs(ref), 1e-6)
        ok      = rel_err < 1e-3
        if not ok:
            all_pass = False
        print(f"{N:>12}  {float(out[0]):>14.6f}  {ref:>14.6f}  {rel_err:>10.2e}  "
              f"{'PASS' if ok else 'FAIL'}")
    return all_pass


# ── Hierarchical reduction kernel ─────────────────────────────────────────────
# Source: test/SPMD/lower-to-gpu-nvptx-hierarchical-reduction.mlir → @hierarchical_sum
# Semantics: out += sum(A[0..N-1])  — same as atomic_sum but different function name.
# When compiled through the SCF host pipeline, spmd.reduce lowers to a serial scf.for;
# the result is numerically identical to the GPU hierarchical path.
#
# LLVM ABI (9 args, identical layout to atomic_sum):
#   void hierarchical_sum(A×5, out×3, N)

def run_reduction_hierarchical(lib_path: str, sizes: list) -> bool:
    fn  = _load(lib_path, "hierarchical_sum", _REDUCTION_ARGTYPES)
    rng = np.random.RandomState(0)
    all_pass = True
    print(f"{'N':>12}  {'cpu_sum':>14}  {'ref_sum':>14}  {'rel_err':>10}  result")
    for N in sizes:
        A   = rng.random_sample(N).astype(np.float32)
        out = np.zeros(1, dtype=np.float32)
        fn(*_desc1d(A, N), *_desc0d(out), c_i(N))
        ref     = float(np.sum(A))
        rel_err = abs(float(out[0]) - ref) / max(abs(ref), 1e-6)
        ok      = rel_err < 1e-3
        if not ok:
            all_pass = False
        print(f"{N:>12}  {float(out[0]):>14.6f}  {ref:>14.6f}  {rel_err:>10.2e}  "
              f"{'PASS' if ok else 'FAIL'}")
    return all_pass


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Run a compiled MLIR kernel shared library against the numpy reference."
    )
    ap.add_argument("--lib",    required=True,
                    help="Path to the compiled shared library (.so)")
    ap.add_argument("--kernel", required=True,
                    choices=["ewise", "stencil", "reduction", "reduction_hierarchical"],
                    help="Which kernel to run")
    ap.add_argument("--sizes",  default="",
                    help="Comma-separated sizes N (ewise, reduction)")
    ap.add_argument("--shapes", default="",
                    help="Comma-separated NxM shapes (stencil)")
    args = ap.parse_args()

    ok = False
    if args.kernel == "ewise":
        sizes = [int(x) for x in args.sizes.split(",") if x]
        print("\n=== Correctness ===")
        ok = run_ewise(args.lib, sizes)
    elif args.kernel == "stencil":
        shapes = [tuple(int(v) for v in s.split("x"))
                  for s in args.shapes.split(",") if s]
        print("\n=== Correctness ===")
        ok = run_stencil(args.lib, shapes)
    elif args.kernel == "reduction":
        sizes = [int(x) for x in args.sizes.split(",") if x]
        print("\n=== Correctness ===")
        ok = run_reduction(args.lib, sizes)
    elif args.kernel == "reduction_hierarchical":
        sizes = [int(x) for x in args.sizes.split(",") if x]
        print("\n=== Correctness ===")
        ok = run_reduction_hierarchical(args.lib, sizes)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
