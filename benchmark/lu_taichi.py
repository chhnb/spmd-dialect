"""C19: LU Decomposition — Taichi implementation.
Two kernels per pivot row: 2*(N-1) launches per step.
Matches CUDA launch pattern: factor_column + update_submatrix per pivot.
"""

import taichi as ti
import numpy as np


def run(N, steps=1, backend="cuda"):
    if backend == "cuda":
        ti.init(arch=ti.cuda, default_fp=ti.f32)
    else:
        ti.init(arch=ti.cpu, default_fp=ti.f32)

    A = ti.field(dtype=ti.f32, shape=(N, N))

    @ti.kernel
    def factor_column(k: ti.i32):
        """Kernel 1: compute L column (A[i,k] /= A[k,k])"""
        for i in range(k + 1, N):
            A[i, k] /= A[k, k]

    @ti.kernel
    def update_submatrix(k: ti.i32):
        """Kernel 2: update submatrix below and right of pivot"""
        for i, j in ti.ndrange((k + 1, N), (k + 1, N)):
            A[i, j] -= A[i, k] * A[k, j]

    # Init: diagonally dominant random-ish matrix for stability
    @ti.kernel
    def init_matrix():
        for i, j in A:
            A[i, j] = ti.cast((i * N + j) % 97, ti.f32) / 97.0 + 0.01
            if i == j:
                A[i, j] += ti.cast(N, ti.f32)

    init_matrix()

    def step_fn():
        for _ in range(steps):
            for k in range(N - 1):  # two kernels per pivot (matching CUDA)
                factor_column(k)
                update_submatrix(k)

    def sync_fn():
        ti.sync()

    # warmup
    step_fn()
    sync_fn()

    # Re-initialize state after warmup so correctness harness sees fresh data
    init_matrix()
    ti.sync()

    return step_fn, sync_fn, A
