"""C19: LU Decomposition — Taichi implementation.
One kernel per pivot row: N-1 launches per step.
Each kernel computes L column and updates submatrix below pivot.
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
    def lu_pivot(k: ti.i32):
        # Phase 1: compute L column (A[i,k] /= A[k,k])
        for i in range(k + 1, N):
            A[i, k] /= A[k, k]
        # Phase 2: update submatrix
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
            for k in range(N - 1):  # one kernel per pivot
                lu_pivot(k)

    def sync_fn():
        ti.sync()

    # warmup
    step_fn()
    sync_fn()

    return step_fn, sync_fn, A
