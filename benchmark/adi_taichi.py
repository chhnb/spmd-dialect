"""C20: ADI (Alternating Direction Implicit) — Taichi implementation.
2*(N-2) kernel launches per step: one per row + one per column.
"""

import taichi as ti
import numpy as np


def run(N, steps=100, backend="cuda"):
    if backend == "cuda":
        ti.init(arch=ti.cuda, default_fp=ti.f32)
    else:
        ti.init(arch=ti.cpu, default_fp=ti.f32)

    u = ti.field(dtype=ti.f32, shape=(N, N))
    v = ti.field(dtype=ti.f32, shape=(N, N))

    a = 0.1
    b = 0.8

    @ti.kernel
    def adi_row(row: ti.i32):
        for j in range(1, N - 1):
            v[row, j] = a * u[row - 1, j] + b * u[row, j] + a * u[row + 1, j]

    @ti.kernel
    def adi_col(col: ti.i32):
        for i in range(1, N - 1):
            u[i, col] = a * v[i, col - 1] + b * v[i, col] + a * v[i, col + 1]

    # Init
    @ti.kernel
    def init_fields():
        for i, j in u:
            u[i, j] = ti.cast(i * (N - 1 - i) * j * (N - 1 - j), ti.f32) / ti.cast(N * N * N * N, ti.f32)

    init_fields()

    def step_fn():
        for _ in range(steps):
            for row in range(1, N - 1):   # N-2 row kernels
                adi_row(row)
            for col in range(1, N - 1):   # N-2 column kernels
                adi_col(col)

    def sync_fn():
        ti.sync()

    # warmup
    step_fn()
    sync_fn()

    return step_fn, sync_fn, u
