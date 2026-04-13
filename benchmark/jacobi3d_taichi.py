"""C2: Jacobi 3D 7-point stencil — Taichi implementation.
2 kernels/step: jacobi3d_step + copy_back.
"""

import taichi as ti
import numpy as np


def run(N, steps=100, backend="cuda"):
    if backend == "cuda":
        ti.init(arch=ti.cuda, default_fp=ti.f32)
    else:
        ti.init(arch=ti.cpu, default_fp=ti.f32)

    u = ti.field(dtype=ti.f32, shape=(N, N, N))
    u_new = ti.field(dtype=ti.f32, shape=(N, N, N))

    @ti.kernel
    def jacobi3d_step():
        for i, j, k in ti.ndrange((1, N - 1), (1, N - 1), (1, N - 1)):
            u_new[i, j, k] = (
                u[i - 1, j, k] + u[i + 1, j, k] +
                u[i, j - 1, k] + u[i, j + 1, k] +
                u[i, j, k - 1] + u[i, j, k + 1]
            ) / 6.0

    @ti.kernel
    def copy_back():
        for i, j, k in u:
            u[i, j, k] = u_new[i, j, k]

    @ti.kernel
    def init_bc():
        for i, j in ti.ndrange(N, N):
            u[i, j, 0] = 1.0

    init_bc()

    def step_fn():
        for _ in range(steps):
            jacobi3d_step()  # kernel 1
            copy_back()      # kernel 2

    def sync_fn():
        ti.sync()

    # warmup
    step_fn()
    sync_fn()

    # Re-initialize state after warmup so correctness harness sees fresh data
    u.fill(0.0)
    u_new.fill(0.0)
    init_bc()
    ti.sync()

    return step_fn, sync_fn, u
