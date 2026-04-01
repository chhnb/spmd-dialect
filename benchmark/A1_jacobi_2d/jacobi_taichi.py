"""Jacobi 2D 5-point stencil — Taichi implementation."""

import taichi as ti
import numpy as np


def run(N, steps=1, backend="cuda"):
    if backend == "cuda":
        ti.init(arch=ti.cuda, default_fp=ti.f32)
    else:
        ti.init(arch=ti.cpu, default_fp=ti.f32)

    u = ti.field(dtype=ti.f32, shape=(N, N))
    u_new = ti.field(dtype=ti.f32, shape=(N, N))

    @ti.kernel
    def jacobi_step():
        for i, j in ti.ndrange((1, N - 1), (1, N - 1)):
            u_new[i, j] = 0.25 * (
                u[i - 1, j] + u[i + 1, j] +
                u[i, j - 1] + u[i, j + 1]
            )

    @ti.kernel
    def copy_back():
        for i, j in u:
            u[i, j] = u_new[i, j]

    @ti.kernel
    def init_bc():
        for j in range(N):
            u[0, j] = 1.0

    init_bc()

    def step():
        for _ in range(steps):
            jacobi_step()
            copy_back()

    sync = ti.sync

    return step, sync, u
