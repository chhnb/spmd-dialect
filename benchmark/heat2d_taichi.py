"""C3: Heat 2D — Taichi implementation.
2 kernels/step: heat_step + copy_back.
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

    alpha = 0.2

    @ti.kernel
    def heat_step():
        for i, j in ti.ndrange((1, N - 1), (1, N - 1)):
            v[i, j] = u[i, j] + alpha * (
                u[i - 1, j] + u[i + 1, j] +
                u[i, j - 1] + u[i, j + 1] - 4.0 * u[i, j]
            )

    @ti.kernel
    def copy_back():
        for i, j in u:
            u[i, j] = v[i, j]

    @ti.kernel
    def init_bc():
        for j in range(N):
            u[0, j] = 1.0

    init_bc()

    def step_fn():
        for _ in range(steps):
            heat_step()   # kernel 1
            copy_back()   # kernel 2

    def sync_fn():
        ti.sync()

    # warmup
    step_fn()
    sync_fn()

    return step_fn, sync_fn, u
