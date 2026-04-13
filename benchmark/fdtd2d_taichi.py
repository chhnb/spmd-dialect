"""C11: FDTD 2D Maxwell — Taichi implementation.
3 kernels/step: update_ey, update_ex, update_hz (staggered Yee grid).
"""

import taichi as ti
import numpy as np


def run(N, steps=100, backend="cuda"):
    if backend == "cuda":
        ti.init(arch=ti.cuda, default_fp=ti.f32)
    else:
        ti.init(arch=ti.cpu, default_fp=ti.f32)

    Nx, Ny = N, N
    ex = ti.field(dtype=ti.f32, shape=(Nx, Ny))
    ey = ti.field(dtype=ti.f32, shape=(Nx, Ny))
    hz = ti.field(dtype=ti.f32, shape=(Nx, Ny))

    COURANT = 0.5

    @ti.kernel
    def update_ey():
        for i, j in ti.ndrange((1, Nx), (0, Ny)):
            ey[i, j] += COURANT * (hz[i, j] - hz[i - 1, j])

    @ti.kernel
    def update_ex():
        for i, j in ti.ndrange((0, Nx), (1, Ny)):
            ex[i, j] -= COURANT * (hz[i, j] - hz[i, j - 1])

    @ti.kernel
    def update_hz():
        for i, j in ti.ndrange((0, Nx - 1), (0, Ny - 1)):
            hz[i, j] -= COURANT * (
                ex[i, j + 1] - ex[i, j] +
                ey[i + 1, j] - ey[i, j]
            )

    # Init: point source at center
    @ti.kernel
    def init_fields():
        for i, j in hz:
            hz[i, j] = 0.0
        hz[Nx // 2, Ny // 2] = 1.0

    init_fields()

    def step_fn():
        for _ in range(steps):
            update_ey()   # kernel 1
            update_ex()   # kernel 2
            update_hz()   # kernel 3

    def sync_fn():
        ti.sync()

    # warmup
    step_fn()
    sync_fn()

    # Re-initialize state after warmup so correctness harness sees fresh data
    ex.fill(0.0)
    ey.fill(0.0)
    init_fields()
    ti.sync()

    return step_fn, sync_fn, hz
