"""C10: Gray-Scott reaction-diffusion — Taichi implementation.
2 kernels/step: gs_step + copy_back.
Du=0.16, Dv=0.08, F=0.06, k=0.062.
"""

import taichi as ti
import numpy as np


def run(N, steps=100, backend="cuda"):
    if backend == "cuda":
        ti.init(arch=ti.cuda, default_fp=ti.f32)
    else:
        ti.init(arch=ti.cpu, default_fp=ti.f32)

    gu = ti.field(dtype=ti.f32, shape=(N, N))
    gv = ti.field(dtype=ti.f32, shape=(N, N))
    gu2 = ti.field(dtype=ti.f32, shape=(N, N))
    gv2 = ti.field(dtype=ti.f32, shape=(N, N))

    Du = 0.16
    Dv = 0.08
    F = 0.06
    k = 0.062

    @ti.kernel
    def gs_step():
        for i, j in ti.ndrange((1, N - 1), (1, N - 1)):
            lu = gu[i - 1, j] + gu[i + 1, j] + gu[i, j - 1] + gu[i, j + 1] - 4.0 * gu[i, j]
            lv = gv[i - 1, j] + gv[i + 1, j] + gv[i, j - 1] + gv[i, j + 1] - 4.0 * gv[i, j]
            uvv = gu[i, j] * gv[i, j] * gv[i, j]
            gu2[i, j] = gu[i, j] + Du * lu - uvv + F * (1.0 - gu[i, j])
            gv2[i, j] = gv[i, j] + Dv * lv + uvv - (F + k) * gv[i, j]

    @ti.kernel
    def copy_back():
        for i, j in gu:
            gu[i, j] = gu2[i, j]
            gv[i, j] = gv2[i, j]

    # Init: u=1 everywhere, v=0.25 in center patch
    @ti.kernel
    def init_fields():
        for i, j in gu:
            gu[i, j] = 1.0
            gv[i, j] = 0.0
        for i, j in ti.ndrange((N // 2 - 10, N // 2 + 10), (N // 2 - 10, N // 2 + 10)):
            gv[i, j] = 0.25
            gu[i, j] = 0.5

    init_fields()

    def step_fn():
        for _ in range(steps):
            gs_step()     # kernel 1
            copy_back()   # kernel 2

    def sync_fn():
        ti.sync()

    # warmup
    step_fn()
    sync_fn()

    # Re-initialize state after warmup so correctness harness sees fresh data
    gu2.fill(0.0)
    gv2.fill(0.0)
    init_fields()
    ti.sync()

    return step_fn, sync_fn, gu
