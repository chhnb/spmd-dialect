"""Global reduction (sum) — Taichi implementation.

Taichi's default reduction strategy: atomic add to a scalar field.
No hierarchical warp-shuffle reduction.
"""

import taichi as ti
import numpy as np


def run(N, backend="cuda"):
    if backend == "cuda":
        ti.init(arch=ti.cuda, default_fp=ti.f32)
    else:
        ti.init(arch=ti.cpu, default_fp=ti.f32)

    data = ti.field(dtype=ti.f32, shape=N)
    result = ti.field(dtype=ti.f32, shape=())

    @ti.kernel
    def fill_random():
        for i in data:
            data[i] = ti.random(dtype=ti.f32) - 0.5

    @ti.kernel
    def reduce_sum():
        for i in data:
            result[None] += data[i]  # atomic add fallback

    fill_random()

    def step():
        result[None] = 0.0
        reduce_sum()

    sync = ti.sync
    return step, sync, result
