"""C17: 3D Convolution — Taichi implementation.
One kernel per z-slice: (NZ-2) launches per step.
3x3x3 stencil with uniform weights (1/27).
"""

import taichi as ti
import numpy as np


def run(N, steps=1, backend="cuda"):
    if backend == "cuda":
        ti.init(arch=ti.cuda, default_fp=ti.f32)
    else:
        ti.init(arch=ti.cpu, default_fp=ti.f32)

    NX, NY, NZ = N, N, N
    A = ti.field(dtype=ti.f32, shape=(NX, NY, NZ))
    B = ti.field(dtype=ti.f32, shape=(NX, NY, NZ))

    @ti.kernel
    def conv3d_slice(z: ti.i32):
        for x, y in ti.ndrange((1, NX - 1), (1, NY - 1)):
            s = ti.cast(0.0, ti.f32)
            for dz, dy, dx in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
                s += A[x + dx, y + dy, z + dz]
            B[x, y, z] = s / 27.0

    # Init with synthetic data
    @ti.kernel
    def init_data():
        for i, j, k in A:
            A[i, j, k] = ti.sin(ti.cast(i * NY * NZ + j * NZ + k, ti.f32) * 0.001)

    init_data()

    @ti.kernel
    def copy_B_to_A():
        for i, j, k in A:
            A[i, j, k] = B[i, j, k]

    def step_fn():
        for _ in range(steps):
            for z in range(1, NZ - 1):   # one kernel per z-slice
                conv3d_slice(z)
            copy_B_to_A()  # propagate output to input for next step

    def sync_fn():
        ti.sync()

    # warmup
    step_fn()
    sync_fn()

    # Re-initialize state after warmup so correctness harness sees fresh data
    init_data()
    B.fill(0.0)
    ti.sync()

    return step_fn, sync_fn, B
