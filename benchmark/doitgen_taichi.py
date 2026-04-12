"""C18: DOITGEN multi-dimensional contraction — Taichi implementation.
One kernel per r-index: NR launches per step.
A_out(p,q,r) = sum_s A(p,q,s) * C4(s,r).
"""

import taichi as ti
import numpy as np


def run(N, steps=1, backend="cuda"):
    if backend == "cuda":
        ti.init(arch=ti.cuda, default_fp=ti.f32)
    else:
        ti.init(arch=ti.cpu, default_fp=ti.f32)

    NP, NQ, NR = N, N, N
    A = ti.field(dtype=ti.f32, shape=(NP, NQ, NR))
    A_out = ti.field(dtype=ti.f32, shape=(NP, NQ, NR))
    C4 = ti.field(dtype=ti.f32, shape=(NR, NR))

    @ti.kernel
    def doitgen_slice(r: ti.i32):
        for p, q in ti.ndrange(NP, NQ):
            s = ti.cast(0.0, ti.f32)
            for ss in range(NR):
                s += A[p, q, ss] * C4[ss, r]
            A_out[p, q, r] = s

    # Init
    @ti.kernel
    def init_data():
        for p, q, r in A:
            A[p, q, r] = ti.cast((p * NQ + q) * NR + r, ti.f32) / ti.cast(NP * NQ * NR, ti.f32)
        for i, j in C4:
            C4[i, j] = ti.cast(i * NR + j, ti.f32) / ti.cast(NR * NR, ti.f32)

    init_data()

    @ti.kernel
    def copy_out_to_in():
        for p, q, r in A:
            A[p, q, r] = A_out[p, q, r]

    def step_fn():
        for _ in range(steps):
            for r in range(NR):  # one kernel per r-index
                doitgen_slice(r)
            copy_out_to_in()  # propagate output to input for next step

    def sync_fn():
        ti.sync()

    # warmup
    step_fn()
    sync_fn()

    return step_fn, sync_fn, A_out
