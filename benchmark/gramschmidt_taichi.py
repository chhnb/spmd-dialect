"""C21: Gram-Schmidt orthogonalization — Taichi implementation.
N normalize kernels + N*(N-1)/2 project kernels = N + N*(N-1)/2 total launches.
Uses fp64 for cross-platform determinism (fp32 Gram-Schmidt diverges at step 1).
Init: identity + sin perturbation (full rank; original (i%97)/97 was rank-deficient).
Both changes applied in lockstep with gramschmidt_benchmark.cu and numpy_refs.py.
"""

import taichi as ti
import numpy as np


def run(N, steps=1, backend="cuda"):
    if backend == "cuda":
        ti.init(arch=ti.cuda, default_fp=ti.f64)
    else:
        ti.init(arch=ti.cpu, default_fp=ti.f64)

    M = N
    Q = ti.field(dtype=ti.f64, shape=(M, N))
    R = ti.field(dtype=ti.f64, shape=(N, N))

    @ti.kernel
    def normalize(k: ti.i32):
        s = ti.cast(0.0, ti.f64)
        ti.loop_config(serialize=True)
        for i in range(M):
            s += Q[i, k] * Q[i, k]
        nrm = ti.sqrt(s)
        R[k, k] = nrm
        inv_nrm = 1.0 / nrm
        ti.loop_config(serialize=True)
        for i in range(M):
            Q[i, k] *= inv_nrm

    @ti.kernel
    def project(k: ti.i32, j: ti.i32):
        dot_val = ti.cast(0.0, ti.f64)
        ti.loop_config(serialize=True)
        for i in range(M):
            dot_val += Q[i, k] * Q[i, j]
        R[k, j] = dot_val
        ti.loop_config(serialize=True)
        for i in range(M):
            Q[i, j] -= dot_val * Q[i, k]

    @ti.kernel
    def init_data():
        for i, j in Q:
            base = 1.0 if i == j else 0.0
            perturb = ti.sin(ti.cast(i * N + j + 1, ti.f64) * 0.1) * 0.3
            Q[i, j] = base + perturb

    init_data()

    def step_fn():
        for _ in range(steps):
            for k in range(N):
                normalize(k)
                for j in range(k + 1, N):
                    project(k, j)

    def sync_fn():
        ti.sync()

    step_fn()
    sync_fn()
    init_data()
    sync_fn()

    return step_fn, sync_fn, Q
