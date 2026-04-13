"""C21: Gram-Schmidt orthogonalization — Taichi implementation.
N normalize kernels + N*(N-1)/2 project kernels = N + N*(N-1)/2 total launches.
"""

import taichi as ti
import numpy as np


def run(N, steps=1, backend="cuda"):
    if backend == "cuda":
        ti.init(arch=ti.cuda, default_fp=ti.f32)
    else:
        ti.init(arch=ti.cpu, default_fp=ti.f32)

    M = N  # M rows x N columns
    Q = ti.field(dtype=ti.f32, shape=(M, N))
    R = ti.field(dtype=ti.f32, shape=(N, N))

    # Temp scalar for dot product / norm
    tmp = ti.field(dtype=ti.f32, shape=())

    @ti.kernel
    def normalize(k: ti.i32):
        # Compute norm of column k (serial for cross-platform determinism — AC-6)
        s = ti.cast(0.0, ti.f32)
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
        # R[k,j] = dot(Q[:,k], Q[:,j]), then Q[:,j] -= R[k,j]*Q[:,k]
        # Serial for cross-platform determinism (AC-6)
        dot_val = ti.cast(0.0, ti.f32)
        ti.loop_config(serialize=True)
        for i in range(M):
            dot_val += Q[i, k] * Q[i, j]
        R[k, j] = dot_val
        ti.loop_config(serialize=True)
        for i in range(M):
            Q[i, j] -= dot_val * Q[i, k]

    # Init Q with random-ish values
    @ti.kernel
    def init_data():
        for i, j in Q:
            Q[i, j] = ti.cast((i * N + j) % 97, ti.f32) / 97.0 + 0.01

    init_data()

    def step_fn():
        for _ in range(steps):
            for k in range(N):
                normalize(k)                 # 1 kernel
                for j in range(k + 1, N):
                    project(k, j)            # 1 kernel per (k,j) pair

    def sync_fn():
        ti.sync()

    # warmup (for JIT compilation)
    step_fn()
    sync_fn()
    # Re-init Q to deterministic state after warmup
    init_data()
    sync_fn()

    return step_fn, sync_fn, Q
