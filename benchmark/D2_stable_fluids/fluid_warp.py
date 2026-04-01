"""Stable Fluids — Warp.
Adapted from warp/examples/core/example_fluid.py
"""
import numpy as np
import warp as wp

@wp.func
def clamp_idx(v: int, lo: int, hi: int) -> int:
    return wp.max(lo, wp.min(v, hi))

@wp.kernel
def advect_kernel(u: wp.array2d(dtype=wp.vec2), u_new: wp.array2d(dtype=wp.vec2),
                  N: int, dt: float):
    i, j = wp.tid()
    vel = u[i, j]
    x = float(i) - dt * float(N) * vel[0]
    y = float(j) - dt * float(N) * vel[1]
    si = clamp_idx(int(x), 0, N - 1)
    sj = clamp_idx(int(y), 0, N - 1)
    u_new[i, j] = u[si, sj]

@wp.kernel
def divergence_kernel(u: wp.array2d(dtype=wp.vec2), div: wp.array2d(dtype=float), N: int):
    i, j = wp.tid()
    il = clamp_idx(i - 1, 0, N - 1); ir = clamp_idx(i + 1, 0, N - 1)
    jl = clamp_idx(j - 1, 0, N - 1); jr = clamp_idx(j + 1, 0, N - 1)
    div[i, j] = 0.5 * (u[ir, j][0] - u[il, j][0] + u[i, jr][1] - u[i, jl][1])

@wp.kernel
def jacobi_kernel(p: wp.array2d(dtype=float), p_new: wp.array2d(dtype=float),
                  div: wp.array2d(dtype=float), N: int):
    i, j = wp.tid()
    il = clamp_idx(i - 1, 0, N - 1); ir = clamp_idx(i + 1, 0, N - 1)
    jl = clamp_idx(j - 1, 0, N - 1); jr = clamp_idx(j + 1, 0, N - 1)
    p_new[i, j] = 0.25 * (p[il, j] + p[ir, j] + p[i, jl] + p[i, jr] - div[i, j])

@wp.kernel
def copy_kernel(src: wp.array2d(dtype=float), dst: wp.array2d(dtype=float)):
    i, j = wp.tid()
    dst[i, j] = src[i, j]

@wp.kernel
def project_kernel(u_new: wp.array2d(dtype=wp.vec2), u: wp.array2d(dtype=wp.vec2),
                   p: wp.array2d(dtype=float), N: int):
    i, j = wp.tid()
    il = clamp_idx(i - 1, 0, N - 1); ir = clamp_idx(i + 1, 0, N - 1)
    jl = clamp_idx(j - 1, 0, N - 1); jr = clamp_idx(j + 1, 0, N - 1)
    u[i, j] = u_new[i, j] - 0.5 * wp.vec2(p[ir, j] - p[il, j], p[i, jr] - p[i, jl])

def run(N, steps=1, jacobi_iters=50, backend="cuda"):
    dt = 0.03
    # Init vortex
    u_np = np.zeros((N, N, 2), dtype=np.float32)
    cx, cy = N * 0.5, N * 0.5
    for i in range(N):
        for j in range(N):
            u_np[i, j, 0] = -(j - cy) * 0.01
            u_np[i, j, 1] = (i - cx) * 0.01

    u = wp.array(u_np, dtype=wp.vec2, device=backend)
    u_new = wp.zeros((N, N), dtype=wp.vec2, device=backend)
    p = wp.zeros((N, N), dtype=float, device=backend)
    p_new = wp.zeros((N, N), dtype=float, device=backend)
    div = wp.zeros((N, N), dtype=float, device=backend)

    dim = (N, N)

    def step_fn():
        for _ in range(steps):
            wp.launch(advect_kernel, dim=dim, inputs=[u, u_new, N, dt], device=backend)
            wp.launch(divergence_kernel, dim=dim, inputs=[u_new, div, N], device=backend)
            for _ in range(jacobi_iters):
                wp.launch(jacobi_kernel, dim=dim, inputs=[p, p_new, div, N], device=backend)
                wp.launch(copy_kernel, dim=dim, inputs=[p_new, p], device=backend)
            wp.launch(project_kernel, dim=dim, inputs=[u_new, u, p, N], device=backend)

    def sync():
        wp.synchronize_device(backend)

    return step_fn, sync, u
