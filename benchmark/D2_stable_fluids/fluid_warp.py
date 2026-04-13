"""Stable Fluids — Warp.
Adapted from warp/examples/core/example_fluid.py
"""
import numpy as np
import warp as wp

@wp.func
def clamp_idx(v: int, lo: int, hi: int) -> int:
    return wp.max(lo, wp.min(v, hi))


@wp.struct
class FluidMesh:
    u: wp.array2d(dtype=wp.vec2)
    u_new: wp.array2d(dtype=wp.vec2)
    p: wp.array2d(dtype=float)
    p_new: wp.array2d(dtype=float)
    div: wp.array2d(dtype=float)
    N: int
    dt: float
    dt_N: float  # precomputed dt * N for exact Taichi match


@wp.kernel
def advect_kernel(m: FluidMesh):
    u = m.u
    u_new = m.u_new
    N = m.N
    dt_N = m.dt_N
    i, j = wp.tid()
    vel = u[i, j]
    # Bilinear interpolation matching Taichi (Stam 1999 semi-Lagrangian)
    x = wp.max(0.0, wp.min(float(i) - dt_N * vel[0], float(N - 1)))
    y = wp.max(0.0, wp.min(float(j) - dt_N * vel[1], float(N - 1)))
    i0 = int(wp.floor(x))
    j0 = int(wp.floor(y))
    i1 = wp.min(i0 + 1, N - 1)
    j1 = wp.min(j0 + 1, N - 1)
    sx = x - float(i0)
    sy = y - float(j0)
    u_new[i, j] = (1.0-sx)*(1.0-sy)*u[i0,j0] + sx*(1.0-sy)*u[i1,j0] + (1.0-sx)*sy*u[i0,j1] + sx*sy*u[i1,j1]

@wp.kernel
def divergence_kernel(m: FluidMesh):
    u = m.u_new
    div = m.div
    N = m.N
    i, j = wp.tid()
    il = clamp_idx(i - 1, 0, N - 1); ir = clamp_idx(i + 1, 0, N - 1)
    jl = clamp_idx(j - 1, 0, N - 1); jr = clamp_idx(j + 1, 0, N - 1)
    div[i, j] = 0.5 * (u[ir, j][0] - u[il, j][0] + u[i, jr][1] - u[i, jl][1])

@wp.kernel
def jacobi_kernel(m: FluidMesh):
    p = m.p
    p_new = m.p_new
    div = m.div
    N = m.N
    i, j = wp.tid()
    il = clamp_idx(i - 1, 0, N - 1); ir = clamp_idx(i + 1, 0, N - 1)
    jl = clamp_idx(j - 1, 0, N - 1); jr = clamp_idx(j + 1, 0, N - 1)
    p_new[i, j] = 0.25 * (p[il, j] + p[ir, j] + p[i, jl] + p[i, jr] - div[i, j])

@wp.kernel
def copy_kernel(m: FluidMesh):
    p = m.p
    p_new = m.p_new
    i, j = wp.tid()
    p[i, j] = p_new[i, j]

@wp.kernel
def project_kernel(m: FluidMesh):
    u_new = m.u_new
    u = m.u
    p = m.p
    N = m.N
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

    mesh = FluidMesh()
    mesh.u = wp.array(u_np, dtype=wp.vec2, device=backend)
    mesh.u_new = wp.zeros((N, N), dtype=wp.vec2, device=backend)
    mesh.p = wp.zeros((N, N), dtype=float, device=backend)
    mesh.p_new = wp.zeros((N, N), dtype=float, device=backend)
    mesh.div = wp.zeros((N, N), dtype=float, device=backend)
    mesh.N = N
    mesh.dt = dt
    mesh.dt_N = dt * N  # precomputed for exact Taichi match

    dim = (N, N)

    def step_fn():
        for _ in range(steps):
            wp.launch(advect_kernel, dim=dim, inputs=[mesh], device=backend)
            wp.launch(divergence_kernel, dim=dim, inputs=[mesh], device=backend)
            for _ in range(jacobi_iters):
                wp.launch(jacobi_kernel, dim=dim, inputs=[mesh], device=backend)
                wp.launch(copy_kernel, dim=dim, inputs=[mesh], device=backend)
            wp.launch(project_kernel, dim=dim, inputs=[mesh], device=backend)

    def sync():
        wp.synchronize_device(backend)

    return step_fn, sync, mesh.u
