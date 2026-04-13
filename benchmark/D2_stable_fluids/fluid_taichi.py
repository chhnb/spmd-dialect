"""Stable Fluids — Taichi.
Adapted from taichi/examples/simulation/stable_fluid.py
"""
import taichi as ti

def run(N, steps=1, jacobi_iters=50, backend="cuda"):
    ti.init(arch=ti.cuda if backend == "cuda" else ti.cpu, default_fp=ti.f32)
    dt = 0.03

    u = ti.Vector.field(2, dtype=ti.f32, shape=(N, N))  # velocity
    u_new = ti.Vector.field(2, dtype=ti.f32, shape=(N, N))
    p = ti.field(dtype=ti.f32, shape=(N, N))  # pressure
    p_new = ti.field(dtype=ti.f32, shape=(N, N))
    div = ti.field(dtype=ti.f32, shape=(N, N))  # divergence

    @ti.func
    def sample_bilinear(field: ti.template(), x: float, y: float) -> ti.math.vec2:
        # Bilinear interpolation (standard Stam 1999 semi-Lagrangian)
        x = ti.max(0.0, ti.min(x, ti.cast(N - 1, ti.f32)))
        y = ti.max(0.0, ti.min(y, ti.cast(N - 1, ti.f32)))
        i0 = ti.cast(ti.floor(x), ti.i32)
        j0 = ti.cast(ti.floor(y), ti.i32)
        i1 = ti.min(i0 + 1, N - 1)
        j1 = ti.min(j0 + 1, N - 1)
        sx = x - ti.cast(i0, ti.f32)
        sy = y - ti.cast(j0, ti.f32)
        return (1-sx)*(1-sy)*field[i0,j0] + sx*(1-sy)*field[i1,j0] + (1-sx)*sy*field[i0,j1] + sx*sy*field[i1,j1]

    @ti.kernel
    def advect():
        for i, j in u:
            coord = ti.Vector([float(i), float(j)]) - dt * N * u[i, j]
            u_new[i, j] = sample_bilinear(u, coord[0], coord[1])

    @ti.kernel
    def divergence_step():
        for i, j in u_new:
            il = ti.max(i - 1, 0); ir = ti.min(i + 1, N - 1)
            jl = ti.max(j - 1, 0); jr = ti.min(j + 1, N - 1)
            div[i, j] = 0.5 * (u_new[ir, j][0] - u_new[il, j][0] + u_new[i, jr][1] - u_new[i, jl][1])

    @ti.kernel
    def pressure_jacobi():
        for i, j in p:
            pl = sample_v(ti.Vector.field(2, dtype=ti.f32, shape=(1,1)), 0, 0)[0]  # placeholder
            # Direct field access with boundary clamp
            il = ti.max(i - 1, 0); ir = ti.min(i + 1, N - 1)
            jl = ti.max(j - 1, 0); jr = ti.min(j + 1, N - 1)
            p_new[i, j] = 0.25 * (p[il, j] + p[ir, j] + p[i, jl] + p[i, jr] - div[i, j])

    @ti.kernel
    def pressure_jacobi_step():
        for i, j in p:
            il = ti.max(i - 1, 0); ir = ti.min(i + 1, N - 1)
            jl = ti.max(j - 1, 0); jr = ti.min(j + 1, N - 1)
            p_new[i, j] = 0.25 * (p[il, j] + p[ir, j] + p[i, jl] + p[i, jr] - div[i, j])

    @ti.kernel
    def copy_pressure():
        for i, j in p:
            p[i, j] = p_new[i, j]

    @ti.kernel
    def pressure_project():
        for i, j in u_new:
            il = ti.max(i - 1, 0); ir = ti.min(i + 1, N - 1)
            jl = ti.max(j - 1, 0); jr = ti.min(j + 1, N - 1)
            u[i, j] = u_new[i, j] - 0.5 * ti.Vector([p[ir, j] - p[il, j], p[i, jr] - p[i, jl]])

    @ti.kernel
    def init():
        for i, j in u:
            # Initial vortex
            cx, cy = N * 0.5, N * 0.5
            dx_ = float(i) - cx
            dy_ = float(j) - cy
            u[i, j] = ti.Vector([-dy_, dx_]) * 0.01

    init()

    def step_fn():
        for _ in range(steps):
            advect()
            divergence_step()
            for _ in range(jacobi_iters):
                pressure_jacobi_step()
                copy_pressure()
            pressure_project()

    return step_fn, ti.sync, u
