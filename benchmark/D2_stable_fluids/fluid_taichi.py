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
    def sample_v(field: ti.template(), x: float, y: float) -> ti.math.vec2:
        i = ti.max(0, ti.min(ti.cast(x, ti.i32), N - 1))
        j = ti.max(0, ti.min(ti.cast(y, ti.i32), N - 1))
        return field[i, j]

    @ti.kernel
    def advect():
        for i, j in u:
            coord = ti.Vector([float(i), float(j)]) - dt * N * u[i, j]
            u_new[i, j] = sample_v(u, coord[0], coord[1])

    @ti.kernel
    def divergence_step():
        for i, j in u_new:
            vl = sample_v(u_new, float(i - 1), float(j))[0]
            vr = sample_v(u_new, float(i + 1), float(j))[0]
            vb = sample_v(u_new, float(i), float(j - 1))[1]
            vt = sample_v(u_new, float(i), float(j + 1))[1]
            div[i, j] = 0.5 * (vr - vl + vt - vb)

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
