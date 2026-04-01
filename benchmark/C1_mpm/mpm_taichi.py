"""MPM (Material Point Method) — Taichi.
Adapted from taichi/examples/simulation/mpm88.py
"""
import taichi as ti
import numpy as np

def run(n_grid=128, n_particles=8192, steps=1, backend="cuda"):
    ti.init(arch=ti.cuda if backend == "cuda" else ti.cpu, default_fp=ti.f32)

    dx = 1.0 / n_grid
    inv_dx = float(n_grid)
    dt = 1e-4
    p_vol = (dx * 0.5) ** 2
    p_rho = 1.0
    p_mass = p_vol * p_rho
    E = 400.0

    x = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
    v = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
    C = ti.Matrix.field(2, 2, dtype=ti.f32, shape=n_particles)
    Jp = ti.field(dtype=ti.f32, shape=n_particles)
    grid_v = ti.Vector.field(2, dtype=ti.f32, shape=(n_grid, n_grid))
    grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))

    @ti.kernel
    def init():
        for i in range(n_particles):
            x[i] = ti.Vector([ti.random() * 0.2 + 0.3, ti.random() * 0.2 + 0.3])
            v[i] = ti.Vector([0.0, 0.0])
            Jp[i] = 1.0

    @ti.kernel
    def substep():
        # Reset grid
        for i, j in grid_m:
            grid_v[i, j] = ti.Vector([0.0, 0.0])
            grid_m[i, j] = 0.0

        # P2G
        for p in x:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            stress = -dt * p_vol * (Jp[p] - 1) * 4 * inv_dx * inv_dx * E
            affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass

        # Grid operations
        for i, j in grid_m:
            if grid_m[i, j] > 0:
                grid_v[i, j] = grid_v[i, j] / grid_m[i, j]
                grid_v[i, j][1] -= dt * 50  # gravity
                # Boundary
                if i < 3 and grid_v[i, j][0] < 0: grid_v[i, j][0] = 0
                if i > n_grid - 3 and grid_v[i, j][0] > 0: grid_v[i, j][0] = 0
                if j < 3 and grid_v[i, j][1] < 0: grid_v[i, j][1] = 0
                if j > n_grid - 3 and grid_v[i, j][1] > 0: grid_v[i, j][1] = 0

        # G2P
        for p in x:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(ti.f32, 2)
            new_C = ti.Matrix.zero(ti.f32, 2, 2)
            for i, j in ti.static(ti.ndrange(3, 3)):
                dpos = ti.Vector([i, j]).cast(float) - fx
                g_v = grid_v[base + ti.Vector([i, j])]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
            v[p] = new_v
            x[p] += dt * new_v
            Jp[p] *= 1 + dt * new_C.trace()
            C[p] = new_C

    init()

    def step_fn():
        for _ in range(steps):
            substep()

    return step_fn, ti.sync, x
