"""Cloth spring-mass system — Taichi."""
import taichi as ti


def run(N=64, steps=1, backend="cuda"):
    ti.init(arch=ti.cuda if backend == "cuda" else ti.cpu, default_fp=ti.f32)
    dt = 0.01
    spring_k = 1200.0
    damping = 0.995
    inv_mass = 1.0

    x = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))
    v = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))
    rest_x = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))

    @ti.kernel
    def init():
        for i, j in x:
            p = ti.Vector([i / (N - 1), 1.0, j / (N - 1)])
            x[i, j] = p
            rest_x[i, j] = p
            v[i, j] = ti.Vector([0.0, 0.0, 0.0])

    @ti.func
    def spring_force(i, j, ni, nj):
        d = x[ni, nj] - x[i, j]
        rest = rest_x[ni, nj] - rest_x[i, j]
        L = ti.max(d.norm(), 1e-6)
        L0 = rest.norm()
        return spring_k * (L - L0) * d / L

    @ti.kernel
    def substep():
        for i, j in x:
            if i == 0 and (j == 0 or j == N - 1):
                continue
            force = ti.Vector([0.0, -9.8, 0.0])
            if i > 0:
                force += spring_force(i, j, i - 1, j)
            if i + 1 < N:
                force += spring_force(i, j, i + 1, j)
            if j > 0:
                force += spring_force(i, j, i, j - 1)
            if j + 1 < N:
                force += spring_force(i, j, i, j + 1)
            if i > 0 and j > 0:
                force += 0.7 * spring_force(i, j, i - 1, j - 1)
            if i > 0 and j + 1 < N:
                force += 0.7 * spring_force(i, j, i - 1, j + 1)
            if i + 1 < N and j > 0:
                force += 0.7 * spring_force(i, j, i + 1, j - 1)
            if i + 1 < N and j + 1 < N:
                force += 0.7 * spring_force(i, j, i + 1, j + 1)
            v[i, j] = damping * (v[i, j] + dt * inv_mass * force)
        for i, j in x:
            if i == 0 and (j == 0 or j == N - 1):
                x[i, j] = rest_x[i, j]
                v[i, j] = ti.Vector([0.0, 0.0, 0.0])
            else:
                x[i, j] += dt * v[i, j]

    init()

    def step_fn():
        for _ in range(steps):
            substep()

    return step_fn, ti.sync, x
