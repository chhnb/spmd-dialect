"""1D electrostatic PIC — Taichi."""
import taichi as ti


def run(n_particles=16384, n_grid=512, steps=1, backend="cuda"):
    ti.init(arch=ti.cuda if backend == "cuda" else ti.cpu, default_fp=ti.f32)
    dt = 0.05
    dx = 1.0 / n_grid
    jacobi_iters = 30

    xp = ti.field(dtype=ti.f32, shape=n_particles)
    vp = ti.field(dtype=ti.f32, shape=n_particles)
    rho = ti.field(dtype=ti.f32, shape=n_grid)
    phi = ti.field(dtype=ti.f32, shape=n_grid)
    phi_new = ti.field(dtype=ti.f32, shape=n_grid)
    ex = ti.field(dtype=ti.f32, shape=n_grid)

    @ti.kernel
    def init():
        for p in xp:
            x = ti.random()
            xp[p] = x
            vp[p] = 0.2 if x < 0.5 else -0.2

    @ti.kernel
    def deposit():
        for i in rho:
            rho[i] = 0.0
        for p in xp:
            gx = xp[p] / dx
            i = ti.cast(ti.floor(gx), ti.i32)
            fx = gx - i
            i0 = (i + n_grid) % n_grid
            i1 = (i + 1) % n_grid
            ti.atomic_add(rho[i0], 1.0 - fx)
            ti.atomic_add(rho[i1], fx)
        mean_rho = 0.0
        for i in rho:
            mean_rho += rho[i]
        mean_rho /= n_grid
        for i in rho:
            rho[i] -= mean_rho
            phi[i] = 0.0

    @ti.kernel
    def poisson_jacobi():
        for i in phi:
            il = (i - 1 + n_grid) % n_grid
            ir = (i + 1) % n_grid
            phi_new[i] = 0.5 * (phi[il] + phi[ir] + dx * dx * rho[i])
        for i in phi:
            phi[i] = phi_new[i]

    @ti.kernel
    def electric_field():
        for i in ex:
            il = (i - 1 + n_grid) % n_grid
            ir = (i + 1) % n_grid
            ex[i] = -0.5 * (phi[ir] - phi[il]) / dx

    @ti.kernel
    def push():
        for p in xp:
            gx = xp[p] / dx
            i = ti.cast(ti.floor(gx), ti.i32)
            fx = gx - i
            i0 = (i + n_grid) % n_grid
            i1 = (i + 1) % n_grid
            e = (1.0 - fx) * ex[i0] + fx * ex[i1]
            vp[p] += dt * e
            x = xp[p] + dt * vp[p]
            if x < 0.0:
                x += 1.0
            if x >= 1.0:
                x -= 1.0
            xp[p] = x

    init()

    def step_fn():
        for _ in range(steps):
            deposit()
            for _ in range(jacobi_iters):
                poisson_jacobi()
            electric_field()
            push()

    return step_fn, ti.sync, xp
