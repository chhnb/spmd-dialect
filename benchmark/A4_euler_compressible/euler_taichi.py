"""Compressible Euler 2D (first-order finite volume, Rusanov flux) — Taichi."""
import taichi as ti

GAMMA = 1.4


def run(N=256, steps=1, backend="cuda"):
    ti.init(arch=ti.cuda if backend == "cuda" else ti.cpu, default_fp=ti.f32)
    dx = 1.0 / N
    dt = 0.12 * dx

    q = ti.Vector.field(4, dtype=ti.f32, shape=(N, N))
    q_new = ti.Vector.field(4, dtype=ti.f32, shape=(N, N))

    @ti.func
    def pressure(U):
        rho = ti.max(U[0], 1e-4)
        ux = U[1] / rho
        uy = U[2] / rho
        return ti.max((GAMMA - 1.0) * (U[3] - 0.5 * rho * (ux * ux + uy * uy)), 1e-4)

    @ti.func
    def flux_x(U):
        rho = ti.max(U[0], 1e-4)
        ux = U[1] / rho
        uy = U[2] / rho
        p = pressure(U)
        return ti.Vector([U[1], U[1] * ux + p, U[1] * uy, (U[3] + p) * ux])

    @ti.func
    def flux_y(U):
        rho = ti.max(U[0], 1e-4)
        ux = U[1] / rho
        uy = U[2] / rho
        p = pressure(U)
        return ti.Vector([U[2], U[2] * ux, U[2] * uy + p, (U[3] + p) * uy])

    @ti.func
    def wavespeed_x(U):
        rho = ti.max(U[0], 1e-4)
        ux = U[1] / rho
        c = ti.sqrt(GAMMA * pressure(U) / rho)
        return ti.abs(ux) + c

    @ti.func
    def wavespeed_y(U):
        rho = ti.max(U[0], 1e-4)
        uy = U[2] / rho
        c = ti.sqrt(GAMMA * pressure(U) / rho)
        return ti.abs(uy) + c

    @ti.func
    def rusanov_x(UL, UR):
        a = ti.max(wavespeed_x(UL), wavespeed_x(UR))
        return 0.5 * (flux_x(UL) + flux_x(UR)) - 0.5 * a * (UR - UL)

    @ti.func
    def rusanov_y(UL, UR):
        a = ti.max(wavespeed_y(UL), wavespeed_y(UR))
        return 0.5 * (flux_y(UL) + flux_y(UR)) - 0.5 * a * (UR - UL)

    @ti.kernel
    def init():
        for i, j in q:
            x = (i + 0.5) * dx
            y = (j + 0.5) * dx
            rho = 1.0
            p = 0.1
            if x < 0.5 and y < 0.5:
                rho = 1.5
                p = 1.5
            elif x >= 0.5 and y < 0.5:
                rho = 0.5323
                p = 0.3
            elif x < 0.5 and y >= 0.5:
                rho = 0.5323
                p = 0.3
            E = p / (GAMMA - 1.0)
            q[i, j] = ti.Vector([rho, 0.0, 0.0, E])

    @ti.kernel
    def step_kernel():
        for i, j in q:
            il = ti.max(i - 1, 0)
            ir = ti.min(i + 1, N - 1)
            jb = ti.max(j - 1, 0)
            jt = ti.min(j + 1, N - 1)
            Uc = q[i, j]
            Fl = rusanov_x(q[il, j], Uc)
            Fr = rusanov_x(Uc, q[ir, j])
            Gb = rusanov_y(q[i, jb], Uc)
            Gt = rusanov_y(Uc, q[i, jt])
            q_new[i, j] = Uc - (dt / dx) * (Fr - Fl + Gt - Gb)
        for i, j in q:
            q[i, j] = q_new[i, j]

    init()

    def step_fn():
        for _ in range(steps):
            step_kernel()

    return step_fn, ti.sync, q
