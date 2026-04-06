"""LBM D2Q9 — Taichi implementation.
Adapted from taichi/examples/simulation/karman_vortex_street.py
"""
import taichi as ti
import numpy as np

def run(nx, ny=None, steps=1, backend="cuda"):
    if ny is None:
        ny = nx // 2
    if backend == "cuda":
        ti.init(arch=ti.cuda, default_fp=ti.f32)
    else:
        ti.init(arch=ti.cpu, default_fp=ti.f32)

    inv_tau = 1.0 / 0.6

    e = ti.Vector.field(2, dtype=ti.i32, shape=9)
    w = ti.field(dtype=ti.f32, shape=9)
    rho = ti.field(dtype=ti.f32, shape=(nx, ny))
    vel = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))
    f_old = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))
    f_new = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))

    e_np = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]], dtype=np.int32)
    w_np = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float32)
    e.from_numpy(e_np)
    w.from_numpy(w_np)

    @ti.func
    def f_eq(i, j):
        usq = vel[i, j].dot(vel[i, j])
        result = ti.Vector.zero(ti.f32, 9)
        for k in ti.static(range(9)):
            eu_k = ti.cast(e[k][0], ti.f32) * vel[i, j][0] + ti.cast(e[k][1], ti.f32) * vel[i, j][1]
            result[k] = w[k] * rho[i, j] * (1.0 + 3.0 * eu_k + 4.5 * eu_k * eu_k - 1.5 * usq)
        return result

    @ti.kernel
    def init():
        for i, j in rho:
            rho[i, j] = 1.0
            vel[i, j] = ti.Vector([0.0, 0.0])
            for k in ti.static(range(9)):
                f_old[i, j][k] = w[k]

    @ti.kernel
    def collide_and_stream():
        for i, j in ti.ndrange((1, nx - 1), (1, ny - 1)):
            feq = f_eq(i, j)
            for k in ti.static(range(9)):
                ip = i - e[k][0]
                jp = j - e[k][1]
                if ip >= 0 and ip < nx and jp >= 0 and jp < ny:
                    f_new[i, j][k] = (1.0 - inv_tau) * f_old[ip, jp][k] + inv_tau * feq[k]

    @ti.kernel
    def update_macro():
        for i, j in ti.ndrange((1, nx - 1), (1, ny - 1)):
            rho[i, j] = 0.0
            vel[i, j] = ti.Vector([0.0, 0.0])
            for k in ti.static(range(9)):
                f_old[i, j][k] = f_new[i, j][k]
                rho[i, j] += f_new[i, j][k]
                vel[i, j] += ti.Vector([ti.cast(e[k][0], ti.f32), ti.cast(e[k][1], ti.f32)]) * f_new[i, j][k]
            vel[i, j] /= rho[i, j]

    init()

    def step_fn():
        for _ in range(steps):
            collide_and_stream()
            update_macro()

    return step_fn, ti.sync, rho
