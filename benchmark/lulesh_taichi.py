"""C13: LULESH-like — Taichi implementation.
4 kernels/step: reset_forces, calc_forces, update_nodes, eos.
Simplified 2D Lagrangian hydro on structured grid.
"""

import taichi as ti
import numpy as np


def run(N, steps=100, backend="cuda"):
    if backend == "cuda":
        ti.init(arch=ti.cuda, default_fp=ti.f32)
    else:
        ti.init(arch=ti.cpu, default_fp=ti.f32)

    NE = N * N          # elements
    NN = (N + 1) * (N + 1)  # nodes
    dt = 0.0001

    # Element fields
    p_el = ti.field(dtype=ti.f32, shape=NE)
    vol = ti.field(dtype=ti.f32, shape=NE)
    rho = ti.field(dtype=ti.f32, shape=NE)
    e_el = ti.field(dtype=ti.f32, shape=NE)
    mass_el = ti.field(dtype=ti.f32, shape=NE)

    # Node fields
    xn_x = ti.field(dtype=ti.f32, shape=NN)
    xn_y = ti.field(dtype=ti.f32, shape=NN)
    vn_x = ti.field(dtype=ti.f32, shape=NN)
    vn_y = ti.field(dtype=ti.f32, shape=NN)
    fn_x = ti.field(dtype=ti.f32, shape=NN)
    fn_y = ti.field(dtype=ti.f32, shape=NN)
    mass_n = ti.field(dtype=ti.f32, shape=NN)

    @ti.kernel
    def reset_forces():
        for i in range(NN):
            fn_x[i] = 0.0
            fn_y[i] = 0.0

    @ti.kernel
    def calc_forces():
        for e in range(NE):
            ei = e // N
            ej = e % N
            n0 = ei * (N + 1) + ej
            n1 = (ei + 1) * (N + 1) + ej
            n2 = (ei + 1) * (N + 1) + ej + 1
            n3 = ei * (N + 1) + ej + 1
            pr = p_el[e] * vol[e] * 0.25
            ti.atomic_add(fn_x[n0], -pr)
            ti.atomic_add(fn_y[n0], -pr)
            ti.atomic_add(fn_x[n1], pr)
            ti.atomic_add(fn_y[n1], -pr)
            ti.atomic_add(fn_x[n2], pr)
            ti.atomic_add(fn_y[n2], pr)
            ti.atomic_add(fn_x[n3], -pr)
            ti.atomic_add(fn_y[n3], pr)

    @ti.kernel
    def update_nodes():
        for i in range(NN):
            if mass_n[i] > 0.0:
                vn_x[i] += fn_x[i] / mass_n[i] * dt
                vn_y[i] += fn_y[i] / mass_n[i] * dt
            xn_x[i] += vn_x[i] * dt
            xn_y[i] += vn_y[i] * dt

    @ti.kernel
    def eos():
        for e in range(NE):
            ei = e // N
            ej = e % N
            n0 = ei * (N + 1) + ej
            n1 = (ei + 1) * (N + 1) + ej
            n2 = (ei + 1) * (N + 1) + ej + 1
            n3 = ei * (N + 1) + ej + 1
            x0 = xn_x[n0]; y0 = xn_y[n0]
            x1 = xn_x[n1]; y1 = xn_y[n1]
            x2 = xn_x[n2]; y2 = xn_y[n2]
            x3 = xn_x[n3]; y3 = xn_y[n3]
            new_vol = 0.5 * ti.abs((x1 - x3) * (y2 - y0) - (x2 - x0) * (y1 - y3))
            new_vol = ti.max(new_vol, 1e-10)
            vol[e] = new_vol
            rho[e] = mass_el[e] / new_vol
            p_el[e] = 0.4 * rho[e] * e_el[e]

    # Init: uniform grid, Sedov-like blast in center element
    @ti.kernel
    def init_mesh():
        for idx in range(NN):
            i = idx // (N + 1)
            j = idx % (N + 1)
            xn_x[idx] = ti.cast(j, ti.f32) / ti.cast(N, ti.f32)
            xn_y[idx] = ti.cast(i, ti.f32) / ti.cast(N, ti.f32)
            vn_x[idx] = 0.0
            vn_y[idx] = 0.0
            mass_n[idx] = 1.0
        for e in range(NE):
            vol[e] = 1.0 / ti.cast(NE, ti.f32)
            rho[e] = 1.0
            mass_el[e] = rho[e] * vol[e]
            e_el[e] = 1e-6
            p_el[e] = 0.4 * rho[e] * e_el[e]
        # Sedov blast at center
        center = (N // 2) * N + N // 2
        e_el[center] = 1.0
        p_el[center] = 0.4 * 1.0 * 1.0

    init_mesh()

    def step_fn():
        for _ in range(steps):
            reset_forces()   # kernel 1
            calc_forces()     # kernel 2
            update_nodes()    # kernel 3
            eos()             # kernel 4

    def sync_fn():
        ti.sync()

    # warmup
    step_fn()
    sync_fn()

    return step_fn, sync_fn, p_el
