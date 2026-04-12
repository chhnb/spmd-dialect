"""C15: CG Solver — Taichi implementation.
5 kernels/step: matvec, dot1 (rr+pAp), update_xr, dot2 (rnew), update_p.
Tridiagonal matrix (2D Laplacian on sqrt(N) x sqrt(N) grid).
"""

import taichi as ti
import numpy as np
import math


def run(N, steps=100, backend="cuda"):
    if backend == "cuda":
        ti.init(arch=ti.cuda, default_fp=ti.f32)
    else:
        ti.init(arch=ti.cpu, default_fp=ti.f32)

    # Use N as total unknowns; grid side = isqrt(N)
    side = int(math.isqrt(N))
    N2 = side * side

    x = ti.field(dtype=ti.f32, shape=N2)
    r = ti.field(dtype=ti.f32, shape=N2)
    p = ti.field(dtype=ti.f32, shape=N2)
    Ap = ti.field(dtype=ti.f32, shape=N2)

    # Scalar reduction results
    d_rr = ti.field(dtype=ti.f32, shape=())
    d_pAp = ti.field(dtype=ti.f32, shape=())
    d_rnew = ti.field(dtype=ti.f32, shape=())

    # Host-side scalars for alpha/beta
    alpha_val = ti.field(dtype=ti.f32, shape=())
    beta_val = ti.field(dtype=ti.f32, shape=())

    @ti.kernel
    def matvec():
        """Ap = A * p (2D 5-point Laplacian)."""
        for idx in range(N2):
            i = idx // side
            j = idx % side
            s = 4.0 * p[idx]
            if i > 0:
                s -= p[idx - side]
            if i < side - 1:
                s -= p[idx + side]
            if j > 0:
                s -= p[idx - 1]
            if j < side - 1:
                s -= p[idx + 1]
            Ap[idx] = s

    @ti.kernel
    def dot1():
        """rr = dot(r,r), pAp = dot(p,Ap)."""
        d_rr[None] = 0.0
        d_pAp[None] = 0.0
        for i in range(N2):
            ti.atomic_add(d_rr[None], r[i] * r[i])
            ti.atomic_add(d_pAp[None], p[i] * Ap[i])

    @ti.kernel
    def update_xr():
        """x += alpha*p, r -= alpha*Ap."""
        a = alpha_val[None]
        for i in range(N2):
            x[i] += a * p[i]
            r[i] -= a * Ap[i]

    @ti.kernel
    def dot2():
        """rnew = dot(r,r)."""
        d_rnew[None] = 0.0
        for i in range(N2):
            ti.atomic_add(d_rnew[None], r[i] * r[i])

    @ti.kernel
    def update_p():
        """p = r + beta*p."""
        b = beta_val[None]
        for i in range(N2):
            p[i] = r[i] + b * p[i]

    # Init: b=1, x=0, r=b, p=r
    @ti.kernel
    def init_cg():
        for i in range(N2):
            x[i] = 0.0
            r[i] = 1.0
            p[i] = 1.0

    init_cg()

    def step_fn():
        for _ in range(steps):
            matvec()        # kernel 1
            dot1()          # kernel 2
            ti.sync()
            rr = d_rr[None]
            pAp = d_pAp[None]
            alpha_val[None] = rr / (pAp + 1e-20)
            update_xr()     # kernel 3
            dot2()          # kernel 4
            ti.sync()
            rnew = d_rnew[None]
            beta_val[None] = rnew / (rr + 1e-20)
            update_p()      # kernel 5

    def sync_fn():
        ti.sync()

    # warmup
    step_fn()
    sync_fn()

    return step_fn, sync_fn, x
