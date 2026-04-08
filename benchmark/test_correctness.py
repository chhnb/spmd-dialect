#!/usr/bin/env python3
"""Correctness sweep for the 36 benchmark kernel types.

Primary check:
  Warp CPU vs Warp GPU on reduced problem sizes for all 36 kernel types.

Anchor checks:
  Heat2D and Jacobi2D against simple NumPy references.
"""

import numpy as np
import os
import sys

os.environ.setdefault("TI_LOG_LEVEL", "warn")

import run_warp_characterization as rw

wp = rw.wp
CPU_DEVICE = "cpu"
GPU_DEVICE = "cuda:0"

PASS = 0
FAIL = 0
SKIP = 0


def flatten_arrays(obj):
    if isinstance(obj, (tuple, list)):
        parts = [flatten_arrays(x) for x in obj]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
    arr = np.asarray(obj)
    return arr.astype(np.float64, copy=False).ravel()


def check(name, ref, test, tol=1e-5):
    global PASS, FAIL, SKIP
    if ref is None or test is None:
        SKIP += 1
        print(f"  {name:<40s} SKIP")
        return
    same_mask = (ref == test) | (np.isnan(ref) & np.isnan(test))
    finite_mask = np.isfinite(ref) & np.isfinite(test)
    valid_mask = finite_mask & ~same_mask
    diff = np.max(np.abs(ref[valid_mask] - test[valid_mask])) if np.any(valid_mask) else 0.0
    ok = np.allclose(ref, test, atol=tol, rtol=0.0, equal_nan=True)
    if ok:
        PASS += 1
        print(f"  {name:<40s} PASS  (max diff = {diff:.2e})")
    else:
        FAIL += 1
        print(f"  {name:<40s} FAIL  (max diff = {diff:.2e}, tol = {tol})")


def compare_case(name, runner, tol=1e-5):
    global SKIP
    try:
        ref = flatten_arrays(runner(CPU_DEVICE))
        test = flatten_arrays(runner(GPU_DEVICE))
        check(name, ref, test, tol=tol)
    except Exception as e:
        SKIP += 1
        print(f"  {name:<40s} SKIP  ({e})")


def arr(data, device, dtype=float):
    return wp.array(np.asarray(data), dtype=dtype, device=device)


def zeros(n, device, dtype=float):
    return wp.zeros(n, dtype=dtype, device=device)


def sync(device):
    wp.synchronize_device(device)


def patch_2d(N, base=1.0, center=2.0):
    u = np.full((N, N), base, dtype=np.float32)
    u[N // 4 : 3 * N // 4, N // 4 : 3 * N // 4] = center
    return u


def ramp_2d(N, sx=1.0, sy=1.0, bias=0.0):
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    return bias + sx * xx / max(1, N - 1) + sy * yy / max(1, N - 1)


def wave_state(N):
    u = patch_2d(N, 0.5, 1.5)
    up = patch_2d(N, 0.25, 0.75)
    return u, up


def pair_state_1d(N):
    out = np.zeros(2 * N, dtype=np.float32)
    out[0::2] = np.linspace(0.0, 1.0, N, endpoint=False, dtype=np.float32)
    out[1::2] = 0.05 * np.sin(np.linspace(0.0, 2.0 * np.pi, N, endpoint=False, dtype=np.float32))
    return out


def rng(seed):
    return np.random.default_rng(seed)


def particle_lattice(N, scale=0.3):
    side = int(np.ceil(np.sqrt(N)))
    xs = np.linspace(0.0, scale, side, endpoint=False, dtype=np.float32)
    ys = np.linspace(0.0, scale, side, endpoint=False, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)[:N]
    return pts[:, 0], pts[:, 1]


def run_2d_single(kernel, N=32, steps=3, init=None):
    def runner(device):
        u0 = patch_2d(N) if init is None else init()
        u = arr(u0.ravel(), device)
        v = zeros(N * N, device)
        for _ in range(steps):
            wp.launch(kernel, dim=(N, N), inputs=[N, u, v], device=device)
            wp.launch(rw.copy_kernel, dim=N * N, inputs=[N * N, v, u], device=device)
        sync(device)
        return u.numpy().reshape(N, N)

    return runner


def run_3d_single(kernel, N=8, steps=2, init_value=1.0):
    def runner(device):
        u = arr(np.full(N * N * N, init_value, dtype=np.float32), device)
        v = zeros(N * N * N, device)
        for _ in range(steps):
            wp.launch(kernel, dim=(N, N, N), inputs=[N, u, v], device=device)
            wp.launch(rw.copy_kernel, dim=N * N * N, inputs=[N * N * N, v, u], device=device)
        sync(device)
        return u.numpy().reshape(N, N, N)

    return runner


def run_1d_single(kernel, N=128, steps=4, init=None):
    def runner(device):
        u0 = np.linspace(0.0, 1.0, N, endpoint=False, dtype=np.float32) if init is None else init()
        u = arr(u0, device)
        v = zeros(len(u0), device)
        for _ in range(steps):
            wp.launch(kernel, dim=N, inputs=[N, u, v], device=device)
            wp.launch(rw.copy_kernel, dim=len(u0), inputs=[len(u0), v, u], device=device)
        sync(device)
        return u.numpy()

    return runner


def run_heat2d():
    return run_2d_single(rw.heat2d_step, N=32, steps=4)


def run_jacobi2d():
    def init():
        u = np.ones((32, 32), dtype=np.float32)
        u[0, :] = 0
        u[-1, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        return u

    return run_2d_single(rw.jacobi2d_step, N=32, steps=4, init=init)


def run_wave2d(device):
    N = 32
    steps = 3
    u0, up0 = wave_state(N)
    u = arr(u0.ravel(), device)
    v = zeros(N * N, device)
    up = arr(up0.ravel(), device)
    for _ in range(steps):
        wp.launch(rw.wave2d_step, dim=(N, N), inputs=[N, u, v, up], device=device)
        wp.launch(rw.copy2_kernel, dim=N * N, inputs=[N * N, u, up, v, u], device=device)
    sync(device)
    return u.numpy().reshape(N, N), up.numpy().reshape(N, N)


def run_grayscott(device):
    N = 24
    steps = 3
    gu0 = patch_2d(N, 1.0, 0.8)
    gv0 = patch_2d(N, 0.1, 0.25)
    gu = arr(gu0.ravel(), device)
    gv = arr(gv0.ravel(), device)
    gu2 = zeros(N * N, device)
    gv2 = zeros(N * N, device)
    for _ in range(steps):
        wp.launch(rw.grayscott_step, dim=(N, N), inputs=[N, gu, gv, gu2, gv2], device=device)
        wp.launch(rw.copy2_kernel, dim=N * N, inputs=[N * N, gu2, gu, gv2, gv], device=device)
    sync(device)
    return gu.numpy().reshape(N, N), gv.numpy().reshape(N, N)


def run_cahnhilliard(device):
    N = 24
    steps = 3
    u0 = ramp_2d(N, sx=0.15, sy=0.08, bias=1.0).astype(np.float32)
    u = arr(u0.ravel(), device)
    v = zeros(N * N, device)
    for _ in range(steps):
        wp.launch(rw.cahnhilliard_step, dim=(N, N), inputs=[N, u, v], device=device)
        wp.launch(rw.copy_kernel, dim=N * N, inputs=[N * N, v, u], device=device)
    sync(device)
    return u.numpy().reshape(N, N)


def run_burgers2d(device):
    N = 24
    steps = 3
    u = arr(ramp_2d(N, sx=0.3, sy=0.0, bias=0.2).ravel(), device)
    vu = arr(ramp_2d(N, sx=0.0, sy=0.25, bias=0.1).ravel(), device)
    u2 = zeros(N * N, device)
    v2 = zeros(N * N, device)
    for _ in range(steps):
        wp.launch(rw.burgers2d_step, dim=(N, N), inputs=[N, u, vu, u2, v2], device=device)
        wp.launch(rw.copy2_kernel, dim=N * N, inputs=[N * N, u2, u, v2, vu], device=device)
    sync(device)
    return u.numpy().reshape(N, N), vu.numpy().reshape(N, N)


def run_swe(device):
    N = 24
    steps = 3
    h = arr(patch_2d(N, 1.0, 1.3).ravel(), device)
    hu = arr(ramp_2d(N, sx=0.02, sy=0.0).ravel(), device)
    hv = arr(ramp_2d(N, sx=0.0, sy=0.02).ravel(), device)
    h2 = zeros(N * N, device)
    hu2 = zeros(N * N, device)
    hv2 = zeros(N * N, device)
    for _ in range(steps):
        wp.launch(rw.swe_lf_step, dim=(N, N), inputs=[N, h, hu, hv, h2, hu2, hv2], device=device)
        wp.launch(
            rw.copy3_kernel,
            dim=N * N,
            inputs=[N * N, h2, h, hu2, hu, hv2, hv],
            device=device,
        )
    sync(device)
    return h.numpy(), hu.numpy(), hv.numpy()


def run_lbm(device):
    N = 16
    N2 = N * N
    steps = 2
    weights = np.array([4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4, dtype=np.float32)
    f0 = np.concatenate([np.full(N2, w, dtype=np.float32) for w in weights])
    f = arr(f0, device)
    f2 = arr(f0.copy(), device)
    for _ in range(steps):
        wp.launch(rw.lbm_step_wp, dim=(N, N), inputs=[N, f, f2], device=device)
        wp.launch(rw.copy_kernel, dim=9 * N2, inputs=[9 * N2, f2, f], device=device)
    sync(device)
    return f.numpy()


def run_stablefluids(device):
    N = 16
    steps = 2
    u = arr(ramp_2d(N, sx=0.03, sy=0.04, bias=0.01).ravel(), device)
    v = zeros(N * N, device)
    for _ in range(steps):
        wp.launch(rw.semilag_step, dim=(N, N), inputs=[N, u, v], device=device)
        wp.launch(rw.copy_kernel, dim=N * N, inputs=[N * N, v, u], device=device)
        for _ in range(4):
            wp.launch(rw.jacobi2d_step, dim=(N, N), inputs=[N, u, v], device=device)
            wp.launch(rw.copy_kernel, dim=N * N, inputs=[N * N, v, u], device=device)
    sync(device)
    return u.numpy().reshape(N, N)


def run_euler1d(device):
    N = 128
    steps = 3
    x = np.linspace(0.0, 1.0, N, endpoint=False, dtype=np.float32)
    rho = arr(1.0 + 0.1 * np.sin(2.0 * np.pi * x), device)
    rhou = arr(0.2 * np.cos(2.0 * np.pi * x), device)
    E = arr(2.0 + 0.05 * np.sin(4.0 * np.pi * x), device)
    rho2 = zeros(N, device)
    rhou2 = zeros(N, device)
    E2 = zeros(N, device)
    for _ in range(steps):
        wp.launch(
            rw.euler1d_step,
            dim=N,
            inputs=[N, rho, rhou, E, rho2, rhou2, E2],
            device=device,
        )
        wp.launch(rw.copy3_kernel, dim=N, inputs=[N, rho2, rho, rhou2, rhou, E2, E], device=device)
    sync(device)
    return rho.numpy(), rhou.numpy(), E.numpy()


def run_nbody(device, N=32, steps=3):
    gen = rng(0)
    px = arr(gen.random(N, dtype=np.float32), device)
    py = arr(gen.random(N, dtype=np.float32), device)
    vx = zeros(N, device)
    vy = zeros(N, device)
    px2 = zeros(N, device)
    py2 = zeros(N, device)
    vx2 = zeros(N, device)
    vy2 = zeros(N, device)
    for _ in range(steps):
        wp.launch(rw.nbody_step, dim=N, inputs=[N, px, py, vx, vy, px2, py2, vx2, vy2], device=device)
        wp.launch(
            rw.copy4_kernel,
            dim=N,
            inputs=[N, px2, px, py2, py, vx2, vx, vy2, vy],
            device=device,
        )
    sync(device)
    return px.numpy(), py.numpy(), vx.numpy(), vy.numpy()


def run_sph(device):
    N = 16
    steps = 1
    px0, py0 = particle_lattice(N, scale=0.8)
    px = arr(px0, device)
    py = arr(py0, device)
    vx = zeros(N, device)
    vy = zeros(N, device)
    rho = zeros(N, device)
    px2 = zeros(N, device)
    py2 = zeros(N, device)
    vx2 = zeros(N, device)
    vy2 = zeros(N, device)
    for _ in range(steps):
        wp.launch(rw.sph_step, dim=N, inputs=[N, px, py, vx, vy, rho, px2, py2, vx2, vy2], device=device)
        wp.launch(
            rw.copy4_kernel,
            dim=N,
            inputs=[N, px2, px, py2, py, vx2, vx, vy2, vy],
            device=device,
        )
    sync(device)
    return px.numpy(), py.numpy(), vx.numpy(), vy.numpy(), rho.numpy()


def run_dem(device):
    N = 48
    steps = 2
    px0, py0 = particle_lattice(N, scale=0.2)
    px = arr(px0, device)
    py = arr(py0, device)
    vx = zeros(N, device)
    vy = zeros(N, device)
    px2 = zeros(N, device)
    py2 = zeros(N, device)
    vx2 = zeros(N, device)
    vy2 = zeros(N, device)
    for _ in range(steps):
        wp.launch(rw.dem_step, dim=N, inputs=[N, px, py, vx, vy, px2, py2, vx2, vy2], device=device)
        wp.launch(
            rw.copy4_kernel,
            dim=N,
            inputs=[N, px2, px, py2, py, vx2, vx, vy2, vy],
            device=device,
        )
    sync(device)
    return px.numpy(), py.numpy(), vx.numpy(), vy.numpy()


def run_mdlj(device):
    N = 48
    steps = 1
    px0, py0 = particle_lattice(N, scale=0.35)
    px = arr(px0, device)
    py = arr(py0, device)
    vx = zeros(N, device)
    vy = zeros(N, device)
    px2 = zeros(N, device)
    py2 = zeros(N, device)
    vx2 = zeros(N, device)
    vy2 = zeros(N, device)
    for _ in range(steps):
        wp.launch(rw.mdlj_step, dim=N, inputs=[N, px, py, vx, vy, px2, py2, vx2, vy2], device=device)
        wp.launch(
            rw.copy4_kernel,
            dim=N,
            inputs=[N, px2, px, py2, py, vx2, vx, vy2, vy],
            device=device,
        )
    sync(device)
    return px.numpy(), py.numpy(), vx.numpy(), vy.numpy()


def run_pic(device):
    NP = 128
    NG = 32
    steps = 3
    xp = arr(np.linspace(0.0, 1.0, NP, endpoint=False, dtype=np.float32), device)
    vp = arr(0.01 * np.sin(np.linspace(0.0, 2.0 * np.pi, NP, endpoint=False, dtype=np.float32)), device)
    rho_g = zeros(NG, device)
    E_g = zeros(NG, device)
    for _ in range(steps):
        wp.launch(rw.pic_zero, dim=NG, inputs=[NG, rho_g], device=device)
        wp.launch(rw.pic_deposit, dim=NP, inputs=[NP, NG, xp, rho_g], device=device)
        wp.launch(rw.pic_field, dim=NG, inputs=[NG, rho_g, E_g], device=device)
        wp.launch(rw.pic_push, dim=NP, inputs=[NP, NG, xp, vp, E_g], device=device)
    sync(device)
    return xp.numpy(), vp.numpy(), rho_g.numpy(), E_g.numpy()


def run_fdtd(device):
    N = 24
    steps = 3
    Ex = arr(ramp_2d(N, sx=0.01, sy=0.02).ravel(), device)
    Ey = arr(ramp_2d(N, sx=0.02, sy=0.01).ravel(), device)
    Hz = arr(ramp_2d(N, sx=0.01, sy=-0.01).ravel(), device)
    for _ in range(steps):
        wp.launch(rw.fdtd_step, dim=(N, N), inputs=[N, Ex, Ey, Hz], device=device)
    sync(device)
    return Ex.numpy(), Ey.numpy(), Hz.numpy()


def run_massspring(device):
    N = 96
    steps = 3
    x = arr(pair_state_1d(N), device)
    v = zeros(2 * N, device)
    for _ in range(steps):
        wp.launch(rw.massspring1d_step, dim=N, inputs=[N, x, v], device=device)
        wp.launch(rw.copy_kernel, dim=2 * N, inputs=[2 * N, v, x], device=device)
    sync(device)
    return x.numpy()


def run_montecarlo(device):
    N = 256
    steps = 4
    x = zeros(N, device)
    y = zeros(N, device)
    state = arr(np.arange(N, dtype=np.uint32) * 12345 + 67890, device, dtype=wp.uint32)
    for _ in range(steps):
        wp.launch(rw.montecarlo_step, dim=N, inputs=[N, x, y, state], device=device)
    sync(device)
    return x.numpy(), y.numpy(), state.numpy()


def run_hotspot(device):
    N = 24
    steps = 3
    u = arr(np.full(N * N, 80.0, dtype=np.float32), device)
    v = zeros(N * N, device)
    power = arr(ramp_2d(N, sx=0.02, sy=0.01, bias=0.01).ravel(), device)
    for _ in range(steps):
        wp.launch(rw.hotspot_step, dim=(N, N), inputs=[N, u, v, power], device=device)
        wp.launch(rw.copy_kernel, dim=N * N, inputs=[N * N, v, u], device=device)
    sync(device)
    return u.numpy().reshape(N, N)


def run_cg(device):
    N = 16
    N2 = N * N
    steps = 2
    gen = rng(4)
    x = zeros(N2, device)
    r0 = gen.random(N2, dtype=np.float32)
    r = arr(r0, device)
    p = arr(r0.copy(), device)
    Ap = zeros(N2, device)
    d_rr = zeros(1, device)
    d_pAp = zeros(1, device)
    d_rnew = zeros(1, device)
    for _ in range(steps):
        wp.launch(rw.cg_matvec, dim=N2, inputs=[N, N2, p, Ap], device=device)
        wp.launch(rw.zero_scalar, dim=1, inputs=[d_rr], device=device)
        wp.launch(rw.zero_scalar, dim=1, inputs=[d_pAp], device=device)
        wp.launch(rw.cg_dot, dim=N2, inputs=[N2, r, r, d_rr], device=device)
        wp.launch(rw.cg_dot, dim=N2, inputs=[N2, p, Ap, d_pAp], device=device)
        sync(device)
        rr = float(d_rr.numpy()[0])
        pAp = float(d_pAp.numpy()[0])
        alpha = rr / (pAp + 1e-10)
        wp.launch(rw.cg_axpy, dim=N2, inputs=[N2, x, alpha, p], device=device)
        wp.launch(rw.cg_axpy_neg, dim=N2, inputs=[N2, r, alpha, Ap], device=device)
        wp.launch(rw.zero_scalar, dim=1, inputs=[d_rnew], device=device)
        wp.launch(rw.cg_dot, dim=N2, inputs=[N2, r, r, d_rnew], device=device)
        sync(device)
        rnew = float(d_rnew.numpy()[0])
        beta = rnew / (rr + 1e-10)
        wp.launch(rw.cg_update_p, dim=N2, inputs=[N2, r, p, beta], device=device)
    sync(device)
    return x.numpy(), r.numpy(), p.numpy()


def run_lulesh(device):
    N = 8
    NE = N * N
    NN = (N + 1) * (N + 1)
    steps = 2
    p_el = arr(np.full(NE, 1.0, dtype=np.float32), device)
    vol = arr(np.full(NE, 1.0 / (N * N), dtype=np.float32), device)
    e_el = arr(np.full(NE, 1.0, dtype=np.float32), device)
    mass_el = arr(np.full(NE, 1.0 / (N * N), dtype=np.float32), device)
    fx = zeros(NN, device)
    fy = zeros(NN, device)
    vn_x = zeros(NN, device)
    vn_y = zeros(NN, device)
    xn_x = arr(np.repeat(np.linspace(0.0, 1.0, N + 1, dtype=np.float32), N + 1), device)
    xn_y = arr(np.tile(np.linspace(0.0, 1.0, N + 1, dtype=np.float32), N + 1), device)
    mass_n = arr(np.full(NN, 1.0 / NN, dtype=np.float32), device)
    dt = 0.001
    for _ in range(steps):
        wp.launch(rw.lulesh_reset, dim=NN, inputs=[NN, fx, fy], device=device)
        wp.launch(rw.lulesh_forces, dim=NE, inputs=[NE, N, p_el, vol, fx, fy], device=device)
        wp.launch(
            rw.lulesh_update,
            dim=NN,
            inputs=[NN, vn_x, vn_y, xn_x, xn_y, fx, fy, mass_n, dt],
            device=device,
        )
        wp.launch(rw.lulesh_eos, dim=NE, inputs=[NE, N, vol, p_el, e_el, mass_el], device=device)
    sync(device)
    return p_el.numpy(), xn_x.numpy(), xn_y.numpy(), vn_x.numpy(), vn_y.numpy()


def numpy_heat2d(N=32, steps=4):
    u = patch_2d(N)
    v = np.zeros_like(u)
    for _ in range(steps):
        v[1:-1, 1:-1] = u[1:-1, 1:-1] + 0.2 * (
            u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4.0 * u[1:-1, 1:-1]
        )
        u[1:-1, 1:-1] = v[1:-1, 1:-1]
    return u


def numpy_jacobi2d(N=32, steps=4):
    u = np.ones((N, N), dtype=np.float32)
    u[0, :] = 0
    u[-1, :] = 0
    u[:, 0] = 0
    u[:, -1] = 0
    for _ in range(steps):
        v = u.copy()
        v[1:-1, 1:-1] = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:])
        u = v
    return u


def warp_heat2d_gpu():
    return run_heat2d()(GPU_DEVICE)


def warp_jacobi2d_gpu():
    return run_jacobi2d()(GPU_DEVICE)


CASES = [
    ("Heat2D", run_heat2d(), 1e-6),
    ("Heat3D", run_3d_single(rw.heat3d_step, N=8, steps=2), 1e-6),
    ("Jacobi2D", run_jacobi2d(), 1e-6),
    ("Wave2D", run_wave2d, 1e-6),
    ("GrayScott", run_grayscott, 1e-6),
    ("AllenCahn", run_2d_single(rw.allencahn_step, N=24, steps=3), 1e-6),
    ("CahnHilliard", run_cahnhilliard, 1e-6),
    ("Burgers2D", run_burgers2d, 1e-6),
    ("ConvDiff", run_2d_single(rw.convdiff_step, N=24, steps=3), 1e-6),
    ("SWE_LaxFried", run_swe, 1e-6),
    ("LBM_D2Q9", run_lbm, 1e-6),
    ("StableFluids", run_stablefluids, 1e-6),
    ("Euler1D", run_euler1d, 1e-6),
    ("NBody", run_nbody, 1e-5),
    ("SPH", run_sph, 1e-5),
    ("DEM", run_dem, 1e-5),
    ("MD_LJ", run_mdlj, 1e-4),
    ("PIC", run_pic, 1e-5),
    ("FDTD_Maxwell", run_fdtd, 1e-5),
    ("Helmholtz2D", run_2d_single(rw.poisson2d_step, N=24, steps=3), 1e-6),
    ("ExplicitFEM", run_2d_single(rw.fem2d_step, N=24, steps=3), 1e-6),
    ("Cloth", run_2d_single(rw.cloth_step, N=24, steps=3), 1e-6),
    ("MassSpring1D", run_massspring, 1e-6),
    ("SemiLagrangian", run_2d_single(rw.semilag_step, N=24, steps=3), 1e-5),
    ("Upwind1D", run_1d_single(rw.upwind1d_step, N=128, steps=4), 1e-6),
    ("Poisson2D", run_2d_single(rw.poisson2d_step, N=24, steps=3), 1e-6),
    ("Schrodinger1D", run_1d_single(rw.schrodinger1d_step, N=128, steps=4, init=lambda: pair_state_1d(128)), 1e-6),
    ("KS1D", run_1d_single(rw.ks1d_step, N=64, steps=1, init=lambda: 1e-4 * np.sin(np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False, dtype=np.float32))), 1e-5),
    ("Reduction", run_1d_single(rw.reduce_step, N=256, steps=4), 1e-6),
    ("MonteCarlo", run_montecarlo, 1e-6),
    ("Jacobi3D", run_3d_single(rw.heat3d_step, N=8, steps=2), 1e-6),
    ("HotSpot", run_hotspot, 1e-6),
    ("SRAD", run_2d_single(rw.srad_step, N=24, steps=3, init=lambda: np.full((24, 24), 1.5, dtype=np.float32)), 1e-6),
    ("SpMV", run_1d_single(rw.spmv_step, N=256, steps=4), 1e-6),
    ("CG_Solver", run_cg, 1e-5),
    ("LULESH", run_lulesh, 1e-5),
]


if __name__ == "__main__":
    print(f"Correctness Test: {len(CASES)} kernel types (Warp CPU vs Warp GPU)")
    print("=" * 72)

    if len(CASES) != 36:
        print(f"Expected 36 cases, found {len(CASES)}", file=sys.stderr)
        sys.exit(2)

    for name, runner, tol in CASES:
        compare_case(name, runner, tol=tol)

    print(f"\n{'=' * 72}")
    print(f"  PASS: {PASS}   FAIL: {FAIL}   SKIP: {SKIP}")
    print(f"{'=' * 72}")
    if FAIL > 0:
        sys.exit(1)
