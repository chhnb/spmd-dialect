#!/usr/bin/env python3
"""Warp characterization: 36 kernel types × multiple sizes.
Mirrors run_overhead_characterization.py but uses Warp instead of Taichi.

Usage:
  spmd-venv/bin/python benchmark/run_warp_characterization.py
"""
import warp as wp
import numpy as np
import time
import os

wp.init()
device = "cuda:0"

def bench(step_fn, sync_fn, warmup=20, steps=500):
    """Benchmark a step function, return μs/step."""
    for _ in range(warmup):
        step_fn()
    sync_fn()
    sync_fn()
    t0 = time.perf_counter()
    for _ in range(steps):
        step_fn()
    sync_fn()
    elapsed = time.perf_counter() - t0
    return elapsed * 1e6 / steps

def sync():
    wp.synchronize_device(device)

def section(name):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")

# =========================================================================
# 1. Heat2D (5-point stencil)
# =========================================================================
@wp.kernel
def heat2d_step(N: int, u: wp.array(dtype=float), v: wp.array(dtype=float)):
    i, j = wp.tid()
    if i >= 1 and i < N-1 and j >= 1 and j < N-1:
        idx = i*N + j
        v[idx] = u[idx] + 0.2*(u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-4.0*u[idx])

@wp.kernel
def copy_kernel(n: int, src: wp.array(dtype=float), dst: wp.array(dtype=float)):
    i = wp.tid()
    if i < n:
        dst[i] = src[i]

def test_heat2d(N, steps=500):
    N2 = N*N
    u = wp.zeros(N2, dtype=float, device=device)
    v = wp.zeros(N2, dtype=float, device=device)
    def step():
        wp.launch(heat2d_step, dim=(N, N), inputs=[N, u, v], device=device)
        wp.launch(copy_kernel, dim=N2, inputs=[N2, v, u], device=device)
    return bench(step, sync, steps=steps)

# =========================================================================
# 2. Heat3D (7-point stencil)
# =========================================================================
@wp.kernel
def heat3d_step(N: int, u: wp.array(dtype=float), v: wp.array(dtype=float)):
    i, j, k = wp.tid()
    N2 = N*N
    if i >= 1 and i < N-1 and j >= 1 and j < N-1 and k >= 1 and k < N-1:
        idx = i*N2 + j*N + k
        v[idx] = u[idx] + 0.1*(u[idx-N2]+u[idx+N2]+u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-6.0*u[idx])

def test_heat3d(N, steps=500):
    N3 = N*N*N
    u = wp.zeros(N3, dtype=float, device=device)
    v = wp.zeros(N3, dtype=float, device=device)
    def step():
        wp.launch(heat3d_step, dim=(N, N, N), inputs=[N, u, v], device=device)
        wp.launch(copy_kernel, dim=N3, inputs=[N3, v, u], device=device)
    return bench(step, sync, steps=steps)

# =========================================================================
# 3. Jacobi2D
# =========================================================================
@wp.kernel
def jacobi2d_step(N: int, u: wp.array(dtype=float), v: wp.array(dtype=float)):
    i, j = wp.tid()
    if i >= 1 and i < N-1 and j >= 1 and j < N-1:
        idx = i*N + j
        v[idx] = 0.25*(u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1])

def test_jacobi2d(N, steps=500):
    N2 = N*N
    u = wp.zeros(N2, dtype=float, device=device)
    v = wp.zeros(N2, dtype=float, device=device)
    def step():
        wp.launch(jacobi2d_step, dim=(N, N), inputs=[N, u, v], device=device)
        wp.launch(copy_kernel, dim=N2, inputs=[N2, v, u], device=device)
    return bench(step, sync, steps=steps)

# =========================================================================
# 4. Wave2D (2nd order time)
# =========================================================================
@wp.kernel
def wave2d_step(N: int, u: wp.array(dtype=float), v: wp.array(dtype=float), u_prev: wp.array(dtype=float)):
    i, j = wp.tid()
    if i >= 1 and i < N-1 and j >= 1 and j < N-1:
        idx = i*N + j
        lap = u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-4.0*u[idx]
        v[idx] = 2.0*u[idx] - u_prev[idx] + 0.01*lap

@wp.kernel
def copy2_kernel(n: int, s1: wp.array(dtype=float), d1: wp.array(dtype=float),
                  s2: wp.array(dtype=float), d2: wp.array(dtype=float)):
    i = wp.tid()
    if i < n:
        d1[i] = s1[i]
        d2[i] = s2[i]

def test_wave2d(N, steps=500):
    N2 = N*N
    u = wp.zeros(N2, dtype=float, device=device)
    v = wp.zeros(N2, dtype=float, device=device)
    up = wp.zeros(N2, dtype=float, device=device)
    def step():
        wp.launch(wave2d_step, dim=(N, N), inputs=[N, u, v, up], device=device)
        wp.launch(copy2_kernel, dim=N2, inputs=[N2, u, up, v, u], device=device)
    return bench(step, sync, steps=steps)

# =========================================================================
# 5. GrayScott (reaction-diffusion)
# =========================================================================
@wp.kernel
def grayscott_step(N: int, gu: wp.array(dtype=float), gv: wp.array(dtype=float),
                    gu2: wp.array(dtype=float), gv2: wp.array(dtype=float)):
    i, j = wp.tid()
    if i >= 1 and i < N-1 and j >= 1 and j < N-1:
        idx = i*N + j
        lu = gu[idx-N]+gu[idx+N]+gu[idx-1]+gu[idx+1]-4.0*gu[idx]
        lv = gv[idx-N]+gv[idx+N]+gv[idx-1]+gv[idx+1]-4.0*gv[idx]
        uvv = gu[idx]*gv[idx]*gv[idx]
        gu2[idx] = gu[idx] + 0.16*lu - uvv + 0.06*(1.0-gu[idx])
        gv2[idx] = gv[idx] + 0.08*lv + uvv - 0.122*gv[idx]

def test_grayscott(N, steps=500):
    N2 = N*N
    gu = wp.zeros(N2, dtype=float, device=device)
    gv = wp.zeros(N2, dtype=float, device=device)
    gu2 = wp.zeros(N2, dtype=float, device=device)
    gv2 = wp.zeros(N2, dtype=float, device=device)
    def step():
        wp.launch(grayscott_step, dim=(N, N), inputs=[N, gu, gv, gu2, gv2], device=device)
        wp.launch(copy2_kernel, dim=N2, inputs=[N2, gu2, gu, gv2, gv], device=device)
    return bench(step, sync, steps=steps)

# =========================================================================
# Generic 2-field stencil template (for AllenCahn, CahnHilliard, ConvDiff, etc.)
# =========================================================================
@wp.kernel
def allencahn_step(N: int, u: wp.array(dtype=float), v: wp.array(dtype=float)):
    i, j = wp.tid()
    if i >= 1 and i < N-1 and j >= 1 and j < N-1:
        idx = i*N + j
        lap = u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-4.0*u[idx]
        phi = u[idx]
        v[idx] = phi + 0.01*(0.01*lap + phi - phi*phi*phi)

@wp.kernel
def cahnhilliard_step(N: int, u: wp.array(dtype=float), v: wp.array(dtype=float)):
    i, j = wp.tid()
    if i >= 2 and i < N-2 and j >= 2 and j < N-2:
        idx = i*N + j
        lap = u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-4.0*u[idx]
        mu_n = -0.01*(u[idx-2*N]+u[idx+2*N]+u[idx-2]+u[idx+2]-4.0*lap)/4.0
        v[idx] = u[idx] + 0.001*(mu_n + lap)

@wp.kernel
def burgers2d_step(N: int, u: wp.array(dtype=float), vu: wp.array(dtype=float),
                    u2: wp.array(dtype=float), v2: wp.array(dtype=float)):
    i, j = wp.tid()
    if i >= 1 and i < N-1 and j >= 1 and j < N-1:
        idx = i*N + j
        nu = 0.01; dt = 0.001; dx = 1.0/float(N)
        dudx = (u[idx]-u[idx-1])/dx; dudy = (u[idx]-u[idx-N])/dx
        dvdx = (vu[idx]-vu[idx-1])/dx; dvdy = (vu[idx]-vu[idx-N])/dx
        lapu = (u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-4.0*u[idx])/(dx*dx)
        lapv = (vu[idx-N]+vu[idx+N]+vu[idx-1]+vu[idx+1]-4.0*vu[idx])/(dx*dx)
        u2[idx] = u[idx] + dt*(-u[idx]*dudx - vu[idx]*dudy + nu*lapu)
        v2[idx] = vu[idx] + dt*(-u[idx]*dvdx - vu[idx]*dvdy + nu*lapv)

@wp.kernel
def convdiff_step(N: int, u: wp.array(dtype=float), v: wp.array(dtype=float)):
    i, j = wp.tid()
    if i >= 1 and i < N-1 and j >= 1 and j < N-1:
        idx = i*N + j
        dx = 1.0/float(N); nu = 0.01; cx = 1.0; cy = 0.5
        lap = (u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-4.0*u[idx])/(dx*dx)
        adv_x = cx*(u[idx+1]-u[idx-1])/(2.0*dx)
        adv_y = cy*(u[idx+N]-u[idx-N])/(2.0*dx)
        v[idx] = u[idx] + 0.0001*(nu*lap - adv_x - adv_y)

@wp.kernel
def swe_lf_step(N: int, h: wp.array(dtype=float), hu: wp.array(dtype=float), hv: wp.array(dtype=float),
                 h2: wp.array(dtype=float), hu2: wp.array(dtype=float), hv2: wp.array(dtype=float)):
    i, j = wp.tid()
    if i >= 1 and i < N-1 and j >= 1 and j < N-1:
        idx = i*N + j; g = 9.81; dx = 1.0/float(N); dt = 0.0001
        h_avg = 0.25*(h[idx-N]+h[idx+N]+h[idx-1]+h[idx+1])
        hu_avg = 0.25*(hu[idx-N]+hu[idx+N]+hu[idx-1]+hu[idx+1])
        dhu_dx = (hu[idx+1]-hu[idx-1])/(2.0*dx)
        dhv_dy = (hv[idx+N]-hv[idx-N])/(2.0*dx)
        h2[idx] = h_avg - dt*(dhu_dx + dhv_dy)
        hu2[idx] = hu_avg
        hv2[idx] = 0.25*(hv[idx-N]+hv[idx+N]+hv[idx-1]+hv[idx+1])

@wp.kernel
def fdtd_step(N: int, Ex: wp.array(dtype=float), Ey: wp.array(dtype=float), Hz: wp.array(dtype=float)):
    i, j = wp.tid()
    dt = 0.001; dx = 0.01
    if i < N-1 and j < N-1:
        idx = i*N + j
        Hz[idx] = Hz[idx] + dt/dx*(Ex[idx+N]-Ex[idx]-Ey[idx+1]+Ey[idx])
    if i >= 1 and i < N and j < N:
        idx = i*N + j
        Ex[idx] = Ex[idx] + dt/dx*(Hz[idx]-Hz[idx-N])
    if i < N and j >= 1 and j < N:
        idx = i*N + j
        Ey[idx] = Ey[idx] - dt/dx*(Hz[idx]-Hz[idx-1])

@wp.kernel
def srad_step(N: int, u: wp.array(dtype=float), v: wp.array(dtype=float)):
    i, j = wp.tid()
    if i >= 1 and i < N-1 and j >= 1 and j < N-1:
        idx = i*N + j
        Jc = u[idx]; dt = 0.05; lam = 0.5
        dN = u[idx-N]-Jc; dS = u[idx+N]-Jc; dW = u[idx-1]-Jc; dE = u[idx+1]-Jc
        G2 = (dN*dN+dS*dS+dW*dW+dE*dE)/(Jc*Jc+1e-10)
        L = (dN+dS+dW+dE)/(Jc+1e-10)
        num = 0.5*G2 - L*L/16.0
        den = (1.0+0.25*L)*(1.0+0.25*L)+1e-10
        q = num/den; q0 = 1.0
        c = 1.0/(1.0+(q-q0*q0)/(q0*q0*(1.0+q0*q0)+1e-10))
        c = wp.clamp(c, 0.0, 1.0)
        v[idx] = Jc + dt*lam*(c*dN + c*dS + c*dW + c*dE)

@wp.kernel
def hotspot_step(N: int, u: wp.array(dtype=float), v: wp.array(dtype=float), power: wp.array(dtype=float)):
    i, j = wp.tid()
    if i >= 1 and i < N-1 and j >= 1 and j < N-1:
        idx = i*N + j
        cap = 0.5; rx = 0.01; ry = 0.01; amb = 80.0
        lap_x = (u[idx-1]+u[idx+1]-2.0*u[idx])*rx
        lap_y = (u[idx-N]+u[idx+N]-2.0*u[idx])*ry
        v[idx] = u[idx] + cap*(lap_x + lap_y + power[idx] + (amb-u[idx])*0.001)

@wp.kernel
def poisson2d_step(N: int, u: wp.array(dtype=float), v: wp.array(dtype=float)):
    i, j = wp.tid()
    if i >= 1 and i < N-1 and j >= 1 and j < N-1:
        idx = i*N + j
        v[idx] = 0.25*(u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1])

@wp.kernel
def semilag_step(N: int, u: wp.array(dtype=float), v: wp.array(dtype=float)):
    i, j = wp.tid()
    if i >= 1 and i < N-1 and j >= 1 and j < N-1:
        idx = i*N + j
        cx = 0.5; cy = 0.3; dt = 0.001; dx = 1.0/float(N)
        xd = float(j)*dx - cx*dt; yd = float(i)*dx - cy*dt
        xd = wp.clamp(xd, dx, float(N-2)*dx); yd = wp.clamp(yd, dx, float(N-2)*dx)
        j0 = int(xd/dx); i0 = int(yd/dx)
        j0 = wp.clamp(j0, 1, N-2); i0 = wp.clamp(i0, 1, N-2)
        fx = xd/dx - float(j0); fy = yd/dx - float(i0)
        v[idx] = (1.0-fx)*(1.0-fy)*u[i0*N+j0] + fx*(1.0-fy)*u[i0*N+j0+1] + (1.0-fx)*fy*u[(i0+1)*N+j0] + fx*fy*u[(i0+1)*N+j0+1]

@wp.kernel
def fem2d_step(N: int, u: wp.array(dtype=float), v: wp.array(dtype=float)):
    i, j = wp.tid()
    if i >= 1 and i < N-1 and j >= 1 and j < N-1:
        idx = i*N + j
        v[idx] = u[idx] + 0.01*(u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]+u[idx-N+1]+u[idx+N-1]-6.0*u[idx])

@wp.kernel
def cloth_step(N: int, u: wp.array(dtype=float), v: wp.array(dtype=float)):
    i, j = wp.tid()
    if i >= 1 and i < N-1 and j >= 1 and j < N-1:
        idx = i*N + j
        k = 50.0; rest = 1.0/float(N)
        fx = k*((u[idx+1]-u[idx])-rest) + k*((u[idx-1]-u[idx])+rest)
        fy = k*((u[idx+N]-u[idx])-rest) + k*((u[idx-N]-u[idx])+rest)
        v[idx] = u[idx] + 0.0001*(fx+fy)

# 1D kernels
@wp.kernel
def upwind1d_step(N: int, u: wp.array(dtype=float), v: wp.array(dtype=float)):
    i = wp.tid()
    if i >= 1 and i < N-1:
        c = 1.0; dx = 1.0/float(N)
        v[i] = u[i] - c*0.0001/dx*(u[i]-u[i-1])

@wp.kernel
def schrodinger1d_step(N: int, u: wp.array(dtype=float), v: wp.array(dtype=float)):
    i = wp.tid()
    if i >= 1 and i < N-1:
        dt = 0.0001; dx = 1.0/float(N)
        lap_r = (u[2*(i-1)]+u[2*(i+1)]-2.0*u[2*i])/(dx*dx)
        lap_i = (u[2*(i-1)+1]+u[2*(i+1)+1]-2.0*u[2*i+1])/(dx*dx)
        v[2*i] = u[2*i] + dt*0.5*lap_i
        v[2*i+1] = u[2*i+1] - dt*0.5*lap_r

@wp.kernel
def ks1d_step(N: int, u: wp.array(dtype=float), v: wp.array(dtype=float)):
    i = wp.tid()
    if i >= 2 and i < N-2:
        dx = 1.0/float(N); dt = 0.00001
        dudx = (u[i+1]-u[i-1])/(2.0*dx)
        d2u = (u[i+1]+u[i-1]-2.0*u[i])/(dx*dx)
        d4u = (u[i+2]-4.0*u[i+1]+6.0*u[i]-4.0*u[i-1]+u[i-2])/(dx*dx*dx*dx)
        v[i] = u[i] + dt*(-u[i]*dudx - d2u - d4u)

@wp.kernel
def massspring1d_step(N: int, x: wp.array(dtype=float), v_out: wp.array(dtype=float)):
    i = wp.tid()
    if i >= 1 and i < N-1:
        k = 100.0; m = 1.0; dt = 0.0001; rest = 1.0/float(N)
        pos = x[2*i]; vel = x[2*i+1]
        f = k*(x[2*(i+1)]-pos-rest) + k*(x[2*(i-1)]-pos+rest)
        v_out[2*i] = pos + dt*vel
        v_out[2*i+1] = vel + dt*f/m

@wp.kernel
def euler1d_step(N: int, rho: wp.array(dtype=float), rhou: wp.array(dtype=float), E: wp.array(dtype=float),
                  rho2: wp.array(dtype=float), rhou2: wp.array(dtype=float), E2: wp.array(dtype=float)):
    i = wp.tid()
    if i >= 1 and i < N-1:
        dx = 1.0/float(N); dt = 0.00001; gamma = 1.4
        u_ = rhou[i]/(rho[i]+1e-10)
        p = (gamma-1.0)*(E[i]-0.5*rho[i]*u_*u_)
        rho2[i] = 0.5*(rho[i-1]+rho[i+1]) - dt/(2.0*dx)*(rhou[i+1]-rhou[i-1])
        rhou2[i] = 0.5*(rhou[i-1]+rhou[i+1]) - dt/(2.0*dx)*(rhou[i+1]*u_+p-rhou[i-1]*u_-p)
        E2[i] = 0.5*(E[i-1]+E[i+1]) - dt/(2.0*dx)*((E[i+1]+p)*u_-(E[i-1]+p)*u_)

@wp.kernel
def reduce_step(N: int, u: wp.array(dtype=float), v: wp.array(dtype=float)):
    i = wp.tid()
    if i >= 1 and i < N-1:
        v[i] = u[i]*0.999 + 0.0005*(u[i-1]+u[i+1])
    elif i < N:
        v[i] = u[i]

@wp.kernel
def montecarlo_step(N: int, x: wp.array(dtype=float), y: wp.array(dtype=float), state: wp.array(dtype=wp.uint32)):
    i = wp.tid()
    if i < N:
        s = state[i]
        s = s * wp.uint32(1664525) + wp.uint32(1013904223)
        r1 = float(int(s & wp.uint32(0xFFFF)))/65536.0 - 0.5
        s = s * wp.uint32(1664525) + wp.uint32(1013904223)
        r2 = float(int(s & wp.uint32(0xFFFF)))/65536.0 - 0.5
        state[i] = s
        x[i] = x[i] + 0.01*r1
        y[i] = y[i] + 0.01*r2

@wp.kernel
def spmv_step(N: int, x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    if i >= 1 and i < N-1:
        y[i] = x[i-1] + 2.0*x[i] + x[i+1]
    elif i == 0:
        y[i] = 2.0*x[i] + x[i+1]
    elif i == N-1:
        y[i] = x[i-1] + 2.0*x[i]

# SPH (brute force O(N²))
@wp.kernel
def sph_step(N: int, px: wp.array(dtype=float), py: wp.array(dtype=float),
              vx: wp.array(dtype=float), vy: wp.array(dtype=float),
              rho_s: wp.array(dtype=float),
              px2: wp.array(dtype=float), py2: wp.array(dtype=float),
              vx2: wp.array(dtype=float), vy2: wp.array(dtype=float)):
    i = wp.tid()
    if i >= N:
        return
    h = 0.1
    dt = 0.0001
    mass = 1.0
    k = 100.0
    rho0 = 1.0
    rho_i = float(0.0)
    for j in range(N):
        dx = px[j]-px[i]
        dy = py[j]-py[i]
        r2 = dx*dx+dy*dy
        if r2 < h*h:
            q = wp.sqrt(r2)/h
            rho_i = rho_i + mass*(1.0-q)*(1.0-q)
    rho_s[i] = wp.max(rho_i, 0.001)
    ax = float(0.0)
    ay = float(0.0)
    pi_ = k*(rho_i - rho0)
    for j in range(N):
        if i != j:
            dx = px[j]-px[i]
            dy = py[j]-py[i]
            r = wp.sqrt(dx*dx+dy*dy) + 1e-6
            if r < h:
                pj = k*(rho_s[j]-rho0)
                f = -mass*(pi_+pj)/(2.0*rho_s[j])*(-2.0*(1.0-r/h)/h)/r
                ax = ax + f*dx
                ay = ay + f*dy
    vx2[i] = vx[i]+dt*ax
    vy2[i] = vy[i]+dt*ay
    px2[i] = px[i]+dt*vx2[i]
    py2[i] = py[i]+dt*vy2[i]

# DEM (spring-dashpot O(N²))
@wp.kernel
def dem_step(N: int, px: wp.array(dtype=float), py: wp.array(dtype=float),
              vx: wp.array(dtype=float), vy: wp.array(dtype=float),
              px2: wp.array(dtype=float), py2: wp.array(dtype=float),
              vx2: wp.array(dtype=float), vy2: wp.array(dtype=float)):
    i = wp.tid()
    if i >= N:
        return
    dt = 0.0001
    rad = 0.01
    kn = 1e4
    gn = 10.0
    ax = float(0.0)
    ay = float(-9.81)
    for j in range(N):
        if i != j:
            dx = px[j]-px[i]
            dy = py[j]-py[i]
            dist = wp.sqrt(dx*dx+dy*dy) + 1e-10
            overlap = 2.0*rad - dist
            if overlap > 0.0:
                nx = dx/dist
                ny = dy/dist
                dvn = (vx[j]-vx[i])*nx + (vy[j]-vy[i])*ny
                fn = kn*overlap - gn*dvn
                ax = ax + fn*nx
                ay = ay + fn*ny
    vx2[i] = vx[i]+dt*ax
    vy2[i] = vy[i]+dt*ay
    px2[i] = px[i]+dt*vx2[i]
    py2[i] = py[i]+dt*vy2[i]

# MD Lennard-Jones (O(N²))
@wp.kernel
def mdlj_step(N: int, px: wp.array(dtype=float), py: wp.array(dtype=float),
               vx: wp.array(dtype=float), vy: wp.array(dtype=float),
               px2: wp.array(dtype=float), py2: wp.array(dtype=float),
               vx2: wp.array(dtype=float), vy2: wp.array(dtype=float)):
    i = wp.tid()
    if i >= N:
        return
    dt = 0.0001
    eps_lj = 1.0
    sigma = 0.01
    ax = float(0.0)
    ay = float(0.0)
    for j in range(N):
        if i != j:
            dx = px[j]-px[i]
            dy = py[j]-py[i]
            r2 = dx*dx+dy*dy+1e-10
            s2 = sigma*sigma/r2
            s6 = s2*s2*s2
            f = 24.0*eps_lj*(2.0*s6*s6-s6)/r2
            ax = ax + f*dx
            ay = ay + f*dy
    vx2[i] = vx[i]+dt*ax
    vy2[i] = vy[i]+dt*ay
    px2[i] = px[i]+dt*vx2[i]
    py2[i] = py[i]+dt*vy2[i]

# PIC 1D (4 phases)
@wp.kernel
def pic_deposit(NP: int, NG: int, xp: wp.array(dtype=float), rho_g: wp.array(dtype=float)):
    i = wp.tid()
    if i >= NP:
        return
    dx = 1.0/float(NG)
    cell = int(xp[i]/dx)
    cell = wp.clamp(cell, 0, NG-1)
    wp.atomic_add(rho_g, cell, 1.0)

@wp.kernel
def pic_field(NG: int, rho_g: wp.array(dtype=float), E_g: wp.array(dtype=float)):
    i = wp.tid()
    if i >= 1 and i < NG-1:
        E_g[i] = -(rho_g[i+1]-rho_g[i-1])*0.5*float(NG)

@wp.kernel
def pic_push(NP: int, NG: int, xp: wp.array(dtype=float), vp: wp.array(dtype=float), E_g: wp.array(dtype=float)):
    i = wp.tid()
    if i >= NP:
        return
    dx = 1.0/float(NG); dt = 0.0001
    cell = int(xp[i]/dx)
    cell = wp.clamp(cell, 0, NG-1)
    vp[i] = vp[i] + dt*E_g[cell]
    xp[i] = xp[i] + dt*vp[i]
    if xp[i] < 0.0:
        xp[i] = xp[i] + 1.0
    if xp[i] >= 1.0:
        xp[i] = xp[i] - 1.0

@wp.kernel
def pic_zero(NG: int, rho_g: wp.array(dtype=float)):
    i = wp.tid()
    if i < NG:
        rho_g[i] = 0.0

# LBM D2Q9 (simplified collision+streaming)
@wp.kernel
def lbm_step_wp(N: int, f: wp.array(dtype=float), f2: wp.array(dtype=float)):
    i, j = wp.tid()
    if i < 1 or i >= N-1 or j < 1 or j >= N-1:
        return
    N2 = N*N
    idx = i*N + j
    # 9 velocities, compute density + velocity
    rho = float(0.0); ux = float(0.0); uy = float(0.0)
    for q in range(9):
        fq = f[q*N2+idx]
        rho = rho + fq
    omega = 1.5
    # simplified: just relax toward equilibrium
    for q in range(9):
        w = 4.0/9.0
        if q >= 1 and q <= 4:
            w = 1.0/9.0
        elif q >= 5:
            w = 1.0/36.0
        feq = w * rho
        f2[q*N2+idx] = f[q*N2+idx] + omega*(feq - f[q*N2+idx])

# CG Solver (5 kernels/step)
@wp.kernel
def cg_matvec(N: int, N2: int, p: wp.array(dtype=float), Ap: wp.array(dtype=float)):
    idx = wp.tid()
    if idx >= N2:
        return
    i = idx / N; j = idx % N
    s = 4.0 * p[idx]
    if i > 0: s = s - p[idx - N]
    if i < N-1: s = s - p[idx + N]
    if j > 0: s = s - p[idx - 1]
    if j < N-1: s = s - p[idx + 1]
    Ap[idx] = s

@wp.kernel
def cg_dot(N2: int, a: wp.array(dtype=float), b: wp.array(dtype=float), result: wp.array(dtype=float)):
    i = wp.tid()
    if i < N2:
        wp.atomic_add(result, 0, a[i]*b[i])

@wp.kernel
def cg_axpy(N2: int, x: wp.array(dtype=float), alpha: float, p: wp.array(dtype=float)):
    i = wp.tid()
    if i < N2:
        x[i] = x[i] + alpha * p[i]

@wp.kernel
def cg_axpy_neg(N2: int, r: wp.array(dtype=float), alpha: float, Ap: wp.array(dtype=float)):
    i = wp.tid()
    if i < N2:
        r[i] = r[i] - alpha * Ap[i]

@wp.kernel
def cg_update_p(N2: int, r: wp.array(dtype=float), p: wp.array(dtype=float), beta: float):
    i = wp.tid()
    if i < N2:
        p[i] = r[i] + beta * p[i]

@wp.kernel
def zero_scalar(s: wp.array(dtype=float)):
    s[0] = 0.0

# LULESH-like (4 kernels/step)
@wp.kernel
def lulesh_reset(NN: int, fx: wp.array(dtype=float), fy: wp.array(dtype=float)):
    i = wp.tid()
    if i < NN:
        fx[i] = 0.0; fy[i] = 0.0

@wp.kernel
def lulesh_forces(NE: int, N: int, p_el: wp.array(dtype=float), vol: wp.array(dtype=float),
                   fx: wp.array(dtype=float), fy: wp.array(dtype=float)):
    e = wp.tid()
    if e >= NE:
        return
    i = e / N; j = e % N
    n0 = i*(N+1)+j; n1 = (i+1)*(N+1)+j
    pr = p_el[e] * vol[e] * 0.25
    wp.atomic_add(fx, n0, -pr); wp.atomic_add(fy, n0, -pr)
    wp.atomic_add(fx, n1, pr); wp.atomic_add(fy, n1, -pr)

@wp.kernel
def lulesh_update(NN: int, vn_x: wp.array(dtype=float), vn_y: wp.array(dtype=float),
                   xn_x: wp.array(dtype=float), xn_y: wp.array(dtype=float),
                   fx: wp.array(dtype=float), fy: wp.array(dtype=float),
                   mass: wp.array(dtype=float), dt: float):
    i = wp.tid()
    if i >= NN:
        return
    if mass[i] > 0.0:
        vn_x[i] = vn_x[i] + fx[i]/mass[i]*dt
        vn_y[i] = vn_y[i] + fy[i]/mass[i]*dt
    xn_x[i] = xn_x[i] + vn_x[i]*dt
    xn_y[i] = xn_y[i] + vn_y[i]*dt

@wp.kernel
def lulesh_eos(NE: int, N: int, vol: wp.array(dtype=float), p_el: wp.array(dtype=float),
                e_el: wp.array(dtype=float), mass_el: wp.array(dtype=float)):
    e = wp.tid()
    if e >= NE:
        return
    nv = wp.max(vol[e], 1e-10)
    rho = mass_el[e] / nv
    p_el[e] = 0.4 * rho * e_el[e]

# NBody
@wp.kernel
def nbody_step(N: int, px: wp.array(dtype=float), py: wp.array(dtype=float),
                vx: wp.array(dtype=float), vy: wp.array(dtype=float),
                px2: wp.array(dtype=float), py2: wp.array(dtype=float),
                vx2: wp.array(dtype=float), vy2: wp.array(dtype=float)):
    i = wp.tid()
    if i >= N:
        return
    ax = float(0.0); ay = float(0.0); dt = 0.001; eps = 0.01
    for j in range(N):
        dx = px[j]-px[i]; dy = py[j]-py[i]
        r2 = dx*dx + dy*dy + eps
        inv = 1.0/wp.sqrt(r2*r2*r2)
        ax = ax + dx*inv; ay = ay + dy*inv
    vx2[i] = vx[i]+dt*ax; vy2[i] = vy[i]+dt*ay
    px2[i] = px[i]+dt*vx2[i]; py2[i] = py[i]+dt*vy2[i]

@wp.kernel
def copy4_kernel(n: int, s1: wp.array(dtype=float), d1: wp.array(dtype=float),
                  s2: wp.array(dtype=float), d2: wp.array(dtype=float),
                  s3: wp.array(dtype=float), d3: wp.array(dtype=float),
                  s4: wp.array(dtype=float), d4: wp.array(dtype=float)):
    i = wp.tid()
    if i < n:
        d1[i]=s1[i]; d2[i]=s2[i]; d3[i]=s3[i]; d4[i]=s4[i]

@wp.kernel
def copy3_kernel(n: int, s1: wp.array(dtype=float), d1: wp.array(dtype=float),
                  s2: wp.array(dtype=float), d2: wp.array(dtype=float),
                  s3: wp.array(dtype=float), d3: wp.array(dtype=float)):
    i = wp.tid()
    if i < n:
        d1[i]=s1[i]; d2[i]=s2[i]; d3[i]=s3[i]

# =========================================================================
# Test functions for 2-field 2D stencils (generic pattern)
# =========================================================================
def test_2d_stencil(kernel_fn, N, steps=500, init_val=1.0):
    N2 = N*N
    u_np = np.full(N2, init_val, dtype=np.float32)
    u = wp.array(u_np, dtype=float, device=device)
    v = wp.zeros(N2, dtype=float, device=device)
    def step():
        wp.launch(kernel_fn, dim=(N, N), inputs=[N, u, v], device=device)
        wp.launch(copy_kernel, dim=N2, inputs=[N2, v, u], device=device)
    return bench(step, sync, steps=steps)

def test_1d(kernel_fn, N, steps=500):
    u_np = np.full(N, 1.0, dtype=np.float32)
    u = wp.array(u_np, dtype=float, device=device)
    v = wp.zeros(N, dtype=float, device=device)
    def step():
        wp.launch(kernel_fn, dim=N, inputs=[N, u, v], device=device)
        wp.launch(copy_kernel, dim=N, inputs=[N, v, u], device=device)
    return bench(step, sync, steps=steps)

# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    results = []
    def report(name, us):
        results.append((name, us))
        if isinstance(us, (int, float)) and us > 0:
            print(f"  {name:<35s} {us:>8.1f} μs/step")
        else:
            print(f"  {name:<35s} {'SKIP':>8s}")

    print(f"Warp Characterization on {wp.get_device(device).name}")
    print(f"{'='*60}")

    section("1-9. Stencil (9 types)")
    report("Heat2D 128sq", test_heat2d(128))
    report("Heat3D 32cu", test_heat3d(32))
    report("Jacobi2D 128sq", test_jacobi2d(128))
    report("Wave2D 128sq", test_wave2d(128))
    report("GrayScott 128sq", test_grayscott(128))
    report("AllenCahn 128sq", test_2d_stencil(allencahn_step, 128))
    report("CahnHilliard 128sq", test_2d_stencil(cahnhilliard_step, 128))
    # Burgers2D special (4 arrays)
    N=128; N2=N*N
    u_b=wp.array(np.full(N2,1.0,dtype=np.float32),dtype=float,device=device)
    v_b=wp.zeros(N2,dtype=float,device=device)
    u2_b=wp.zeros(N2,dtype=float,device=device)
    v2_b=wp.zeros(N2,dtype=float,device=device)
    def step_burgers():
        wp.launch(burgers2d_step, dim=(N,N), inputs=[N,u_b,v_b,u2_b,v2_b], device=device)
        wp.launch(copy2_kernel, dim=N2, inputs=[N2,u2_b,u_b,v2_b,v_b], device=device)
    report("Burgers2D 128sq", bench(step_burgers, sync))
    report("ConvDiff 128sq", test_2d_stencil(convdiff_step, 128))

    section("10-13. CFD (4 types)")
    # SWE LaxFried
    N=128; N2=N*N
    h_s=wp.array(np.full(N2,1.0,dtype=np.float32),dtype=float,device=device)
    hu_s=wp.zeros(N2,dtype=float,device=device); hv_s=wp.zeros(N2,dtype=float,device=device)
    h2_s=wp.zeros(N2,dtype=float,device=device); hu2_s=wp.zeros(N2,dtype=float,device=device); hv2_s=wp.zeros(N2,dtype=float,device=device)
    def step_swe():
        wp.launch(swe_lf_step, dim=(N,N), inputs=[N,h_s,hu_s,hv_s,h2_s,hu2_s,hv2_s], device=device)
        wp.launch(copy3_kernel, dim=N2, inputs=[N2,h2_s,h_s,hu2_s,hu_s,hv2_s,hv_s], device=device)
    report("SWE_LaxFried 128sq", bench(step_swe, sync))

    # LBM D2Q9
    N=64; N2=N*N
    f_lbm = wp.array(np.full(9*N2, 1.0/9.0, dtype=np.float32), dtype=float, device=device)
    f2_lbm = wp.array(np.full(9*N2, 1.0/9.0, dtype=np.float32), dtype=float, device=device)
    def step_lbm():
        wp.launch(lbm_step_wp, dim=(N,N), inputs=[N, f_lbm, f2_lbm], device=device)
        wp.launch(copy_kernel, dim=9*N2, inputs=[9*N2, f2_lbm, f_lbm], device=device)
    report("LBM D2Q9 64sq", bench(step_lbm, sync))

    # StableFluids (simplified: advect + divergence + jacobi×20 + project)
    N=128; N2=N*N
    u_sf=wp.array(np.full(N2,0.01,dtype=np.float32),dtype=float,device=device)
    v_sf=wp.zeros(N2,dtype=float,device=device)
    def step_sf():
        # simplified: just run advect-like step + copy (overhead measurement)
        wp.launch(semilag_step, dim=(N,N), inputs=[N,u_sf,v_sf], device=device)
        wp.launch(copy_kernel, dim=N2, inputs=[N2,v_sf,u_sf], device=device)
        # 20 jacobi iterations
        for _ in range(20):
            wp.launch(jacobi2d_step, dim=(N,N), inputs=[N,u_sf,v_sf], device=device)
            wp.launch(copy_kernel, dim=N2, inputs=[N2,v_sf,u_sf], device=device)
    report("StableFluids 128sq", bench(step_sf, sync, steps=200))
    # Euler1D
    N=4096
    rho_e=wp.array(np.full(N,1.0,dtype=np.float32),dtype=float,device=device)
    rhou_e=wp.zeros(N,dtype=float,device=device)
    E_e=wp.array(np.full(N,1.0,dtype=np.float32),dtype=float,device=device)
    rho2_e=wp.zeros(N,dtype=float,device=device); rhou2_e=wp.zeros(N,dtype=float,device=device); E2_e=wp.zeros(N,dtype=float,device=device)
    def step_euler():
        wp.launch(euler1d_step, dim=N, inputs=[N,rho_e,rhou_e,E_e,rho2_e,rhou2_e,E2_e], device=device)
        wp.launch(copy3_kernel, dim=N, inputs=[N,rho2_e,rho_e,rhou2_e,rhou_e,E2_e,E_e], device=device)
    report("Euler1D N=4096", bench(step_euler, sync))

    section("14-18. Particle (5 types)")
    # NBody
    N=256
    px_n=wp.array(np.random.rand(N).astype(np.float32),dtype=float,device=device)
    py_n=wp.array(np.random.rand(N).astype(np.float32),dtype=float,device=device)
    vx_n=wp.zeros(N,dtype=float,device=device); vy_n=wp.zeros(N,dtype=float,device=device)
    px2_n=wp.zeros(N,dtype=float,device=device); py2_n=wp.zeros(N,dtype=float,device=device)
    vx2_n=wp.zeros(N,dtype=float,device=device); vy2_n=wp.zeros(N,dtype=float,device=device)
    def step_nbody():
        wp.launch(nbody_step, dim=N, inputs=[N,px_n,py_n,vx_n,vy_n,px2_n,py2_n,vx2_n,vy2_n], device=device)
        wp.launch(copy4_kernel, dim=N, inputs=[N,px2_n,px_n,py2_n,py_n,vx2_n,vx_n,vy2_n,vy_n], device=device)
    report("NBody N=256", bench(step_nbody, sync, steps=200))

    # SPH
    N=1024
    px_s=wp.array(np.random.rand(N).astype(np.float32)*0.5,dtype=float,device=device)
    py_s=wp.array(np.random.rand(N).astype(np.float32)*0.5,dtype=float,device=device)
    vx_s=wp.zeros(N,dtype=float,device=device); vy_s=wp.zeros(N,dtype=float,device=device)
    rho_sp=wp.zeros(N,dtype=float,device=device)
    px2_s=wp.zeros(N,dtype=float,device=device); py2_s=wp.zeros(N,dtype=float,device=device)
    vx2_s=wp.zeros(N,dtype=float,device=device); vy2_s=wp.zeros(N,dtype=float,device=device)
    def step_sph():
        wp.launch(sph_step, dim=N, inputs=[N,px_s,py_s,vx_s,vy_s,rho_sp,px2_s,py2_s,vx2_s,vy2_s], device=device)
        wp.launch(copy4_kernel, dim=N, inputs=[N,px2_s,px_s,py2_s,py_s,vx2_s,vx_s,vy2_s,vy_s], device=device)
    report("SPH N=1024", bench(step_sph, sync, steps=50))

    # DEM
    px_d=wp.array(np.random.rand(N).astype(np.float32)*0.5,dtype=float,device=device)
    py_d=wp.array(np.random.rand(N).astype(np.float32)*0.5,dtype=float,device=device)
    vx_d=wp.zeros(N,dtype=float,device=device); vy_d=wp.zeros(N,dtype=float,device=device)
    px2_d=wp.zeros(N,dtype=float,device=device); py2_d=wp.zeros(N,dtype=float,device=device)
    vx2_d=wp.zeros(N,dtype=float,device=device); vy2_d=wp.zeros(N,dtype=float,device=device)
    def step_dem():
        wp.launch(dem_step, dim=N, inputs=[N,px_d,py_d,vx_d,vy_d,px2_d,py2_d,vx2_d,vy2_d], device=device)
        wp.launch(copy4_kernel, dim=N, inputs=[N,px2_d,px_d,py2_d,py_d,vx2_d,vx_d,vy2_d,vy_d], device=device)
    report("DEM N=1024", bench(step_dem, sync, steps=50))

    # MD_LJ
    px_m=wp.array(np.random.rand(N).astype(np.float32)*0.5,dtype=float,device=device)
    py_m=wp.array(np.random.rand(N).astype(np.float32)*0.5,dtype=float,device=device)
    vx_m=wp.zeros(N,dtype=float,device=device); vy_m=wp.zeros(N,dtype=float,device=device)
    px2_m=wp.zeros(N,dtype=float,device=device); py2_m=wp.zeros(N,dtype=float,device=device)
    vx2_m=wp.zeros(N,dtype=float,device=device); vy2_m=wp.zeros(N,dtype=float,device=device)
    def step_mdlj():
        wp.launch(mdlj_step, dim=N, inputs=[N,px_m,py_m,vx_m,vy_m,px2_m,py2_m,vx2_m,vy2_m], device=device)
        wp.launch(copy4_kernel, dim=N, inputs=[N,px2_m,px_m,py2_m,py_m,vx2_m,vx_m,vy2_m,vy_m], device=device)
    report("MD_LJ N=1024", bench(step_mdlj, sync, steps=50))

    # PIC 1D
    NP=4096; NG=256
    xp_p=wp.array(np.linspace(0,1,NP,endpoint=False,dtype=np.float32),dtype=float,device=device)
    vp_p=wp.zeros(NP,dtype=float,device=device)
    rho_g=wp.zeros(NG,dtype=float,device=device); E_g=wp.zeros(NG,dtype=float,device=device)
    def step_pic():
        wp.launch(pic_zero, dim=NG, inputs=[NG, rho_g], device=device)
        wp.launch(pic_deposit, dim=NP, inputs=[NP, NG, xp_p, rho_g], device=device)
        wp.launch(pic_field, dim=NG, inputs=[NG, rho_g, E_g], device=device)
        wp.launch(pic_push, dim=NP, inputs=[NP, NG, xp_p, vp_p, E_g], device=device)
    report("PIC NP=4096", bench(step_pic, sync, steps=500))

    section("19-20. EM (2 types)")
    N=128; N2=N*N
    Ex_f=wp.zeros(N2,dtype=float,device=device)
    Ey_f=wp.zeros(N2,dtype=float,device=device)
    Hz_f=wp.zeros(N2,dtype=float,device=device)
    def step_fdtd():
        wp.launch(fdtd_step, dim=(N,N), inputs=[N,Ex_f,Ey_f,Hz_f], device=device)
    report("FDTD Maxwell 128sq", bench(step_fdtd, sync))
    report("Helmholtz2D 128sq", test_2d_stencil(poisson2d_step, 128))

    section("21-23. FEM/Structure (3 types)")
    report("ExplicitFEM 128sq", test_2d_stencil(fem2d_step, 128))
    report("Cloth 128sq", test_2d_stencil(cloth_step, 128))
    report("MassSpring1D N=4096", test_1d(massspring1d_step, 4096))

    section("24-25. Transport (2 types)")
    report("SemiLagrangian 128sq", test_2d_stencil(semilag_step, 128))
    report("Upwind1D N=4096", test_1d(upwind1d_step, 4096))

    section("26-28. PDE (3 types)")
    report("Poisson2D 128sq", test_2d_stencil(poisson2d_step, 128))
    report("Schrodinger1D N=4096", test_1d(schrodinger1d_step, 4096))
    report("KS 1D N=4096", test_1d(ks1d_step, 4096))

    section("29-30. Other (2 types)")
    report("Reduction N=16384", test_1d(reduce_step, 16384))
    # MonteCarlo
    N=4096
    x_mc=wp.zeros(N,dtype=float,device=device)
    y_mc=wp.zeros(N,dtype=float,device=device)
    st_mc=wp.array(np.arange(N,dtype=np.uint32)*12345+67890,dtype=wp.uint32,device=device)
    def step_mc():
        wp.launch(montecarlo_step, dim=N, inputs=[N,x_mc,y_mc,st_mc], device=device)
    report("MonteCarlo N=4096", bench(step_mc, sync))

    section("31-36. Classic (6 types)")
    report("Jacobi3D 32cu", test_heat3d(32))  # same stencil pattern
    # HotSpot special (needs power array)
    N=128; N2=N*N
    u_hs=wp.array(np.full(N2,80.0,dtype=np.float32),dtype=float,device=device)
    v_hs=wp.zeros(N2,dtype=float,device=device)
    pow_hs=wp.array(np.random.rand(N2).astype(np.float32)*0.1,dtype=float,device=device)
    def step_hotspot():
        wp.launch(hotspot_step, dim=(N,N), inputs=[N,u_hs,v_hs,pow_hs], device=device)
        wp.launch(copy_kernel, dim=N2, inputs=[N2,v_hs,u_hs], device=device)
    report("HotSpot 128sq", bench(step_hotspot, sync))

    report("SRAD 128sq", test_2d_stencil(srad_step, 128, init_val=1.5))
    report("SpMV N=4096", test_1d(spmv_step, 4096))
    # CG Solver (5 kernels/step + host readback)
    N=128; N2=N*N
    x_cg=wp.zeros(N2,dtype=float,device=device)
    r_cg=wp.array(np.random.rand(N2).astype(np.float32),dtype=float,device=device)
    p_cg=wp.array(r_cg.numpy().copy(),dtype=float,device=device)
    Ap_cg=wp.zeros(N2,dtype=float,device=device)
    d_rr=wp.zeros(1,dtype=float,device=device)
    d_pAp=wp.zeros(1,dtype=float,device=device)
    d_rnew=wp.zeros(1,dtype=float,device=device)
    def step_cg():
        wp.launch(cg_matvec, dim=N2, inputs=[N, N2, p_cg, Ap_cg], device=device)
        wp.launch(zero_scalar, dim=1, inputs=[d_rr], device=device)
        wp.launch(zero_scalar, dim=1, inputs=[d_pAp], device=device)
        wp.launch(cg_dot, dim=N2, inputs=[N2, r_cg, r_cg, d_rr], device=device)
        wp.launch(cg_dot, dim=N2, inputs=[N2, p_cg, Ap_cg, d_pAp], device=device)
        sync()
        rr = d_rr.numpy()[0]; pAp = d_pAp.numpy()[0]
        alpha = rr / (pAp + 1e-10)
        wp.launch(cg_axpy, dim=N2, inputs=[N2, x_cg, alpha, p_cg], device=device)
        wp.launch(cg_axpy_neg, dim=N2, inputs=[N2, r_cg, alpha, Ap_cg], device=device)
        wp.launch(zero_scalar, dim=1, inputs=[d_rnew], device=device)
        wp.launch(cg_dot, dim=N2, inputs=[N2, r_cg, r_cg, d_rnew], device=device)
        sync()
        rnew = d_rnew.numpy()[0]
        beta = rnew / (rr + 1e-10)
        wp.launch(cg_update_p, dim=N2, inputs=[N2, r_cg, p_cg, beta], device=device)
    report("CG Solver 128sq", bench(step_cg, sync, warmup=5, steps=100))

    # LULESH (4 kernels/step)
    N=64; NE=N*N; NN=(N+1)*(N+1)
    p_el=wp.array(np.full(NE,1.0,dtype=np.float32),dtype=float,device=device)
    vol_l=wp.array(np.full(NE,1.0/(N*N),dtype=np.float32),dtype=float,device=device)
    e_el=wp.array(np.full(NE,1.0,dtype=np.float32),dtype=float,device=device)
    mass_el=wp.array(np.full(NE,1.0/(N*N),dtype=np.float32),dtype=float,device=device)
    fx_l=wp.zeros(NN,dtype=float,device=device); fy_l=wp.zeros(NN,dtype=float,device=device)
    vn_x=wp.zeros(NN,dtype=float,device=device); vn_y=wp.zeros(NN,dtype=float,device=device)
    xn_x=wp.array(np.repeat(np.linspace(0,1,N+1),N+1).astype(np.float32),dtype=float,device=device)
    xn_y=wp.array(np.tile(np.linspace(0,1,N+1),N+1).astype(np.float32),dtype=float,device=device)
    mass_n=wp.array(np.full(NN,1.0/NN,dtype=np.float32),dtype=float,device=device)
    dt_l = 0.001
    def step_lulesh():
        wp.launch(lulesh_reset, dim=NN, inputs=[NN, fx_l, fy_l], device=device)
        wp.launch(lulesh_forces, dim=NE, inputs=[NE, N, p_el, vol_l, fx_l, fy_l], device=device)
        wp.launch(lulesh_update, dim=NN, inputs=[NN, vn_x, vn_y, xn_x, xn_y, fx_l, fy_l, mass_n, dt_l], device=device)
        wp.launch(lulesh_eos, dim=NE, inputs=[NE, N, vol_l, p_el, e_el, mass_el], device=device)
    report("LULESH 64sq", bench(step_lulesh, sync, steps=200))

    print(f"\n{'='*60}")
    print(f"SUMMARY: Warp Characterization")
    print(f"{'='*60}")
    for name, us in results:
        if us > 0:
            print(f"  {name:<35s} {us:>8.1f} μs/step")
        else:
            print(f"  {name:<35s} {'SKIP':>8s}")
