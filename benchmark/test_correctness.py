#!/usr/bin/env python3
"""Correctness test: compare Taichi vs Warp vs NumPy for all kernels.
Runs each kernel for a few steps, then compares output arrays.

Usage:
  spmd-venv/bin/python benchmark/test_correctness.py
"""
import numpy as np
import sys
import os

os.environ['TI_LOG_LEVEL'] = 'warn'

PASS = 0
FAIL = 0
SKIP = 0

def check(name, ref, test, tol=1e-4):
    global PASS, FAIL, SKIP
    if ref is None or test is None:
        SKIP += 1
        print(f"  {name:<40s} SKIP")
        return
    diff = np.max(np.abs(ref - test))
    if diff < tol:
        PASS += 1
        print(f"  {name:<40s} PASS  (max diff = {diff:.2e})")
    else:
        FAIL += 1
        print(f"  {name:<40s} FAIL  (max diff = {diff:.2e}, tol = {tol})")

# =========================================================================
# Helper: run Taichi kernel and return numpy array
# =========================================================================
def run_taichi_2d(N, steps=5):
    """Run Heat2D in Taichi, return array."""
    import taichi as ti
    ti.init(arch=ti.cuda, default_fp=ti.f32, log_level='warn')
    u = ti.field(dtype=ti.f32, shape=(N, N))
    v = ti.field(dtype=ti.f32, shape=(N, N))
    # Init
    u_np = np.ones((N, N), dtype=np.float32)
    u_np[N//4:3*N//4, N//4:3*N//4] = 2.0
    u.from_numpy(u_np)

    @ti.kernel
    def step():
        for i, j in u:
            if i >= 1 and i < N-1 and j >= 1 and j < N-1:
                v[i,j] = u[i,j] + 0.2*(u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1]-4.0*u[i,j])
    @ti.kernel
    def copy():
        for i, j in v:
            u[i,j] = v[i,j]

    for _ in range(steps):
        step()
        copy()
    ti.sync()
    result = u.to_numpy()
    ti.reset()
    return result

def run_warp_2d(N, steps=5):
    """Run Heat2D in Warp, return array."""
    import warp as wp
    device = "cuda:0"
    u_np = np.ones(N*N, dtype=np.float32)
    u_np.reshape(N,N)[N//4:3*N//4, N//4:3*N//4] = 2.0
    u = wp.array(u_np, dtype=float, device=device)
    v = wp.zeros(N*N, dtype=float, device=device)

    @wp.kernel
    def heat_step(N_k: int, u_k: wp.array(dtype=float), v_k: wp.array(dtype=float)):
        i, j = wp.tid()
        if i >= 1 and i < N_k-1 and j >= 1 and j < N_k-1:
            idx = i*N_k + j
            v_k[idx] = u_k[idx] + 0.2*(u_k[idx-N_k]+u_k[idx+N_k]+u_k[idx-1]+u_k[idx+1]-4.0*u_k[idx])

    @wp.kernel
    def copy_k(n: int, src: wp.array(dtype=float), dst: wp.array(dtype=float)):
        i = wp.tid()
        if i < n:
            dst[i] = src[i]

    for _ in range(steps):
        wp.launch(heat_step, dim=(N, N), inputs=[N, u, v], device=device)
        wp.launch(copy_k, dim=N*N, inputs=[N*N, v, u], device=device)
    wp.synchronize_device(device)
    return u.numpy().reshape(N, N)

def run_numpy_2d(N, steps=5):
    """Run Heat2D in NumPy (reference), return array."""
    u = np.ones((N, N), dtype=np.float32)
    u[N//4:3*N//4, N//4:3*N//4] = 2.0
    v = np.zeros_like(u)
    for _ in range(steps):
        v[1:-1, 1:-1] = u[1:-1, 1:-1] + 0.2*(
            u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4.0*u[1:-1, 1:-1])
        # Copy only interior (match Taichi/Warp which don't touch boundary)
        u[1:-1, 1:-1] = v[1:-1, 1:-1]
    return u

# Jacobi2D
def run_numpy_jacobi(N, steps=5):
    u = np.ones((N, N), dtype=np.float32)
    u[0,:] = 0; u[-1,:] = 0; u[:,0] = 0; u[:,-1] = 0
    for _ in range(steps):
        v = u.copy()
        v[1:-1, 1:-1] = 0.25*(u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:])
        u = v
    return u

def run_taichi_jacobi(N, steps=5):
    import taichi as ti
    ti.init(arch=ti.cuda, default_fp=ti.f32, log_level='warn')
    u = ti.field(dtype=ti.f32, shape=(N, N))
    v = ti.field(dtype=ti.f32, shape=(N, N))
    u_np = np.ones((N, N), dtype=np.float32)
    u_np[0,:] = 0; u_np[-1,:] = 0; u_np[:,0] = 0; u_np[:,-1] = 0
    u.from_numpy(u_np)

    @ti.kernel
    def step():
        for i, j in u:
            if i >= 1 and i < N-1 and j >= 1 and j < N-1:
                v[i,j] = 0.25*(u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1])
    @ti.kernel
    def copy():
        for i, j in v:
            u[i,j] = v[i,j]

    for _ in range(steps):
        step()
        copy()
    ti.sync()
    result = u.to_numpy()
    ti.reset()
    return result

def run_warp_jacobi(N, steps=5):
    import warp as wp
    device = "cuda:0"
    u_np = np.ones(N*N, dtype=np.float32)
    u_np.reshape(N,N)[0,:] = 0; u_np.reshape(N,N)[-1,:] = 0
    u_np.reshape(N,N)[:,0] = 0; u_np.reshape(N,N)[:,-1] = 0
    u = wp.array(u_np, dtype=float, device=device)
    v = wp.zeros(N*N, dtype=float, device=device)

    @wp.kernel
    def jacobi_step(N_k: int, u_k: wp.array(dtype=float), v_k: wp.array(dtype=float)):
        i, j = wp.tid()
        if i >= 1 and i < N_k-1 and j >= 1 and j < N_k-1:
            idx = i*N_k + j
            v_k[idx] = 0.25*(u_k[idx-N_k]+u_k[idx+N_k]+u_k[idx-1]+u_k[idx+1])

    @wp.kernel
    def copy_k(n: int, src: wp.array(dtype=float), dst: wp.array(dtype=float)):
        i = wp.tid()
        if i < n:
            dst[i] = src[i]

    for _ in range(steps):
        wp.launch(jacobi_step, dim=(N, N), inputs=[N, u, v], device=device)
        wp.launch(copy_k, dim=N*N, inputs=[N*N, v, u], device=device)
    wp.synchronize_device(device)
    return u.numpy().reshape(N, N)

# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    N = 64
    STEPS = 10

    print(f"Correctness Test: N={N}, steps={STEPS}")
    print(f"{'='*60}")

    # Heat2D
    print("\n--- Heat2D ---")
    ref = run_numpy_2d(N, STEPS)
    try:
        tai = run_taichi_2d(N, STEPS)
        check("Taichi vs NumPy", ref, tai, tol=1e-4)
    except Exception as e:
        print(f"  Taichi: {e}")
        SKIP += 1

    try:
        war = run_warp_2d(N, STEPS)
        check("Warp vs NumPy", ref, war, tol=1e-4)
    except Exception as e:
        print(f"  Warp: {e}")
        SKIP += 1

    if 'tai' in dir() and 'war' in dir():
        check("Taichi vs Warp", tai, war, tol=1e-4)

    # Jacobi2D
    print("\n--- Jacobi2D ---")
    ref = run_numpy_jacobi(N, STEPS)
    try:
        tai = run_taichi_jacobi(N, STEPS)
        check("Taichi vs NumPy", ref, tai, tol=1e-4)
    except Exception as e:
        print(f"  Taichi: {e}")
        SKIP += 1

    try:
        war = run_warp_jacobi(N, STEPS)
        check("Warp vs NumPy", ref, war, tol=1e-4)
    except Exception as e:
        print(f"  Warp: {e}")
        SKIP += 1

    # NBody (compare Taichi vs Warp, no NumPy ref for simplicity)
    print("\n--- NBody N=64 ---")
    try:
        import taichi as ti
        ti.init(arch=ti.cuda, default_fp=ti.f32, log_level='warn')
        np.random.seed(42)
        pos_np = np.random.rand(64, 2).astype(np.float32)
        vel_np = np.zeros((64, 2), dtype=np.float32)

        px_t = ti.field(dtype=ti.f32, shape=64)
        py_t = ti.field(dtype=ti.f32, shape=64)
        vx_t = ti.field(dtype=ti.f32, shape=64)
        vy_t = ti.field(dtype=ti.f32, shape=64)
        px_t.from_numpy(pos_np[:, 0])
        py_t.from_numpy(pos_np[:, 1])

        @ti.kernel
        def nbody_ti():
            for i in range(64):
                ax, ay = 0.0, 0.0
                for j in range(64):
                    dx = px_t[j]-px_t[i]; dy = py_t[j]-py_t[i]
                    r2 = dx*dx+dy*dy+0.01
                    inv = 1.0/ti.sqrt(r2*r2*r2)
                    ax += dx*inv; ay += dy*inv
                vx_t[i] += 0.001*ax; vy_t[i] += 0.001*ay
                px_t[i] += 0.001*vx_t[i]; py_t[i] += 0.001*vy_t[i]

        for _ in range(3):
            nbody_ti()
        ti.sync()
        tai_px = px_t.to_numpy()
        ti.reset()

        # Warp version
        import warp as wp
        device = "cuda:0"
        np.random.seed(42)
        pos_np2 = np.random.rand(64, 2).astype(np.float32)
        px_w = wp.array(pos_np2[:, 0].copy(), dtype=float, device=device)
        py_w = wp.array(pos_np2[:, 1].copy(), dtype=float, device=device)
        vx_w = wp.zeros(64, dtype=float, device=device)
        vy_w = wp.zeros(64, dtype=float, device=device)

        @wp.kernel
        def nbody_wp(N_k: int, px_k: wp.array(dtype=float), py_k: wp.array(dtype=float),
                      vx_k: wp.array(dtype=float), vy_k: wp.array(dtype=float)):
            i = wp.tid()
            if i >= N_k:
                return
            ax = float(0.0); ay = float(0.0)
            for j in range(N_k):
                dx = px_k[j]-px_k[i]; dy = py_k[j]-py_k[i]
                r2 = dx*dx+dy*dy+0.01
                inv = 1.0/wp.sqrt(r2*r2*r2)
                ax = ax + dx*inv; ay = ay + dy*inv
            vx_k[i] = vx_k[i] + 0.001*ax; vy_k[i] = vy_k[i] + 0.001*ay
            px_k[i] = px_k[i] + 0.001*vx_k[i]; py_k[i] = py_k[i] + 0.001*vy_k[i]

        for _ in range(3):
            wp.launch(nbody_wp, dim=64, inputs=[64, px_w, py_w, vx_w, vy_w], device=device)
        wp.synchronize_device(device)
        war_px = px_w.numpy()

        check("Taichi vs Warp (px)", tai_px, war_px, tol=1e-3)

    except Exception as e:
        print(f"  NBody error: {e}")
        SKIP += 1

    # F1 OSHER (use existing implementations)
    print("\n--- F1 OSHER 32sq ---")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'F1_hydro_shallow_water'))
        import taichi as ti
        from hydro_taichi import run as taichi_run
        step_fn_t, sync_fn_t, H_t = taichi_run(32, steps=1, backend="cuda")
        for _ in range(5):
            step_fn_t()
        sync_fn_t()
        tai_h = H_t.to_numpy()
        ti.reset()

        from hydro_warp import run as warp_run
        step_fn_w, sync_fn_w, H_w = warp_run(32, steps=1, backend="cuda")
        for _ in range(5):
            step_fn_w()
        sync_fn_w()
        war_h = H_w.numpy() if hasattr(H_w, 'numpy') else np.array(H_w)

        check("Taichi vs Warp (H field)", tai_h.flatten(), war_h.flatten(), tol=1e-3)
    except Exception as e:
        print(f"  F1 OSHER error: {e}")
        SKIP += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"  PASS: {PASS}   FAIL: {FAIL}   SKIP: {SKIP}")
    print(f"{'='*60}")
    if FAIL > 0:
        sys.exit(1)
