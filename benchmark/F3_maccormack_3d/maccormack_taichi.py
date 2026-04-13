"""
MacCormack 3D Advection — Taichi implementation.
AsyncTaichi benchmark: predictor-corrector scheme, 2 kernels per step.

Usage:
  python maccormack_taichi.py [N] [steps]
"""
import taichi as ti
import time
import sys
import numpy as np


def run(N=64, steps=200, backend="cuda"):
    ti.init(arch=ti.cuda if backend == "cuda" else ti.cpu, default_fp=ti.f32)

    # Velocity field (constant uniform advection)
    vel_x, vel_y, vel_z = 1.0, 0.5, 0.3
    dt = 0.5 / N  # CFL-safe

    u = ti.field(ti.f32, shape=(N, N, N))
    u_pred = ti.field(ti.f32, shape=(N, N, N))
    u_new = ti.field(ti.f32, shape=(N, N, N))

    @ti.kernel
    def init():
        for i, j, k in u:
            # Gaussian blob in center
            ci = float(i) / N - 0.3
            cj = float(j) / N - 0.3
            ck = float(k) / N - 0.3
            u[i, j, k] = ti.exp(-50.0 * (ci*ci + cj*cj + ck*ck))

    @ti.kernel
    def predict():
        """MacCormack predictor: forward differences."""
        for i, j, k in ti.ndrange((1, N-1), (1, N-1), (1, N-1)):
            dudx = (u[i+1, j, k] - u[i, j, k])
            dudy = (u[i, j+1, k] - u[i, j, k])
            dudz = (u[i, j, k+1] - u[i, j, k])
            u_pred[i, j, k] = u[i, j, k] - dt * (vel_x * dudx + vel_y * dudy + vel_z * dudz)

    @ti.kernel
    def correct():
        """MacCormack corrector: backward differences on predicted + average."""
        for i, j, k in ti.ndrange((1, N-1), (1, N-1), (1, N-1)):
            dudx = (u_pred[i, j, k] - u_pred[i-1, j, k])
            dudy = (u_pred[i, j, k] - u_pred[i, j-1, k])
            dudz = (u_pred[i, j, k] - u_pred[i, j, k-1])
            u_corr = u_pred[i, j, k] - dt * (vel_x * dudx + vel_y * dudy + vel_z * dudz)
            u_new[i, j, k] = 0.5 * (u[i, j, k] + u_corr)

    @ti.kernel
    def copy_back():
        for i, j, k in u:
            u[i, j, k] = u_new[i, j, k]

    init()

    def step_fn():
        for _ in range(steps):
            predict()
            correct()
            copy_back()

    def sync_fn():
        ti.sync()

    # Warmup
    step_fn()
    sync_fn()

    return step_fn, sync_fn, u


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    print(f"MacCormack 3D Advection (Taichi, CUDA)")
    for n in [32, 64, 128]:
        us = run(n, steps, "cuda")
        print(f"  N={n:>4} ({n**3:>8} cells, 3 kernels/step): {us:>8.1f} μs/step")
        ti.reset()
