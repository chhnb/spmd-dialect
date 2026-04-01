"""Jacobi 2D 5-point stencil — Warp implementation."""

import numpy as np
import warp as wp


@wp.kernel
def jacobi_step_kernel(
    u: wp.array2d(dtype=float),
    u_new: wp.array2d(dtype=float),
    N: int,
):
    i, j = wp.tid()
    if i >= 1 and i < N - 1 and j >= 1 and j < N - 1:
        u_new[i, j] = 0.25 * (
            u[i - 1, j] + u[i + 1, j] +
            u[i, j - 1] + u[i, j + 1]
        )


@wp.kernel
def copy_kernel(
    src: wp.array2d(dtype=float),
    dst: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    dst[i, j] = src[i, j]


def run(N, steps=1, backend="cuda"):
    device = backend

    u_np = np.zeros((N, N), dtype=np.float32)
    u_np[0, :] = 1.0

    u = wp.array(u_np, dtype=float, device=device)
    u_new = wp.array(u_np, dtype=float, device=device)

    def step():
        for _ in range(steps):
            wp.launch(jacobi_step_kernel, dim=(N, N), inputs=[u, u_new, N], device=device)
            wp.launch(copy_kernel, dim=(N, N), inputs=[u_new, u], device=device)

    def sync():
        wp.synchronize_device(device)

    return step, sync, u
