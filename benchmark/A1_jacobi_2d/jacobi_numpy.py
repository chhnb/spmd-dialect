"""Jacobi 2D 5-point stencil — NumPy baseline."""

import numpy as np


def create_field(N):
    """Create NxN field with boundary conditions."""
    u = np.zeros((N, N), dtype=np.float32)
    # simple boundary: top=1, others=0
    u[0, :] = 1.0
    return u


def jacobi_step(u, u_new):
    """One Jacobi iteration: u_new[i,j] = 0.25*(u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1])"""
    u_new[1:-1, 1:-1] = 0.25 * (
        u[0:-2, 1:-1] +  # up
        u[2:,   1:-1] +  # down
        u[1:-1, 0:-2] +  # left
        u[1:-1, 2:]      # right
    )


def run(N, steps=1):
    u = create_field(N)
    u_new = u.copy()
    def step():
        for _ in range(steps):
            jacobi_step(u, u_new)
            u[:] = u_new
    sync = None
    return step, sync, u
