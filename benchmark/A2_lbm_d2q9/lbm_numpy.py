"""LBM D2Q9 — NumPy baseline."""
import numpy as np

# D2Q9 lattice constants
E = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]], dtype=np.int32)
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float32)

def run(nx, ny=None, steps=1, backend="cpu"):
    if ny is None:
        ny = nx // 2
    inv_tau = 1.0 / 0.6

    rho = np.ones((nx, ny), dtype=np.float32)
    vel = np.zeros((nx, ny, 2), dtype=np.float32)
    f_old = np.zeros((nx, ny, 9), dtype=np.float32)
    f_new = np.zeros((nx, ny, 9), dtype=np.float32)

    # Init equilibrium
    for k in range(9):
        f_old[:, :, k] = W[k] * rho

    def step():
        for _ in range(steps):
            # Collision + streaming (gather)
            for k in range(9):
                ex, ey = int(E[k, 0]), int(E[k, 1])
                # Gather from upstream neighbor
                f_old_shifted = np.roll(np.roll(f_old[:, :, k], ex, axis=0), ey, axis=1)
                # BGK collision
                eu = vel[:, :, 0] * E[k, 0] + vel[:, :, 1] * E[k, 1]
                usq = vel[:, :, 0]**2 + vel[:, :, 1]**2
                feq = W[k] * rho * (1.0 + 3.0*eu + 4.5*eu**2 - 1.5*usq)
                f_new[:, :, k] = (1 - inv_tau) * f_old_shifted + inv_tau * feq
            # Update macro
            rho[:] = np.sum(f_new, axis=2)
            vel[:, :, 0] = np.sum(f_new * E[:, 0].reshape(1, 1, 9), axis=2) / rho
            vel[:, :, 1] = np.sum(f_new * E[:, 1].reshape(1, 1, 9), axis=2) / rho
            f_old[:] = f_new

    return step, None, rho
