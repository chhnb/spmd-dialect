"""N-body (direct all-pairs) — NumPy baseline."""
import numpy as np

G = 1.0
SOFTENING = 1e-5

def run(N, steps=1, backend="cpu"):
    pos = np.random.randn(N, 3).astype(np.float32)
    vel = np.zeros((N, 3), dtype=np.float32)
    mass = np.ones(N, dtype=np.float32)
    dt = 0.001

    def step():
        nonlocal pos, vel
        for _ in range(steps):
            force = np.zeros_like(pos)
            for i in range(N):
                diff = pos - pos[i]  # (N, 3)
                dist = np.sqrt(np.sum(diff**2, axis=1) + SOFTENING**2)  # (N,)
                f = G * mass[i] * mass / dist**3  # (N,)
                force[i] = np.sum(diff * f[:, None], axis=0)
            vel += force * dt
            pos += vel * dt
    return step, None, pos
