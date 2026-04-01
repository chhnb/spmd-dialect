"""2D Wave Equation — NumPy baseline."""
import numpy as np

def run(N, steps=1, backend="cpu"):
    c = 1.0; dt = 0.1; dx = 1.0
    coeff = (c * dt / dx) ** 2
    h_prev = np.zeros((N, N), dtype=np.float32)
    h_curr = np.zeros((N, N), dtype=np.float32)
    mask = np.fromfunction(lambda i,j: ((i-N//4)**2+(j-N//4)**2) < (N//16)**2, (N,N))
    h_curr[mask] = 1.0

    def step():
        for _ in range(steps):
            lap = (h_curr[:-2,1:-1] + h_curr[2:,1:-1] + h_curr[1:-1,:-2] + h_curr[1:-1,2:] - 4*h_curr[1:-1,1:-1])
            h_new = 2*h_curr[1:-1,1:-1] - h_prev[1:-1,1:-1] + coeff * lap
            h_prev[1:-1,1:-1] = h_curr[1:-1,1:-1]
            h_curr[1:-1,1:-1] = h_new
    return step, None, h_curr
