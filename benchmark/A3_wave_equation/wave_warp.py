"""2D Wave Equation — Warp."""
import numpy as np
import warp as wp


@wp.struct
class WaveMesh:
    h_prev: wp.array2d(dtype=float)
    h_curr: wp.array2d(dtype=float)
    N: int
    coeff: float


@wp.kernel
def wave_step_kernel(m: WaveMesh):
    h_prev = m.h_prev
    h_curr = m.h_curr
    N = m.N
    coeff = m.coeff
    i, j = wp.tid()
    if i >= 1 and i < N-1 and j >= 1 and j < N-1:
        lap = h_curr[i-1,j] + h_curr[i+1,j] + h_curr[i,j-1] + h_curr[i,j+1] - 4.0*h_curr[i,j]
        h_new = 2.0*h_curr[i,j] - h_prev[i,j] + coeff * lap
        h_prev[i,j] = h_curr[i,j]
        h_curr[i,j] = h_new

def run(N, steps=1, backend="cuda"):
    c = 1.0; dt = 0.1; dx = 1.0
    coeff = (c * dt / dx) ** 2
    h_np = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if (i - N//4)**2 + (j - N//4)**2 < (N//16)**2:
                h_np[i, j] = 1.0

    mesh = WaveMesh()
    mesh.h_prev = wp.array(np.zeros((N,N), dtype=np.float32), dtype=float, device=backend)
    mesh.h_curr = wp.array(h_np, dtype=float, device=backend)
    mesh.N = N
    mesh.coeff = coeff

    def step():
        for _ in range(steps):
            wp.launch(wave_step_kernel, dim=(N, N), inputs=[mesh], device=backend)
    def sync():
        wp.synchronize_device(backend)
    return step, sync, mesh.h_curr
