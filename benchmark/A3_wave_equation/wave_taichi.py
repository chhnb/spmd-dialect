"""2D Wave Equation — Taichi. h_new = 2h - h_old + dt^2 * c^2 * laplacian(h)"""
import taichi as ti

def run(N, steps=1, backend="cuda"):
    ti.init(arch=ti.cuda if backend == "cuda" else ti.cpu, default_fp=ti.f32)
    c = 1.0; dt = 0.1; dx = 1.0
    coeff = (c * dt / dx) ** 2

    h_prev = ti.field(dtype=ti.f32, shape=(N, N))
    h_curr = ti.field(dtype=ti.f32, shape=(N, N))

    @ti.kernel
    def init():
        for i, j in h_curr:
            if (i - N//4)**2 + (j - N//4)**2 < (N//16)**2:
                h_curr[i, j] = 1.0

    @ti.kernel
    def wave_step():
        for i, j in ti.ndrange((1, N-1), (1, N-1)):
            lap = (h_curr[i-1,j] + h_curr[i+1,j] + h_curr[i,j-1] + h_curr[i,j+1] - 4.0*h_curr[i,j])
            h_new = 2.0 * h_curr[i,j] - h_prev[i,j] + coeff * lap
            h_prev[i,j] = h_curr[i,j]
            h_curr[i,j] = h_new

    init()
    def step():
        for _ in range(steps):
            wave_step()
    return step, ti.sync, h_curr
