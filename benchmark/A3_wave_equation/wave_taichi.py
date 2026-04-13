"""2D Wave Equation — Taichi. h_next = 2h_curr - h_prev + dt^2 * c^2 * laplacian(h_curr)
Uses 3-buffer rotation matching the CUDA scheme to avoid race conditions."""
import taichi as ti

def run(N, steps=1, backend="cuda"):
    ti.init(arch=ti.cuda if backend == "cuda" else ti.cpu, default_fp=ti.f32)
    c = 1.0; dt = 0.1; dx = 1.0
    coeff = (c * dt / dx) ** 2

    h0 = ti.field(dtype=ti.f32, shape=(N, N))
    h1 = ti.field(dtype=ti.f32, shape=(N, N))
    h2 = ti.field(dtype=ti.f32, shape=(N, N))
    h_out = ti.field(dtype=ti.f32, shape=(N, N))  # persistent output

    @ti.kernel
    def init():
        for i, j in h1:
            if (i - N//4)**2 + (j - N//4)**2 < (N//16)**2:
                h1[i, j] = 1.0

    @ti.kernel
    def wave_step_01(h_prev: ti.template(), h_curr: ti.template(), h_next: ti.template()):
        for i, j in ti.ndrange((1, N-1), (1, N-1)):
            lap = (h_curr[i-1,j] + h_curr[i+1,j] + h_curr[i,j-1] + h_curr[i,j+1] - 4.0*h_curr[i,j])
            h_next[i,j] = 2.0 * h_curr[i,j] - h_prev[i,j] + coeff * lap

    @ti.kernel
    def copy_field(src: ti.template(), dst: ti.template()):
        for i, j in src:
            dst[i, j] = src[i, j]

    init()

    bufs = [h0, h1, h2]
    rot = [0, 1, 2]  # prev=h0, curr=h1, next=h2

    def step():
        for _ in range(steps):
            wave_step_01(bufs[rot[0]], bufs[rot[1]], bufs[rot[2]])
            rot[0], rot[1], rot[2] = rot[1], rot[2], rot[0]
        # Export current buffer to persistent output
        copy_field(bufs[rot[1]], h_out)

    return step, ti.sync, h_out
