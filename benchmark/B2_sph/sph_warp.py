"""SPH density computation — Warp.
Aligned with sph_benchmark.cu: H=0.05, DOMAIN=1.0, libc srand(42) init,
3 kernels/step: build_grid + compute_density + jitter_positions.
Uses unnormalized poly6 + 4/(pi*h^8) normalization matching CUDA."""
import numpy as np
import warp as wp

H = 0.05
H2 = H * H
DOMAIN = 1.0
DT_SPH = 0.0001
MASS = 1.0


@wp.struct
class SPHMesh:
    pos: wp.array(dtype=wp.vec2)
    rho: wp.array(dtype=float)
    grid_count: wp.array2d(dtype=int)
    grid_entries: wp.array3d(dtype=int)
    grid_x: int
    grid_y: int
    max_per_cell: int
    N: int
    cell_size: float


@wp.kernel
def build_grid_kernel(m: SPHMesh):
    p = wp.tid()
    if p < m.N:
        pi = m.pos[p]
        ci = int(pi[0] / m.cell_size)
        cj = int(pi[1] / m.cell_size)
        ci = wp.max(0, wp.min(ci, m.grid_x - 1))
        cj = wp.max(0, wp.min(cj, m.grid_y - 1))
        idx = wp.atomic_add(m.grid_count, ci, cj, 1)
        if idx < m.max_per_cell:
            m.grid_entries[ci, cj, idx] = p


@wp.kernel
def compute_density_kernel(m: SPHMesh):
    p = wp.tid()
    if p < m.N:
        pi = m.pos[p]
        ci = int(pi[0] / m.cell_size)
        cj = int(pi[1] / m.cell_size)
        density = float(0.0)
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ni = ci + di
                nj = cj + dj
                if ni >= 0 and ni < m.grid_x and nj >= 0 and nj < m.grid_y:
                    cnt = m.grid_count[ni, nj]
                    for k in range(cnt):
                        if k < m.max_per_cell:
                            q = m.grid_entries[ni, nj, k]
                            diff = pi - m.pos[q]
                            r2 = wp.dot(diff, diff)
                            if r2 < H2:
                                d = H2 - r2
                                density += MASS * d * d * d
        # Normalize: 4/(pi*h^8) matching CUDA
        h8 = H2 * H2 * H2 * H2
        m.rho[p] = density * 4.0 / (3.14159265 * h8)


@wp.kernel
def jitter_kernel(pos: wp.array(dtype=wp.vec2), N: int, step_val: int):
    i = wp.tid()
    if i < N:
        dx = wp.sin(float(i + step_val * 7)) * DT_SPH
        dy = wp.cos(float(i + step_val * 13)) * DT_SPH
        x = wp.min(wp.max(pos[i][0] + dx, 0.001), DOMAIN - 0.001)
        y = wp.min(wp.max(pos[i][1] + dy, 0.001), DOMAIN - 0.001)
        pos[i] = wp.vec2(x, y)


def run(N, steps=1, backend="cuda"):
    cell_size = H
    grid_x = int(DOMAIN / cell_size)
    grid_y = int(DOMAIN / cell_size)
    max_per_cell = 64

    import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from crand_init import sph_init
    pos_np = sph_init(N, seed=42, domain=DOMAIN)

    mesh = SPHMesh()
    mesh.pos = wp.array(pos_np, dtype=wp.vec2, device=backend)
    mesh.rho = wp.zeros(N, dtype=float, device=backend)
    mesh.grid_count = wp.zeros((grid_x, grid_y), dtype=int, device=backend)
    mesh.grid_entries = wp.zeros((grid_x, grid_y, max_per_cell), dtype=int, device=backend)
    mesh.grid_x = grid_x
    mesh.grid_y = grid_y
    mesh.max_per_cell = max_per_cell
    mesh.N = N
    mesh.cell_size = cell_size
    step_val = [0]

    def step_fn():
        for _ in range(steps):
            # Clear grid via fill (not a separate kernel launch)
            mesh.grid_count.zero_()
            # 3 kernels matching CUDA: build_grid + compute_density + jitter
            wp.launch(build_grid_kernel, dim=N, inputs=[mesh], device=backend)
            wp.launch(compute_density_kernel, dim=N, inputs=[mesh], device=backend)
            wp.launch(jitter_kernel, dim=N, inputs=[mesh.pos, N, step_val[0]], device=backend)
            step_val[0] += 1

    def sync():
        wp.synchronize_device(backend)

    return step_fn, sync, mesh.rho
