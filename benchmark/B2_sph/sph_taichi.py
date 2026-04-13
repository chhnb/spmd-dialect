"""SPH density computation — Taichi.
Aligned with sph_benchmark.cu: H=0.05, DOMAIN=1.0, srand(42) init,
3 kernels/step: build_grid + compute_density + jitter_positions."""
import taichi as ti
import numpy as np

H = 0.05
H2 = H * H
DOMAIN = 1.0
MASS = 1.0
DT_SPH = 0.0001
POLY6_COEFF = 315.0 / (64.0 * np.pi)

def run(N, steps=1, backend="cuda"):
    ti.init(arch=ti.cuda if backend == "cuda" else ti.cpu, default_fp=ti.f32)

    cell_size = H
    grid_x = int(DOMAIN / cell_size)
    grid_y = int(DOMAIN / cell_size)
    max_per_cell = 64

    pos = ti.Vector.field(2, dtype=ti.f32, shape=N)
    rho = ti.field(dtype=ti.f32, shape=N)
    grid_count = ti.field(dtype=ti.i32, shape=(grid_x, grid_y))
    grid_entries = ti.field(dtype=ti.i32, shape=(grid_x, grid_y, max_per_cell))
    step_counter = ti.field(dtype=ti.i32, shape=())

    # Init matching CUDA srand(42) exactly via libc rand()
    import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from crand_init import sph_init
    pos_np = sph_init(N, seed=42, domain=DOMAIN)
    pos.from_numpy(pos_np)

    @ti.func
    def poly6_unnorm(r2_val: ti.f32) -> ti.f32:
        """Unnormalized poly6: (h²-r²)³. Matches CUDA sph_benchmark.cu."""
        result = ti.cast(0.0, ti.f32)
        if r2_val < H2:
            d = H2 - r2_val
            result = d * d * d
        return result

    @ti.kernel
    def build_grid():
        for i, j in grid_count:
            grid_count[i, j] = 0
        for p in pos:
            ci = ti.cast(pos[p][0] / cell_size, ti.i32)
            cj = ti.cast(pos[p][1] / cell_size, ti.i32)
            ci = ti.max(0, ti.min(ci, grid_x - 1))
            cj = ti.max(0, ti.min(cj, grid_y - 1))
            idx = ti.atomic_add(grid_count[ci, cj], 1)
            if idx < max_per_cell:
                grid_entries[ci, cj, idx] = p

    @ti.kernel
    def compute_density():
        # Match CUDA: unnormalized poly6 + 4/(pi*h^8) normalization
        h8 = ti.cast(H2 * H2 * H2 * H2, ti.f32)
        norm = 4.0 / (3.14159265 * h8)
        for p in pos:
            pi = pos[p]
            ci = ti.cast(pi[0] / cell_size, ti.i32)
            cj = ti.cast(pi[1] / cell_size, ti.i32)
            density = ti.cast(0.0, ti.f32)
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni = ci + di
                    nj = cj + dj
                    if ni >= 0 and ni < grid_x and nj >= 0 and nj < grid_y:
                        for k in range(grid_count[ni, nj]):
                            q = grid_entries[ni, nj, k]
                            diff = pi - pos[q]
                            r2 = diff.dot(diff)
                            density += ti.cast(MASS, ti.f32) * poly6_unnorm(r2)
            rho[p] = density * norm

    @ti.kernel
    def jitter_positions(step_val: ti.i32):
        for i in pos:
            dx = ti.sin(ti.cast(i + step_val * 7, ti.f32)) * ti.cast(DT_SPH, ti.f32)
            dy = ti.cos(ti.cast(i + step_val * 13, ti.f32)) * ti.cast(DT_SPH, ti.f32)
            x = ti.min(ti.max(pos[i][0] + dx, 0.001), ti.cast(DOMAIN, ti.f32) - 0.001)
            y = ti.min(ti.max(pos[i][1] + dy, 0.001), ti.cast(DOMAIN, ti.f32) - 0.001)
            pos[i] = ti.Vector([x, y])

    step_counter[None] = 0

    def step_fn():
        for _ in range(steps):
            build_grid()
            compute_density()
            jitter_positions(step_counter[None])
            step_counter[None] += 1

    return step_fn, ti.sync, rho
