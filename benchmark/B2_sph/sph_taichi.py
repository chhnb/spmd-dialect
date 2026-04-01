"""SPH density computation — Taichi.
Adapted from taichi/examples/simulation/pbf2d.py
"""
import taichi as ti
import numpy as np

KERNEL_RADIUS = 1.0
POLY6_COEFF = 315.0 / (64.0 * np.pi)  # 2D poly6 coefficient

def run(N, steps=1, backend="cuda"):
    ti.init(arch=ti.cuda if backend == "cuda" else ti.cpu, default_fp=ti.f32)

    h = KERNEL_RADIUS
    grid_size = 64
    cell_size = h
    max_neighbors = 64

    pos = ti.Vector.field(2, dtype=ti.f32, shape=N)
    vel = ti.Vector.field(2, dtype=ti.f32, shape=N)
    rho = ti.field(dtype=ti.f32, shape=N)

    # Grid for neighbor search
    grid_count = ti.field(dtype=ti.i32, shape=(grid_size, grid_size))
    grid_entries = ti.field(dtype=ti.i32, shape=(grid_size, grid_size, 64))

    @ti.func
    def poly6(r, h):
        result = 0.0
        if r < h:
            x = (h * h - r * r) / (h * h * h)
            result = POLY6_COEFF * x * x * x
        return result

    @ti.kernel
    def init():
        for i in pos:
            pos[i] = ti.Vector([ti.random() * grid_size * 0.5 + grid_size * 0.1,
                                ti.random() * grid_size * 0.5 + grid_size * 0.1])

    @ti.kernel
    def build_grid():
        for i, j in grid_count:
            grid_count[i, j] = 0
        for p in pos:
            ci = ti.cast(pos[p][0] / cell_size, ti.i32)
            cj = ti.cast(pos[p][1] / cell_size, ti.i32)
            ci = ti.max(0, ti.min(ci, grid_size - 1))
            cj = ti.max(0, ti.min(cj, grid_size - 1))
            idx = ti.atomic_add(grid_count[ci, cj], 1)
            if idx < 64:
                grid_entries[ci, cj, idx] = p

    @ti.kernel
    def compute_density():
        for p in pos:
            pi = pos[p]
            ci = ti.cast(pi[0] / cell_size, ti.i32)
            cj = ti.cast(pi[1] / cell_size, ti.i32)
            density = 0.0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni = ci + di
                    nj = cj + dj
                    if ni >= 0 and ni < grid_size and nj >= 0 and nj < grid_size:
                        for k in range(grid_count[ni, nj]):
                            q = grid_entries[ni, nj, k]
                            dist = (pi - pos[q]).norm()
                            density += poly6(dist, h)
            rho[p] = density

    init()

    def step_fn():
        for _ in range(steps):
            build_grid()
            compute_density()

    return step_fn, ti.sync, rho
