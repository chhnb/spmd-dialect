"""2D DEM with uniform-grid neighbor search — Taichi."""
import taichi as ti


def run(N=8192, steps=1, backend="cuda"):
    ti.init(arch=ti.cuda if backend == "cuda" else ti.cpu, default_fp=ti.f32)
    domain = 64.0
    radius = 0.25
    stiffness = 200.0
    dt = 0.002
    grid_size = 128
    cell_size = domain / grid_size
    max_per_cell = 64

    pos = ti.Vector.field(2, dtype=ti.f32, shape=N)
    vel = ti.Vector.field(2, dtype=ti.f32, shape=N)
    force = ti.Vector.field(2, dtype=ti.f32, shape=N)
    grid_count = ti.field(dtype=ti.i32, shape=(grid_size, grid_size))
    grid_entries = ti.field(dtype=ti.i32, shape=(grid_size, grid_size, max_per_cell))

    @ti.kernel
    def init():
        for p in pos:
            pos[p] = ti.Vector([ti.random() * (domain * 0.6) + domain * 0.2,
                                ti.random() * (domain * 0.6) + domain * 0.2])
            vel[p] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def build_grid():
        for i, j in grid_count:
            grid_count[i, j] = 0
        for p in pos:
            ci = ti.max(0, ti.min(ti.cast(pos[p][0] / cell_size, ti.i32), grid_size - 1))
            cj = ti.max(0, ti.min(ti.cast(pos[p][1] / cell_size, ti.i32), grid_size - 1))
            slot = ti.atomic_add(grid_count[ci, cj], 1)
            if slot < max_per_cell:
                grid_entries[ci, cj, slot] = p

    @ti.kernel
    def contact_and_integrate():
        for p in pos:
            pi = pos[p]
            ci = ti.cast(pi[0] / cell_size, ti.i32)
            cj = ti.cast(pi[1] / cell_size, ti.i32)
            f = ti.Vector([0.0, -9.8])
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni = ci + di
                    nj = cj + dj
                    if 0 <= ni < grid_size and 0 <= nj < grid_size:
                        for k in range(grid_count[ni, nj]):
                            q = grid_entries[ni, nj, k]
                            if q != p:
                                d = pi - pos[q]
                                dist = d.norm() + 1e-5
                                overlap = 2.0 * radius - dist
                                if overlap > 0:
                                    n = d / dist
                                    rel = vel[p] - vel[q]
                                    f += stiffness * overlap * n - 0.5 * rel
            force[p] = f
        for p in pos:
            vel[p] += dt * force[p]
            pos[p] += dt * vel[p]
            if pos[p][0] < radius:
                pos[p][0] = radius; vel[p][0] *= -0.5
            if pos[p][0] > domain - radius:
                pos[p][0] = domain - radius; vel[p][0] *= -0.5
            if pos[p][1] < radius:
                pos[p][1] = radius; vel[p][1] *= -0.5
            if pos[p][1] > domain - radius:
                pos[p][1] = domain - radius; vel[p][1] *= -0.5

    init()

    def step_fn():
        for _ in range(steps):
            build_grid()
            contact_and_integrate()

    return step_fn, ti.sync, pos
