"""SPH density computation — Warp with HashGrid.
Adapted from warp/examples/core/example_sph.py
"""
import numpy as np
import warp as wp

KERNEL_RADIUS = 1.0
POLY6_COEFF = 315.0 / (64.0 * 3.14159265)

@wp.func
def poly6(r: float, h: float) -> float:
    result = float(0.0)
    if r < h:
        x = (h * h - r * r) / (h * h * h)
        result = POLY6_COEFF * x * x * x
    return result


@wp.struct
class SPHMesh:
    grid: wp.uint64
    pos: wp.array(dtype=wp.vec3)
    rho: wp.array(dtype=float)
    h: float
    N: int


@wp.kernel
def compute_density_kernel(m: SPHMesh):
    grid = m.grid
    pos = m.pos
    rho = m.rho
    h = m.h
    N = m.N
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i >= N:
        return
    pi = pos[i]
    density = float(0.0)

    neighbors = wp.hash_grid_query(grid, pi, h)
    for idx in neighbors:
        dist = wp.length(pi - pos[idx])
        density = density + poly6(dist, h)

    rho[i] = density

def run(N, steps=1, backend="cuda"):
    h = KERNEL_RADIUS
    domain = 32.0

    # Deterministic init matching Taichi: golden-ratio quasi-random
    g = 1.618033988749895
    idx = np.arange(N, dtype=np.float32)
    pos_np = np.stack([(idx * g % 1.0) * domain * 0.5 + domain * 0.1,
                       (idx * 7 * g % 1.0) * domain * 0.5 + domain * 0.1,
                       np.zeros(N, dtype=np.float32)], axis=1).astype(np.float32)

    pos = wp.array(pos_np, dtype=wp.vec3, device=backend)
    rho = wp.zeros(N, dtype=float, device=backend)

    grid = wp.HashGrid(dim_x=128, dim_y=128, dim_z=1, device=backend)

    mesh = SPHMesh()
    mesh.pos = pos
    mesh.rho = rho
    mesh.h = h
    mesh.N = N

    def step_fn():
        for _ in range(steps):
            grid.build(pos, h)
            mesh.grid = grid.id
            wp.launch(compute_density_kernel, dim=N,
                      inputs=[mesh], device=backend)

    def sync():
        wp.synchronize_device(backend)

    return step_fn, sync, rho
