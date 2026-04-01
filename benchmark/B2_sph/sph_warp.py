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

@wp.kernel
def compute_density_kernel(
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    rho: wp.array(dtype=float),
    h: float,
    N: int,
):
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

    pos_np = (np.random.rand(N, 3).astype(np.float32) * domain * 0.5 + domain * 0.1)
    pos_np[:, 2] = 0.0  # 2D in XY plane

    pos = wp.array(pos_np, dtype=wp.vec3, device=backend)
    rho = wp.zeros(N, dtype=float, device=backend)

    grid = wp.HashGrid(dim_x=128, dim_y=128, dim_z=1, device=backend)

    def step_fn():
        for _ in range(steps):
            grid.build(pos, h)
            wp.launch(compute_density_kernel, dim=N,
                      inputs=[grid.id, pos, rho, h, N], device=backend)

    def sync():
        wp.synchronize_device(backend)

    return step_fn, sync, rho
