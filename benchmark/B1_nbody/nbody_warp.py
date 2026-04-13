"""N-body (direct all-pairs) — Warp.
Adapted from warp/examples/tile/example_tile_nbody.py
"""
import numpy as np
import warp as wp

G = 1.0
SOFTENING = 1e-5


@wp.struct
class NBodyMesh:
    pos: wp.array(dtype=wp.vec3)
    vel: wp.array(dtype=wp.vec3)
    force: wp.array(dtype=wp.vec3)
    dt: float
    N: int


@wp.kernel
def compute_force_kernel(m: NBodyMesh):
    pos = m.pos
    force = m.force
    N = m.N
    i = wp.tid()
    if i >= N:
        return
    pi = pos[i]
    f = wp.vec3(0.0, 0.0, 0.0)
    for j in range(N):
        if j != i:
            diff = pos[j] - pi
            dist = wp.length(diff) + SOFTENING
            f = f + G / (dist * dist * dist) * diff
    force[i] = f

@wp.kernel
def integrate_kernel(m: NBodyMesh):
    pos = m.pos
    vel = m.vel
    force = m.force
    dt = m.dt
    N = m.N
    i = wp.tid()
    if i >= N:
        return
    vel[i] = vel[i] + force[i] * dt
    pos[i] = pos[i] + vel[i] * dt

def run(N, steps=1, backend="cuda"):
    dt = 0.001
    # Deterministic init matching Taichi: golden-ratio quasi-random
    g = 1.618033988749895
    i = np.arange(N, dtype=np.float32)
    pos_np = np.stack([(i * g) % 1.0 - 0.5,
                       (i * 7 * g) % 1.0 - 0.5,
                       (i * 13 * g) % 1.0 - 0.5], axis=1).astype(np.float32)

    mesh = NBodyMesh()
    mesh.pos = wp.array(pos_np, dtype=wp.vec3, device=backend)
    mesh.vel = wp.zeros(N, dtype=wp.vec3, device=backend)
    mesh.force = wp.zeros(N, dtype=wp.vec3, device=backend)
    mesh.dt = dt
    mesh.N = N

    def step_fn():
        for _ in range(steps):
            wp.launch(compute_force_kernel, dim=N, inputs=[mesh], device=backend)
            wp.launch(integrate_kernel, dim=N, inputs=[mesh], device=backend)

    def sync():
        wp.synchronize_device(backend)

    return step_fn, sync, mesh.pos
