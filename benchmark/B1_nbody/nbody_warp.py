"""N-body (direct all-pairs) — Warp.
Aligned with nbody_benchmark.cu: srand(42) positions in [-1,1]^3,
unit mass, EPS2=0.01, DT=0.001."""
import numpy as np
import warp as wp

EPS2 = 0.01  # softening squared, matching CUDA


@wp.struct
class NBodyMesh:
    pos: wp.array(dtype=wp.vec3)
    vel: wp.array(dtype=wp.vec3)
    acc: wp.array(dtype=wp.vec3)
    dt: float
    N: int


@wp.kernel
def compute_force_kernel(m: NBodyMesh):
    pos = m.pos
    acc = m.acc
    N = m.N
    i = wp.tid()
    if i >= N:
        return
    pi = pos[i]
    a = wp.vec3(0.0, 0.0, 0.0)
    for j in range(N):
        diff = pos[j] - pi
        r2 = wp.dot(diff, diff) + EPS2
        inv_r3 = 1.0 / (r2 * wp.sqrt(r2))
        a = a + diff * inv_r3  # unit mass
    acc[i] = a

@wp.kernel
def integrate_kernel(m: NBodyMesh):
    pos = m.pos
    vel = m.vel
    acc = m.acc
    dt = m.dt
    N = m.N
    i = wp.tid()
    if i >= N:
        return
    vel[i] = vel[i] + acc[i] * dt
    pos[i] = pos[i] + vel[i] * dt

def run(N, steps=1, backend="cuda"):
    # Match CUDA srand(42) init
    rng = np.random.RandomState(42)
    pos_np = (rng.rand(N, 3).astype(np.float32) * 2 - 1)

    mesh = NBodyMesh()
    mesh.pos = wp.array(pos_np, dtype=wp.vec3, device=backend)
    mesh.vel = wp.zeros(N, dtype=wp.vec3, device=backend)
    mesh.acc = wp.zeros(N, dtype=wp.vec3, device=backend)
    mesh.dt = 0.001
    mesh.N = N

    def step_fn():
        for _ in range(steps):
            wp.launch(compute_force_kernel, dim=N, inputs=[mesh], device=backend)
            wp.launch(integrate_kernel, dim=N, inputs=[mesh], device=backend)

    def sync():
        wp.synchronize_device(backend)

    return step_fn, sync, mesh.pos
