"""N-body (direct all-pairs) — Taichi.
Aligned with nbody_benchmark.cu: srand(42) positions in [-1,1]^3,
unit mass, EPS2=0.01, DT=0.001, 2 kernels/step (compute_force + integrate).
"""
import taichi as ti
import numpy as np

EPS2 = 0.01  # softening squared, matching CUDA
DT = 0.001

def run(N, steps=1, backend="cuda"):
    ti.init(arch=ti.cuda if backend == "cuda" else ti.cpu, default_fp=ti.f32)

    pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
    vel = ti.Vector.field(3, dtype=ti.f32, shape=N)
    acc = ti.Vector.field(3, dtype=ti.f32, shape=N)

    # Init matching CUDA srand(42) exactly via libc rand()
    import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from crand_init import nbody_init
    pos_np = nbody_init(N, seed=42)

    @ti.kernel
    def init():
        for i in pos:
            vel[i] = ti.Vector([0.0, 0.0, 0.0])
            acc[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def compute_force():
        for i in range(N):
            a = ti.Vector([0.0, 0.0, 0.0])
            for j in range(N):
                diff = pos[j] - pos[i]
                r2 = diff.dot(diff) + EPS2
                inv_r3 = 1.0 / (r2 * ti.sqrt(r2))
                a += diff * inv_r3  # unit mass
            acc[i] = a

    @ti.kernel
    def integrate():
        for i in range(N):
            vel[i] += acc[i] * DT
            pos[i] += vel[i] * DT

    pos.from_numpy(pos_np)
    init()

    def step_fn():
        for _ in range(steps):
            compute_force()
            integrate()

    return step_fn, ti.sync, pos
