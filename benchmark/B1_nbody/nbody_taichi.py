"""N-body (direct all-pairs) — Taichi.
Adapted from taichi/examples/simulation/nbody.py
"""
import taichi as ti
import numpy as np

G = 1.0
SOFTENING = 1e-5

def run(N, steps=1, backend="cuda"):
    ti.init(arch=ti.cuda if backend == "cuda" else ti.cpu, default_fp=ti.f32)
    dt = 0.001

    pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
    vel = ti.Vector.field(3, dtype=ti.f32, shape=N)
    force = ti.Vector.field(3, dtype=ti.f32, shape=N)

    @ti.kernel
    def init():
        for i in pos:
            pos[i] = ti.Vector([ti.random()-0.5, ti.random()-0.5, ti.random()-0.5])
            vel[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def compute_force():
        for i in range(N):
            f = ti.Vector([0.0, 0.0, 0.0])
            for j in range(N):
                if i != j:
                    diff = pos[j] - pos[i]
                    dist = diff.norm() + SOFTENING
                    f += G / (dist * dist * dist) * diff
            force[i] = f

    @ti.kernel
    def integrate():
        for i in range(N):
            vel[i] += force[i] * dt
            pos[i] += vel[i] * dt

    init()

    def step_fn():
        for _ in range(steps):
            compute_force()
            integrate()

    return step_fn, ti.sync, pos
