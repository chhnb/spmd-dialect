"""Jacobi 2D 5-point stencil — TileLang implementation.

Uses T.Kernel with explicit block/thread bindings for element-wise stencil.
Requires CUDA_HOME to be set for TileLang's TVM-based JIT compilation.
"""
import os
os.environ.setdefault("CUDA_HOME", "/home/scratch.huanhuanc_gpu/spmd/cuda-toolkit")

import numpy as np
import torch
import tilelang
import tilelang.language as T


BLOCK = 256


def make_kernels(N):
    @T.prim_func
    def jacobi_step(
        u: T.Buffer((N * N,), "float32"),
        u_new: T.Buffer((N * N,), "float32"),
    ):
        with T.Kernel(T.ceildiv(N * N, BLOCK), threads=BLOCK):
            bx = T.get_block_binding(0)
            tx = T.get_thread_binding(0)
            idx = bx * BLOCK + tx
            if idx < N * N:
                i = idx // N
                j = idx % N
                if i >= 1 and i < N - 1 and j >= 1 and j < N - 1:
                    u_new[idx] = T.float32(0.25) * (
                        u[(i - 1) * N + j] + u[(i + 1) * N + j] +
                        u[i * N + (j - 1)] + u[i * N + (j + 1)]
                    )
                else:
                    u_new[idx] = u[idx]

    @T.prim_func
    def copy_kernel(
        src: T.Buffer((N * N,), "float32"),
        dst: T.Buffer((N * N,), "float32"),
    ):
        with T.Kernel(T.ceildiv(N * N, BLOCK), threads=BLOCK):
            bx = T.get_block_binding(0)
            tx = T.get_thread_binding(0)
            idx = bx * BLOCK + tx
            if idx < N * N:
                dst[idx] = src[idx]

    return jacobi_step, copy_kernel


def run(N=64, steps=1, backend="cuda"):
    jacobi_step_fn, copy_fn = make_kernels(N)
    step_mod = tilelang.JITKernel(jacobi_step_fn, out_idx=[1])
    copy_mod = tilelang.JITKernel(copy_fn, out_idx=[1])

    # Init: top row = 1.0
    u_np = np.zeros((N * N,), dtype=np.float32)
    u_np[:N] = 1.0
    u = torch.tensor(u_np, device="cuda", dtype=torch.float32)

    def step_fn():
        nonlocal u
        for _ in range(steps):
            u_new = step_mod(u)
            u = copy_mod(u_new)

    def sync_fn():
        torch.cuda.synchronize()

    return step_fn, sync_fn, u
