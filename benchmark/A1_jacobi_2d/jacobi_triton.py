"""Jacobi 2D 5-point stencil — Triton implementation.

This demonstrates that Triton CAN express a stencil kernel,
but without automatic shared memory promotion for halo reuse.
Each of the 4 neighbor loads is an independent global memory access.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def jacobi_kernel(
    u_ptr, u_new_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # 2D block index
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    # Global coordinates for this block
    i_base = pid_i * BLOCK
    j_base = pid_j * BLOCK

    i_offsets = i_base + tl.arange(0, BLOCK)
    j_offsets = j_base + tl.arange(0, BLOCK)

    # 2D mask: interior only (skip boundaries)
    i_idx = i_offsets[:, None]  # (BLOCK, 1)
    j_idx = j_offsets[None, :]  # (1, BLOCK)

    mask = (i_idx >= 1) & (i_idx < N - 1) & (j_idx >= 1) & (j_idx < N - 1)

    # Load 4 neighbors — each is an independent global memory access
    # No shared memory reuse of overlapping halo data
    center_offset = i_idx * N + j_idx
    up    = tl.load(u_ptr + (i_idx - 1) * N + j_idx, mask=mask, other=0.0)
    down  = tl.load(u_ptr + (i_idx + 1) * N + j_idx, mask=mask, other=0.0)
    left  = tl.load(u_ptr + i_idx * N + (j_idx - 1), mask=mask, other=0.0)
    right = tl.load(u_ptr + i_idx * N + (j_idx + 1), mask=mask, other=0.0)

    result = 0.25 * (up + down + left + right)
    tl.store(u_new_ptr + center_offset, result, mask=mask)


def run(N, steps=1, backend="cuda"):
    assert backend == "cuda", "Triton only supports CUDA"

    u = torch.zeros(N, N, dtype=torch.float32, device="cuda")
    u[0, :] = 1.0
    u_new = u.clone()

    BLOCK = 32
    grid = (triton.cdiv(N, BLOCK), triton.cdiv(N, BLOCK))

    def step():
        for _ in range(steps):
            jacobi_kernel[grid](u, u_new, N, BLOCK)
            u.copy_(u_new)

    def sync():
        torch.cuda.synchronize()

    return step, sync, u
