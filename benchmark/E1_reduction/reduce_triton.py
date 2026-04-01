"""Global reduction (sum) — Triton implementation.

Triton has tl.sum as a first-class op → warp-level shuffle reduction.
Then atomic_add across CTAs.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def reduce_sum_kernel(
    data_ptr,
    result_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(data_ptr + offsets, mask=mask, other=0.0)
    partial = tl.sum(x, axis=0)  # warp-level reduction
    tl.atomic_add(result_ptr, partial)


def run(N, backend="cuda"):
    assert backend == "cuda", "Triton only supports CUDA"

    data = torch.randn(N, dtype=torch.float32, device="cuda")
    result = torch.zeros(1, dtype=torch.float32, device="cuda")

    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)

    def step():
        result.zero_()
        reduce_sum_kernel[grid](data, result, N, BLOCK)

    def sync():
        torch.cuda.synchronize()

    return step, sync, result
