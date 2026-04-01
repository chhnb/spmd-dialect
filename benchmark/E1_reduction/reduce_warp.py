"""Global reduction (sum) — Warp implementation using tile operations."""

import numpy as np
import warp as wp

TILE_SIZE = 256


@wp.kernel
def reduce_sum_kernel(
    data: wp.array(dtype=float),
    result: wp.array(dtype=float),
    N: int,
):
    tid = wp.tid()
    if tid < N:
        wp.atomic_add(result, 0, data[tid])


def run(N, backend="cuda"):
    device = backend

    data_np = np.random.randn(N).astype(np.float32)
    data = wp.array(data_np, dtype=float, device=device)
    result = wp.zeros(1, dtype=float, device=device)

    def step():
        result.zero_()
        wp.launch(reduce_sum_kernel, dim=N, inputs=[data, result, N], device=device)

    def sync():
        wp.synchronize_device(device)

    return step, sync, result
