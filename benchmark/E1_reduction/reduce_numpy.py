"""Global reduction (sum) — NumPy baseline."""

import numpy as np


def run(N, backend="cpu"):
    data = np.random.randn(N).astype(np.float32)

    def step():
        _ = np.sum(data)

    return step, None, data
