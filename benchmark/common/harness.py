"""Common benchmark harness for simulation kernel benchmarks."""

import csv
import os
import time
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    kernel: str
    framework: str
    backend: str  # "cpu" or "cuda"
    problem_size: str  # e.g. "4096x4096"
    warmup_runs: int = 5
    timed_runs: int = 20
    times_ms: list = field(default_factory=list)

    @property
    def min_ms(self):
        return min(self.times_ms) if self.times_ms else 0.0

    @property
    def max_ms(self):
        return max(self.times_ms) if self.times_ms else 0.0

    @property
    def avg_ms(self):
        return sum(self.times_ms) / len(self.times_ms) if self.times_ms else 0.0

    @property
    def median_ms(self):
        if not self.times_ms:
            return 0.0
        s = sorted(self.times_ms)
        n = len(s)
        return (s[n // 2] + s[(n - 1) // 2]) / 2.0

    def summary(self):
        return (
            f"{self.framework:12s} {self.backend:5s} {self.problem_size:>12s}  "
            f"min={self.min_ms:8.3f}ms  median={self.median_ms:8.3f}ms  "
            f"avg={self.avg_ms:8.3f}ms  max={self.max_ms:8.3f}ms"
        )


class Timer:
    """Simple CUDA-aware timer."""

    def __init__(self):
        self._start = None

    def start(self):
        self._start = time.perf_counter()

    def stop(self):
        return (time.perf_counter() - self._start) * 1000.0  # ms


def run_kernel_benchmark(
    kernel_fn,
    sync_fn=None,
    warmup=5,
    repeat=20,
):
    """Run a kernel function with warmup and timing.

    Args:
        kernel_fn: callable that runs one iteration of the kernel
        sync_fn: callable for device synchronization (e.g. cuda sync), or None
        warmup: number of warmup iterations
        repeat: number of timed iterations

    Returns:
        list of times in ms
    """
    # warmup
    for _ in range(warmup):
        kernel_fn()
        if sync_fn:
            sync_fn()

    # timed runs
    timer = Timer()
    times = []
    for _ in range(repeat):
        if sync_fn:
            sync_fn()
        timer.start()
        kernel_fn()
        if sync_fn:
            sync_fn()
        elapsed = timer.stop()
        times.append(elapsed)

    return times


def save_results(results: list, output_path: str = "results.csv"):
    """Save benchmark results to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "kernel", "framework", "backend", "problem_size",
            "min_ms", "median_ms", "avg_ms", "max_ms", "runs",
        ])
        for r in results:
            writer.writerow([
                r.kernel, r.framework, r.backend, r.problem_size,
                f"{r.min_ms:.3f}", f"{r.median_ms:.3f}",
                f"{r.avg_ms:.3f}", f"{r.max_ms:.3f}",
                len(r.times_ms),
            ])
    print(f"Results saved to {output_path}")


def print_results(results: list):
    """Print results table to stdout."""
    print()
    print("=" * 80)
    for r in results:
        print(r.summary())
    print("=" * 80)
    print()
