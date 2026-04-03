# F1 Hydro Shallow Water — Benchmark Results

**GPU**: NVIDIA B200 (sm_100, 178GB HBM3e, ~8TB/s)
**Date**: 2026-04-02
**Kernel**: Osher Riemann solver, dam-break on NxN structured quad mesh, 10 time steps

## 1. Full Comparison: fp64 (double precision)

| N | Cells | CUDA naive | Kokkos | Taichi | Triton |
|---|-------|-----------|--------|--------|--------|
| 512 | 262K | 0.455 ms | 0.461 ms | 0.427 ms | 3.577 ms |
| 1024 | 1M | 1.794 ms | 1.636 ms | 1.733 ms | 13.533 ms |
| 2048 | 4M | 7.024 ms | 6.083 ms | 5.373 ms | 53.232 ms |
| 4096 | 16M | 27.280 ms | **23.768 ms** | 25.970 ms | 207.795 ms |

**Ranking at N=4096 (fp64):**
1. Kokkos — 23.8ms (fastest)
2. Taichi — 26.0ms (1.09x)
3. CUDA naive — 27.3ms (1.15x)
4. Triton — 207.8ms (8.74x slower)

## 2. Full Comparison: fp32 (single precision)

| N | Cells | Kokkos | Taichi | Warp |
|---|-------|--------|--------|------|
| 512 | 262K | 0.275 ms | 0.328 ms | 0.360 ms |
| 1024 | 1M | 0.929 ms | 1.266 ms | 0.963 ms |
| 2048 | 4M | 3.431 ms | 3.587 ms | 3.573 ms |
| 4096 | 16M | **13.326 ms** | 20.253 ms | 13.970 ms |

**Ranking at N=4096 (fp32):**
1. Kokkos — 13.3ms (fastest)
2. Warp — 14.0ms (1.05x)
3. Taichi — 20.3ms (1.52x)

## 3. fp64 vs fp32 Speedup

| N=4096 | fp64 (ms) | fp32 (ms) | Speedup |
|--------|----------|----------|---------|
| Kokkos | 23.768 | 13.326 | **1.78x** |
| Taichi | 25.970 | 20.253 | 1.28x |

## 4. Gather Promotion Experiment (CUDA, fp64)

| N | Naive (ms) | Promoted (ms) | Speedup |
|---|-----------|---------------|---------|
| 64 | 1.314 | 1.440 | 0.91x |
| 128 | 1.427 | 1.441 | 0.99x |
| 256 | 1.442 | 1.645 | 0.88x |
| 512 | 4.320 | 5.127 | 0.84x |

**Result: No benefit. Shared memory gather promotion is slower on B200.**

## 5. Correctness

| Comparison | Max abs diff | Status |
|-----------|-------------|--------|
| Taichi fp64 vs NumPy (20 steps, 64x64) | 8.88e-16 | PASS |
| Warp fp32 vs NumPy (20 steps, 64x64) | 6.00e-07 | PASS (fp32) |
| CUDA promoted vs naive (1 step) | 0.00e+00 | PASS |

## 6. Not Benchmarked

- **TileLang**: pip v0.1.8 TVM FFI bug (`T.Tensor` Buffer construction fails)
- **Warp fp64**: Warp's `float` type is fp32; changing to fp64 requires modifying all literal constants

## 7. Key Findings

1. **Kokkos is fastest** in both fp32 and fp64, beating hand-written CUDA by 15%
2. **Taichi matches CUDA** in fp64, but lags in fp32 (1.52x slower than Kokkos)
3. **Warp ≈ Kokkos** in fp32 (within 5%) — both leverage efficient fp32 codegen
4. **fp32 gives 1.28-1.78x speedup** over fp64 (kernel is partially memory-bound)
5. **Triton is 8.7x slower** — precompute-all-select-one is catastrophic for branchy kernels
6. **Gather promotion has no benefit** on B200 (60MB L2 cache covers working set)
7. **Framework compilers > hand-written CUDA** — Kokkos/Taichi produce better optimized code
