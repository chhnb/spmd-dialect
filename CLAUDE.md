# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repo has two major components:
1. **SPMD Dialect** — An MLIR-based structured IR for simulation kernels (in `include/`, `lib/`, `tools/`)
2. **Benchmark Suite** — A comprehensive GPU simulation overhead characterization and optimization study (in `benchmark/`)

The active research focus is the **benchmark suite** (launch overhead wall characterization, persistent kernel fusion, strategy selection). The MLIR dialect is functional but secondary to the current research direction.

## Python Environment

All Python work uses the venv at `/home/scratch.huanhuanc_gpu/spmd/spmd-venv`.

```bash
/home/scratch.huanhuanc_gpu/spmd/spmd-venv/bin/python  # always use this
```

Taichi, Warp, NumPy are installed in this venv. Do NOT use system Python.

## Building

### SPMD Dialect (MLIR)

```bash
bash scripts/build.sh                    # full build (LLVM + dialect), ~1-2h first time
cmake --build build                      # rebuild dialect only (~5 min)
```

Binary: `build/bin/spmd-opt`. Requires GPU compute node (GLIBC ≥ 2.32).

### CUDA Benchmark Experiments

```bash
# nvcc is at /home/scratch.huanhuanc_gpu/spmd/cuda-toolkit/bin/nvcc
export PATH=/home/scratch.huanhuanc_gpu/spmd/cuda-toolkit/bin:$PATH
export LD_LIBRARY_PATH=/home/scratch.huanhuanc_gpu/spmd/cuda-toolkit/lib64:$LD_LIBRARY_PATH

# Compile (use sm_90 for B200, sm_86 for RTX 3060)
nvcc -O3 -arch=sm_90 -rdc=true benchmark/hydro_osher_benchmark.cu -o hydro_osher -lcudadevrt
nvcc -O3 -arch=sm_90 -rdc=true benchmark/overhead_solutions.cu -o overhead_solutions -lcudadevrt
nvcc -O3 -arch=sm_90 -rdc=true benchmark/persistent_async_copy.cu -o persistent_async_copy -lcudadevrt
```

The `-rdc=true -lcudadevrt` flags are required for cooperative kernel launch (persistent kernels use `cooperative_groups::this_grid().sync()`).

### Kokkos Benchmarks

```bash
cd benchmark/cpp/kokkos
cmake --build build-cuda
```

## Running Tests

### MLIR Lit Tests
```bash
bash scripts/check-quick.sh     # fast (~30s)
bash scripts/check-full.sh      # full suite with GPU
```

### Benchmark Characterization (36 kernel types × multiple sizes)
```bash
/home/scratch.huanhuanc_gpu/spmd/spmd-venv/bin/python benchmark/run_overhead_characterization.py
```

### Individual Benchmark Suites
```bash
/home/scratch.huanhuanc_gpu/spmd/spmd-venv/bin/python benchmark/A1_jacobi_2d/run.py
/home/scratch.huanhuanc_gpu/spmd/spmd-venv/bin/python benchmark/F1_hydro_shallow_water/run.py
# etc.
```

## Architecture

### Benchmark Suite (`benchmark/`)

```
benchmark/
├── A1_jacobi_2d/ ... F3_maccormack_3d/  — Per-kernel benchmark dirs
│   └── {kernel}_taichi.py, {kernel}_warp.py, run.py
├── cpp/kokkos/                   — C++/Kokkos reference implementations
├── results/                      — Collected timing data (B200 + 3060)
│   ├── b200_*.txt, 3060_*.txt    — Characterization results
│   └── ncu_*.csv                 — NCU profiling data
├── run_overhead_characterization.py  — Master script: 36 kernels × multi-size
├── hydro_osher_benchmark.cu      — Real OSHER solver: 4-strategy comparison
├── overhead_solutions.cu         — Simplified Heat2D/GrayScott: 4-strategy
├── persistent_async_copy.cu      — Validates DMA ∥ compute overlap
├── cost_model_v3.py              — Fusion cost-benefit model
├── strategy_selector.py          — Auto Graph/Persistent/Async selection
├── KERNEL_COVERAGE.md            — 36 kernel types, 15 patterns, related work mapping
└── RUN_ON_3060.md                — Step-by-step execution instructions
```

**Kernel categories**: Stencil(9), CFD(4), Particle(5), EM(2), FEM/Structure(3), Transport(2), PDE(3), Other(2), Classic(6) = 36 types.

**Multi-kernel-per-step cases** (best fusion targets): CG Solver (5 kern/step), Stable Fluids (22), LULESH-like (3), PIC (4), hydro-cal (2).

### MLIR Dialect (`include/`, `lib/`)

```
include/SPMD/         — ODS/TableGen definitions
lib/IR/               — SPMDDialect, SPMDOps, SPMDAttrs
lib/Transforms/       — NormalizeSPMD, PromoteGroupMemory, ReduceToHierarchicalGPU, ...
lib/Conversion/       — SPMDToSCF, SPMDToOpenMP, SPMDToGPU
lib/Analysis/         — AccessSummaryAnalysis, PromotionPlanAnalysis
test/SPMD/            — 34 lit tests
```

Key pipeline: `S0 (semantic) → S1 (scheduled) → S2 (materialized) → GPU/CPU lowering`

### Research Documents (`docs/`)

- `research-plan-v6.md` — Current plan: legality-aware loop-level step-time model
- `advisor-report.md` — Latest advisor report with cross-GPU (3060 vs B200) analysis
- `literature.md` — Related work survey (AsyncTaichi, PERKS, PyGraph, etc.)
- `semantic-spec-v1.md` — SPMD IR specification

## Key Technical Concepts

**Overhead Wall**: Python DSLs (Taichi/Warp) pay ~15μs (B200) to ~80μs (3060) fixed overhead per timestep from Python→driver→launch→sync. For small/medium meshes this exceeds GPU compute time.

**Four execution strategies** (tested via CUDA experiments):
- Sync loop: `launch → sync → launch → sync` (Taichi/Warp default)
- Async loop: `launch → launch → ... → sync` (Kokkos-like)
- CUDA Graph: capture launch sequence, GPU replays
- Persistent kernel: single cooperative launch, `grid_sync()` between phases

**Cooperative launch limit**: `cudaLaunchCooperativeKernel` requires all blocks simultaneously resident. Max grid = `blocks_per_SM × num_SMs`. Exceeding this → must fall back to CUDA Graph.

**Compute-Communication overlap**: Persistent kernel runs on Compute Engine while `cudaMemcpyAsync` runs on DMA Copy Engine — verified to work in parallel (benchmark/persistent_async_copy.cu).

## Important Experimental Notes

- `hydro_osher_benchmark.cu` uses the **real OSHER Riemann solver** (aligned with Taichi/Kokkos). Earlier `hydro_persistent.cu` used a simplified Rusanov flux — **do not cite those numbers as representative of hydro-cal**.
- The characterization script's overhead classification depends on measuring the real overhead per GPU. B200 overhead ≈ 15μs, 3060 overhead ≈ 70-80μs.
- Binary mesh data for F2 is at `benchmark/F2_hydro_refactored/data/binary/`. F1 loads from text files in `benchmark/F1_hydro_shallow_water/data/`.
- Warp cache can fill home disk. Set `WARP_CACHE_PATH=/home/scratch.huanhuanc_gpu/spmd/.warp_cache`.
