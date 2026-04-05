# Launch Overhead Elimination: Validated Solutions

## Setup
- GPU: NVIDIA B200 (148 SMs, Compute 10.0)
- Benchmark: 9 synthetic stencil configs + hydro-cal real workload (6675 cells)
- 4 strategies compared against Python DSL sync-per-step pattern

## Solution Comparison (μs/step)

### Synthetic Stencils

| Kernel | Sync (DSL) | Async (C++) | CUDA Graph | Persistent | Taichi | Warp |
|--------|-----------|-------------|------------|------------|--------|------|
| Heat2D 128² | 12.3 | 7.2 (1.7x) | **2.8** (4.3x) | 3.9 (3.2x) | 18.4 | 15.0 |
| Heat2D 256² | 12.9 | 8.1 (1.6x) | **3.3** (3.9x) | 4.8 (2.7x) | 14.9 | 14.9 |
| Heat2D 512² | 15.4 | 10.2 (1.5x) | **5.6** (2.7x) | 9.7 (1.6x) | 15.3 | 15.3 |
| Heat2D 1024² | 25.9 | 19.2 (1.4x) | **16.2** (1.6x) | N/A | 16.5 | 15.2 |
| Heat2D 2048² | 67.0 | 61.5 (1.1x) | **57.4** (1.2x) | N/A | 20.6 | 40.2 |
| GrayScott 128² | 16.5 | 10.4 (1.6x) | **4.8** (3.4x) | 5.4 (3.1x) | 14.8 | 17.7 |
| GrayScott 256² | 17.5 | 12.3 (1.4x) | **5.7** (3.1x) | 6.5 (2.7x) | 15.0 | 17.7 |
| GrayScott 512² | 22.7 | 18.4 (1.2x) | **10.6** (2.1x) | 14.2 (1.6x) | 15.0 | 17.9 |
| GrayScott 1024² | 45.3 | 39.5 (1.1x) | **33.8** (1.3x) | N/A | 16.7 | 18.6 |

### Real Workload: Hydro-Cal (6675 cells, 26700 edges, 9000 steps)

| Method | μs/step | ms/day | Speedup vs Sync |
|--------|---------|--------|-----------------|
| Sync loop (Taichi-like) | 15.23 | 13.7 | 1.00x |
| Async loop (Kokkos-like) | 8.20 | 7.4 | **1.86x** |
| CUDA Graph (900-step batch) | 5.29 | 4.8 | **2.88x** |
| Persistent kernel (coop grid) | 5.73 | 5.2 | **2.66x** |

Overhead breakdown for hydro-cal:
- Pure GPU compute: ~5.3 μs (Graph lower bound)
- Launch overhead: 2.9 μs (Async − Graph)
- Sync overhead: 7.0 μs (Sync − Async)
- **Total overhead: 9.9 μs = 65% of runtime**

## Validated Solutions

### 1. CUDA Graph (Best overall: 2.9–4.3x)
- **Mechanism**: Capture kernel launch sequence, GPU replays without CPU involvement
- **Pros**: Universal, no code restructuring, works at any grid size
- **Cons**: Static graph only (no data-dependent control flow), memory for graph storage
- **Best for**: Fixed time-stepping loops (90%+ of simulation kernels)

### 2. C++ Async Loop (Simple: 1.5–1.9x)
- **Mechanism**: Remove per-step `cudaDeviceSynchronize()`, let GPU pipeline launches
- **Pros**: Simplest change (just remove sync), always applicable
- **Cons**: Still pays cudaLaunchKernel cost per step
- **Best for**: Any kernel; baseline improvement over Python DSL

### 3. Persistent Kernel (Strong: 2.7–3.2x, small grids)
- **Mechanism**: Single kernel launch, loop inside kernel, cooperative grid sync between phases
- **Pros**: Near-Graph performance, zero launch overhead, data stays in registers/cache
- **Cons**: Grid size ≤ occupancy × SM count (e.g., ≤444 blocks on B200)
- **Best for**: Small-to-medium meshes (hydro-cal's 105 blocks fits perfectly)
- **Note**: This is the only solution implementable purely at the compiler level

### 4. Stream Pipelining (Marginal: ~1.2x)
- **Mechanism**: Round-robin kernel launches across 2+ streams
- **Pros**: Easy to implement
- **Cons**: Still has launch overhead, marginal benefit
- **Not recommended** as primary solution

## Key Insight: Two Regimes

```
Overhead fraction vs problem size:

N=128²:  |████████████████████░░░░|  ~85% overhead  → Graph/Persistent: 3-4x
N=256²:  |██████████████████░░░░░░|  ~75% overhead  → Graph/Persistent: 2.7-3.9x
N=512²:  |████████████████░░░░░░░░|  ~65% overhead  → Graph: 2.1-2.7x
N=1024²: |███████░░░░░░░░░░░░░░░░░|  ~30% overhead  → Graph: 1.3-1.6x
N=2048²: |███░░░░░░░░░░░░░░░░░░░░░|  ~15% overhead  → Diminishing returns
```

**Simulation codes overwhelmingly operate in the overhead-dominated regime:**
- hydro-cal: 6675 cells → 105 blocks (overhead = 65%)
- Typical CFD: 10K-100K cells → similar
- FEM/DEM: Often 10K-50K elements

## Taichi/Warp-Specific Observations

**Taichi has a ~15 μs floor** regardless of problem size (even 128² = 16K elements):
- Python runtime: ~3 μs
- Taichi runtime/JIT dispatch: ~2–3 μs
- cudaLaunchKernel: ~3–5 μs
- Implicit synchronization: ~5+ μs

**Warp** has similar ~15 μs floor for simple kernels, ~18 μs for multi-field.

**Neither framework exposes any of the 4 solutions to users** — they always sync per step, always launch from Python, and don't support CUDA Graph capture or persistent kernels.

## Compiler/Runtime Actionable Items

1. **Transparent CUDA Graph capture** — detect fixed time-stepping pattern, auto-capture → 2.9x
2. **Persistent kernel transformation** — for small grids, fuse multi-kernel step into single cooperative launch → 2.7x
3. **Sync elimination** — static analysis to remove unnecessary per-step sync → 1.9x
4. **Combined**: Persistent + register tuning (from compute study) → potentially 3.7x

## Reproducing

```bash
# Compile
nvcc -O3 -arch=sm_90 -rdc=true overhead_solutions_v2.cu -o overhead_solutions -lcudadevrt
nvcc -O3 -arch=sm_90 -rdc=true hydro_persistent.cu -o hydro_persistent -lcudadevrt

# Run
./overhead_solutions
./hydro_persistent
```
