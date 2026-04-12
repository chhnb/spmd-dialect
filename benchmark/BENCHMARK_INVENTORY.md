# Benchmark Inventory

Tested on NVIDIA H100 80GB HBM3 (sm_90, CUDA 12.6). Date: 2026-04-12.

## Source Attribution

### GPU Benchmark Suites (overhead / kernel optimization focus)

| Suite | Paper | Year | Original GPU | URL | Status |
|-------|-------|------|-------------|-----|--------|
| PERKS | Zhang et al., ICS'23 | 2023 | V100 + A100 | github.com/neozhang307/PERKS | **69 exe running** |
| EBISU | Matsumura et al., ICS'23 | 2023 | V100 + A100 | github.com/neozhang307/EBISU-ICS23 | **Built, running** |
| PolyBenchGPU | Grauer-Gray et al., InPar'12 | 2012 | Tesla C2070 | github.com/cavazos-lab/PolyBench-ACC | **21 exe running** |
| Parboil | Stratton et al., IMPACT'12 | 2012 | Tesla C2050 | github.com/abduld/Parboil | **18 exe, 17 running** |
| Rodinia | Che et al., IISWC'09 | 2009 | GTX 280 | github.com/yuhc/gpu-rodinia | **23/24 compiled** |
| LULESH | Karlin et al., LLNL-TR | 2013 | CPU only | github.com/LLNL/LULESH | **Running (C++/OpenMP)** |

### Simulation-Specific Projects (domain-representative kernels)

| Project | Paper/Reference | Year | Domain | URL | Benchmark Cases | Status |
|---------|----------------|------|--------|-----|----------------|--------|
| PIConGPU | Bussmann et al., SC'13 | 2013 | Plasma PIC | github.com/ComputationalRadiationPhysics/picongpu | LaserWakefield, KelvinHelmholtz, etc. (10) | **Cloned** |
| PICSAR | ECP WarpX project | 2020 | Plasma PIC proxy | github.com/ECP-WarpX/picsar | DOE acceptance tests | **Cloned** |
| GPUSPH | Hérault et al., 2010 | 2010 | SPH fluid | github.com/GPUSPH/gpusph | DamBreak3D, Bubble, etc. (20+) | **Cloned** |
| DEM-Engine | Chrono project (Tasora et al.) | 2023 | DEM granular | github.com/projectchrono/DEM-Engine | Mixer, Centrifuge, Sieve, etc. (28) | **Cloned** |
| taichi_mpm | Hu et al., SIGGRAPH'18 | 2018 | MPM | github.com/yuanming-hu/taichi_mpm | MLS-MPM solver | **Cloned** |
| GPUMPM | Gao et al., SIGGRAPH'18 | 2018 | MPM (GPU) | github.com/kuiwuchn/GPUMPM | GPU-optimized MPM | **Cloned** |
| DiffTaichi | Hu et al., ICLR'20 | 2020 | Diff. physics | github.com/yuanming-hu/difftaichi | diffmpm, wave, smoke, etc. (8 ok) | **8 running** |
| hydro-cal | Collaborator (Chen) | 2024 | Hydro SWE | github.com/9triver/hydro-cal | 6675 + 24K + 207K cell mesh | **Fully aligned** |
| hydro-cal | Collaborator (Chen) | 2024 | RTX 3060/3090 | github.com/9triver/hydro-cal |

## All Kernels by Type (merged across suites)

### Stencil 2D

| Kernel | Suites | Implementations | Status |
|--------|--------|----------------|--------|
| Jacobi 2D 5pt | Ours + PERKS + EBISU + PolyBench | Taichi/Warp/Triton/NumPy + CUDA naive/baseline/persistent/temporal | All running |
| Jacobi 2D 9pt | PERKS + EBISU | CUDA 4 strategies + temporal | Running |
| Jacobi 2D 13pt | PERKS | CUDA 4 strategies | Running |
| Jacobi 2D 17pt | PERKS | CUDA 4 strategies | Running |
| Jacobi 2D 21pt | PERKS | CUDA 4 strategies | Running |
| Jacobi 2D 25pt | PERKS + EBISU | CUDA 4 strategies + temporal | Running |
| Jacobi 2D 49pt | EBISU | CUDA temporal | Running |
| Star 2D 9pt | PERKS + EBISU | CUDA + temporal | Running |
| Star 2D 25pt | PERKS + EBISU | CUDA + temporal | Running |
| Jacobi 1D | PolyBench | CUDA | Running (N=1M, 1000 steps) |
| 2D Convolution | PolyBench | CUDA | Running (8192x8192) |
| 3D Convolution | PolyBench | CUDA | Running (512^3, 510 launches) |
| Heat 2D | Ours | CUDA sync/async/graph/persistent | Running |
| Wave 2D | Ours | Taichi, Warp | Running |
| HotSpot 2D | Rodinia | CUDA | Compiled, needs data |
| SRAD | Rodinia | CUDA (v1+v2) | Running (v2) |
| FDTD 2D | PolyBench | CUDA | Running (4096x4096, 3 kern/step) |

### Stencil 3D

| Kernel | Suites | Status |
|--------|--------|--------|
| Jacobi 3D 7pt | PERKS + EBISU + Parboil | All running |
| Jacobi 3D 13pt | PERKS + EBISU (EBISU crashes) | PERKS running |
| Jacobi 3D 17pt | PERKS + EBISU | Running |
| Jacobi 3D 27pt | PERKS + EBISU | Running |
| Poisson 3D | PERKS + EBISU | Running |
| HotSpot 3D | Rodinia | Compiled, needs data |

### CFD / Hydro

| Kernel | Suites | Implementations | Status |
|--------|--------|----------------|--------|
| **Hydro SWE Osher (cell-parallel, fp64)** | Ours (hydro-cal) | CUDA 4-strategy, Kokkos, Taichi, Warp, Triton | **Fully aligned, 6675+207K cells** |
| **Hydro Osher (edge-parallel, fp32)** | Ours (hydro-cal) | CUDA 4-strategy, Kokkos, Taichi, Warp, Triton | **Fully aligned, 24K+207K cells** |
| LBM D2Q9 | Ours | Taichi, NumPy | Running |
| LBM D3Q19 | Parboil | CUDA | Running (167us/kern) |
| **Stable Fluids** | Ours | Taichi, Warp | **Running (102 kern/step, overhead showcase)** |
| CFD Euler | Rodinia | CUDA (4 variants) | Compiled, needs mesh data |
| GrayScott | Ours | CUDA sync/async/graph/persistent | Running |

### Particle Methods

| Kernel | Suites | Implementations | Status |
|--------|--------|----------------|--------|
| N-body (all-pairs) | Ours | Taichi, Warp | Running (4K-32K) |
| SPH Density | Ours | Taichi, Warp | Running |
| MPM (P2G+G2P) | Ours + DiffTaichi | Taichi (1 impl) | Running (needs Warp/CUDA to be useful) |
| PIC 1D | Ours | Taichi (1 impl) | Running (4 kern/step, needs Warp/CUDA) |
| lavaMD | Rodinia | CUDA | Running (1.1ms kernel) |
| particlefilter | Rodinia | CUDA (2 variants) | Running |

### Linear Algebra / Solvers

| Kernel | Suites | Launches | Status |
|--------|--------|---------|--------|
| **CG Solver (fused)** | Ours | 5/step + host readback | Running (CUDA sync/persistent/graph) |
| CG Solver (SpMV) | PERKS | baseline/persistent | Running |
| SpMV (JDS) | Parboil | 1 | Running |
| GEMM | PolyBench | 1 | Running (2048^3) |
| 2MM / 3MM | PolyBench | 2-3 | Running |
| **LU** | PolyBench | **4096** | **Running (overhead-heavy)** |
| **Gram-Schmidt** | PolyBench | **6144** | **Running (extreme overhead)** |
| **ADI** | PolyBench | **4097/step** | **Running (most extreme)** |
| ATAX / BICG / MVT | PolyBench | 2 | Running |
| GEMVER / GESUMMV | PolyBench | 1-3 | Running |
| SYR2K / SYRK | PolyBench | 1 | Running |
| CORR / COVAR | PolyBench | 3-4 | Running |
| DOITGEN | PolyBench | 512 | Running |
| Gaussian Elimination | Rodinia | — | Running (0.041s) |
| LUD | Rodinia | — | Running (1.45ms) |

### Multi-Kernel Fusion Targets (sorted by launch count)

| Kernel | Kern/Step | Suite | Overhead% (H100) |
|--------|----------|-------|------------------|
| Gram-Schmidt | 6144 | PolyBench | >90% |
| ADI | 4097 | PolyBench | ~90% |
| LU | 4096 | PolyBench | ~80% |
| 3D Convolution | 510 | PolyBench | ~95% |
| DOITGEN | 512 | PolyBench | ~70% |
| Stable Fluids | 102 | Ours | ~89% |
| CG Solver | 5 (+readback) | Ours | ~93% |
| LULESH-like | 4 | Ours | ~69% |
| PIC 1D | 4 | Ours | TBD |
| FDTD-2D | 3 | PolyBench | ~30% |
| Hydro Refactored | 2 | Ours | ~53% |

### Lagrangian Hydro / Structural

| Kernel | Suites | Status |
|--------|--------|--------|
| **LULESH-like** | Ours | CUDA 4-strategy, running (4 kern/step) |
| LULESH 2.0 | LULESH-official | C++/OpenMP, running (30^3, 0.84s) |

### Graph / Irregular

| Kernel | Suites | Status |
|--------|--------|--------|
| BFS | Parboil + Rodinia | Both compiled (Parboil running, Rodinia needs data) |
| B+tree | Rodinia | Compiled, needs data |
| Needleman-Wunsch | Rodinia | Running |
| Pathfinder | Rodinia | Running |
| Streamcluster | Rodinia | Running (0.36s) |

### Imaging / Signal / ML

| Kernel | Suites | Status |
|--------|--------|--------|
| FFT | Parboil | Compiled |
| Histogram | Parboil | Compiled |
| SAD | Parboil | Compiled |
| MRI-Q / MRI-FHD | Parboil | Running |
| DWT2D | Rodinia | Compiled, needs data |
| Backprop | Rodinia | Running |
| Hybridsort | Rodinia | Running |
| K-means | Rodinia | Compiled, needs data |
| NN | Rodinia | Running |

### Differentiable Physics (DiffTaichi)

| Kernel | Type | Status |
|--------|------|--------|
| diffmpm / diffmpm_simple / diffmpm_checkpointing | 2D MPM | Running (CUDA) |
| wave | 2D wave equation | Running (CUDA) |
| smoke_taichi / smoke_taichi_gpu | 2D Euler smoke | Running (CUDA) |
| mass_spring / mass_spring_simple | Mass-spring robot | Running (CPU) |

## Summary

| Category | Unique Kernels | Running | Sources |
|----------|---------------|---------|---------|
| 2D Stencil | 17 | 15 | Ours + PERKS + EBISU + PolyBench + Rodinia |
| 3D Stencil | 6 | 5 | PERKS + EBISU + Parboil + Rodinia |
| CFD / Hydro | 7 | 6 | Ours (hydro-cal) + Parboil + Rodinia |
| Particle | 6 | 6 | Ours + Rodinia |
| Linear Algebra | 16 | 16 | Ours + PERKS + PolyBench + Rodinia |
| Fusion Targets | 11 | 10 | Ours + PolyBench |
| Lagrangian | 2 | 2 | Ours + LULESH |
| Graph/Irregular | 5 | 3 | Parboil + Rodinia |
| Imaging/Signal/ML | 9 | 5 | Parboil + Rodinia |
| Diff. Physics | 8 | 8 | DiffTaichi |
| **Total** | **~87** | **~76** | **8 suites** |

## Removed (single implementation, no external counterpart, no research value)

- ~~A4_euler_compressible~~ (1 Taichi impl, Rodinia cfd is similar)
- ~~B3_dem~~ (1 Taichi impl, no external)
- ~~D1_cloth~~ (1 Taichi impl, DiffTaichi mass_spring is similar)
- ~~F3_maccormack_3d~~ (1 Taichi impl, no external)
- ~~E1_reduction~~ (not a simulation kernel, Warp impl has bug)
