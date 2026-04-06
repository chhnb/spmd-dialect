# Benchmark Kernel Coverage

## Overview

36 kernel types (15 distinct computational patterns) across 9 domains, ~120 configurations.

## Computational Pattern Classification

### Pattern 1: 5-point stencil (8 variants)
Heat2D, Jacobi2D, Poisson2D, HotSpot(Rodinia), AllenCahn, Wave2D, ConvDiff, SRAD(Rodinia)

### Pattern 2: 7-point stencil (2 variants)
Heat3D, Jacobi3D(Parboil/PERKS standard)

### Pattern 3: Multi-field stencil (4 variants)
GrayScott, Burgers2D, CahnHilliard, SWE(Lax-Friedrichs)

### Pattern 4: 4th-order stencil
Kuramoto-Sivashinsky (4th derivative)

### Pattern 5: LBM stream+collide
LBM D2Q9 (unique streaming pattern)

### Pattern 6: Multi-kernel pressure projection
Stable Fluids = divergence + 20× Jacobi pressure + project = **22 kernels/step**

### Pattern 7: O(N²) pairwise
N-body, MD(Lennard-Jones)

### Pattern 8: Neighbor-list particle
SPH, DEM(spring-dashpot)

### Pattern 9: Particle-grid coupling
PIC 1D = deposit + Poisson + E-field + push = **4 kernels/step**

### Pattern 10: FDTD leapfrog stagger
FDTD Maxwell 2D (E/H field update)

### Pattern 11: FEM element scatter
Explicit FEM 2D triangles (force assembly + node update)

### Pattern 12: Sparse indirect access
SpMV CSR (inside CG iteration)

### Pattern 13: CG multi-kernel iteration
CG Solver = matvec + 2 dot products + 2 vector updates = **5 kernels/step**

### Pattern 14: Lagrangian hydro multi-kernel
LULESH-like = forces + node update + EOS = **3 kernels/step**

### Pattern 15: Other
Reduction, Monte Carlo random walk, Schrodinger 1D, Helmholtz, Upwind1D, MassSpring1D

## Multi-Kernel-per-Step Cases (最重要 — fusion 价值最高)

These are the cases where persistent kernel fusion gives the biggest advantage over PERKS (which only handles single-kernel loops):

| Case | Kernels/Step | Pattern | PERKS | Our Fusion | Launch OH per Step |
|---|---|---|---|---|---|
| **CG Solver** | 5 | matvec + 2 dots + 2 updates | 每个单独 persistent | **融合成 1 个** | 5 × ~10μs = 50μs |
| **Stable Fluids** | 22 | div + 20×jacobi + project | 不行 | **融合成 1 个** | 22 × ~10μs = 220μs |
| **PIC 1D** | 4 | deposit + poisson + efield + push | 不行 | **融合成 1 个** | 4 × ~10μs = 40μs |
| **LULESH-like** | 3 | forces + update + EOS | 不行 | **融合成 1 个** | 3 × ~10μs = 30μs |
| **Hydro-cal** | 2 | flux + update | 不行 | **融合成 1 个** | 2 × ~10μs = 20μs |

## Related Work Overlap

Each benchmark must be positioned against prior work:

| Benchmark | Prior Work | Relationship |
|---|---|---|
| **Jacobi3D** | **PERKS (ICS'23)**: 2.12x persistent kernel | 直接对比baseline。我们 multi-kernel fusion 在 CG 上应该更优 |
| **CG Solver** | **PERKS (ICS'23)**: 4.86x on CG (SpMV) | 关键差异化：PERKS 每个 kernel 单独 persistent，我们融合 5 个 kernel 为 1 个 |
| **LBM** | **Leonid (ICS'25)**: 1.26x auto fusion on GPU | 需要对比。我们的 persistent 路径可能更优 |
| **LULESH** | 大量 MPI+CUDA 优化论文 | 没人从 launch overhead 角度分析过 LULESH |
| **HotSpot/SRAD** | **Rodinia (IISWC'09)**: 架构 benchmark 标准 | 大量测量数据，但无 overhead 分析 |
| **SpMV** | 数百篇格式/负载优化论文 | 我们不单独优化 SpMV，只作为 CG 的子组件 |
| **Stable Fluids** | Taichi 官方示例，NVIDIA 示例 | 22 kernel/step 是最强的 fusion showcase |
| **MPM** | **G2P2G fusion (SIGGRAPH'20)**: 手动 fusion | AsyncTaichi 自动 fusion 过，但已废弃 |
| **Hydro-cal SWE** | 我们的原始真实工程案例 | 已验证 persistent + async DMA overlap |
| **N-body** | Taichi 官方 benchmark | DSL overhead 对小 N 尤为严重 |

## Key Thesis Support

### For "overhead wall" characterization:
所有 36 种 kernel × 小/中 grid size → 测量 overhead fraction → 证明 90% 是 OH-dominated

### For "persistent kernel fusion" value:
Multi-kernel/step cases (CG/StableFluids/LULESH/PIC/hydro-cal) → 证明 fusion 比 PERKS 的单 kernel 方案更有效

### For "CUDA Graph vs Persistent" strategy selection:
Static loops → Graph wins; Dynamic loops (CG with convergence) → Persistent wins; Large grid → Graph (no coop limit)

### For "compute-communication overlap" (async DMA):
Hydro-cal persistent + save → 验证 Compute Engine ∥ Copy Engine 并行
