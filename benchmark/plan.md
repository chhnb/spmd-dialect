# Benchmark Plan: N Strategies × M Cases

## Goal

Run a comprehensive N×M benchmark matrix: every execution strategy against every simulation kernel, on H100 (and A100 for PERKS alignment). Measure per-step time, overhead fraction, and speedup.

## N = 11 Execution Strategies

| ID | Strategy | Type | Source | Description |
|----|----------|------|--------|-------------|
| S1 | CUDA Sync | Baseline | Ours | `cudaDeviceSynchronize()` after each step |
| S2 | CUDA Async | Optimization | Ours | Launch all steps, sync once at end |
| S3 | CUDA Graph | Optimization | Ours | `cudaGraphLaunch()` replay captured pattern |
| S4 | CUDA Persistent | Optimization | Ours | Cooperative launch + `grid_sync()` between steps |
| S5 | PERKS | SOTA (ICS'23) | Zhang et al. | Persistent + register/shared memory cache |
| S6 | EBISU | SOTA (ICS'23) | Matsumura et al. | Deep temporal blocking, low occupancy |
| S7 | Kokkos | Framework | Ours | `parallel_for` + `fence()`, async by default |
| S8 | Taichi | DSL | Ours | Python DSL, closure-based fields |
| S9 | Warp | DSL | Ours | Python DSL, `wp.launch()` + struct packing |
| S10 | Triton | DSL | Ours | Python DSL, `tl.load/store` blocks |
| S11 | TileLang | DSL | Ours | Python DSL, tile-based programming |

### Strategy applicability constraints

| Strategy | Constraint |
|----------|-----------|
| S3 (Graph) | Requires static launch pattern (no data-dependent control flow) |
| S4 (Persistent) | Grid must fit cooperative limit (blocks ≤ blocks_per_SM × num_SMs) |
| S5 (PERKS) | Only iterative single-kernel stencils |
| S6 (EBISU) | Only iterative single-kernel stencils |
| S7-S11 | Requires per-case implementation in that framework |

## M = 21 Benchmark Cases (deduplicated across all suites)

### Single kernel per step (8 cases)

| ID | Case | Domain | Precision | Problem Sizes | Current Sources |
|----|------|--------|-----------|---------------|----------------|
| C1 | Jacobi 2D 5pt | Stencil | fp32 | 1024², 4096², 8192² | Ours, PERKS, EBISU, PolyBench |
| C2 | Jacobi 3D 7pt | Stencil | fp32 | 128³, 256³, 512³ | PERKS, EBISU, Parboil |
| C3 | Heat 2D | Stencil | fp32 | 128², 256², 512², 1024² | Ours (overhead_solutions.cu) |
| C4 | Wave 2D | PDE | fp32 | 1024², 4096², 8192² | Ours |
| C5 | LBM D2Q9 | CFD | fp32 | 512×256, 1024×512, 2048×1024 | Ours |
| C6 | N-body | Particle | fp32 | 4096, 16384, 32768 | Ours |
| C7 | SPH | Particle | fp32 | 8192, 32768, 65536 | Ours |
| C8 | Hydro SWE F1 (Osher) | CFD | fp64 | 6675, 207234 (real mesh) | Ours (hydro-cal) |

### 2 kernels per step (2 cases)

| ID | Case | Domain | Precision | Problem Sizes | Current Sources |
|----|------|--------|-----------|---------------|----------------|
| C9 | Hydro Refactored F2 | CFD | fp32 | 24020, 207234 (real mesh) | Ours (hydro-cal) |
| C10 | GrayScott | Reaction-Diffusion | fp32 | 128², 256², 512² | Ours (overhead_solutions.cu) |

### 3-5 kernels per step (5 cases)

| ID | Case | Domain | Precision | Problem Sizes | Current Sources |
|----|------|--------|-----------|---------------|----------------|
| C11 | FDTD-2D | EM | fp32 | 1024², 4096² | PolyBench |
| C12 | MacCormack 3D | CFD | fp32 | 64³, 128³ | Ours |
| C13 | LULESH-like | Lagrangian Hydro | fp32 | 32³, 64³ (4096 elem) | Ours |
| C14 | PIC 1D | Plasma | fp32 | 4096 particles, 256 grid | Ours |
| C15 | CG Solver | Linear Algebra | fp32 | 16384 unknowns | Ours, PERKS |

### Extreme multi-kernel per step (6 cases)

| ID | Case | Domain | Kern/Step | Problem Sizes | Current Sources |
|----|------|--------|----------|---------------|----------------|
| C16 | Stable Fluids | CFD | 102 | 512², 1024² | Ours |
| C17 | 3D Convolution | Stencil | 510 | 128³, 256³ | PolyBench |
| C18 | DOITGEN | Tensor | 512 | 128³, 256³ | PolyBench |
| C19 | LU Decomposition | Linear Algebra | 4096 | N=1024, 2048 | PolyBench |
| C20 | ADI | Implicit Solver | 4097 | N=1024, 2048 | PolyBench |
| C21 | Gram-Schmidt | Linear Algebra | 6144 | N=1024, 2048 | PolyBench |

## N×M Matrix: Current Status

Legend: ✓ = implemented and tested, ○ = impl exists but not tested, △ = need to implement, ✗ = N/A (strategy doesn't apply)

```
Case                S1    S2    S3    S4    S5    S6    S7    S8    S9    S10   S11
                    Sync  Asyn  Grph  Pers  PERK  EBIS  Kokk  Tai   Warp  Trit  TLng
────────────────────────────────────────────────────────────────────────────────────────
C1  Jacobi2D 5pt    △     △     △     △     ✓     ✓     △     ✓     ✓     ✓     △
C2  Jacobi3D 7pt    △     △     △     △     ✓     ✓     △     △     △     △     △
C3  Heat 2D         ✓     ✓     ✓     ✓     ✗     ✗     △     △     △     △     △
C4  Wave 2D         △     △     △     △     ✗     ✗     △     ✓     ✓     △     △
C5  LBM D2Q9        △     △     △     △     ✗     ✗     △     ✓     △     △     △
C6  N-body          △     △     △     △     ✗     ✗     △     ✓     ✓     △     △
C7  SPH             △     △     △     △     ✗     ✗     △     ✓     ✓     △     △
C8  Hydro F1        ✓     ✓     ✓     ✓     ✗     ✗     ✓     ✓     ✓     ✓     △
─── 2 kern/step ────────────────────────────────────────────────────────────────────
C9  Hydro F2        ✓     ✓     ✓     ✓     ✗     ✗     ✓     ✓     ✓     ✓     △
C10 GrayScott       ✓     ✓     ✓     ✓     ✗     ✗     △     △     △     △     △
─── 3-5 kern/step ──────────────────────────────────────────────────────────────────
C11 FDTD-2D         △     △     △     △     ✗     ✗     △     △     △     △     △
C12 MacCormack 3D   △     △     △     △     ✗     ✗     △     ✓     △     △     △
C13 LULESH-like     ✓     ✓     ✓     ✓     ✗     ✗     △     △     △     △     △
C14 PIC 1D          △     △     △     △     ✗     ✗     △     ✓     △     △     △
C15 CG Solver       ✓     ✗     ✓*    ✓     ✓     ✗     △     △     △     △     △
─── Extreme multi-kernel ───────────────────────────────────────────────────────────
C16 Stable Fluids   △     △     △     △     ✗     ✗     △     ✓     ✓     △     △
C17 3DConv          △     △     △     △     ✗     ✗     △     △     △     △     △
C18 DOITGEN         △     △     △     △     ✗     ✗     △     △     △     △     △
C19 LU              △     △     △     △     ✗     ✗     △     △     △     △     △
C20 ADI             △     △     △     △     ✗     ✗     △     △     △     △     △
C21 Gram-Schmidt    △     △     △     △     ✗     ✗     △     △     △     △     △
```

### Cell counts

| Status | Count | % |
|--------|-------|---|
| ✓ Implemented + tested | 42 | 18% |
| ✗ N/A (strategy doesn't apply) | 42 | 18% |
| △ Need to implement | 147 | 64% |
| **Total cells** | **231** | |

## Execution Plan

### Phase 1: CUDA 4-strategy coverage for all 21 cases

**Goal:** Fill S1-S4 columns for all cases. This is the core overhead characterization.

For each case, write a `{case}_benchmark.cu` file implementing sync/async/graph/persistent, similar to the existing `hydro_osher_benchmark.cu` pattern.

| Priority | Cases | Effort | Notes |
|----------|-------|--------|-------|
| P0 (done) | C3, C8, C9, C10, C13, C15 | — | Already have CUDA 4-strategy |
| P1 (stencil) | C1, C2, C4 | Low | Simple stencils, template from Heat2D |
| P1 (particle) | C5, C6, C7 | Low | Single kernel, straightforward |
| P2 (multi-kern) | C11, C12, C14 | Medium | 3-4 kernels per step |
| P2 (extreme) | C16, C17, C18 | Medium | Many launches, graph capture may be large |
| P3 (extreme) | C19, C20, C21 | High | 4000+ launches, need loop-based graph or persistent redesign |

**Deliverable:** 21 `.cu` benchmark files, each with sync/async/graph/persistent timing.

### Phase 2: DSL coverage (Taichi/Warp/Triton) for key cases

**Goal:** Fill S8-S10 columns. Priority order by research value.

| Priority | Cases | Effort |
|----------|-------|--------|
| P0 (done) | C1, C8, C9 (Jacobi, Hydro F1/F2) | — |
| P1 | C4, C6, C7 (Wave, N-body, SPH) | Partially done (Taichi+Warp exist) |
| P1 | C13, C15, C16 (LULESH, CG, Stable Fluids) | High value multi-kernel cases |
| P2 | C2, C3, C5, C10, C11, C12, C14 | Remaining single/few kernel cases |
| P3 | C17-C21 (extreme multi-kernel) | Very high effort, may not be meaningful for DSL comparison |

**Note on extreme cases (C17-C21):** PolyBench's ADI/LU/Gram-Schmidt launch thousands of tiny kernels from the host. A Taichi/Warp implementation would likely restructure the algorithm (fuse loops), which changes the overhead profile. These are valuable as "what happens when you DON'T fuse" baselines, not as DSL comparison targets.

### Phase 3: Kokkos + TileLang coverage

**Goal:** Fill S7, S11 columns for key cases.

Kokkos is valuable because it's the C++ portable performance layer (used in DOE apps). TileLang is a newer DSL worth exploring.

| Cases | Kokkos | TileLang |
|-------|--------|---------|
| C1 (Jacobi 2D) | Implement | Implement |
| C8, C9 (Hydro F1/F2) | Done | Implement |
| C13 (LULESH) | Use LULESH-official Kokkos port | — |
| C15 (CG) | Implement | — |

### Phase 4: Run full matrix

**Hardware:**
- Primary: NVIDIA H100 80GB HBM3 (sm_90, 132 SMs)
- Secondary: NVIDIA A100 80GB PCIe (sm_80, 108 SMs) — for PERKS paper alignment

**Parameters per cell:**
- Problem sizes: small (overhead-dominated) + large (compute-dominated)
- Steps: 100 iterations (or 1 day for hydro cases)
- Warmup: 5 runs
- Timed: 10 runs
- Metric: median per-step time (μs)

**Output format (CSV):**
```
case,strategy,gpu,problem_size,steps,median_us,min_us,max_us,overhead_pct
```

### Phase 5: Analysis and visualization

**Key plots:**

1. **Overhead% vs kernels/step** (x-axis: C1→C21 sorted by kern/step, y-axis: overhead%)
   - One line per strategy → shows how each strategy scales with multi-kernel complexity

2. **Speedup heatmap** (N×M matrix, color = speedup vs S1 Sync)
   - Shows which strategy wins for which case

3. **DSL overhead decomposition** (for F1/F2 where we have full coverage)
   - Stack bar: GPU compute | launch overhead | Python overhead | codegen penalty

4. **Problem size scaling** (for Jacobi 2D + Hydro F2)
   - x-axis: problem size, y-axis: overhead%
   - Shows the "overhead wall" crossing point

5. **H100 vs A100 comparison** (for PERKS-overlapping cases)
   - Shows how newer GPU architecture changes overhead dynamics

## Implementation Priority

| Phase | Work | Cells Filled | Timeline |
|-------|------|-------------|----------|
| Phase 1 | CUDA 4-strategy for 15 remaining cases | +60 cells | Priority |
| Phase 2 | DSL for ~10 cases × 3 DSLs | +30 cells | After Phase 1 |
| Phase 3 | Kokkos + TileLang for ~5 cases | +10 cells | Parallel with Phase 2 |
| Phase 4 | Run everything | 0 (data collection) | After Phase 1-3 |
| Phase 5 | Analysis | 0 (output) | After Phase 4 |

**After Phase 1-3: ~142/147 △ cells filled → ~95% coverage**

## Case-Source Mapping (provenance)

| Case | Kernel Code Source | Data Source | Reference Paper |
|------|-------------------|-------------|-----------------|
| C1 Jacobi 2D | Ours (multi-framework) + PERKS + EBISU + PolyBench | Synthetic (random init) | Zhang ICS'23, Grauer-Gray InPar'12 |
| C2 Jacobi 3D | PERKS + EBISU + Parboil stencil | Synthetic | Zhang ICS'23, Stratton IMPACT'12 |
| C3 Heat 2D | Ours (overhead_solutions.cu) | Synthetic | — |
| C4 Wave 2D | Ours (Taichi/Warp) | Synthetic | — |
| C5 LBM D2Q9 | Ours (Taichi) + Parboil lbm (D3Q19) | Synthetic | Stratton IMPACT'12 |
| C6 N-body | Ours (Taichi/Warp) + Rodinia lavaMD | Synthetic | Che IISWC'09 |
| C7 SPH | Ours (Taichi/Warp), ref: GPUSPH | Synthetic, ref: DamBreak3D | Hérault 2010 |
| C8 Hydro F1 | Ours (5 frameworks) | hydro-cal real mesh (6675+207K) | hydro-cal project |
| C9 Hydro F2 | Ours (5 frameworks) | hydro-cal real mesh (24K+207K) | hydro-cal project |
| C10 GrayScott | Ours (overhead_solutions.cu) | Synthetic | — |
| C11 FDTD-2D | PolyBench | Synthetic | Grauer-Gray InPar'12 |
| C12 MacCormack 3D | Ours (Taichi) | Synthetic | Selle & Fedkiw (Unconditionally Stable MacCormack) |
| C13 LULESH-like | Ours (lulesh_fusion_benchmark.cu) | Synthetic | Karlin LLNL-TR |
| C14 PIC 1D | Ours (Taichi), ref: PIConGPU, PICSAR | Synthetic, ref: LaserWakefield | Bussmann SC'13, ECP WarpX |
| C15 CG Solver | Ours (cg_fusion_benchmark.cu) + PERKS | Synthetic (tridiagonal) | Zhang ICS'23 |
| C16 Stable Fluids | Ours (Taichi/Warp) | Synthetic | Stam 1999, NVIDIA GPU Gems |
| C17 3DConv | PolyBench | Synthetic | Grauer-Gray InPar'12 |
| C18 DOITGEN | PolyBench | Synthetic | Grauer-Gray InPar'12 |
| C19 LU | PolyBench | Synthetic | Grauer-Gray InPar'12 |
| C20 ADI | PolyBench | Synthetic | Grauer-Gray InPar'12 |
| C21 Gram-Schmidt | PolyBench | Synthetic | Grauer-Gray InPar'12 |

## Simulation-Specific Project References

These larger simulation projects provide domain credibility for our kernel types:

| Kernel Type | Project | Scale | Reference |
|-------------|---------|-------|-----------|
| PIC | PIConGPU | Exascale (1024³ grid, multi-GPU) | Bussmann et al., SC'13 |
| PIC | WarpX/PICSAR | DOE ECP flagship | Vay et al. |
| SPH | GPUSPH | DamBreak3D, 20+ problems | Hérault et al., 2010 |
| DEM | DEM-Engine (Chrono) | 150M elements on dual-A100 | Tasora et al., 2023 |
| MPM | taichi_mpm / GPUMPM | MLS-MPM, SIGGRAPH'18 | Hu et al., Gao et al. |
| MPM | DiffTaichi diffmpm | Differentiable MPM | Hu et al., ICLR'20 |
| Hydro SWE | hydro-cal | Real engineering mesh, 207K cells | Collaborator (Chen) |
| Stable Fluids | GPU Gems Ch.38 | NVIDIA official example | Harris 2004 |
| MacCormack | NICAM | Atmospheric model, 320 GPU | Shimomura et al. |
