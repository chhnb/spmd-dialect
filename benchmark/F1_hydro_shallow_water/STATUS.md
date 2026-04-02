# F1 — 2D Shallow Water Equations (Osher Riemann Solver)

**Source**: Ported from hydro-cal project (`calculate_gpu.cu`)

## Problem
Dam-break on NxN structured quad mesh using unstructured mesh representation
(cell-neighbor arrays). Exercises the same Osher Riemann solver and finite-volume
update kernel as the hydro-cal 1D-2D coupled hydrodynamic simulation.

## Kernel
Per-cell computation:
1. For each of 4 edges: compute left/right Riemann states
2. Apply Osher approximate Riemann solver for shallow water equations
3. Accumulate flux divergence (mass, x-momentum, y-momentum)
4. Time-step update with Manning friction source term
5. Wet-dry cell handling

## Implementations
- [x] NumPy (CPU reference, per-cell loop)
- [x] Taichi (CPU + CUDA)
- [x] Warp (CUDA)
- [x] Triton (CUDA) — branchless precompute-and-select for Osher 16-case dispatch
- [x] TileLang (CUDA) — T.if_then_else precompute-and-select, no tile/shared benefit

## Grid sizes
Default: 32, 64, 128, 256 (NxN cells)
NumPy is O(N^2) per step with a Python loop, so it's slow for large N.

## Physics
- Gravity: g = 9.81 m/s²
- Manning roughness: n = 0.03
- Drying threshold: HM1 = 0.001 m
- Shallow threshold: HM2 = 0.01 m
- Boundary: reflective walls (KLAS=4)
