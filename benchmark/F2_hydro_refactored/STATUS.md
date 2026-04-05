# F2 — Refactored Hydro-Cal (Edge-Parallel, fp32)

**Source**: hydro-cal branch `20251011_cuda_refactor_chensc`

## Key Differences from F1

| Feature | F1 (Original) | F2 (Refactored) |
|---------|---------------|-----------------|
| Precision | fp64 | fp32 |
| Parallelism | Cell-parallel (1 thread/cell) | Edge-parallel flux + Cell-parallel update |
| Kernels | 1 monolithic kernel + transfer | 2 kernels: CalculateFlux + UpdateCell |
| Transfer | Explicit copy kernel | No transfer (FLUX stored in side arrays) |
| Data layout | [5][CEL+1] (1-indexed) | [4*CELL] flat (0-indexed) |
| CUDA Graph | No | Yes (2.83x speedup from launch overhead elimination) |
| Mesh | 6675 cells, real hydro-cal data | Same |

## Baseline Performance (B200)

```
Standard mode:    21.8 ms/day (65% launch overhead!)
CUDA Graph mode:   7.7 ms/day (pure compute)
Steps per day:    900 (MDT=3600s, DT=4s)
Kernels per step: 2 (CalculateFlux + UpdateCell)
```

## Implementations
- [x] CUDA (original, baseline)
- [ ] Taichi
- [ ] Warp
- [ ] Kokkos
- [ ] Triton

## Register Tuning Result
No effect (fp32, low register pressure, occupancy already high)
