# Simulation Kernel Benchmark Suite

Cross-framework benchmark of classic simulation kernels.

## Kernel Matrix (12 kernels × 4+ frameworks)

### Group A: Structured Grid
| ID | Kernel | Pattern | Taichi | Warp | Kokkos | Halide | CUDA |
|----|--------|---------|--------|------|--------|--------|------|
| A1 | Jacobi 2D (5-point) | simple stencil | ✓ | ✓ | ✓ | ✓ | TODO |
| A2 | LBM D2Q9 | multi-component stencil | ✓ | TODO | - | - | TODO |
| A3 | Wave Equation | time-stepping stencil | ✓ | ✓ | - | - | TODO |
| A4 | Euler (compressible) | high-order stencil + Riemann | ✓ | - | - | - | TODO |

### Group B: Particle
| ID | Kernel | Pattern | Taichi | Warp | Kokkos | CUDA |
|----|--------|---------|--------|------|--------|------|
| B1 | N-body (direct) | all-pairs gather + reduction | ✓ | ✓ (tiled) | - | TODO |
| B2 | SPH density+force | neighbor query + gather | ✓ | ✓ | - | TODO |
| B3 | DEM contact | neighbor query + atomic scatter | - | ✓ | - | TODO |

### Group C: Hybrid Particle-Grid
| ID | Kernel | Pattern | Taichi | Warp |
|----|--------|---------|--------|------|
| C1 | MPM (P2G+grid+G2P) | scatter + stencil + gather | ✓ | ✓ (APIC) |
| C2 | PIC (Particle-in-Cell) | charge deposition + Poisson + force interp | ✓ | - |

### Group D: Mesh / Multi-kernel Pipeline
| ID | Kernel | Pattern | Taichi | Warp | NumPy | PyTorch | JAX |
|----|--------|---------|--------|------|-------|---------|-----|
| D1 | Cloth spring mass | per-edge + atomic scatter | ✓ | ✓ | ✓ | ✓ | ✓ |
| D2 | Stable Fluids | advection + divergence + Jacobi | ✓ | ✓ | - | - | - |

### Group E: Primitives
| ID | Kernel | Pattern | Taichi | Warp | Kokkos | NumPy |
|----|--------|---------|--------|------|--------|-------|
| E1 | Global Reduction | hierarchical vs atomic | ✓ | TODO | ✓ | ✓ |

## Source status

- **existing**: kernel code taken from framework's own examples/benchmarks
- **adapted**: kernel logic from framework examples, wrapped in our harness
- **written**: implemented from scratch to match the reference kernel

## Running

```bash
# Run single benchmark
cd A1_jacobi_2d && python run.py

# Run all
python run_all.py
```

## Output

Each benchmark produces `results.csv` with columns:
  framework, backend, problem_size, kernel_time_ms, total_time_ms
