# Launch Overhead Study: 127 Configurations, 60 Kernel Types

**GPU**: NVIDIA B200 (sm_100) | **Framework**: Taichi (Python DSL)

## Summary

- **90% (114/127) of configurations are launch-overhead-dominated (>50%)**
- Python loop overhead = ~15-26 μs/step (constant across all kernels)
- Only O(N²) algorithms, large grids (4096²+), and multi-sub-step solvers are compute-dominated

## Overhead Decomposition (Heat2D 512², 1000 steps)

| Method | μs/step | Overhead Source |
|--------|---------|----------------|
| CUDA Graph | 6.1 | Pure GPU compute |
| C++ async loop | 11.1 | +5.0 μs launch OH |
| Taichi Python | ~16 | +5.0 μs Python OH |
| C++ sync loop | 225 | +214 μs sync OH |

## 60 Kernel Types

### Grid PDE (26)
Heat 1D/2D/3D, Wave, Jacobi, Poisson RB-GS, Burgers 2D, ConvDiff,
AdvDiff, Allen-Cahn, Cahn-Hilliard, Gray-Scott 2D/3D, Fisher-KPP,
Kuramoto-Sivashinsky, Biharmonic, Schrodinger, SWE (Lax-Fr/Roe/Osher),
FDTD Maxwell, Elastic Wave, Lid-Driven Cavity, Level Set, VOF,
DG P1, Pseudo-Spectral, MHD Ideal, Richards, Saltwater Intrusion

### Particle Methods (13)
N-Body, SPH, PBF, DEM, PIC Plasma, Vortex, Monte Carlo,
Spring Network, LJ-MD, Langevin, Peridynamics, DLA, DPD

### Structural/Mesh (4)
FEM Explicit, Cloth Spring, Meshfree RBF, Lattice Spring

### Multi-Kernel Solvers (8)
MPM, Stable Fluids, Hydro-Cal SWE, TopOpt SIMP,
Multigrid V-cycle, CG Iteration, NS Fractional Step, BEM

### Statistical/Stochastic (5)
Ising MC, SDE Ornstein-Uhlenbeck, Boltzmann Transport,
Compressible Euler 1D, Lotka-Volterra RD

### Biology (2)
3-Species Diffusion, Lotka-Volterra

## Application Domains (30+)
Thermal, Acoustics, Astrophysics, Biology, CFD, Chaos, Chemistry,
Electromagnetics, Fluids, Granular, Geochemistry, Groundwater,
Linear Algebra, Materials Science, Mathematics, Molecular Dynamics,
Nonlinear Waves, Optimization, Plasma Physics, Quantum Mechanics,
Seismology, Statistical Mechanics, Stochastic Processes,
Structural Mechanics, Textile, Transport, Compressible Flow,
Population Dynamics, Topology Optimization, Boundary Elements
