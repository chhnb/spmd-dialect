# C1: MPM (Material Point Method)

## Existing code
- **Taichi**: `taichi/python/taichi/examples/simulation/mpm99.py` (9000 particles, 128x128 grid)
- **Warp**: `warp/warp/examples/fem/example_apic_fluid.py` (APIC variant)

## Kernel specification
Three-phase kernel per substep:
1. P2G (Particle-to-Grid): scatter particle mass/momentum to grid via quadratic B-spline
2. Grid operations: apply gravity, enforce boundary conditions
3. G2P (Grid-to-Particle): gather velocity from grid, update particle position

## Implementation status
- [ ] taichi (adapt from mpm99.py)
- [ ] warp (adapt from example_apic_fluid.py)
