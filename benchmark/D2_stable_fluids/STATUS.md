# D2: Stable Fluids (Incompressible Navier-Stokes)

## Existing code
- **Taichi**: `taichi/python/taichi/examples/simulation/stable_fluid.py` (512x512)
- **Warp**: `warp/warp/examples/core/example_fluid.py` (256x128)

## Kernel specification
1. Advection (semi-Lagrangian backtracing + bilinear interpolation)
2. Divergence (2-point stencil)
3. Pressure Jacobi (5-point stencil, 50-500 iterations)
4. Pressure gradient subtraction

## Implementation status
- [ ] taichi (adapt from stable_fluid.py)
- [ ] warp (adapt from example_fluid.py)
- [ ] numpy baseline
