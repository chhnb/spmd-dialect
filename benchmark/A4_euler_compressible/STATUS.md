# A4: Compressible Euler Equations

## Existing code
- **Taichi**: `taichi/python/taichi/examples/simulation/euler.py`
  - MUSCL reconstruction (5-point stencil) + HLLC Riemann solver
  - 512x512 grid

## Kernel specification
- Compressible Euler: conservation of mass, momentum, energy
- MUSCL-Hancock reconstruction: 5-point stencil per direction
- HLLC approximate Riemann solver for flux computation
- RK2 or forward Euler time integration
- Grid size: 512x512, 1024x1024

## Implementation status
- [ ] taichi (adapt from euler.py)
- [ ] warp (implement from spec)
- [ ] numpy baseline
