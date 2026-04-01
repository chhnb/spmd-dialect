# A3: Wave Equation (2D)

## Existing code
- **Taichi**: `taichi/python/taichi/examples/simulation/waterwave.py` (shallow water, 512x512)
- **Warp**: `warp/warp/examples/core/example_wave.py` (5-point Laplacian, 128x128)

## Kernel specification
- 2D wave: h_new = 2*h - h_old + dt^2 * c^2 * laplacian(h)
- 5-point Laplacian stencil + double buffer swap
- Grid size: 1024x1024, 2048x2048
- Steps per timed call: 100

## Implementation status
- [ ] taichi (adapt from waterwave.py)
- [ ] warp (adapt from example_wave.py, scale up grid)
- [ ] numpy baseline
