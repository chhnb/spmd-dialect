# A2: LBM D2Q9 (Lattice Boltzmann Method)

## Existing code
- **Taichi**: `taichi/python/taichi/examples/simulation/karman_vortex_street.py`
  - collide_and_stream() + update_macro_var(), needs GUI removal
- **Warp**: needs implementation
- **Triton**: needs implementation (demonstrates stencil limitation)

## Kernel specification
- D2Q9 lattice: 9 velocity directions per grid point
- Two steps: collision (BGK) + streaming (gather from shifted neighbors)
- Grid size: 1024x512, 2048x1024
- Steps per timed call: 100

## Implementation status
- [ ] taichi (adapt from karman_vortex_street.py)
- [ ] warp
- [ ] triton
- [ ] numpy baseline
