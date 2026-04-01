# B2: SPH (Smoothed Particle Hydrodynamics)

## Existing code
- **Taichi**: `taichi/python/taichi/examples/simulation/pbf2d.py` (PBF, 1200 particles)
- **Warp**: `warp/warp/examples/core/example_sph.py` (56K particles, wp.HashGrid)

## Kernel specification
- Phase 1: Build spatial hash grid (O(N))
- Phase 2: For each particle, query neighbors within smoothing length
- Phase 3: Compute density via kernel interpolation (variable-length inner loop)
- Phase 4: Compute pressure + viscosity forces
- N: 16K, 32K, 65K particles

## Key computation patterns
- Spatial hash neighbor query (data-dependent, variable-length)
- Gather (read neighbor positions)
- Per-particle reduction (accumulate density/forces)

## Implementation status
- [ ] taichi (adapt from pbf2d.py, scale up)
- [ ] warp (adapt from example_sph.py)
- [ ] numpy baseline
