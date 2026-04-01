# B3: DEM (Discrete Element Method)

## Existing code
- **Warp**: `warp/warp/examples/core/example_dem.py` (65K particles, HashGrid + contact forces)

## Kernel specification
- Neighbor query via spatial hash
- Pairwise contact force: normal + tangential + cohesion
- Atomic force scatter to both particles
- N: 32K, 65K particles

## Implementation status
- [ ] warp (adapt from example_dem.py)
- [ ] taichi (implement from spec)
- [ ] numpy baseline
