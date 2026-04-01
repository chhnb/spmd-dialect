# B1: N-body (Direct, All-pairs)

## Existing code
- **Taichi**: `taichi/python/taichi/examples/simulation/nbody.py` (3000 particles, O(N^2))
- **Warp**: `warp/warp/examples/tile/example_tile_nbody.py` (16K bodies, tiled with wp.tile_load)

## Kernel specification
- All-pairs gravitational/LJ force: F_i = sum_{j!=i} f(pos_i, pos_j)
- O(N^2) computation, compute-bound
- N: 4096, 8192, 16384, 32768

## Key comparison
- Taichi: naive O(N^2), no shared memory tiling
- Warp: wp.tile_load() for j-tile blocking → shared memory reuse
- Triton: can be written but no tensor core, no pairwise optimization

## Implementation status
- [ ] taichi (adapt from nbody.py, scale up N)
- [ ] warp (adapt from example_tile_nbody.py)
- [ ] triton (implement to show limitation)
- [ ] numpy baseline
