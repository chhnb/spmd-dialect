# C2: PIC (Particle-in-Cell)

## Existing code
- **Taichi**: `taichi/python/taichi/examples/simulation/two_stream_instability.py`

## Kernel specification
1. Charge deposition (scatter): particles → grid
2. Poisson solve (stencil): charge → E field
3. Force interpolation (gather): E field → particle acceleration
4. Particle push: update velocity and position

## Implementation status
- [ ] taichi (adapt from two_stream_instability.py)
- [ ] warp (implement from spec)
