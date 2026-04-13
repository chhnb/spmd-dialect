"""1D electrostatic PIC — Taichi.
Aligned with pic1d_benchmark.cu: DT=0.1, DX=1.0, Gauss-law field solve,
4 kernels/step: deposit, field_solve, gather, push."""
import taichi as ti


def run(n_particles=16384, n_grid=512, steps=1, backend="cuda"):
    ti.init(arch=ti.cuda if backend == "cuda" else ti.cpu, default_fp=ti.f32)
    Np = n_particles
    Ng = n_grid
    DT = 0.1
    DX = 1.0
    QM = -1.0  # charge/mass ratio

    xp = ti.field(dtype=ti.f32, shape=Np)
    vp = ti.field(dtype=ti.f32, shape=Np)
    rho = ti.field(dtype=ti.f32, shape=Ng)
    E = ti.field(dtype=ti.f32, shape=Ng)
    Ep = ti.field(dtype=ti.f32, shape=Np)  # E at particle positions

    @ti.kernel
    def init():
        # Uniform particle distribution matching CUDA
        for p in xp:
            xp[p] = (ti.cast(p, ti.f32) + 0.5) * (ti.cast(Ng, ti.f32) * DX / ti.cast(Np, ti.f32))
            vp[p] = 0.0

    @ti.kernel
    def deposit():
        # CIC charge deposition
        for i in rho:
            rho[i] = 0.0
        for p in xp:
            xpos = xp[p]
            ic = ti.cast(ti.floor(xpos / DX), ti.i32)
            if ic < 0: ic = 0
            if ic >= Ng - 1: ic = Ng - 2
            frac = xpos / DX - ti.cast(ic, ti.f32)
            ti.atomic_add(rho[ic], (1.0 - frac) / DX)
            ti.atomic_add(rho[ic + 1], frac / DX)

    @ti.kernel
    def field_solve():
        # Gauss's law scan matching CUDA
        n0 = ti.cast(Np, ti.f32) / (ti.cast(Ng, ti.f32) * DX)
        cumsum = ti.cast(0.0, ti.f32)
        ti.loop_config(serialize=True)
        for i in range(Ng):
            cumsum += (rho[i] - n0) * DX
            E[i] = -cumsum

    @ti.kernel
    def gather():
        # CIC interpolation of E to particle positions
        for p in xp:
            xpos = xp[p]
            ic = ti.cast(ti.floor(xpos / DX), ti.i32)
            if ic < 0: ic = 0
            if ic >= Ng - 1: ic = Ng - 2
            frac = xpos / DX - ti.cast(ic, ti.f32)
            Ep[p] = (1.0 - frac) * E[ic] + frac * E[ic + 1]

    @ti.kernel
    def push():
        # Leapfrog push
        for p in xp:
            vp[p] += QM * Ep[p] * DT
            xnew = xp[p] + vp[p] * DT
            # Periodic boundary
            L = ti.cast(Ng, ti.f32) * DX
            if xnew < 0.0:
                xnew += L
            if xnew >= L:
                xnew -= L
            xp[p] = xnew

    init()

    def step_fn():
        for _ in range(steps):
            deposit()
            field_solve()
            gather()
            push()

    return step_fn, ti.sync, xp
