"""Independent NumPy reference implementations for correctness validation.

Each function implements the EXACT same algorithm, initial conditions,
coefficients, and output field as the corresponding Taichi kernel.
These serve as ground-truth references for cross-implementation checks.

Interface: run_<case>(params) -> (step_fn, sync_fn, output_array)
"""
import numpy as np


def _noop():
    pass


# C2: Jacobi 3D 7-point stencil
# Matches jacobi3d_taichi.py: u initialized to 0, boundary u[:,:,0]=1.0
def run_jacobi3d(N=32, steps=50):
    u = np.zeros((N, N, N), dtype=np.float32)
    u_new = np.zeros_like(u)
    # Boundary: z=0 plane set to 1.0 (matches Taichi init_boundary)
    u[:, :, 0] = 1.0

    for _ in range(steps):
        u_new[1:-1, 1:-1, 1:-1] = (
            u[0:-2, 1:-1, 1:-1] + u[2:, 1:-1, 1:-1] +
            u[1:-1, 0:-2, 1:-1] + u[1:-1, 2:, 1:-1] +
            u[1:-1, 1:-1, 0:-2] + u[1:-1, 1:-1, 2:]
        ) / 6.0
        # Taichi copy_back: u[i,j,k] = u_new[i,j,k] for ALL cells
        u[:] = u_new[:]

    return _noop, _noop, u


# C3: Heat 2D — 5-point stencil, alpha=0.2
# Matches heat2d_taichi.py: u initialized to 0, boundary u[0,:]=1.0
def run_heat2d(N=64, steps=100):
    u = np.zeros((N, N), dtype=np.float32)
    v = np.zeros_like(u)
    # Boundary: top row set to 1.0 (matches Taichi init_boundary)
    u[0, :] = 1.0
    alpha = np.float32(0.2)

    for _ in range(steps):
        v[1:-1, 1:-1] = u[1:-1, 1:-1] + alpha * (
            u[0:-2, 1:-1] + u[2:, 1:-1] +
            u[1:-1, 0:-2] + u[1:-1, 2:] -
            np.float32(4.0) * u[1:-1, 1:-1]
        )
        # Taichi copy_back: u[i,j] = v[i,j] for ALL cells
        u[:] = v[:]

    return _noop, _noop, u


# C10: Gray-Scott reaction-diffusion
# Matches grayscott_taichi.py: F=0.06, k=0.062, returns gu (not gv)
# Interior stencil (no wrap), center seed at [N//2-10:N//2+10]
def run_grayscott(N=64, steps=100):
    Du = np.float32(0.16)
    Dv = np.float32(0.08)
    F = np.float32(0.06)
    k = np.float32(0.062)

    gu = np.ones((N, N), dtype=np.float32)
    gv = np.zeros((N, N), dtype=np.float32)
    gu2 = np.zeros_like(gu)
    gv2 = np.zeros_like(gv)

    # Center seed (matches Taichi init)
    lo, hi = N // 2 - 10, N // 2 + 10
    gu[lo:hi, lo:hi] = 0.5
    gv[lo:hi, lo:hi] = 0.25

    for _ in range(steps):
        # Interior 5-point Laplacian (no wrapping — matches Taichi [1,N-1) bounds)
        lu = (gu[0:-2, 1:-1] + gu[2:, 1:-1] +
              gu[1:-1, 0:-2] + gu[1:-1, 2:] -
              4.0 * gu[1:-1, 1:-1])
        lv = (gv[0:-2, 1:-1] + gv[2:, 1:-1] +
              gv[1:-1, 0:-2] + gv[1:-1, 2:] -
              4.0 * gv[1:-1, 1:-1])
        uvv = gu[1:-1, 1:-1] * gv[1:-1, 1:-1] * gv[1:-1, 1:-1]
        gu2[1:-1, 1:-1] = gu[1:-1, 1:-1] + Du * lu - uvv + F * (1.0 - gu[1:-1, 1:-1])
        gv2[1:-1, 1:-1] = gv[1:-1, 1:-1] + Dv * lv + uvv - (F + k) * gv[1:-1, 1:-1]
        # Taichi copy_back: gu = gu2, gv = gv2 for ALL cells
        gu[:] = gu2[:]
        gv[:] = gv2[:]

    return _noop, _noop, gu  # returns gu, matching Taichi


# C11: FDTD-2D — 3 field updates per step
# Matches fdtd2d_taichi.py: COURANT=0.5, hz[N//2,N//2]=1.0 init, no time source
def run_fdtd2d(N=64, steps=100):
    Nx, Ny = N, N
    COURANT = np.float32(0.5)
    ex = np.zeros((Nx, Ny), dtype=np.float32)
    ey = np.zeros((Nx, Ny), dtype=np.float32)
    hz = np.zeros((Nx, Ny), dtype=np.float32)
    # Point source at center (matches Taichi init)
    hz[Nx // 2, Ny // 2] = np.float32(1.0)

    for _ in range(steps):
        # update_ey: ey[i,j] += COURANT * (hz[i,j] - hz[i-1,j]) for i in [1,Nx), j in [0,Ny)
        ey[1:, :] += COURANT * (hz[1:, :] - hz[:-1, :])
        # update_ex: ex[i,j] -= COURANT * (hz[i,j] - hz[i,j-1]) for i in [0,Nx), j in [1,Ny)
        ex[:, 1:] -= COURANT * (hz[:, 1:] - hz[:, :-1])
        # update_hz: hz[i,j] -= COURANT * (ex[i,j+1]-ex[i,j] + ey[i+1,j]-ey[i,j])
        hz[:-1, :-1] -= COURANT * (
            ex[:-1, 1:] - ex[:-1, :-1] +
            ey[1:, :-1] - ey[:-1, :-1]
        )

    return _noop, _noop, hz


# C17: 3D Convolution — 27-point (3x3x3) uniform stencil
# Matches conv3d_taichi.py: A init = sin(linear_index * 0.001), weight = 1/27
def run_conv3d(N=32, steps=1):
    NX, NY, NZ = N, N, N
    A = np.zeros((NX, NY, NZ), dtype=np.float32)
    B = np.zeros_like(A)

    # Init: A[i,j,k] = sin((i*NY*NZ + j*NZ + k) * 0.001)
    for i in range(NX):
        for j in range(NY):
            for k in range(NZ):
                A[i, j, k] = np.sin((i * NY * NZ + j * NZ + k) * 0.001)

    for _ in range(steps):
        # 27-point stencil: sum over 3x3x3 neighborhood
        for z in range(1, NZ - 1):
            for x in range(1, NX - 1):
                for y in range(1, NY - 1):
                    s = np.float32(0.0)
                    for dz in range(-1, 2):
                        for dy in range(-1, 2):
                            for dx in range(-1, 2):
                                s += A[x + dx, y + dy, z + dz]
                    B[x, y, z] = s / np.float32(27.0)
        # Taichi copy_B_to_A: A[i,j,k] = B[i,j,k] for ALL cells
        A[:] = B[:]

    return _noop, _noop, B


# C18: DOITGEN — tensor contraction A[p,q,:] = A[p,q,:] @ C4
# Matches doitgen_taichi.py: A[p,q,r] = ((p*NQ+q)*NR+r)/(NP*NQ*NR), C4[i,j] = (i*NR+j)/(NR*NR)
def run_doitgen(N=32, steps=1):
    NP, NQ, NR = N, N, N
    A = np.zeros((NP, NQ, NR), dtype=np.float32)
    C4 = np.zeros((NR, NR), dtype=np.float32)
    A_out = np.zeros_like(A)

    # Init A
    for p in range(NP):
        for q in range(NQ):
            for r in range(NR):
                A[p, q, r] = ((p * NQ + q) * NR + r) / (NP * NQ * NR)
    # Init C4
    for i in range(NR):
        for j in range(NR):
            C4[i, j] = (i * NR + j) / (NR * NR)

    for _ in range(steps):
        for r in range(NR):
            for p in range(NP):
                for q in range(NQ):
                    s = np.float32(0.0)
                    for ss in range(NR):
                        s += A[p, q, ss] * C4[ss, r]
                    A_out[p, q, r] = s
        A[:] = A_out[:]

    return _noop, _noop, A_out


# C19: LU Decomposition
# Matches lu_taichi.py: A[i,j] = ((i*N+j)%97)/97.0+0.01, diagonal += N
def run_lu(N=64, steps=1):
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            A[i, j] = ((i * N + j) % 97) / 97.0 + 0.01
            if i == j:
                A[i, j] += N

    for _ in range(steps):
        for k in range(N - 1):
            # Factor column
            for i in range(k + 1, N):
                A[i, k] /= A[k, k]
            # Update submatrix
            for i in range(k + 1, N):
                for j in range(k + 1, N):
                    A[i, j] -= A[i, k] * A[k, j]

    return _noop, _noop, A


# C20: ADI — Alternating Direction Implicit (simplified smoothing sweep)
# Matches adi_taichi.py: u init = (i*(N-1-i)*j*(N-1-j))/(N^4), a=0.1, b=0.8
# Row sweep: v[row,j] = a*u[row-1,j] + b*u[row,j] + a*u[row+1,j]
# Col sweep: u[i,col] = a*v[i,col-1] + b*v[i,col] + a*v[i,col+1]
def run_adi(N=64, steps=3):
    a = np.float32(0.1)
    b = np.float32(0.8)
    u = np.zeros((N, N), dtype=np.float32)
    v = np.zeros((N, N), dtype=np.float32)

    # Init: u[i,j] = (i*(N-1-i)*j*(N-1-j)) / (N*N*N*N)
    Nf = np.float32(N)
    for i in range(N):
        for j in range(N):
            u[i, j] = np.float32(i * (N - 1 - i) * j * (N - 1 - j)) / (Nf * Nf * Nf * Nf)

    for _ in range(steps):
        # Row sweep: v[row,j] for row in [1,N-1), j in [1,N-1)
        for row in range(1, N - 1):
            for j in range(1, N - 1):
                v[row, j] = a * u[row - 1, j] + b * u[row, j] + a * u[row + 1, j]
        # Col sweep: u[i,col] for col in [1,N-1), i in [1,N-1)
        for col in range(1, N - 1):
            for i in range(1, N - 1):
                u[i, col] = a * v[i, col - 1] + b * v[i, col] + a * v[i, col + 1]

    return _noop, _noop, u


# C21: Gram-Schmidt orthogonalization
# Matches gramschmidt_taichi.py: Q[i,j] = ((i*N+j)%97)/97.0+0.01, column-major ops
def run_gramschmidt(N=64, steps=1):
    M = N
    Q = np.zeros((M, N), dtype=np.float32)
    R = np.zeros((N, N), dtype=np.float32)

    # Init Q (matches Taichi init)
    for i in range(M):
        for j in range(N):
            Q[i, j] = ((i * N + j) % 97) / 97.0 + 0.01

    for _ in range(steps):
        for k in range(N):
            # Normalize column k
            nrm = np.float32(0.0)
            for i in range(M):
                nrm += Q[i, k] * Q[i, k]
            nrm = np.sqrt(nrm)
            R[k, k] = nrm
            inv_nrm = np.float32(1.0) / nrm
            Q[:, k] *= inv_nrm
            # Project: for j > k
            for j in range(k + 1, N):
                dot_val = np.float32(0.0)
                for i in range(M):
                    dot_val += Q[i, k] * Q[i, j]
                R[k, j] = dot_val
                Q[:, j] -= dot_val * Q[:, k]

    return _noop, _noop, Q
