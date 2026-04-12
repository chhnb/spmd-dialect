"""Independent NumPy reference implementations for correctness validation.

Each function implements the same algorithm as the corresponding Taichi/CUDA
kernel but using pure NumPy array operations. These serve as ground-truth
references for cross-implementation correctness checks.

Interface: run_<case>(params) -> (step_fn, sync_fn, output_array)
  - step_fn: callable that advances the simulation by the configured steps
  - sync_fn: no-op (NumPy is synchronous)
  - output_array: numpy array of the final state
"""
import numpy as np


def _noop():
    pass


# C2: Jacobi 3D 7-point stencil
def run_jacobi3d(N=32, steps=50):
    u = np.random.RandomState(42).rand(N, N, N).astype(np.float32)
    u_new = np.zeros_like(u)

    def step():
        nonlocal u, u_new
        for _ in range(steps):
            u_new[1:-1, 1:-1, 1:-1] = (
                u[0:-2, 1:-1, 1:-1] + u[2:, 1:-1, 1:-1] +
                u[1:-1, 0:-2, 1:-1] + u[1:-1, 2:, 1:-1] +
                u[1:-1, 1:-1, 0:-2] + u[1:-1, 1:-1, 2:]
            ) / 6.0
            u, u_new = u_new, u

    step()
    return _noop, _noop, u


# C3: Heat 2D (5-point stencil with diffusion coefficient)
def run_heat2d(N=64, steps=100):
    u = np.zeros((N, N), dtype=np.float32)
    # Initial hot spot in center
    u[N//4:3*N//4, N//4:3*N//4] = 1.0
    alpha = 0.25  # diffusion coefficient (stable for dx=1, dt=1)

    def step():
        nonlocal u
        for _ in range(steps):
            u_new = u.copy()
            u_new[1:-1, 1:-1] = u[1:-1, 1:-1] + alpha * (
                u[0:-2, 1:-1] + u[2:, 1:-1] +
                u[1:-1, 0:-2] + u[1:-1, 2:] -
                4.0 * u[1:-1, 1:-1]
            )
            u = u_new

    step()
    return _noop, _noop, u


# C10: Gray-Scott reaction-diffusion
def run_grayscott(N=64, steps=100):
    Du, Dv, F, k = 0.16, 0.08, 0.035, 0.065
    u = np.ones((N, N), dtype=np.float32)
    v = np.zeros((N, N), dtype=np.float32)
    # Seed
    r = N // 4
    u[N//2-r:N//2+r, N//2-r:N//2+r] = 0.5
    v[N//2-r:N//2+r, N//2-r:N//2+r] = 0.25

    def laplacian(a):
        return (np.roll(a, 1, 0) + np.roll(a, -1, 0) +
                np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4 * a)

    def step():
        nonlocal u, v
        for _ in range(steps):
            uvv = u * v * v
            u += Du * laplacian(u) - uvv + F * (1 - u)
            v += Dv * laplacian(v) + uvv - (F + k) * v

    step()
    return _noop, _noop, v  # return v field for comparison


# C11: FDTD-2D (3 field updates per step)
def run_fdtd2d(N=64, steps=100):
    ex = np.zeros((N, N), dtype=np.float32)
    ey = np.zeros((N, N), dtype=np.float32)
    hz = np.zeros((N, N), dtype=np.float32)
    coeff = 0.5

    def step():
        nonlocal ex, ey, hz
        for t in range(steps):
            # Update ey
            ey[1:, :] -= coeff * (hz[1:, :] - hz[:-1, :])
            # Update ex
            ex[:, 1:] -= coeff * (hz[:, 1:] - hz[:, :-1])
            # Source
            ex[N//2, N//2] = np.sin(0.1 * t).astype(np.float32)
            # Update hz
            hz[:-1, :-1] -= coeff * (
                ex[:-1, 1:] - ex[:-1, :-1] +
                ey[1:, :-1] - ey[:-1, :-1]
            )

    step()
    return _noop, _noop, hz


# C17: 3D Convolution
def run_conv3d(N=32, steps=1):
    A = np.random.RandomState(42).rand(N, N, N).astype(np.float32)
    B = np.zeros_like(A)

    def step():
        nonlocal B
        for _ in range(steps):
            B[1:-1, 1:-1, 1:-1] = (
                A[0:-2, 1:-1, 1:-1] + A[2:, 1:-1, 1:-1] +
                A[1:-1, 0:-2, 1:-1] + A[1:-1, 2:, 1:-1] +
                A[1:-1, 1:-1, 0:-2] + A[1:-1, 1:-1, 2:] +
                A[1:-1, 1:-1, 1:-1]
            ) / 7.0

    step()
    return _noop, _noop, B


# C18: DOITGEN (tensor contraction)
def run_doitgen(N=32, steps=1):
    NR, NQ, NP = N, N, N
    A = np.random.RandomState(42).rand(NR, NQ, NP).astype(np.float32)
    C4 = np.random.RandomState(43).rand(NP, NP).astype(np.float32)

    def step():
        nonlocal A
        for _ in range(steps):
            for r in range(NR):
                for q in range(NQ):
                    A[r, q, :] = A[r, q, :] @ C4

    step()
    return _noop, _noop, A


# C19: LU Decomposition
def run_lu(N=64, steps=1):
    np.random.seed(42)
    A = np.random.rand(N, N).astype(np.float32) + N * np.eye(N, dtype=np.float32)

    def step():
        nonlocal A
        for _ in range(steps):
            for k in range(N):
                for j in range(k + 1, N):
                    A[j, k] /= A[k, k]
                    for i in range(k + 1, N):
                        A[j, i] -= A[j, k] * A[k, i]

    step()
    return _noop, _noop, A


# C20: ADI (Alternating Direction Implicit)
def run_adi(N=64, steps=3):
    u = np.zeros((N, N), dtype=np.float32)
    v = np.zeros((N, N), dtype=np.float32)
    p = np.zeros((N, N), dtype=np.float32)
    q = np.zeros((N, N), dtype=np.float32)
    DX = 1.0 / N
    DY = 1.0 / N
    DT = 1.0 / steps
    B1 = 2.0
    B2 = 1.0
    mul1 = B1 * DT / (DX * DX)
    mul2 = B2 * DT / (DY * DY)
    a = -mul1 / 2.0
    b = 1.0 + mul1
    c = a
    d = -mul2 / 2.0
    e = 1.0 + mul2
    f = d

    def step():
        nonlocal u, v, p, q
        for t in range(1, steps + 1):
            # Column sweep
            for i in range(1, N - 1):
                v[0, i] = 1.0
                p[i, 0] = 0.0
                q[i, 0] = v[0, i]
                for j in range(1, N - 1):
                    p[i, j] = -c / (a * p[i, j-1] + b)
                    q[i, j] = (-d * u[j, i-1] + (1.0 + 2.0*d) * u[j, i] -
                               f * u[j, i+1] - a * q[i, j-1]) / (a * p[i, j-1] + b)
                v[N-1, i] = 1.0
                for j in range(N - 2, 0, -1):
                    v[j, i] = p[i, j] * v[j+1, i] + q[i, j]
            # Row sweep
            for i in range(1, N - 1):
                u[i, 0] = 1.0
                p[i, 0] = 0.0
                q[i, 0] = u[i, 0]
                for j in range(1, N - 1):
                    p[i, j] = -f / (d * p[i, j-1] + e)
                    q[i, j] = (-a * v[i-1, j] + (1.0 + 2.0*a) * v[i, j] -
                               c * v[i+1, j] - d * q[i, j-1]) / (d * p[i, j-1] + e)
                u[i, N-1] = 1.0
                for j in range(N - 2, 0, -1):
                    u[i, j] = p[i, j] * u[i, j+1] + q[i, j]

    step()
    return _noop, _noop, u


# C21: Gram-Schmidt orthogonalization
def run_gramschmidt(N=64, steps=1):
    M = N
    A = np.array([(i % 97) / 97.0 + 0.01 for i in range(M * M)],
                 dtype=np.float32).reshape(M, M)
    Q = A.copy()
    R = np.zeros((M, M), dtype=np.float32)

    def step():
        nonlocal Q, R
        for _ in range(steps):
            for k in range(M):
                nrm = np.sqrt(np.dot(Q[:, k], Q[:, k]))
                R[k, k] = nrm
                Q[:, k] /= nrm
                for j in range(k + 1, M):
                    R[k, j] = np.dot(Q[:, k], Q[:, j])
                    Q[:, j] -= Q[:, k] * R[k, j]

    step()
    return _noop, _noop, Q
