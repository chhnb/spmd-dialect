"""2D Shallow Water Equations (Osher Riemann solver) — NumPy baseline.

Dam-break problem on an unstructured quad mesh (stored as cell-neighbor arrays).
Faithfully translates the hydro-cal CUDA kernel (calculate_gpu.cu).
"""
import numpy as np
import math


# ---------------------------------------------------------------------------
# Physical / solver constants
# ---------------------------------------------------------------------------
G = 9.81
HALF_G = 4.905  # g/2
HM1 = 0.001     # drying threshold
HM2 = 0.01      # shallow threshold
VMIN = 0.001     # minimum velocity for shallow cells
C0 = 1.0         # weir coefficient
C1 = 0.3         # submergence coefficient
MANNING_N = 0.03 # Manning roughness


# ---------------------------------------------------------------------------
# Mesh generation: structured quad grid stored in unstructured format
# ---------------------------------------------------------------------------
def make_dam_break_mesh(N):
    """Generate an NxN structured quad mesh for dam-break.

    Returns dict with 1-indexed arrays matching hydro-cal conventions.
    Cell index 0 is unused (hydro-cal uses 1-based indexing).
    """
    CEL = N * N
    dx = 1.0  # uniform spacing

    # Allocate 1-indexed arrays (index 0 unused)
    NAC = np.zeros((5, CEL + 1), dtype=np.int32)   # [edge 1..4][cell]
    KLAS = np.zeros((5, CEL + 1), dtype=np.int32)
    SIDE = np.zeros((5, CEL + 1), dtype=np.float64)
    COSF = np.zeros((5, CEL + 1), dtype=np.float64)
    SINF = np.zeros((5, CEL + 1), dtype=np.float64)
    AREA = np.zeros(CEL + 1, dtype=np.float64)
    ZBC = np.zeros(CEL + 1, dtype=np.float64)       # bed elevation (flat=0)
    FNC = np.zeros(CEL + 1, dtype=np.float64)        # g * n^2

    # Edge normal directions for structured quad:
    # edge 1: south (-y), edge 2: east (+x), edge 3: north (+y), edge 4: west (-x)
    edge_cos = [0.0, 0.0, 1.0, 0.0, -1.0]  # index 1..4
    edge_sin = [0.0, -1.0, 0.0, 1.0, 0.0]

    for i in range(N):
        for j in range(N):
            pos = i * N + j + 1  # 1-indexed
            AREA[pos] = dx * dx
            FNC[pos] = G * MANNING_N * MANNING_N

            for e in range(1, 5):
                SIDE[e][pos] = dx
                COSF[e][pos] = edge_cos[e]
                SINF[e][pos] = edge_sin[e]

            # Neighbors: south, east, north, west
            # south (i-1)
            if i > 0:
                NAC[1][pos] = (i - 1) * N + j + 1
            else:
                KLAS[1][pos] = 4  # wall

            # east (j+1)
            if j < N - 1:
                NAC[2][pos] = i * N + (j + 1) + 1
            else:
                KLAS[2][pos] = 4  # wall

            # north (i+1)
            if i < N - 1:
                NAC[3][pos] = (i + 1) * N + j + 1
            else:
                KLAS[3][pos] = 4  # wall

            # west (j-1)
            if j > 0:
                NAC[4][pos] = i * N + (j - 1) + 1
            else:
                KLAS[4][pos] = 4  # wall

    # Pre-compute SLCOS, SLSIN
    SLCOS = SIDE * COSF
    SLSIN = SIDE * SINF

    # Initial conditions: dam break
    H = np.full(CEL + 1, HM1, dtype=np.float64)
    U = np.zeros(CEL + 1, dtype=np.float64)
    V = np.zeros(CEL + 1, dtype=np.float64)
    Z = np.zeros(CEL + 1, dtype=np.float64)
    W = np.zeros(CEL + 1, dtype=np.float64)

    for i in range(N):
        for j in range(N):
            pos = i * N + j + 1
            if j < N // 2:
                H[pos] = 2.0  # left: deep
            else:
                H[pos] = 0.5  # right: shallow
            Z[pos] = H[pos] + ZBC[pos]

    return dict(
        CEL=CEL, N=N, dx=dx,
        NAC=NAC, KLAS=KLAS, SIDE=SIDE, COSF=COSF, SINF=SINF,
        SLCOS=SLCOS, SLSIN=SLSIN, AREA=AREA, ZBC=ZBC, FNC=FNC,
        H=H, U=U, V=V, Z=Z, W=W,
    )


# ---------------------------------------------------------------------------
# Osher Riemann solver (faithful port from calculate_gpu.cu)
# ---------------------------------------------------------------------------
def QF(h, u, v):
    """Compute flux components."""
    f0 = h * u
    return np.array([f0, f0 * u, f0 * v, HALF_G * h * h])


def osher(QL, QR, FIL, H_pos):
    """Osher approximate Riemann solver for shallow water equations."""
    CR = math.sqrt(G * QR[0])
    FIR = QR[1] - 2.0 * CR
    UA = (FIL + FIR) / 2.0
    CA = abs((FIL - FIR) / 4.0)
    CL = math.sqrt(G * H_pos)

    FLR = np.zeros(4)

    # Determine K2
    if CA < UA:
        K2 = 1
    elif UA >= 0.0 and UA < CA:
        K2 = 2
    elif UA >= -CA and UA < 0.0:
        K2 = 3
    else:
        K2 = 4

    # Determine K1
    if QL[1] < CL and QR[1] >= -CR:
        K1 = 1
    elif QL[1] >= CL and QR[1] >= -CR:
        K1 = 2
    elif QL[1] < CL and QR[1] < -CR:
        K1 = 3
    else:
        K1 = 4

    # QS template calls — state variables may be mutated (FIL, FIR)
    fil = FIL
    fir = FIR

    def qs(T, sign):
        nonlocal fil, fir
        if T == 1:
            F = QF(QL[0], QL[1], QL[2])
        elif T == 2:
            US = fil / 3.0
            HS = US * US / G
            F = QF(HS, US, QL[2])
        elif T == 3:
            ua_ = (fil + fir) / 2.0
            fil = fil - ua_
            HA = fil * fil / (4.0 * G)
            F = QF(HA, ua_, QL[2])
        elif T == 5:
            ua_ = (fil + fir) / 2.0
            fir = fir - ua_
            HA = fir * fir / (4.0 * G)
            F = QF(HA, ua_, QR[2])
        elif T == 6:
            US = fir / 3.0
            HS = US * US / G
            F = QF(HS, US, QR[2])
        elif T == 7:
            F = QF(QR[0], QR[1], QR[2])
        else:
            return
        FLR[:] += F * sign

    # Dispatch table matching the CUDA switch-case
    dispatch = {
        (1, 1): [(2, 1)],
        (1, 2): [(3, 1)],
        (1, 3): [(5, 1)],
        (1, 4): [(6, 1)],
        (2, 1): [(1, 1)],
        (2, 2): [(1, 1), (2, -1), (3, 1)],
        (2, 3): [(1, 1), (2, -1), (5, 1)],
        (2, 4): [(1, 1), (2, -1), (6, 1)],
        (3, 1): [(2, 1), (6, -1), (7, 1)],
        (3, 2): [(3, 1), (6, -1), (7, 1)],
        (3, 3): [(5, 1), (6, -1), (7, 1)],
        (3, 4): [(7, 1)],
        (4, 1): [(1, 1), (6, -1), (7, 1)],
        (4, 2): [(1, 1), (2, -1), (3, 1), (6, -1), (7, 1)],
        (4, 3): [(1, 1), (2, -1), (5, 1), (6, -1), (7, 1)],
        (4, 4): [(1, 1), (2, -1), (7, 1)],
    }

    for T, sign in dispatch.get((K1, K2), []):
        qs(T, sign)

    return FLR


# ---------------------------------------------------------------------------
# Per-cell flux and state update (port of calculate_FLUX + calculate_HUV)
# ---------------------------------------------------------------------------
def compute_step(mesh):
    """One time step of the 2D shallow water kernel."""
    CEL = mesh["CEL"]
    NAC = mesh["NAC"]
    KLAS = mesh["KLAS"]
    SIDE = mesh["SIDE"]
    COSF = mesh["COSF"]
    SINF = mesh["SINF"]
    SLCOS = mesh["SLCOS"]
    SLSIN = mesh["SLSIN"]
    AREA = mesh["AREA"]
    ZBC = mesh["ZBC"]
    FNC = mesh["FNC"]
    H_pre = mesh["H"]
    U_pre = mesh["U"]
    V_pre = mesh["V"]
    Z_pre = mesh["Z"]

    # CFL-based time step
    DT = 0.5 * mesh["dx"] / (math.sqrt(G * 2.0) + 1e-6)
    QLUA = 0.0

    H_res = np.copy(H_pre)
    U_res = np.copy(U_pre)
    V_res = np.copy(V_pre)
    Z_res = np.copy(Z_pre)
    W_res = np.copy(mesh["W"])

    for pos in range(1, CEL + 1):
        H1 = H_pre[pos]
        U1 = U_pre[pos]
        V1 = V_pre[pos]
        BI = ZBC[pos]

        HI = max(H1, HM1)
        UI = U1
        VI = V1
        if HI <= HM2:
            UI = math.copysign(VMIN, UI) if UI != 0 else VMIN
            VI = math.copysign(VMIN, VI) if VI != 0 else VMIN
        ZI = max(Z_pre[pos], ZBC[pos])

        WH = 0.0
        WU = 0.0
        WV = 0.0

        for j in range(1, 5):
            NC = NAC[j][pos]
            KP = KLAS[j][pos]
            COSJ = COSF[j][pos]
            SINJ = SINF[j][pos]

            # Left state in edge-local coords
            QL = np.array([HI, UI * COSJ + VI * SINJ, VI * COSJ - UI * SINJ])
            CL_val = math.sqrt(G * HI)
            FIL = QL[1] + 2.0 * CL_val

            # Neighbor state
            if NC == 0:
                HC = 0.0
                BC = 0.0
                ZC = 0.0
                UC = 0.0
                VC = 0.0
            else:
                HC = max(H_pre[NC], HM1)
                BC = ZBC[NC]
                ZC = max(ZBC[NC], Z_pre[NC])
                UC = U_pre[NC]
                VC = V_pre[NC]

            # --- Flux computation (port of calculate_FLUX) ---
            flux = np.zeros(4)

            if KP == 4:
                # Wall boundary: no flow, pressure only
                flux[3] = HALF_G * H1 * H1
            elif KP == 5:
                # Outflow boundary
                ql1 = max(QL[1], 0.0)
                flux[0] = H1 * ql1
                flux[1] = flux[0] * ql1
                flux[3] = HALF_G * H1 * H1
            elif KP != 0:
                # Other boundary types: treat as wall for dam-break
                flux[3] = HALF_G * H1 * H1
            elif HI <= HM1 and HC <= HM1:
                # Both dry
                pass
            elif ZI <= BC:
                flux[0] = -C1 * HC ** 1.5
                flux[1] = HI * QL[1] * abs(QL[1])
                flux[3] = HALF_G * HI * HI
            elif ZC <= BI:
                flux[0] = C1 * HI ** 1.5
                flux[1] = HI * abs(QL[1]) * QL[1]
                flux[2] = HI * abs(QL[1]) * QL[2]
            elif HI <= HM2:
                if ZC > ZI:
                    DH = max(ZC - ZBC[pos], HM1)
                    UN = -C1 * math.sqrt(DH)
                    flux[0] = DH * UN
                    flux[1] = flux[0] * UN
                    flux[2] = flux[0] * (VC * COSJ - UC * SINJ)
                    flux[3] = HALF_G * HI * HI
                else:
                    flux[0] = C1 * HI ** 1.5
                    flux[3] = HALF_G * HI * HI
            elif HC <= HM2:
                if ZI > ZC:
                    DH = max(ZI - BC, HM1)
                    UN = C1 * math.sqrt(DH)
                    HC1 = ZC - ZBC[pos]
                    flux[0] = DH * UN
                    flux[1] = flux[0] * UN
                    flux[2] = flux[0] * QL[2]
                    flux[3] = HALF_G * HC1 * HC1
                else:
                    flux[0] = -C1 * HC ** 1.5
                    flux[1] = HI * QL[1] * QL[1]
                    flux[3] = HALF_G * HI * HI
            else:
                # Both wet, interior edge — Osher Riemann solver
                if pos < NC:
                    QR = np.zeros(3)
                    QR[0] = max(ZC - ZBC[pos], HM1)
                    UR = UC * COSJ + VC * SINJ
                    QR[1] = UR * min(HC / QR[0], 1.5)
                    if HC <= HM2 or QR[0] <= HM2:
                        QR[1] = math.copysign(VMIN, UR) if UR != 0 else VMIN
                    QR[2] = VC * COSJ - UC * SINJ
                    FLR_OSHER = osher(QL, QR, FIL, H_pre[pos])
                    flux[0] = FLR_OSHER[0]
                    flux[1] = FLR_OSHER[1] + (1.0 - min(HC / QR[0], 1.5)) * HC * UR * UR / 2.0
                    flux[2] = FLR_OSHER[2]
                    flux[3] = FLR_OSHER[3]
                else:
                    # Mirror: compute from neighbor perspective
                    COSJ1 = -COSJ
                    SINJ1 = -SINJ
                    QL1 = np.array([
                        H_pre[NC],
                        U_pre[NC] * COSJ1 + V_pre[NC] * SINJ1,
                        V_pre[NC] * COSJ1 - U_pre[NC] * SINJ1,
                    ])
                    CL1 = math.sqrt(G * H_pre[NC])
                    FIL1 = QL1[1] + 2.0 * CL1
                    HC2 = max(HI, HM1)
                    ZC1 = max(ZBC[pos], ZI)
                    QR1 = np.zeros(3)
                    QR1[0] = max(ZC1 - ZBC[NC], HM1)
                    UR1 = UI * COSJ1 + VI * SINJ1
                    QR1[1] = UR1 * min(HC2 / QR1[0], 1.5)
                    if HC2 <= HM2 or QR1[0] <= HM2:
                        QR1[1] = math.copysign(VMIN, UR1) if UR1 != 0 else VMIN
                    QR1[2] = VI * COSJ1 - UI * SINJ1
                    FLR1 = osher(QL1, QR1, FIL1, H_pre[NC])
                    flux[0] = -FLR1[0]
                    flux[1] = FLR1[1] + (1.0 - min(HC2 / QR1[0], 1.5)) * HC2 * UR1 * UR1 / 2.0
                    flux[2] = FLR1[2]
                    ZA = math.sqrt(FLR1[3] / HALF_G) + BC
                    HC3 = max(ZA - ZBC[pos], 0.0)
                    flux[3] = HALF_G * HC3 * HC3

            # Accumulate fluxes (port of calculate_WHUV)
            SL = SIDE[j][pos]
            SLCA = SLCOS[j][pos]
            SLSA = SLSIN[j][pos]
            FLR_1 = flux[1] + flux[3]
            FLR_2 = flux[2]
            WH += SL * flux[0]
            WU += SLCA * FLR_1 - SLSA * FLR_2
            WV += SLSA * FLR_1 + SLCA * FLR_2

        # --- State update (port of calculate_HUV) ---
        DTA = DT / AREA[pos]
        WDTA = DTA
        H2 = max(H1 - WDTA * WH + QLUA, HM1)
        Z2 = H2 + BI

        if H2 <= HM1:
            U2 = 0.0
            V2 = 0.0
        elif H2 <= HM2:
            U2 = math.copysign(min(VMIN, abs(U1)), U1) if U1 != 0 else 0.0
            V2 = math.copysign(min(VMIN, abs(V1)), V1) if V1 != 0 else 0.0
        else:
            QX1 = H1 * U1
            QY1 = H1 * V1
            DTAU = WDTA * WU
            DTAV = WDTA * WV
            WSF = FNC[pos] * math.sqrt(U1 * U1 + V1 * V1) / (H1 ** 0.33333)
            U2 = (QX1 - DTAU - DT * WSF * U1) / H2
            V2 = (QY1 - DTAV - DT * WSF * V1) / H2
            if H2 > HM2:
                U2 = math.copysign(min(abs(U2), 15.0), U2)
                V2 = math.copysign(min(abs(V2), 15.0), V2)

        W2 = math.sqrt(U2 * U2 + V2 * V2)

        H_res[pos] = H2
        U_res[pos] = U2
        V_res[pos] = V2
        Z_res[pos] = Z2
        W_res[pos] = W2

    # Update mesh state
    mesh["H"][:] = H_res
    mesh["U"][:] = U_res
    mesh["V"][:] = V_res
    mesh["Z"][:] = Z_res
    mesh["W"][:] = W_res


# ---------------------------------------------------------------------------
# Benchmark interface
# ---------------------------------------------------------------------------
def run(N, steps=1, backend="cpu"):
    mesh = make_dam_break_mesh(N)

    def step():
        for _ in range(steps):
            compute_step(mesh)

    return step, None, mesh["H"]


def run_real(steps=1, backend="cpu"):
    """Run on the real hydro-cal mesh (6675 cells)."""
    from mesh_loader import load_hydro_mesh
    mesh = load_hydro_mesh()
    # compute_step needs dx for DT; estimate from min SIDE
    min_side = mesh["SIDE"][1][1:mesh["CEL"]+1][mesh["SIDE"][1][1:mesh["CEL"]+1] > 0].min()
    mesh["dx"] = min_side

    def step():
        for _ in range(steps):
            compute_step(mesh)

    return step, None, mesh["H"]
