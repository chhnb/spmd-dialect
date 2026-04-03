"""2D Shallow Water Equations (Osher Riemann solver) — Warp.

Dam-break on unstructured quad mesh. Port of hydro-cal calculate_gpu.cu.
"""
import numpy as np
import warp as wp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G = 9.81
HALF_G = 4.905
HM1_C = 0.001
HM2_C = 0.01
VMIN_C = 0.001
C0_C = 1.0
C1_C = 0.3
MANNING_N = 0.03


# ---------------------------------------------------------------------------
# Warp helper functions
# ---------------------------------------------------------------------------
@wp.func
def QF(h: wp.float64, u: wp.float64, v: wp.float64) -> wp.vec4d:
    f0 = h * u
    return wp.vec4d(f0, f0 * u, f0 * v, wp.float64(4.905) * h * h)


@wp.func
def safe_copysign(val: wp.float64, sign_val: wp.float64) -> wp.float64:
    if sign_val >= 0.0:
        return wp.abs(val)
    else:
        return -wp.abs(val)


@wp.func
def osher_solver(QL_h: wp.float64, QL_u: wp.float64, QL_v: wp.float64,
                 QR_h: wp.float64, QR_u: wp.float64, QR_v: wp.float64,
                 FIL_in: wp.float64, H_pos: wp.float64) -> wp.vec4d:
    CR = wp.sqrt(wp.float64(9.81) * QR_h)
    FIR_v = QR_u - wp.float64(2.0) * CR
    UA = (FIL_in + FIR_v) / wp.float64(2.0)
    CA = wp.abs((FIL_in - FIR_v) / wp.float64(4.0))
    CL_v = wp.sqrt(wp.float64(9.81) * H_pos)

    FLR = wp.vec4d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))

    K2 = 0
    if CA < UA:
        K2 = 1
    elif UA >= 0.0 and UA < CA:
        K2 = 2
    elif UA >= -CA and UA < 0.0:
        K2 = 3
    else:
        K2 = 4

    K1 = 0
    if QL_u < CL_v and QR_u >= -CR:
        K1 = 1
    elif QL_u >= CL_v and QR_u >= -CR:
        K1 = 2
    elif QL_u < CL_v and QR_u < -CR:
        K1 = 3
    else:
        K1 = 4

    fil = FIL_in
    fir = FIR_v

    # Dispatch: encode each (K1,K2) case inline
    if K1 == 1:
        if K2 == 1:  # QS<2>(+1)
            US = fil / wp.float64(3.0)
            HS = US * US / wp.float64(9.81)
            FLR = FLR + QF(HS, US, QL_v)
        elif K2 == 2:  # QS<3>(+1)
            ua_ = (fil + fir) / wp.float64(2.0)
            fil = fil - ua_
            HA = fil * fil / (wp.float64(4.0) * wp.float64(9.81))
            FLR = FLR + QF(HA, ua_, QL_v)
        elif K2 == 3:  # QS<5>(+1)
            ua_ = (fil + fir) / wp.float64(2.0)
            fir = fir - ua_
            HA = fir * fir / (wp.float64(4.0) * wp.float64(9.81))
            FLR = FLR + QF(HA, ua_, QR_v)
        else:  # K2 == 4, QS<6>(+1)
            US = fir / wp.float64(3.0)
            HS = US * US / wp.float64(9.81)
            FLR = FLR + QF(HS, US, QR_v)
    elif K1 == 2:
        if K2 == 1:  # QS<1>(+1)
            FLR = FLR + QF(QL_h, QL_u, QL_v)
        elif K2 == 2:  # QS<1>(+1), QS<2>(-1), QS<3>(+1)
            FLR = FLR + QF(QL_h, QL_u, QL_v)
            US2 = fil / wp.float64(3.0)
            HS2 = US2 * US2 / wp.float64(9.81)
            FLR = FLR + QF(HS2, US2, QL_v)
            ua_ = (fil + fir) / wp.float64(2.0)
            fil = fil - ua_
            HA = fil * fil / (wp.float64(4.0) * wp.float64(9.81))
            FLR = FLR + QF(HA, ua_, QL_v)
        elif K2 == 3:  # QS<1>(+1), QS<2>(-1), QS<5>(+1)
            FLR = FLR + QF(QL_h, QL_u, QL_v)
            US2 = fil / wp.float64(3.0)
            HS2 = US2 * US2 / wp.float64(9.81)
            FLR = FLR + QF(HS2, US2, QL_v)
            ua_ = (fil + fir) / wp.float64(2.0)
            fir = fir - ua_
            HA = fir * fir / (wp.float64(4.0) * wp.float64(9.81))
            FLR = FLR + QF(HA, ua_, QR_v)
        else:  # K2 == 4, QS<1>(+1), QS<2>(-1), QS<6>(+1)
            FLR = FLR + QF(QL_h, QL_u, QL_v)
            US2 = fil / wp.float64(3.0)
            HS2 = US2 * US2 / wp.float64(9.81)
            FLR = FLR + QF(HS2, US2, QL_v)
            US6 = fir / wp.float64(3.0)
            HS6 = US6 * US6 / wp.float64(9.81)
            FLR = FLR + QF(HS6, US6, QR_v)
    elif K1 == 3:
        if K2 == 1:  # QS<2>(+1), QS<6>(-1), QS<7>(+1)
            US2 = fil / wp.float64(3.0)
            HS2 = US2 * US2 / wp.float64(9.81)
            FLR = FLR + QF(HS2, US2, QL_v)
            US6 = fir / wp.float64(3.0)
            HS6 = US6 * US6 / wp.float64(9.81)
            FLR = FLR + QF(HS6, US6, QR_v)
            FLR = FLR + QF(QR_h, QR_u, QR_v)
        elif K2 == 2:  # QS<3>(+1), QS<6>(-1), QS<7>(+1)
            ua_ = (fil + fir) / wp.float64(2.0)
            fil = fil - ua_
            HA = fil * fil / (wp.float64(4.0) * wp.float64(9.81))
            FLR = FLR + QF(HA, ua_, QL_v)
            US6 = fir / wp.float64(3.0)
            HS6 = US6 * US6 / wp.float64(9.81)
            FLR = FLR + QF(HS6, US6, QR_v)
            FLR = FLR + QF(QR_h, QR_u, QR_v)
        elif K2 == 3:  # QS<5>(+1), QS<6>(-1), QS<7>(+1)
            ua_ = (fil + fir) / wp.float64(2.0)
            fir = fir - ua_
            HA = fir * fir / (wp.float64(4.0) * wp.float64(9.81))
            FLR = FLR + QF(HA, ua_, QR_v)
            US6b = fir / wp.float64(3.0)
            HS6b = US6b * US6b / wp.float64(9.81)
            FLR = FLR + QF(HS6b, US6b, QR_v)
            FLR = FLR + QF(QR_h, QR_u, QR_v)
        else:  # K2 == 4, QS<7>(+1)
            FLR = FLR + QF(QR_h, QR_u, QR_v)
    else:  # K1 == 4
        if K2 == 1:  # QS<1>(+1), QS<6>(-1), QS<7>(+1)
            FLR = FLR + QF(QL_h, QL_u, QL_v)
            US6 = fir / wp.float64(3.0)
            HS6 = US6 * US6 / wp.float64(9.81)
            FLR = FLR + QF(HS6, US6, QR_v)
            FLR = FLR + QF(QR_h, QR_u, QR_v)
        elif K2 == 2:  # QS<1>(+1), QS<2>(-1), QS<3>(+1), QS<6>(-1), QS<7>(+1)
            FLR = FLR + QF(QL_h, QL_u, QL_v)
            US2 = fil / wp.float64(3.0)
            HS2 = US2 * US2 / wp.float64(9.81)
            FLR = FLR + QF(HS2, US2, QL_v)
            ua_ = (fil + fir) / wp.float64(2.0)
            fil = fil - ua_
            HA = fil * fil / (wp.float64(4.0) * wp.float64(9.81))
            FLR = FLR + QF(HA, ua_, QL_v)
            US6 = fir / wp.float64(3.0)
            HS6 = US6 * US6 / wp.float64(9.81)
            FLR = FLR + QF(HS6, US6, QR_v)
            FLR = FLR + QF(QR_h, QR_u, QR_v)
        elif K2 == 3:  # QS<1>(+1), QS<2>(-1), QS<5>(+1), QS<6>(-1), QS<7>(+1)
            FLR = FLR + QF(QL_h, QL_u, QL_v)
            US2 = fil / wp.float64(3.0)
            HS2 = US2 * US2 / wp.float64(9.81)
            FLR = FLR + QF(HS2, US2, QL_v)
            ua_ = (fil + fir) / wp.float64(2.0)
            fir = fir - ua_
            HA = fir * fir / (wp.float64(4.0) * wp.float64(9.81))
            FLR = FLR + QF(HA, ua_, QR_v)
            US6 = fir / wp.float64(3.0)
            HS6 = US6 * US6 / wp.float64(9.81)
            FLR = FLR + QF(HS6, US6, QR_v)
            FLR = FLR + QF(QR_h, QR_u, QR_v)
        else:  # K2 == 4, QS<1>(+1), QS<2>(-1), QS<7>(+1)
            FLR = FLR + QF(QL_h, QL_u, QL_v)
            US2 = fil / wp.float64(3.0)
            HS2 = US2 * US2 / wp.float64(9.81)
            FLR = FLR + QF(HS2, US2, QL_v)
            FLR = FLR + QF(QR_h, QR_u, QR_v)

    return FLR


# ---------------------------------------------------------------------------
# Main shallow water kernel
# ---------------------------------------------------------------------------
@wp.kernel
def shallow_water_step(
    CEL: int,
    DT: wp.float64,
    NAC: wp.array(dtype=int, ndim=2),
    KLAS: wp.array(dtype=int, ndim=2),
    SIDE: wp.array(dtype=wp.float64, ndim=2),
    COSF: wp.array(dtype=wp.float64, ndim=2),
    SINF: wp.array(dtype=wp.float64, ndim=2),
    SLCOS: wp.array(dtype=wp.float64, ndim=2),
    SLSIN: wp.array(dtype=wp.float64, ndim=2),
    AREA: wp.array(dtype=wp.float64),
    ZBC: wp.array(dtype=wp.float64),
    FNC: wp.array(dtype=wp.float64),
    H_pre: wp.array(dtype=wp.float64),
    U_pre: wp.array(dtype=wp.float64),
    V_pre: wp.array(dtype=wp.float64),
    Z_pre: wp.array(dtype=wp.float64),
    H_res: wp.array(dtype=wp.float64),
    U_res: wp.array(dtype=wp.float64),
    V_res: wp.array(dtype=wp.float64),
    Z_res: wp.array(dtype=wp.float64),
    W_res: wp.array(dtype=wp.float64),
):
    tid = wp.tid()
    pos = tid + 1  # 1-indexed
    if pos > CEL:
        return

    H1 = H_pre[pos]
    U1 = U_pre[pos]
    V1 = V_pre[pos]
    BI = ZBC[pos]

    HI = wp.max(H1, wp.float64(0.001))
    UI = U1
    VI = V1
    if HI <= wp.float64(0.01):
        UI = safe_copysign(wp.float64(0.001), UI)
        VI = safe_copysign(wp.float64(0.001), VI)
    ZI = wp.max(Z_pre[pos], ZBC[pos])

    WH = wp.float64(0.0)
    WU = wp.float64(0.0)
    WV = wp.float64(0.0)

    for j in range(1, 5):
        NC = NAC[j, pos]
        KP = KLAS[j, pos]
        COSJ = COSF[j, pos]
        SINJ = SINF[j, pos]

        QL_h = HI
        QL_u = UI * COSJ + VI * SINJ
        QL_v = VI * COSJ - UI * SINJ
        CL_v = wp.sqrt(wp.float64(9.81) * HI)
        FIL_v = QL_u + wp.float64(2.0) * CL_v

        HC = wp.float64(0.0)
        BC = wp.float64(0.0)
        ZC = wp.float64(0.0)
        UC = wp.float64(0.0)
        VC = wp.float64(0.0)
        if NC != 0:
            HC = wp.max(H_pre[NC], wp.float64(0.001))
            BC = ZBC[NC]
            ZC = wp.max(ZBC[NC], Z_pre[NC])
            UC = U_pre[NC]
            VC = V_pre[NC]

        flux0 = wp.float64(0.0)
        flux1 = wp.float64(0.0)
        flux2 = wp.float64(0.0)
        flux3 = wp.float64(0.0)

        if KP == 4 or (KP != 0):
            # Wall / other boundary
            flux3 = wp.float64(4.905) * H1 * H1
        elif HI <= wp.float64(0.001) and HC <= wp.float64(0.001):
            pass
        elif ZI <= BC:
            flux0 = wp.float64(-0.3) * HC * wp.sqrt(HC)
            flux1 = HI * QL_u * wp.abs(QL_u)
            flux3 = wp.float64(4.905) * HI * HI
        elif ZC <= BI:
            flux0 = wp.float64(0.3) * HI * wp.sqrt(HI)
            flux1 = HI * wp.abs(QL_u) * QL_u
            flux2 = HI * wp.abs(QL_u) * QL_v
        elif HI <= wp.float64(0.01):
            if ZC > ZI:
                DH = wp.max(ZC - ZBC[pos], wp.float64(0.001))
                UN = wp.float64(-0.3) * wp.sqrt(DH)
                flux0 = DH * UN
                flux1 = flux0 * UN
                flux2 = flux0 * (VC * COSJ - UC * SINJ)
                flux3 = wp.float64(4.905) * HI * HI
            else:
                flux0 = wp.float64(0.3) * HI * wp.sqrt(HI)
                flux3 = wp.float64(4.905) * HI * HI
        elif HC <= wp.float64(0.01):
            if ZI > ZC:
                DH = wp.max(ZI - BC, wp.float64(0.001))
                UN = wp.float64(0.3) * wp.sqrt(DH)
                HC1 = ZC - ZBC[pos]
                flux0 = DH * UN
                flux1 = flux0 * UN
                flux2 = flux0 * QL_v
                flux3 = wp.float64(4.905) * HC1 * HC1
            else:
                flux0 = wp.float64(-0.3) * HC * wp.sqrt(HC)
                flux1 = HI * QL_u * QL_u
                flux3 = wp.float64(4.905) * HI * HI
        else:
            # Both wet — Osher Riemann solver
            if pos < NC:
                QR_h = wp.max(ZC - ZBC[pos], wp.float64(0.001))
                UR = UC * COSJ + VC * SINJ
                ratio = wp.min(HC / QR_h, wp.float64(1.5))
                QR_u = UR * ratio
                if HC <= wp.float64(0.01) or QR_h <= wp.float64(0.01):
                    QR_u = safe_copysign(wp.float64(0.001), UR)
                QR_v_ = VC * COSJ - UC * SINJ
                FLR_OS = osher_solver(QL_h, QL_u, QL_v, QR_h, QR_u, QR_v_, FIL_v, H_pre[pos])
                flux0 = FLR_OS[0]
                flux1 = FLR_OS[1] + (wp.float64(1.0) - ratio) * HC * UR * UR / wp.float64(2.0)
                flux2 = FLR_OS[2]
                flux3 = FLR_OS[3]
            else:
                COSJ1 = -COSJ
                SINJ1 = -SINJ
                QL1_h = H_pre[NC]
                QL1_u = U_pre[NC] * COSJ1 + V_pre[NC] * SINJ1
                QL1_v = V_pre[NC] * COSJ1 - U_pre[NC] * SINJ1
                CL1 = wp.sqrt(wp.float64(9.81) * H_pre[NC])
                FIL1 = QL1_u + wp.float64(2.0) * CL1
                HC2 = wp.max(HI, wp.float64(0.001))
                ZC1 = wp.max(ZBC[pos], ZI)
                QR1_h = wp.max(ZC1 - ZBC[NC], wp.float64(0.001))
                UR1 = UI * COSJ1 + VI * SINJ1
                ratio1 = wp.min(HC2 / QR1_h, wp.float64(1.5))
                QR1_u = UR1 * ratio1
                if HC2 <= wp.float64(0.01) or QR1_h <= wp.float64(0.01):
                    QR1_u = safe_copysign(wp.float64(0.001), UR1)
                QR1_v_ = VI * COSJ1 - UI * SINJ1
                FLR1 = osher_solver(QL1_h, QL1_u, QL1_v, QR1_h, QR1_u, QR1_v_, FIL1, H_pre[NC])
                flux0 = -FLR1[0]
                flux1 = FLR1[1] + (wp.float64(1.0) - ratio1) * HC2 * UR1 * UR1 / wp.float64(2.0)
                flux2 = FLR1[2]
                ZA = wp.sqrt(FLR1[3] / wp.float64(4.905)) + BC
                HC3 = wp.max(ZA - ZBC[pos], wp.float64(0.0))
                flux3 = wp.float64(4.905) * HC3 * HC3

        # Accumulate fluxes
        SL = SIDE[j, pos]
        SLCA = SLCOS[j, pos]
        SLSA = SLSIN[j, pos]
        FLR_1 = flux1 + flux3
        FLR_2 = flux2
        WH = WH + SL * flux0
        WU = WU + SLCA * FLR_1 - SLSA * FLR_2
        WV = WV + SLSA * FLR_1 + SLCA * FLR_2

    # State update with Manning friction
    DTA = DT / AREA[pos]
    WDTA = DTA
    H2 = wp.max(H1 - WDTA * WH, wp.float64(0.001))
    Z2 = H2 + BI

    U2 = wp.float64(0.0)
    V2 = wp.float64(0.0)
    if H2 > wp.float64(0.001):
        if H2 <= wp.float64(0.01):
            U2 = safe_copysign(wp.min(wp.float64(0.001), wp.abs(U1)), U1)
            V2 = safe_copysign(wp.min(wp.float64(0.001), wp.abs(V1)), V1)
        else:
            QX1 = H1 * U1
            QY1 = H1 * V1
            DTAU = WDTA * WU
            DTAV = WDTA * WV
            WSF = FNC[pos] * wp.sqrt(U1 * U1 + V1 * V1) / wp.pow(H1, wp.float64(0.33333))
            U2 = (QX1 - DTAU - DT * WSF * U1) / H2
            V2 = (QY1 - DTAV - DT * WSF * V1) / H2
            if H2 > wp.float64(0.01):
                U2 = safe_copysign(wp.min(wp.abs(U2), wp.float64(15.0)), U2)
                V2 = safe_copysign(wp.min(wp.abs(V2), wp.float64(15.0)), V2)

    H_res[pos] = H2
    U_res[pos] = U2
    V_res[pos] = V2
    Z_res[pos] = Z2
    W_res[pos] = wp.sqrt(U2 * U2 + V2 * V2)


@wp.kernel
def transfer_data(
    CEL: int,
    H_pre: wp.array(dtype=wp.float64),
    U_pre: wp.array(dtype=wp.float64),
    V_pre: wp.array(dtype=wp.float64),
    Z_pre: wp.array(dtype=wp.float64),
    W_pre: wp.array(dtype=wp.float64),
    H_res: wp.array(dtype=wp.float64),
    U_res: wp.array(dtype=wp.float64),
    V_res: wp.array(dtype=wp.float64),
    Z_res: wp.array(dtype=wp.float64),
    W_res: wp.array(dtype=wp.float64),
):
    tid = wp.tid()
    pos = tid + 1
    if pos > CEL:
        return
    H_pre[pos] = H_res[pos]
    U_pre[pos] = U_res[pos]
    V_pre[pos] = V_res[pos]
    Z_pre[pos] = Z_res[pos]
    W_pre[pos] = W_res[pos]


# ---------------------------------------------------------------------------
# Benchmark interface
# ---------------------------------------------------------------------------
def run(N, steps=1, backend="cuda"):
    CEL = N * N
    dx = 1.0
    DT = np.float64(0.5 * 1.0 / (np.sqrt(G * 2.0) + 1e-6))

    # Build mesh on host
    nac_np = np.zeros((5, CEL + 1), dtype=np.int32)
    klas_np = np.zeros((5, CEL + 1), dtype=np.int32)
    side_np = np.zeros((5, CEL + 1), dtype=np.float64)
    cosf_np = np.zeros((5, CEL + 1), dtype=np.float64)
    sinf_np = np.zeros((5, CEL + 1), dtype=np.float64)
    area_np = np.zeros(CEL + 1, dtype=np.float64)
    zbc_np = np.zeros(CEL + 1, dtype=np.float64)
    fnc_np = np.full(CEL + 1, G * MANNING_N * MANNING_N, dtype=np.float64)

    edge_cos = [0.0, 0.0, 1.0, 0.0, -1.0]
    edge_sin = [0.0, -1.0, 0.0, 1.0, 0.0]

    h_np = np.full(CEL + 1, HM1_C, dtype=np.float64)
    z_np = np.zeros(CEL + 1, dtype=np.float64)

    for i in range(N):
        for jj in range(N):
            pos = i * N + jj + 1
            area_np[pos] = dx * dx
            for e in range(1, 5):
                side_np[e][pos] = dx
                cosf_np[e][pos] = edge_cos[e]
                sinf_np[e][pos] = edge_sin[e]
            if i > 0:
                nac_np[1][pos] = (i - 1) * N + jj + 1
            else:
                klas_np[1][pos] = 4
            if jj < N - 1:
                nac_np[2][pos] = i * N + (jj + 1) + 1
            else:
                klas_np[2][pos] = 4
            if i < N - 1:
                nac_np[3][pos] = (i + 1) * N + jj + 1
            else:
                klas_np[3][pos] = 4
            if jj > 0:
                nac_np[4][pos] = i * N + (jj - 1) + 1
            else:
                klas_np[4][pos] = 4

            h_np[pos] = 2.0 if jj < N // 2 else 0.5
            z_np[pos] = h_np[pos]

    slcos_np = side_np * cosf_np
    slsin_np = side_np * sinf_np

    # Upload to device
    NAC = wp.array(nac_np, dtype=int, device=backend)
    KLAS_d = wp.array(klas_np, dtype=int, device=backend)
    SIDE_d = wp.array(side_np, dtype=wp.float64, device=backend)
    COSF_d = wp.array(cosf_np, dtype=wp.float64, device=backend)
    SINF_d = wp.array(sinf_np, dtype=wp.float64, device=backend)
    SLCOS_d = wp.array(slcos_np, dtype=wp.float64, device=backend)
    SLSIN_d = wp.array(slsin_np, dtype=wp.float64, device=backend)
    AREA_d = wp.array(area_np, dtype=wp.float64, device=backend)
    ZBC_d = wp.array(zbc_np, dtype=wp.float64, device=backend)
    FNC_d = wp.array(fnc_np, dtype=wp.float64, device=backend)

    H_pre = wp.array(h_np, dtype=wp.float64, device=backend)
    U_pre = wp.array(np.zeros(CEL + 1, dtype=np.float64), dtype=wp.float64, device=backend)
    V_pre = wp.array(np.zeros(CEL + 1, dtype=np.float64), dtype=wp.float64, device=backend)
    Z_pre = wp.array(z_np, dtype=wp.float64, device=backend)
    W_pre = wp.array(np.zeros(CEL + 1, dtype=np.float64), dtype=wp.float64, device=backend)

    H_res = wp.zeros(CEL + 1, dtype=wp.float64, device=backend)
    U_res = wp.zeros(CEL + 1, dtype=wp.float64, device=backend)
    V_res = wp.zeros(CEL + 1, dtype=wp.float64, device=backend)
    Z_res = wp.zeros(CEL + 1, dtype=wp.float64, device=backend)
    W_res = wp.zeros(CEL + 1, dtype=wp.float64, device=backend)

    def step():
        for _ in range(steps):
            wp.launch(shallow_water_step, dim=CEL,
                      inputs=[CEL, DT,
                              NAC, KLAS_d, SIDE_d, COSF_d, SINF_d, SLCOS_d, SLSIN_d,
                              AREA_d, ZBC_d, FNC_d,
                              H_pre, U_pre, V_pre, Z_pre,
                              H_res, U_res, V_res, Z_res, W_res],
                      device=backend)
            wp.launch(transfer_data, dim=CEL,
                      inputs=[CEL, H_pre, U_pre, V_pre, Z_pre, W_pre,
                              H_res, U_res, V_res, Z_res, W_res],
                      device=backend)

    def sync():
        wp.synchronize_device(backend)

    return step, sync, H_pre
