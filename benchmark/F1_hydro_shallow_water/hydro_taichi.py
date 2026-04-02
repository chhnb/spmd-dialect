"""2D Shallow Water Equations (Osher Riemann solver) — Taichi.

Dam-break on unstructured quad mesh. Port of hydro-cal calculate_gpu.cu.
"""
import taichi as ti
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G = 9.81
HALF_G = 4.905
HM1 = 0.001
HM2 = 0.01
VMIN = 0.001
C0 = 1.0
C1 = 0.3
MANNING_N = 0.03


def run(N, steps=1, backend="cuda"):
    ti.init(arch=ti.cuda if backend == "cuda" else ti.cpu, default_fp=ti.f64)
    CEL = N * N
    dx = 1.0
    DT = 0.5 * dx / (ti.sqrt(G * 2.0) + 1e-6)

    # --- Fields ---
    NAC = ti.field(dtype=ti.i32, shape=(5, CEL + 1))
    KLAS = ti.field(dtype=ti.i32, shape=(5, CEL + 1))
    SIDE = ti.field(dtype=ti.f64, shape=(5, CEL + 1))
    COSF = ti.field(dtype=ti.f64, shape=(5, CEL + 1))
    SINF = ti.field(dtype=ti.f64, shape=(5, CEL + 1))
    SLCOS = ti.field(dtype=ti.f64, shape=(5, CEL + 1))
    SLSIN = ti.field(dtype=ti.f64, shape=(5, CEL + 1))
    AREA = ti.field(dtype=ti.f64, shape=(CEL + 1,))
    ZBC = ti.field(dtype=ti.f64, shape=(CEL + 1,))
    FNC = ti.field(dtype=ti.f64, shape=(CEL + 1,))

    H_pre = ti.field(dtype=ti.f64, shape=(CEL + 1,))
    U_pre = ti.field(dtype=ti.f64, shape=(CEL + 1,))
    V_pre = ti.field(dtype=ti.f64, shape=(CEL + 1,))
    Z_pre = ti.field(dtype=ti.f64, shape=(CEL + 1,))
    W_pre = ti.field(dtype=ti.f64, shape=(CEL + 1,))

    H_res = ti.field(dtype=ti.f64, shape=(CEL + 1,))
    U_res = ti.field(dtype=ti.f64, shape=(CEL + 1,))
    V_res = ti.field(dtype=ti.f64, shape=(CEL + 1,))
    Z_res = ti.field(dtype=ti.f64, shape=(CEL + 1,))
    W_res = ti.field(dtype=ti.f64, shape=(CEL + 1,))

    # --- Mesh init on host, then copy to fields ---
    nac_np = np.zeros((5, CEL + 1), dtype=np.int32)
    klas_np = np.zeros((5, CEL + 1), dtype=np.int32)
    side_np = np.zeros((5, CEL + 1), dtype=np.float64)
    cosf_np = np.zeros((5, CEL + 1), dtype=np.float64)
    sinf_np = np.zeros((5, CEL + 1), dtype=np.float64)
    area_np = np.zeros(CEL + 1, dtype=np.float64)
    fnc_np = np.full(CEL + 1, G * MANNING_N * MANNING_N, dtype=np.float64)

    edge_cos = [0.0, 0.0, 1.0, 0.0, -1.0]
    edge_sin = [0.0, -1.0, 0.0, 1.0, 0.0]

    h_np = np.full(CEL + 1, HM1, dtype=np.float64)
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

    NAC.from_numpy(nac_np)
    KLAS.from_numpy(klas_np)
    SIDE.from_numpy(side_np)
    COSF.from_numpy(cosf_np)
    SINF.from_numpy(sinf_np)
    slcos_np = side_np * cosf_np
    slsin_np = side_np * sinf_np
    SLCOS.from_numpy(slcos_np)
    SLSIN.from_numpy(slsin_np)
    AREA.from_numpy(area_np)
    ZBC.from_numpy(np.zeros(CEL + 1, dtype=np.float64))
    FNC.from_numpy(fnc_np)
    H_pre.from_numpy(h_np)
    U_pre.from_numpy(np.zeros(CEL + 1, dtype=np.float64))
    V_pre.from_numpy(np.zeros(CEL + 1, dtype=np.float64))
    Z_pre.from_numpy(z_np)
    W_pre.from_numpy(np.zeros(CEL + 1, dtype=np.float64))

    # ------------------------------------------------------------------
    # Taichi functions
    # ------------------------------------------------------------------
    @ti.func
    def QF(h: ti.f64, u: ti.f64, v: ti.f64) -> ti.types.vector(4, ti.f64):
        f0 = h * u
        return ti.Vector([f0, f0 * u, f0 * v, HALF_G * h * h])

    @ti.func
    def osher(QL: ti.types.vector(3, ti.f64),
              QR: ti.types.vector(3, ti.f64),
              FIL_in: ti.f64, H_pos: ti.f64) -> ti.types.vector(4, ti.f64):
        CR = ti.sqrt(G * QR[0])
        FIR_v = QR[1] - 2.0 * CR
        UA = (FIL_in + FIR_v) / 2.0
        CA = ti.abs((FIL_in - FIR_v) / 4.0)
        CL_v = ti.sqrt(G * H_pos)

        FLR = ti.Vector([0.0, 0.0, 0.0, 0.0])

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
        if QL[1] < CL_v and QR[1] >= -CR:
            K1 = 1
        elif QL[1] >= CL_v and QR[1] >= -CR:
            K1 = 2
        elif QL[1] < CL_v and QR[1] < -CR:
            K1 = 3
        else:
            K1 = 4

        fil = FIL_in
        fir = FIR_v

        # Inline QS calls — encode the dispatch table
        # T=1: left state
        # T=2: sonic on FIL
        # T=3: intermediate (updates fil)
        # T=5: intermediate (updates fir)
        # T=6: sonic on FIR
        # T=7: right state

        # We encode each (K1,K2) case explicitly.
        # Each case is a sequence of (T, sign) pairs.
        if K1 == 1:
            if K2 == 1:
                US = fil / 3.0; HS = US * US / G
                FLR += QF(HS, US, QL[2]) * 1.0
            elif K2 == 2:
                ua_ = (fil + fir) / 2.0; fil = fil - ua_; HA = fil * fil / (4.0 * G)
                FLR += QF(HA, ua_, QL[2]) * 1.0
            elif K2 == 3:
                ua_ = (fil + fir) / 2.0; fir = fir - ua_; HA = fir * fir / (4.0 * G)
                FLR += QF(HA, ua_, QR[2]) * 1.0
            else:  # K2 == 4
                US = fir / 3.0; HS = US * US / G
                FLR += QF(HS, US, QR[2]) * 1.0
        elif K1 == 2:
            if K2 == 1:
                FLR += QF(QL[0], QL[1], QL[2]) * 1.0
            elif K2 == 2:
                FLR += QF(QL[0], QL[1], QL[2]) * 1.0
                US2 = fil / 3.0; HS2 = US2 * US2 / G
                FLR += QF(HS2, US2, QL[2]) * (-1.0)
                ua_ = (fil + fir) / 2.0; fil = fil - ua_; HA = fil * fil / (4.0 * G)
                FLR += QF(HA, ua_, QL[2]) * 1.0
            elif K2 == 3:
                FLR += QF(QL[0], QL[1], QL[2]) * 1.0
                US2 = fil / 3.0; HS2 = US2 * US2 / G
                FLR += QF(HS2, US2, QL[2]) * (-1.0)
                ua_ = (fil + fir) / 2.0; fir = fir - ua_; HA = fir * fir / (4.0 * G)
                FLR += QF(HA, ua_, QR[2]) * 1.0
            else:  # K2 == 4
                FLR += QF(QL[0], QL[1], QL[2]) * 1.0
                US2 = fil / 3.0; HS2 = US2 * US2 / G
                FLR += QF(HS2, US2, QL[2]) * (-1.0)
                US6 = fir / 3.0; HS6 = US6 * US6 / G
                FLR += QF(HS6, US6, QR[2]) * 1.0
        elif K1 == 3:
            if K2 == 1:
                US2 = fil / 3.0; HS2 = US2 * US2 / G
                FLR += QF(HS2, US2, QL[2]) * 1.0
                US6 = fir / 3.0; HS6 = US6 * US6 / G
                FLR += QF(HS6, US6, QR[2]) * (-1.0)
                FLR += QF(QR[0], QR[1], QR[2]) * 1.0
            elif K2 == 2:
                ua_ = (fil + fir) / 2.0; fil = fil - ua_; HA = fil * fil / (4.0 * G)
                FLR += QF(HA, ua_, QL[2]) * 1.0
                US6 = fir / 3.0; HS6 = US6 * US6 / G
                FLR += QF(HS6, US6, QR[2]) * (-1.0)
                FLR += QF(QR[0], QR[1], QR[2]) * 1.0
            elif K2 == 3:
                ua_ = (fil + fir) / 2.0; fir = fir - ua_; HA = fir * fir / (4.0 * G)
                FLR += QF(HA, ua_, QR[2]) * 1.0
                US6b = fir / 3.0; HS6b = US6b * US6b / G
                FLR += QF(HS6b, US6b, QR[2]) * (-1.0)
                FLR += QF(QR[0], QR[1], QR[2]) * 1.0
            else:  # K2 == 4
                FLR += QF(QR[0], QR[1], QR[2]) * 1.0
        else:  # K1 == 4
            if K2 == 1:
                FLR += QF(QL[0], QL[1], QL[2]) * 1.0
                US6 = fir / 3.0; HS6 = US6 * US6 / G
                FLR += QF(HS6, US6, QR[2]) * (-1.0)
                FLR += QF(QR[0], QR[1], QR[2]) * 1.0
            elif K2 == 2:
                FLR += QF(QL[0], QL[1], QL[2]) * 1.0
                US2 = fil / 3.0; HS2 = US2 * US2 / G
                FLR += QF(HS2, US2, QL[2]) * (-1.0)
                ua_ = (fil + fir) / 2.0; fil = fil - ua_; HA = fil * fil / (4.0 * G)
                FLR += QF(HA, ua_, QL[2]) * 1.0
                US6 = fir / 3.0; HS6 = US6 * US6 / G
                FLR += QF(HS6, US6, QR[2]) * (-1.0)
                FLR += QF(QR[0], QR[1], QR[2]) * 1.0
            elif K2 == 3:
                FLR += QF(QL[0], QL[1], QL[2]) * 1.0
                US2 = fil / 3.0; HS2 = US2 * US2 / G
                FLR += QF(HS2, US2, QL[2]) * (-1.0)
                ua_ = (fil + fir) / 2.0; fir = fir - ua_; HA = fir * fir / (4.0 * G)
                FLR += QF(HA, ua_, QR[2]) * 1.0
                US6 = fir / 3.0; HS6 = US6 * US6 / G
                FLR += QF(HS6, US6, QR[2]) * (-1.0)
                FLR += QF(QR[0], QR[1], QR[2]) * 1.0
            else:  # K2 == 4
                FLR += QF(QL[0], QL[1], QL[2]) * 1.0
                US2 = fil / 3.0; HS2 = US2 * US2 / G
                FLR += QF(HS2, US2, QL[2]) * (-1.0)
                FLR += QF(QR[0], QR[1], QR[2]) * 1.0

        return FLR

    # ------------------------------------------------------------------
    @ti.kernel
    def shallow_water_step():
        for pos in range(1, CEL + 1):
            H1 = H_pre[pos]
            U1 = U_pre[pos]
            V1 = V_pre[pos]
            BI = ZBC[pos]

            HI = ti.max(H1, HM1)
            UI = U1
            VI = V1
            if HI <= HM2:
                UI = ti.select(UI >= 0.0, VMIN, -VMIN)
                VI = ti.select(VI >= 0.0, VMIN, -VMIN)
            ZI = ti.max(Z_pre[pos], ZBC[pos])

            WH = 0.0
            WU = 0.0
            WV = 0.0

            for j in ti.static(range(1, 5)):
                NC = NAC[j, pos]
                KP = KLAS[j, pos]
                COSJ = COSF[j, pos]
                SINJ = SINF[j, pos]

                QL = ti.Vector([HI, UI * COSJ + VI * SINJ, VI * COSJ - UI * SINJ])
                CL_v = ti.sqrt(G * HI)
                FIL_v = QL[1] + 2.0 * CL_v

                HC = 0.0; BC = 0.0; ZC = 0.0; UC = 0.0; VC = 0.0
                if NC != 0:
                    HC = ti.max(H_pre[NC], HM1)
                    BC = ZBC[NC]
                    ZC = ti.max(ZBC[NC], Z_pre[NC])
                    UC = U_pre[NC]
                    VC = V_pre[NC]

                flux = ti.Vector([0.0, 0.0, 0.0, 0.0])

                if KP == 4:
                    flux[3] = HALF_G * H1 * H1
                elif KP != 0:
                    flux[3] = HALF_G * H1 * H1
                elif HI <= HM1 and HC <= HM1:
                    pass
                elif ZI <= BC:
                    flux[0] = -C1 * ti.pow(HC, 1.5)
                    flux[1] = HI * QL[1] * ti.abs(QL[1])
                    flux[3] = HALF_G * HI * HI
                elif ZC <= BI:
                    flux[0] = C1 * ti.pow(HI, 1.5)
                    flux[1] = HI * ti.abs(QL[1]) * QL[1]
                    flux[2] = HI * ti.abs(QL[1]) * QL[2]
                elif HI <= HM2:
                    if ZC > ZI:
                        DH = ti.max(ZC - ZBC[pos], HM1)
                        UN = -C1 * ti.sqrt(DH)
                        flux[0] = DH * UN
                        flux[1] = flux[0] * UN
                        flux[2] = flux[0] * (VC * COSJ - UC * SINJ)
                        flux[3] = HALF_G * HI * HI
                    else:
                        flux[0] = C1 * ti.pow(HI, 1.5)
                        flux[3] = HALF_G * HI * HI
                elif HC <= HM2:
                    if ZI > ZC:
                        DH = ti.max(ZI - BC, HM1)
                        UN = C1 * ti.sqrt(DH)
                        HC1 = ZC - ZBC[pos]
                        flux[0] = DH * UN
                        flux[1] = flux[0] * UN
                        flux[2] = flux[0] * QL[2]
                        flux[3] = HALF_G * HC1 * HC1
                    else:
                        flux[0] = -C1 * ti.pow(HC, 1.5)
                        flux[1] = HI * QL[1] * QL[1]
                        flux[3] = HALF_G * HI * HI
                else:
                    # Both wet — Osher Riemann solver
                    if pos < NC:
                        QR_h = ti.max(ZC - ZBC[pos], HM1)
                        UR = UC * COSJ + VC * SINJ
                        ratio = ti.min(HC / QR_h, 1.5)
                        QR_u = UR * ratio
                        if HC <= HM2 or QR_h <= HM2:
                            QR_u = ti.select(UR >= 0.0, VMIN, -VMIN)
                        QR_v_ = VC * COSJ - UC * SINJ
                        QR_vec = ti.Vector([QR_h, QR_u, QR_v_])
                        FLR_OS = osher(QL, QR_vec, FIL_v, H_pre[pos])
                        flux[0] = FLR_OS[0]
                        flux[1] = FLR_OS[1] + (1.0 - ratio) * HC * UR * UR / 2.0
                        flux[2] = FLR_OS[2]
                        flux[3] = FLR_OS[3]
                    else:
                        COSJ1 = -COSJ
                        SINJ1 = -SINJ
                        QL1 = ti.Vector([
                            H_pre[NC],
                            U_pre[NC] * COSJ1 + V_pre[NC] * SINJ1,
                            V_pre[NC] * COSJ1 - U_pre[NC] * SINJ1,
                        ])
                        CL1 = ti.sqrt(G * H_pre[NC])
                        FIL1 = QL1[1] + 2.0 * CL1
                        HC2 = ti.max(HI, HM1)
                        ZC1 = ti.max(ZBC[pos], ZI)
                        QR1_h = ti.max(ZC1 - ZBC[NC], HM1)
                        UR1 = UI * COSJ1 + VI * SINJ1
                        ratio1 = ti.min(HC2 / QR1_h, 1.5)
                        QR1_u = UR1 * ratio1
                        if HC2 <= HM2 or QR1_h <= HM2:
                            QR1_u = ti.select(UR1 >= 0.0, VMIN, -VMIN)
                        QR1_v_ = VI * COSJ1 - UI * SINJ1
                        QR1_vec = ti.Vector([QR1_h, QR1_u, QR1_v_])
                        FLR1 = osher(QL1, QR1_vec, FIL1, H_pre[NC])
                        flux[0] = -FLR1[0]
                        flux[1] = FLR1[1] + (1.0 - ratio1) * HC2 * UR1 * UR1 / 2.0
                        flux[2] = FLR1[2]
                        ZA = ti.sqrt(FLR1[3] / HALF_G) + BC
                        HC3 = ti.max(ZA - ZBC[pos], 0.0)
                        flux[3] = HALF_G * HC3 * HC3

                # Accumulate fluxes
                SL = SIDE[j, pos]
                SLCA = SLCOS[j, pos]
                SLSA = SLSIN[j, pos]
                FLR_1 = flux[1] + flux[3]
                FLR_2 = flux[2]
                WH += SL * flux[0]
                WU += SLCA * FLR_1 - SLSA * FLR_2
                WV += SLSA * FLR_1 + SLCA * FLR_2

            # State update with Manning friction
            DTA = DT / AREA[pos]
            WDTA = DTA
            H2 = ti.max(H1 - WDTA * WH, HM1)
            Z2 = H2 + BI

            U2 = 0.0
            V2 = 0.0
            if H2 > HM1:
                if H2 <= HM2:
                    U2 = ti.select(U1 >= 0.0, ti.min(VMIN, ti.abs(U1)), -ti.min(VMIN, ti.abs(U1)))
                    V2 = ti.select(V1 >= 0.0, ti.min(VMIN, ti.abs(V1)), -ti.min(VMIN, ti.abs(V1)))
                else:
                    QX1 = H1 * U1
                    QY1 = H1 * V1
                    DTAU = WDTA * WU
                    DTAV = WDTA * WV
                    WSF = FNC[pos] * ti.sqrt(U1 * U1 + V1 * V1) / ti.pow(H1, 0.33333)
                    U2 = (QX1 - DTAU - DT * WSF * U1) / H2
                    V2 = (QY1 - DTAV - DT * WSF * V1) / H2
                    if H2 > HM2:
                        U2 = ti.select(U2 >= 0.0, ti.min(ti.abs(U2), 15.0), -ti.min(ti.abs(U2), 15.0))
                        V2 = ti.select(V2 >= 0.0, ti.min(ti.abs(V2), 15.0), -ti.min(ti.abs(V2), 15.0))

            H_res[pos] = H2
            U_res[pos] = U2
            V_res[pos] = V2
            Z_res[pos] = Z2
            W_res[pos] = ti.sqrt(U2 * U2 + V2 * V2)

    @ti.kernel
    def transfer():
        for pos in range(1, CEL + 1):
            H_pre[pos] = H_res[pos]
            U_pre[pos] = U_res[pos]
            V_pre[pos] = V_res[pos]
            Z_pre[pos] = Z_res[pos]
            W_pre[pos] = W_res[pos]

    def step():
        for _ in range(steps):
            shallow_water_step()
            transfer()

    return step, ti.sync, H_pre
