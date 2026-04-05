"""F2: Refactored Hydro-Cal — Taichi (fp32, edge-parallel flux + cell-parallel update).

Two-kernel design matching the refactored CUDA kernel:
  1. CalculateFluxKernel: 1 thread per edge (4*CELL), computes flux per edge
  2. UpdateCellKernel:    1 thread per cell (CELL),   accumulates fluxes, updates state
"""
import os
import sys
import numpy as np
import taichi as ti

sys.path.insert(0, os.path.dirname(__file__))
from mesh_loader import load_mesh

# ---------------------------------------------------------------------------
# Constants (fp32)
# ---------------------------------------------------------------------------
G: ti.f32 = 9.81
HALF_G: ti.f32 = 4.905
C0: ti.f32 = 1.33
C1: ti.f32 = 1.7
VMIN: ti.f32 = 0.001
QLUA: ti.f32 = 0.0
BRDTH: ti.f32 = 100.0


def run(days=10, backend="cuda"):
    ti.init(arch=ti.cuda if backend == "cuda" else ti.cpu, default_fp=ti.f32)
    mesh = load_mesh()

    CELL = mesh["CELL"]
    NE = 4 * CELL
    HM1 = float(mesh["HM1"])
    HM2 = float(mesh["HM2"])
    DT = float(mesh["DT"])
    steps_per_day = mesh["steps_per_day"]
    total_steps = steps_per_day * days

    # --- Fields: edges [4*CELL] ---
    NAC   = ti.field(ti.i32, shape=NE)
    KLAS  = ti.field(ti.f32, shape=NE)
    SIDE  = ti.field(ti.f32, shape=NE)
    COSF  = ti.field(ti.f32, shape=NE)
    SINF  = ti.field(ti.f32, shape=NE)
    SLCOS = ti.field(ti.f32, shape=NE)
    SLSIN = ti.field(ti.f32, shape=NE)
    FLUX0 = ti.field(ti.f32, shape=NE)
    FLUX1 = ti.field(ti.f32, shape=NE)
    FLUX2 = ti.field(ti.f32, shape=NE)
    FLUX3 = ti.field(ti.f32, shape=NE)

    # --- Fields: cells [CELL] ---
    H    = ti.field(ti.f32, shape=CELL)
    U    = ti.field(ti.f32, shape=CELL)
    V    = ti.field(ti.f32, shape=CELL)
    Z    = ti.field(ti.f32, shape=CELL)
    W    = ti.field(ti.f32, shape=CELL)
    ZBC  = ti.field(ti.f32, shape=CELL)
    ZB1  = ti.field(ti.f32, shape=CELL)
    AREA = ti.field(ti.f32, shape=CELL)
    FNC  = ti.field(ti.f32, shape=CELL)

    # --- Load data into fields ---
    NAC.from_numpy(mesh["NAC"])
    KLAS.from_numpy(mesh["KLAS"])
    SIDE.from_numpy(mesh["SIDE"])
    COSF.from_numpy(mesh["COSF"])
    SINF.from_numpy(mesh["SINF"])
    SLCOS.from_numpy(mesh["SLCOS"])
    SLSIN.from_numpy(mesh["SLSIN"])
    H.from_numpy(mesh["H"])
    U.from_numpy(mesh["U"])
    V.from_numpy(mesh["V"])
    Z.from_numpy(mesh["Z"])
    W.from_numpy(mesh["W"])
    ZBC.from_numpy(mesh["ZBC"])
    ZB1.from_numpy(mesh["ZB1"])
    AREA.from_numpy(mesh["AREA"])
    FNC.from_numpy(mesh["FNC"])

    # ------------------------------------------------------------------
    # Taichi functions
    # ------------------------------------------------------------------
    @ti.func
    def QF(h: ti.f32, u: ti.f32, v: ti.f32) -> ti.types.vector(4, ti.f32):
        hu = h * u
        return ti.Vector([hu, hu * u, hu * v, HALF_G * h * h])

    @ti.func
    def osher(QL: ti.types.vector(3, ti.f32),
              QR: ti.types.vector(3, ti.f32),
              FIL_in: ti.f32, H_pos: ti.f32) -> ti.types.vector(4, ti.f32):
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

        if K1 == 1:
            if K2 == 1:
                US = fil / 3.0; HS = US * US / G
                FLR += QF(HS, US, QL[2])
            elif K2 == 2:
                ua_ = (fil + fir) / 2.0; fil = fil - ua_; HA = fil * fil / (4.0 * G)
                FLR += QF(HA, ua_, QL[2])
            elif K2 == 3:
                ua_ = (fil + fir) / 2.0; fir = fir - ua_; HA = fir * fir / (4.0 * G)
                FLR += QF(HA, ua_, QR[2])
            else:
                US = fir / 3.0; HS = US * US / G
                FLR += QF(HS, US, QR[2])
        elif K1 == 2:
            if K2 == 1:
                FLR += QF(QL[0], QL[1], QL[2])
            elif K2 == 2:
                FLR += QF(QL[0], QL[1], QL[2])
                US2 = fil / 3.0; HS2 = US2 * US2 / G
                FLR -= QF(HS2, US2, QL[2])
                ua_ = (fil + fir) / 2.0; fil = fil - ua_; HA = fil * fil / (4.0 * G)
                FLR += QF(HA, ua_, QL[2])
            elif K2 == 3:
                FLR += QF(QL[0], QL[1], QL[2])
                US2 = fil / 3.0; HS2 = US2 * US2 / G
                FLR -= QF(HS2, US2, QL[2])
                ua_ = (fil + fir) / 2.0; fir = fir - ua_; HA = fir * fir / (4.0 * G)
                FLR += QF(HA, ua_, QR[2])
            else:
                FLR += QF(QL[0], QL[1], QL[2])
                US2 = fil / 3.0; HS2 = US2 * US2 / G
                FLR -= QF(HS2, US2, QL[2])
                US6 = fir / 3.0; HS6 = US6 * US6 / G
                FLR += QF(HS6, US6, QR[2])
        elif K1 == 3:
            if K2 == 1:
                US2 = fil / 3.0; HS2 = US2 * US2 / G
                FLR += QF(HS2, US2, QL[2])
                US6 = fir / 3.0; HS6 = US6 * US6 / G
                FLR -= QF(HS6, US6, QR[2])
                FLR += QF(QR[0], QR[1], QR[2])
            elif K2 == 2:
                ua_ = (fil + fir) / 2.0; fil = fil - ua_; HA = fil * fil / (4.0 * G)
                FLR += QF(HA, ua_, QL[2])
                US6 = fir / 3.0; HS6 = US6 * US6 / G
                FLR -= QF(HS6, US6, QR[2])
                FLR += QF(QR[0], QR[1], QR[2])
            elif K2 == 3:
                ua_ = (fil + fir) / 2.0; fir = fir - ua_; HA = fir * fir / (4.0 * G)
                FLR += QF(HA, ua_, QR[2])
                US6b = fir / 3.0; HS6b = US6b * US6b / G
                FLR -= QF(HS6b, US6b, QR[2])
                FLR += QF(QR[0], QR[1], QR[2])
            else:
                FLR += QF(QR[0], QR[1], QR[2])
        else:  # K1 == 4
            if K2 == 1:
                FLR += QF(QL[0], QL[1], QL[2])
                US6 = fir / 3.0; HS6 = US6 * US6 / G
                FLR -= QF(HS6, US6, QR[2])
                FLR += QF(QR[0], QR[1], QR[2])
            elif K2 == 2:
                FLR += QF(QL[0], QL[1], QL[2])
                US2 = fil / 3.0; HS2 = US2 * US2 / G
                FLR -= QF(HS2, US2, QL[2])
                ua_ = (fil + fir) / 2.0; fil = fil - ua_; HA = fil * fil / (4.0 * G)
                FLR += QF(HA, ua_, QL[2])
                US6 = fir / 3.0; HS6 = US6 * US6 / G
                FLR -= QF(HS6, US6, QR[2])
                FLR += QF(QR[0], QR[1], QR[2])
            elif K2 == 3:
                FLR += QF(QL[0], QL[1], QL[2])
                US2 = fil / 3.0; HS2 = US2 * US2 / G
                FLR -= QF(HS2, US2, QL[2])
                ua_ = (fil + fir) / 2.0; fir = fir - ua_; HA = fir * fir / (4.0 * G)
                FLR += QF(HA, ua_, QR[2])
                US6 = fir / 3.0; HS6 = US6 * US6 / G
                FLR -= QF(HS6, US6, QR[2])
                FLR += QF(QR[0], QR[1], QR[2])
            else:
                FLR += QF(QL[0], QL[1], QL[2])
                US2 = fil / 3.0; HS2 = US2 * US2 / G
                FLR -= QF(HS2, US2, QL[2])
                FLR += QF(QR[0], QR[1], QR[2])
        return FLR

    # ------------------------------------------------------------------
    # Kernel 1: CalculateFluxKernel — 1 thread per edge
    # ------------------------------------------------------------------
    @ti.kernel
    def calculate_flux():
        for idx in range(NE):
            cell_i = idx // 4
            KP = ti.cast(KLAS[idx], ti.i32)
            NC_raw = NAC[idx]
            NC = NC_raw - 1  # convert 1-indexed to 0-indexed; -1 means no neighbor

            HI = ti.max(H[cell_i], HM1)
            UI = U[cell_i]
            VI = V[cell_i]
            BI = ZBC[cell_i]
            ZI = ti.max(Z[cell_i], BI)
            H1 = H[cell_i]

            if HI <= HM2:
                UI = ti.select(UI >= 0.0, VMIN, -VMIN)
                VI = ti.select(VI >= 0.0, VMIN, -VMIN)

            COSJ = COSF[idx]
            SINJ = SINF[idx]

            QL = ti.Vector([HI, UI * COSJ + VI * SINJ, VI * COSJ - UI * SINJ])
            CL_v = ti.sqrt(G * HI)
            FIL_v = QL[1] + 2.0 * CL_v

            f0 = ti.cast(0.0, ti.f32)
            f1 = ti.cast(0.0, ti.f32)
            f2 = ti.cast(0.0, ti.f32)
            f3 = ti.cast(0.0, ti.f32)

            if KP == 4 or KP == 1:
                # Wall (KP=4) or water level boundary (KP=1, simplified as wall)
                f3 = HALF_G * H1 * H1
            elif KP != 0:
                # Other boundary types — treat as wall
                f3 = HALF_G * H1 * H1
            elif NC < 0:
                # No neighbor (shouldn't happen for KP==0, but safety)
                f3 = HALF_G * H1 * H1
            else:
                # Interior edge (KP == 0)
                HC = ti.max(H[NC], HM1)
                BC = ZBC[NC]
                ZC = ti.max(BC, Z[NC])
                UC = U[NC]
                VC = V[NC]

                if HI <= HM1 and HC <= HM1:
                    pass  # both dry
                elif ZI <= BC:
                    f0 = -C1 * ti.pow(HC, ti.cast(1.5, ti.f32))
                    f1 = HI * QL[1] * ti.abs(QL[1])
                    f3 = HALF_G * HI * HI
                elif ZC <= BI:
                    f0 = C1 * ti.pow(HI, ti.cast(1.5, ti.f32))
                    f1 = HI * ti.abs(QL[1]) * QL[1]
                    f2 = HI * ti.abs(QL[1]) * QL[2]
                elif HI <= HM2:
                    if ZC > ZI:
                        DH = ti.max(ZC - BI, HM1)
                        UN = -C1 * ti.sqrt(DH)
                        f0 = DH * UN
                        f1 = f0 * UN
                        f2 = f0 * (VC * COSJ - UC * SINJ)
                        f3 = HALF_G * HI * HI
                    else:
                        f0 = C1 * ti.pow(HI, ti.cast(1.5, ti.f32))
                        f3 = HALF_G * HI * HI
                elif HC <= HM2:
                    if ZI > ZC:
                        DH = ti.max(ZI - BC, HM1)
                        UN = C1 * ti.sqrt(DH)
                        HC1 = ZC - BI
                        f0 = DH * UN
                        f1 = f0 * UN
                        f2 = f0 * QL[2]
                        f3 = HALF_G * HC1 * HC1
                    else:
                        f0 = -C1 * ti.pow(HC, ti.cast(1.5, ti.f32))
                        f1 = HI * QL[1] * QL[1]
                        f3 = HALF_G * HI * HI
                else:
                    # Both wet — Osher Riemann solver
                    if cell_i < NC:
                        QR_h = ti.max(ZC - BI, HM1)
                        UR = UC * COSJ + VC * SINJ
                        ratio = ti.min(HC / QR_h, ti.cast(1.5, ti.f32))
                        QR_u = UR * ratio
                        if HC <= HM2 or QR_h <= HM2:
                            QR_u = ti.select(UR >= 0.0, VMIN, -VMIN)
                        QR_v_ = VC * COSJ - UC * SINJ
                        QR_vec = ti.Vector([QR_h, QR_u, QR_v_])
                        FLR_OS = osher(QL, QR_vec, FIL_v, H1)
                        f0 = FLR_OS[0]
                        f1 = FLR_OS[1] + (1.0 - ratio) * HC * UR * UR / 2.0
                        f2 = FLR_OS[2]
                        f3 = FLR_OS[3]
                    else:
                        COSJ1 = -COSJ
                        SINJ1 = -SINJ
                        QL1 = ti.Vector([
                            H[NC],
                            U[NC] * COSJ1 + V[NC] * SINJ1,
                            V[NC] * COSJ1 - U[NC] * SINJ1,
                        ])
                        CL1 = ti.sqrt(G * H[NC])
                        FIL1 = QL1[1] + 2.0 * CL1
                        HC2 = ti.max(HI, HM1)
                        ZC1 = ti.max(BI, ZI)
                        QR1_h = ti.max(ZC1 - BC, HM1)
                        UR1 = UI * COSJ1 + VI * SINJ1
                        ratio1 = ti.min(HC2 / QR1_h, ti.cast(1.5, ti.f32))
                        QR1_u = UR1 * ratio1
                        if HC2 <= HM2 or QR1_h <= HM2:
                            QR1_u = ti.select(UR1 >= 0.0, VMIN, -VMIN)
                        QR1_v_ = VI * COSJ1 - UI * SINJ1
                        QR1_vec = ti.Vector([QR1_h, QR1_u, QR1_v_])
                        FLR1 = osher(QL1, QR1_vec, FIL1, H[NC])
                        f0 = -FLR1[0]
                        f1 = FLR1[1] + (1.0 - ratio1) * HC2 * UR1 * UR1 / 2.0
                        f2 = FLR1[2]
                        ZA = ti.sqrt(FLR1[3] / HALF_G) + BC
                        HC3 = ti.max(ZA - BI, ti.cast(0.0, ti.f32))
                        f3 = HALF_G * HC3 * HC3

            FLUX0[idx] = f0
            FLUX1[idx] = f1
            FLUX2[idx] = f2
            FLUX3[idx] = f3

    # ------------------------------------------------------------------
    # Kernel 2: UpdateCellKernel — 1 thread per cell
    # ------------------------------------------------------------------
    @ti.kernel
    def update_cell():
        for i in range(CELL):
            H1 = H[i]
            U1 = U[i]
            V1 = V[i]
            BI = ZBC[i]

            WH = ti.cast(0.0, ti.f32)
            WU = ti.cast(0.0, ti.f32)
            WV = ti.cast(0.0, ti.f32)

            for j in ti.static(range(4)):
                idx = 4 * i + j
                SL = SIDE[idx]
                SLCA = SLCOS[idx]
                SLSA = SLSIN[idx]
                FLR_1 = FLUX1[idx] + FLUX3[idx]
                FLR_2 = FLUX2[idx]
                WH += SL * FLUX0[idx]
                WU += SLCA * FLR_1 - SLSA * FLR_2
                WV += SLSA * FLR_1 + SLCA * FLR_2

            DTA = ti.cast(DT, ti.f32) / AREA[i]
            H2 = ti.max(H1 - DTA * WH, HM1)
            Z2 = H2 + BI

            U2 = ti.cast(0.0, ti.f32)
            V2 = ti.cast(0.0, ti.f32)
            if H2 > HM1:
                if H2 <= HM2:
                    U2 = ti.select(U1 >= 0.0,
                                   ti.min(VMIN, ti.abs(U1)),
                                   -ti.min(VMIN, ti.abs(U1)))
                    V2 = ti.select(V1 >= 0.0,
                                   ti.min(VMIN, ti.abs(V1)),
                                   -ti.min(VMIN, ti.abs(V1)))
                else:
                    QX1 = H1 * U1
                    QY1 = H1 * V1
                    DTAU = DTA * WU
                    DTAV = DTA * WV
                    WSF = FNC[i] * ti.sqrt(U1 * U1 + V1 * V1) / ti.pow(H1, ti.cast(0.33333, ti.f32))
                    U2 = (QX1 - DTAU - ti.cast(DT, ti.f32) * WSF * U1) / H2
                    V2 = (QY1 - DTAV - ti.cast(DT, ti.f32) * WSF * V1) / H2
                    if H2 > HM2:
                        U2 = ti.select(U2 >= 0.0,
                                       ti.min(ti.abs(U2), ti.cast(15.0, ti.f32)),
                                       -ti.min(ti.abs(U2), ti.cast(15.0, ti.f32)))
                        V2 = ti.select(V2 >= 0.0,
                                       ti.min(ti.abs(V2), ti.cast(15.0, ti.f32)),
                                       -ti.min(ti.abs(V2), ti.cast(15.0, ti.f32)))

            H[i] = H2
            U[i] = U2
            V[i] = V2
            Z[i] = Z2
            W[i] = ti.sqrt(U2 * U2 + V2 * V2)

    # ------------------------------------------------------------------
    # Step function
    # ------------------------------------------------------------------
    def step_fn():
        for _ in range(total_steps):
            calculate_flux()
            update_cell()

    def sync_fn():
        ti.sync()

    # Warm-compile
    calculate_flux()
    update_cell()
    ti.sync()
    # Reload initial state after warm-compile
    H.from_numpy(mesh["H"])
    U.from_numpy(mesh["U"])
    V.from_numpy(mesh["V"])
    Z.from_numpy(mesh["Z"])
    W.from_numpy(mesh["W"])

    return step_fn, sync_fn, H


if __name__ == "__main__":
    import time
    days = 10
    step_fn, sync_fn, H_field = run(days=days, backend="cuda")
    sync_fn()
    t0 = time.perf_counter()
    step_fn()
    sync_fn()
    t1 = time.perf_counter()
    elapsed = (t1 - t0) * 1000
    print(f"{days} days: {elapsed:.1f} ms  ({elapsed/days:.2f} ms/day)")
    h = H_field.to_numpy()
    print(f"H range: [{h.min():.6f}, {h.max():.6f}]")
