"""Refactored Hydro-Cal (edge-parallel, fp32) — Warp.

Two-kernel design:
  1. CalculateFluxKernel: 1 thread per edge (4*CELL edges)
  2. UpdateCellKernel:    1 thread per cell, accumulates fluxes and updates H,U,V,Z,W

Real mesh data loaded via mesh_loader.  All arrays use fp32 (float in Warp).
"""
import numpy as np
import warp as wp

from mesh_loader import load_mesh

# ---------------------------------------------------------------------------
# Constants (plain Python floats — Warp treats as fp32 in float-typed code)
# ---------------------------------------------------------------------------
C0 = 1.33
C1 = 1.7
VMIN = 0.001
QLUA = 0.0
BRDTH = 100.0
G = 9.81
HALF_G = 4.905


# ---------------------------------------------------------------------------
# Warp helper functions
# ---------------------------------------------------------------------------
@wp.func
def safe_copysign(x: float, val: float) -> float:
    """copysign(x, val): return |x| with the sign of val."""
    ax = wp.abs(x)
    return wp.select(val >= 0.0, ax, -ax)


@wp.func
def QF(h: float, u: float, v: float) -> wp.vec4f:
    f0 = h * u
    return wp.vec4f(f0, f0 * u, f0 * v, 4.905 * h * h)


@wp.func
def osher_solver(
    QL_h: float, QL_u: float, QL_v: float,
    QR_h: float, QR_u: float, QR_v: float,
    FIL_in: float, H_pos: float,
) -> wp.vec4f:
    CR = wp.sqrt(9.81 * QR_h)
    FIR_v = QR_u - 2.0 * CR
    UA = (FIL_in + FIR_v) / 2.0
    CA = wp.abs((FIL_in - FIR_v) / 4.0)
    CL_v = wp.sqrt(9.81 * H_pos)

    FLR = wp.vec4f(0.0, 0.0, 0.0, 0.0)

    # K2 classification
    K2 = 0
    if CA < UA:
        K2 = 1
    elif UA >= 0.0 and UA < CA:
        K2 = 2
    elif UA >= -CA and UA < 0.0:
        K2 = 3
    else:
        K2 = 4

    # K1 classification
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

    # 16-case dispatch
    if K1 == 1:
        if K2 == 1:
            US = fil / 3.0
            HS = US * US / 9.81
            FLR = FLR + QF(HS, US, QL_v)
        elif K2 == 2:
            ua_ = (fil + fir) / 2.0
            fil = fil - ua_
            HA = fil * fil / (4.0 * 9.81)
            FLR = FLR + QF(HA, ua_, QL_v)
        elif K2 == 3:
            ua_ = (fil + fir) / 2.0
            fir = fir - ua_
            HA = fir * fir / (4.0 * 9.81)
            FLR = FLR + QF(HA, ua_, QR_v)
        else:
            US = fir / 3.0
            HS = US * US / 9.81
            FLR = FLR + QF(HS, US, QR_v)
    elif K1 == 2:
        if K2 == 1:
            FLR = FLR + QF(QL_h, QL_u, QL_v)
        elif K2 == 2:
            FLR = FLR + QF(QL_h, QL_u, QL_v)
            US2 = fil / 3.0
            HS2 = US2 * US2 / 9.81
            FLR = FLR - QF(HS2, US2, QL_v)
            ua_ = (fil + fir) / 2.0
            fil = fil - ua_
            HA = fil * fil / (4.0 * 9.81)
            FLR = FLR + QF(HA, ua_, QL_v)
        elif K2 == 3:
            FLR = FLR + QF(QL_h, QL_u, QL_v)
            US2 = fil / 3.0
            HS2 = US2 * US2 / 9.81
            FLR = FLR - QF(HS2, US2, QL_v)
            ua_ = (fil + fir) / 2.0
            fir = fir - ua_
            HA = fir * fir / (4.0 * 9.81)
            FLR = FLR + QF(HA, ua_, QR_v)
        else:
            FLR = FLR + QF(QL_h, QL_u, QL_v)
            US2 = fil / 3.0
            HS2 = US2 * US2 / 9.81
            FLR = FLR - QF(HS2, US2, QL_v)
            US6 = fir / 3.0
            HS6 = US6 * US6 / 9.81
            FLR = FLR + QF(HS6, US6, QR_v)
    elif K1 == 3:
        if K2 == 1:
            US2 = fil / 3.0
            HS2 = US2 * US2 / 9.81
            FLR = FLR + QF(HS2, US2, QL_v)
            US6 = fir / 3.0
            HS6 = US6 * US6 / 9.81
            FLR = FLR - QF(HS6, US6, QR_v)
            FLR = FLR + QF(QR_h, QR_u, QR_v)
        elif K2 == 2:
            ua_ = (fil + fir) / 2.0
            fil = fil - ua_
            HA = fil * fil / (4.0 * 9.81)
            FLR = FLR + QF(HA, ua_, QL_v)
            US6 = fir / 3.0
            HS6 = US6 * US6 / 9.81
            FLR = FLR - QF(HS6, US6, QR_v)
            FLR = FLR + QF(QR_h, QR_u, QR_v)
        elif K2 == 3:
            ua_ = (fil + fir) / 2.0
            fir = fir - ua_
            HA = fir * fir / (4.0 * 9.81)
            FLR = FLR + QF(HA, ua_, QR_v)
            US6b = fir / 3.0
            HS6b = US6b * US6b / 9.81
            FLR = FLR - QF(HS6b, US6b, QR_v)
            FLR = FLR + QF(QR_h, QR_u, QR_v)
        else:
            FLR = FLR + QF(QR_h, QR_u, QR_v)
    else:  # K1 == 4
        if K2 == 1:
            FLR = FLR + QF(QL_h, QL_u, QL_v)
            US6 = fir / 3.0
            HS6 = US6 * US6 / 9.81
            FLR = FLR - QF(HS6, US6, QR_v)
            FLR = FLR + QF(QR_h, QR_u, QR_v)
        elif K2 == 2:
            FLR = FLR + QF(QL_h, QL_u, QL_v)
            US2 = fil / 3.0
            HS2 = US2 * US2 / 9.81
            FLR = FLR - QF(HS2, US2, QL_v)
            ua_ = (fil + fir) / 2.0
            fil = fil - ua_
            HA = fil * fil / (4.0 * 9.81)
            FLR = FLR + QF(HA, ua_, QL_v)
            US6 = fir / 3.0
            HS6 = US6 * US6 / 9.81
            FLR = FLR - QF(HS6, US6, QR_v)
            FLR = FLR + QF(QR_h, QR_u, QR_v)
        elif K2 == 3:
            FLR = FLR + QF(QL_h, QL_u, QL_v)
            US2 = fil / 3.0
            HS2 = US2 * US2 / 9.81
            FLR = FLR - QF(HS2, US2, QL_v)
            ua_ = (fil + fir) / 2.0
            fir = fir - ua_
            HA = fir * fir / (4.0 * 9.81)
            FLR = FLR + QF(HA, ua_, QR_v)
            US6 = fir / 3.0
            HS6 = US6 * US6 / 9.81
            FLR = FLR - QF(HS6, US6, QR_v)
            FLR = FLR + QF(QR_h, QR_u, QR_v)
        else:
            FLR = FLR + QF(QL_h, QL_u, QL_v)
            US2 = fil / 3.0
            HS2 = US2 * US2 / 9.81
            FLR = FLR - QF(HS2, US2, QL_v)
            FLR = FLR + QF(QR_h, QR_u, QR_v)

    return FLR


# ---------------------------------------------------------------------------
# Kernel 1: CalculateFlux  (1 thread per edge, 4*CELL threads)
# ---------------------------------------------------------------------------
@wp.kernel
def calculate_flux_kernel(
    CELL: int,
    HM1: float,
    HM2: float,
    # Cell arrays [CELL]
    H: wp.array(dtype=float),
    U: wp.array(dtype=float),
    V: wp.array(dtype=float),
    Z: wp.array(dtype=float),
    ZBC: wp.array(dtype=float),
    ZB1: wp.array(dtype=float),
    # Edge arrays [4*CELL]
    NAC: wp.array(dtype=int),
    KLAS: wp.array(dtype=float),
    COSF: wp.array(dtype=float),
    SINF: wp.array(dtype=float),
    # Flux outputs [4*CELL]
    FLUX0: wp.array(dtype=float),
    FLUX1: wp.array(dtype=float),
    FLUX2: wp.array(dtype=float),
    FLUX3: wp.array(dtype=float),
):
    idx = wp.tid()
    if idx >= 4 * CELL:
        return

    i = idx / 4   # cell index (0-based)
    # j = idx % 4  # edge within cell (not needed explicitly)

    # Load cell i state
    H1 = H[i]
    U1 = U[i]
    V1 = V[i]
    BI = ZBC[i]

    HI = wp.max(H1, HM1)
    UI = U1
    VI = V1
    if HI <= HM2:
        UI = wp.select(UI >= 0.0, 0.001, -0.001)
        VI = wp.select(VI >= 0.0, 0.001, -0.001)
    ZI = wp.max(Z[i], ZBC[i])

    # Edge geometry
    COSJ = COSF[idx]
    SINJ = SINF[idx]

    # Rotate to edge-local frame
    QL_u = UI * COSJ + VI * SINJ
    QL_v = VI * COSJ - UI * SINJ
    CL_v = wp.sqrt(9.81 * HI)
    FIL_v = QL_u + 2.0 * CL_v

    # Neighbor cell (NAC stores 1-indexed; NC = NAC[idx] - 1, NC=-1 means no neighbor)
    NC = NAC[idx] - 1
    KP = KLAS[idx]
    KP_int = int(KP)

    HC = float(0.0)
    BC = float(0.0)
    ZC = float(0.0)
    UC = float(0.0)
    VC = float(0.0)
    if NC >= 0 and NC < CELL:
        HC = wp.max(H[NC], HM1)
        BC = ZBC[NC]
        ZC = wp.max(ZBC[NC], Z[NC])
        UC = U[NC]
        VC = V[NC]

    flux0 = float(0.0)
    flux1 = float(0.0)
    flux2 = float(0.0)
    flux3 = float(0.0)

    if KP_int == 4:
        # Wall boundary
        flux3 = 4.905 * H1 * H1
    elif KP_int == 1:
        # Water level boundary: treat like wall for simplified benchmark
        flux3 = 4.905 * H1 * H1
    elif KP_int != 0:
        # Other boundary types: treat as wall
        flux3 = 4.905 * H1 * H1
    elif HI <= HM1 and HC <= HM1:
        # Both dry — zero flux
        pass
    elif ZI <= BC:
        flux0 = -1.7 * wp.pow(HC, 1.5)
        flux1 = HI * QL_u * wp.abs(QL_u)
        flux3 = 4.905 * HI * HI
    elif ZC <= BI:
        flux0 = 1.7 * wp.pow(HI, 1.5)
        flux1 = HI * wp.abs(QL_u) * QL_u
        flux2 = HI * wp.abs(QL_u) * QL_v
    elif HI <= HM2:
        if ZC > ZI:
            DH = wp.max(ZC - ZBC[i], HM1)
            UN = -1.7 * wp.sqrt(DH)
            flux0 = DH * UN
            flux1 = flux0 * UN
            flux2 = flux0 * (VC * COSJ - UC * SINJ)
            flux3 = 4.905 * HI * HI
        else:
            flux0 = 1.7 * wp.pow(HI, 1.5)
            flux3 = 4.905 * HI * HI
    elif HC <= HM2:
        if ZI > ZC:
            DH = wp.max(ZI - BC, HM1)
            UN = 1.7 * wp.sqrt(DH)
            HC1 = ZC - ZBC[i]
            flux0 = DH * UN
            flux1 = flux0 * UN
            flux2 = flux0 * QL_v
            flux3 = 4.905 * HC1 * HC1
        else:
            flux0 = -1.7 * wp.pow(HC, 1.5)
            flux1 = HI * QL_u * QL_u
            flux3 = 4.905 * HI * HI
    else:
        # Both wet — Osher Riemann solver
        if i < NC:
            QR_h = wp.max(ZC - ZBC[i], HM1)
            UR = UC * COSJ + VC * SINJ
            ratio = wp.min(HC / QR_h, 1.5)
            QR_u = UR * ratio
            if HC <= HM2 or QR_h <= HM2:
                QR_u = wp.select(UR >= 0.0, 0.001, -0.001)
            QR_v_ = VC * COSJ - UC * SINJ
            FLR_OS = osher_solver(HI, QL_u, QL_v, QR_h, QR_u, QR_v_, FIL_v, H[i])
            flux0 = FLR_OS[0]
            flux1 = FLR_OS[1] + (1.0 - ratio) * HC * UR * UR / 2.0
            flux2 = FLR_OS[2]
            flux3 = FLR_OS[3]
        else:
            COSJ1 = -COSJ
            SINJ1 = -SINJ
            QL1_h = H[NC]
            QL1_u = U[NC] * COSJ1 + V[NC] * SINJ1
            QL1_v = V[NC] * COSJ1 - U[NC] * SINJ1
            CL1 = wp.sqrt(9.81 * H[NC])
            FIL1 = QL1_u + 2.0 * CL1
            HC2 = wp.max(HI, HM1)
            ZC1 = wp.max(ZBC[i], ZI)
            QR1_h = wp.max(ZC1 - ZBC[NC], HM1)
            UR1 = UI * COSJ1 + VI * SINJ1
            ratio1 = wp.min(HC2 / QR1_h, 1.5)
            QR1_u = UR1 * ratio1
            if HC2 <= HM2 or QR1_h <= HM2:
                QR1_u = wp.select(UR1 >= 0.0, 0.001, -0.001)
            QR1_v_ = VI * COSJ1 - UI * SINJ1
            FLR1 = osher_solver(QL1_h, QL1_u, QL1_v, QR1_h, QR1_u, QR1_v_, FIL1, H[NC])
            flux0 = -FLR1[0]
            flux1 = FLR1[1] + (1.0 - ratio1) * HC2 * UR1 * UR1 / 2.0
            flux2 = FLR1[2]
            ZA = wp.sqrt(FLR1[3] / 4.905) + BC
            HC3 = wp.max(ZA - ZBC[i], 0.0)
            flux3 = 4.905 * HC3 * HC3

    FLUX0[idx] = flux0
    FLUX1[idx] = flux1
    FLUX2[idx] = flux2
    FLUX3[idx] = flux3


# ---------------------------------------------------------------------------
# Kernel 2: UpdateCell  (1 thread per cell)
# ---------------------------------------------------------------------------
@wp.kernel
def update_cell_kernel(
    CELL: int,
    DT: float,
    HM1: float,
    HM2: float,
    # Cell arrays [CELL]
    H: wp.array(dtype=float),
    U: wp.array(dtype=float),
    V: wp.array(dtype=float),
    Z: wp.array(dtype=float),
    W: wp.array(dtype=float),
    ZBC: wp.array(dtype=float),
    AREA: wp.array(dtype=float),
    FNC: wp.array(dtype=float),
    NV: wp.array(dtype=int),
    # Edge arrays [4*CELL]
    SIDE: wp.array(dtype=float),
    SLCOS: wp.array(dtype=float),
    SLSIN: wp.array(dtype=float),
    FLUX0: wp.array(dtype=float),
    FLUX1: wp.array(dtype=float),
    FLUX2: wp.array(dtype=float),
    FLUX3: wp.array(dtype=float),
):
    i = wp.tid()
    if i >= CELL:
        return

    H1 = H[i]
    U1 = U[i]
    V1 = V[i]
    BI = ZBC[i]
    nv = NV[i]

    # Accumulate fluxes over edges of this cell
    WH = float(0.0)
    WU = float(0.0)
    WV = float(0.0)

    base = 4 * i
    for j in range(4):
        if j >= nv:
            break
        eidx = base + j
        SL = SIDE[eidx]
        SLCA = SLCOS[eidx]
        SLSA = SLSIN[eidx]
        f0 = FLUX0[eidx]
        f1 = FLUX1[eidx]
        f2 = FLUX2[eidx]
        f3 = FLUX3[eidx]
        FLR_1 = f1 + f3
        FLR_2 = f2
        WH = WH + SL * f0
        WU = WU + SLCA * FLR_1 - SLSA * FLR_2
        WV = WV + SLSA * FLR_1 + SLCA * FLR_2

    # State update with Manning friction
    DTA = DT / AREA[i]
    WDTA = DTA
    H2 = wp.max(H1 - WDTA * WH, HM1)
    Z2 = H2 + BI

    U2 = float(0.0)
    V2 = float(0.0)
    if H2 > HM1:
        if H2 <= HM2:
            U2 = wp.select(U1 >= 0.0, wp.min(VMIN, wp.abs(U1)), -wp.min(VMIN, wp.abs(U1)))
            V2 = wp.select(V1 >= 0.0, wp.min(VMIN, wp.abs(V1)), -wp.min(VMIN, wp.abs(V1)))
        else:
            QX1 = H1 * U1
            QY1 = H1 * V1
            DTAU = WDTA * WU
            DTAV = WDTA * WV
            WSF = FNC[i] * wp.sqrt(U1 * U1 + V1 * V1) / wp.pow(H1, 0.33333)
            U2 = (QX1 - DTAU - DT * WSF * U1) / H2
            V2 = (QY1 - DTAV - DT * WSF * V1) / H2
            if H2 > HM2:
                U2 = wp.select(U2 >= 0.0, wp.min(wp.abs(U2), 15.0), -wp.min(wp.abs(U2), 15.0))
                V2 = wp.select(V2 >= 0.0, wp.min(wp.abs(V2), 15.0), -wp.min(wp.abs(V2), 15.0))

    H[i] = H2
    U[i] = U2
    V[i] = V2
    Z[i] = Z2
    W[i] = wp.sqrt(U2 * U2 + V2 * V2)


# ---------------------------------------------------------------------------
# Benchmark interface
# ---------------------------------------------------------------------------
def run(days=10, backend="cuda"):
    """Set up and return (step_fn, sync_fn, H_array)."""
    wp.init()

    mesh = load_mesh()
    CELL = mesh["CELL"]
    DT_val = float(mesh["DT"])
    HM1_val = float(mesh["HM1"])
    HM2_val = float(mesh["HM2"])
    steps_per_day = mesh["steps_per_day"]
    total_steps = steps_per_day * days

    # Upload cell arrays [CELL] — fp32
    H_d = wp.array(mesh["H"], dtype=float, device=backend)
    U_d = wp.array(mesh["U"], dtype=float, device=backend)
    V_d = wp.array(mesh["V"], dtype=float, device=backend)
    Z_d = wp.array(mesh["Z"], dtype=float, device=backend)
    W_d = wp.array(mesh["W"], dtype=float, device=backend)
    ZBC_d = wp.array(mesh["ZBC"], dtype=float, device=backend)
    ZB1_d = wp.array(mesh["ZB1"], dtype=float, device=backend)
    AREA_d = wp.array(mesh["AREA"], dtype=float, device=backend)
    FNC_d = wp.array(mesh["FNC"], dtype=float, device=backend)
    NV_d = wp.array(mesh["NV"], dtype=int, device=backend)

    # Upload edge arrays [4*CELL] — fp32
    NAC_d = wp.array(mesh["NAC"], dtype=int, device=backend)
    KLAS_d = wp.array(mesh["KLAS"], dtype=float, device=backend)
    SIDE_d = wp.array(mesh["SIDE"], dtype=float, device=backend)
    COSF_d = wp.array(mesh["COSF"], dtype=float, device=backend)
    SINF_d = wp.array(mesh["SINF"], dtype=float, device=backend)
    SLCOS_d = wp.array(mesh["SLCOS"], dtype=float, device=backend)
    SLSIN_d = wp.array(mesh["SLSIN"], dtype=float, device=backend)

    # Flux buffers [4*CELL]
    FLUX0_d = wp.zeros(4 * CELL, dtype=float, device=backend)
    FLUX1_d = wp.zeros(4 * CELL, dtype=float, device=backend)
    FLUX2_d = wp.zeros(4 * CELL, dtype=float, device=backend)
    FLUX3_d = wp.zeros(4 * CELL, dtype=float, device=backend)

    num_edges = 4 * CELL

    def step_fn():
        for _ in range(total_steps):
            wp.launch(
                calculate_flux_kernel,
                dim=num_edges,
                inputs=[
                    CELL, HM1_val, HM2_val,
                    H_d, U_d, V_d, Z_d, ZBC_d, ZB1_d,
                    NAC_d, KLAS_d, COSF_d, SINF_d,
                    FLUX0_d, FLUX1_d, FLUX2_d, FLUX3_d,
                ],
                device=backend,
            )
            wp.launch(
                update_cell_kernel,
                dim=CELL,
                inputs=[
                    CELL, DT_val, HM1_val, HM2_val,
                    H_d, U_d, V_d, Z_d, W_d,
                    ZBC_d, AREA_d, FNC_d, NV_d,
                    SIDE_d, SLCOS_d, SLSIN_d,
                    FLUX0_d, FLUX1_d, FLUX2_d, FLUX3_d,
                ],
                device=backend,
            )

    def sync_fn():
        wp.synchronize_device(backend)

    # Warm-up: compile kernels
    step_fn()
    sync_fn()

    # Reset state for benchmark
    H_d.assign(wp.array(mesh["H"], dtype=float, device=backend))
    U_d.assign(wp.array(mesh["U"], dtype=float, device=backend))
    V_d.assign(wp.array(mesh["V"], dtype=float, device=backend))
    Z_d.assign(wp.array(mesh["Z"], dtype=float, device=backend))
    W_d.assign(wp.array(mesh["W"], dtype=float, device=backend))
    sync_fn()

    return step_fn, sync_fn, H_d


if __name__ == "__main__":
    import time

    step_fn, sync_fn, H_arr = run(days=10, backend="cuda")

    # Benchmark
    sync_fn()
    t0 = time.perf_counter()
    step_fn()
    sync_fn()
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000.0
    print(f"10 days: {elapsed_ms:.2f} ms  ({elapsed_ms / 10.0:.2f} ms/day)")

    H_np = H_arr.numpy()
    print(f"H range: [{H_np.min():.6f}, {H_np.max():.6f}]")
