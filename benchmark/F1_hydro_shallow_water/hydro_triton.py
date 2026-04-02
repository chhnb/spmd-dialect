"""2D Shallow Water Equations (Osher Riemann solver) — Triton.

Dam-break on unstructured quad mesh.  Port of hydro-cal calculate_gpu.cu.

Triton is block-oriented, so the Osher solver's 16-case dispatch is handled
by pre-computing QF for all template states and selecting via tl.where masks.
Indirect neighbor access uses tl.load with arbitrary offsets (gather).
"""
import numpy as np
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G: tl.constexpr = 9.81
HALF_G: tl.constexpr = 4.905
HM1_C: tl.constexpr = 0.001
HM2_C: tl.constexpr = 0.01
VMIN_C: tl.constexpr = 0.001
C1_C: tl.constexpr = 0.3
MANNING_N_SQ_G: tl.constexpr = 9.81 * 0.03 * 0.03  # g * n^2


# ---------------------------------------------------------------------------
# Osher Riemann solver — branchless precompute-and-select
# ---------------------------------------------------------------------------
@triton.jit
def osher_flux(
    QL_h, QL_u, QL_v,
    QR_h, QR_u, QR_v,
    FIL_in, H_pos,
):
    """Osher solver returning (FLR0, FLR1, FLR2, FLR3) as block tensors."""
    CR = tl.sqrt(9.81 * QR_h)
    FIR = QR_u - 2.0 * CR
    fil = FIL_in
    fir = FIR
    UA = (fil + fir) / 2.0
    CA = tl.abs((fil - fir) / 4.0)
    CL_v = tl.sqrt(9.81 * H_pos)

    # --- K1, K2 as integer tensors ---
    K2 = tl.where(CA < UA, 1,
         tl.where((UA >= 0.0) & (UA < CA), 2,
         tl.where((UA >= -CA) & (UA < 0.0), 3, 4)))
    K1 = tl.where((QL_u < CL_v) & (QR_u >= -CR), 1,
         tl.where((QL_u >= CL_v) & (QR_u >= -CR), 2,
         tl.where((QL_u < CL_v) & (QR_u < -CR), 3, 4)))

    # --- Pre-compute QF for each template state ---
    # T=1: left state  (QL)
    F1_0 = QL_h * QL_u
    F1_1 = F1_0 * QL_u
    F1_2 = F1_0 * QL_v
    F1_3 = 4.905 * QL_h * QL_h

    # T=2: sonic point on FIL line
    US2 = fil / 3.0
    HS2 = US2 * US2 / 9.81
    F2_0 = HS2 * US2
    F2_1 = F2_0 * US2
    F2_2 = F2_0 * QL_v
    F2_3 = 4.905 * HS2 * HS2

    # T=3: intermediate — uses ua3 = (fil+fir)/2 as velocity,
    #       fil_new = fil - ua3 = (fil-fir)/2 for depth
    ua3 = (fil + fir) / 2.0
    fil3 = fil - ua3   # (fil - fir) / 2
    HA3 = fil3 * fil3 / (4.0 * 9.81)
    F3_0 = HA3 * ua3
    F3_1 = F3_0 * ua3
    F3_2 = F3_0 * QL_v
    F3_3 = 4.905 * HA3 * HA3

    # T=5: intermediate — symmetric of T=3 for right side
    ua5 = (fil + fir) / 2.0
    fir5 = fir - ua5   # (fir - fil) / 2
    HA5 = fir5 * fir5 / (4.0 * 9.81)
    F5_0 = HA5 * ua5
    F5_1 = F5_0 * ua5
    F5_2 = F5_0 * QR_v
    F5_3 = 4.905 * HA5 * HA5

    # T=6: sonic point on FIR line (original fir)
    US6 = fir / 3.0
    HS6 = US6 * US6 / 9.81
    F6_0 = HS6 * US6
    F6_1 = F6_0 * US6
    F6_2 = F6_0 * QR_v
    F6_3 = 4.905 * HS6 * HS6

    # T=6m: sonic on FIR AFTER T=5 modified fir → fir5 = (fir-fil)/2
    US6m = fir5 / 3.0
    HS6m = US6m * US6m / 9.81
    F6m_0 = HS6m * US6m
    F6m_1 = F6m_0 * US6m
    F6m_2 = F6m_0 * QR_v
    F6m_3 = 4.905 * HS6m * HS6m

    # T=7: right state (QR)
    F7_0 = QR_h * QR_u
    F7_1 = F7_0 * QR_u
    F7_2 = F7_0 * QR_v
    F7_3 = 4.905 * QR_h * QR_h

    # --- Linear combinations for each (K1,K2) case ---
    # (1,1): +F2
    R11_0 = F2_0; R11_1 = F2_1; R11_2 = F2_2; R11_3 = F2_3
    # (1,2): +F3
    R12_0 = F3_0; R12_1 = F3_1; R12_2 = F3_2; R12_3 = F3_3
    # (1,3): +F5
    R13_0 = F5_0; R13_1 = F5_1; R13_2 = F5_2; R13_3 = F5_3
    # (1,4): +F6
    R14_0 = F6_0; R14_1 = F6_1; R14_2 = F6_2; R14_3 = F6_3
    # (2,1): +F1
    R21_0 = F1_0; R21_1 = F1_1; R21_2 = F1_2; R21_3 = F1_3
    # (2,2): +F1 -F2 +F3
    R22_0 = F1_0 - F2_0 + F3_0
    R22_1 = F1_1 - F2_1 + F3_1
    R22_2 = F1_2 - F2_2 + F3_2
    R22_3 = F1_3 - F2_3 + F3_3
    # (2,3): +F1 -F2 +F5
    R23_0 = F1_0 - F2_0 + F5_0
    R23_1 = F1_1 - F2_1 + F5_1
    R23_2 = F1_2 - F2_2 + F5_2
    R23_3 = F1_3 - F2_3 + F5_3
    # (2,4): +F1 -F2 +F6
    R24_0 = F1_0 - F2_0 + F6_0
    R24_1 = F1_1 - F2_1 + F6_1
    R24_2 = F1_2 - F2_2 + F6_2
    R24_3 = F1_3 - F2_3 + F6_3
    # (3,1): +F2 -F6 +F7
    R31_0 = F2_0 - F6_0 + F7_0
    R31_1 = F2_1 - F6_1 + F7_1
    R31_2 = F2_2 - F6_2 + F7_2
    R31_3 = F2_3 - F6_3 + F7_3
    # (3,2): +F3 -F6 +F7
    R32_0 = F3_0 - F6_0 + F7_0
    R32_1 = F3_1 - F6_1 + F7_1
    R32_2 = F3_2 - F6_2 + F7_2
    R32_3 = F3_3 - F6_3 + F7_3
    # (3,3): +F5 -F6m +F7   (T=5 modifies fir, T=6 uses modified)
    R33_0 = F5_0 - F6m_0 + F7_0
    R33_1 = F5_1 - F6m_1 + F7_1
    R33_2 = F5_2 - F6m_2 + F7_2
    R33_3 = F5_3 - F6m_3 + F7_3
    # (3,4): +F7
    R34_0 = F7_0; R34_1 = F7_1; R34_2 = F7_2; R34_3 = F7_3
    # (4,1): +F1 -F6 +F7
    R41_0 = F1_0 - F6_0 + F7_0
    R41_1 = F1_1 - F6_1 + F7_1
    R41_2 = F1_2 - F6_2 + F7_2
    R41_3 = F1_3 - F6_3 + F7_3
    # (4,2): +F1 -F2 +F3 -F6 +F7
    R42_0 = F1_0 - F2_0 + F3_0 - F6_0 + F7_0
    R42_1 = F1_1 - F2_1 + F3_1 - F6_1 + F7_1
    R42_2 = F1_2 - F2_2 + F3_2 - F6_2 + F7_2
    R42_3 = F1_3 - F2_3 + F3_3 - F6_3 + F7_3
    # (4,3): +F1 -F2 +F5 -F6m +F7  (T=5 modifies fir, T=6 uses modified)
    R43_0 = F1_0 - F2_0 + F5_0 - F6m_0 + F7_0
    R43_1 = F1_1 - F2_1 + F5_1 - F6m_1 + F7_1
    R43_2 = F1_2 - F2_2 + F5_2 - F6m_2 + F7_2
    R43_3 = F1_3 - F2_3 + F5_3 - F6m_3 + F7_3
    # (4,4): +F1 -F2 +F7
    R44_0 = F1_0 - F2_0 + F7_0
    R44_1 = F1_1 - F2_1 + F7_1
    R44_2 = F1_2 - F2_2 + F7_2
    R44_3 = F1_3 - F2_3 + F7_3

    # --- Select result via nested tl.where on K1, K2 ---
    # Encode case as K1*10 + K2 for unique selection
    case = K1 * 10 + K2

    zero = tl.zeros_like(QL_h)
    FLR0 = tl.where(case == 11, R11_0,
            tl.where(case == 12, R12_0,
            tl.where(case == 13, R13_0,
            tl.where(case == 14, R14_0,
            tl.where(case == 21, R21_0,
            tl.where(case == 22, R22_0,
            tl.where(case == 23, R23_0,
            tl.where(case == 24, R24_0,
            tl.where(case == 31, R31_0,
            tl.where(case == 32, R32_0,
            tl.where(case == 33, R33_0,
            tl.where(case == 34, R34_0,
            tl.where(case == 41, R41_0,
            tl.where(case == 42, R42_0,
            tl.where(case == 43, R43_0,
            tl.where(case == 44, R44_0, zero))))))))))))))))

    FLR1 = tl.where(case == 11, R11_1,
            tl.where(case == 12, R12_1,
            tl.where(case == 13, R13_1,
            tl.where(case == 14, R14_1,
            tl.where(case == 21, R21_1,
            tl.where(case == 22, R22_1,
            tl.where(case == 23, R23_1,
            tl.where(case == 24, R24_1,
            tl.where(case == 31, R31_1,
            tl.where(case == 32, R32_1,
            tl.where(case == 33, R33_1,
            tl.where(case == 34, R34_1,
            tl.where(case == 41, R41_1,
            tl.where(case == 42, R42_1,
            tl.where(case == 43, R43_1,
            tl.where(case == 44, R44_1, zero))))))))))))))))

    FLR2 = tl.where(case == 11, R11_2,
            tl.where(case == 12, R12_2,
            tl.where(case == 13, R13_2,
            tl.where(case == 14, R14_2,
            tl.where(case == 21, R21_2,
            tl.where(case == 22, R22_2,
            tl.where(case == 23, R23_2,
            tl.where(case == 24, R24_2,
            tl.where(case == 31, R31_2,
            tl.where(case == 32, R32_2,
            tl.where(case == 33, R33_2,
            tl.where(case == 34, R34_2,
            tl.where(case == 41, R41_2,
            tl.where(case == 42, R42_2,
            tl.where(case == 43, R43_2,
            tl.where(case == 44, R44_2, zero))))))))))))))))

    FLR3 = tl.where(case == 11, R11_3,
            tl.where(case == 12, R12_3,
            tl.where(case == 13, R13_3,
            tl.where(case == 14, R14_3,
            tl.where(case == 21, R21_3,
            tl.where(case == 22, R22_3,
            tl.where(case == 23, R23_3,
            tl.where(case == 24, R24_3,
            tl.where(case == 31, R31_3,
            tl.where(case == 32, R32_3,
            tl.where(case == 33, R33_3,
            tl.where(case == 34, R34_3,
            tl.where(case == 41, R41_3,
            tl.where(case == 42, R42_3,
            tl.where(case == 43, R43_3,
            tl.where(case == 44, R44_3, zero))))))))))))))))

    return FLR0, FLR1, FLR2, FLR3


# ---------------------------------------------------------------------------
# Per-edge flux computation (called 4 times per cell, once per edge)
# ---------------------------------------------------------------------------
@triton.jit
def compute_edge_flux(
    pos,          # [BLOCK] cell indices (1-indexed)
    j_stride,     # stride for edge dimension in 2D arrays
    j_offset,     # j * j_stride (pre-computed for this edge)
    NAC_ptr, KLAS_ptr, SIDE_ptr, COSF_ptr, SINF_ptr,
    ZBC_ptr, H_ptr, U_ptr, V_ptr, Z_ptr,
    HI, UI, VI, ZI, BI,   # pre-computed local cell state
    mask,         # valid cell mask
):
    """Compute flux for one edge direction. Returns (flux0..3, SL, SLCA, SLSA)."""
    # Load edge data
    edge_off = j_offset + pos
    NC = tl.load(NAC_ptr + edge_off, mask=mask, other=0)
    KP = tl.load(KLAS_ptr + edge_off, mask=mask, other=0)
    COSJ = tl.load(COSF_ptr + edge_off, mask=mask, other=0.0)
    SINJ = tl.load(SINF_ptr + edge_off, mask=mask, other=0.0)
    SL = tl.load(SIDE_ptr + edge_off, mask=mask, other=0.0)
    SLCA = SL * COSJ
    SLSA = SL * SINJ

    # Left state in edge-local coords
    QL_h = HI
    QL_u = UI * COSJ + VI * SINJ
    QL_v = VI * COSJ - UI * SINJ
    CL_v = tl.sqrt(9.81 * HI)
    FIL_v = QL_u + 2.0 * CL_v

    # Neighbor state (gather load)
    nc_valid = (NC != 0) & mask
    HC = tl.load(H_ptr + NC, mask=nc_valid, other=0.0)
    HC = tl.where(nc_valid, tl.maximum(HC, 0.001), 0.0)
    BC = tl.load(ZBC_ptr + NC, mask=nc_valid, other=0.0)
    ZC = tl.load(Z_ptr + NC, mask=nc_valid, other=0.0)
    ZC = tl.where(nc_valid, tl.maximum(BC, ZC), 0.0)
    UC = tl.load(U_ptr + NC, mask=nc_valid, other=0.0)
    VC = tl.load(V_ptr + NC, mask=nc_valid, other=0.0)

    H1 = tl.load(H_ptr + pos, mask=mask, other=0.0)

    # ---- Flux computation with branchless masking ----
    # Start with zero flux
    f0 = tl.zeros_like(HI)
    f1 = tl.zeros_like(HI)
    f2 = tl.zeros_like(HI)
    f3 = tl.zeros_like(HI)

    # Case: boundary (KP != 0) → wall: pressure only
    is_bnd = (KP != 0)
    f3 = tl.where(is_bnd, 4.905 * H1 * H1, f3)

    # Case: both dry
    both_dry = (~is_bnd) & (HI <= 0.001) & (HC <= 0.001)
    # flux stays zero

    # Case: ZI <= BC (local below neighbor bed)
    zi_le_bc = (~is_bnd) & (~both_dry) & (ZI <= BC)
    f0 = tl.where(zi_le_bc, -0.3 * tl.math.pow(HC, 1.5), f0)
    f1 = tl.where(zi_le_bc, HI * QL_u * tl.abs(QL_u), f1)
    f3 = tl.where(zi_le_bc, 4.905 * HI * HI, f3)

    # Case: ZC <= BI (neighbor below local bed)
    zc_le_bi = (~is_bnd) & (~both_dry) & (~zi_le_bc) & (ZC <= BI)
    f0 = tl.where(zc_le_bi, 0.3 * tl.math.pow(HI, 1.5), f0)
    f1 = tl.where(zc_le_bi, HI * tl.abs(QL_u) * QL_u, f1)
    f2 = tl.where(zc_le_bi, HI * tl.abs(QL_u) * QL_v, f2)

    # Case: HI <= HM2 (local shallow)
    hi_shallow = (~is_bnd) & (~both_dry) & (~zi_le_bc) & (~zc_le_bi) & (HI <= 0.01)
    hi_zc_gt = hi_shallow & (ZC > ZI)
    DH_a = tl.maximum(ZC - tl.load(ZBC_ptr + pos, mask=mask, other=0.0), 0.001)
    UN_a = -0.3 * tl.sqrt(DH_a)
    f0 = tl.where(hi_zc_gt, DH_a * UN_a, f0)
    f1 = tl.where(hi_zc_gt, DH_a * UN_a * UN_a, f1)
    f2 = tl.where(hi_zc_gt, DH_a * UN_a * (VC * COSJ - UC * SINJ), f2)
    f3 = tl.where(hi_zc_gt, 4.905 * HI * HI, f3)
    hi_zc_le = hi_shallow & (~hi_zc_gt)
    f0 = tl.where(hi_zc_le, 0.3 * tl.math.pow(HI, 1.5), f0)
    f3 = tl.where(hi_zc_le, 4.905 * HI * HI, f3)

    # Case: HC <= HM2 (neighbor shallow)
    hc_shallow = (~is_bnd) & (~both_dry) & (~zi_le_bc) & (~zc_le_bi) & (~hi_shallow) & (HC <= 0.01)
    hc_zi_gt = hc_shallow & (ZI > ZC)
    DH_b = tl.maximum(ZI - BC, 0.001)
    UN_b = 0.3 * tl.sqrt(DH_b)
    HC1_b = ZC - tl.load(ZBC_ptr + pos, mask=mask, other=0.0)
    f0 = tl.where(hc_zi_gt, DH_b * UN_b, f0)
    f1 = tl.where(hc_zi_gt, DH_b * UN_b * UN_b, f1)
    f2 = tl.where(hc_zi_gt, DH_b * UN_b * QL_v, f2)
    f3 = tl.where(hc_zi_gt, 4.905 * HC1_b * HC1_b, f3)
    hc_zi_le = hc_shallow & (~hc_zi_gt)
    f0 = tl.where(hc_zi_le, -0.3 * tl.math.pow(HC, 1.5), f0)
    f1 = tl.where(hc_zi_le, HI * QL_u * QL_u, f1)
    f3 = tl.where(hc_zi_le, 4.905 * HI * HI, f3)

    # Case: both wet — Osher Riemann solver
    both_wet = (~is_bnd) & (~both_dry) & (~zi_le_bc) & (~zc_le_bi) & (~hi_shallow) & (~hc_shallow)
    pos_lt_nc = both_wet & (pos < NC)
    pos_ge_nc = both_wet & (~pos_lt_nc)

    zbc_pos = tl.load(ZBC_ptr + pos, mask=mask, other=0.0)

    # --- pos < NC path ---
    QR_h = tl.maximum(ZC - zbc_pos, 0.001)
    UR = UC * COSJ + VC * SINJ
    ratio = tl.minimum(HC / QR_h, 1.5)
    QR_u = UR * ratio
    shallow_qr = (HC <= 0.01) | (QR_h <= 0.01)
    QR_u = tl.where(shallow_qr, tl.where(UR >= 0.0, 0.001, -0.001), QR_u)
    QR_v = VC * COSJ - UC * SINJ
    H_pos_val = tl.load(H_ptr + pos, mask=mask, other=0.001)

    os0, os1, os2, os3 = osher_flux(QL_h, QL_u, QL_v, QR_h, QR_u, QR_v, FIL_v, H_pos_val)
    os1_adj = os1 + (1.0 - ratio) * HC * UR * UR / 2.0

    f0 = tl.where(pos_lt_nc, os0, f0)
    f1 = tl.where(pos_lt_nc, os1_adj, f1)
    f2 = tl.where(pos_lt_nc, os2, f2)
    f3 = tl.where(pos_lt_nc, os3, f3)

    # --- pos >= NC path (mirror) ---
    COSJ1 = -COSJ
    SINJ1 = -SINJ
    H_NC = tl.load(H_ptr + NC, mask=nc_valid, other=0.001)
    U_NC = tl.load(U_ptr + NC, mask=nc_valid, other=0.0)
    V_NC = tl.load(V_ptr + NC, mask=nc_valid, other=0.0)
    QL1_h = H_NC
    QL1_u = U_NC * COSJ1 + V_NC * SINJ1
    QL1_v = V_NC * COSJ1 - U_NC * SINJ1
    CL1 = tl.sqrt(9.81 * H_NC)
    FIL1 = QL1_u + 2.0 * CL1
    HC2 = tl.maximum(HI, 0.001)
    ZC1 = tl.maximum(zbc_pos, ZI)
    ZBC_NC = tl.load(ZBC_ptr + NC, mask=nc_valid, other=0.0)
    QR1_h = tl.maximum(ZC1 - ZBC_NC, 0.001)
    UR1 = UI * COSJ1 + VI * SINJ1
    ratio1 = tl.minimum(HC2 / QR1_h, 1.5)
    QR1_u = UR1 * ratio1
    shallow_qr1 = (HC2 <= 0.01) | (QR1_h <= 0.01)
    QR1_u = tl.where(shallow_qr1, tl.where(UR1 >= 0.0, 0.001, -0.001), QR1_u)
    QR1_v = VI * COSJ1 - UI * SINJ1

    mr0, mr1, mr2, mr3 = osher_flux(QL1_h, QL1_u, QL1_v, QR1_h, QR1_u, QR1_v, FIL1, H_NC)
    mr1_adj = mr1 + (1.0 - ratio1) * HC2 * UR1 * UR1 / 2.0
    ZA = tl.sqrt(mr3 / 4.905) + BC
    HC3 = tl.maximum(ZA - zbc_pos, 0.0)

    f0 = tl.where(pos_ge_nc, -mr0, f0)
    f1 = tl.where(pos_ge_nc, mr1_adj, f1)
    f2 = tl.where(pos_ge_nc, mr2, f2)
    f3 = tl.where(pos_ge_nc, 4.905 * HC3 * HC3, f3)

    return f0, f1, f2, f3, SL, SLCA, SLSA


# ---------------------------------------------------------------------------
# Main kernel
# ---------------------------------------------------------------------------
@triton.jit
def shallow_water_kernel(
    CEL,
    DT,
    NAC_ptr, KLAS_ptr, SIDE_ptr, COSF_ptr, SINF_ptr,
    SLCOS_ptr, SLSIN_ptr,
    AREA_ptr, ZBC_ptr, FNC_ptr,
    H_pre_ptr, U_pre_ptr, V_pre_ptr, Z_pre_ptr,
    H_res_ptr, U_res_ptr, V_res_ptr, Z_res_ptr, W_res_ptr,
    stride_edge,   # stride for edge dimension: CEL+1
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    pos = offsets + 1   # 1-indexed cell IDs
    mask = pos <= CEL

    # Load cell state
    H1 = tl.load(H_pre_ptr + pos, mask=mask, other=0.001)
    U1 = tl.load(U_pre_ptr + pos, mask=mask, other=0.0)
    V1 = tl.load(V_pre_ptr + pos, mask=mask, other=0.0)
    Z1 = tl.load(Z_pre_ptr + pos, mask=mask, other=0.0)
    BI = tl.load(ZBC_ptr + pos, mask=mask, other=0.0)

    HI = tl.maximum(H1, 0.001)
    UI = tl.where(HI <= 0.01, tl.where(U1 >= 0.0, 0.001, -0.001), U1)
    VI = tl.where(HI <= 0.01, tl.where(V1 >= 0.0, 0.001, -0.001), V1)
    ZI = tl.maximum(Z1, BI)

    WH = tl.zeros_like(H1)
    WU = tl.zeros_like(H1)
    WV = tl.zeros_like(H1)

    # Unroll 4 edges
    for j in tl.static_range(1, 5):
        j_off = j * stride_edge
        f0, f1, f2, f3, SL, SLCA, SLSA = compute_edge_flux(
            pos, stride_edge, j_off,
            NAC_ptr, KLAS_ptr, SIDE_ptr, COSF_ptr, SINF_ptr,
            ZBC_ptr, H_pre_ptr, U_pre_ptr, V_pre_ptr, Z_pre_ptr,
            HI, UI, VI, ZI, BI, mask,
        )
        FLR_1 = f1 + f3
        FLR_2 = f2
        WH += SL * f0
        WU += SLCA * FLR_1 - SLSA * FLR_2
        WV += SLSA * FLR_1 + SLCA * FLR_2

    # State update with Manning friction
    AREA_v = tl.load(AREA_ptr + pos, mask=mask, other=1.0)
    FNC_v = tl.load(FNC_ptr + pos, mask=mask, other=0.0)
    DTA = DT / AREA_v
    WDTA = DTA
    H2 = tl.maximum(H1 - WDTA * WH, 0.001)
    Z2 = H2 + BI

    # Velocity update — three regimes
    is_dry = H2 <= 0.001
    is_shallow = (~is_dry) & (H2 <= 0.01)
    is_wet = (~is_dry) & (~is_shallow)

    # Dry: zero velocity
    U2 = tl.zeros_like(H1)
    V2 = tl.zeros_like(H1)

    # Shallow: clamp
    U2 = tl.where(is_shallow, tl.where(U1 >= 0.0, 1.0, -1.0) * tl.minimum(0.001, tl.abs(U1)), U2)
    V2 = tl.where(is_shallow, tl.where(V1 >= 0.0, 1.0, -1.0) * tl.minimum(0.001, tl.abs(V1)), V2)

    # Wet: full momentum update with Manning friction
    QX1 = H1 * U1
    QY1 = H1 * V1
    DTAU = WDTA * WU
    DTAV = WDTA * WV
    speed = tl.sqrt(U1 * U1 + V1 * V1)
    WSF = FNC_v * speed / tl.math.pow(H1, 0.33333)
    U2_wet = (QX1 - DTAU - DT * WSF * U1) / H2
    V2_wet = (QY1 - DTAV - DT * WSF * V1) / H2
    # Velocity cap at 15 m/s
    U2_wet = tl.where(U2_wet >= 0.0, 1.0, -1.0) * tl.minimum(tl.abs(U2_wet), 15.0)
    V2_wet = tl.where(V2_wet >= 0.0, 1.0, -1.0) * tl.minimum(tl.abs(V2_wet), 15.0)
    U2 = tl.where(is_wet, U2_wet, U2)
    V2 = tl.where(is_wet, V2_wet, V2)

    W2 = tl.sqrt(U2 * U2 + V2 * V2)

    tl.store(H_res_ptr + pos, H2, mask=mask)
    tl.store(U_res_ptr + pos, U2, mask=mask)
    tl.store(V_res_ptr + pos, V2, mask=mask)
    tl.store(Z_res_ptr + pos, Z2, mask=mask)
    tl.store(W_res_ptr + pos, W2, mask=mask)


@triton.jit
def transfer_kernel(
    CEL,
    H_pre_ptr, U_pre_ptr, V_pre_ptr, Z_pre_ptr, W_pre_ptr,
    H_res_ptr, U_res_ptr, V_res_ptr, Z_res_ptr, W_res_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    pos = offsets + 1
    mask = pos <= CEL
    tl.store(H_pre_ptr + pos, tl.load(H_res_ptr + pos, mask=mask, other=0.0), mask=mask)
    tl.store(U_pre_ptr + pos, tl.load(U_res_ptr + pos, mask=mask, other=0.0), mask=mask)
    tl.store(V_pre_ptr + pos, tl.load(V_res_ptr + pos, mask=mask, other=0.0), mask=mask)
    tl.store(Z_pre_ptr + pos, tl.load(Z_res_ptr + pos, mask=mask, other=0.0), mask=mask)
    tl.store(W_pre_ptr + pos, tl.load(W_res_ptr + pos, mask=mask, other=0.0), mask=mask)


# ---------------------------------------------------------------------------
# Benchmark interface
# ---------------------------------------------------------------------------
import torch

BLOCK_SIZE = 256


def run(N, steps=1, backend="cuda"):
    assert backend == "cuda", "Triton requires CUDA backend"
    CEL = N * N
    dx = 1.0
    DT = float(0.5 * dx / (np.sqrt(9.81 * 2.0) + 1e-6))
    stride_edge = CEL + 1  # stride for edge dimension in [5][CEL+1] arrays

    # Build mesh on host (same as Warp/Taichi versions)
    nac_np = np.zeros((5, CEL + 1), dtype=np.int32)
    klas_np = np.zeros((5, CEL + 1), dtype=np.int32)
    side_np = np.zeros((5, CEL + 1), dtype=np.float64)
    cosf_np = np.zeros((5, CEL + 1), dtype=np.float64)
    sinf_np = np.zeros((5, CEL + 1), dtype=np.float64)
    area_np = np.zeros(CEL + 1, dtype=np.float64)
    zbc_np = np.zeros(CEL + 1, dtype=np.float64)
    fnc_np = np.full(CEL + 1, 9.81 * 0.03 * 0.03, dtype=np.float64)

    edge_cos = [0.0, 0.0, 1.0, 0.0, -1.0]
    edge_sin = [0.0, -1.0, 0.0, 1.0, 0.0]

    h_np = np.full(CEL + 1, 0.001, dtype=np.float64)
    z_np = np.zeros(CEL + 1, dtype=np.float64)

    for i in range(N):
        for jj in range(N):
            pos = i * N + jj + 1
            area_np[pos] = dx * dx
            for e in range(1, 5):
                side_np[e][pos] = dx
                cosf_np[e][pos] = edge_cos[e]
                sinf_np[e][pos] = edge_sin[e]
            if i > 0:     nac_np[1][pos] = (i - 1) * N + jj + 1
            else:         klas_np[1][pos] = 4
            if jj < N-1:  nac_np[2][pos] = i * N + (jj + 1) + 1
            else:         klas_np[2][pos] = 4
            if i < N-1:   nac_np[3][pos] = (i + 1) * N + jj + 1
            else:         klas_np[3][pos] = 4
            if jj > 0:    nac_np[4][pos] = i * N + (jj - 1) + 1
            else:         klas_np[4][pos] = 4
            h_np[pos] = 2.0 if jj < N // 2 else 0.5
            z_np[pos] = h_np[pos]

    # Upload to GPU as contiguous torch tensors
    dev = "cuda"
    NAC = torch.from_numpy(nac_np.ravel()).to(dev)          # [5*(CEL+1)]
    KLAS = torch.from_numpy(klas_np.ravel()).to(dev)
    SIDE = torch.from_numpy(side_np.ravel()).to(dtype=torch.float64, device=dev)
    COSF = torch.from_numpy(cosf_np.ravel()).to(dtype=torch.float64, device=dev)
    SINF = torch.from_numpy(sinf_np.ravel()).to(dtype=torch.float64, device=dev)
    AREA = torch.from_numpy(area_np).to(dtype=torch.float64, device=dev)
    ZBC = torch.from_numpy(zbc_np).to(dtype=torch.float64, device=dev)
    FNC = torch.from_numpy(fnc_np).to(dtype=torch.float64, device=dev)

    H_pre = torch.from_numpy(h_np).to(dtype=torch.float64, device=dev)
    U_pre = torch.zeros(CEL + 1, dtype=torch.float64, device=dev)
    V_pre = torch.zeros(CEL + 1, dtype=torch.float64, device=dev)
    Z_pre = torch.from_numpy(z_np).to(dtype=torch.float64, device=dev)
    W_pre = torch.zeros(CEL + 1, dtype=torch.float64, device=dev)
    H_res = torch.zeros_like(H_pre)
    U_res = torch.zeros_like(U_pre)
    V_res = torch.zeros_like(V_pre)
    Z_res = torch.zeros_like(Z_pre)
    W_res = torch.zeros_like(W_pre)

    grid = lambda meta: (triton.cdiv(CEL, meta["BLOCK_SIZE"]),)

    def step():
        for _ in range(steps):
            shallow_water_kernel[grid](
                CEL, DT,
                NAC, KLAS, SIDE, COSF, SINF, COSF, SINF,  # SLCOS/SLSIN = SIDE*COS/SIN computed in kernel
                AREA, ZBC, FNC,
                H_pre, U_pre, V_pre, Z_pre,
                H_res, U_res, V_res, Z_res, W_res,
                stride_edge,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            transfer_kernel[grid](
                CEL,
                H_pre, U_pre, V_pre, Z_pre, W_pre,
                H_res, U_res, V_res, Z_res, W_res,
                BLOCK_SIZE=BLOCK_SIZE,
            )

    def sync():
        torch.cuda.synchronize()

    return step, sync, H_pre


