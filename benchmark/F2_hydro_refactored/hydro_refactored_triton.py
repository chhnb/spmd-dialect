"""F2: Refactored Hydro-Cal — Triton (fp32, edge-parallel flux + cell-parallel update).

Two-kernel design matching the refactored CUDA kernel:
  1. calculate_flux_kernel: 1 thread per edge (4*CELL), computes flux per edge
  2. update_cell_kernel:    1 thread per cell (CELL),   accumulates fluxes, updates state
"""
import os
import sys
import numpy as np
import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(__file__))
from mesh_loader import load_mesh

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G = tl.constexpr(9.81)
HALF_G = tl.constexpr(4.905)
C1 = tl.constexpr(1.7)
VMIN = tl.constexpr(0.001)

BLOCK_SIZE_FLUX = 256
BLOCK_SIZE_CELL = 256


# ---------------------------------------------------------------------------
# Helper: QF inline (returns f0, f1, f2, f3)
# Triton doesn't support returning tuples from device functions easily,
# so we inline QF directly in the kernel.
# QF(h, u, v) = (h*u, h*u*u, h*u*v, HALF_G*h*h)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Kernel 1: calculate_flux_kernel — 1 thread per edge
# ---------------------------------------------------------------------------
@triton.jit
def calculate_flux_kernel(
    # Cell arrays
    H_ptr, U_ptr, V_ptr, Z_ptr, ZBC_ptr,
    # Edge arrays
    NAC_ptr, KLAS_ptr, COSF_ptr, SINF_ptr,
    # Output flux arrays
    FLUX0_ptr, FLUX1_ptr, FLUX2_ptr, FLUX3_ptr,
    # Scalars
    NE: tl.constexpr, HM1: tl.constexpr, HM2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < NE

    cell_i = idx // 4
    KP_f = tl.load(KLAS_ptr + idx, mask=mask, other=0.0)
    KP = KP_f.to(tl.int32)
    NC_raw = tl.load(NAC_ptr + idx, mask=mask, other=0)
    NC = NC_raw - 1  # convert 1-indexed to 0-indexed

    # Load cell_i data
    HI_raw = tl.load(H_ptr + cell_i, mask=mask, other=0.0)
    HI = tl.maximum(HI_raw, HM1)
    UI = tl.load(U_ptr + cell_i, mask=mask, other=0.0)
    VI = tl.load(V_ptr + cell_i, mask=mask, other=0.0)
    BI = tl.load(ZBC_ptr + cell_i, mask=mask, other=0.0)
    ZI_raw = tl.load(Z_ptr + cell_i, mask=mask, other=0.0)
    ZI = tl.maximum(ZI_raw, BI)
    H1 = HI_raw

    # Shallow cell velocity clamping
    shallow = HI <= HM2
    UI = tl.where(shallow & (UI >= 0.0), VMIN, tl.where(shallow, -VMIN, UI))
    VI = tl.where(shallow & (VI >= 0.0), VMIN, tl.where(shallow, -VMIN, VI))

    COSJ = tl.load(COSF_ptr + idx, mask=mask, other=0.0)
    SINJ = tl.load(SINF_ptr + idx, mask=mask, other=0.0)

    QL_h = HI
    QL_u = UI * COSJ + VI * SINJ
    QL_v = VI * COSJ - UI * SINJ
    CL_v = tl.sqrt(G * HI)
    FIL_v = QL_u + 2.0 * CL_v

    # Initialize flux outputs
    f0 = tl.zeros_like(HI)
    f1 = tl.zeros_like(HI)
    f2 = tl.zeros_like(HI)
    f3 = tl.zeros_like(HI)

    # --- Boundary cases: KP != 0, or no neighbor ---
    is_boundary = (KP != 0) | (NC < 0)
    wall_f3 = HALF_G * H1 * H1
    f3 = tl.where(is_boundary, wall_f3, f3)

    # --- Interior edge: KP == 0 and NC >= 0 ---
    is_interior = (~is_boundary) & mask

    # Load neighbor data (use NC, valid only when is_interior)
    # Clamp NC to 0 for invalid loads (the result will be masked out)
    NC_safe = tl.maximum(NC, 0)
    HC_raw = tl.load(H_ptr + NC_safe, mask=is_interior, other=0.0)
    HC = tl.maximum(HC_raw, HM1)
    BC = tl.load(ZBC_ptr + NC_safe, mask=is_interior, other=0.0)
    ZC_raw = tl.load(Z_ptr + NC_safe, mask=is_interior, other=0.0)
    ZC = tl.maximum(BC, ZC_raw)
    UC = tl.load(U_ptr + NC_safe, mask=is_interior, other=0.0)
    VC = tl.load(V_ptr + NC_safe, mask=is_interior, other=0.0)

    # Sub-cases for interior edges
    both_dry = is_interior & (HI <= HM1) & (HC <= HM1)

    zi_le_bc = is_interior & (~both_dry) & (ZI <= BC)
    zc_le_bi = is_interior & (~both_dry) & (~(ZI <= BC)) & (ZC <= BI)
    hi_le_hm2 = is_interior & (~both_dry) & (~(ZI <= BC)) & (~(ZC <= BI)) & (HI <= HM2)
    hc_le_hm2 = is_interior & (~both_dry) & (~(ZI <= BC)) & (~(ZC <= BI)) & (~(HI <= HM2)) & (HC <= HM2)
    both_wet = is_interior & (~both_dry) & (~(ZI <= BC)) & (~(ZC <= BI)) & (~(HI <= HM2)) & (~(HC <= HM2))

    # --- zi_le_bc: ZI <= BC ---
    HC_pow = tl.extra.cuda.libdevice.pow(HC, 1.5)
    f0 = tl.where(zi_le_bc, -C1 * HC_pow, f0)
    f1 = tl.where(zi_le_bc, HI * QL_u * tl.abs(QL_u), f1)
    f3 = tl.where(zi_le_bc, HALF_G * HI * HI, f3)

    # --- zc_le_bi: ZC <= BI ---
    HI_pow = tl.extra.cuda.libdevice.pow(HI, 1.5)
    f0 = tl.where(zc_le_bi, C1 * HI_pow, f0)
    f1 = tl.where(zc_le_bi, HI * tl.abs(QL_u) * QL_u, f1)
    f2 = tl.where(zc_le_bi, HI * tl.abs(QL_u) * QL_v, f2)

    # --- hi_le_hm2: HI <= HM2 ---
    zc_gt_zi = ZC > ZI
    hi_hm2_case1 = hi_le_hm2 & zc_gt_zi
    hi_hm2_case2 = hi_le_hm2 & (~zc_gt_zi)

    DH_1 = tl.maximum(ZC - BI, HM1)
    UN_1 = -C1 * tl.sqrt(DH_1)
    f0_hm2_c1 = DH_1 * UN_1
    f0 = tl.where(hi_hm2_case1, f0_hm2_c1, f0)
    f1 = tl.where(hi_hm2_case1, f0_hm2_c1 * UN_1, f1)
    f2 = tl.where(hi_hm2_case1, f0_hm2_c1 * (VC * COSJ - UC * SINJ), f2)
    f3 = tl.where(hi_hm2_case1, HALF_G * HI * HI, f3)

    f0 = tl.where(hi_hm2_case2, C1 * HI_pow, f0)
    f3 = tl.where(hi_hm2_case2, HALF_G * HI * HI, f3)

    # --- hc_le_hm2: HC <= HM2 ---
    zi_gt_zc = ZI > ZC
    hc_hm2_case1 = hc_le_hm2 & zi_gt_zc
    hc_hm2_case2 = hc_le_hm2 & (~zi_gt_zc)

    DH_2 = tl.maximum(ZI - BC, HM1)
    UN_2 = C1 * tl.sqrt(DH_2)
    HC1_v = ZC - BI
    f0_hcm2_c1 = DH_2 * UN_2
    f0 = tl.where(hc_hm2_case1, f0_hcm2_c1, f0)
    f1 = tl.where(hc_hm2_case1, f0_hcm2_c1 * UN_2, f1)
    f2 = tl.where(hc_hm2_case1, f0_hcm2_c1 * QL_v, f2)
    f3 = tl.where(hc_hm2_case1, HALF_G * HC1_v * HC1_v, f3)

    f0 = tl.where(hc_hm2_case2, -C1 * HC_pow, f0)
    f1 = tl.where(hc_hm2_case2, HI * QL_u * QL_u, f1)
    f3 = tl.where(hc_hm2_case2, HALF_G * HI * HI, f3)

    # --- both_wet: Osher Riemann solver ---
    # Two sub-cases: cell_i < NC and cell_i >= NC
    wet_fwd = both_wet & (cell_i < NC)
    wet_bwd = both_wet & (cell_i >= NC)

    # Forward case (cell_i < NC)
    QR_h_fwd = tl.maximum(ZC - BI, HM1)
    UR_fwd = UC * COSJ + VC * SINJ
    ratio_fwd = tl.minimum(HC / QR_h_fwd, 1.5)
    QR_u_fwd = UR_fwd * ratio_fwd
    # If HC <= HM2 or QR_h <= HM2, clamp
    fwd_shallow = (HC <= HM2) | (QR_h_fwd <= HM2)
    QR_u_fwd = tl.where(fwd_shallow & (UR_fwd >= 0.0), VMIN,
               tl.where(fwd_shallow, -VMIN, QR_u_fwd))
    QR_v_fwd = VC * COSJ - UC * SINJ

    # Call Osher for forward case
    os_f0_fwd, os_f1_fwd, os_f2_fwd, os_f3_fwd = _osher_inline(
        QL_h, QL_u, QL_v, QR_h_fwd, QR_u_fwd, QR_v_fwd, FIL_v, H1)
    f0 = tl.where(wet_fwd, os_f0_fwd, f0)
    f1 = tl.where(wet_fwd, os_f1_fwd + (1.0 - ratio_fwd) * HC * UR_fwd * UR_fwd / 2.0, f1)
    f2 = tl.where(wet_fwd, os_f2_fwd, f2)
    f3 = tl.where(wet_fwd, os_f3_fwd, f3)

    # Backward case (cell_i >= NC)
    COSJ1 = -COSJ
    SINJ1 = -SINJ
    H_NC = tl.load(H_ptr + NC_safe, mask=wet_bwd, other=0.0)
    U_NC = tl.load(U_ptr + NC_safe, mask=wet_bwd, other=0.0)
    V_NC = tl.load(V_ptr + NC_safe, mask=wet_bwd, other=0.0)

    QL1_h = H_NC
    QL1_u = U_NC * COSJ1 + V_NC * SINJ1
    QL1_v = V_NC * COSJ1 - U_NC * SINJ1
    CL1 = tl.sqrt(G * tl.maximum(H_NC, HM1))
    FIL1 = QL1_u + 2.0 * CL1

    HC2 = tl.maximum(HI, HM1)
    ZC1 = tl.maximum(BI, ZI)
    QR1_h = tl.maximum(ZC1 - BC, HM1)
    UR1 = UI * COSJ1 + VI * SINJ1
    ratio1 = tl.minimum(HC2 / QR1_h, 1.5)
    QR1_u = UR1 * ratio1
    bwd_shallow = (HC2 <= HM2) | (QR1_h <= HM2)
    QR1_u = tl.where(bwd_shallow & (UR1 >= 0.0), VMIN,
             tl.where(bwd_shallow, -VMIN, QR1_u))
    QR1_v = VI * COSJ1 - UI * SINJ1

    os_f0_bwd, os_f1_bwd, os_f2_bwd, os_f3_bwd = _osher_inline(
        QL1_h, QL1_u, QL1_v, QR1_h, QR1_u, QR1_v, FIL1, H_NC)

    ZA = tl.sqrt(os_f3_bwd / HALF_G) + BC
    HC3 = tl.maximum(ZA - BI, 0.0)

    f0 = tl.where(wet_bwd, -os_f0_bwd, f0)
    f1 = tl.where(wet_bwd, os_f1_bwd + (1.0 - ratio1) * HC2 * UR1 * UR1 / 2.0, f1)
    f2 = tl.where(wet_bwd, os_f2_bwd, f2)
    f3 = tl.where(wet_bwd, HALF_G * HC3 * HC3, f3)

    # both_dry => all zeros (already initialized)

    # Store results
    tl.store(FLUX0_ptr + idx, f0, mask=mask)
    tl.store(FLUX1_ptr + idx, f1, mask=mask)
    tl.store(FLUX2_ptr + idx, f2, mask=mask)
    tl.store(FLUX3_ptr + idx, f3, mask=mask)


@triton.jit
def _osher_inline(
    QL_h, QL_u, QL_v,
    QR_h, QR_u, QR_v,
    FIL_in, H_pos,
):
    """Osher Riemann solver — vectorized branchless implementation.

    All 16 K1xK2 combinations computed; final result selected via masks.
    Returns (f0, f1, f2, f3).
    """
    CR = tl.sqrt(G * QR_h)
    FIR_v = QR_u - 2.0 * CR
    UA = (FIL_in + FIR_v) / 2.0
    CA = tl.abs((FIL_in - FIR_v) / 4.0)
    CL_v = tl.sqrt(G * H_pos)

    fil = FIL_in
    fir = FIR_v

    # K2 determination
    k2_1 = CA < UA
    k2_2 = (~k2_1) & (UA >= 0.0) & (UA < CA)
    k2_3 = (~k2_1) & (~k2_2) & (UA >= -CA) & (UA < 0.0)
    k2_4 = (~k2_1) & (~k2_2) & (~k2_3)

    # K1 determination
    k1_1 = (QL_u < CL_v) & (QR_u >= -CR)
    k1_2 = (QL_u >= CL_v) & (QR_u >= -CR)
    k1_3 = (QL_u < CL_v) & (QR_u < -CR)
    k1_4 = (QL_u >= CL_v) & (QR_u < -CR)

    # Pre-compute common sub-expressions
    # QF(h, u, v) = (h*u, h*u*u, h*u*v, HALF_G*h*h)
    # For QL
    ql_f0 = QL_h * QL_u
    ql_f1 = ql_f0 * QL_u
    ql_f2 = ql_f0 * QL_v
    ql_f3 = HALF_G * QL_h * QL_h

    # For QR
    qr_f0 = QR_h * QR_u
    qr_f1 = qr_f0 * QR_u
    qr_f2 = qr_f0 * QR_v
    qr_f3 = HALF_G * QR_h * QR_h

    # US = fil/3, HS = US*US/G => QF(HS, US, QL_v or QR_v)
    US_fil = fil / 3.0
    HS_fil = US_fil * US_fil / G
    qf_fil_ql_f0 = HS_fil * US_fil
    qf_fil_ql_f1 = qf_fil_ql_f0 * US_fil
    qf_fil_ql_f2 = qf_fil_ql_f0 * QL_v
    qf_fil_ql_f3 = HALF_G * HS_fil * HS_fil

    US_fir = fir / 3.0
    HS_fir = US_fir * US_fir / G
    qf_fir_qr_f0 = HS_fir * US_fir
    qf_fir_qr_f1 = qf_fir_qr_f0 * US_fir
    qf_fir_qr_f2 = qf_fir_qr_f0 * QR_v
    qf_fir_qr_f3 = HALF_G * HS_fir * HS_fir

    # ua_k2 = (fil + fir) / 2.0
    ua_k2 = (fil + fir) / 2.0
    # For K2==2: fil2 = fil - ua_k2, HA2 = fil2*fil2/(4*G)
    fil2 = fil - ua_k2
    HA2 = fil2 * fil2 / (4.0 * G)
    qf_ha2_ql_f0 = HA2 * ua_k2
    qf_ha2_ql_f1 = qf_ha2_ql_f0 * ua_k2
    qf_ha2_ql_f2 = qf_ha2_ql_f0 * QL_v
    qf_ha2_ql_f3 = HALF_G * HA2 * HA2

    # For K2==3: fir3 = fir - ua_k2, HA3 = fir3*fir3/(4*G)
    fir3 = fir - ua_k2
    HA3 = fir3 * fir3 / (4.0 * G)
    qf_ha3_qr_f0 = HA3 * ua_k2
    qf_ha3_qr_f1 = qf_ha3_qr_f0 * ua_k2
    qf_ha3_qr_f2 = qf_ha3_qr_f0 * QR_v
    qf_ha3_qr_f3 = HALF_G * HA3 * HA3

    # Now compute all 16 K1xK2 results
    zero = tl.zeros_like(QL_h)
    out_f0 = zero
    out_f1 = zero
    out_f2 = zero
    out_f3 = zero

    # === K1==1 ===
    # K1==1, K2==1: QF(HS_fil, US_fil, QL_v)
    m = k1_1 & k2_1
    out_f0 = tl.where(m, qf_fil_ql_f0, out_f0)
    out_f1 = tl.where(m, qf_fil_ql_f1, out_f1)
    out_f2 = tl.where(m, qf_fil_ql_f2, out_f2)
    out_f3 = tl.where(m, qf_fil_ql_f3, out_f3)

    # K1==1, K2==2: QF(HA2, ua_k2, QL_v)
    m = k1_1 & k2_2
    out_f0 = tl.where(m, qf_ha2_ql_f0, out_f0)
    out_f1 = tl.where(m, qf_ha2_ql_f1, out_f1)
    out_f2 = tl.where(m, qf_ha2_ql_f2, out_f2)
    out_f3 = tl.where(m, qf_ha2_ql_f3, out_f3)

    # K1==1, K2==3: QF(HA3, ua_k2, QR_v)
    m = k1_1 & k2_3
    out_f0 = tl.where(m, qf_ha3_qr_f0, out_f0)
    out_f1 = tl.where(m, qf_ha3_qr_f1, out_f1)
    out_f2 = tl.where(m, qf_ha3_qr_f2, out_f2)
    out_f3 = tl.where(m, qf_ha3_qr_f3, out_f3)

    # K1==1, K2==4: QF(HS_fir, US_fir, QR_v)
    m = k1_1 & k2_4
    out_f0 = tl.where(m, qf_fir_qr_f0, out_f0)
    out_f1 = tl.where(m, qf_fir_qr_f1, out_f1)
    out_f2 = tl.where(m, qf_fir_qr_f2, out_f2)
    out_f3 = tl.where(m, qf_fir_qr_f3, out_f3)

    # === K1==2 ===
    # K1==2, K2==1: QF(QL)
    m = k1_2 & k2_1
    out_f0 = tl.where(m, ql_f0, out_f0)
    out_f1 = tl.where(m, ql_f1, out_f1)
    out_f2 = tl.where(m, ql_f2, out_f2)
    out_f3 = tl.where(m, ql_f3, out_f3)

    # K1==2, K2==2: QF(QL) - QF(HS_fil, US_fil, QL_v) + QF(HA2, ua_k2, QL_v)
    m = k1_2 & k2_2
    out_f0 = tl.where(m, ql_f0 - qf_fil_ql_f0 + qf_ha2_ql_f0, out_f0)
    out_f1 = tl.where(m, ql_f1 - qf_fil_ql_f1 + qf_ha2_ql_f1, out_f1)
    out_f2 = tl.where(m, ql_f2 - qf_fil_ql_f2 + qf_ha2_ql_f2, out_f2)
    out_f3 = tl.where(m, ql_f3 - qf_fil_ql_f3 + qf_ha2_ql_f3, out_f3)

    # K1==2, K2==3: QF(QL) - QF(HS_fil, US_fil, QL_v) + QF(HA3, ua_k2, QR_v)
    m = k1_2 & k2_3
    out_f0 = tl.where(m, ql_f0 - qf_fil_ql_f0 + qf_ha3_qr_f0, out_f0)
    out_f1 = tl.where(m, ql_f1 - qf_fil_ql_f1 + qf_ha3_qr_f1, out_f1)
    out_f2 = tl.where(m, ql_f2 - qf_fil_ql_f2 + qf_ha3_qr_f2, out_f2)
    out_f3 = tl.where(m, ql_f3 - qf_fil_ql_f3 + qf_ha3_qr_f3, out_f3)

    # K1==2, K2==4: QF(QL) - QF(HS_fil, US_fil, QL_v) + QF(HS_fir, US_fir, QR_v)
    m = k1_2 & k2_4
    out_f0 = tl.where(m, ql_f0 - qf_fil_ql_f0 + qf_fir_qr_f0, out_f0)
    out_f1 = tl.where(m, ql_f1 - qf_fil_ql_f1 + qf_fir_qr_f1, out_f1)
    out_f2 = tl.where(m, ql_f2 - qf_fil_ql_f2 + qf_fir_qr_f2, out_f2)
    out_f3 = tl.where(m, ql_f3 - qf_fil_ql_f3 + qf_fir_qr_f3, out_f3)

    # === K1==3 ===
    # K1==3, K2==1: QF(HS_fil, US_fil, QL_v) - QF(HS_fir, US_fir, QR_v) + QF(QR)
    m = k1_3 & k2_1
    out_f0 = tl.where(m, qf_fil_ql_f0 - qf_fir_qr_f0 + qr_f0, out_f0)
    out_f1 = tl.where(m, qf_fil_ql_f1 - qf_fir_qr_f1 + qr_f1, out_f1)
    out_f2 = tl.where(m, qf_fil_ql_f2 - qf_fir_qr_f2 + qr_f2, out_f2)
    out_f3 = tl.where(m, qf_fil_ql_f3 - qf_fir_qr_f3 + qr_f3, out_f3)

    # K1==3, K2==2: QF(HA2, ua_k2, QL_v) - QF(HS_fir, US_fir, QR_v) + QF(QR)
    m = k1_3 & k2_2
    out_f0 = tl.where(m, qf_ha2_ql_f0 - qf_fir_qr_f0 + qr_f0, out_f0)
    out_f1 = tl.where(m, qf_ha2_ql_f1 - qf_fir_qr_f1 + qr_f1, out_f1)
    out_f2 = tl.where(m, qf_ha2_ql_f2 - qf_fir_qr_f2 + qr_f2, out_f2)
    out_f3 = tl.where(m, qf_ha2_ql_f3 - qf_fir_qr_f3 + qr_f3, out_f3)

    # K1==3, K2==3: QF(HA3, ua_k2, QR_v) - QF(HS_fir_b, US_fir_b, QR_v) + QF(QR)
    # Note: in Taichi, this uses fir3 for HA, then US6b = fir/3 but fir was modified
    # Actually looking at the Taichi code more carefully:
    # fir was NOT modified in K2==3 path for K1==3. Let me re-check.
    # K1==3, K2==3: ua_ = (fil+fir)/2; fir = fir - ua_; HA = fir*fir/(4*G) => QF(HA, ua_, QR_v)
    #               then US6b = fir/3; HS6b = US6b*US6b/G => -QF(HS6b, US6b, QR_v)
    #               then + QF(QR)
    # But fir was modified! fir = fir - ua_k2. So US6b = (fir - ua_k2)/3 = fir3/3
    # HA = fir3*fir3/(4*G) = HA3
    # And HS6b = fir3*fir3/(9*G)  which is different from HS_fir
    fir3_div3 = fir3 / 3.0
    HS6b = fir3_div3 * fir3_div3 / G
    qf_hs6b_f0 = HS6b * fir3_div3
    qf_hs6b_f1 = qf_hs6b_f0 * fir3_div3
    qf_hs6b_f2 = qf_hs6b_f0 * QR_v
    qf_hs6b_f3 = HALF_G * HS6b * HS6b

    m = k1_3 & k2_3
    out_f0 = tl.where(m, qf_ha3_qr_f0 - qf_hs6b_f0 + qr_f0, out_f0)
    out_f1 = tl.where(m, qf_ha3_qr_f1 - qf_hs6b_f1 + qr_f1, out_f1)
    out_f2 = tl.where(m, qf_ha3_qr_f2 - qf_hs6b_f2 + qr_f2, out_f2)
    out_f3 = tl.where(m, qf_ha3_qr_f3 - qf_hs6b_f3 + qr_f3, out_f3)

    # K1==3, K2==4: QF(QR)
    m = k1_3 & k2_4
    out_f0 = tl.where(m, qr_f0, out_f0)
    out_f1 = tl.where(m, qr_f1, out_f1)
    out_f2 = tl.where(m, qr_f2, out_f2)
    out_f3 = tl.where(m, qr_f3, out_f3)

    # === K1==4 ===
    # K1==4, K2==1: QF(QL) - QF(HS_fir, US_fir, QR_v) + QF(QR)
    m = k1_4 & k2_1
    out_f0 = tl.where(m, ql_f0 - qf_fir_qr_f0 + qr_f0, out_f0)
    out_f1 = tl.where(m, ql_f1 - qf_fir_qr_f1 + qr_f1, out_f1)
    out_f2 = tl.where(m, ql_f2 - qf_fir_qr_f2 + qr_f2, out_f2)
    out_f3 = tl.where(m, ql_f3 - qf_fir_qr_f3 + qr_f3, out_f3)

    # K1==4, K2==2: QF(QL) - QF(HS_fil, US_fil, QL_v) + QF(HA2, ua_k2, QL_v) - QF(HS_fir, US_fir, QR_v) + QF(QR)
    m = k1_4 & k2_2
    out_f0 = tl.where(m, ql_f0 - qf_fil_ql_f0 + qf_ha2_ql_f0 - qf_fir_qr_f0 + qr_f0, out_f0)
    out_f1 = tl.where(m, ql_f1 - qf_fil_ql_f1 + qf_ha2_ql_f1 - qf_fir_qr_f1 + qr_f1, out_f1)
    out_f2 = tl.where(m, ql_f2 - qf_fil_ql_f2 + qf_ha2_ql_f2 - qf_fir_qr_f2 + qr_f2, out_f2)
    out_f3 = tl.where(m, ql_f3 - qf_fil_ql_f3 + qf_ha2_ql_f3 - qf_fir_qr_f3 + qr_f3, out_f3)

    # K1==4, K2==3: QF(QL) - QF(HS_fil, US_fil, QL_v) + QF(HA3, ua_k2, QR_v) - QF(HS_fir, US_fir, QR_v) + QF(QR)
    m = k1_4 & k2_3
    out_f0 = tl.where(m, ql_f0 - qf_fil_ql_f0 + qf_ha3_qr_f0 - qf_fir_qr_f0 + qr_f0, out_f0)
    out_f1 = tl.where(m, ql_f1 - qf_fil_ql_f1 + qf_ha3_qr_f1 - qf_fir_qr_f1 + qr_f1, out_f1)
    out_f2 = tl.where(m, ql_f2 - qf_fil_ql_f2 + qf_ha3_qr_f2 - qf_fir_qr_f2 + qr_f2, out_f2)
    out_f3 = tl.where(m, ql_f3 - qf_fil_ql_f3 + qf_ha3_qr_f3 - qf_fir_qr_f3 + qr_f3, out_f3)

    # K1==4, K2==4: QF(QL) - QF(HS_fil, US_fil, QL_v) + QF(QR)
    m = k1_4 & k2_4
    out_f0 = tl.where(m, ql_f0 - qf_fil_ql_f0 + qr_f0, out_f0)
    out_f1 = tl.where(m, ql_f1 - qf_fil_ql_f1 + qr_f1, out_f1)
    out_f2 = tl.where(m, ql_f2 - qf_fil_ql_f2 + qr_f2, out_f2)
    out_f3 = tl.where(m, ql_f3 - qf_fil_ql_f3 + qr_f3, out_f3)

    return out_f0, out_f1, out_f2, out_f3


# ---------------------------------------------------------------------------
# Kernel 2: update_cell_kernel — 1 thread per cell
# ---------------------------------------------------------------------------
@triton.jit
def update_cell_kernel(
    # Cell arrays (read/write)
    H_ptr, U_ptr, V_ptr, Z_ptr, W_ptr,
    # Cell arrays (read-only)
    ZBC_ptr, AREA_ptr, FNC_ptr,
    # Edge arrays (read-only)
    SIDE_ptr, SLCOS_ptr, SLSIN_ptr,
    FLUX0_ptr, FLUX1_ptr, FLUX2_ptr, FLUX3_ptr,
    # Scalars
    CELL: tl.constexpr, DT: tl.constexpr,
    HM1: tl.constexpr, HM2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = i < CELL

    H1 = tl.load(H_ptr + i, mask=mask, other=0.0)
    U1 = tl.load(U_ptr + i, mask=mask, other=0.0)
    V1 = tl.load(V_ptr + i, mask=mask, other=0.0)
    BI = tl.load(ZBC_ptr + i, mask=mask, other=0.0)

    WH = tl.zeros_like(H1)
    WU = tl.zeros_like(H1)
    WV = tl.zeros_like(H1)

    # Unroll 4 edges per cell
    base_idx = 4 * i

    # Edge 0
    idx0 = base_idx
    SL0 = tl.load(SIDE_ptr + idx0, mask=mask, other=0.0)
    SLCA0 = tl.load(SLCOS_ptr + idx0, mask=mask, other=0.0)
    SLSA0 = tl.load(SLSIN_ptr + idx0, mask=mask, other=0.0)
    FLR1_0 = tl.load(FLUX1_ptr + idx0, mask=mask, other=0.0) + tl.load(FLUX3_ptr + idx0, mask=mask, other=0.0)
    FLR2_0 = tl.load(FLUX2_ptr + idx0, mask=mask, other=0.0)
    WH += SL0 * tl.load(FLUX0_ptr + idx0, mask=mask, other=0.0)
    WU += SLCA0 * FLR1_0 - SLSA0 * FLR2_0
    WV += SLSA0 * FLR1_0 + SLCA0 * FLR2_0

    # Edge 1
    idx1 = base_idx + 1
    SL1 = tl.load(SIDE_ptr + idx1, mask=mask, other=0.0)
    SLCA1 = tl.load(SLCOS_ptr + idx1, mask=mask, other=0.0)
    SLSA1 = tl.load(SLSIN_ptr + idx1, mask=mask, other=0.0)
    FLR1_1 = tl.load(FLUX1_ptr + idx1, mask=mask, other=0.0) + tl.load(FLUX3_ptr + idx1, mask=mask, other=0.0)
    FLR2_1 = tl.load(FLUX2_ptr + idx1, mask=mask, other=0.0)
    WH += SL1 * tl.load(FLUX0_ptr + idx1, mask=mask, other=0.0)
    WU += SLCA1 * FLR1_1 - SLSA1 * FLR2_1
    WV += SLSA1 * FLR1_1 + SLCA1 * FLR2_1

    # Edge 2
    idx2 = base_idx + 2
    SL2 = tl.load(SIDE_ptr + idx2, mask=mask, other=0.0)
    SLCA2 = tl.load(SLCOS_ptr + idx2, mask=mask, other=0.0)
    SLSA2 = tl.load(SLSIN_ptr + idx2, mask=mask, other=0.0)
    FLR1_2 = tl.load(FLUX1_ptr + idx2, mask=mask, other=0.0) + tl.load(FLUX3_ptr + idx2, mask=mask, other=0.0)
    FLR2_2 = tl.load(FLUX2_ptr + idx2, mask=mask, other=0.0)
    WH += SL2 * tl.load(FLUX0_ptr + idx2, mask=mask, other=0.0)
    WU += SLCA2 * FLR1_2 - SLSA2 * FLR2_2
    WV += SLSA2 * FLR1_2 + SLCA2 * FLR2_2

    # Edge 3
    idx3 = base_idx + 3
    SL3 = tl.load(SIDE_ptr + idx3, mask=mask, other=0.0)
    SLCA3 = tl.load(SLCOS_ptr + idx3, mask=mask, other=0.0)
    SLSA3 = tl.load(SLSIN_ptr + idx3, mask=mask, other=0.0)
    FLR1_3 = tl.load(FLUX1_ptr + idx3, mask=mask, other=0.0) + tl.load(FLUX3_ptr + idx3, mask=mask, other=0.0)
    FLR2_3 = tl.load(FLUX2_ptr + idx3, mask=mask, other=0.0)
    WH += SL3 * tl.load(FLUX0_ptr + idx3, mask=mask, other=0.0)
    WU += SLCA3 * FLR1_3 - SLSA3 * FLR2_3
    WV += SLSA3 * FLR1_3 + SLCA3 * FLR2_3

    # Update state
    AREA_i = tl.load(AREA_ptr + i, mask=mask, other=1.0)
    DTA = DT / AREA_i
    H2 = tl.maximum(H1 - DTA * WH, HM1)
    Z2 = H2 + BI

    # Velocity update
    U2 = tl.zeros_like(H1)
    V2 = tl.zeros_like(H1)

    wet = H2 > HM1
    shallow2 = wet & (H2 <= HM2)
    deep = wet & (H2 > HM2)

    # Shallow: clamp velocity
    U2 = tl.where(shallow2 & (U1 >= 0.0), tl.minimum(VMIN, tl.abs(U1)),
         tl.where(shallow2, -tl.minimum(VMIN, tl.abs(U1)), U2))
    V2 = tl.where(shallow2 & (V1 >= 0.0), tl.minimum(VMIN, tl.abs(V1)),
         tl.where(shallow2, -tl.minimum(VMIN, tl.abs(V1)), V2))

    # Deep: momentum update with friction
    QX1 = H1 * U1
    QY1 = H1 * V1
    DTAU = DTA * WU
    DTAV = DTA * WV
    FNC_i = tl.load(FNC_ptr + i, mask=mask, other=0.0)
    WSF = FNC_i * tl.sqrt(U1 * U1 + V1 * V1) / tl.extra.cuda.libdevice.pow(H1, 0.33333)
    U2_deep = (QX1 - DTAU - DT * WSF * U1) / H2
    V2_deep = (QY1 - DTAV - DT * WSF * V1) / H2

    # Clamp deep velocity to 15
    deep_clamp = deep & (H2 > HM2)
    U2_deep = tl.where(deep_clamp & (U2_deep >= 0.0), tl.minimum(tl.abs(U2_deep), 15.0),
              tl.where(deep_clamp, -tl.minimum(tl.abs(U2_deep), 15.0), U2_deep))
    V2_deep = tl.where(deep_clamp & (V2_deep >= 0.0), tl.minimum(tl.abs(V2_deep), 15.0),
              tl.where(deep_clamp, -tl.minimum(tl.abs(V2_deep), 15.0), V2_deep))

    U2 = tl.where(deep, U2_deep, U2)
    V2 = tl.where(deep, V2_deep, V2)

    # Store
    tl.store(H_ptr + i, H2, mask=mask)
    tl.store(U_ptr + i, U2, mask=mask)
    tl.store(V_ptr + i, V2, mask=mask)
    tl.store(Z_ptr + i, Z2, mask=mask)
    tl.store(W_ptr + i, tl.sqrt(U2 * U2 + V2 * V2), mask=mask)


# ---------------------------------------------------------------------------
# run() — matching Taichi/Warp interface
# ---------------------------------------------------------------------------
def run(days=10, backend="cuda", mesh="default"):
    assert backend == "cuda", "Triton only supports CUDA backend"
    device = torch.device("cuda")

    mesh_data = load_mesh(mesh=mesh)

    CELL = mesh_data["CELL"]
    NE = 4 * CELL
    HM1 = float(mesh_data["HM1"])
    HM2 = float(mesh_data["HM2"])
    DT = float(mesh_data["DT"])
    steps_per_day = mesh_data["steps_per_day"]
    total_steps = steps_per_day * days

    # Load cell arrays to GPU
    H = torch.from_numpy(mesh_data["H"]).to(device=device, dtype=torch.float32)
    U = torch.from_numpy(mesh_data["U"]).to(device=device, dtype=torch.float32)
    V = torch.from_numpy(mesh_data["V"]).to(device=device, dtype=torch.float32)
    Z = torch.from_numpy(mesh_data["Z"]).to(device=device, dtype=torch.float32)
    W = torch.from_numpy(mesh_data["W"]).to(device=device, dtype=torch.float32)
    ZBC = torch.from_numpy(mesh_data["ZBC"]).to(device=device, dtype=torch.float32)
    AREA = torch.from_numpy(mesh_data["AREA"]).to(device=device, dtype=torch.float32)
    FNC = torch.from_numpy(mesh_data["FNC"]).to(device=device, dtype=torch.float32)

    # Load edge arrays to GPU
    NAC = torch.from_numpy(mesh_data["NAC"]).to(device=device, dtype=torch.int32)
    KLAS = torch.from_numpy(mesh_data["KLAS"]).to(device=device, dtype=torch.float32)
    SIDE = torch.from_numpy(mesh_data["SIDE"]).to(device=device, dtype=torch.float32)
    COSF = torch.from_numpy(mesh_data["COSF"]).to(device=device, dtype=torch.float32)
    SINF = torch.from_numpy(mesh_data["SINF"]).to(device=device, dtype=torch.float32)
    SLCOS = torch.from_numpy(mesh_data["SLCOS"]).to(device=device, dtype=torch.float32)
    SLSIN = torch.from_numpy(mesh_data["SLSIN"]).to(device=device, dtype=torch.float32)

    # Flux buffers
    FLUX0 = torch.zeros(NE, device=device, dtype=torch.float32)
    FLUX1 = torch.zeros(NE, device=device, dtype=torch.float32)
    FLUX2 = torch.zeros(NE, device=device, dtype=torch.float32)
    FLUX3 = torch.zeros(NE, device=device, dtype=torch.float32)

    # Save initial state for reload after warm-compile
    H_init = H.clone()
    U_init = U.clone()
    V_init = V.clone()
    Z_init = Z.clone()
    W_init = W.clone()

    # Grid sizes
    grid_edges = ((NE + BLOCK_SIZE_FLUX - 1) // BLOCK_SIZE_FLUX,)
    grid_cells = ((CELL + BLOCK_SIZE_CELL - 1) // BLOCK_SIZE_CELL,)

    def step_fn():
        for _ in range(total_steps):
            calculate_flux_kernel[grid_edges](
                H, U, V, Z, ZBC,
                NAC, KLAS, COSF, SINF,
                FLUX0, FLUX1, FLUX2, FLUX3,
                NE, HM1, HM2,
                BLOCK_SIZE=BLOCK_SIZE_FLUX,
            )
            update_cell_kernel[grid_cells](
                H, U, V, Z, W,
                ZBC, AREA, FNC,
                SIDE, SLCOS, SLSIN,
                FLUX0, FLUX1, FLUX2, FLUX3,
                CELL, DT, HM1, HM2,
                BLOCK_SIZE=BLOCK_SIZE_CELL,
            )

    def sync_fn():
        torch.cuda.synchronize()

    # Warm-compile: run one step
    calculate_flux_kernel[grid_edges](
        H, U, V, Z, ZBC,
        NAC, KLAS, COSF, SINF,
        FLUX0, FLUX1, FLUX2, FLUX3,
        NE, HM1, HM2,
        BLOCK_SIZE=BLOCK_SIZE_FLUX,
    )
    update_cell_kernel[grid_cells](
        H, U, V, Z, W,
        ZBC, AREA, FNC,
        SIDE, SLCOS, SLSIN,
        FLUX0, FLUX1, FLUX2, FLUX3,
        CELL, DT, HM1, HM2,
        BLOCK_SIZE=BLOCK_SIZE_CELL,
    )
    torch.cuda.synchronize()

    # Reload initial state after warm-compile
    H.copy_(H_init)
    U.copy_(U_init)
    V.copy_(V_init)
    Z.copy_(Z_init)
    W.copy_(W_init)

    return step_fn, sync_fn, H


if __name__ == "__main__":
    import time
    days = 10
    step_fn, sync_fn, H_tensor = run(days=days, backend="cuda")
    sync_fn()
    t0 = time.perf_counter()
    step_fn()
    sync_fn()
    t1 = time.perf_counter()
    elapsed = (t1 - t0) * 1000
    print(f"{days} days: {elapsed:.1f} ms  ({elapsed/days:.2f} ms/day)")
    print(f"H range: [{H_tensor.min().item():.6f}, {H_tensor.max().item():.6f}]")
