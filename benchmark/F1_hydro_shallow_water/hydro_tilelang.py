"""2D Shallow Water Equations (Osher Riemann solver) — TileLang.

Dam-break on unstructured quad mesh.  Port of hydro-cal calculate_gpu.cu.

TileLang is tile-oriented; this kernel does per-element computation with
indirect neighbor access (gather), so TileLang's tile/shared-memory
abstractions provide no benefit.  We use T.Parallel + direct global access.
The Osher 16-case dispatch is precompute-and-select via T.if_then_else.
"""
import numpy as np
import torch
import tilelang
import tilelang.language as T

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G = 9.81
HALF_G = 4.905
HM1_C = 0.001
HM2_C = 0.01
VMIN_C = 0.001
C1_C = 0.3

BLOCK_SIZE = 256


# ---------------------------------------------------------------------------
# Kernel builder
# ---------------------------------------------------------------------------
def build_swe_kernel(CEL, DT_val):
    """Build and JIT-compile the shallow-water step kernel for a given mesh size."""
    stride = CEL + 1  # row stride in flattened [5][CEL+1] arrays

    # out_idx: positions of output tensors in the argument list
    # Arguments: NAC KLAS SIDE COSF SINF AREA ZBC FNC  H_pre U_pre V_pre Z_pre  H_res U_res V_res Z_res W_res
    #   idx:      0    1    2    3    4    5    6   7    8     9    10    11      12    13    14    15    16
    edge_dim = 5 * stride  # pre-compute outside prim_func

    @tilelang.jit(out_idx=[12, 13, 14, 15, 16])
    def swe_step_kernel(CEL: int = CEL, stride: int = stride,
                        DT: float = DT_val, BLOCK: int = BLOCK_SIZE,
                        edge_dim: int = edge_dim):
        @T.prim_func
        def kernel(
            NAC: T.Tensor((edge_dim,), T.int32),
            KLAS: T.Tensor((edge_dim,), T.int32),
            SIDE: T.Tensor((edge_dim,), T.float32),
            COSF: T.Tensor((edge_dim,), T.float32),
            SINF: T.Tensor((edge_dim,), T.float32),
            AREA: T.Tensor((stride,), T.float32),
            ZBC: T.Tensor((stride,), T.float32),
            FNC: T.Tensor((stride,), T.float32),
            H_pre: T.Tensor((stride,), T.float32),
            U_pre: T.Tensor((stride,), T.float32),
            V_pre: T.Tensor((stride,), T.float32),
            Z_pre: T.Tensor((stride,), T.float32),
            H_res: T.Tensor((stride,), T.float32),
            U_res: T.Tensor((stride,), T.float32),
            V_res: T.Tensor((stride,), T.float32),
            Z_res: T.Tensor((stride,), T.float32),
            W_res: T.Tensor((stride,), T.float32),
        ):
            with T.Kernel(T.ceildiv(CEL, BLOCK), threads=BLOCK) as bx:
                for li in T.Parallel(BLOCK):
                    pos = bx * BLOCK + li + 1  # 1-indexed

                    # --- Load cell state ---
                    H1 = H_pre[pos]
                    U1 = U_pre[pos]
                    V1 = V_pre[pos]
                    BI = ZBC[pos]

                    HI = T.max(H1, T.cast(HM1_C, T.float32))
                    UI = T.if_then_else(HI <= T.cast(HM2_C, T.float32),
                                        T.if_then_else(U1 >= T.cast(0.0, T.float32),
                                                        T.cast(VMIN_C, T.float32),
                                                        T.cast(-VMIN_C, T.float32)),
                                        U1)
                    VI = T.if_then_else(HI <= T.cast(HM2_C, T.float32),
                                        T.if_then_else(V1 >= T.cast(0.0, T.float32),
                                                        T.cast(VMIN_C, T.float32),
                                                        T.cast(-VMIN_C, T.float32)),
                                        V1)
                    ZI = T.max(Z_pre[pos], BI)

                    WH = T.cast(0.0, T.float32)
                    WU = T.cast(0.0, T.float32)
                    WV = T.cast(0.0, T.float32)

                    # --- 4 edges (Python-level unroll) ---
                    for j in range(1, 5):
                        j_off = j * stride
                        NC = NAC[j_off + pos]
                        KP = KLAS[j_off + pos]
                        COSJ = COSF[j_off + pos]
                        SINJ = SINF[j_off + pos]
                        SL = SIDE[j_off + pos]
                        SLCA = SL * COSJ
                        SLSA = SL * SINJ

                        # Left state
                        QL_h = HI
                        QL_u = UI * COSJ + VI * SINJ
                        QL_v = VI * COSJ - UI * SINJ
                        CL_v = T.sqrt(G * HI)
                        FIL_v = QL_u + T.cast(2.0, T.float32) * CL_v

                        # Neighbor state (gather)
                        nc_valid = NC != 0
                        HC = T.if_then_else(nc_valid, T.max(H_pre[T.max(NC, 0)], T.cast(HM1_C, T.float32)), T.cast(0.0, T.float32))
                        BC = T.if_then_else(nc_valid, ZBC[T.max(NC, 0)], T.cast(0.0, T.float32))
                        ZC_raw = T.if_then_else(nc_valid, Z_pre[T.max(NC, 0)], T.cast(0.0, T.float32))
                        ZC = T.if_then_else(nc_valid, T.max(BC, ZC_raw), T.cast(0.0, T.float32))
                        UC = T.if_then_else(nc_valid, U_pre[T.max(NC, 0)], T.cast(0.0, T.float32))
                        VC = T.if_then_else(nc_valid, V_pre[T.max(NC, 0)], T.cast(0.0, T.float32))

                        # ---- Flux: default zero ----
                        f0 = T.cast(0.0, T.float32)
                        f1 = T.cast(0.0, T.float32)
                        f2 = T.cast(0.0, T.float32)
                        f3 = T.cast(0.0, T.float32)

                        is_bnd = KP != 0
                        both_dry = (HI <= T.cast(HM1_C, T.float32)) & (HC <= T.cast(HM1_C, T.float32))

                        # Boundary: wall pressure
                        wall_f3 = T.cast(HALF_G, T.float32) * H1 * H1

                        # ZI <= BC
                        zi_bc_f0 = T.cast(-C1_C, T.float32) * T.pow(HC, T.cast(1.5, T.float32))
                        zi_bc_f1 = HI * QL_u * T.abs(QL_u)
                        zi_bc_f3 = T.cast(HALF_G, T.float32) * HI * HI

                        # ZC <= BI
                        zc_bi_f0 = T.cast(C1_C, T.float32) * T.pow(HI, T.cast(1.5, T.float32))
                        zc_bi_f1 = HI * T.abs(QL_u) * QL_u
                        zc_bi_f2 = HI * T.abs(QL_u) * QL_v

                        # HI shallow, ZC>ZI
                        DH_a = T.max(ZC - ZBC[pos], T.cast(HM1_C, T.float32))
                        UN_a = T.cast(-C1_C, T.float32) * T.sqrt(DH_a)
                        hish_gt_f0 = DH_a * UN_a
                        hish_gt_f1 = hish_gt_f0 * UN_a
                        hish_gt_f2 = hish_gt_f0 * (VC * COSJ - UC * SINJ)
                        hish_gt_f3 = T.cast(HALF_G, T.float32) * HI * HI

                        # HI shallow, ZC<=ZI
                        hish_le_f0 = T.cast(C1_C, T.float32) * T.pow(HI, T.cast(1.5, T.float32))

                        # HC shallow, ZI>ZC
                        DH_b = T.max(ZI - BC, T.cast(HM1_C, T.float32))
                        UN_b = T.cast(C1_C, T.float32) * T.sqrt(DH_b)
                        HC1_b = ZC - ZBC[pos]
                        hcsh_gt_f0 = DH_b * UN_b
                        hcsh_gt_f1 = hcsh_gt_f0 * UN_b
                        hcsh_gt_f2 = hcsh_gt_f0 * QL_v
                        hcsh_gt_f3 = T.cast(HALF_G, T.float32) * HC1_b * HC1_b

                        # HC shallow, ZI<=ZC
                        hcsh_le_f0 = T.cast(-C1_C, T.float32) * T.pow(HC, T.cast(1.5, T.float32))
                        hcsh_le_f1 = HI * QL_u * QL_u

                        # ---- OSHER solver (precompute all template QFs) ----
                        # Riemann invariants
                        CR = T.sqrt(T.cast(G, T.float32) * T.max(HC, T.cast(HM1_C, T.float32)))
                        FIR = QL_u - T.cast(2.0, T.float32) * CR  # use QR_u approx
                        # For proper OSHER we need QR states
                        QR_h = T.max(ZC - ZBC[pos], T.cast(HM1_C, T.float32))
                        UR = UC * COSJ + VC * SINJ
                        ratio = T.min(HC / T.max(QR_h, T.cast(HM1_C, T.float32)), T.cast(1.5, T.float32))
                        QR_u = T.if_then_else((HC <= T.cast(HM2_C, T.float32)) | (QR_h <= T.cast(HM2_C, T.float32)),
                                              T.if_then_else(UR >= T.cast(0.0, T.float32), T.cast(VMIN_C, T.float32), T.cast(-VMIN_C, T.float32)),
                                              UR * ratio)
                        QR_v = VC * COSJ - UC * SINJ

                        CR_os = T.sqrt(T.cast(G, T.float32) * QR_h)
                        FIR_os = QR_u - T.cast(2.0, T.float32) * CR_os
                        fil = FIL_v
                        fir = FIR_os
                        UA = (fil + fir) * T.cast(0.5, T.float32)
                        CA = T.abs((fil - fir) * T.cast(0.25, T.float32))

                        # K1, K2
                        K2 = T.if_then_else(CA < UA, 1,
                             T.if_then_else((UA >= T.cast(0.0, T.float32)) & (UA < CA), 2,
                             T.if_then_else((UA >= -CA) & (UA < T.cast(0.0, T.float32)), 3, 4)))
                        K1 = T.if_then_else((QL_u < CL_v) & (QR_u >= -CR_os), 1,
                             T.if_then_else((QL_u >= CL_v) & (QR_u >= -CR_os), 2,
                             T.if_then_else((QL_u < CL_v) & (QR_u < -CR_os), 3, 4)))

                        # QF helper inline: QF(h,u,v) = (h*u, h*u*u, h*u*v, 4.905*h*h)
                        # T1: left state
                        t1_0 = QL_h * QL_u; t1_1 = t1_0 * QL_u; t1_2 = t1_0 * QL_v; t1_3 = T.cast(HALF_G, T.float32) * QL_h * QL_h
                        # T2: sonic FIL
                        us2 = fil / T.cast(3.0, T.float32); hs2 = us2 * us2 / T.cast(G, T.float32)
                        t2_0 = hs2 * us2; t2_1 = t2_0 * us2; t2_2 = t2_0 * QL_v; t2_3 = T.cast(HALF_G, T.float32) * hs2 * hs2
                        # T3: intermediate (fil modified)
                        ua3 = (fil + fir) * T.cast(0.5, T.float32); fil3 = fil - ua3
                        ha3 = fil3 * fil3 / T.cast(4.0 * G, T.float32)
                        t3_0 = ha3 * ua3; t3_1 = t3_0 * ua3; t3_2 = t3_0 * QL_v; t3_3 = T.cast(HALF_G, T.float32) * ha3 * ha3
                        # T5: intermediate (fir modified)
                        ua5 = (fil + fir) * T.cast(0.5, T.float32); fir5 = fir - ua5
                        ha5 = fir5 * fir5 / T.cast(4.0 * G, T.float32)
                        t5_0 = ha5 * ua5; t5_1 = t5_0 * ua5; t5_2 = t5_0 * QR_v; t5_3 = T.cast(HALF_G, T.float32) * ha5 * ha5
                        # T6: sonic FIR (original)
                        us6 = fir / T.cast(3.0, T.float32); hs6 = us6 * us6 / T.cast(G, T.float32)
                        t6_0 = hs6 * us6; t6_1 = t6_0 * us6; t6_2 = t6_0 * QR_v; t6_3 = T.cast(HALF_G, T.float32) * hs6 * hs6
                        # T6m: sonic FIR after T5 (modified fir)
                        us6m = fir5 / T.cast(3.0, T.float32); hs6m = us6m * us6m / T.cast(G, T.float32)
                        t6m_0 = hs6m * us6m; t6m_1 = t6m_0 * us6m; t6m_2 = t6m_0 * QR_v; t6m_3 = T.cast(HALF_G, T.float32) * hs6m * hs6m
                        # T7: right state
                        t7_0 = QR_h * QR_u; t7_1 = t7_0 * QR_u; t7_2 = t7_0 * QR_v; t7_3 = T.cast(HALF_G, T.float32) * QR_h * QR_h

                        # 16-case linear combinations
                        case = K1 * 10 + K2
                        zero = T.cast(0.0, T.float32)

                        def sel4(c, r11, r12, r13, r14, r21, r22, r23, r24,
                                 r31, r32, r33, r34, r41, r42, r43, r44):
                            return T.if_then_else(c == 11, r11,
                                   T.if_then_else(c == 12, r12,
                                   T.if_then_else(c == 13, r13,
                                   T.if_then_else(c == 14, r14,
                                   T.if_then_else(c == 21, r21,
                                   T.if_then_else(c == 22, r22,
                                   T.if_then_else(c == 23, r23,
                                   T.if_then_else(c == 24, r24,
                                   T.if_then_else(c == 31, r31,
                                   T.if_then_else(c == 32, r32,
                                   T.if_then_else(c == 33, r33,
                                   T.if_then_else(c == 34, r34,
                                   T.if_then_else(c == 41, r41,
                                   T.if_then_else(c == 42, r42,
                                   T.if_then_else(c == 43, r43,
                                   T.if_then_else(c == 44, r44, zero))))))))))))))))

                        os_f0 = sel4(case,
                            t2_0, t3_0, t5_0, t6_0,                                         # K1=1
                            t1_0, t1_0-t2_0+t3_0, t1_0-t2_0+t5_0, t1_0-t2_0+t6_0,         # K1=2
                            t2_0-t6_0+t7_0, t3_0-t6_0+t7_0, t5_0-t6m_0+t7_0, t7_0,        # K1=3
                            t1_0-t6_0+t7_0, t1_0-t2_0+t3_0-t6_0+t7_0, t1_0-t2_0+t5_0-t6m_0+t7_0, t1_0-t2_0+t7_0)
                        os_f1 = sel4(case,
                            t2_1, t3_1, t5_1, t6_1,
                            t1_1, t1_1-t2_1+t3_1, t1_1-t2_1+t5_1, t1_1-t2_1+t6_1,
                            t2_1-t6_1+t7_1, t3_1-t6_1+t7_1, t5_1-t6m_1+t7_1, t7_1,
                            t1_1-t6_1+t7_1, t1_1-t2_1+t3_1-t6_1+t7_1, t1_1-t2_1+t5_1-t6m_1+t7_1, t1_1-t2_1+t7_1)
                        os_f2 = sel4(case,
                            t2_2, t3_2, t5_2, t6_2,
                            t1_2, t1_2-t2_2+t3_2, t1_2-t2_2+t5_2, t1_2-t2_2+t6_2,
                            t2_2-t6_2+t7_2, t3_2-t6_2+t7_2, t5_2-t6m_2+t7_2, t7_2,
                            t1_2-t6_2+t7_2, t1_2-t2_2+t3_2-t6_2+t7_2, t1_2-t2_2+t5_2-t6m_2+t7_2, t1_2-t2_2+t7_2)
                        os_f3 = sel4(case,
                            t2_3, t3_3, t5_3, t6_3,
                            t1_3, t1_3-t2_3+t3_3, t1_3-t2_3+t5_3, t1_3-t2_3+t6_3,
                            t2_3-t6_3+t7_3, t3_3-t6_3+t7_3, t5_3-t6m_3+t7_3, t7_3,
                            t1_3-t6_3+t7_3, t1_3-t2_3+t3_3-t6_3+t7_3, t1_3-t2_3+t5_3-t6m_3+t7_3, t1_3-t2_3+t7_3)

                        # Momentum correction for OSHER
                        os_f1_adj = os_f1 + (T.cast(1.0, T.float32) - ratio) * HC * UR * UR * T.cast(0.5, T.float32)

                        # ---- Mirror OSHER (pos >= NC) ----
                        COSJ1 = -COSJ; SINJ1 = -SINJ
                        H_NC = T.if_then_else(nc_valid, H_pre[T.max(NC, 0)], T.cast(HM1_C, T.float32))
                        U_NC = T.if_then_else(nc_valid, U_pre[T.max(NC, 0)], T.cast(0.0, T.float32))
                        V_NC = T.if_then_else(nc_valid, V_pre[T.max(NC, 0)], T.cast(0.0, T.float32))
                        QL1_h = H_NC
                        QL1_u = U_NC * COSJ1 + V_NC * SINJ1
                        QL1_v = V_NC * COSJ1 - U_NC * SINJ1
                        CL1 = T.sqrt(T.cast(G, T.float32) * H_NC)
                        FIL1 = QL1_u + T.cast(2.0, T.float32) * CL1
                        HC2 = T.max(HI, T.cast(HM1_C, T.float32))
                        ZC1 = T.max(ZBC[pos], ZI)
                        ZBC_NC = T.if_then_else(nc_valid, ZBC[T.max(NC, 0)], T.cast(0.0, T.float32))
                        QR1_h = T.max(ZC1 - ZBC_NC, T.cast(HM1_C, T.float32))
                        UR1 = UI * COSJ1 + VI * SINJ1
                        ratio1 = T.min(HC2 / T.max(QR1_h, T.cast(HM1_C, T.float32)), T.cast(1.5, T.float32))
                        QR1_u = T.if_then_else((HC2 <= T.cast(HM2_C, T.float32)) | (QR1_h <= T.cast(HM2_C, T.float32)),
                                               T.if_then_else(UR1 >= T.cast(0.0, T.float32), T.cast(VMIN_C, T.float32), T.cast(-VMIN_C, T.float32)),
                                               UR1 * ratio1)
                        QR1_v = VI * COSJ1 - UI * SINJ1

                        # Mirror OSHER QF templates
                        CR1_os = T.sqrt(T.cast(G, T.float32) * QR1_h)
                        FIR1_os = QR1_u - T.cast(2.0, T.float32) * CR1_os
                        mfil = FIL1; mfir = FIR1_os
                        mUA = (mfil + mfir) * T.cast(0.5, T.float32)
                        mCA = T.abs((mfil - mfir) * T.cast(0.25, T.float32))
                        mK2 = T.if_then_else(mCA < mUA, 1,
                              T.if_then_else((mUA >= T.cast(0.0, T.float32)) & (mUA < mCA), 2,
                              T.if_then_else((mUA >= -mCA) & (mUA < T.cast(0.0, T.float32)), 3, 4)))
                        mCL = T.sqrt(T.cast(G, T.float32) * H_NC)
                        mK1 = T.if_then_else((QL1_u < mCL) & (QR1_u >= -CR1_os), 1,
                              T.if_then_else((QL1_u >= mCL) & (QR1_u >= -CR1_os), 2,
                              T.if_then_else((QL1_u < mCL) & (QR1_u < -CR1_os), 3, 4)))

                        # Mirror templates
                        m1_0 = QL1_h*QL1_u; m1_1 = m1_0*QL1_u; m1_2 = m1_0*QL1_v; m1_3 = T.cast(HALF_G,T.float32)*QL1_h*QL1_h
                        mus2 = mfil/T.cast(3.0,T.float32); mhs2 = mus2*mus2/T.cast(G,T.float32)
                        m2_0 = mhs2*mus2; m2_1 = m2_0*mus2; m2_2 = m2_0*QL1_v; m2_3 = T.cast(HALF_G,T.float32)*mhs2*mhs2
                        mua3 = (mfil+mfir)*T.cast(0.5,T.float32); mfil3 = mfil-mua3
                        mha3 = mfil3*mfil3/T.cast(4.0*G,T.float32)
                        m3_0 = mha3*mua3; m3_1 = m3_0*mua3; m3_2 = m3_0*QL1_v; m3_3 = T.cast(HALF_G,T.float32)*mha3*mha3
                        mua5 = (mfil+mfir)*T.cast(0.5,T.float32); mfir5 = mfir-mua5
                        mha5 = mfir5*mfir5/T.cast(4.0*G,T.float32)
                        m5_0 = mha5*mua5; m5_1 = m5_0*mua5; m5_2 = m5_0*QR1_v; m5_3 = T.cast(HALF_G,T.float32)*mha5*mha5
                        mus6 = mfir/T.cast(3.0,T.float32); mhs6 = mus6*mus6/T.cast(G,T.float32)
                        m6_0 = mhs6*mus6; m6_1 = m6_0*mus6; m6_2 = m6_0*QR1_v; m6_3 = T.cast(HALF_G,T.float32)*mhs6*mhs6
                        mus6m = mfir5/T.cast(3.0,T.float32); mhs6m = mus6m*mus6m/T.cast(G,T.float32)
                        m6m_0 = mhs6m*mus6m; m6m_1 = m6m_0*mus6m; m6m_2 = m6m_0*QR1_v; m6m_3 = T.cast(HALF_G,T.float32)*mhs6m*mhs6m
                        m7_0 = QR1_h*QR1_u; m7_1 = m7_0*QR1_u; m7_2 = m7_0*QR1_v; m7_3 = T.cast(HALF_G,T.float32)*QR1_h*QR1_h

                        mcase = mK1*10+mK2
                        mr0 = sel4(mcase, m2_0,m3_0,m5_0,m6_0, m1_0,m1_0-m2_0+m3_0,m1_0-m2_0+m5_0,m1_0-m2_0+m6_0,
                                   m2_0-m6_0+m7_0,m3_0-m6_0+m7_0,m5_0-m6m_0+m7_0,m7_0,
                                   m1_0-m6_0+m7_0,m1_0-m2_0+m3_0-m6_0+m7_0,m1_0-m2_0+m5_0-m6m_0+m7_0,m1_0-m2_0+m7_0)
                        mr1 = sel4(mcase, m2_1,m3_1,m5_1,m6_1, m1_1,m1_1-m2_1+m3_1,m1_1-m2_1+m5_1,m1_1-m2_1+m6_1,
                                   m2_1-m6_1+m7_1,m3_1-m6_1+m7_1,m5_1-m6m_1+m7_1,m7_1,
                                   m1_1-m6_1+m7_1,m1_1-m2_1+m3_1-m6_1+m7_1,m1_1-m2_1+m5_1-m6m_1+m7_1,m1_1-m2_1+m7_1)
                        mr1_adj = mr1 + (T.cast(1.0,T.float32) - ratio1) * HC2 * UR1 * UR1 * T.cast(0.5,T.float32)
                        mr3 = sel4(mcase, m2_3,m3_3,m5_3,m6_3, m1_3,m1_3-m2_3+m3_3,m1_3-m2_3+m5_3,m1_3-m2_3+m6_3,
                                   m2_3-m6_3+m7_3,m3_3-m6_3+m7_3,m5_3-m6m_3+m7_3,m7_3,
                                   m1_3-m6_3+m7_3,m1_3-m2_3+m3_3-m6_3+m7_3,m1_3-m2_3+m5_3-m6m_3+m7_3,m1_3-m2_3+m7_3)
                        ZA = T.sqrt(mr3 / T.cast(HALF_G, T.float32)) + BC
                        HC3 = T.max(ZA - ZBC[pos], T.cast(0.0, T.float32))
                        mr2 = sel4(mcase, m2_2,m3_2,m5_2,m6_2, m1_2,m1_2-m2_2+m3_2,m1_2-m2_2+m5_2,m1_2-m2_2+m6_2,
                                   m2_2-m6_2+m7_2,m3_2-m6_2+m7_2,m5_2-m6m_2+m7_2,m7_2,
                                   m1_2-m6_2+m7_2,m1_2-m2_2+m3_2-m6_2+m7_2,m1_2-m2_2+m5_2-m6m_2+m7_2,m1_2-m2_2+m7_2)

                        # ---- Select flux by case priority ----
                        is_interior = ~is_bnd & ~both_dry
                        zi_le_bc = is_interior & (ZI <= BC)
                        zc_le_bi = is_interior & ~zi_le_bc & (ZC <= BI)
                        hi_sh = is_interior & ~zi_le_bc & ~zc_le_bi & (HI <= T.cast(HM2_C, T.float32))
                        hc_sh = is_interior & ~zi_le_bc & ~zc_le_bi & ~hi_sh & (HC <= T.cast(HM2_C, T.float32))
                        both_wet = is_interior & ~zi_le_bc & ~zc_le_bi & ~hi_sh & ~hc_sh
                        pos_lt = both_wet & (pos < NC)
                        pos_ge = both_wet & ~pos_lt

                        # Build flux bottom-up (last matching wins)
                        # Mirror OSHER (pos >= NC)
                        f0 = T.if_then_else(pos_ge, -mr0, f0)
                        f1 = T.if_then_else(pos_ge, mr1_adj, f1)
                        f2 = T.if_then_else(pos_ge, mr2, f2)
                        f3 = T.if_then_else(pos_ge, T.cast(HALF_G,T.float32)*HC3*HC3, f3)
                        # Direct OSHER (pos < NC)
                        f0 = T.if_then_else(pos_lt, os_f0, f0)
                        f1 = T.if_then_else(pos_lt, os_f1_adj, f1)
                        f2 = T.if_then_else(pos_lt, os_f2, f2)
                        f3 = T.if_then_else(pos_lt, os_f3, f3)
                        # HC shallow
                        f0 = T.if_then_else(hc_sh & (ZI > ZC), hcsh_gt_f0, T.if_then_else(hc_sh, hcsh_le_f0, f0))
                        f1 = T.if_then_else(hc_sh & (ZI > ZC), hcsh_gt_f1, T.if_then_else(hc_sh, hcsh_le_f1, f1))
                        f2 = T.if_then_else(hc_sh & (ZI > ZC), hcsh_gt_f2, f2)
                        f3 = T.if_then_else(hc_sh & (ZI > ZC), hcsh_gt_f3, T.if_then_else(hc_sh, zi_bc_f3, f3))
                        # HI shallow
                        f0 = T.if_then_else(hi_sh & (ZC > ZI), hish_gt_f0, T.if_then_else(hi_sh, hish_le_f0, f0))
                        f1 = T.if_then_else(hi_sh & (ZC > ZI), hish_gt_f1, f1)
                        f2 = T.if_then_else(hi_sh & (ZC > ZI), hish_gt_f2, f2)
                        f3 = T.if_then_else(hi_sh, hish_gt_f3, f3)
                        # ZC <= BI
                        f0 = T.if_then_else(zc_le_bi, zc_bi_f0, f0)
                        f1 = T.if_then_else(zc_le_bi, zc_bi_f1, f1)
                        f2 = T.if_then_else(zc_le_bi, zc_bi_f2, f2)
                        # ZI <= BC
                        f0 = T.if_then_else(zi_le_bc, zi_bc_f0, f0)
                        f1 = T.if_then_else(zi_le_bc, zi_bc_f1, f1)
                        f3 = T.if_then_else(zi_le_bc, zi_bc_f3, f3)
                        # Boundary
                        f3 = T.if_then_else(is_bnd, wall_f3, f3)
                        f0 = T.if_then_else(is_bnd, zero, f0)
                        f1 = T.if_then_else(is_bnd, zero, f1)
                        f2 = T.if_then_else(is_bnd, zero, f2)

                        # ---- Accumulate fluxes ----
                        FLR_1 = f1 + f3
                        FLR_2 = f2
                        WH = WH + SL * f0
                        WU = WU + SLCA * FLR_1 - SLSA * FLR_2
                        WV = WV + SLSA * FLR_1 + SLCA * FLR_2

                    # ---- State update ----
                    DTA = T.cast(DT, T.float32) / AREA[pos]
                    WDTA = DTA
                    H2 = T.max(H1 - WDTA * WH, T.cast(HM1_C, T.float32))
                    Z2 = H2 + BI

                    is_dry = H2 <= T.cast(HM1_C, T.float32)
                    is_shallow = (~is_dry) & (H2 <= T.cast(HM2_C, T.float32))
                    is_wet = (~is_dry) & (~is_shallow)

                    speed = T.sqrt(U1 * U1 + V1 * V1)
                    WSF = FNC[pos] * speed / T.pow(T.max(H1, T.cast(HM1_C, T.float32)), T.cast(0.33333, T.float32))
                    QX1 = H1 * U1; QY1 = H1 * V1
                    U2_wet = (QX1 - WDTA * WU - T.cast(DT, T.float32) * WSF * U1) / T.max(H2, T.cast(HM1_C, T.float32))
                    V2_wet = (QY1 - WDTA * WV - T.cast(DT, T.float32) * WSF * V1) / T.max(H2, T.cast(HM1_C, T.float32))
                    # Cap at 15
                    U2_wet = T.if_then_else(U2_wet >= T.cast(0.0, T.float32), T.min(U2_wet, T.cast(15.0, T.float32)), T.max(U2_wet, T.cast(-15.0, T.float32)))
                    V2_wet = T.if_then_else(V2_wet >= T.cast(0.0, T.float32), T.min(V2_wet, T.cast(15.0, T.float32)), T.max(V2_wet, T.cast(-15.0, T.float32)))

                    U2_sh = T.if_then_else(U1 >= T.cast(0.0, T.float32),
                                           T.min(T.cast(VMIN_C, T.float32), T.abs(U1)),
                                           -T.min(T.cast(VMIN_C, T.float32), T.abs(U1)))
                    V2_sh = T.if_then_else(V1 >= T.cast(0.0, T.float32),
                                           T.min(T.cast(VMIN_C, T.float32), T.abs(V1)),
                                           -T.min(T.cast(VMIN_C, T.float32), T.abs(V1)))

                    U2 = T.if_then_else(is_wet, U2_wet, T.if_then_else(is_shallow, U2_sh, zero))
                    V2 = T.if_then_else(is_wet, V2_wet, T.if_then_else(is_shallow, V2_sh, zero))

                    H_res[pos] = H2
                    U_res[pos] = U2
                    V_res[pos] = V2
                    Z_res[pos] = Z2
                    W_res[pos] = T.sqrt(U2 * U2 + V2 * V2)

        return kernel

    return swe_step_kernel


# ---------------------------------------------------------------------------
# Transfer kernel (copy res → pre)
# ---------------------------------------------------------------------------
def build_transfer_kernel(CEL):
    stride = CEL + 1

    @tilelang.jit(out_idx=[0, 1, 2, 3, 4])
    def transfer(CEL: int = CEL, BLOCK: int = BLOCK_SIZE):
        @T.prim_func
        def kernel(
            H_pre: T.Tensor((stride,), T.float32),
            U_pre: T.Tensor((stride,), T.float32),
            V_pre: T.Tensor((stride,), T.float32),
            Z_pre: T.Tensor((stride,), T.float32),
            W_pre: T.Tensor((stride,), T.float32),
            H_res: T.Tensor((stride,), T.float32),
            U_res: T.Tensor((stride,), T.float32),
            V_res: T.Tensor((stride,), T.float32),
            Z_res: T.Tensor((stride,), T.float32),
            W_res: T.Tensor((stride,), T.float32),
        ):
            with T.Kernel(T.ceildiv(CEL, BLOCK), threads=BLOCK) as bx:
                for li in T.Parallel(BLOCK):
                    pos = bx * BLOCK + li + 1
                    H_pre[pos] = H_res[pos]
                    U_pre[pos] = U_res[pos]
                    V_pre[pos] = V_res[pos]
                    Z_pre[pos] = Z_res[pos]
                    W_pre[pos] = W_res[pos]
        return kernel

    return transfer


# ---------------------------------------------------------------------------
# Benchmark interface
# ---------------------------------------------------------------------------
def run(N, steps=1, backend="cuda"):
    assert backend == "cuda", "TileLang requires CUDA"
    CEL = N * N
    dx = 1.0
    DT = float(0.5 * dx / (np.sqrt(G * 2.0) + 1e-6))
    stride = CEL + 1

    # Build mesh on host (same as other implementations)
    nac_np = np.zeros((5, stride), dtype=np.int32)
    klas_np = np.zeros((5, stride), dtype=np.int32)
    side_np = np.zeros((5, stride), dtype=np.float32)
    cosf_np = np.zeros((5, stride), dtype=np.float32)
    sinf_np = np.zeros((5, stride), dtype=np.float32)
    area_np = np.zeros(stride, dtype=np.float32)
    zbc_np = np.zeros(stride, dtype=np.float32)
    fnc_np = np.full(stride, G * 0.03 * 0.03, dtype=np.float32)

    edge_cos = [0.0, 0.0, 1.0, 0.0, -1.0]
    edge_sin = [0.0, -1.0, 0.0, 1.0, 0.0]

    h_np = np.full(stride, HM1_C, dtype=np.float32)
    z_np = np.zeros(stride, dtype=np.float32)

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

    dev = "cuda"
    NAC = torch.from_numpy(nac_np.ravel()).to(dev)
    KLAS = torch.from_numpy(klas_np.ravel()).to(dev)
    SIDE_t = torch.from_numpy(side_np.ravel()).to(dtype=torch.float32, device=dev)
    COSF_t = torch.from_numpy(cosf_np.ravel()).to(dtype=torch.float32, device=dev)
    SINF_t = torch.from_numpy(sinf_np.ravel()).to(dtype=torch.float32, device=dev)
    AREA_t = torch.from_numpy(area_np).to(dtype=torch.float32, device=dev)
    ZBC_t = torch.from_numpy(zbc_np).to(dtype=torch.float32, device=dev)
    FNC_t = torch.from_numpy(fnc_np).to(dtype=torch.float32, device=dev)

    H_pre = torch.from_numpy(h_np).to(dtype=torch.float32, device=dev)
    U_pre = torch.zeros(stride, dtype=torch.float32, device=dev)
    V_pre = torch.zeros(stride, dtype=torch.float32, device=dev)
    Z_pre = torch.from_numpy(z_np).to(dtype=torch.float32, device=dev)
    W_pre = torch.zeros(stride, dtype=torch.float32, device=dev)
    H_res = torch.zeros_like(H_pre)
    U_res = torch.zeros_like(U_pre)
    V_res = torch.zeros_like(V_pre)
    Z_res = torch.zeros_like(Z_pre)
    W_res = torch.zeros_like(W_pre)

    # Compile kernels
    swe_kernel = build_swe_kernel(CEL, DT)
    xfer_kernel = build_transfer_kernel(CEL)

    def step():
        for _ in range(steps):
            swe_kernel(NAC, KLAS, SIDE_t, COSF_t, SINF_t, AREA_t, ZBC_t, FNC_t,
                       H_pre, U_pre, V_pre, Z_pre,
                       H_res, U_res, V_res, Z_res, W_res)
            xfer_kernel(H_pre, U_pre, V_pre, Z_pre, W_pre,
                        H_res, U_res, V_res, Z_res, W_res)

    def sync():
        torch.cuda.synchronize()

    return step, sync, H_pre
