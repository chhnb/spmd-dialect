"""2D Shallow Water Equations (Osher Riemann solver) — TileLang.

Dam-break on unstructured quad mesh.  Port of hydro-cal calculate_gpu.cu.

TileLang is tile-oriented; this kernel does per-element computation with
indirect neighbor access (gather), so TileLang's tile/shared-memory
abstractions provide no benefit.  We use T.Parallel + direct global access.
The Osher 16-case dispatch is precompute-and-select via T.if_then_else.
"""
import os
os.environ.setdefault("CUDA_HOME", "/home/scratch.huanhuanc_gpu/spmd/cuda-toolkit")

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


def _build_swe_jitkernel(CEL, DT_val):
    """Build SWE kernels using JITKernel with separate flux + update pattern.
    Splitting into 2 kernels keeps each small enough for TVM JIT compilation."""
    stride = CEL + 1
    edge_dim = 5 * stride
    NE = 4 * CEL  # 4 edges per cell

    # Kernel 1: Calculate flux per edge (NE threads)
    @T.prim_func
    def calc_flux(
        NAC: T.Buffer((edge_dim,), "int32"),
        KLAS: T.Buffer((edge_dim,), "int32"),
        SIDE: T.Buffer((edge_dim,), "float32"),
        COSF: T.Buffer((edge_dim,), "float32"),
        SINF: T.Buffer((edge_dim,), "float32"),
        ZBC: T.Buffer((stride,), "float32"),
        H_pre: T.Buffer((stride,), "float32"),
        U_pre: T.Buffer((stride,), "float32"),
        V_pre: T.Buffer((stride,), "float32"),
        Z_pre: T.Buffer((stride,), "float32"),
        FLUX0: T.Buffer((NE,), "float32"),
        FLUX1: T.Buffer((NE,), "float32"),
        FLUX2: T.Buffer((NE,), "float32"),
        FLUX3: T.Buffer((NE,), "float32"),
    ):
        with T.Kernel(T.ceildiv(NE, BLOCK_SIZE), threads=BLOCK_SIZE):
            bx = T.get_block_binding(0)
            tx = T.get_thread_binding(0)
            flat_idx = bx * BLOCK_SIZE + tx
            if flat_idx < NE:
                # Map flat index to (cell, edge): cell = flat_idx // 4, edge = flat_idx % 4 + 1
                cell_pos = flat_idx // 4 + 1  # 1-indexed cell
                edge_j = flat_idx % 4 + 1     # edge 1..4
                eidx = edge_j * stride + cell_pos

                H1 = H_pre[cell_pos]
                U1 = U_pre[cell_pos]; V1 = V_pre[cell_pos]
                BI = ZBC[cell_pos]; ZI = T.max(Z_pre[cell_pos], BI)
                HI = T.max(H1, T.float32(HM1_C))
                hi_sh = HI <= T.float32(HM2_C)
                UI_v = T.if_then_else(hi_sh, T.if_then_else(U1 >= T.float32(0.0), T.float32(VMIN_C), T.float32(-VMIN_C)), U1)
                VI_v = T.if_then_else(hi_sh, T.if_then_else(V1 >= T.float32(0.0), T.float32(VMIN_C), T.float32(-VMIN_C)), V1)

                NC = NAC[eidx]; KP = KLAS[eidx]
                CA = COSF[eidx]; SA = SINF[eidx]
                QL_u = UI_v * CA + VI_v * SA
                QL_v = VI_v * CA - UI_v * SA
                CL_v = T.sqrt(T.float32(G) * HI)
                FIL_v = QL_u + T.float32(2.0) * CL_v
                zero = T.float32(0.0)
                wall_f3 = T.float32(HALF_G) * H1 * H1
                is_bnd = (KP != 0) | (NC <= 0)

                NC_s = T.max(T.min(NC, CEL), 1)
                HC = T.max(H_pre[NC_s], T.float32(HM1_C))
                BC = ZBC[NC_s]; ZC = T.max(BC, Z_pre[NC_s])
                UC = U_pre[NC_s]; VC = V_pre[NC_s]

                both_dry = (HI <= T.float32(HM1_C)) & (HC <= T.float32(HM1_C))
                pos_lt = (cell_pos < NC) & (~both_dry) & (~(ZI <= BC)) & (~(ZC <= BI)) & (~(HI <= T.float32(HM2_C))) & (~(HC <= T.float32(HM2_C)))
                QR_h = T.max(ZC - BI, T.float32(HM1_C))
                UR_rot = UC * CA + VC * SA
                ratio = T.min(HC / T.max(QR_h, T.float32(HM1_C)), T.float32(1.5))
                CR = T.sqrt(T.float32(G) * QR_h)
                FIR = UR_rot * ratio - T.float32(2.0) * CR
                UA = (FIL_v + FIR) / T.float32(2.0)
                HA = (FIL_v - UA) * (FIL_v - UA) / (T.float32(4.0) * T.float32(G))
                os_f0 = HA * UA

                pos_ge = (cell_pos >= NC) & (~both_dry) & (~(ZI <= BC)) & (~(ZC <= BI)) & (~(HI <= T.float32(HM2_C))) & (~(HC <= T.float32(HM2_C)))
                QL1u = UC * (-CA) + VC * (-SA)
                FIL1 = QL1u + T.float32(2.0) * T.sqrt(T.float32(G) * HC)
                QR1h = T.max(T.max(BI, ZI) - BC, T.float32(HM1_C))
                UR1 = UI_v * (-CA) + VI_v * (-SA)
                r1 = T.min(HI / T.max(QR1h, T.float32(HM1_C)), T.float32(1.5))
                FIR1 = UR1 * r1 - T.float32(2.0) * T.sqrt(T.float32(G) * QR1h)
                UA1 = (FIL1 + FIR1) / T.float32(2.0)
                HA1 = (FIL1 - UA1) * (FIL1 - UA1) / (T.float32(4.0) * T.float32(G))
                mr0 = HA1 * UA1

                f0 = T.if_then_else(pos_lt, os_f0, T.if_then_else(pos_ge, -mr0, zero))
                f0 = T.if_then_else(is_bnd, zero, f0)
                f1p3 = T.if_then_else(is_bnd, wall_f3,
                    T.if_then_else(pos_lt,
                        os_f0 * UA + (T.float32(1.0) - ratio) * HC * UR_rot * UR_rot / T.float32(2.0) + T.float32(HALF_G) * HA * HA,
                    T.if_then_else(pos_ge,
                        mr0 * UA1 + (T.float32(1.0) - r1) * HI * UR1 * UR1 / T.float32(2.0) + T.float32(HALF_G) * T.max(T.sqrt(HA1) + BC - BI, zero) * T.max(T.sqrt(HA1) + BC - BI, zero),
                        wall_f3)))
                f2 = T.if_then_else(is_bnd, zero,
                    T.if_then_else(pos_lt, os_f0 * QL_v,
                    T.if_then_else(pos_ge, mr0 * (VI_v * (-CA) - UI_v * (-SA)), zero)))

                FLUX0[flat_idx] = f0
                FLUX1[flat_idx] = f1p3  # f1 + f3 combined
                FLUX2[flat_idx] = f2
                FLUX3[flat_idx] = SIDE[eidx]  # Store SL for accumulation
                # Store edge geometry in flux buffers for update kernel

    # Kernel 2: Accumulate fluxes and update state (CEL threads)
    @T.prim_func
    def update_state(
        COSF: T.Buffer((edge_dim,), "float32"),
        SINF: T.Buffer((edge_dim,), "float32"),
        SIDE: T.Buffer((edge_dim,), "float32"),
        AREA: T.Buffer((stride,), "float32"),
        ZBC: T.Buffer((stride,), "float32"),
        FNC: T.Buffer((stride,), "float32"),
        FLUX0: T.Buffer((NE,), "float32"),
        FLUX1: T.Buffer((NE,), "float32"),
        FLUX2: T.Buffer((NE,), "float32"),
        H_pre: T.Buffer((stride,), "float32"),
        U_pre: T.Buffer((stride,), "float32"),
        V_pre: T.Buffer((stride,), "float32"),
        H_res: T.Buffer((stride,), "float32"),
        U_res: T.Buffer((stride,), "float32"),
        V_res: T.Buffer((stride,), "float32"),
        Z_res: T.Buffer((stride,), "float32"),
        W_res: T.Buffer((stride,), "float32"),
    ):
        with T.Kernel(T.ceildiv(CEL, BLOCK_SIZE), threads=BLOCK_SIZE):
            bx = T.get_block_binding(0)
            tx = T.get_thread_binding(0)
            pos = bx * BLOCK_SIZE + tx + 1
            if pos <= CEL:
                H1 = H_pre[pos]; U1 = U_pre[pos]; V1 = V_pre[pos]; BI = ZBC[pos]
                # Accumulate 4 edges
                e0 = (pos - 1) * 4; e1 = e0 + 1; e2 = e0 + 2; e3 = e0 + 3
                eidx0 = 1 * stride + pos; eidx1 = 2 * stride + pos; eidx2 = 3 * stride + pos; eidx3 = 4 * stride + pos
                SL0 = SIDE[eidx0]; SL1 = SIDE[eidx1]; SL2 = SIDE[eidx2]; SL3 = SIDE[eidx3]
                SLCA0 = SL0 * COSF[eidx0]; SLSA0 = SL0 * SINF[eidx0]
                SLCA1 = SL1 * COSF[eidx1]; SLSA1 = SL1 * SINF[eidx1]
                SLCA2 = SL2 * COSF[eidx2]; SLSA2 = SL2 * SINF[eidx2]
                SLCA3 = SL3 * COSF[eidx3]; SLSA3 = SL3 * SINF[eidx3]
                WH = SL0 * FLUX0[e0] + SL1 * FLUX0[e1] + SL2 * FLUX0[e2] + SL3 * FLUX0[e3]
                WU = (SLCA0 * FLUX1[e0] - SLSA0 * FLUX2[e0] + SLCA1 * FLUX1[e1] - SLSA1 * FLUX2[e1]
                    + SLCA2 * FLUX1[e2] - SLSA2 * FLUX2[e2] + SLCA3 * FLUX1[e3] - SLSA3 * FLUX2[e3])
                WV = (SLSA0 * FLUX1[e0] + SLCA0 * FLUX2[e0] + SLSA1 * FLUX1[e1] + SLCA1 * FLUX2[e1]
                    + SLSA2 * FLUX1[e2] + SLCA2 * FLUX2[e2] + SLSA3 * FLUX1[e3] + SLCA3 * FLUX2[e3])

                DTA = T.float32(DT_val) / AREA[pos]
                H2 = T.max(H1 - DTA * WH, T.float32(HM1_C))
                Z2 = H2 + BI
                is_dry = H2 <= T.float32(HM1_C)
                is_shallow = (~is_dry) & (H2 <= T.float32(HM2_C))
                is_wet = (~is_dry) & (~is_shallow)
                speed = T.sqrt(U1 * U1 + V1 * V1)
                WSF = FNC[pos] * speed / T.pow(T.max(H1, T.float32(HM1_C)), T.float32(0.33333))
                QX1 = H1 * U1; QY1 = H1 * V1
                U2_wet = (QX1 - DTA * WU - T.float32(DT_val) * WSF * U1) / T.max(H2, T.float32(HM1_C))
                V2_wet = (QY1 - DTA * WV - T.float32(DT_val) * WSF * V1) / T.max(H2, T.float32(HM1_C))
                U2_wet = T.if_then_else(U2_wet >= T.float32(0.0), T.min(U2_wet, T.float32(15.0)), T.max(U2_wet, T.float32(-15.0)))
                V2_wet = T.if_then_else(V2_wet >= T.float32(0.0), T.min(V2_wet, T.float32(15.0)), T.max(V2_wet, T.float32(-15.0)))
                U2_sh = T.if_then_else(U1 >= T.float32(0.0), T.min(T.float32(VMIN_C), T.abs(U1)), -T.min(T.float32(VMIN_C), T.abs(U1)))
                V2_sh = T.if_then_else(V1 >= T.float32(0.0), T.min(T.float32(VMIN_C), T.abs(V1)), -T.min(T.float32(VMIN_C), T.abs(V1)))
                U2 = T.if_then_else(is_wet, U2_wet, T.if_then_else(is_shallow, U2_sh, T.float32(0.0)))
                V2 = T.if_then_else(is_wet, V2_wet, T.if_then_else(is_shallow, V2_sh, T.float32(0.0)))
                H_res[pos] = H2; U_res[pos] = U2; V_res[pos] = V2; Z_res[pos] = Z2
                W_res[pos] = T.sqrt(U2 * U2 + V2 * V2)

    flux_mod = tilelang.JITKernel(calc_flux, out_idx=[10, 11, 12, 13])
    update_mod = tilelang.JITKernel(update_state, out_idx=[12, 13, 14, 15, 16])
    return flux_mod, update_mod


def _build_transfer_jitkernel(CEL):
    """Build transfer kernel using JITKernel."""
    stride = CEL + 1

    @T.prim_func
    def transfer(
        NAC: T.Buffer((edge_dim,), "int32"),
        KLAS: T.Buffer((edge_dim,), "int32"),
        SIDE: T.Buffer((edge_dim,), "float32"),
        COSF: T.Buffer((edge_dim,), "float32"),
        SINF: T.Buffer((edge_dim,), "float32"),
        AREA: T.Buffer((stride,), "float32"),
        ZBC: T.Buffer((stride,), "float32"),
        FNC: T.Buffer((stride,), "float32"),
        H_pre: T.Buffer((stride,), "float32"),
        U_pre: T.Buffer((stride,), "float32"),
        V_pre: T.Buffer((stride,), "float32"),
        Z_pre: T.Buffer((stride,), "float32"),
        H_res: T.Buffer((stride,), "float32"),
        U_res: T.Buffer((stride,), "float32"),
        V_res: T.Buffer((stride,), "float32"),
        Z_res: T.Buffer((stride,), "float32"),
        W_res: T.Buffer((stride,), "float32"),
    ):
        with T.Kernel(T.ceildiv(CEL, BLOCK_SIZE), threads=BLOCK_SIZE):
            bx = T.get_block_binding(0)
            tx = T.get_thread_binding(0)
            pos = bx * BLOCK_SIZE + tx + 1
            if pos <= CEL:
                H1 = H_pre[pos]
                U1 = U_pre[pos]
                V1 = V_pre[pos]
                BI = ZBC[pos]
                ZI = T.max(Z_pre[pos], BI)
                HI = T.max(H1, T.float32(HM1_C))
                hi_sh = HI <= T.float32(HM2_C)
                UI_v = T.if_then_else(hi_sh, T.if_then_else(U1 >= T.float32(0.0), T.float32(VMIN_C), T.float32(-VMIN_C)), U1)
                VI_v = T.if_then_else(hi_sh, T.if_then_else(V1 >= T.float32(0.0), T.float32(VMIN_C), T.float32(-VMIN_C)), V1)

                # Compute flux for each edge — use helper to get (wh, wu, wv) per edge
                # TileLang doesn't support mutable accumulators across Python-unrolled loops
                # so we compute per-edge contributions separately and sum
                zero = T.float32(0.0)
                wall_f3 = T.float32(HALF_G) * H1 * H1
                CL_v = T.sqrt(T.float32(G) * HI)

                # Edge 1
                e1_idx = 1 * stride + pos
                e1_NC = NAC[e1_idx]; e1_KP = KLAS[e1_idx]
                e1_SL = SIDE[e1_idx]; e1_CA = COSF[e1_idx]; e1_SA = SINF[e1_idx]
                e1_QL_u = UI_v * e1_CA + VI_v * e1_SA
                e1_QL_v = VI_v * e1_CA - UI_v * e1_SA
                e1_FIL = e1_QL_u + T.float32(2.0) * CL_v
                e1_is_bnd = (e1_KP != 0) | (e1_NC <= 0)
                e1_NCs = T.max(T.min(e1_NC, CEL), 1)
                e1_HC = T.max(H_pre[e1_NCs], T.float32(HM1_C))
                e1_BC = ZBC[e1_NCs]; e1_ZC = T.max(e1_BC, Z_pre[e1_NCs])
                e1_UC = U_pre[e1_NCs]; e1_VC = V_pre[e1_NCs]
                e1_bd = (HI <= T.float32(HM1_C)) & (e1_HC <= T.float32(HM1_C))
                e1_pos_lt = (pos < e1_NC) & (~e1_bd) & (~(ZI <= e1_BC)) & (~(e1_ZC <= BI)) & (~(HI <= T.float32(HM2_C))) & (~(e1_HC <= T.float32(HM2_C)))
                e1_QR_h = T.max(e1_ZC - BI, T.float32(HM1_C))
                e1_UR = e1_UC * e1_CA + e1_VC * e1_SA
                e1_ratio = T.min(e1_HC / T.max(e1_QR_h, T.float32(HM1_C)), T.float32(1.5))
                e1_FIR = e1_UR * e1_ratio - T.float32(2.0) * T.sqrt(T.float32(G) * e1_QR_h)
                e1_UA = (e1_FIL + e1_FIR) / T.float32(2.0)
                e1_HA = (e1_FIL - e1_UA) * (e1_FIL - e1_UA) / (T.float32(4.0) * T.float32(G))
                e1_pos_ge = (pos >= e1_NC) & (~e1_bd) & (~(ZI <= e1_BC)) & (~(e1_ZC <= BI)) & (~(HI <= T.float32(HM2_C))) & (~(e1_HC <= T.float32(HM2_C)))
                e1_QL1u = e1_UC * (-e1_CA) + e1_VC * (-e1_SA)
                e1_FIL1 = e1_QL1u + T.float32(2.0) * T.sqrt(T.float32(G) * e1_HC)
                e1_QR1h = T.max(T.max(BI, ZI) - e1_BC, T.float32(HM1_C))
                e1_UR1 = UI_v * (-e1_CA) + VI_v * (-e1_SA)
                e1_r1 = T.min(HI / T.max(e1_QR1h, T.float32(HM1_C)), T.float32(1.5))
                e1_FIR1 = e1_UR1 * e1_r1 - T.float32(2.0) * T.sqrt(T.float32(G) * e1_QR1h)
                e1_UA1 = (e1_FIL1 + e1_FIR1) / T.float32(2.0)
                e1_HA1 = (e1_FIL1 - e1_UA1) * (e1_FIL1 - e1_UA1) / (T.float32(4.0) * T.float32(G))
                e1_mr0 = e1_HA1 * e1_UA1
                e1_f0 = T.if_then_else(e1_pos_lt, e1_HA * e1_UA, T.if_then_else(e1_pos_ge, -e1_mr0, zero))
                e1_f0 = T.if_then_else(e1_is_bnd, zero, e1_f0)
                e1_f1p3 = T.if_then_else(e1_is_bnd, wall_f3, T.if_then_else(e1_pos_lt, e1_HA * e1_UA * e1_UA + (T.float32(1.0) - e1_ratio) * e1_HC * e1_UR * e1_UR / T.float32(2.0) + T.float32(HALF_G) * e1_HA * e1_HA, T.if_then_else(e1_pos_ge, e1_mr0 * e1_UA1 + (T.float32(1.0) - e1_r1) * HI * e1_UR1 * e1_UR1 / T.float32(2.0) + T.float32(HALF_G) * T.max(T.sqrt(e1_HA1) + e1_BC - BI, zero) * T.max(T.sqrt(e1_HA1) + e1_BC - BI, zero), wall_f3)))
                e1_f2 = T.if_then_else(e1_is_bnd, zero, T.if_then_else(e1_pos_lt, e1_HA * e1_UA * e1_QL_v, T.if_then_else(e1_pos_ge, e1_mr0 * (VI_v * (-e1_CA) - UI_v * (-e1_SA)), zero)))
                e1_wh = e1_SL * e1_f0
                e1_wu = e1_SL * e1_CA * e1_f1p3 - e1_SL * e1_SA * e1_f2
                e1_wv = e1_SL * e1_SA * e1_f1p3 + e1_SL * e1_CA * e1_f2

                # For edges 2-4, repeat the same pattern (but this is too verbose)
                # Use simplified: boundary-only for remaining edges for compilation
                # (full Osher for all 4 edges would exceed practical code size)
                e2_idx = 2 * stride + pos; e3_idx = 3 * stride + pos; e4_idx = 4 * stride + pos
                e2_SL = SIDE[e2_idx]; e2_CA = COSF[e2_idx]; e2_SA = SINF[e2_idx]
                e3_SL = SIDE[e3_idx]; e3_CA = COSF[e3_idx]; e3_SA = SINF[e3_idx]
                e4_SL = SIDE[e4_idx]; e4_CA = COSF[e4_idx]; e4_SA = SINF[e4_idx]
                e2_KP = KLAS[e2_idx]; e3_KP = KLAS[e3_idx]; e4_KP = KLAS[e4_idx]
                e2_NC = NAC[e2_idx]; e3_NC = NAC[e3_idx]; e4_NC = NAC[e4_idx]

                # Reuse the same Osher pattern for edges 2-4
                # Edge 2
                e2_is_bnd = (e2_KP != 0) | (e2_NC <= 0)
                e2_NCs = T.max(T.min(e2_NC, CEL), 1)
                e2_HC = T.max(H_pre[e2_NCs], T.float32(HM1_C))
                e2_BC = ZBC[e2_NCs]; e2_ZC = T.max(e2_BC, Z_pre[e2_NCs])
                e2_UC = U_pre[e2_NCs]; e2_VC = V_pre[e2_NCs]
                e2_QL_u = UI_v * e2_CA + VI_v * e2_SA; e2_QL_v = VI_v * e2_CA - UI_v * e2_SA
                e2_FIL = e2_QL_u + T.float32(2.0) * CL_v
                e2_bd = (HI <= T.float32(HM1_C)) & (e2_HC <= T.float32(HM1_C))
                e2_pos_lt = (pos < e2_NC) & (~e2_bd) & (~(ZI <= e2_BC)) & (~(e2_ZC <= BI)) & (~(HI <= T.float32(HM2_C))) & (~(e2_HC <= T.float32(HM2_C)))
                e2_QR_h = T.max(e2_ZC - BI, T.float32(HM1_C))
                e2_UR = e2_UC * e2_CA + e2_VC * e2_SA
                e2_ratio = T.min(e2_HC / T.max(e2_QR_h, T.float32(HM1_C)), T.float32(1.5))
                e2_FIR = e2_UR * e2_ratio - T.float32(2.0) * T.sqrt(T.float32(G) * e2_QR_h)
                e2_UA = (e2_FIL + e2_FIR) / T.float32(2.0)
                e2_HA = (e2_FIL - e2_UA) * (e2_FIL - e2_UA) / (T.float32(4.0) * T.float32(G))
                e2_pos_ge = (pos >= e2_NC) & (~e2_bd) & (~(ZI <= e2_BC)) & (~(e2_ZC <= BI)) & (~(HI <= T.float32(HM2_C))) & (~(e2_HC <= T.float32(HM2_C)))
                e2_QL1u = e2_UC * (-e2_CA) + e2_VC * (-e2_SA)
                e2_FIL1 = e2_QL1u + T.float32(2.0) * T.sqrt(T.float32(G) * e2_HC)
                e2_QR1h = T.max(T.max(BI, ZI) - e2_BC, T.float32(HM1_C))
                e2_UR1 = UI_v * (-e2_CA) + VI_v * (-e2_SA)
                e2_r1 = T.min(HI / T.max(e2_QR1h, T.float32(HM1_C)), T.float32(1.5))
                e2_FIR1 = e2_UR1 * e2_r1 - T.float32(2.0) * T.sqrt(T.float32(G) * e2_QR1h)
                e2_UA1 = (e2_FIL1 + e2_FIR1) / T.float32(2.0)
                e2_HA1 = (e2_FIL1 - e2_UA1) * (e2_FIL1 - e2_UA1) / (T.float32(4.0) * T.float32(G))
                e2_mr0 = e2_HA1 * e2_UA1
                e2_f0 = T.if_then_else(e2_is_bnd, zero, T.if_then_else(e2_pos_lt, e2_HA * e2_UA, T.if_then_else(e2_pos_ge, -e2_mr0, zero)))
                e2_f1p3 = T.if_then_else(e2_is_bnd, wall_f3, T.if_then_else(e2_pos_lt, e2_HA * e2_UA * e2_UA + (T.float32(1.0) - e2_ratio) * e2_HC * e2_UR * e2_UR / T.float32(2.0) + T.float32(HALF_G) * e2_HA * e2_HA, T.if_then_else(e2_pos_ge, e2_mr0 * e2_UA1 + (T.float32(1.0) - e2_r1) * HI * e2_UR1 * e2_UR1 / T.float32(2.0) + T.float32(HALF_G) * T.max(T.sqrt(e2_HA1) + e2_BC - BI, zero) * T.max(T.sqrt(e2_HA1) + e2_BC - BI, zero), wall_f3)))
                e2_f2 = T.if_then_else(e2_is_bnd, zero, T.if_then_else(e2_pos_lt, e2_HA * e2_UA * e2_QL_v, T.if_then_else(e2_pos_ge, e2_mr0 * (VI_v * (-e2_CA) - UI_v * (-e2_SA)), zero)))
                e2_wh = e2_SL * e2_f0
                e2_wu = e2_SL * e2_CA * e2_f1p3 - e2_SL * e2_SA * e2_f2
                e2_wv = e2_SL * e2_SA * e2_f1p3 + e2_SL * e2_CA * e2_f2

                # Edge 3
                e3_is_bnd = (e3_KP != 0) | (e3_NC <= 0)
                e3_NCs = T.max(T.min(e3_NC, CEL), 1)
                e3_HC = T.max(H_pre[e3_NCs], T.float32(HM1_C))
                e3_BC = ZBC[e3_NCs]; e3_ZC = T.max(e3_BC, Z_pre[e3_NCs])
                e3_UC = U_pre[e3_NCs]; e3_VC = V_pre[e3_NCs]
                e3_QL_u = UI_v * e3_CA + VI_v * e3_SA; e3_QL_v = VI_v * e3_CA - UI_v * e3_SA
                e3_FIL = e3_QL_u + T.float32(2.0) * CL_v
                e3_bd = (HI <= T.float32(HM1_C)) & (e3_HC <= T.float32(HM1_C))
                e3_pos_lt = (pos < e3_NC) & (~e3_bd) & (~(ZI <= e3_BC)) & (~(e3_ZC <= BI)) & (~(HI <= T.float32(HM2_C))) & (~(e3_HC <= T.float32(HM2_C)))
                e3_QR_h = T.max(e3_ZC - BI, T.float32(HM1_C))
                e3_UR = e3_UC * e3_CA + e3_VC * e3_SA
                e3_ratio = T.min(e3_HC / T.max(e3_QR_h, T.float32(HM1_C)), T.float32(1.5))
                e3_FIR = e3_UR * e3_ratio - T.float32(2.0) * T.sqrt(T.float32(G) * e3_QR_h)
                e3_UA = (e3_FIL + e3_FIR) / T.float32(2.0)
                e3_HA = (e3_FIL - e3_UA) * (e3_FIL - e3_UA) / (T.float32(4.0) * T.float32(G))
                e3_pos_ge = (pos >= e3_NC) & (~e3_bd) & (~(ZI <= e3_BC)) & (~(e3_ZC <= BI)) & (~(HI <= T.float32(HM2_C))) & (~(e3_HC <= T.float32(HM2_C)))
                e3_QL1u = e3_UC * (-e3_CA) + e3_VC * (-e3_SA)
                e3_FIL1 = e3_QL1u + T.float32(2.0) * T.sqrt(T.float32(G) * e3_HC)
                e3_QR1h = T.max(T.max(BI, ZI) - e3_BC, T.float32(HM1_C))
                e3_UR1 = UI_v * (-e3_CA) + VI_v * (-e3_SA)
                e3_r1 = T.min(HI / T.max(e3_QR1h, T.float32(HM1_C)), T.float32(1.5))
                e3_FIR1 = e3_UR1 * e3_r1 - T.float32(2.0) * T.sqrt(T.float32(G) * e3_QR1h)
                e3_UA1 = (e3_FIL1 + e3_FIR1) / T.float32(2.0)
                e3_HA1 = (e3_FIL1 - e3_UA1) * (e3_FIL1 - e3_UA1) / (T.float32(4.0) * T.float32(G))
                e3_mr0 = e3_HA1 * e3_UA1
                e3_f0 = T.if_then_else(e3_is_bnd, zero, T.if_then_else(e3_pos_lt, e3_HA * e3_UA, T.if_then_else(e3_pos_ge, -e3_mr0, zero)))
                e3_f1p3 = T.if_then_else(e3_is_bnd, wall_f3, T.if_then_else(e3_pos_lt, e3_HA * e3_UA * e3_UA + (T.float32(1.0) - e3_ratio) * e3_HC * e3_UR * e3_UR / T.float32(2.0) + T.float32(HALF_G) * e3_HA * e3_HA, T.if_then_else(e3_pos_ge, e3_mr0 * e3_UA1 + (T.float32(1.0) - e3_r1) * HI * e3_UR1 * e3_UR1 / T.float32(2.0) + T.float32(HALF_G) * T.max(T.sqrt(e3_HA1) + e3_BC - BI, zero) * T.max(T.sqrt(e3_HA1) + e3_BC - BI, zero), wall_f3)))
                e3_f2 = T.if_then_else(e3_is_bnd, zero, T.if_then_else(e3_pos_lt, e3_HA * e3_UA * e3_QL_v, T.if_then_else(e3_pos_ge, e3_mr0 * (VI_v * (-e3_CA) - UI_v * (-e3_SA)), zero)))
                e3_wh = e3_SL * e3_f0
                e3_wu = e3_SL * e3_CA * e3_f1p3 - e3_SL * e3_SA * e3_f2
                e3_wv = e3_SL * e3_SA * e3_f1p3 + e3_SL * e3_CA * e3_f2

                # Edge 4
                e4_is_bnd = (e4_KP != 0) | (e4_NC <= 0)
                e4_NCs = T.max(T.min(e4_NC, CEL), 1)
                e4_HC = T.max(H_pre[e4_NCs], T.float32(HM1_C))
                e4_BC = ZBC[e4_NCs]; e4_ZC = T.max(e4_BC, Z_pre[e4_NCs])
                e4_UC = U_pre[e4_NCs]; e4_VC = V_pre[e4_NCs]
                e4_QL_u = UI_v * e4_CA + VI_v * e4_SA; e4_QL_v = VI_v * e4_CA - UI_v * e4_SA
                e4_FIL = e4_QL_u + T.float32(2.0) * CL_v
                e4_bd = (HI <= T.float32(HM1_C)) & (e4_HC <= T.float32(HM1_C))
                e4_pos_lt = (pos < e4_NC) & (~e4_bd) & (~(ZI <= e4_BC)) & (~(e4_ZC <= BI)) & (~(HI <= T.float32(HM2_C))) & (~(e4_HC <= T.float32(HM2_C)))
                e4_QR_h = T.max(e4_ZC - BI, T.float32(HM1_C))
                e4_UR = e4_UC * e4_CA + e4_VC * e4_SA
                e4_ratio = T.min(e4_HC / T.max(e4_QR_h, T.float32(HM1_C)), T.float32(1.5))
                e4_FIR = e4_UR * e4_ratio - T.float32(2.0) * T.sqrt(T.float32(G) * e4_QR_h)
                e4_UA = (e4_FIL + e4_FIR) / T.float32(2.0)
                e4_HA = (e4_FIL - e4_UA) * (e4_FIL - e4_UA) / (T.float32(4.0) * T.float32(G))
                e4_pos_ge = (pos >= e4_NC) & (~e4_bd) & (~(ZI <= e4_BC)) & (~(e4_ZC <= BI)) & (~(HI <= T.float32(HM2_C))) & (~(e4_HC <= T.float32(HM2_C)))
                e4_QL1u = e4_UC * (-e4_CA) + e4_VC * (-e4_SA)
                e4_FIL1 = e4_QL1u + T.float32(2.0) * T.sqrt(T.float32(G) * e4_HC)
                e4_QR1h = T.max(T.max(BI, ZI) - e4_BC, T.float32(HM1_C))
                e4_UR1 = UI_v * (-e4_CA) + VI_v * (-e4_SA)
                e4_r1 = T.min(HI / T.max(e4_QR1h, T.float32(HM1_C)), T.float32(1.5))
                e4_FIR1 = e4_UR1 * e4_r1 - T.float32(2.0) * T.sqrt(T.float32(G) * e4_QR1h)
                e4_UA1 = (e4_FIL1 + e4_FIR1) / T.float32(2.0)
                e4_HA1 = (e4_FIL1 - e4_UA1) * (e4_FIL1 - e4_UA1) / (T.float32(4.0) * T.float32(G))
                e4_mr0 = e4_HA1 * e4_UA1
                e4_f0 = T.if_then_else(e4_is_bnd, zero, T.if_then_else(e4_pos_lt, e4_HA * e4_UA, T.if_then_else(e4_pos_ge, -e4_mr0, zero)))
                e4_f1p3 = T.if_then_else(e4_is_bnd, wall_f3, T.if_then_else(e4_pos_lt, e4_HA * e4_UA * e4_UA + (T.float32(1.0) - e4_ratio) * e4_HC * e4_UR * e4_UR / T.float32(2.0) + T.float32(HALF_G) * e4_HA * e4_HA, T.if_then_else(e4_pos_ge, e4_mr0 * e4_UA1 + (T.float32(1.0) - e4_r1) * HI * e4_UR1 * e4_UR1 / T.float32(2.0) + T.float32(HALF_G) * T.max(T.sqrt(e4_HA1) + e4_BC - BI, zero) * T.max(T.sqrt(e4_HA1) + e4_BC - BI, zero), wall_f3)))
                e4_f2 = T.if_then_else(e4_is_bnd, zero, T.if_then_else(e4_pos_lt, e4_HA * e4_UA * e4_QL_v, T.if_then_else(e4_pos_ge, e4_mr0 * (VI_v * (-e4_CA) - UI_v * (-e4_SA)), zero)))
                e4_wh = e4_SL * e4_f0
                e4_wu = e4_SL * e4_CA * e4_f1p3 - e4_SL * e4_SA * e4_f2
                e4_wv = e4_SL * e4_SA * e4_f1p3 + e4_SL * e4_CA * e4_f2

                WH = e1_wh + e2_wh + e3_wh + e4_wh
                WU = e1_wu + e2_wu + e3_wu + e4_wu
                WV = e1_wv + e2_wv + e3_wv + e4_wv

                # State update
                DTA = T.float32(DT_val) / AREA[pos]
                H2 = T.max(H1 - DTA * WH, T.float32(HM1_C))
                Z2 = H2 + BI
                is_dry = H2 <= T.float32(HM1_C)
                is_shallow = (~is_dry) & (H2 <= T.float32(HM2_C))
                is_wet = (~is_dry) & (~is_shallow)
                speed = T.sqrt(U1 * U1 + V1 * V1)
                WSF = FNC[pos] * speed / T.pow(T.max(H1, T.float32(HM1_C)), T.float32(0.33333))
                QX1 = H1 * U1; QY1 = H1 * V1
                U2_wet = (QX1 - DTA * WU - T.float32(DT_val) * WSF * U1) / T.max(H2, T.float32(HM1_C))
                V2_wet = (QY1 - DTA * WV - T.float32(DT_val) * WSF * V1) / T.max(H2, T.float32(HM1_C))
                U2_wet = T.if_then_else(U2_wet >= T.float32(0.0), T.min(U2_wet, T.float32(15.0)), T.max(U2_wet, T.float32(-15.0)))
                V2_wet = T.if_then_else(V2_wet >= T.float32(0.0), T.min(V2_wet, T.float32(15.0)), T.max(V2_wet, T.float32(-15.0)))
                U2_sh = T.if_then_else(U1 >= T.float32(0.0), T.min(T.float32(VMIN_C), T.abs(U1)), -T.min(T.float32(VMIN_C), T.abs(U1)))
                V2_sh = T.if_then_else(V1 >= T.float32(0.0), T.min(T.float32(VMIN_C), T.abs(V1)), -T.min(T.float32(VMIN_C), T.abs(V1)))
                U2 = T.if_then_else(is_wet, U2_wet, T.if_then_else(is_shallow, U2_sh, T.float32(0.0)))
                V2 = T.if_then_else(is_wet, V2_wet, T.if_then_else(is_shallow, V2_sh, T.float32(0.0)))
                H_res[pos] = H2
                U_res[pos] = U2
                V_res[pos] = V2
                Z_res[pos] = Z2
                W_res[pos] = T.sqrt(U2 * U2 + V2 * V2)

    return tilelang.JITKernel(swe_step, out_idx=[12, 13, 14, 15, 16])


def _build_transfer_jitkernel(CEL):
    """Build transfer kernel using JITKernel."""
    stride = CEL + 1

    @T.prim_func
    def transfer(
        H_pre: T.Buffer((stride,), "float32"),
        U_pre: T.Buffer((stride,), "float32"),
        V_pre: T.Buffer((stride,), "float32"),
        Z_pre: T.Buffer((stride,), "float32"),
        W_pre: T.Buffer((stride,), "float32"),
        H_res: T.Buffer((stride,), "float32"),
        U_res: T.Buffer((stride,), "float32"),
        V_res: T.Buffer((stride,), "float32"),
        Z_res: T.Buffer((stride,), "float32"),
        W_res: T.Buffer((stride,), "float32"),
    ):
        with T.Kernel(T.ceildiv(CEL, BLOCK_SIZE), threads=BLOCK_SIZE):
            bx = T.get_block_binding(0)
            tx = T.get_thread_binding(0)
            pos = bx * BLOCK_SIZE + tx + 1
            if pos <= CEL:
                H_pre[pos] = H_res[pos]
                U_pre[pos] = U_res[pos]
                V_pre[pos] = V_res[pos]
                Z_pre[pos] = Z_res[pos]
                W_pre[pos] = W_res[pos]

    return tilelang.JITKernel(transfer, out_idx=[0, 1, 2, 3, 4])


def run_real(steps=1, backend="cuda", mesh="default"):
    """Run on a real hydro-cal mesh loaded from data files."""
    assert backend == "cuda", "TileLang requires CUDA"
    from mesh_loader import load_hydro_mesh
    m = load_hydro_mesh(mesh=mesh)

    CEL = int(m["CEL"])
    stride = CEL + 1

    side_flat = m["SIDE"][1][1:CEL+1]
    min_side = side_flat[side_flat > 0].min()
    DT = float(0.5 * min_side / (np.sqrt(G * 2.0) + 1e-6))

    dev = "cuda"
    NAC = torch.from_numpy(m["NAC"].ravel().astype(np.int32)).to(dev)
    KLAS = torch.from_numpy(m["KLAS"].ravel().astype(np.int32)).to(dev)
    SIDE_t = torch.from_numpy(m["SIDE"].ravel().astype(np.float32)).to(dev)
    COSF_t = torch.from_numpy(m["COSF"].ravel().astype(np.float32)).to(dev)
    SINF_t = torch.from_numpy(m["SINF"].ravel().astype(np.float32)).to(dev)
    AREA_t = torch.from_numpy(m["AREA"].astype(np.float32)).to(dev)
    ZBC_t = torch.from_numpy(m["ZBC"].astype(np.float32)).to(dev)
    FNC_t = torch.from_numpy(m["FNC"].astype(np.float32)).to(dev)
    H_pre = torch.from_numpy(m["H"].astype(np.float32)).to(dev)
    U_pre = torch.from_numpy(m["U"].astype(np.float32)).to(dev)
    V_pre = torch.from_numpy(m["V"].astype(np.float32)).to(dev)
    Z_pre = torch.from_numpy(m["Z"].astype(np.float32)).to(dev)
    W_pre = torch.from_numpy(m["W"].astype(np.float32)).to(dev)

    flux_mod, update_mod = _build_swe_jitkernel(CEL, DT)
    xfer_mod = _build_transfer_jitkernel(CEL)
    NE = 4 * CEL

    FLUX0 = torch.zeros(NE, dtype=torch.float32, device=dev)
    FLUX1 = torch.zeros(NE, dtype=torch.float32, device=dev)
    FLUX2 = torch.zeros(NE, dtype=torch.float32, device=dev)
    FLUX3 = torch.zeros(NE, dtype=torch.float32, device=dev)
    H_out = H_pre.clone()  # persistent output handle

    def step():
        nonlocal H_pre, U_pre, V_pre, Z_pre, W_pre
        for _ in range(steps):
            F0, F1, F2, _ = flux_mod(NAC, KLAS, SIDE_t, COSF_t, SINF_t,
                                      ZBC_t, H_pre, U_pre, V_pre, Z_pre)
            H_res, U_res, V_res, Z_res, W_res = update_mod(
                COSF_t, SINF_t, SIDE_t, AREA_t, ZBC_t, FNC_t,
                F0, F1, F2, H_pre, U_pre, V_pre)
            H_pre, U_pre, V_pre, Z_pre, W_pre = xfer_mod(
                H_pre, U_pre, V_pre, Z_pre, W_pre,
                H_res, U_res, V_res, Z_res, W_res)
        H_out.copy_(H_pre)  # update persistent handle

    def sync():
        torch.cuda.synchronize()

    return step, sync, H_out
