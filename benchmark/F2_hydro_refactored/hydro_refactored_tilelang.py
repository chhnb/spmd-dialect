"""F2 Hydro Refactored — TileLang implementation.

2-kernel-per-step pattern matching Taichi/Warp/Triton F2 ports:
  1. calculate_flux: 1 thread per edge (4*CELL threads)
  2. update_cell: 1 thread per cell (CELL threads)

Uses tilelang.JITKernel with T.Kernel + T.get_block_binding for reliable
repeated invocation (the @tilelang.jit path fails on second call).
"""
import os
os.environ.setdefault("CUDA_HOME", "/home/scratch.huanhuanc_gpu/spmd/cuda-toolkit")

import numpy as np
import torch
import tilelang
import tilelang.language as T

G_C = 9.81
HALF_G_C = 4.905
C1_C = 1.7
VMIN_C = 0.001
BLK = 256


def build_kernels(CELL, NE, DT_val, HM1_val, HM2_val):
    """Build flux + update kernels using JITKernel for reliable invocation."""

    @T.prim_func
    def calculate_flux(
        NAC: T.Buffer((NE,), "int32"),
        KLAS: T.Buffer((NE,), "float32"),
        COSF: T.Buffer((NE,), "float32"),
        SINF: T.Buffer((NE,), "float32"),
        H: T.Buffer((CELL,), "float32"),
        U: T.Buffer((CELL,), "float32"),
        V: T.Buffer((CELL,), "float32"),
        Z: T.Buffer((CELL,), "float32"),
        ZBC: T.Buffer((CELL,), "float32"),
        FLUX0: T.Buffer((NE,), "float32"),
        FLUX1: T.Buffer((NE,), "float32"),
        FLUX2: T.Buffer((NE,), "float32"),
        FLUX3: T.Buffer((NE,), "float32"),
    ):
        with T.Kernel(T.ceildiv(NE, BLK), threads=BLK):
            bx = T.get_block_binding(0)
            tx = T.get_thread_binding(0)
            idx = bx * BLK + tx
            if idx < NE:
                cell_i = idx // 4
                KP = T.cast(KLAS[idx], "int32")
                NC = NAC[idx] - 1

                HI = T.max(H[cell_i], T.float32(HM1_val))
                H1 = H[cell_i]
                BI = ZBC[cell_i]
                ZI = T.max(Z[cell_i], BI)
                UI_raw = U[cell_i]
                VI_raw = V[cell_i]
                hi_sh = HI <= T.float32(HM2_val)
                UI_v = T.if_then_else(hi_sh, T.if_then_else(UI_raw >= T.float32(0.0), T.float32(VMIN_C), T.float32(-VMIN_C)), UI_raw)
                VI_v = T.if_then_else(hi_sh, T.if_then_else(VI_raw >= T.float32(0.0), T.float32(VMIN_C), T.float32(-VMIN_C)), VI_raw)

                COSJ = COSF[idx]
                SINJ = SINF[idx]
                QL_u = UI_v * COSJ + VI_v * SINJ
                QL_v = VI_v * COSJ - UI_v * SINJ
                CL_v = T.sqrt(T.float32(G_C) * HI)
                FIL_v = QL_u + T.float32(2.0) * CL_v

                zero = T.float32(0.0)
                wall_f3 = T.float32(HALF_G_C) * H1 * H1
                is_bnd = (KP != 0) | (NC < 0)

                # Neighbor data (clamped for safety)
                NC_s = T.max(T.min(NC, CELL - 1), 0)
                HC = T.max(H[NC_s], T.float32(HM1_val))
                BC = ZBC[NC_s]
                ZC = T.max(BC, Z[NC_s])
                UC = U[NC_s]
                VC = V[NC_s]

                both_dry = (HI <= T.float32(HM1_val)) & (HC <= T.float32(HM1_val))
                zi_le_bc = ZI <= BC
                zc_le_bi = ZC <= BI
                hi_shallow = HI <= T.float32(HM2_val)
                hc_shallow = HC <= T.float32(HM2_val)

                # Osher direct (cell_i < NC, both wet)
                pos_lt = (cell_i < NC) & (~both_dry) & (~zi_le_bc) & (~zc_le_bi) & (~hi_shallow) & (~hc_shallow)
                QR_h = T.max(ZC - BI, T.float32(HM1_val))
                UR_rot = UC * COSJ + VC * SINJ
                ratio = T.min(HC / T.max(QR_h, T.float32(HM1_val)), T.float32(1.5))
                CR = T.sqrt(T.float32(G_C) * QR_h)
                FIR_v = UR_rot * ratio - T.float32(2.0) * CR
                UA = (FIL_v + FIR_v) / T.float32(2.0)
                HA = (FIL_v - UA) * (FIL_v - UA) / (T.float32(4.0) * T.float32(G_C))
                os_f0 = HA * UA
                os_f1 = os_f0 * UA + (T.float32(1.0) - ratio) * HC * UR_rot * UR_rot / T.float32(2.0)
                os_f2 = os_f0 * QL_v
                os_f3 = T.float32(HALF_G_C) * HA * HA

                # Mirror Osher (cell_i >= NC, both wet)
                pos_ge = (cell_i >= NC) & (~both_dry) & (~zi_le_bc) & (~zc_le_bi) & (~hi_shallow) & (~hc_shallow)
                QL1_u = UC * (-COSJ) + VC * (-SINJ)
                CL1 = T.sqrt(T.float32(G_C) * HC)
                FIL1 = QL1_u + T.float32(2.0) * CL1
                QR1_h = T.max(T.max(BI, ZI) - BC, T.float32(HM1_val))
                UR1 = UI_v * (-COSJ) + VI_v * (-SINJ)
                ratio1 = T.min(HI / T.max(QR1_h, T.float32(HM1_val)), T.float32(1.5))
                CR1 = T.sqrt(T.float32(G_C) * QR1_h)
                FIR1 = UR1 * ratio1 - T.float32(2.0) * CR1
                UA1 = (FIL1 + FIR1) / T.float32(2.0)
                HA1 = (FIL1 - UA1) * (FIL1 - UA1) / (T.float32(4.0) * T.float32(G_C))
                mr0 = HA1 * UA1
                mr1 = mr0 * UA1 + (T.float32(1.0) - ratio1) * HI * UR1 * UR1 / T.float32(2.0)
                mr2 = mr0 * (VI_v * (-COSJ) - UI_v * (-SINJ))
                ZA = T.sqrt(HA1) + BC
                HC3 = T.max(ZA - BI, zero)
                mr3 = T.float32(HALF_G_C) * HC3 * HC3

                # Shallow cases
                DH_zi = T.max(ZI - BC, T.float32(HM1_val))
                UN_zi = T.float32(C1_C) * T.sqrt(DH_zi)
                DH_zc = T.max(ZC - BI, T.float32(HM1_val))
                UN_zc = T.float32(C1_C) * T.sqrt(DH_zc)

                # Flux assembly
                r0 = zero; r1 = zero; r2 = zero; r3 = zero
                r0 = T.if_then_else(pos_lt, os_f0, r0)
                r1 = T.if_then_else(pos_lt, os_f1, r1)
                r2 = T.if_then_else(pos_lt, os_f2, r2)
                r3 = T.if_then_else(pos_lt, os_f3, r3)
                r0 = T.if_then_else(pos_ge, -mr0, r0)
                r1 = T.if_then_else(pos_ge, mr1, r1)
                r2 = T.if_then_else(pos_ge, mr2, r2)
                r3 = T.if_then_else(pos_ge, mr3, r3)
                hc_gt = hc_shallow & (~hi_shallow) & (~both_dry) & (~zi_le_bc) & (~zc_le_bi) & (ZI > ZC)
                r0 = T.if_then_else(hc_gt, DH_zi * UN_zi, r0)
                r1 = T.if_then_else(hc_gt, DH_zi * UN_zi * UN_zi, r1)
                r2 = T.if_then_else(hc_gt, DH_zi * UN_zi * QL_v, r2)
                r3 = T.if_then_else(hc_gt, T.float32(HALF_G_C) * (ZC - BI) * (ZC - BI), r3)
                hc_le = hc_shallow & (~hi_shallow) & (~both_dry) & (~zi_le_bc) & (~zc_le_bi) & (~(ZI > ZC))
                r0 = T.if_then_else(hc_le, -T.float32(C1_C) * T.pow(HC, T.float32(1.5)), r0)
                r1 = T.if_then_else(hc_le, HI * QL_u * QL_u, r1)
                r3 = T.if_then_else(hc_le, wall_f3, r3)
                hi_gt = hi_shallow & (~both_dry) & (~zi_le_bc) & (~zc_le_bi) & (ZC > ZI)
                r0 = T.if_then_else(hi_gt, DH_zc * (-UN_zc), r0)
                r1 = T.if_then_else(hi_gt, DH_zc * UN_zc * UN_zc, r1)
                r2 = T.if_then_else(hi_gt, DH_zc * (-UN_zc) * (VC * COSJ - UC * SINJ), r2)
                r3 = T.if_then_else(hi_gt, wall_f3, r3)
                hi_le = hi_shallow & (~both_dry) & (~zi_le_bc) & (~zc_le_bi) & (~(ZC > ZI))
                r0 = T.if_then_else(hi_le, T.float32(C1_C) * T.pow(HI, T.float32(1.5)), r0)
                r3 = T.if_then_else(hi_le, wall_f3, r3)
                zc_cond = zc_le_bi & (~both_dry) & (~zi_le_bc)
                r0 = T.if_then_else(zc_cond, T.float32(C1_C) * T.pow(HI, T.float32(1.5)), r0)
                r1 = T.if_then_else(zc_cond, HI * T.abs(QL_u) * QL_u, r1)
                r2 = T.if_then_else(zc_cond, HI * T.abs(QL_u) * QL_v, r2)
                zi_cond = zi_le_bc & (~both_dry)
                r0 = T.if_then_else(zi_cond, -T.float32(C1_C) * T.pow(HC, T.float32(1.5)), r0)
                r1 = T.if_then_else(zi_cond, HI * QL_u * T.abs(QL_u), r1)
                r3 = T.if_then_else(zi_cond, wall_f3, r3)
                r0 = T.if_then_else(is_bnd, zero, r0)
                r1 = T.if_then_else(is_bnd, zero, r1)
                r2 = T.if_then_else(is_bnd, zero, r2)
                r3 = T.if_then_else(is_bnd, wall_f3, r3)

                FLUX0[idx] = r0
                FLUX1[idx] = r1
                FLUX2[idx] = r2
                FLUX3[idx] = r3

    @T.prim_func
    def update_cell(
        SIDE: T.Buffer((NE,), "float32"),
        SLCOS: T.Buffer((NE,), "float32"),
        SLSIN: T.Buffer((NE,), "float32"),
        AREA: T.Buffer((CELL,), "float32"),
        ZBC: T.Buffer((CELL,), "float32"),
        FNC: T.Buffer((CELL,), "float32"),
        FLUX0: T.Buffer((NE,), "float32"),
        FLUX1: T.Buffer((NE,), "float32"),
        FLUX2: T.Buffer((NE,), "float32"),
        FLUX3: T.Buffer((NE,), "float32"),
        H: T.Buffer((CELL,), "float32"),
        U: T.Buffer((CELL,), "float32"),
        V: T.Buffer((CELL,), "float32"),
        Z: T.Buffer((CELL,), "float32"),
        W: T.Buffer((CELL,), "float32"),
    ):
        with T.Kernel(T.ceildiv(CELL, BLK), threads=BLK):
            bx = T.get_block_binding(0)
            tx = T.get_thread_binding(0)
            i = bx * BLK + tx
            if i < CELL:
                H1 = H[i]; U1 = U[i]; V1 = V[i]; BI = ZBC[i]
                # Unroll 4 edges manually (T.serial creates scoping issues with JITKernel)
                e0 = 4 * i; e1 = e0 + 1; e2 = e0 + 2; e3 = e0 + 3
                FLR1_0 = FLUX1[e0] + FLUX3[e0]; FLR2_0 = FLUX2[e0]
                FLR1_1 = FLUX1[e1] + FLUX3[e1]; FLR2_1 = FLUX2[e1]
                FLR1_2 = FLUX1[e2] + FLUX3[e2]; FLR2_2 = FLUX2[e2]
                FLR1_3 = FLUX1[e3] + FLUX3[e3]; FLR2_3 = FLUX2[e3]
                WH = SIDE[e0]*FLUX0[e0] + SIDE[e1]*FLUX0[e1] + SIDE[e2]*FLUX0[e2] + SIDE[e3]*FLUX0[e3]
                WU = (SLCOS[e0]*FLR1_0 - SLSIN[e0]*FLR2_0 + SLCOS[e1]*FLR1_1 - SLSIN[e1]*FLR2_1
                    + SLCOS[e2]*FLR1_2 - SLSIN[e2]*FLR2_2 + SLCOS[e3]*FLR1_3 - SLSIN[e3]*FLR2_3)
                WV = (SLSIN[e0]*FLR1_0 + SLCOS[e0]*FLR2_0 + SLSIN[e1]*FLR1_1 + SLCOS[e1]*FLR2_1
                    + SLSIN[e2]*FLR1_2 + SLCOS[e2]*FLR2_2 + SLSIN[e3]*FLR1_3 + SLCOS[e3]*FLR2_3)

                DTA = T.float32(DT_val) / AREA[i]
                H2 = T.max(H1 - DTA * WH, T.float32(HM1_val))
                Z2 = H2 + BI
                is_dry = H2 <= T.float32(HM1_val)
                is_shallow = (~is_dry) & (H2 <= T.float32(HM2_val))
                is_wet = (~is_dry) & (~is_shallow)
                QX1 = H1 * U1; QY1 = H1 * V1
                speed = T.sqrt(U1 * U1 + V1 * V1)
                WSF = FNC[i] * speed / T.pow(T.max(H1, T.float32(HM1_val)), T.float32(0.33333))
                U2_wet = (QX1 - DTA * WU - T.float32(DT_val) * WSF * U1) / T.max(H2, T.float32(HM1_val))
                V2_wet = (QY1 - DTA * WV - T.float32(DT_val) * WSF * V1) / T.max(H2, T.float32(HM1_val))
                U2_wet = T.if_then_else(U2_wet >= T.float32(0.0), T.min(U2_wet, T.float32(15.0)), T.max(U2_wet, T.float32(-15.0)))
                V2_wet = T.if_then_else(V2_wet >= T.float32(0.0), T.min(V2_wet, T.float32(15.0)), T.max(V2_wet, T.float32(-15.0)))
                U2_sh = T.if_then_else(U1 >= T.float32(0.0), T.min(T.float32(VMIN_C), T.abs(U1)), -T.min(T.float32(VMIN_C), T.abs(U1)))
                V2_sh = T.if_then_else(V1 >= T.float32(0.0), T.min(T.float32(VMIN_C), T.abs(V1)), -T.min(T.float32(VMIN_C), T.abs(V1)))
                U2 = T.if_then_else(is_wet, U2_wet, T.if_then_else(is_shallow, U2_sh, T.float32(0.0)))
                V2 = T.if_then_else(is_wet, V2_wet, T.if_then_else(is_shallow, V2_sh, T.float32(0.0)))
                H[i] = H2; U[i] = U2; V[i] = V2; Z[i] = Z2
                W[i] = T.sqrt(U2 * U2 + V2 * V2)

    flux_mod = tilelang.JITKernel(calculate_flux, out_idx=[9, 10, 11, 12])
    update_mod = tilelang.JITKernel(update_cell, out_idx=[10, 11, 12, 13, 14])
    return flux_mod, update_mod


def run(days=10, backend="cuda", mesh="default"):
    """Run F2 hydro refactored solver on real mesh."""
    assert backend == "cuda", "TileLang requires CUDA"
    from mesh_loader import load_mesh
    m = load_mesh(mesh=mesh)

    CELL = int(m["CELL"])
    NE = 4 * CELL
    HM1 = float(m["HM1"])
    HM2 = float(m["HM2"])
    DT = float(m["DT"])
    steps_per_day = int(m["steps_per_day"])
    total_steps = steps_per_day * days

    dev = "cuda"
    NAC_t = torch.from_numpy(m["NAC"].astype(np.int32)).to(dev)
    KLAS_t = torch.from_numpy(m["KLAS"].astype(np.float32)).to(dev)
    SIDE_t = torch.from_numpy(m["SIDE"].astype(np.float32)).to(dev)
    COSF_t = torch.from_numpy(m["COSF"].astype(np.float32)).to(dev)
    SINF_t = torch.from_numpy(m["SINF"].astype(np.float32)).to(dev)
    SLCOS_t = torch.from_numpy(m["SLCOS"].astype(np.float32)).to(dev)
    SLSIN_t = torch.from_numpy(m["SLSIN"].astype(np.float32)).to(dev)
    FLUX0 = torch.zeros(NE, dtype=torch.float32, device=dev)
    FLUX1 = torch.zeros(NE, dtype=torch.float32, device=dev)
    FLUX2 = torch.zeros(NE, dtype=torch.float32, device=dev)
    FLUX3 = torch.zeros(NE, dtype=torch.float32, device=dev)

    H = torch.from_numpy(m["H"].astype(np.float32)).to(dev)
    U = torch.from_numpy(m["U"].astype(np.float32)).to(dev)
    V = torch.from_numpy(m["V"].astype(np.float32)).to(dev)
    Z = torch.from_numpy(m["Z"].astype(np.float32)).to(dev)
    W = torch.from_numpy(m["W"].astype(np.float32)).to(dev)
    ZBC_t = torch.from_numpy(m["ZBC"].astype(np.float32)).to(dev)
    AREA_t = torch.from_numpy(m["AREA"].astype(np.float32)).to(dev)
    FNC_t = torch.from_numpy(m["FNC"].astype(np.float32)).to(dev)

    flux_mod, update_mod = build_kernels(CELL, NE, DT, HM1, HM2)

    def step_fn():
        nonlocal H, U, V, Z, W
        for _ in range(total_steps):
            F0, F1, F2, F3 = flux_mod(NAC_t, KLAS_t, COSF_t, SINF_t,
                                       H, U, V, Z, ZBC_t)
            H, U, V, Z, W = update_mod(SIDE_t, SLCOS_t, SLSIN_t, AREA_t,
                                        ZBC_t, FNC_t, F0, F1, F2, F3)

    def sync_fn():
        torch.cuda.synchronize()

    return step_fn, sync_fn, H
