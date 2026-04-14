/**
 * F2: Refactored Hydro-Cal — CUDA benchmark with OSHER Riemann solver.
 * Tests 4 strategies: Sync loop, Async loop, CUDA Graph, Persistent Kernel.
 * Loads real binary mesh data from F2_hydro_refactored/data/binary/.
 *
 * Build: nvcc -O3 -arch=sm_90 -rdc=true hydro_osher_benchmark.cu -o hydro_osher -lcudadevrt
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <string>
#include <chrono>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// ===== Constants (matching Kokkos version exactly) =====
__constant__ float G_c     = 9.81f;
__constant__ float HALF_G_c = 4.905f;
__constant__ float C0_c    = 1.33f;
__constant__ float C1_C_c  = 1.7f;
__constant__ float VMIN_c  = 0.001f;
__constant__ float QLUA_c  = 0.0f;
__constant__ float BRDTH_c = 100.0f;

// Host-side constants for persistent kernel params
constexpr float G     = 9.81f;
constexpr float HALF_G = 4.905f;
constexpr float C0    = 1.33f;
constexpr float C1_C  = 1.7f;
constexpr float VMIN  = 0.001f;
constexpr float QLUA  = 0.0f;

// --- Device Graph tail launch support ---
__device__ cudaGraphExec_t d_graph_exec;
__global__ void tail_launch_kernel(int* steps_remaining) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int rem = atomicSub(steps_remaining, 1);
        if (rem > 1) cudaGraphLaunch(d_graph_exec, cudaStreamGraphTailLaunch);
    }
}

// ===== Device: QF flux function =====
__device__ __forceinline__
void QF(float h, float u, float v, float& F0, float& F1, float& F2, float& F3) {
    F0 = h * u;
    F1 = F0 * u;
    F2 = F0 * v;
    F3 = HALF_G_c * h * h;
}

// ===== Device: OSHER Riemann solver (all 16 K1xK2 cases) =====
__device__ __forceinline__
void osher(float QL_h, float QL_u, float QL_v,
           float QR_h, float QR_u, float QR_v,
           float FIL_in, float H_pos,
           float& R0, float& R1, float& R2, float& R3) {
    float CR = sqrtf(G_c * QR_h);
    float FIR_v = QR_u - 2.0f * CR;
    float fil = FIL_in, fir = FIR_v;
    float UA = (fil + fir) / 2.0f;
    float CA = fabsf((fil - fir) / 4.0f);
    float CL_v = sqrtf(G_c * H_pos);
    R0 = R1 = R2 = R3 = 0.0f;

    int K2 = (CA < UA) ? 1 : (UA >= 0 && UA < CA) ? 2 : (UA >= -CA && UA < 0) ? 3 : 4;
    int K1 = (QL_u < CL_v && QR_u >= -CR) ? 1 :
             (QL_u >= CL_v && QR_u >= -CR) ? 2 :
             (QL_u < CL_v && QR_u < -CR) ? 3 : 4;

    // Lambda-like helpers inlined as local functions via macros
    // qs1: add(QL_h, QL_u, QL_v, s)
    // qs2: U=fil/3, H=U*U/G, add(H, U, QL_v, s)
    // qs3: ua=(fil+fir)/2, fil-=ua, H=fil*fil/(4*G), add(H, ua, QL_v, s)
    // qs5: ua=(fil+fir)/2, fir-=ua, H=fir*fir/(4*G), add(H, ua, QR_v, s)
    // qs6: U=fir/3, H=U*U/G, add(H, U, QR_v, s)
    // qs7: add(QR_h, QR_u, QR_v, s)

    #define ADD(hh, uu, vv, ss) { \
        float _f0, _f1, _f2, _f3; QF(hh, uu, vv, _f0, _f1, _f2, _f3); \
        R0 += _f0*(ss); R1 += _f1*(ss); R2 += _f2*(ss); R3 += _f3*(ss); }
    #define QS1(ss) ADD(QL_h, QL_u, QL_v, ss)
    #define QS2(ss) { float _U=fil/3.0f, _H=_U*_U/G_c; ADD(_H, _U, QL_v, ss) }
    #define QS3(ss) { float _ua=(fil+fir)/2.0f; float _fl=fil-_ua; float _H=_fl*_fl/(4.0f*G_c); ADD(_H, _ua, QL_v, ss) }
    #define QS5(ss) { float _ua=(fil+fir)/2.0f; float _fr=fir-_ua; float _H=_fr*_fr/(4.0f*G_c); ADD(_H, _ua, QR_v, ss) }
    #define QS6(ss) { float _U=fir/3.0f, _H=_U*_U/G_c; ADD(_H, _U, QR_v, ss) }
    #define QS7(ss) ADD(QR_h, QR_u, QR_v, ss)

    switch(K1) {
    case 1: switch(K2) { case 1:QS2(1);break;case 2:QS3(1);break;case 3:QS5(1);break;case 4:QS6(1);break; } break;
    case 2: switch(K2) { case 1:QS1(1);break;case 2:QS1(1);QS2(-1);QS3(1);break;case 3:QS1(1);QS2(-1);QS5(1);break;case 4:QS1(1);QS2(-1);QS6(1);break; } break;
    case 3: switch(K2) { case 1:QS2(1);QS6(-1);QS7(1);break;case 2:QS3(1);QS6(-1);QS7(1);break;case 3:QS5(1);QS6(-1);QS7(1);break;case 4:QS7(1);break; } break;
    case 4: switch(K2) { case 1:QS1(1);QS6(-1);QS7(1);break;case 2:QS1(1);QS2(-1);QS3(1);QS6(-1);QS7(1);break;case 3:QS1(1);QS2(-1);QS5(1);QS6(-1);QS7(1);break;case 4:QS1(1);QS2(-1);QS7(1);break; } break;
    }

    #undef ADD
    #undef QS1
    #undef QS2
    #undef QS3
    #undef QS5
    #undef QS6
    #undef QS7
}

// ===== Kernel: calculate_flux (edge-parallel, 4*CELL threads) =====
__global__
void calculate_flux(int CELL, float HM1, float HM2,
                    const float* __restrict__ H, const float* __restrict__ U,
                    const float* __restrict__ V, const float* __restrict__ Z,
                    const float* __restrict__ ZBC, const float* __restrict__ ZB1,
                    const int* __restrict__ NAC, const float* __restrict__ KLAS,
                    const float* __restrict__ SIDE, const float* __restrict__ COSF,
                    const float* __restrict__ SINF,
                    float* __restrict__ FLUX0, float* __restrict__ FLUX1,
                    float* __restrict__ FLUX2, float* __restrict__ FLUX3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nSides = CELL * 4;
    if (idx >= nSides) return;

    int pos = idx / 4;

    float H1 = H[pos], U1 = U[pos], V1 = V[pos], Z1 = Z[pos];
    float v_ZB1 = ZB1[pos];
    int NC = NAC[idx] - 1;
    float KP = KLAS[idx];
    float COSJ = COSF[idx], SINJ = SINF[idx];

    float QL_h = H1;
    float QL_u = U1 * COSJ + V1 * SINJ;
    float QL_v = V1 * COSJ - U1 * SINJ;
    float CL = sqrtf(G_c * H1);
    float FIL = QL_u + 2.0f * CL;
    float ZI = fmaxf(Z1, v_ZB1);

    float HC = 0, BC = 0, ZC = 0, UC = 0, VC = 0;
    if (NC >= 0 && NC < CELL) {
        HC = fmaxf(H[NC], HM1);
        BC = ZBC[NC];
        ZC = fmaxf(BC, Z[NC]);
        UC = U[NC]; VC = V[NC];
    }

    float f0 = 0, f1 = 0, f2 = 0, f3 = 0;

    if (KP >= 1 && KP <= 8 || KP >= 10) {
        // Boundary: wall
        f3 = HALF_G_c * H1 * H1;
    } else if (H1 <= HM1 && HC <= HM1) {
        // both dry
    } else if (ZI <= BC) {
        f0 = -C1_C_c * powf(HC, 1.5f);
        f1 = H1 * QL_u * fabsf(QL_u);
        f3 = HALF_G_c * H1 * H1;
    } else if (ZC <= ZBC[pos]) {
        f0 = C1_C_c * powf(H1, 1.5f);
        f1 = H1 * fabsf(QL_u) * QL_u;
        f2 = H1 * fabsf(QL_u) * QL_v;
    } else if (H1 <= HM2) {
        if (ZC > ZI) {
            float DH = fmaxf(ZC - ZBC[pos], HM1);
            float UN = -C1_C_c * sqrtf(DH);
            f0 = DH * UN; f1 = f0 * UN;
            f2 = f0 * (VC * COSJ - UC * SINJ);
            f3 = HALF_G_c * H1 * H1;
        } else {
            f0 = C1_C_c * powf(H1, 1.5f);
            f3 = HALF_G_c * H1 * H1;
        }
    } else if (HC <= HM2) {
        if (ZI > ZC) {
            float DH = fmaxf(ZI - BC, HM1);
            float UN = C1_C_c * sqrtf(DH);
            float HC1 = ZC - ZBC[pos];
            f0 = DH * UN; f1 = f0 * UN; f2 = f0 * QL_v;
            f3 = HALF_G_c * HC1 * HC1;
        } else {
            f0 = -C1_C_c * powf(HC, 1.5f);
            f1 = H1 * QL_u * QL_u;
            f3 = HALF_G_c * H1 * H1;
        }
    } else {
        // Interior: OSHER
        if ((int)KP == 0 && pos < NC) {
            float QR_h = fmaxf(ZC - ZBC[pos], HM1);
            float UR = UC * COSJ + VC * SINJ;
            float ratio = fminf(HC / QR_h, 1.5f);
            float QR_u = UR * ratio;
            if (HC <= HM2 || QR_h <= HM2) QR_u = copysignf(VMIN_c, UR);
            float QR_v = VC * COSJ - UC * SINJ;
            float r0, r1, r2, r3;
            osher(QL_h, QL_u, QL_v, QR_h, QR_u, QR_v, FIL, H[pos], r0, r1, r2, r3);
            f0 = r0;
            f1 = r1 + (1.0f - ratio) * HC * UR * UR / 2.0f;
            f2 = r2; f3 = r3;
        } else {
            float COSJ1 = -COSJ, SINJ1 = -SINJ;
            float L1h = H[NC], L1u = U[NC]*COSJ1+V[NC]*SINJ1, L1v = V[NC]*COSJ1-U[NC]*SINJ1;
            float CL1 = sqrtf(G_c * H[NC]), FIL1 = L1u + 2.0f*CL1;
            float HC2 = fmaxf(H1, HM1), ZC1 = fmaxf(ZBC[pos], Z1);
            float R1h = fmaxf(ZC1 - ZBC[NC], HM1);
            float UR1 = U1*COSJ1 + V1*SINJ1;
            float ratio1 = fminf(HC2 / R1h, 1.5f);
            float R1u = UR1 * ratio1;
            if (HC2 <= HM2 || R1h <= HM2) R1u = copysignf(VMIN_c, UR1);
            float R1v = V1*COSJ1 - U1*SINJ1;
            float mr0, mr1, mr2, mr3;
            osher(L1h, L1u, L1v, R1h, R1u, R1v, FIL1, H[NC], mr0, mr1, mr2, mr3);
            f0 = -mr0;
            f1 = mr1 + (1.0f - ratio1) * HC2 * UR1 * UR1 / 2.0f;
            f2 = mr2;
            float ZA = sqrtf(mr3 / HALF_G_c) + BC;
            float HC3 = fmaxf(ZA - ZBC[pos], 0.0f);
            f3 = HALF_G_c * HC3 * HC3;
        }
    }
    FLUX0[idx] = f0; FLUX1[idx] = f1; FLUX2[idx] = f2; FLUX3[idx] = f3;
}

// ===== Kernel: update_cell (cell-parallel, CELL threads) =====
__global__
void update_cell(int CELL, float DT, float HM1, float HM2,
                 float* __restrict__ H, float* __restrict__ U,
                 float* __restrict__ V, float* __restrict__ Z,
                 float* __restrict__ W,
                 const float* __restrict__ ZBC, const float* __restrict__ AREA,
                 const float* __restrict__ FNC,
                 const float* __restrict__ SIDE,
                 const float* __restrict__ SLCOS, const float* __restrict__ SLSIN,
                 const float* __restrict__ FLUX0, const float* __restrict__ FLUX1,
                 const float* __restrict__ FLUX2, const float* __restrict__ FLUX3) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= CELL) return;

    float WH = 0, WU = 0, WV = 0;
    for (int i = 0; i < 4; i++) {
        int idx = 4 * pos + i;
        float FLR1 = FLUX1[idx] + FLUX3[idx];
        float FLR2 = FLUX2[idx];
        float SL = SIDE[idx], SLCA = SLCOS[idx], SLSA = SLSIN[idx];
        WH += SL * FLUX0[idx];
        WU += SLCA * FLR1 - SLSA * FLR2;
        WV += SLSA * FLR1 + SLCA * FLR2;
    }
    float H1 = H[pos], U1 = U[pos], V1 = V[pos];
    float DTA = DT / AREA[pos];
    float H2 = fmaxf(H1 - DTA * WH + QLUA_c, HM1);
    float Z2 = H2 + ZBC[pos];
    float U2 = 0, V2 = 0;
    if (H2 > HM1) {
        if (H2 <= HM2) {
            U2 = copysignf(fminf(VMIN_c, fabsf(U1)), U1);
            V2 = copysignf(fminf(VMIN_c, fabsf(V1)), V1);
        } else {
            float WSF = FNC[pos] * sqrtf(U1*U1+V1*V1) / powf(H1, 0.33333f);
            U2 = (H1*U1 - DTA*WU - DT*WSF*U1) / H2;
            V2 = (H1*V1 - DTA*WV - DT*WSF*V1) / H2;
            U2 = copysignf(fminf(fabsf(U2), 15.0f), U2);
            V2 = copysignf(fminf(fabsf(V2), 15.0f), V2);
        }
    }
    H[pos] = H2; U[pos] = U2; V[pos] = V2;
    Z[pos] = Z2; W[pos] = sqrtf(U2*U2 + V2*V2);
}

// ===== Persistent kernel: fused flux + update with grid sync =====
__global__
void persistent_fused(int CELL, float DT, float HM1, float HM2, int steps,
                      float* __restrict__ H, float* __restrict__ U,
                      float* __restrict__ V, float* __restrict__ Z,
                      float* __restrict__ W,
                      const float* __restrict__ ZBC, const float* __restrict__ ZB1,
                      const float* __restrict__ AREA, const float* __restrict__ FNC,
                      const int* __restrict__ NAC, const float* __restrict__ KLAS,
                      const float* __restrict__ SIDE,
                      const float* __restrict__ COSF, const float* __restrict__ SINF,
                      const float* __restrict__ SLCOS, const float* __restrict__ SLSIN,
                      float* __restrict__ FLUX0, float* __restrict__ FLUX1,
                      float* __restrict__ FLUX2, float* __restrict__ FLUX3) {
    cg::grid_group grid = cg::this_grid();
    int nSides = CELL * 4;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int step = 0; step < steps; step++) {
        // Phase 1: calculate_flux (edge-parallel)
        for (int idx = tid; idx < nSides; idx += total_threads) {
            int pos = idx / 4;

            float H1 = H[pos], U1 = U[pos], V1 = V[pos], Z1 = Z[pos];
            float v_ZB1 = ZB1[pos];
            int NC = NAC[idx] - 1;
            float KP = KLAS[idx];
            float COSJ = COSF[idx], SINJ = SINF[idx];

            float QL_h = H1;
            float QL_u = U1 * COSJ + V1 * SINJ;
            float QL_v = V1 * COSJ - U1 * SINJ;
            float CL = sqrtf(G_c * H1);
            float FIL = QL_u + 2.0f * CL;
            float ZI = fmaxf(Z1, v_ZB1);

            float HC = 0, BC = 0, ZC = 0, UC = 0, VC = 0;
            if (NC >= 0 && NC < CELL) {
                HC = fmaxf(H[NC], HM1);
                BC = ZBC[NC];
                ZC = fmaxf(BC, Z[NC]);
                UC = U[NC]; VC = V[NC];
            }

            float f0 = 0, f1 = 0, f2 = 0, f3 = 0;

            if (KP >= 1 && KP <= 8 || KP >= 10) {
                f3 = HALF_G_c * H1 * H1;
            } else if (H1 <= HM1 && HC <= HM1) {
                // dry
            } else if (ZI <= BC) {
                f0 = -C1_C_c * powf(HC, 1.5f);
                f1 = H1 * QL_u * fabsf(QL_u);
                f3 = HALF_G_c * H1 * H1;
            } else if (ZC <= ZBC[pos]) {
                f0 = C1_C_c * powf(H1, 1.5f);
                f1 = H1 * fabsf(QL_u) * QL_u;
                f2 = H1 * fabsf(QL_u) * QL_v;
            } else if (H1 <= HM2) {
                if (ZC > ZI) {
                    float DH = fmaxf(ZC - ZBC[pos], HM1);
                    float UN = -C1_C_c * sqrtf(DH);
                    f0 = DH * UN; f1 = f0 * UN;
                    f2 = f0 * (VC * COSJ - UC * SINJ);
                    f3 = HALF_G_c * H1 * H1;
                } else {
                    f0 = C1_C_c * powf(H1, 1.5f);
                    f3 = HALF_G_c * H1 * H1;
                }
            } else if (HC <= HM2) {
                if (ZI > ZC) {
                    float DH = fmaxf(ZI - BC, HM1);
                    float UN = C1_C_c * sqrtf(DH);
                    float HC1 = ZC - ZBC[pos];
                    f0 = DH * UN; f1 = f0 * UN; f2 = f0 * QL_v;
                    f3 = HALF_G_c * HC1 * HC1;
                } else {
                    f0 = -C1_C_c * powf(HC, 1.5f);
                    f1 = H1 * QL_u * QL_u;
                    f3 = HALF_G_c * H1 * H1;
                }
            } else {
                if ((int)KP == 0 && pos < NC) {
                    float QR_h = fmaxf(ZC - ZBC[pos], HM1);
                    float UR = UC * COSJ + VC * SINJ;
                    float ratio = fminf(HC / QR_h, 1.5f);
                    float QR_u = UR * ratio;
                    if (HC <= HM2 || QR_h <= HM2) QR_u = copysignf(VMIN_c, UR);
                    float QR_v = VC * COSJ - UC * SINJ;
                    float r0, r1, r2, r3;
                    osher(QL_h, QL_u, QL_v, QR_h, QR_u, QR_v, FIL, H[pos], r0, r1, r2, r3);
                    f0 = r0;
                    f1 = r1 + (1.0f - ratio) * HC * UR * UR / 2.0f;
                    f2 = r2; f3 = r3;
                } else {
                    float COSJ1 = -COSJ, SINJ1 = -SINJ;
                    float L1h = H[NC], L1u = U[NC]*COSJ1+V[NC]*SINJ1, L1v = V[NC]*COSJ1-U[NC]*SINJ1;
                    float CL1 = sqrtf(G_c * H[NC]), FIL1 = L1u + 2.0f*CL1;
                    float HC2 = fmaxf(H1, HM1), ZC1 = fmaxf(ZBC[pos], Z1);
                    float R1h = fmaxf(ZC1 - ZBC[NC], HM1);
                    float UR1 = U1*COSJ1 + V1*SINJ1;
                    float ratio1 = fminf(HC2 / R1h, 1.5f);
                    float R1u = UR1 * ratio1;
                    if (HC2 <= HM2 || R1h <= HM2) R1u = copysignf(VMIN_c, UR1);
                    float R1v = V1*COSJ1 - U1*SINJ1;
                    float mr0, mr1, mr2, mr3;
                    osher(L1h, L1u, L1v, R1h, R1u, R1v, FIL1, H[NC], mr0, mr1, mr2, mr3);
                    f0 = -mr0;
                    f1 = mr1 + (1.0f - ratio1) * HC2 * UR1 * UR1 / 2.0f;
                    f2 = mr2;
                    float ZA = sqrtf(mr3 / HALF_G_c) + BC;
                    float HC3 = fmaxf(ZA - ZBC[pos], 0.0f);
                    f3 = HALF_G_c * HC3 * HC3;
                }
            }
            FLUX0[idx] = f0; FLUX1[idx] = f1; FLUX2[idx] = f2; FLUX3[idx] = f3;
        }

        grid.sync();

        // Phase 2: update_cell (cell-parallel)
        for (int pos = tid; pos < CELL; pos += total_threads) {
            float WH = 0, WU = 0, WV = 0;
            for (int i = 0; i < 4; i++) {
                int sidx = 4 * pos + i;
                float FLR1 = FLUX1[sidx] + FLUX3[sidx];
                float FLR2 = FLUX2[sidx];
                float SL = SIDE[sidx], SLCA = SLCOS[sidx], SLSA = SLSIN[sidx];
                WH += SL * FLUX0[sidx];
                WU += SLCA * FLR1 - SLSA * FLR2;
                WV += SLSA * FLR1 + SLCA * FLR2;
            }
            float H1 = H[pos], U1 = U[pos], V1 = V[pos];
            float DTA = DT / AREA[pos];
            float H2 = fmaxf(H1 - DTA * WH + QLUA_c, HM1);
            float Z2 = H2 + ZBC[pos];
            float U2 = 0, V2 = 0;
            if (H2 > HM1) {
                if (H2 <= HM2) {
                    U2 = copysignf(fminf(VMIN_c, fabsf(U1)), U1);
                    V2 = copysignf(fminf(VMIN_c, fabsf(V1)), V1);
                } else {
                    float WSF = FNC[pos] * sqrtf(U1*U1+V1*V1) / powf(H1, 0.33333f);
                    U2 = (H1*U1 - DTA*WU - DT*WSF*U1) / H2;
                    V2 = (H1*V1 - DTA*WV - DT*WSF*V1) / H2;
                    U2 = copysignf(fminf(fabsf(U2), 15.0f), U2);
                    V2 = copysignf(fminf(fabsf(V2), 15.0f), V2);
                }
            }
            H[pos] = H2; U[pos] = U2; V[pos] = V2;
            Z[pos] = Z2; W[pos] = sqrtf(U2*U2 + V2*V2);
        }

        // Sync before next step
        if (step < steps - 1)
            grid.sync();
    }
}

// ===== Binary loader =====
template<typename T>
void loadBinary(const std::string& path, T* dst, int count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "ERROR: Cannot open %s\n", path.c_str()); exit(1); }
    f.read(reinterpret_cast<char*>(dst), count * sizeof(T));
}

// ===== State snapshot for resetting between benchmarks =====
struct MeshState {
    std::vector<float> H, U, V, Z, W;
};

void saveState(float* d_H, float* d_U, float* d_V, float* d_Z, float* d_W, int CELL, MeshState& s) {
    s.H.resize(CELL); s.U.resize(CELL); s.V.resize(CELL); s.Z.resize(CELL); s.W.resize(CELL);
    cudaMemcpy(s.H.data(), d_H, CELL*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(s.U.data(), d_U, CELL*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(s.V.data(), d_V, CELL*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(s.Z.data(), d_Z, CELL*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(s.W.data(), d_W, CELL*sizeof(float), cudaMemcpyDeviceToHost);
}

void restoreState(float* d_H, float* d_U, float* d_V, float* d_Z, float* d_W, int CELL, const MeshState& s) {
    cudaMemcpy(d_H, s.H.data(), CELL*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, s.U.data(), CELL*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, s.V.data(), CELL*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Z, s.Z.data(), CELL*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, s.W.data(), CELL*sizeof(float), cudaMemcpyHostToDevice);
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

int main(int argc, char* argv[]) {
    // ===== Parse params =====
    // Usage: hydro_osher [steps] [repeat] [data_dir]
    std::string binDir = "/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/benchmark/F2_hydro_refactored/data/binary/";
    if (argc > 3) binDir = std::string(argv[3]);
    std::ifstream pf(binDir + "params.txt");
    if (!pf) { fprintf(stderr, "Cannot open %sparams.txt\n", binDir.c_str()); return 1; }
    std::string line;
    std::getline(pf, line); int CELL = std::stoi(line);
    std::getline(pf, line); float HM1 = std::stof(line);
    std::getline(pf, line); float HM2 = std::stof(line);
    std::getline(pf, line); float DT = std::stof(line);
    std::getline(pf, line); int steps_per_day = std::stoi(line);
    pf.close();

    int steps = (argc > 1) ? atoi(argv[1]) : steps_per_day;
    int repeat = (argc > 2) ? atoi(argv[2]) : 10;
    int nSides = CELL * 4;

    printf("=== F2 Hydro-Cal OSHER Benchmark (CUDA) ===\n");
    printf("CELL=%d, HM1=%.6f, HM2=%.6f, DT=%.1f, steps=%d, repeat=%d\n",
           CELL, HM1, HM2, DT, steps, repeat);

    // ===== GPU info =====
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s, SMs=%d, maxThreadsPerSM=%d\n",
           prop.name, prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor);

    // ===== Allocate host arrays =====
    std::vector<float> h_H(CELL), h_U(CELL), h_V(CELL), h_Z(CELL), h_W(CELL);
    std::vector<float> h_ZBC(CELL), h_ZB1(CELL), h_AREA(CELL), h_FNC(CELL);
    std::vector<int>   h_NAC(nSides);
    std::vector<float> h_KLAS(nSides), h_SIDE(nSides);
    std::vector<float> h_COSF(nSides), h_SINF(nSides);
    std::vector<float> h_SLCOS(nSides), h_SLSIN(nSides);

    // ===== Load binary data =====
    loadBinary(binDir + "H.bin",    h_H.data(), CELL);
    loadBinary(binDir + "U.bin",    h_U.data(), CELL);
    loadBinary(binDir + "V.bin",    h_V.data(), CELL);
    loadBinary(binDir + "Z.bin",    h_Z.data(), CELL);
    loadBinary(binDir + "W.bin",    h_W.data(), CELL);
    loadBinary(binDir + "ZBC.bin",  h_ZBC.data(), CELL);
    loadBinary(binDir + "ZB1.bin",  h_ZB1.data(), CELL);
    loadBinary(binDir + "AREA.bin", h_AREA.data(), CELL);
    loadBinary(binDir + "FNC.bin",  h_FNC.data(), CELL);
    loadBinary(binDir + "NAC.bin",  h_NAC.data(), nSides);
    loadBinary(binDir + "KLAS.bin", h_KLAS.data(), nSides);
    loadBinary(binDir + "SIDE.bin", h_SIDE.data(), nSides);
    loadBinary(binDir + "COSF.bin", h_COSF.data(), nSides);
    loadBinary(binDir + "SINF.bin", h_SINF.data(), nSides);
    loadBinary(binDir + "SLCOS.bin", h_SLCOS.data(), nSides);
    loadBinary(binDir + "SLSIN.bin", h_SLSIN.data(), nSides);
    printf("Mesh loaded: %d cells, %d sides\n", CELL, nSides);

    // ===== Allocate device arrays =====
    float *d_H, *d_U, *d_V, *d_Z, *d_W;
    float *d_ZBC, *d_ZB1, *d_AREA, *d_FNC;
    int   *d_NAC;
    float *d_KLAS, *d_SIDE, *d_COSF, *d_SINF, *d_SLCOS, *d_SLSIN;
    float *d_FLUX0, *d_FLUX1, *d_FLUX2, *d_FLUX3;

    CUDA_CHECK(cudaMalloc(&d_H,    CELL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_U,    CELL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V,    CELL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Z,    CELL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W,    CELL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ZBC,  CELL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ZB1,  CELL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_AREA, CELL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_FNC,  CELL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_NAC,  nSides * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_KLAS, nSides * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_SIDE, nSides * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_COSF, nSides * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_SINF, nSides * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_SLCOS, nSides * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_SLSIN, nSides * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_FLUX0, nSides * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_FLUX1, nSides * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_FLUX2, nSides * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_FLUX3, nSides * sizeof(float)));

    // Copy constant data to device
    CUDA_CHECK(cudaMemcpy(d_ZBC,  h_ZBC.data(),  CELL * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ZB1,  h_ZB1.data(),  CELL * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_AREA, h_AREA.data(),  CELL * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_FNC,  h_FNC.data(),   CELL * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_NAC,  h_NAC.data(),   nSides * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_KLAS, h_KLAS.data(),  nSides * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_SIDE, h_SIDE.data(),  nSides * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_COSF, h_COSF.data(),  nSides * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_SINF, h_SINF.data(),  nSides * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_SLCOS, h_SLCOS.data(), nSides * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_SLSIN, h_SLSIN.data(), nSides * sizeof(float), cudaMemcpyHostToDevice));

    // Copy mutable state
    auto uploadState = [&]() {
        CUDA_CHECK(cudaMemcpy(d_H, h_H.data(), CELL * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), CELL * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), CELL * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Z, h_Z.data(), CELL * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), CELL * sizeof(float), cudaMemcpyHostToDevice));
    };
    uploadState();

    // ===== Kernel launch config =====
    int fluxThreads = 256;
    int fluxBlocks = (nSides + fluxThreads - 1) / fluxThreads;
    int updateThreads = 256;
    int updateBlocks = (CELL + updateThreads - 1) / updateThreads;

    printf("Flux kernel:   %d blocks x %d threads = %d (need %d)\n", fluxBlocks, fluxThreads, fluxBlocks*fluxThreads, nSides);
    printf("Update kernel: %d blocks x %d threads = %d (need %d)\n", updateBlocks, updateThreads, updateBlocks*updateThreads, CELL);

    // ===== Helper: run one benchmark =====
    auto benchmarkSync = [&](const char* name) -> double {
        // Warmup
        uploadState();
        for (int s = 0; s < 5; s++) {
            calculate_flux<<<fluxBlocks, fluxThreads>>>(CELL, HM1, HM2, d_H, d_U, d_V, d_Z, d_ZBC, d_ZB1, d_NAC, d_KLAS, d_SIDE, d_COSF, d_SINF, d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3);
            update_cell<<<updateBlocks, updateThreads>>>(CELL, DT, HM1, HM2, d_H, d_U, d_V, d_Z, d_W, d_ZBC, d_AREA, d_FNC, d_SIDE, d_SLCOS, d_SLSIN, d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t a0, a1;
        CUDA_CHECK(cudaEventCreate(&a0));
        CUDA_CHECK(cudaEventCreate(&a1));
        std::vector<double> times;
        for (int r = 0; r < repeat; r++) {
            uploadState();
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(a0));
            for (int s = 0; s < steps; s++) {
                calculate_flux<<<fluxBlocks, fluxThreads>>>(CELL, HM1, HM2, d_H, d_U, d_V, d_Z, d_ZBC, d_ZB1, d_NAC, d_KLAS, d_SIDE, d_COSF, d_SINF, d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3);
                update_cell<<<updateBlocks, updateThreads>>>(CELL, DT, HM1, HM2, d_H, d_U, d_V, d_Z, d_W, d_ZBC, d_AREA, d_FNC, d_SIDE, d_SLCOS, d_SLSIN, d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3);
            }
            CUDA_CHECK(cudaEventRecord(a1));
            CUDA_CHECK(cudaEventSynchronize(a1));
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, a0, a1));
            times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        double median = times[repeat / 2];
        CUDA_CHECK(cudaEventDestroy(a0));
        CUDA_CHECK(cudaEventDestroy(a1));
        double per_step = median / steps * 1000.0;  // us/step
        printf("[%s] %d steps: median=%.3f ms, %.2f us/step\n", name, steps, median, per_step);
        return median;
    };

    // ===== Strategy 1: Sync loop =====
    printf("\n--- Strategy 1: Sync Loop (cudaDeviceSynchronize per step) ---\n");
    {
        uploadState();
        // Warmup
        for (int s = 0; s < 5; s++) {
            calculate_flux<<<fluxBlocks, fluxThreads>>>(CELL, HM1, HM2, d_H, d_U, d_V, d_Z, d_ZBC, d_ZB1, d_NAC, d_KLAS, d_SIDE, d_COSF, d_SINF, d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3);
            update_cell<<<updateBlocks, updateThreads>>>(CELL, DT, HM1, HM2, d_H, d_U, d_V, d_Z, d_W, d_ZBC, d_AREA, d_FNC, d_SIDE, d_SLCOS, d_SLSIN, d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        std::vector<double> times;
        for (int r = 0; r < repeat; r++) {
            uploadState();
            CUDA_CHECK(cudaDeviceSynchronize());
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int s = 0; s < steps; s++) {
                calculate_flux<<<fluxBlocks, fluxThreads>>>(CELL, HM1, HM2, d_H, d_U, d_V, d_Z, d_ZBC, d_ZB1, d_NAC, d_KLAS, d_SIDE, d_COSF, d_SINF, d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3);
                CUDA_CHECK(cudaDeviceSynchronize());
                update_cell<<<updateBlocks, updateThreads>>>(CELL, DT, HM1, HM2, d_H, d_U, d_V, d_Z, d_W, d_ZBC, d_AREA, d_FNC, d_SIDE, d_SLCOS, d_SLSIN, d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            double ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        double median = times[repeat / 2];
        printf("[Sync Loop] %d steps: median=%.3f ms, %.2f us/step\n", steps, median, median / steps * 1000.0);
    }

    // ===== Strategy 2: Async loop (no sync between kernels) =====
    printf("\n--- Strategy 2: Async Loop (no intermediate sync) ---\n");
    benchmarkSync("Async Loop");

    // ===== Strategy 3: CUDA Graph =====
    printf("\n--- Strategy 3: CUDA Graph ---\n");
    {
        uploadState();
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // Capture graph
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;

        // Full capture: all steps in one graph (no per-step host overhead)
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int s = 0; s < steps; s++) {
            calculate_flux<<<fluxBlocks, fluxThreads, 0, stream>>>(CELL, HM1, HM2, d_H, d_U, d_V, d_Z, d_ZBC, d_ZB1, d_NAC, d_KLAS, d_SIDE, d_COSF, d_SINF, d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3);
            update_cell<<<updateBlocks, updateThreads, 0, stream>>>(CELL, DT, HM1, HM2, d_H, d_U, d_V, d_Z, d_W, d_ZBC, d_AREA, d_FNC, d_SIDE, d_SLCOS, d_SLSIN, d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3);
        }
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

        size_t numNodes;
        CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &numNodes));
        printf("Graph: %zu nodes (%d steps x 2 kernels)\n", numNodes, steps);

        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

        // Warmup
        uploadState();
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        cudaEvent_t g0, g1;
        CUDA_CHECK(cudaEventCreate(&g0));
        CUDA_CHECK(cudaEventCreate(&g1));
        std::vector<double> times;
        for (int r = 0; r < repeat; r++) {
            uploadState();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaEventRecord(g0, stream));
            CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
            CUDA_CHECK(cudaEventRecord(g1, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, g0, g1));
            times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        double median = times[repeat / 2];
        printf("[CUDA Graph] %d steps: median=%.3f ms, %.2f us/step\n", steps, median, median / steps * 1000.0);
        CUDA_CHECK(cudaEventDestroy(g0));
        CUDA_CHECK(cudaEventDestroy(g1));

        CUDA_CHECK(cudaGraphExecDestroy(graphExec));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    // ===== Strategy 3b: Device Graph (tail launch) =====
    printf("\n--- Strategy 3b: Device Graph (tail launch) ---\n");
    {
        int *d_steps_dg;
        CUDA_CHECK(cudaMalloc(&d_steps_dg, sizeof(int)));
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;

        // Capture ONE step: calculate_flux + update_cell
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        calculate_flux<<<fluxBlocks, fluxThreads, 0, stream>>>(CELL, HM1, HM2, d_H, d_U, d_V, d_Z, d_ZBC, d_ZB1, d_NAC, d_KLAS, d_SIDE, d_COSF, d_SINF, d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3);
        update_cell<<<updateBlocks, updateThreads, 0, stream>>>(CELL, DT, HM1, HM2, d_H, d_U, d_V, d_Z, d_W, d_ZBC, d_AREA, d_FNC, d_SIDE, d_SLCOS, d_SLSIN, d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3);
        tail_launch_kernel<<<1, 1, 0, stream>>>(d_steps_dg);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

        // Instantiate for device-side launch
        CUDA_CHECK(cudaGraphInstantiateWithFlags(&graphExec, graph,
              cudaGraphInstantiateFlagDeviceLaunch));
        CUDA_CHECK(cudaGraphUpload(graphExec, stream));

        // Copy graph exec handle to device symbol
        cudaGraphExec_t* d_sym_ptr;
        CUDA_CHECK(cudaGetSymbolAddress((void**)&d_sym_ptr, d_graph_exec));
        CUDA_CHECK(cudaMemcpy(d_sym_ptr, &graphExec, sizeof(cudaGraphExec_t),
              cudaMemcpyHostToDevice));

        // Warmup
        uploadState();
        for (int w = 0; w < 5; w++) {
            int sv = steps;
            CUDA_CHECK(cudaMemcpy(d_steps_dg, &sv, sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        std::vector<double> times;
        for (int r = 0; r < repeat; r++) {
            uploadState();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto t0 = std::chrono::high_resolution_clock::now();
            int sv = steps;
            CUDA_CHECK(cudaMemcpy(d_steps_dg, &sv, sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            double ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        double median = times[repeat / 2];
        printf("[DevGraph] %d steps: median=%.3f ms, %.2f us/step\n", steps, median, median / steps * 1000.0);

        CUDA_CHECK(cudaGraphExecDestroy(graphExec));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFree(d_steps_dg));
    }

    // ===== Strategy 4: Persistent Kernel (cooperative launch) =====
    printf("\n--- Strategy 4: Persistent Kernel (cooperative grid sync) ---\n");
    {
        int persistentThreads = 256;
        int maxBlocksPerSM = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM, persistent_fused, persistentThreads, 0));
        int maxBlocks = maxBlocksPerSM * prop.multiProcessorCount;

        // We need enough threads to cover nSides (the larger kernel)
        int neededBlocks = (nSides + persistentThreads - 1) / persistentThreads;
        int persistentBlocks = std::min(maxBlocks, neededBlocks);

        printf("Persistent: %d blocks x %d threads (max occupancy: %d blocks/SM, total max: %d)\n",
               persistentBlocks, persistentThreads, maxBlocksPerSM, maxBlocks);

        // Check cooperative launch support
        int supportsCoopLaunch = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, 0));

        if (supportsCoopLaunch && maxBlocksPerSM > 0) {
            void* kernelArgs[] = {
                &CELL, &DT, &HM1, &HM2, &steps,
                &d_H, &d_U, &d_V, &d_Z, &d_W,
                &d_ZBC, &d_ZB1, &d_AREA, &d_FNC,
                &d_NAC, &d_KLAS, &d_SIDE,
                &d_COSF, &d_SINF, &d_SLCOS, &d_SLSIN,
                &d_FLUX0, &d_FLUX1, &d_FLUX2, &d_FLUX3
            };

            // Warmup
            uploadState();
            CUDA_CHECK(cudaLaunchCooperativeKernel(
                (void*)persistent_fused,
                dim3(persistentBlocks), dim3(persistentThreads),
                kernelArgs));
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<double> times;
            for (int r = 0; r < repeat; r++) {
                uploadState();
                CUDA_CHECK(cudaDeviceSynchronize());

                // Need to pass steps as kernel arg each time
                auto t0 = std::chrono::high_resolution_clock::now();
                CUDA_CHECK(cudaLaunchCooperativeKernel(
                    (void*)persistent_fused,
                    dim3(persistentBlocks), dim3(persistentThreads),
                    kernelArgs));
                CUDA_CHECK(cudaDeviceSynchronize());
                double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t0).count();
                times.push_back(ms);
            }
            std::sort(times.begin(), times.end());
            double median = times[repeat / 2];
            printf("[Persistent] %d steps: median=%.3f ms, %.2f us/step\n", steps, median, median / steps * 1000.0);
        } else {
            printf("[Persistent] SKIPPED: cooperative launch not supported or occupancy=0\n");
        }
    }

    // ===== Overhead breakdown: measure single-step components =====
    printf("\n--- Overhead Breakdown (single step timing with CUDA events) ---\n");
    {
        uploadState();
        cudaEvent_t e0, e1, e2, e3, e4;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventCreate(&e2));
        CUDA_CHECK(cudaEventCreate(&e3));
        CUDA_CHECK(cudaEventCreate(&e4));

        // Warmup
        for (int w = 0; w < 10; w++) {
            calculate_flux<<<fluxBlocks, fluxThreads>>>(CELL, HM1, HM2, d_H, d_U, d_V, d_Z, d_ZBC, d_ZB1, d_NAC, d_KLAS, d_SIDE, d_COSF, d_SINF, d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3);
            update_cell<<<updateBlocks, updateThreads>>>(CELL, DT, HM1, HM2, d_H, d_U, d_V, d_Z, d_W, d_ZBC, d_AREA, d_FNC, d_SIDE, d_SLCOS, d_SLSIN, d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Time 100 steps to get avg
        int N = 100;
        float flux_ms = 0, update_ms = 0;

        // Measure flux kernel
        CUDA_CHECK(cudaEventRecord(e0));
        for (int i = 0; i < N; i++)
            calculate_flux<<<fluxBlocks, fluxThreads>>>(CELL, HM1, HM2, d_H, d_U, d_V, d_Z, d_ZBC, d_ZB1, d_NAC, d_KLAS, d_SIDE, d_COSF, d_SINF, d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&flux_ms, e0, e1));

        // Measure update kernel
        CUDA_CHECK(cudaEventRecord(e2));
        for (int i = 0; i < N; i++)
            update_cell<<<updateBlocks, updateThreads>>>(CELL, DT, HM1, HM2, d_H, d_U, d_V, d_Z, d_W, d_ZBC, d_AREA, d_FNC, d_SIDE, d_SLCOS, d_SLSIN, d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3);
        CUDA_CHECK(cudaEventRecord(e3));
        CUDA_CHECK(cudaEventSynchronize(e3));
        CUDA_CHECK(cudaEventElapsedTime(&update_ms, e2, e3));

        printf("Flux kernel:   %.2f us/call\n", flux_ms / N * 1000.0);
        printf("Update kernel: %.2f us/call\n", update_ms / N * 1000.0);
        printf("GPU total:     %.2f us/step\n", (flux_ms + update_ms) / N * 1000.0);

        CUDA_CHECK(cudaEventDestroy(e0));
        CUDA_CHECK(cudaEventDestroy(e1));
        CUDA_CHECK(cudaEventDestroy(e2));
        CUDA_CHECK(cudaEventDestroy(e3));
        CUDA_CHECK(cudaEventDestroy(e4));
    }

    // ===== Dump final state for correctness validation =====
    // Run N steps with Async and dump H to binary
    {
        uploadState();
        for (int s = 0; s < steps; s++) {
            calculate_flux<<<fluxBlocks, fluxThreads>>>(CELL, HM1, HM2, d_H, d_U, d_V, d_Z, d_ZBC, d_ZB1, d_NAC, d_KLAS, d_SIDE, d_COSF, d_SINF, d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3);
            update_cell<<<updateBlocks, updateThreads>>>(CELL, DT, HM1, HM2, d_H, d_U, d_V, d_Z, d_W, d_ZBC, d_AREA, d_FNC, d_SIDE, d_SLCOS, d_SLSIN, d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> h_out(CELL);
        cudaMemcpy(h_out.data(), d_H, CELL*sizeof(float), cudaMemcpyDeviceToHost);
        // Write to /tmp/cuda_H.bin
        FILE* fp = fopen("/tmp/cuda_H.bin", "wb");
        if (fp) {
            fwrite(h_out.data(), sizeof(float), CELL, fp);
            fclose(fp);
            printf("\nDumped CUDA H[%d] to /tmp/cuda_H.bin after %d steps\n", CELL, steps);
            printf("H range: [%.6f, %.6f]\n", *std::min_element(h_out.begin(), h_out.end()),
                   *std::max_element(h_out.begin(), h_out.end()));
        }
    }

    // ===== Summary =====
    printf("\n=== Summary ===\n");
    printf("Benchmark complete. Compare us/step across strategies to see launch overhead impact.\n");
    printf("For register tuning, recompile with: nvcc -O3 -arch=sm_90 -rdc=true -maxrregcount=64 ...\n");

    // Cleanup
    cudaFree(d_H); cudaFree(d_U); cudaFree(d_V); cudaFree(d_Z); cudaFree(d_W);
    cudaFree(d_ZBC); cudaFree(d_ZB1); cudaFree(d_AREA); cudaFree(d_FNC);
    cudaFree(d_NAC); cudaFree(d_KLAS); cudaFree(d_SIDE);
    cudaFree(d_COSF); cudaFree(d_SINF); cudaFree(d_SLCOS); cudaFree(d_SLSIN);
    cudaFree(d_FLUX0); cudaFree(d_FLUX1); cudaFree(d_FLUX2); cudaFree(d_FLUX3);

    return 0;
}
