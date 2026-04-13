/**
 * F1: Shallow Water Equations — CUDA benchmark with OSHER Riemann solver.
 * Tests 4 strategies: Sync loop, Async loop, CUDA Graph, Persistent Kernel.
 * Loads real binary mesh data from F1_hydro_shallow_water/data/binary/.
 *
 * F1 differences from F2:
 *   - fp64 (double) throughout
 *   - Single monolithic kernel per step (cell-parallel, 1 thread per cell)
 *   - Data layout [5*(CEL+1)], 1-indexed (index 0 unused)
 *   - Each cell processes 4 edges internally (NAC[1..4][cell], KLAS[1..4][cell])
 *
 * Build: nvcc -O3 -arch=sm_90 -rdc=true hydro_f1_benchmark.cu -o hydro_f1 -lcudadevrt
 * Usage: ./hydro_f1 [steps] [repeat] [data_dir]
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

// ===== Constants (fp64) =====
__constant__ double G_c     = 9.81;
__constant__ double HALF_G_c = 4.905;
__constant__ double C1_C_c  = 1.7;
__constant__ double VMIN_c  = 0.001;

// Host-side constants (kept for reference)
// constexpr double G     = 9.81;
// constexpr double HALF_G = 4.905;
// constexpr double C1_C  = 1.7;
// constexpr double VMIN  = 0.001;

// ===== Device: QF flux function (fp64) =====
__device__ __forceinline__
void QF(double h, double u, double v, double& F0, double& F1, double& F2, double& F3) {
    F0 = h * u;
    F1 = F0 * u;
    F2 = F0 * v;
    F3 = HALF_G_c * h * h;
}

// ===== Device: OSHER Riemann solver (all 16 K1xK2 cases, fp64) =====
__device__ __forceinline__
void osher(double QL_h, double QL_u, double QL_v,
           double QR_h, double QR_u, double QR_v,
           double FIL_in, double H_pos,
           double& R0, double& R1, double& R2, double& R3) {
    double CR = sqrt(G_c * QR_h);
    double FIR_v = QR_u - 2.0 * CR;
    double fil = FIL_in, fir = FIR_v;
    double UA = (fil + fir) / 2.0;
    double CA = fabs((fil - fir) / 4.0);
    double CL_v = sqrt(G_c * H_pos);
    R0 = R1 = R2 = R3 = 0.0;

    int K2 = (CA < UA) ? 1 : (UA >= 0 && UA < CA) ? 2 : (UA >= -CA && UA < 0) ? 3 : 4;
    int K1 = (QL_u < CL_v && QR_u >= -CR) ? 1 :
             (QL_u >= CL_v && QR_u >= -CR) ? 2 :
             (QL_u < CL_v && QR_u < -CR) ? 3 : 4;

    #define ADD(hh, uu, vv, ss) { \
        double _f0, _f1, _f2, _f3; QF(hh, uu, vv, _f0, _f1, _f2, _f3); \
        R0 += _f0*(ss); R1 += _f1*(ss); R2 += _f2*(ss); R3 += _f3*(ss); }
    #define QS1(ss) ADD(QL_h, QL_u, QL_v, ss)
    #define QS2(ss) { double _U=fil/3.0, _H=_U*_U/G_c; ADD(_H, _U, QL_v, ss) }
    #define QS3(ss) { double _ua=(fil+fir)/2.0; double _fl=fil-_ua; double _H=_fl*_fl/(4.0*G_c); ADD(_H, _ua, QL_v, ss) }
    #define QS5(ss) { double _ua=(fil+fir)/2.0; double _fr=fir-_ua; double _H=_fr*_fr/(4.0*G_c); ADD(_H, _ua, QR_v, ss) }
    #define QS6(ss) { double _U=fir/3.0, _H=_U*_U/G_c; ADD(_H, _U, QR_v, ss) }
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

// ===== Kernel: monolithic shallow water step (cell-parallel, 1 thread per cell) =====
// F1 layout: arrays are [5*(CEL+1)], 1-indexed.
//   Edge j of cell i: index = j*(CEL+1) + i, where j=1..4
//   Cell arrays: index i (1-based), index 0 unused.
__global__
void shallow_water_step(int CEL, double DT, double HM1, double HM2,
                        double* __restrict__ H, double* __restrict__ U,
                        double* __restrict__ V, double* __restrict__ Z,
                        double* __restrict__ W,
                        const double* __restrict__ ZBC, const double* __restrict__ ZB1,
                        const double* __restrict__ AREA, const double* __restrict__ FNC,
                        const int* __restrict__ NAC, const int* __restrict__ KLAS,
                        const double* __restrict__ SIDE,
                        const double* __restrict__ COSF, const double* __restrict__ SINF,
                        const double* __restrict__ SLCOS, const double* __restrict__ SLSIN,
                        const int* __restrict__ NV) {
    // 1-indexed: cells are 1..CEL
    int pos = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (pos > CEL) return;

    int stride = CEL + 1;  // stride for [5][CEL+1] layout

    double H1 = H[pos];
    double U1 = U[pos];
    double V1 = V[pos];
    double BI = ZBC[pos];

    double HI = fmax(H1, HM1);
    double UI = U1;
    double VI = V1;
    if (HI <= HM2) {
        UI = (UI >= 0.0) ? VMIN_c : -VMIN_c;
        VI = (VI >= 0.0) ? VMIN_c : -VMIN_c;
    }
    double ZI = fmax(Z[pos], ZBC[pos]);

    double WH = 0.0, WU = 0.0, WV = 0.0;

    int nv = NV[pos];  // number of edges for this cell (typically 4)
    for (int j = 1; j <= nv; j++) {
        int eidx = j * stride + pos;  // index into [5][CEL+1] edge arrays
        int NC = NAC[eidx];
        int KP = KLAS[eidx];
        double COSJ = COSF[eidx];
        double SINJ = SINF[eidx];

        double QL_h = HI;
        double QL_u = UI * COSJ + VI * SINJ;
        double QL_v = VI * COSJ - UI * SINJ;
        double CL_v = sqrt(G_c * HI);
        double FIL_v = QL_u + 2.0 * CL_v;

        double HC = 0.0, BC = 0.0, ZC = 0.0, UC = 0.0, VC = 0.0;
        if (NC != 0) {
            HC = fmax(H[NC], HM1);
            BC = ZBC[NC];
            ZC = fmax(ZBC[NC], Z[NC]);
            UC = U[NC];
            VC = V[NC];
        }

        double f0 = 0.0, f1 = 0.0, f2 = 0.0, f3 = 0.0;

        if (KP == 4) {
            // Wall boundary
            f3 = HALF_G_c * H1 * H1;
        } else if (KP != 0) {
            // Other boundary types
            f3 = HALF_G_c * H1 * H1;
        } else if (HI <= HM1 && HC <= HM1) {
            // Both dry — no flux
        } else if (ZI <= BC) {
            f0 = -C1_C_c * pow(HC, 1.5);
            f1 = HI * QL_u * fabs(QL_u);
            f3 = HALF_G_c * HI * HI;
        } else if (ZC <= BI) {
            f0 = C1_C_c * pow(HI, 1.5);
            f1 = HI * fabs(QL_u) * QL_u;
            f2 = HI * fabs(QL_u) * QL_v;
        } else if (HI <= HM2) {
            if (ZC > ZI) {
                double DH = fmax(ZC - ZBC[pos], HM1);
                double UN = -C1_C_c * sqrt(DH);
                f0 = DH * UN;
                f1 = f0 * UN;
                f2 = f0 * (VC * COSJ - UC * SINJ);
                f3 = HALF_G_c * HI * HI;
            } else {
                f0 = C1_C_c * pow(HI, 1.5);
                f3 = HALF_G_c * HI * HI;
            }
        } else if (HC <= HM2) {
            if (ZI > ZC) {
                double DH = fmax(ZI - BC, HM1);
                double UN = C1_C_c * sqrt(DH);
                double HC1 = ZC - ZBC[pos];
                f0 = DH * UN;
                f1 = f0 * UN;
                f2 = f0 * QL_v;
                f3 = HALF_G_c * HC1 * HC1;
            } else {
                f0 = -C1_C_c * pow(HC, 1.5);
                f1 = HI * QL_u * QL_u;
                f3 = HALF_G_c * HI * HI;
            }
        } else {
            // Both wet — Osher Riemann solver
            if (pos < NC) {
                double QR_h = fmax(ZC - ZBC[pos], HM1);
                double UR = UC * COSJ + VC * SINJ;
                double ratio = fmin(HC / QR_h, 1.5);
                double QR_u = UR * ratio;
                if (HC <= HM2 || QR_h <= HM2)
                    QR_u = (UR >= 0.0) ? VMIN_c : -VMIN_c;
                double QR_v = VC * COSJ - UC * SINJ;
                double r0, r1, r2, r3;
                osher(QL_h, QL_u, QL_v, QR_h, QR_u, QR_v, FIL_v, H[pos], r0, r1, r2, r3);
                f0 = r0;
                f1 = r1 + (1.0 - ratio) * HC * UR * UR / 2.0;
                f2 = r2;
                f3 = r3;
            } else {
                double COSJ1 = -COSJ, SINJ1 = -SINJ;
                double L1h = H[NC];
                double L1u = U[NC] * COSJ1 + V[NC] * SINJ1;
                double L1v = V[NC] * COSJ1 - U[NC] * SINJ1;
                double CL1 = sqrt(G_c * H[NC]);
                double FIL1 = L1u + 2.0 * CL1;
                double HC2 = fmax(H1, HM1);
                double ZC1 = fmax(ZBC[pos], ZI);
                double R1h = fmax(ZC1 - ZBC[NC], HM1);
                double UR1 = UI * COSJ1 + VI * SINJ1;
                double ratio1 = fmin(HC2 / R1h, 1.5);
                double R1u = UR1 * ratio1;
                if (HC2 <= HM2 || R1h <= HM2)
                    R1u = (UR1 >= 0.0) ? VMIN_c : -VMIN_c;
                double R1v = VI * COSJ1 - UI * SINJ1;
                double mr0, mr1, mr2, mr3;
                osher(L1h, L1u, L1v, R1h, R1u, R1v, FIL1, H[NC], mr0, mr1, mr2, mr3);
                f0 = -mr0;
                f1 = mr1 + (1.0 - ratio1) * HC2 * UR1 * UR1 / 2.0;
                f2 = mr2;
                double ZA = sqrt(mr3 / HALF_G_c) + BC;
                double HC3 = fmax(ZA - ZBC[pos], 0.0);
                f3 = HALF_G_c * HC3 * HC3;
            }
        }

        // Accumulate fluxes
        double SL = SIDE[eidx];
        double SLCA = SLCOS[eidx];
        double SLSA = SLSIN[eidx];
        double FLR_1 = f1 + f3;
        double FLR_2 = f2;
        WH += SL * f0;
        WU += SLCA * FLR_1 - SLSA * FLR_2;
        WV += SLSA * FLR_1 + SLCA * FLR_2;
    }

    // State update with Manning friction
    double DTA = DT / AREA[pos];
    double WDTA = DTA;
    double H2 = fmax(H1 - WDTA * WH, HM1);
    double Z2 = H2 + BI;

    double U2 = 0.0, V2 = 0.0;
    if (H2 > HM1) {
        if (H2 <= HM2) {
            U2 = copysign(fmin(VMIN_c, fabs(U1)), U1);
            V2 = copysign(fmin(VMIN_c, fabs(V1)), V1);
        } else {
            double QX1 = H1 * U1;
            double QY1 = H1 * V1;
            double DTAU = WDTA * WU;
            double DTAV = WDTA * WV;
            double WSF = FNC[pos] * sqrt(U1 * U1 + V1 * V1) / pow(H1, 0.33333);
            U2 = (QX1 - DTAU - DT * WSF * U1) / H2;
            V2 = (QY1 - DTAV - DT * WSF * V1) / H2;
            if (H2 > HM2) {
                U2 = copysign(fmin(fabs(U2), 15.0), U2);
                V2 = copysign(fmin(fabs(V2), 15.0), V2);
            }
        }
    }

    H[pos] = H2;
    U[pos] = U2;
    V[pos] = V2;
    Z[pos] = Z2;
    W[pos] = sqrt(U2 * U2 + V2 * V2);
}

// ===== Persistent kernel: monolithic step with grid_sync between steps =====
__global__
void persistent_shallow_water(int CEL, double DT, double HM1, double HM2, int steps,
                              double* __restrict__ H, double* __restrict__ U,
                              double* __restrict__ V, double* __restrict__ Z,
                              double* __restrict__ W,
                              const double* __restrict__ ZBC, const double* __restrict__ ZB1,
                              const double* __restrict__ AREA, const double* __restrict__ FNC,
                              const int* __restrict__ NAC, const int* __restrict__ KLAS,
                              const double* __restrict__ SIDE,
                              const double* __restrict__ COSF, const double* __restrict__ SINF,
                              const double* __restrict__ SLCOS, const double* __restrict__ SLSIN,
                              const int* __restrict__ NV) {
    cg::grid_group grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    int stride = CEL + 1;

    for (int step = 0; step < steps; step++) {
        // Each thread handles one or more cells via grid-stride loop
        for (int pos = tid + 1; pos <= CEL; pos += total_threads) {
            double H1 = H[pos];
            double U1 = U[pos];
            double V1 = V[pos];
            double BI = ZBC[pos];

            double HI = fmax(H1, HM1);
            double UI = U1;
            double VI = V1;
            if (HI <= HM2) {
                UI = (UI >= 0.0) ? VMIN_c : -VMIN_c;
                VI = (VI >= 0.0) ? VMIN_c : -VMIN_c;
            }
            double ZI = fmax(Z[pos], ZBC[pos]);

            double WH = 0.0, WU = 0.0, WV = 0.0;

            int nv = NV[pos];
            for (int j = 1; j <= nv; j++) {
                int eidx = j * stride + pos;
                int NC = NAC[eidx];
                int KP = KLAS[eidx];
                double COSJ = COSF[eidx];
                double SINJ = SINF[eidx];

                double QL_h = HI;
                double QL_u = UI * COSJ + VI * SINJ;
                double QL_v = VI * COSJ - UI * SINJ;
                double CL_v = sqrt(G_c * HI);
                double FIL_v = QL_u + 2.0 * CL_v;

                double HC = 0.0, BC = 0.0, ZC = 0.0, UC = 0.0, VC = 0.0;
                if (NC != 0) {
                    HC = fmax(H[NC], HM1);
                    BC = ZBC[NC];
                    ZC = fmax(ZBC[NC], Z[NC]);
                    UC = U[NC];
                    VC = V[NC];
                }

                double f0 = 0.0, f1 = 0.0, f2 = 0.0, f3 = 0.0;

                if (KP == 4) {
                    f3 = HALF_G_c * H1 * H1;
                } else if (KP != 0) {
                    f3 = HALF_G_c * H1 * H1;
                } else if (HI <= HM1 && HC <= HM1) {
                    // dry
                } else if (ZI <= BC) {
                    f0 = -C1_C_c * pow(HC, 1.5);
                    f1 = HI * QL_u * fabs(QL_u);
                    f3 = HALF_G_c * HI * HI;
                } else if (ZC <= BI) {
                    f0 = C1_C_c * pow(HI, 1.5);
                    f1 = HI * fabs(QL_u) * QL_u;
                    f2 = HI * fabs(QL_u) * QL_v;
                } else if (HI <= HM2) {
                    if (ZC > ZI) {
                        double DH = fmax(ZC - ZBC[pos], HM1);
                        double UN = -C1_C_c * sqrt(DH);
                        f0 = DH * UN;
                        f1 = f0 * UN;
                        f2 = f0 * (VC * COSJ - UC * SINJ);
                        f3 = HALF_G_c * HI * HI;
                    } else {
                        f0 = C1_C_c * pow(HI, 1.5);
                        f3 = HALF_G_c * HI * HI;
                    }
                } else if (HC <= HM2) {
                    if (ZI > ZC) {
                        double DH = fmax(ZI - BC, HM1);
                        double UN = C1_C_c * sqrt(DH);
                        double HC1 = ZC - ZBC[pos];
                        f0 = DH * UN;
                        f1 = f0 * UN;
                        f2 = f0 * QL_v;
                        f3 = HALF_G_c * HC1 * HC1;
                    } else {
                        f0 = -C1_C_c * pow(HC, 1.5);
                        f1 = HI * QL_u * QL_u;
                        f3 = HALF_G_c * HI * HI;
                    }
                } else {
                    if (pos < NC) {
                        double QR_h = fmax(ZC - ZBC[pos], HM1);
                        double UR = UC * COSJ + VC * SINJ;
                        double ratio = fmin(HC / QR_h, 1.5);
                        double QR_u = UR * ratio;
                        if (HC <= HM2 || QR_h <= HM2)
                            QR_u = (UR >= 0.0) ? VMIN_c : -VMIN_c;
                        double QR_v = VC * COSJ - UC * SINJ;
                        double r0, r1, r2, r3;
                        osher(QL_h, QL_u, QL_v, QR_h, QR_u, QR_v, FIL_v, H[pos], r0, r1, r2, r3);
                        f0 = r0;
                        f1 = r1 + (1.0 - ratio) * HC * UR * UR / 2.0;
                        f2 = r2;
                        f3 = r3;
                    } else {
                        double COSJ1 = -COSJ, SINJ1 = -SINJ;
                        double L1h = H[NC];
                        double L1u = U[NC] * COSJ1 + V[NC] * SINJ1;
                        double L1v = V[NC] * COSJ1 - U[NC] * SINJ1;
                        double CL1 = sqrt(G_c * H[NC]);
                        double FIL1 = L1u + 2.0 * CL1;
                        double HC2 = fmax(H1, HM1);
                        double ZC1 = fmax(ZBC[pos], ZI);
                        double R1h = fmax(ZC1 - ZBC[NC], HM1);
                        double UR1 = UI * COSJ1 + VI * SINJ1;
                        double ratio1 = fmin(HC2 / R1h, 1.5);
                        double R1u = UR1 * ratio1;
                        if (HC2 <= HM2 || R1h <= HM2)
                            R1u = (UR1 >= 0.0) ? VMIN_c : -VMIN_c;
                        double R1v = VI * COSJ1 - UI * SINJ1;
                        double mr0, mr1, mr2, mr3;
                        osher(L1h, L1u, L1v, R1h, R1u, R1v, FIL1, H[NC], mr0, mr1, mr2, mr3);
                        f0 = -mr0;
                        f1 = mr1 + (1.0 - ratio1) * HC2 * UR1 * UR1 / 2.0;
                        f2 = mr2;
                        double ZA = sqrt(mr3 / HALF_G_c) + BC;
                        double HC3 = fmax(ZA - ZBC[pos], 0.0);
                        f3 = HALF_G_c * HC3 * HC3;
                    }
                }

                double SL = SIDE[eidx];
                double SLCA = SLCOS[eidx];
                double SLSA = SLSIN[eidx];
                double FLR_1 = f1 + f3;
                double FLR_2 = f2;
                WH += SL * f0;
                WU += SLCA * FLR_1 - SLSA * FLR_2;
                WV += SLSA * FLR_1 + SLCA * FLR_2;
            }

            double DTA = DT / AREA[pos];
            double WDTA = DTA;
            double H2 = fmax(H1 - WDTA * WH, HM1);
            double Z2 = H2 + BI;
            double U2 = 0.0, V2 = 0.0;
            if (H2 > HM1) {
                if (H2 <= HM2) {
                    U2 = copysign(fmin(VMIN_c, fabs(U1)), U1);
                    V2 = copysign(fmin(VMIN_c, fabs(V1)), V1);
                } else {
                    double QX1 = H1 * U1;
                    double QY1 = H1 * V1;
                    double DTAU = WDTA * WU;
                    double DTAV = WDTA * WV;
                    double WSF = FNC[pos] * sqrt(U1 * U1 + V1 * V1) / pow(H1, 0.33333);
                    U2 = (QX1 - DTAU - DT * WSF * U1) / H2;
                    V2 = (QY1 - DTAV - DT * WSF * V1) / H2;
                    if (H2 > HM2) {
                        U2 = copysign(fmin(fabs(U2), 15.0), U2);
                        V2 = copysign(fmin(fabs(V2), 15.0), V2);
                    }
                }
            }

            H[pos] = H2;
            U[pos] = U2;
            V[pos] = V2;
            Z[pos] = Z2;
            W[pos] = sqrt(U2 * U2 + V2 * V2);
        }

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

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

int main(int argc, char* argv[]) {
    // ===== Parse params =====
    // Usage: hydro_f1 [steps] [repeat] [data_dir]
    std::string binDir = "/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/benchmark/F1_hydro_shallow_water/data/binary/";
    if (argc > 3) binDir = std::string(argv[3]);
    // Ensure trailing slash
    if (binDir.back() != '/') binDir += '/';

    std::ifstream pf(binDir + "params.txt");
    if (!pf) { fprintf(stderr, "Cannot open %sparams.txt\n", binDir.c_str()); return 1; }
    std::string line;
    std::getline(pf, line); int CEL = std::stoi(line);
    std::getline(pf, line); int NOD = std::stoi(line);
    std::getline(pf, line); double HM1 = std::stod(line);
    std::getline(pf, line); double HM2 = std::stod(line);
    std::getline(pf, line); int NZ = std::stoi(line);
    std::getline(pf, line); int NQ = std::stoi(line);
    pf.close();

    int steps = (argc > 1) ? atoi(argv[1]) : 100;
    int repeat = (argc > 2) ? atoi(argv[2]) : 10;
    double DT = 1.0;  // Benchmark-only: timing, not correctness

    int cellSize = CEL + 1;   // 1-indexed, index 0 unused
    int edgeSize = 5 * cellSize;  // [5][CEL+1] layout

    printf("=== F1 Hydro Shallow Water OSHER Benchmark (CUDA, fp64) ===\n");
    printf("CEL=%d, NOD=%d, HM1=%.6f, HM2=%.6f, DT=%.1f, steps=%d, repeat=%d\n",
           CEL, NOD, HM1, HM2, DT, steps, repeat);

    // ===== GPU info =====
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s, SMs=%d, maxThreadsPerSM=%d\n",
           prop.name, prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor);

    // ===== Allocate host arrays =====
    // Cell arrays: [CEL+1] fp64
    std::vector<double> h_H(cellSize), h_U(cellSize), h_V(cellSize);
    std::vector<double> h_Z(cellSize), h_W(cellSize);
    std::vector<double> h_ZBC(cellSize), h_ZB1(cellSize);
    std::vector<double> h_AREA(cellSize), h_FNC(cellSize);
    // Edge arrays: [5*(CEL+1)] fp64
    std::vector<double> h_SIDE(edgeSize), h_COSF(edgeSize), h_SINF(edgeSize);
    std::vector<double> h_SLCOS(edgeSize), h_SLSIN(edgeSize);
    // Int arrays: [5*(CEL+1)] int32
    std::vector<int> h_NAC(edgeSize), h_KLAS(edgeSize);
    // Cell int array: [CEL+1] int32
    std::vector<int> h_NV(cellSize);

    // ===== Load binary data =====
    loadBinary(binDir + "H.bin",     h_H.data(),    cellSize);
    loadBinary(binDir + "U.bin",     h_U.data(),    cellSize);
    loadBinary(binDir + "V.bin",     h_V.data(),    cellSize);
    loadBinary(binDir + "Z.bin",     h_Z.data(),    cellSize);
    loadBinary(binDir + "W.bin",     h_W.data(),    cellSize);
    loadBinary(binDir + "ZBC.bin",   h_ZBC.data(),  cellSize);
    loadBinary(binDir + "ZB1.bin",   h_ZB1.data(),  cellSize);
    loadBinary(binDir + "AREA.bin",  h_AREA.data(), cellSize);
    loadBinary(binDir + "FNC.bin",   h_FNC.data(),  cellSize);
    loadBinary(binDir + "SIDE.bin",  h_SIDE.data(), edgeSize);
    loadBinary(binDir + "COSF.bin",  h_COSF.data(), edgeSize);
    loadBinary(binDir + "SINF.bin",  h_SINF.data(), edgeSize);
    loadBinary(binDir + "SLCOS.bin", h_SLCOS.data(), edgeSize);
    loadBinary(binDir + "SLSIN.bin", h_SLSIN.data(), edgeSize);
    loadBinary(binDir + "NAC.bin",   h_NAC.data(),  edgeSize);
    loadBinary(binDir + "KLAS.bin",  h_KLAS.data(), edgeSize);
    loadBinary(binDir + "NV.bin",    h_NV.data(),   cellSize);
    printf("Mesh loaded: %d cells, edge arrays %d entries\n", CEL, edgeSize);

    // ===== Allocate device arrays =====
    double *d_H, *d_U, *d_V, *d_Z, *d_W;
    double *d_ZBC, *d_ZB1, *d_AREA, *d_FNC;
    double *d_SIDE, *d_COSF, *d_SINF, *d_SLCOS, *d_SLSIN;
    int *d_NAC, *d_KLAS, *d_NV;

    CUDA_CHECK(cudaMalloc(&d_H,     cellSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_U,     cellSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_V,     cellSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Z,     cellSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W,     cellSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ZBC,   cellSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ZB1,   cellSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_AREA,  cellSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_FNC,   cellSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_SIDE,  edgeSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_COSF,  edgeSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_SINF,  edgeSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_SLCOS, edgeSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_SLSIN, edgeSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_NAC,   edgeSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_KLAS,  edgeSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_NV,    cellSize * sizeof(int)));

    // Copy constant data to device
    CUDA_CHECK(cudaMemcpy(d_ZBC,   h_ZBC.data(),   cellSize * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ZB1,   h_ZB1.data(),   cellSize * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_AREA,  h_AREA.data(),  cellSize * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_FNC,   h_FNC.data(),   cellSize * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_SIDE,  h_SIDE.data(),  edgeSize * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_COSF,  h_COSF.data(),  edgeSize * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_SINF,  h_SINF.data(),  edgeSize * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_SLCOS, h_SLCOS.data(), edgeSize * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_SLSIN, h_SLSIN.data(), edgeSize * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_NAC,   h_NAC.data(),   edgeSize * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_KLAS,  h_KLAS.data(),  edgeSize * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_NV,    h_NV.data(),    cellSize * sizeof(int),    cudaMemcpyHostToDevice));

    // Upload mutable state
    auto uploadState = [&]() {
        CUDA_CHECK(cudaMemcpy(d_H, h_H.data(), cellSize * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), cellSize * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), cellSize * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Z, h_Z.data(), cellSize * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), cellSize * sizeof(double), cudaMemcpyHostToDevice));
    };
    uploadState();

    // ===== Kernel launch config =====
    int blockSize = 256;
    int gridSize = (CEL + blockSize - 1) / blockSize;

    printf("Kernel: %d blocks x %d threads = %d (need %d cells)\n",
           gridSize, blockSize, gridSize * blockSize, CEL);

    // Macro for launching the monolithic step kernel
    #define LAUNCH_STEP() \
        shallow_water_step<<<gridSize, blockSize>>>(CEL, DT, HM1, HM2, \
            d_H, d_U, d_V, d_Z, d_W, d_ZBC, d_ZB1, d_AREA, d_FNC, \
            d_NAC, d_KLAS, d_SIDE, d_COSF, d_SINF, d_SLCOS, d_SLSIN, d_NV)

    #define LAUNCH_STEP_STREAM(s) \
        shallow_water_step<<<gridSize, blockSize, 0, s>>>(CEL, DT, HM1, HM2, \
            d_H, d_U, d_V, d_Z, d_W, d_ZBC, d_ZB1, d_AREA, d_FNC, \
            d_NAC, d_KLAS, d_SIDE, d_COSF, d_SINF, d_SLCOS, d_SLSIN, d_NV)

    // ===== Strategy 1: Sync Loop (cudaDeviceSynchronize after each step) =====
    printf("\n--- Strategy 1: Sync Loop (cudaDeviceSynchronize per step) ---\n");
    {
        uploadState();
        // Warmup
        for (int s = 0; s < 5; s++) {
            LAUNCH_STEP();
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        std::vector<double> times;
        for (int r = 0; r < repeat; r++) {
            uploadState();
            CUDA_CHECK(cudaDeviceSynchronize());
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int s = 0; s < steps; s++) {
                LAUNCH_STEP();
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            double ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        double median = times[repeat / 2];
        printf("[Sync Loop] %d steps: median=%.3f ms, %.2f us/step\n",
               steps, median, median / steps * 1000.0);
    }

    // ===== Strategy 2: Async Loop (no intermediate sync) =====
    printf("\n--- Strategy 2: Async Loop (no intermediate sync) ---\n");
    {
        uploadState();
        // Warmup
        for (int s = 0; s < 5; s++)
            LAUNCH_STEP();
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> times;
        for (int r = 0; r < repeat; r++) {
            uploadState();
            CUDA_CHECK(cudaDeviceSynchronize());
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int s = 0; s < steps; s++)
                LAUNCH_STEP();
            CUDA_CHECK(cudaDeviceSynchronize());
            double ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        double median = times[repeat / 2];
        printf("[Async Loop] %d steps: median=%.3f ms, %.2f us/step\n",
               steps, median, median / steps * 1000.0);
    }

    // ===== Strategy 3: CUDA Graph =====
    printf("\n--- Strategy 3: CUDA Graph ---\n");
    {
        uploadState();
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // Capture graph for one step
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;

        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        LAUNCH_STEP_STREAM(stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

        // Warmup
        uploadState();
        for (int s = 0; s < 5; s++)
            CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<double> times;
        for (int r = 0; r < repeat; r++) {
            uploadState();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int s = 0; s < steps; s++)
                CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            double ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        double median = times[repeat / 2];
        printf("[CUDA Graph] %d steps: median=%.3f ms, %.2f us/step\n",
               steps, median, median / steps * 1000.0);

        CUDA_CHECK(cudaGraphExecDestroy(graphExec));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    // ===== Strategy 4: Persistent Kernel (cooperative launch) =====
    printf("\n--- Strategy 4: Persistent Kernel (cooperative grid sync) ---\n");
    {
        int persistentThreads = 256;
        int maxBlocksPerSM = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM, persistent_shallow_water, persistentThreads, 0));
        int maxBlocks = maxBlocksPerSM * prop.multiProcessorCount;

        int neededBlocks = (CEL + persistentThreads - 1) / persistentThreads;
        int persistentBlocks = std::min(maxBlocks, neededBlocks);

        printf("Persistent: %d blocks x %d threads (max occupancy: %d blocks/SM, total max: %d)\n",
               persistentBlocks, persistentThreads, maxBlocksPerSM, maxBlocks);

        int supportsCoopLaunch = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, 0));

        if (supportsCoopLaunch && maxBlocksPerSM > 0) {
            void* kernelArgs[] = {
                &CEL, &DT, &HM1, &HM2, &steps,
                &d_H, &d_U, &d_V, &d_Z, &d_W,
                &d_ZBC, &d_ZB1, &d_AREA, &d_FNC,
                &d_NAC, &d_KLAS, &d_SIDE,
                &d_COSF, &d_SINF, &d_SLCOS, &d_SLSIN,
                &d_NV
            };

            // Warmup
            uploadState();
            CUDA_CHECK(cudaLaunchCooperativeKernel(
                (void*)persistent_shallow_water,
                dim3(persistentBlocks), dim3(persistentThreads),
                kernelArgs));
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<double> times;
            for (int r = 0; r < repeat; r++) {
                uploadState();
                CUDA_CHECK(cudaDeviceSynchronize());

                auto t0 = std::chrono::high_resolution_clock::now();
                CUDA_CHECK(cudaLaunchCooperativeKernel(
                    (void*)persistent_shallow_water,
                    dim3(persistentBlocks), dim3(persistentThreads),
                    kernelArgs));
                CUDA_CHECK(cudaDeviceSynchronize());
                double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t0).count();
                times.push_back(ms);
            }
            std::sort(times.begin(), times.end());
            double median = times[repeat / 2];
            printf("[Persistent] %d steps: median=%.3f ms, %.2f us/step\n",
                   steps, median, median / steps * 1000.0);
        } else {
            printf("[Persistent] SKIPPED: cooperative launch not supported or occupancy=0\n");
        }
    }

    // ===== Overhead breakdown: single step timing with CUDA events =====
    printf("\n--- Overhead Breakdown (single step timing with CUDA events) ---\n");
    {
        uploadState();
        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));

        // Warmup
        for (int w = 0; w < 10; w++)
            LAUNCH_STEP();
        CUDA_CHECK(cudaDeviceSynchronize());

        // Time N steps to get avg
        int N = 100;
        float kernel_ms = 0;

        CUDA_CHECK(cudaEventRecord(e0));
        for (int i = 0; i < N; i++)
            LAUNCH_STEP();
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, e0, e1));

        printf("Monolithic kernel: %.2f us/call\n", kernel_ms / N * 1000.0);
        printf("GPU total:         %.2f us/step (single kernel per step)\n", kernel_ms / N * 1000.0);

        CUDA_CHECK(cudaEventDestroy(e0));
        CUDA_CHECK(cudaEventDestroy(e1));
    }

    // ===== Summary =====
    printf("\n=== Summary ===\n");
    printf("F1 Benchmark complete. Compare us/step across strategies to see launch overhead impact.\n");
    printf("F1 uses fp64, single monolithic kernel, cell-parallel (1 thread/cell).\n");
    printf("For register tuning, recompile with: nvcc -O3 -arch=sm_90 -rdc=true -maxrregcount=64 ...\n");

    // Cleanup
    cudaFree(d_H); cudaFree(d_U); cudaFree(d_V); cudaFree(d_Z); cudaFree(d_W);
    cudaFree(d_ZBC); cudaFree(d_ZB1); cudaFree(d_AREA); cudaFree(d_FNC);
    cudaFree(d_SIDE); cudaFree(d_COSF); cudaFree(d_SINF);
    cudaFree(d_SLCOS); cudaFree(d_SLSIN);
    cudaFree(d_NAC); cudaFree(d_KLAS); cudaFree(d_NV);

    #undef LAUNCH_STEP
    #undef LAUNCH_STEP_STREAM

    return 0;
}
