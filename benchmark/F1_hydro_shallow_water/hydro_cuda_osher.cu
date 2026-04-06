/**
 * 2D Shallow Water Equations (Osher Riemann solver) — CUDA benchmark.
 *
 * Exact port of hydro_taichi.py (dam-break on structured quad mesh).
 *
 * Compile:
 *   nvcc -O3 -arch=sm_86 -rdc=true hydro_cuda_osher.cu -o hydro_cuda_osher -lcudadevrt
 *
 * Four launch strategies:
 *   [1] Sync loop:  kernel launch + cudaDeviceSynchronize each step
 *   [2] Async loop:  kernel launches without per-step sync
 *   [3] CUDA Graph:  capture + replay
 *   [4] Persistent:  cooperative launch with grid_sync (fused kernels)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// Constants (match Taichi exactly)
// ---------------------------------------------------------------------------
static constexpr double G         = 9.81;
static constexpr double HALF_G    = 4.905;
static constexpr double HM1       = 0.001;
static constexpr double HM2       = 0.01;
static constexpr double VMIN      = 0.001;
static constexpr double C0        = 1.0;
static constexpr double C1        = 0.3;
static constexpr double MANNING_N = 0.03;

#define CHECK_CUDA(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// QF helper (inline on device)
// ---------------------------------------------------------------------------
struct Vec4 { double x, y, z, w; };
struct Vec3 { double x, y, z; };

__device__ __forceinline__ Vec4 QF(double h, double u, double v) {
    double f0 = h * u;
    return {f0, f0 * u, f0 * v, HALF_G * h * h};
}

__device__ __forceinline__ Vec4 vec4_add(Vec4 a, Vec4 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

__device__ __forceinline__ Vec4 vec4_sub(Vec4 a, Vec4 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

// ---------------------------------------------------------------------------
// Osher Riemann solver (exact port of Taichi osher())
// ---------------------------------------------------------------------------
__device__ Vec4 osher(Vec3 QL, Vec3 QR, double FIL_in, double H_pos) {
    double CR = sqrt(G * QR.x);
    double FIR_v = QR.y - 2.0 * CR;
    double UA = (FIL_in + FIR_v) / 2.0;
    double CA = fabs((FIL_in - FIR_v) / 4.0);
    double CL_v = sqrt(G * H_pos);

    Vec4 FLR = {0.0, 0.0, 0.0, 0.0};

    int K2 = 0;
    if (CA < UA)                       K2 = 1;
    else if (UA >= 0.0 && UA < CA)     K2 = 2;
    else if (UA >= -CA && UA < 0.0)    K2 = 3;
    else                               K2 = 4;

    int K1 = 0;
    if (QL.y < CL_v && QR.y >= -CR)        K1 = 1;
    else if (QL.y >= CL_v && QR.y >= -CR)   K1 = 2;
    else if (QL.y < CL_v && QR.y < -CR)     K1 = 3;
    else                                     K1 = 4;

    double fil = FIL_in;
    double fir = FIR_v;

    double US, HS, ua_, HA, US2, HS2, US6, HS6, US6b, HS6b;

    if (K1 == 1) {
        if (K2 == 1) {
            US = fil / 3.0; HS = US * US / G;
            FLR = QF(HS, US, QL.z);
        } else if (K2 == 2) {
            ua_ = (fil + fir) / 2.0; fil = fil - ua_; HA = fil * fil / (4.0 * G);
            FLR = QF(HA, ua_, QL.z);
        } else if (K2 == 3) {
            ua_ = (fil + fir) / 2.0; fir = fir - ua_; HA = fir * fir / (4.0 * G);
            FLR = QF(HA, ua_, QR.z);
        } else {
            US = fir / 3.0; HS = US * US / G;
            FLR = QF(HS, US, QR.z);
        }
    } else if (K1 == 2) {
        if (K2 == 1) {
            FLR = QF(QL.x, QL.y, QL.z);
        } else if (K2 == 2) {
            FLR = QF(QL.x, QL.y, QL.z);
            US2 = fil / 3.0; HS2 = US2 * US2 / G;
            FLR = vec4_sub(FLR, QF(HS2, US2, QL.z));
            ua_ = (fil + fir) / 2.0; fil = fil - ua_; HA = fil * fil / (4.0 * G);
            FLR = vec4_add(FLR, QF(HA, ua_, QL.z));
        } else if (K2 == 3) {
            FLR = QF(QL.x, QL.y, QL.z);
            US2 = fil / 3.0; HS2 = US2 * US2 / G;
            FLR = vec4_sub(FLR, QF(HS2, US2, QL.z));
            ua_ = (fil + fir) / 2.0; fir = fir - ua_; HA = fir * fir / (4.0 * G);
            FLR = vec4_add(FLR, QF(HA, ua_, QR.z));
        } else {
            FLR = QF(QL.x, QL.y, QL.z);
            US2 = fil / 3.0; HS2 = US2 * US2 / G;
            FLR = vec4_sub(FLR, QF(HS2, US2, QL.z));
            US6 = fir / 3.0; HS6 = US6 * US6 / G;
            FLR = vec4_add(FLR, QF(HS6, US6, QR.z));
        }
    } else if (K1 == 3) {
        if (K2 == 1) {
            US2 = fil / 3.0; HS2 = US2 * US2 / G;
            FLR = QF(HS2, US2, QL.z);
            US6 = fir / 3.0; HS6 = US6 * US6 / G;
            FLR = vec4_sub(FLR, QF(HS6, US6, QR.z));
            FLR = vec4_add(FLR, QF(QR.x, QR.y, QR.z));
        } else if (K2 == 2) {
            ua_ = (fil + fir) / 2.0; fil = fil - ua_; HA = fil * fil / (4.0 * G);
            FLR = QF(HA, ua_, QL.z);
            US6 = fir / 3.0; HS6 = US6 * US6 / G;
            FLR = vec4_sub(FLR, QF(HS6, US6, QR.z));
            FLR = vec4_add(FLR, QF(QR.x, QR.y, QR.z));
        } else if (K2 == 3) {
            ua_ = (fil + fir) / 2.0; fir = fir - ua_; HA = fir * fir / (4.0 * G);
            FLR = QF(HA, ua_, QR.z);
            US6b = fir / 3.0; HS6b = US6b * US6b / G;
            FLR = vec4_sub(FLR, QF(HS6b, US6b, QR.z));
            FLR = vec4_add(FLR, QF(QR.x, QR.y, QR.z));
        } else {
            FLR = QF(QR.x, QR.y, QR.z);
        }
    } else { // K1 == 4
        if (K2 == 1) {
            FLR = QF(QL.x, QL.y, QL.z);
            US6 = fir / 3.0; HS6 = US6 * US6 / G;
            FLR = vec4_sub(FLR, QF(HS6, US6, QR.z));
            FLR = vec4_add(FLR, QF(QR.x, QR.y, QR.z));
        } else if (K2 == 2) {
            FLR = QF(QL.x, QL.y, QL.z);
            US2 = fil / 3.0; HS2 = US2 * US2 / G;
            FLR = vec4_sub(FLR, QF(HS2, US2, QL.z));
            ua_ = (fil + fir) / 2.0; fil = fil - ua_; HA = fil * fil / (4.0 * G);
            FLR = vec4_add(FLR, QF(HA, ua_, QL.z));
            US6 = fir / 3.0; HS6 = US6 * US6 / G;
            FLR = vec4_sub(FLR, QF(HS6, US6, QR.z));
            FLR = vec4_add(FLR, QF(QR.x, QR.y, QR.z));
        } else if (K2 == 3) {
            FLR = QF(QL.x, QL.y, QL.z);
            US2 = fil / 3.0; HS2 = US2 * US2 / G;
            FLR = vec4_sub(FLR, QF(HS2, US2, QL.z));
            ua_ = (fil + fir) / 2.0; fir = fir - ua_; HA = fir * fir / (4.0 * G);
            FLR = vec4_add(FLR, QF(HA, ua_, QR.z));
            US6 = fir / 3.0; HS6 = US6 * US6 / G;
            FLR = vec4_sub(FLR, QF(HS6, US6, QR.z));
            FLR = vec4_add(FLR, QF(QR.x, QR.y, QR.z));
        } else {
            FLR = QF(QL.x, QL.y, QL.z);
            US2 = fil / 3.0; HS2 = US2 * US2 / G;
            FLR = vec4_sub(FLR, QF(HS2, US2, QL.z));
            FLR = vec4_add(FLR, QF(QR.x, QR.y, QR.z));
        }
    }

    return FLR;
}

// ---------------------------------------------------------------------------
// Kernel 1: shallow_water_step  (exact port of Taichi kernel)
// ---------------------------------------------------------------------------
__global__ void shallow_water_step(
    int CEL, double DT,
    const int* __restrict__ NAC,        // [5*(CEL+1)]
    const int* __restrict__ KLAS,
    const double* __restrict__ SIDE,
    const double* __restrict__ COSF,
    const double* __restrict__ SINF,
    const double* __restrict__ SLCOS,
    const double* __restrict__ SLSIN,
    const double* __restrict__ AREA,
    const double* __restrict__ ZBC,
    const double* __restrict__ FNC,
    const double* __restrict__ H_pre,
    const double* __restrict__ U_pre,
    const double* __restrict__ V_pre,
    const double* __restrict__ Z_pre,
    const double* __restrict__ W_pre,
    double* __restrict__ H_res,
    double* __restrict__ U_res,
    double* __restrict__ V_res,
    double* __restrict__ Z_res,
    double* __restrict__ W_res)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x + 1; // 1-indexed
    if (pos > CEL) return;

    int stride = CEL + 1; // stride for 2D arrays [5][CEL+1]

    double H1 = H_pre[pos];
    double U1 = U_pre[pos];
    double V1 = V_pre[pos];
    double BI = ZBC[pos];

    double HI = fmax(H1, HM1);
    double UI = U1;
    double VI = V1;
    if (HI <= HM2) {
        UI = (UI >= 0.0) ? VMIN : -VMIN;
        VI = (VI >= 0.0) ? VMIN : -VMIN;
    }
    double ZI = fmax(Z_pre[pos], ZBC[pos]);

    double WH = 0.0, WU = 0.0, WV = 0.0;

    // 4 edges (j=1..4)
    for (int j = 1; j <= 4; j++) {
        int idx = j * stride + pos;
        int NC  = NAC[idx];
        int KP  = KLAS[idx];
        double COSJ = COSF[idx];
        double SINJ = SINF[idx];

        Vec3 QL = {HI, UI * COSJ + VI * SINJ, VI * COSJ - UI * SINJ};
        double CL_v = sqrt(G * HI);
        double FIL_v = QL.y + 2.0 * CL_v;

        double HC = 0.0, BC = 0.0, ZC = 0.0, UC = 0.0, VC = 0.0;
        if (NC != 0) {
            HC = fmax(H_pre[NC], HM1);
            BC = ZBC[NC];
            ZC = fmax(ZBC[NC], Z_pre[NC]);
            UC = U_pre[NC];
            VC = V_pre[NC];
        }

        double flux0 = 0.0, flux1 = 0.0, flux2 = 0.0, flux3 = 0.0;

        if (KP == 4) {
            flux3 = HALF_G * H1 * H1;
        } else if (KP != 0) {
            flux3 = HALF_G * H1 * H1;
        } else if (HI <= HM1 && HC <= HM1) {
            // pass
        } else if (ZI <= BC) {
            flux0 = -C1 * pow(HC, 1.5);
            flux1 = HI * QL.y * fabs(QL.y);
            flux3 = HALF_G * HI * HI;
        } else if (ZC <= BI) {
            flux0 = C1 * pow(HI, 1.5);
            flux1 = HI * fabs(QL.y) * QL.y;
            flux2 = HI * fabs(QL.y) * QL.z;
        } else if (HI <= HM2) {
            if (ZC > ZI) {
                double DH = fmax(ZC - ZBC[pos], HM1);
                double UN = -C1 * sqrt(DH);
                flux0 = DH * UN;
                flux1 = flux0 * UN;
                flux2 = flux0 * (VC * COSJ - UC * SINJ);
                flux3 = HALF_G * HI * HI;
            } else {
                flux0 = C1 * pow(HI, 1.5);
                flux3 = HALF_G * HI * HI;
            }
        } else if (HC <= HM2) {
            if (ZI > ZC) {
                double DH = fmax(ZI - BC, HM1);
                double UN = C1 * sqrt(DH);
                double HC1 = ZC - ZBC[pos];
                flux0 = DH * UN;
                flux1 = flux0 * UN;
                flux2 = flux0 * QL.z;
                flux3 = HALF_G * HC1 * HC1;
            } else {
                flux0 = -C1 * pow(HC, 1.5);
                flux1 = HI * QL.y * QL.y;
                flux3 = HALF_G * HI * HI;
            }
        } else {
            // Both wet — Osher Riemann solver
            if (pos < NC) {
                double QR_h = fmax(ZC - ZBC[pos], HM1);
                double UR = UC * COSJ + VC * SINJ;
                double ratio = fmin(HC / QR_h, 1.5);
                double QR_u = UR * ratio;
                if (HC <= HM2 || QR_h <= HM2) {
                    QR_u = (UR >= 0.0) ? VMIN : -VMIN;
                }
                double QR_v_ = VC * COSJ - UC * SINJ;
                Vec3 QR_vec = {QR_h, QR_u, QR_v_};
                Vec4 FLR_OS = osher(QL, QR_vec, FIL_v, H_pre[pos]);
                flux0 = FLR_OS.x;
                flux1 = FLR_OS.y + (1.0 - ratio) * HC * UR * UR / 2.0;
                flux2 = FLR_OS.z;
                flux3 = FLR_OS.w;
            } else {
                double COSJ1 = -COSJ;
                double SINJ1 = -SINJ;
                Vec3 QL1 = {
                    H_pre[NC],
                    U_pre[NC] * COSJ1 + V_pre[NC] * SINJ1,
                    V_pre[NC] * COSJ1 - U_pre[NC] * SINJ1
                };
                double CL1 = sqrt(G * H_pre[NC]);
                double FIL1 = QL1.y + 2.0 * CL1;
                double HC2 = fmax(HI, HM1);
                double ZC1 = fmax(ZBC[pos], ZI);
                double QR1_h = fmax(ZC1 - ZBC[NC], HM1);
                double UR1 = UI * COSJ1 + VI * SINJ1;
                double ratio1 = fmin(HC2 / QR1_h, 1.5);
                double QR1_u = UR1 * ratio1;
                if (HC2 <= HM2 || QR1_h <= HM2) {
                    QR1_u = (UR1 >= 0.0) ? VMIN : -VMIN;
                }
                double QR1_v_ = VI * COSJ1 - UI * SINJ1;
                Vec3 QR1_vec = {QR1_h, QR1_u, QR1_v_};
                Vec4 FLR1 = osher(QL1, QR1_vec, FIL1, H_pre[NC]);
                flux0 = -FLR1.x;
                flux1 = FLR1.y + (1.0 - ratio1) * HC2 * UR1 * UR1 / 2.0;
                flux2 = FLR1.z;
                double ZA = sqrt(FLR1.w / HALF_G) + BC;
                double HC3 = fmax(ZA - ZBC[pos], 0.0);
                flux3 = HALF_G * HC3 * HC3;
            }
        }

        // Accumulate fluxes
        double SL   = SIDE[idx];
        double SLCA = SLCOS[idx];
        double SLSA = SLSIN[idx];
        double FLR_1 = flux1 + flux3;
        double FLR_2 = flux2;
        WH += SL * flux0;
        WU += SLCA * FLR_1 - SLSA * FLR_2;
        WV += SLSA * FLR_1 + SLCA * FLR_2;
    }

    // State update with Manning friction
    double DTA  = DT / AREA[pos];
    double WDTA = DTA;
    double H2   = fmax(H1 - WDTA * WH, HM1);
    double Z2   = H2 + BI;

    double U2 = 0.0, V2 = 0.0;
    if (H2 > HM1) {
        if (H2 <= HM2) {
            U2 = (U1 >= 0.0) ? fmin(VMIN, fabs(U1)) : -fmin(VMIN, fabs(U1));
            V2 = (V1 >= 0.0) ? fmin(VMIN, fabs(V1)) : -fmin(VMIN, fabs(V1));
        } else {
            double QX1  = H1 * U1;
            double QY1  = H1 * V1;
            double DTAU = WDTA * WU;
            double DTAV = WDTA * WV;
            double WSF  = FNC[pos] * sqrt(U1 * U1 + V1 * V1) / pow(H1, 0.33333);
            U2 = (QX1 - DTAU - DT * WSF * U1) / H2;
            V2 = (QY1 - DTAV - DT * WSF * V1) / H2;
            if (H2 > HM2) {
                U2 = (U2 >= 0.0) ? fmin(fabs(U2), 15.0) : -fmin(fabs(U2), 15.0);
                V2 = (V2 >= 0.0) ? fmin(fabs(V2), 15.0) : -fmin(fabs(V2), 15.0);
            }
        }
    }

    H_res[pos] = H2;
    U_res[pos] = U2;
    V_res[pos] = V2;
    Z_res[pos] = Z2;
    W_res[pos] = sqrt(U2 * U2 + V2 * V2);
}

// ---------------------------------------------------------------------------
// Kernel 2: transfer (copy res -> pre)
// ---------------------------------------------------------------------------
__global__ void transfer(
    int CEL,
    const double* __restrict__ H_res,
    const double* __restrict__ U_res,
    const double* __restrict__ V_res,
    const double* __restrict__ Z_res,
    const double* __restrict__ W_res,
    double* __restrict__ H_pre,
    double* __restrict__ U_pre,
    double* __restrict__ V_pre,
    double* __restrict__ Z_pre,
    double* __restrict__ W_pre)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (pos > CEL) return;
    H_pre[pos] = H_res[pos];
    U_pre[pos] = U_res[pos];
    V_pre[pos] = V_res[pos];
    Z_pre[pos] = Z_res[pos];
    W_pre[pos] = W_res[pos];
}

// ---------------------------------------------------------------------------
// Persistent kernel: fused shallow_water_step + grid_sync + transfer
// ---------------------------------------------------------------------------
__global__ void persistent_kernel(
    int CEL, double DT, int STEPS,
    int* __restrict__ NAC,
    int* __restrict__ KLAS,
    double* __restrict__ SIDE,
    double* __restrict__ COSF,
    double* __restrict__ SINF,
    double* __restrict__ SLCOS,
    double* __restrict__ SLSIN,
    double* __restrict__ AREA,
    double* __restrict__ ZBC,
    double* __restrict__ FNC,
    double* __restrict__ H_pre,
    double* __restrict__ U_pre,
    double* __restrict__ V_pre,
    double* __restrict__ Z_pre,
    double* __restrict__ W_pre,
    double* __restrict__ H_res,
    double* __restrict__ U_res,
    double* __restrict__ V_res,
    double* __restrict__ Z_res,
    double* __restrict__ W_res)
{
    cg::grid_group grid = cg::this_grid();
    int stride = CEL + 1;

    for (int step = 0; step < STEPS; step++) {
        // Phase 1: shallow_water_step (strided loop for persistent kernel)
        for (int pos = blockIdx.x * blockDim.x + threadIdx.x + 1;
             pos <= CEL;
             pos += gridDim.x * blockDim.x) {

            double H1 = H_pre[pos];
            double U1 = U_pre[pos];
            double V1 = V_pre[pos];
            double BI = ZBC[pos];

            double HI = fmax(H1, HM1);
            double UI = U1;
            double VI = V1;
            if (HI <= HM2) {
                UI = (UI >= 0.0) ? VMIN : -VMIN;
                VI = (VI >= 0.0) ? VMIN : -VMIN;
            }
            double ZI = fmax(Z_pre[pos], ZBC[pos]);

            double WH = 0.0, WU_acc = 0.0, WV_acc = 0.0;

            for (int j = 1; j <= 4; j++) {
                int idx = j * stride + pos;
                int NC  = NAC[idx];
                int KP  = KLAS[idx];
                double COSJ = COSF[idx];
                double SINJ = SINF[idx];

                Vec3 QL = {HI, UI * COSJ + VI * SINJ, VI * COSJ - UI * SINJ};
                double CL_v = sqrt(G * HI);
                double FIL_v = QL.y + 2.0 * CL_v;

                double HC = 0.0, BC = 0.0, ZC = 0.0, UC = 0.0, VC = 0.0;
                if (NC != 0) {
                    HC = fmax(H_pre[NC], HM1);
                    BC = ZBC[NC];
                    ZC = fmax(ZBC[NC], Z_pre[NC]);
                    UC = U_pre[NC];
                    VC = V_pre[NC];
                }

                double flux0 = 0.0, flux1 = 0.0, flux2 = 0.0, flux3 = 0.0;

                if (KP == 4) {
                    flux3 = HALF_G * H1 * H1;
                } else if (KP != 0) {
                    flux3 = HALF_G * H1 * H1;
                } else if (HI <= HM1 && HC <= HM1) {
                    // pass
                } else if (ZI <= BC) {
                    flux0 = -C1 * pow(HC, 1.5);
                    flux1 = HI * QL.y * fabs(QL.y);
                    flux3 = HALF_G * HI * HI;
                } else if (ZC <= BI) {
                    flux0 = C1 * pow(HI, 1.5);
                    flux1 = HI * fabs(QL.y) * QL.y;
                    flux2 = HI * fabs(QL.y) * QL.z;
                } else if (HI <= HM2) {
                    if (ZC > ZI) {
                        double DH = fmax(ZC - ZBC[pos], HM1);
                        double UN = -C1 * sqrt(DH);
                        flux0 = DH * UN;
                        flux1 = flux0 * UN;
                        flux2 = flux0 * (VC * COSJ - UC * SINJ);
                        flux3 = HALF_G * HI * HI;
                    } else {
                        flux0 = C1 * pow(HI, 1.5);
                        flux3 = HALF_G * HI * HI;
                    }
                } else if (HC <= HM2) {
                    if (ZI > ZC) {
                        double DH = fmax(ZI - BC, HM1);
                        double UN = C1 * sqrt(DH);
                        double HC1 = ZC - ZBC[pos];
                        flux0 = DH * UN;
                        flux1 = flux0 * UN;
                        flux2 = flux0 * QL.z;
                        flux3 = HALF_G * HC1 * HC1;
                    } else {
                        flux0 = -C1 * pow(HC, 1.5);
                        flux1 = HI * QL.y * QL.y;
                        flux3 = HALF_G * HI * HI;
                    }
                } else {
                    if (pos < NC) {
                        double QR_h = fmax(ZC - ZBC[pos], HM1);
                        double UR = UC * COSJ + VC * SINJ;
                        double ratio = fmin(HC / QR_h, 1.5);
                        double QR_u = UR * ratio;
                        if (HC <= HM2 || QR_h <= HM2) {
                            QR_u = (UR >= 0.0) ? VMIN : -VMIN;
                        }
                        double QR_v_ = VC * COSJ - UC * SINJ;
                        Vec3 QR_vec = {QR_h, QR_u, QR_v_};
                        Vec4 FLR_OS = osher(QL, QR_vec, FIL_v, H_pre[pos]);
                        flux0 = FLR_OS.x;
                        flux1 = FLR_OS.y + (1.0 - ratio) * HC * UR * UR / 2.0;
                        flux2 = FLR_OS.z;
                        flux3 = FLR_OS.w;
                    } else {
                        double COSJ1 = -COSJ;
                        double SINJ1 = -SINJ;
                        Vec3 QL1 = {
                            H_pre[NC],
                            U_pre[NC] * COSJ1 + V_pre[NC] * SINJ1,
                            V_pre[NC] * COSJ1 - U_pre[NC] * SINJ1
                        };
                        double CL1 = sqrt(G * H_pre[NC]);
                        double FIL1 = QL1.y + 2.0 * CL1;
                        double HC2 = fmax(HI, HM1);
                        double ZC1 = fmax(ZBC[pos], ZI);
                        double QR1_h = fmax(ZC1 - ZBC[NC], HM1);
                        double UR1 = UI * COSJ1 + VI * SINJ1;
                        double ratio1 = fmin(HC2 / QR1_h, 1.5);
                        double QR1_u = UR1 * ratio1;
                        if (HC2 <= HM2 || QR1_h <= HM2) {
                            QR1_u = (UR1 >= 0.0) ? VMIN : -VMIN;
                        }
                        double QR1_v_ = VI * COSJ1 - UI * SINJ1;
                        Vec3 QR1_vec = {QR1_h, QR1_u, QR1_v_};
                        Vec4 FLR1 = osher(QL1, QR1_vec, FIL1, H_pre[NC]);
                        flux0 = -FLR1.x;
                        flux1 = FLR1.y + (1.0 - ratio1) * HC2 * UR1 * UR1 / 2.0;
                        flux2 = FLR1.z;
                        double ZA = sqrt(FLR1.w / HALF_G) + BC;
                        double HC3 = fmax(ZA - ZBC[pos], 0.0);
                        flux3 = HALF_G * HC3 * HC3;
                    }
                }

                double SL   = SIDE[idx];
                double SLCA = SLCOS[idx];
                double SLSA = SLSIN[idx];
                double FLR_1 = flux1 + flux3;
                double FLR_2 = flux2;
                WH     += SL * flux0;
                WU_acc += SLCA * FLR_1 - SLSA * FLR_2;
                WV_acc += SLSA * FLR_1 + SLCA * FLR_2;
            }

            double DTA  = DT / AREA[pos];
            double WDTA = DTA;
            double H2   = fmax(H1 - WDTA * WH, HM1);
            double Z2   = H2 + BI;

            double U2 = 0.0, V2 = 0.0;
            if (H2 > HM1) {
                if (H2 <= HM2) {
                    U2 = (U1 >= 0.0) ? fmin(VMIN, fabs(U1)) : -fmin(VMIN, fabs(U1));
                    V2 = (V1 >= 0.0) ? fmin(VMIN, fabs(V1)) : -fmin(VMIN, fabs(V1));
                } else {
                    double QX1  = H1 * U1;
                    double QY1  = H1 * V1;
                    double DTAU = WDTA * WU_acc;
                    double DTAV = WDTA * WV_acc;
                    double WSF  = FNC[pos] * sqrt(U1 * U1 + V1 * V1) / pow(H1, 0.33333);
                    U2 = (QX1 - DTAU - DT * WSF * U1) / H2;
                    V2 = (QY1 - DTAV - DT * WSF * V1) / H2;
                    if (H2 > HM2) {
                        U2 = (U2 >= 0.0) ? fmin(fabs(U2), 15.0) : -fmin(fabs(U2), 15.0);
                        V2 = (V2 >= 0.0) ? fmin(fabs(V2), 15.0) : -fmin(fabs(V2), 15.0);
                    }
                }
            }

            H_res[pos] = H2;
            U_res[pos] = U2;
            V_res[pos] = V2;
            Z_res[pos] = Z2;
            W_res[pos] = sqrt(U2 * U2 + V2 * V2);
        }

        grid.sync();

        // Phase 2: transfer (res -> pre)
        for (int pos = blockIdx.x * blockDim.x + threadIdx.x + 1;
             pos <= CEL;
             pos += gridDim.x * blockDim.x) {
            H_pre[pos] = H_res[pos];
            U_pre[pos] = U_res[pos];
            V_pre[pos] = V_res[pos];
            Z_pre[pos] = Z_res[pos];
            W_pre[pos] = W_res[pos];
        }

        grid.sync();
    }
}

// ---------------------------------------------------------------------------
// Host: mesh setup (mirrors Taichi init exactly)
// ---------------------------------------------------------------------------
struct MeshData {
    int CEL;
    double DT;
    // Host arrays (row-major [5][CEL+1])
    int    *h_NAC, *h_KLAS;
    double *h_SIDE, *h_COSF, *h_SINF, *h_SLCOS, *h_SLSIN;
    double *h_AREA, *h_ZBC, *h_FNC;
    double *h_H, *h_U, *h_V, *h_Z, *h_W;
    // Device arrays
    int    *d_NAC, *d_KLAS;
    double *d_SIDE, *d_COSF, *d_SINF, *d_SLCOS, *d_SLSIN;
    double *d_AREA, *d_ZBC, *d_FNC;
    double *d_H_pre, *d_U_pre, *d_V_pre, *d_Z_pre, *d_W_pre;
    double *d_H_res, *d_U_res, *d_V_res, *d_Z_res, *d_W_res;
};

void init_mesh(MeshData &m, int N) {
    int CEL = N * N;
    m.CEL = CEL;
    double dx = 1.0;
    m.DT = 0.5 * dx / (sqrt(G * 2.0) + 1e-6);

    int sz5 = 5 * (CEL + 1);
    int sz1 = CEL + 1;

    // Allocate host
    m.h_NAC   = (int*)calloc(sz5, sizeof(int));
    m.h_KLAS  = (int*)calloc(sz5, sizeof(int));
    m.h_SIDE  = (double*)calloc(sz5, sizeof(double));
    m.h_COSF  = (double*)calloc(sz5, sizeof(double));
    m.h_SINF  = (double*)calloc(sz5, sizeof(double));
    m.h_SLCOS = (double*)calloc(sz5, sizeof(double));
    m.h_SLSIN = (double*)calloc(sz5, sizeof(double));
    m.h_AREA  = (double*)calloc(sz1, sizeof(double));
    m.h_ZBC   = (double*)calloc(sz1, sizeof(double));
    m.h_FNC   = (double*)calloc(sz1, sizeof(double));
    m.h_H     = (double*)calloc(sz1, sizeof(double));
    m.h_U     = (double*)calloc(sz1, sizeof(double));
    m.h_V     = (double*)calloc(sz1, sizeof(double));
    m.h_Z     = (double*)calloc(sz1, sizeof(double));
    m.h_W     = (double*)calloc(sz1, sizeof(double));

    // FNC = G * MANNING_N^2
    for (int i = 0; i < sz1; i++)
        m.h_FNC[i] = G * MANNING_N * MANNING_N;

    // H init to HM1
    for (int i = 0; i < sz1; i++)
        m.h_H[i] = HM1;

    double edge_cos[5] = {0.0, 0.0, 1.0, 0.0, -1.0};
    double edge_sin[5] = {0.0, -1.0, 0.0, 1.0, 0.0};

    for (int i = 0; i < N; i++) {
        for (int jj = 0; jj < N; jj++) {
            int pos = i * N + jj + 1;
            m.h_AREA[pos] = dx * dx;

            for (int e = 1; e <= 4; e++) {
                int idx = e * (CEL + 1) + pos;
                m.h_SIDE[idx] = dx;
                m.h_COSF[idx] = edge_cos[e];
                m.h_SINF[idx] = edge_sin[e];
            }

            // Edge 1: i-1 neighbor (south)
            if (i > 0) {
                m.h_NAC[1 * (CEL + 1) + pos] = (i - 1) * N + jj + 1;
            } else {
                m.h_KLAS[1 * (CEL + 1) + pos] = 4;
            }
            // Edge 2: jj+1 neighbor (east)
            if (jj < N - 1) {
                m.h_NAC[2 * (CEL + 1) + pos] = i * N + (jj + 1) + 1;
            } else {
                m.h_KLAS[2 * (CEL + 1) + pos] = 4;
            }
            // Edge 3: i+1 neighbor (north)
            if (i < N - 1) {
                m.h_NAC[3 * (CEL + 1) + pos] = (i + 1) * N + jj + 1;
            } else {
                m.h_KLAS[3 * (CEL + 1) + pos] = 4;
            }
            // Edge 4: jj-1 neighbor (west)
            if (jj > 0) {
                m.h_NAC[4 * (CEL + 1) + pos] = i * N + (jj - 1) + 1;
            } else {
                m.h_KLAS[4 * (CEL + 1) + pos] = 4;
            }

            // Dam-break IC
            m.h_H[pos] = (jj < N / 2) ? 2.0 : 0.5;
            m.h_Z[pos] = m.h_H[pos];
        }
    }

    // Compute SLCOS, SLSIN
    for (int i = 0; i < sz5; i++) {
        m.h_SLCOS[i] = m.h_SIDE[i] * m.h_COSF[i];
        m.h_SLSIN[i] = m.h_SIDE[i] * m.h_SINF[i];
    }

    // Allocate device
    CHECK_CUDA(cudaMalloc(&m.d_NAC,   sz5 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&m.d_KLAS,  sz5 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&m.d_SIDE,  sz5 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&m.d_COSF,  sz5 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&m.d_SINF,  sz5 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&m.d_SLCOS, sz5 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&m.d_SLSIN, sz5 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&m.d_AREA,  sz1 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&m.d_ZBC,   sz1 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&m.d_FNC,   sz1 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&m.d_H_pre, sz1 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&m.d_U_pre, sz1 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&m.d_V_pre, sz1 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&m.d_Z_pre, sz1 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&m.d_W_pre, sz1 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&m.d_H_res, sz1 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&m.d_U_res, sz1 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&m.d_V_res, sz1 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&m.d_Z_res, sz1 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&m.d_W_res, sz1 * sizeof(double)));

    // Copy to device
    CHECK_CUDA(cudaMemcpy(m.d_NAC,   m.h_NAC,   sz5 * sizeof(int),    cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m.d_KLAS,  m.h_KLAS,  sz5 * sizeof(int),    cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m.d_SIDE,  m.h_SIDE,  sz5 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m.d_COSF,  m.h_COSF,  sz5 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m.d_SINF,  m.h_SINF,  sz5 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m.d_SLCOS, m.h_SLCOS, sz5 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m.d_SLSIN, m.h_SLSIN, sz5 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m.d_AREA,  m.h_AREA,  sz1 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m.d_ZBC,   m.h_ZBC,   sz1 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m.d_FNC,   m.h_FNC,   sz1 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m.d_H_pre, m.h_H,     sz1 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m.d_U_pre, m.h_U,     sz1 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m.d_V_pre, m.h_V,     sz1 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m.d_Z_pre, m.h_Z,     sz1 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m.d_W_pre, m.h_W,     sz1 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(m.d_H_res, 0, sz1 * sizeof(double)));
    CHECK_CUDA(cudaMemset(m.d_U_res, 0, sz1 * sizeof(double)));
    CHECK_CUDA(cudaMemset(m.d_V_res, 0, sz1 * sizeof(double)));
    CHECK_CUDA(cudaMemset(m.d_Z_res, 0, sz1 * sizeof(double)));
    CHECK_CUDA(cudaMemset(m.d_W_res, 0, sz1 * sizeof(double)));
}

void reset_state(MeshData &m) {
    int sz1 = m.CEL + 1;
    CHECK_CUDA(cudaMemcpy(m.d_H_pre, m.h_H, sz1 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m.d_U_pre, m.h_U, sz1 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m.d_V_pre, m.h_V, sz1 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m.d_Z_pre, m.h_Z, sz1 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m.d_W_pre, m.h_W, sz1 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(m.d_H_res, 0, sz1 * sizeof(double)));
    CHECK_CUDA(cudaMemset(m.d_U_res, 0, sz1 * sizeof(double)));
    CHECK_CUDA(cudaMemset(m.d_V_res, 0, sz1 * sizeof(double)));
    CHECK_CUDA(cudaMemset(m.d_Z_res, 0, sz1 * sizeof(double)));
    CHECK_CUDA(cudaMemset(m.d_W_res, 0, sz1 * sizeof(double)));
}

void free_mesh(MeshData &m) {
    free(m.h_NAC);  free(m.h_KLAS);
    free(m.h_SIDE); free(m.h_COSF); free(m.h_SINF);
    free(m.h_SLCOS); free(m.h_SLSIN);
    free(m.h_AREA); free(m.h_ZBC); free(m.h_FNC);
    free(m.h_H); free(m.h_U); free(m.h_V); free(m.h_Z); free(m.h_W);
    cudaFree(m.d_NAC);  cudaFree(m.d_KLAS);
    cudaFree(m.d_SIDE); cudaFree(m.d_COSF); cudaFree(m.d_SINF);
    cudaFree(m.d_SLCOS); cudaFree(m.d_SLSIN);
    cudaFree(m.d_AREA); cudaFree(m.d_ZBC); cudaFree(m.d_FNC);
    cudaFree(m.d_H_pre); cudaFree(m.d_U_pre); cudaFree(m.d_V_pre);
    cudaFree(m.d_Z_pre); cudaFree(m.d_W_pre);
    cudaFree(m.d_H_res); cudaFree(m.d_U_res); cudaFree(m.d_V_res);
    cudaFree(m.d_Z_res); cudaFree(m.d_W_res);
}

// ---------------------------------------------------------------------------
// Benchmark helpers
// ---------------------------------------------------------------------------
double bench_sync(MeshData &m, int STEPS, int blocks, int threads) {
    reset_state(m);
    // Warmup
    for (int i = 0; i < 5; i++) {
        shallow_water_step<<<blocks, threads>>>(
            m.CEL, m.DT, m.d_NAC, m.d_KLAS, m.d_SIDE, m.d_COSF, m.d_SINF,
            m.d_SLCOS, m.d_SLSIN, m.d_AREA, m.d_ZBC, m.d_FNC,
            m.d_H_pre, m.d_U_pre, m.d_V_pre, m.d_Z_pre, m.d_W_pre,
            m.d_H_res, m.d_U_res, m.d_V_res, m.d_Z_res, m.d_W_res);
        transfer<<<blocks, threads>>>(
            m.CEL, m.d_H_res, m.d_U_res, m.d_V_res, m.d_Z_res, m.d_W_res,
            m.d_H_pre, m.d_U_pre, m.d_V_pre, m.d_Z_pre, m.d_W_pre);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    reset_state(m);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    for (int s = 0; s < STEPS; s++) {
        shallow_water_step<<<blocks, threads>>>(
            m.CEL, m.DT, m.d_NAC, m.d_KLAS, m.d_SIDE, m.d_COSF, m.d_SINF,
            m.d_SLCOS, m.d_SLSIN, m.d_AREA, m.d_ZBC, m.d_FNC,
            m.d_H_pre, m.d_U_pre, m.d_V_pre, m.d_Z_pre, m.d_W_pre,
            m.d_H_res, m.d_U_res, m.d_V_res, m.d_Z_res, m.d_W_res);
        transfer<<<blocks, threads>>>(
            m.CEL, m.d_H_res, m.d_U_res, m.d_V_res, m.d_Z_res, m.d_W_res,
            m.d_H_pre, m.d_U_pre, m.d_V_pre, m.d_Z_pre, m.d_W_pre);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return (double)ms * 1000.0 / STEPS; // us/step
}

double bench_async(MeshData &m, int STEPS, int blocks, int threads) {
    reset_state(m);
    // Warmup
    for (int i = 0; i < 5; i++) {
        shallow_water_step<<<blocks, threads>>>(
            m.CEL, m.DT, m.d_NAC, m.d_KLAS, m.d_SIDE, m.d_COSF, m.d_SINF,
            m.d_SLCOS, m.d_SLSIN, m.d_AREA, m.d_ZBC, m.d_FNC,
            m.d_H_pre, m.d_U_pre, m.d_V_pre, m.d_Z_pre, m.d_W_pre,
            m.d_H_res, m.d_U_res, m.d_V_res, m.d_Z_res, m.d_W_res);
        transfer<<<blocks, threads>>>(
            m.CEL, m.d_H_res, m.d_U_res, m.d_V_res, m.d_Z_res, m.d_W_res,
            m.d_H_pre, m.d_U_pre, m.d_V_pre, m.d_Z_pre, m.d_W_pre);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    reset_state(m);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    for (int s = 0; s < STEPS; s++) {
        shallow_water_step<<<blocks, threads>>>(
            m.CEL, m.DT, m.d_NAC, m.d_KLAS, m.d_SIDE, m.d_COSF, m.d_SINF,
            m.d_SLCOS, m.d_SLSIN, m.d_AREA, m.d_ZBC, m.d_FNC,
            m.d_H_pre, m.d_U_pre, m.d_V_pre, m.d_Z_pre, m.d_W_pre,
            m.d_H_res, m.d_U_res, m.d_V_res, m.d_Z_res, m.d_W_res);
        transfer<<<blocks, threads>>>(
            m.CEL, m.d_H_res, m.d_U_res, m.d_V_res, m.d_Z_res, m.d_W_res,
            m.d_H_pre, m.d_U_pre, m.d_V_pre, m.d_Z_pre, m.d_W_pre);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return (double)ms * 1000.0 / STEPS;
}

double bench_graph(MeshData &m, int STEPS, int blocks, int threads) {
    reset_state(m);
    // Warmup
    for (int i = 0; i < 5; i++) {
        shallow_water_step<<<blocks, threads>>>(
            m.CEL, m.DT, m.d_NAC, m.d_KLAS, m.d_SIDE, m.d_COSF, m.d_SINF,
            m.d_SLCOS, m.d_SLSIN, m.d_AREA, m.d_ZBC, m.d_FNC,
            m.d_H_pre, m.d_U_pre, m.d_V_pre, m.d_Z_pre, m.d_W_pre,
            m.d_H_res, m.d_U_res, m.d_V_res, m.d_Z_res, m.d_W_res);
        transfer<<<blocks, threads>>>(
            m.CEL, m.d_H_res, m.d_U_res, m.d_V_res, m.d_Z_res, m.d_W_res,
            m.d_H_pre, m.d_U_pre, m.d_V_pre, m.d_Z_pre, m.d_W_pre);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    reset_state(m);

    // Capture graph for one step (2 kernels)
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    shallow_water_step<<<blocks, threads, 0, stream>>>(
        m.CEL, m.DT, m.d_NAC, m.d_KLAS, m.d_SIDE, m.d_COSF, m.d_SINF,
        m.d_SLCOS, m.d_SLSIN, m.d_AREA, m.d_ZBC, m.d_FNC,
        m.d_H_pre, m.d_U_pre, m.d_V_pre, m.d_Z_pre, m.d_W_pre,
        m.d_H_res, m.d_U_res, m.d_V_res, m.d_Z_res, m.d_W_res);
    transfer<<<blocks, threads, 0, stream>>>(
        m.CEL, m.d_H_res, m.d_U_res, m.d_V_res, m.d_Z_res, m.d_W_res,
        m.d_H_pre, m.d_U_pre, m.d_V_pre, m.d_Z_pre, m.d_W_pre);

    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, stream));

    for (int s = 0; s < STEPS; s++) {
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    }

    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return (double)ms * 1000.0 / STEPS;
}

double bench_persistent(MeshData &m, int STEPS, int threads) {
    reset_state(m);

    // Query max cooperative blocks
    int numBlocksPerSm = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm, persistent_kernel, threads, 0));
    int devId = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, devId));
    int maxBlocks = numBlocksPerSm * prop.multiProcessorCount;
    // Cap to what we need
    int needed = (m.CEL + threads - 1) / threads;
    int blocks = (maxBlocks < needed) ? maxBlocks : needed;
    if (blocks < 1) blocks = 1;

    // Warmup with a small number of steps
    int warmup_steps = 5;
    void *args_warmup[] = {
        &m.CEL, &m.DT, &warmup_steps,
        &m.d_NAC, &m.d_KLAS, &m.d_SIDE, &m.d_COSF, &m.d_SINF,
        &m.d_SLCOS, &m.d_SLSIN, &m.d_AREA, &m.d_ZBC, &m.d_FNC,
        &m.d_H_pre, &m.d_U_pre, &m.d_V_pre, &m.d_Z_pre, &m.d_W_pre,
        &m.d_H_res, &m.d_U_res, &m.d_V_res, &m.d_Z_res, &m.d_W_res
    };
    CHECK_CUDA(cudaLaunchCooperativeKernel(
        (void*)persistent_kernel, dim3(blocks), dim3(threads), args_warmup, 0, 0));
    CHECK_CUDA(cudaDeviceSynchronize());
    reset_state(m);

    void *args[] = {
        &m.CEL, &m.DT, &STEPS,
        &m.d_NAC, &m.d_KLAS, &m.d_SIDE, &m.d_COSF, &m.d_SINF,
        &m.d_SLCOS, &m.d_SLSIN, &m.d_AREA, &m.d_ZBC, &m.d_FNC,
        &m.d_H_pre, &m.d_U_pre, &m.d_V_pre, &m.d_Z_pre, &m.d_W_pre,
        &m.d_H_res, &m.d_U_res, &m.d_V_res, &m.d_Z_res, &m.d_W_res
    };

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    CHECK_CUDA(cudaLaunchCooperativeKernel(
        (void*)persistent_kernel, dim3(blocks), dim3(threads), args, 0, 0));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return (double)ms * 1000.0 / STEPS;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    int Ns[]    = {32, 64, 128};
    int STEPS   = 500;

    auto pick_threads = [](int CEL) {
        if (CEL <= 1024) return 128;
        return 256;
    };

    printf("===================================================================\n");
    printf("  Shallow Water (Osher) CUDA Benchmark — fp64, STEPS=%d\n", STEPS);
    printf("===================================================================\n");
    printf("%-6s | %-12s | %-12s | %-12s | %-12s\n",
           "N", "Sync (us)", "Async (us)", "Graph (us)", "Persist (us)");
    printf("-------+");
    for (int i = 0; i < 4; i++) printf("--------------+");
    printf("\n");

    double sync_us[3], async_us[3], graph_us[3], persist_us[3];

    for (int ni = 0; ni < 3; ni++) {
        int N = Ns[ni];
        int CEL = N * N;
        int THREADS = pick_threads(CEL);
        int blocks = (CEL + THREADS - 1) / THREADS;

        MeshData m;
        init_mesh(m, N);

        sync_us[ni]    = bench_sync(m, STEPS, blocks, THREADS);
        async_us[ni]   = bench_async(m, STEPS, blocks, THREADS);
        graph_us[ni]   = bench_graph(m, STEPS, blocks, THREADS);
        persist_us[ni] = bench_persistent(m, STEPS, THREADS);

        printf("%-6d | %10.2f   | %10.2f   | %10.2f   | %10.2f\n",
               N, sync_us[ni], async_us[ni], graph_us[ni], persist_us[ni]);

        free_mesh(m);
    }

    printf("\n");
    printf("%-6s | %-12s | %-12s | %-12s | %-12s\n",
           "N", "Sync (1.0x)", "Async spdup", "Graph spdup", "Persist spdup");
    printf("-------+");
    for (int i = 0; i < 4; i++) printf("--------------+");
    printf("\n");

    for (int ni = 0; ni < 3; ni++) {
        printf("%-6d | %10.2fx  | %10.2fx  | %10.2fx  | %10.2fx\n",
               Ns[ni],
               1.0,
               sync_us[ni] / async_us[ni],
               sync_us[ni] / graph_us[ni],
               sync_us[ni] / persist_us[ni]);
    }

    printf("===================================================================\n");
    return 0;
}
