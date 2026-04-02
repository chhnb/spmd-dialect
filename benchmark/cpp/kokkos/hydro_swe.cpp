/**
 * 2D Shallow Water Equations (Osher Riemann solver) — Kokkos implementation.
 *
 * Dam-break on NxN structured quad mesh (unstructured representation).
 * Port of hydro-cal calculate_gpu.cu.
 *
 * Build:
 *   cmake -B build-cuda \
 *     -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_SERIAL=ON \
 *     -DKokkos_ARCH_AMPERE80=ON \
 *     -DCMAKE_CXX_COMPILER=$(pwd)/../../../kokkos/bin/nvcc_wrapper
 *   cmake --build build-cuda -j8
 *
 * Run:
 *   ./build-cuda/hydro_swe_kokkos 128 10 20
 *                                  ^N  ^steps ^repeat
 */

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr double G      = 9.81;
constexpr double HALF_G = 4.905;
constexpr double HM1    = 0.001;
constexpr double HM2    = 0.01;
constexpr double VMIN   = 0.001;
constexpr double C1_C   = 0.3;
constexpr double MANNING_N = 0.03;

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------
using View1Di = Kokkos::View<int*>;
using View1D  = Kokkos::View<double*>;
using View2Di = Kokkos::View<int**>;
using View2D  = Kokkos::View<double**>;

// ---------------------------------------------------------------------------
// QF: shallow-water flux function
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void QF(double h, double u, double v, double &F0, double &F1, double &F2, double &F3) {
    F0 = h * u;
    F1 = F0 * u;
    F2 = F0 * v;
    F3 = HALF_G * h * h;
}

// ---------------------------------------------------------------------------
// Osher approximate Riemann solver
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void osher(double QL_h, double QL_u, double QL_v,
           double QR_h, double QR_u, double QR_v,
           double FIL_in, double H_pos,
           double &FLR0, double &FLR1, double &FLR2, double &FLR3)
{
    double CR = Kokkos::sqrt(G * QR_h);
    double FIR_v = QR_u - 2.0 * CR;
    double fil = FIL_in;
    double fir = FIR_v;
    double UA = (fil + fir) / 2.0;
    double CA = Kokkos::fabs((fil - fir) / 4.0);
    double CL_v = Kokkos::sqrt(G * H_pos);

    FLR0 = 0; FLR1 = 0; FLR2 = 0; FLR3 = 0;

    int K2 = (CA < UA) ? 1 : (UA >= 0.0 && UA < CA) ? 2 : (UA >= -CA && UA < 0.0) ? 3 : 4;
    int K1 = (QL_u < CL_v && QR_u >= -CR) ? 1 :
             (QL_u >= CL_v && QR_u >= -CR) ? 2 :
             (QL_u < CL_v && QR_u < -CR)   ? 3 : 4;

    // Helper lambdas for QS templates
    auto add_QF = [&](double h, double u, double v, double sign) {
        double f0, f1, f2, f3;
        QF(h, u, v, f0, f1, f2, f3);
        FLR0 += f0 * sign; FLR1 += f1 * sign; FLR2 += f2 * sign; FLR3 += f3 * sign;
    };

    auto qs1 = [&](double s) { add_QF(QL_h, QL_u, QL_v, s); };
    auto qs2 = [&](double s) {
        double US = fil / 3.0, HS = US * US / G;
        add_QF(HS, US, QL_v, s);
    };
    auto qs3 = [&](double s) {
        double ua = (fil + fir) / 2.0;
        fil = fil - ua;
        double HA = fil * fil / (4.0 * G);
        add_QF(HA, ua, QL_v, s);
    };
    auto qs5 = [&](double s) {
        double ua = (fil + fir) / 2.0;
        fir = fir - ua;
        double HA = fir * fir / (4.0 * G);
        add_QF(HA, ua, QR_v, s);
    };
    auto qs6 = [&](double s) {
        double US = fir / 3.0, HS = US * US / G;
        add_QF(HS, US, QR_v, s);
    };
    auto qs7 = [&](double s) { add_QF(QR_h, QR_u, QR_v, s); };

    // Dispatch table (matching CUDA switch-case)
    switch (K1) {
    case 1:
        switch (K2) {
        case 1: qs2(1); break;
        case 2: qs3(1); break;
        case 3: qs5(1); break;
        case 4: qs6(1); break;
        } break;
    case 2:
        switch (K2) {
        case 1: qs1(1); break;
        case 2: qs1(1); qs2(-1); qs3(1); break;
        case 3: qs1(1); qs2(-1); qs5(1); break;
        case 4: qs1(1); qs2(-1); qs6(1); break;
        } break;
    case 3:
        switch (K2) {
        case 1: qs2(1); qs6(-1); qs7(1); break;
        case 2: qs3(1); qs6(-1); qs7(1); break;
        case 3: qs5(1); qs6(-1); qs7(1); break;
        case 4: qs7(1); break;
        } break;
    case 4:
        switch (K2) {
        case 1: qs1(1); qs6(-1); qs7(1); break;
        case 2: qs1(1); qs2(-1); qs3(1); qs6(-1); qs7(1); break;
        case 3: qs1(1); qs2(-1); qs5(1); qs6(-1); qs7(1); break;
        case 4: qs1(1); qs2(-1); qs7(1); break;
        } break;
    }
}

// ---------------------------------------------------------------------------
// Main kernel functor
// ---------------------------------------------------------------------------
struct SWEStep {
    int CEL;
    double DT;
    View2Di NAC, KLAS;
    View2D  SIDE, COSF, SINF, SLCOS, SLSIN;
    View1D  AREA, ZBC, FNC;
    View1D  H_pre, U_pre, V_pre, Z_pre;
    View1D  H_res, U_res, V_res, Z_res, W_res;

    KOKKOS_INLINE_FUNCTION
    void operator()(const int idx) const {
        int pos = idx + 1; // 1-indexed
        if (pos > CEL) return;

        double H1 = H_pre(pos), U1 = U_pre(pos), V1 = V_pre(pos);
        double BI = ZBC(pos);
        double HI = Kokkos::fmax(H1, HM1);
        double UI = U1, VI = V1;
        if (HI <= HM2) {
            UI = Kokkos::copysign(VMIN, UI);
            VI = Kokkos::copysign(VMIN, VI);
        }
        double ZI = Kokkos::fmax(Z_pre(pos), ZBC(pos));
        double WH = 0.0, WU = 0.0, WV = 0.0;

        for (int j = 1; j <= 4; ++j) {
            int NC = NAC(j, pos);
            int KP = KLAS(j, pos);
            double COSJ = COSF(j, pos), SINJ = SINF(j, pos);
            double SL = SIDE(j, pos), SLCA = SLCOS(j, pos), SLSA = SLSIN(j, pos);

            double QL_h = HI, QL_u = UI*COSJ + VI*SINJ, QL_v = VI*COSJ - UI*SINJ;
            double CL_v = Kokkos::sqrt(G * HI);
            double FIL_v = QL_u + 2.0 * CL_v;

            double HC = 0, BC = 0, ZC = 0, UC = 0, VC = 0;
            if (NC != 0) {
                HC = Kokkos::fmax(H_pre(NC), HM1);
                BC = ZBC(NC);
                ZC = Kokkos::fmax(ZBC(NC), Z_pre(NC));
                UC = U_pre(NC); VC = V_pre(NC);
            }

            double f0 = 0, f1 = 0, f2 = 0, f3 = 0;

            if (KP == 4 || KP != 0) {
                f3 = HALF_G * H1 * H1;
            } else if (HI <= HM1 && HC <= HM1) {
                // both dry
            } else if (ZI <= BC) {
                f0 = -C1_C * Kokkos::pow(HC, 1.5);
                f1 = HI * QL_u * Kokkos::fabs(QL_u);
                f3 = HALF_G * HI * HI;
            } else if (ZC <= BI) {
                f0 = C1_C * Kokkos::pow(HI, 1.5);
                f1 = HI * Kokkos::fabs(QL_u) * QL_u;
                f2 = HI * Kokkos::fabs(QL_u) * QL_v;
            } else if (HI <= HM2) {
                if (ZC > ZI) {
                    double DH = Kokkos::fmax(ZC - ZBC(pos), HM1);
                    double UN = -C1_C * Kokkos::sqrt(DH);
                    f0 = DH * UN; f1 = f0 * UN;
                    f2 = f0 * (VC*COSJ - UC*SINJ);
                    f3 = HALF_G * HI * HI;
                } else {
                    f0 = C1_C * Kokkos::pow(HI, 1.5);
                    f3 = HALF_G * HI * HI;
                }
            } else if (HC <= HM2) {
                if (ZI > ZC) {
                    double DH = Kokkos::fmax(ZI - BC, HM1);
                    double UN = C1_C * Kokkos::sqrt(DH);
                    double HC1 = ZC - ZBC(pos);
                    f0 = DH * UN; f1 = f0 * UN; f2 = f0 * QL_v;
                    f3 = HALF_G * HC1 * HC1;
                } else {
                    f0 = -C1_C * Kokkos::pow(HC, 1.5);
                    f1 = HI * QL_u * QL_u;
                    f3 = HALF_G * HI * HI;
                }
            } else {
                // Both wet — Osher Riemann solver
                if (pos < NC) {
                    double QR_h = Kokkos::fmax(ZC - ZBC(pos), HM1);
                    double UR = UC*COSJ + VC*SINJ;
                    double ratio = Kokkos::fmin(HC / QR_h, 1.5);
                    double QR_u = UR * ratio;
                    if (HC <= HM2 || QR_h <= HM2)
                        QR_u = Kokkos::copysign(VMIN, UR);
                    double QR_v = VC*COSJ - UC*SINJ;
                    double os0, os1, os2, os3;
                    osher(QL_h, QL_u, QL_v, QR_h, QR_u, QR_v, FIL_v, H_pre(pos),
                          os0, os1, os2, os3);
                    f0 = os0;
                    f1 = os1 + (1.0 - ratio) * HC * UR * UR / 2.0;
                    f2 = os2; f3 = os3;
                } else {
                    double COSJ1 = -COSJ, SINJ1 = -SINJ;
                    double QL1_h = H_pre(NC);
                    double QL1_u = U_pre(NC)*COSJ1 + V_pre(NC)*SINJ1;
                    double QL1_v = V_pre(NC)*COSJ1 - U_pre(NC)*SINJ1;
                    double CL1 = Kokkos::sqrt(G * H_pre(NC));
                    double FIL1 = QL1_u + 2.0 * CL1;
                    double HC2 = Kokkos::fmax(HI, HM1);
                    double ZC1 = Kokkos::fmax(ZBC(pos), ZI);
                    double QR1_h = Kokkos::fmax(ZC1 - ZBC(NC), HM1);
                    double UR1 = UI*COSJ1 + VI*SINJ1;
                    double ratio1 = Kokkos::fmin(HC2 / QR1_h, 1.5);
                    double QR1_u = UR1 * ratio1;
                    if (HC2 <= HM2 || QR1_h <= HM2)
                        QR1_u = Kokkos::copysign(VMIN, UR1);
                    double QR1_v = VI*COSJ1 - UI*SINJ1;
                    double mr0, mr1, mr2, mr3;
                    osher(QL1_h, QL1_u, QL1_v, QR1_h, QR1_u, QR1_v, FIL1, H_pre(NC),
                          mr0, mr1, mr2, mr3);
                    f0 = -mr0;
                    f1 = mr1 + (1.0 - ratio1) * HC2 * UR1 * UR1 / 2.0;
                    f2 = mr2;
                    double ZA = Kokkos::sqrt(mr3 / HALF_G) + BC;
                    double HC3 = Kokkos::fmax(ZA - ZBC(pos), 0.0);
                    f3 = HALF_G * HC3 * HC3;
                }
            }

            double FLR_1 = f1 + f3, FLR_2 = f2;
            WH += SL * f0;
            WU += SLCA * FLR_1 - SLSA * FLR_2;
            WV += SLSA * FLR_1 + SLCA * FLR_2;
        }

        // State update + Manning friction
        double DTA = DT / AREA(pos);
        double WDTA = DTA;
        double H2 = Kokkos::fmax(H1 - WDTA * WH, HM1);
        double Z2 = H2 + BI;
        double U2 = 0.0, V2 = 0.0;

        if (H2 > HM1) {
            if (H2 <= HM2) {
                U2 = Kokkos::copysign(Kokkos::fmin(VMIN, Kokkos::fabs(U1)), U1);
                V2 = Kokkos::copysign(Kokkos::fmin(VMIN, Kokkos::fabs(V1)), V1);
            } else {
                double QX1 = H1*U1, QY1 = H1*V1;
                double WSF = G * MANNING_N * MANNING_N *
                             Kokkos::sqrt(U1*U1 + V1*V1) / Kokkos::pow(H1, 0.33333);
                U2 = (QX1 - WDTA*WU - DT*WSF*U1) / H2;
                V2 = (QY1 - WDTA*WV - DT*WSF*V1) / H2;
                U2 = Kokkos::copysign(Kokkos::fmin(Kokkos::fabs(U2), 15.0), U2);
                V2 = Kokkos::copysign(Kokkos::fmin(Kokkos::fabs(V2), 15.0), V2);
            }
        }

        H_res(pos) = H2; U_res(pos) = U2; V_res(pos) = V2;
        Z_res(pos) = Z2; W_res(pos) = Kokkos::sqrt(U2*U2 + V2*V2);
    }
};

// ---------------------------------------------------------------------------
// Transfer kernel
// ---------------------------------------------------------------------------
struct Transfer {
    int CEL;
    View1D H_pre, U_pre, V_pre, Z_pre, W_pre;
    View1D H_res, U_res, V_res, Z_res, W_res;

    KOKKOS_INLINE_FUNCTION
    void operator()(const int idx) const {
        int pos = idx + 1;
        if (pos > CEL) return;
        H_pre(pos) = H_res(pos); U_pre(pos) = U_res(pos);
        V_pre(pos) = V_res(pos); Z_pre(pos) = Z_res(pos);
        W_pre(pos) = W_res(pos);
    }
};

// ---------------------------------------------------------------------------
// Main: build dam-break mesh, run benchmark
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        int N      = (argc > 1) ? atoi(argv[1]) : 128;
        int steps  = (argc > 2) ? atoi(argv[2]) : 10;
        int repeat = (argc > 3) ? atoi(argv[3]) : 20;
        int warmup = 5;
        int CEL = N * N;
        double dx = 1.0;
        double DT = 0.5 * dx / (std::sqrt(G * 2.0) + 1e-6);

        printf("Kokkos SWE Osher: N=%d (%d cells), steps=%d, warmup=%d, repeat=%d\n",
               N, CEL, steps, warmup, repeat);

        int stride = CEL + 1;

        // Allocate views
        View2Di NAC("NAC", 5, stride);
        View2Di KLAS("KLAS", 5, stride);
        View2D  SIDE("SIDE", 5, stride);
        View2D  COSF("COSF", 5, stride);
        View2D  SINF("SINF", 5, stride);
        View2D  SLCOS("SLCOS", 5, stride);
        View2D  SLSIN("SLSIN", 5, stride);
        View1D  AREA("AREA", stride);
        View1D  ZBC("ZBC", stride);
        View1D  FNC("FNC", stride);
        View1D  H_pre("H_pre", stride), U_pre("U_pre", stride);
        View1D  V_pre("V_pre", stride), Z_pre("Z_pre", stride);
        View1D  W_pre("W_pre", stride);
        View1D  H_res("H_res", stride), U_res("U_res", stride);
        View1D  V_res("V_res", stride), Z_res("Z_res", stride);
        View1D  W_res("W_res", stride);

        // Host mirrors for init
        auto h_NAC  = Kokkos::create_mirror_view(NAC);
        auto h_KLAS = Kokkos::create_mirror_view(KLAS);
        auto h_SIDE = Kokkos::create_mirror_view(SIDE);
        auto h_COSF = Kokkos::create_mirror_view(COSF);
        auto h_SINF = Kokkos::create_mirror_view(SINF);
        auto h_AREA = Kokkos::create_mirror_view(AREA);
        auto h_ZBC  = Kokkos::create_mirror_view(ZBC);
        auto h_FNC  = Kokkos::create_mirror_view(FNC);
        auto h_H    = Kokkos::create_mirror_view(H_pre);
        auto h_Z    = Kokkos::create_mirror_view(Z_pre);

        double edge_cos[] = {0.0, 0.0, 1.0, 0.0, -1.0};
        double edge_sin[] = {0.0, -1.0, 0.0, 1.0, 0.0};

        for (int i = 0; i < N; ++i) {
            for (int jj = 0; jj < N; ++jj) {
                int pos = i * N + jj + 1;
                h_AREA(pos) = dx * dx;
                h_FNC(pos) = G * MANNING_N * MANNING_N;
                for (int e = 1; e <= 4; ++e) {
                    h_SIDE(e, pos) = dx;
                    h_COSF(e, pos) = edge_cos[e];
                    h_SINF(e, pos) = edge_sin[e];
                }
                // South
                if (i > 0) h_NAC(1, pos) = (i-1)*N + jj + 1;
                else       h_KLAS(1, pos) = 4;
                // East
                if (jj < N-1) h_NAC(2, pos) = i*N + (jj+1) + 1;
                else          h_KLAS(2, pos) = 4;
                // North
                if (i < N-1) h_NAC(3, pos) = (i+1)*N + jj + 1;
                else         h_KLAS(3, pos) = 4;
                // West
                if (jj > 0) h_NAC(4, pos) = i*N + (jj-1) + 1;
                else        h_KLAS(4, pos) = 4;

                h_H(pos) = (jj < N/2) ? 2.0 : 0.5;
                h_Z(pos) = h_H(pos);
            }
        }

        // Compute SLCOS, SLSIN
        auto h_SLCOS = Kokkos::create_mirror_view(SLCOS);
        auto h_SLSIN = Kokkos::create_mirror_view(SLSIN);
        for (int e = 1; e <= 4; ++e)
            for (int p = 1; p <= CEL; ++p) {
                h_SLCOS(e, p) = h_SIDE(e, p) * h_COSF(e, p);
                h_SLSIN(e, p) = h_SIDE(e, p) * h_SINF(e, p);
            }

        // Copy to device
        Kokkos::deep_copy(NAC, h_NAC);
        Kokkos::deep_copy(KLAS, h_KLAS);
        Kokkos::deep_copy(SIDE, h_SIDE);
        Kokkos::deep_copy(COSF, h_COSF);
        Kokkos::deep_copy(SINF, h_SINF);
        Kokkos::deep_copy(SLCOS, h_SLCOS);
        Kokkos::deep_copy(SLSIN, h_SLSIN);
        Kokkos::deep_copy(AREA, h_AREA);
        Kokkos::deep_copy(ZBC, h_ZBC);
        Kokkos::deep_copy(FNC, h_FNC);
        Kokkos::deep_copy(H_pre, h_H);
        Kokkos::deep_copy(Z_pre, h_Z);
        Kokkos::fence();

        SWEStep swe{CEL, DT, NAC, KLAS, SIDE, COSF, SINF, SLCOS, SLSIN,
                     AREA, ZBC, FNC, H_pre, U_pre, V_pre, Z_pre,
                     H_res, U_res, V_res, Z_res, W_res};
        Transfer xfer{CEL, H_pre, U_pre, V_pre, Z_pre, W_pre,
                       H_res, U_res, V_res, Z_res, W_res};

        // Warmup
        for (int w = 0; w < warmup; ++w) {
            for (int s = 0; s < steps; ++s) {
                Kokkos::parallel_for("swe_step", CEL, swe);
                Kokkos::parallel_for("transfer", CEL, xfer);
            }
            Kokkos::fence();
        }

        // Timed runs
        std::vector<double> times;
        for (int r = 0; r < repeat; ++r) {
            Kokkos::fence();
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int s = 0; s < steps; ++s) {
                Kokkos::parallel_for("swe_step", CEL, swe);
                Kokkos::parallel_for("transfer", CEL, xfer);
            }
            Kokkos::fence();
            auto t1 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        std::sort(times.begin(), times.end());
        double sum = std::accumulate(times.begin(), times.end(), 0.0);
        printf("  min=%.3fms  median=%.3fms  avg=%.3fms  max=%.3fms\n",
               times.front(), times[times.size()/2], sum/times.size(), times.back());
        printf("CSV: kokkos_hydro_swe,%d,%d,%.3f,%.3f,%.3f,%.3f\n",
               N, steps, times.front(), times[times.size()/2],
               sum/times.size(), times.back());
    }
    Kokkos::finalize();
    return 0;
}
