/**
 * F2: Refactored Hydro-Cal — Kokkos implementation.
 * Edge-parallel flux + Cell-parallel update, fp32.
 *
 * Simplified for benchmark: only interior (KLAS=0), wall (KLAS=4),
 * and water level boundary (KLAS=1). Other KLAS treated as wall.
 *
 * Build: cmake --build build-cuda --target hydro_refactored_kokkos
 * Run:   ./hydro_refactored_kokkos <steps> <repeat>
 */

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <string>

// Constants (matching common.hpp)
constexpr float G = 9.81f;
constexpr float HALF_G = 4.905f;
constexpr float C0 = 1.33f;
constexpr float C1_C = 1.7f;
constexpr float VMIN = 0.001f;
constexpr float QLUA = 0.0f;
constexpr float BRDTH = 100.0f;

using View1Df = Kokkos::View<float*>;
using View1Di = Kokkos::View<int*>;

// ===== QF: flux function =====
KOKKOS_INLINE_FUNCTION
void QF(float h, float u, float v, float& F0, float& F1, float& F2, float& F3) {
    F0 = h * u;
    F1 = F0 * u;
    F2 = F0 * v;
    F3 = HALF_G * h * h;
}

// ===== OSHER Riemann solver =====
KOKKOS_INLINE_FUNCTION
void osher(float QL_h, float QL_u, float QL_v,
           float QR_h, float QR_u, float QR_v,
           float FIL_in, float H_pos,
           float& R0, float& R1, float& R2, float& R3) {
    float CR = Kokkos::sqrt(G * QR_h);
    float FIR_v = QR_u - 2.0f * CR;
    float fil = FIL_in, fir = FIR_v;
    float UA = (fil + fir) / 2.0f;
    float CA = Kokkos::fabs((fil - fir) / 4.0f);
    float CL_v = Kokkos::sqrt(G * H_pos);
    R0 = R1 = R2 = R3 = 0.0f;

    int K2 = (CA < UA) ? 1 : (UA >= 0 && UA < CA) ? 2 : (UA >= -CA && UA < 0) ? 3 : 4;
    int K1 = (QL_u < CL_v && QR_u >= -CR) ? 1 :
             (QL_u >= CL_v && QR_u >= -CR) ? 2 :
             (QL_u < CL_v && QR_u < -CR) ? 3 : 4;

    auto add = [&](float h, float u, float v, float s) {
        float f0, f1, f2, f3; QF(h, u, v, f0, f1, f2, f3);
        R0 += f0*s; R1 += f1*s; R2 += f2*s; R3 += f3*s;
    };
    auto qs1 = [&](float s) { add(QL_h, QL_u, QL_v, s); };
    auto qs2 = [&](float s) { float U=fil/3, H=U*U/G; add(H, U, QL_v, s); };
    auto qs3 = [&](float s) { float ua=(fil+fir)/2; fil-=ua; float H=fil*fil/(4*G); add(H, ua, QL_v, s); };
    auto qs5 = [&](float s) { float ua=(fil+fir)/2; fir-=ua; float H=fir*fir/(4*G); add(H, ua, QR_v, s); };
    auto qs6 = [&](float s) { float U=fir/3, H=U*U/G; add(H, U, QR_v, s); };
    auto qs7 = [&](float s) { add(QR_h, QR_u, QR_v, s); };

    switch(K1) {
    case 1: switch(K2) { case 1:qs2(1);break;case 2:qs3(1);break;case 3:qs5(1);break;case 4:qs6(1);break; } break;
    case 2: switch(K2) { case 1:qs1(1);break;case 2:qs1(1);qs2(-1);qs3(1);break;case 3:qs1(1);qs2(-1);qs5(1);break;case 4:qs1(1);qs2(-1);qs6(1);break; } break;
    case 3: switch(K2) { case 1:qs2(1);qs6(-1);qs7(1);break;case 2:qs3(1);qs6(-1);qs7(1);break;case 3:qs5(1);qs6(-1);qs7(1);break;case 4:qs7(1);break; } break;
    case 4: switch(K2) { case 1:qs1(1);qs6(-1);qs7(1);break;case 2:qs1(1);qs2(-1);qs3(1);qs6(-1);qs7(1);break;case 3:qs1(1);qs2(-1);qs5(1);qs6(-1);qs7(1);break;case 4:qs1(1);qs2(-1);qs7(1);break; } break;
    }
}

// ===== CalculateFlux functor (edge-parallel) =====
struct CalcFlux {
    int CELL;
    float HM1, HM2;
    View1Df H, U, V, Z, ZBC, ZB1;
    View1Di NAC;
    View1Df KLAS, SIDE_arr, COSF, SINF;
    View1Df FLUX0, FLUX1, FLUX2, FLUX3;

    KOKKOS_INLINE_FUNCTION
    void operator()(const int idx) const {
        if (idx >= CELL * 4) return;
        int pos = idx / 4;
        if (pos >= CELL) return;

        float H1 = H(pos), U1 = U(pos), V1 = V(pos), Z1 = Z(pos);
        float v_ZB1 = ZB1(pos);
        int NC = NAC(idx) - 1;
        float KP = KLAS(idx);
        float COSJ = COSF(idx), SINJ = SINF(idx);

        float QL_h = H1;
        float QL_u = U1 * COSJ + V1 * SINJ;
        float QL_v = V1 * COSJ - U1 * SINJ;
        float CL = Kokkos::sqrt(G * H1);
        float FIL = QL_u + 2 * CL;
        float ZI = Kokkos::fmax(Z1, v_ZB1);

        float HC = 0, BC = 0, ZC = 0, UC = 0, VC = 0;
        if (NC >= 0 && NC < CELL) {
            HC = Kokkos::fmax(H(NC), HM1);
            BC = ZBC(NC);
            ZC = Kokkos::fmax(BC, Z(NC));
            UC = U(NC); VC = V(NC);
        }

        float f0 = 0, f1 = 0, f2 = 0, f3 = 0;

        if (KP >= 1 && KP <= 8 || KP >= 10) {
            // Boundary: simplified as wall
            f3 = HALF_G * H1 * H1;
        } else if (H1 <= HM1 && HC <= HM1) {
            // both dry
        } else if (ZI <= BC) {
            f0 = -C1_C * Kokkos::pow(HC, 1.5f);
            f1 = H1 * QL_u * Kokkos::fabs(QL_u);
            f3 = HALF_G * H1 * H1;
        } else if (ZC <= ZBC(pos)) {
            f0 = C1_C * Kokkos::pow(H1, 1.5f);
            f1 = H1 * Kokkos::fabs(QL_u) * QL_u;
            f2 = H1 * Kokkos::fabs(QL_u) * QL_v;
        } else if (H1 <= HM2) {
            if (ZC > ZI) {
                float DH = Kokkos::fmax(ZC - ZBC(pos), HM1);
                float UN = -C1_C * Kokkos::sqrt(DH);
                f0 = DH * UN; f1 = f0 * UN;
                f2 = f0 * (VC * COSJ - UC * SINJ);
                f3 = HALF_G * H1 * H1;
            } else {
                f0 = C1_C * Kokkos::pow(H1, 1.5f);
                f3 = HALF_G * H1 * H1;
            }
        } else if (HC <= HM2) {
            if (ZI > ZC) {
                float DH = Kokkos::fmax(ZI - BC, HM1);
                float UN = C1_C * Kokkos::sqrt(DH);
                float HC1 = ZC - ZBC(pos);
                f0 = DH * UN; f1 = f0 * UN; f2 = f0 * QL_v;
                f3 = HALF_G * HC1 * HC1;
            } else {
                f0 = -C1_C * Kokkos::pow(HC, 1.5f);
                f1 = H1 * QL_u * QL_u;
                f3 = HALF_G * H1 * H1;
            }
        } else {
            // Interior: OSHER
            if ((int)KP == 0 && pos < NC) {
                float QR_h = Kokkos::fmax(ZC - ZBC(pos), HM1);
                float UR = UC * COSJ + VC * SINJ;
                float ratio = Kokkos::fmin(HC / QR_h, 1.5f);
                float QR_u = UR * ratio;
                if (HC <= HM2 || QR_h <= HM2) QR_u = Kokkos::copysign(VMIN, UR);
                float QR_v = VC * COSJ - UC * SINJ;
                float r0, r1, r2, r3;
                osher(QL_h, QL_u, QL_v, QR_h, QR_u, QR_v, FIL, H(pos), r0, r1, r2, r3);
                f0 = r0;
                f1 = r1 + (1 - ratio) * HC * UR * UR / 2;
                f2 = r2; f3 = r3;
            } else {
                float COSJ1 = -COSJ, SINJ1 = -SINJ;
                float L1h = H(NC), L1u = U(NC)*COSJ1+V(NC)*SINJ1, L1v = V(NC)*COSJ1-U(NC)*SINJ1;
                float CL1 = Kokkos::sqrt(G * H(NC)), FIL1 = L1u + 2*CL1;
                float HC2 = Kokkos::fmax(H1, HM1), ZC1 = Kokkos::fmax(ZBC(pos), Z1);
                float R1h = Kokkos::fmax(ZC1 - ZBC(NC), HM1);
                float UR1 = U1*COSJ1 + V1*SINJ1;
                float ratio1 = Kokkos::fmin(HC2 / R1h, 1.5f);
                float R1u = UR1 * ratio1;
                if (HC2 <= HM2 || R1h <= HM2) R1u = Kokkos::copysign(VMIN, UR1);
                float R1v = V1*COSJ1 - U1*SINJ1;
                float mr0, mr1, mr2, mr3;
                osher(L1h, L1u, L1v, R1h, R1u, R1v, FIL1, H(NC), mr0, mr1, mr2, mr3);
                f0 = -mr0;
                f1 = mr1 + (1 - ratio1) * HC2 * UR1 * UR1 / 2;
                f2 = mr2;
                float ZA = Kokkos::sqrt(mr3 / HALF_G) + BC;
                float HC3 = Kokkos::fmax(ZA - ZBC(pos), 0.0f);
                f3 = HALF_G * HC3 * HC3;
            }
        }
        FLUX0(idx) = f0; FLUX1(idx) = f1; FLUX2(idx) = f2; FLUX3(idx) = f3;
    }
};

// ===== UpdateCell functor (cell-parallel) =====
struct UpdateCell {
    int CELL;
    float DT, HM1, HM2;
    View1Df H, U, V, Z, W, ZBC, AREA, FNC;
    View1Df SIDE_arr, SLCOS, SLSIN;
    View1Df FLUX0, FLUX1, FLUX2, FLUX3;

    KOKKOS_INLINE_FUNCTION
    void operator()(const int pos) const {
        if (pos >= CELL) return;
        float WH = 0, WU = 0, WV = 0;
        for (int idx = 4*pos; idx < 4*(pos+1); idx++) {
            float FLR1 = FLUX1(idx) + FLUX3(idx);
            float FLR2 = FLUX2(idx);
            float SL = SIDE_arr(idx), SLCA = SLCOS(idx), SLSA = SLSIN(idx);
            WH += SL * FLUX0(idx);
            WU += SLCA * FLR1 - SLSA * FLR2;
            WV += SLSA * FLR1 + SLCA * FLR2;
        }
        float H1 = H(pos), U1 = U(pos), V1 = V(pos);
        float DTA = DT / AREA(pos);
        float H2 = Kokkos::fmax(H1 - DTA * WH + QLUA, HM1);
        float Z2 = H2 + ZBC(pos);
        float U2 = 0, V2 = 0;
        if (H2 > HM1) {
            if (H2 <= HM2) {
                U2 = Kokkos::copysign(Kokkos::fmin(VMIN, Kokkos::fabs(U1)), U1);
                V2 = Kokkos::copysign(Kokkos::fmin(VMIN, Kokkos::fabs(V1)), V1);
            } else {
                float WSF = FNC(pos) * Kokkos::sqrt(U1*U1+V1*V1) / Kokkos::pow(H1, 0.33333f);
                U2 = (H1*U1 - DTA*WU - DT*WSF*U1) / H2;
                V2 = (H1*V1 - DTA*WV - DT*WSF*V1) / H2;
                U2 = Kokkos::copysign(Kokkos::fmin(Kokkos::fabs(U2), 15.0f), U2);
                V2 = Kokkos::copysign(Kokkos::fmin(Kokkos::fabs(V2), 15.0f), V2);
            }
        }
        H(pos) = H2; U(pos) = U2; V(pos) = V2;
        Z(pos) = Z2; W(pos) = Kokkos::sqrt(U2*U2 + V2*V2);
    }
};

// ===== Binary file loader =====
template<typename T>
void loadBinary(const std::string& path, T* dst, int count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); exit(1); }
    f.read(reinterpret_cast<char*>(dst), count * sizeof(T));
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        int steps = (argc > 1) ? atoi(argv[1]) : 900;
        int repeat = (argc > 2) ? atoi(argv[2]) : 10;
        std::string bin = (argc > 3) ? std::string(argv[3])
            : "/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/benchmark/F2_hydro_refactored/data/binary/";

        // Read params line by line
        std::ifstream pf(bin + "params.txt");
        std::string line;
        std::getline(pf, line); int CELL = std::stoi(line);
        std::getline(pf, line); float HM1 = std::stof(line);
        std::getline(pf, line); float HM2 = std::stof(line);
        std::getline(pf, line); float DT = std::stof(line);
        std::getline(pf, line); int steps_per_day = std::stoi(line);
        printf("CELL=%d, HM1=%.4f, HM2=%.4f, DT=%.1f, steps=%d\n", CELL, HM1, HM2, DT, steps);

        int nSides = CELL * 4;
        View1Df H("H", CELL), U("U", CELL), V("V", CELL), Z("Z", CELL), W("W", CELL);
        View1Df ZBC_v("ZBC", CELL), ZB1("ZB1", CELL), AREA("AREA", CELL), FNC("FNC", CELL);
        View1Di NAC("NAC", nSides);
        View1Df KLAS("KLAS", nSides), SIDE_v("SIDE", nSides);
        View1Df COSF("COSF", nSides), SINF("SINF", nSides);
        View1Df SLCOS("SLCOS", nSides), SLSIN("SLSIN", nSides);
        View1Df FLUX0("FLUX0", nSides), FLUX1("FLUX1", nSides);
        View1Df FLUX2("FLUX2", nSides), FLUX3("FLUX3", nSides);

        // Load from binary files via host mirrors
        {
            auto hH=Kokkos::create_mirror_view(H); loadBinary(bin+"H.bin",hH.data(),CELL); Kokkos::deep_copy(H,hH);
            auto hU=Kokkos::create_mirror_view(U); loadBinary(bin+"U.bin",hU.data(),CELL); Kokkos::deep_copy(U,hU);
            auto hV=Kokkos::create_mirror_view(V); loadBinary(bin+"V.bin",hV.data(),CELL); Kokkos::deep_copy(V,hV);
            auto hZ=Kokkos::create_mirror_view(Z); loadBinary(bin+"Z.bin",hZ.data(),CELL); Kokkos::deep_copy(Z,hZ);
            auto hW=Kokkos::create_mirror_view(W); loadBinary(bin+"W.bin",hW.data(),CELL); Kokkos::deep_copy(W,hW);
            auto hZBC=Kokkos::create_mirror_view(ZBC_v); loadBinary(bin+"ZBC.bin",hZBC.data(),CELL); Kokkos::deep_copy(ZBC_v,hZBC);
            auto hZB1=Kokkos::create_mirror_view(ZB1); loadBinary(bin+"ZB1.bin",hZB1.data(),CELL); Kokkos::deep_copy(ZB1,hZB1);
            auto hAR=Kokkos::create_mirror_view(AREA); loadBinary(bin+"AREA.bin",hAR.data(),CELL); Kokkos::deep_copy(AREA,hAR);
            auto hFNC=Kokkos::create_mirror_view(FNC); loadBinary(bin+"FNC.bin",hFNC.data(),CELL); Kokkos::deep_copy(FNC,hFNC);
            auto hNAC=Kokkos::create_mirror_view(NAC); loadBinary(bin+"NAC.bin",hNAC.data(),nSides); Kokkos::deep_copy(NAC,hNAC);
            auto hKL=Kokkos::create_mirror_view(KLAS); loadBinary(bin+"KLAS.bin",hKL.data(),nSides); Kokkos::deep_copy(KLAS,hKL);
            auto hSD=Kokkos::create_mirror_view(SIDE_v); loadBinary(bin+"SIDE.bin",hSD.data(),nSides); Kokkos::deep_copy(SIDE_v,hSD);
            auto hCF=Kokkos::create_mirror_view(COSF); loadBinary(bin+"COSF.bin",hCF.data(),nSides); Kokkos::deep_copy(COSF,hCF);
            auto hSF=Kokkos::create_mirror_view(SINF); loadBinary(bin+"SINF.bin",hSF.data(),nSides); Kokkos::deep_copy(SINF,hSF);
            auto hSC=Kokkos::create_mirror_view(SLCOS); loadBinary(bin+"SLCOS.bin",hSC.data(),nSides); Kokkos::deep_copy(SLCOS,hSC);
            auto hSS=Kokkos::create_mirror_view(SLSIN); loadBinary(bin+"SLSIN.bin",hSS.data(),nSides); Kokkos::deep_copy(SLSIN,hSS);
        }
        Kokkos::fence();
        printf("Mesh loaded: %d cells, %d sides\n", CELL, nSides);

        // Benchmark
        CalcFlux flux_fn{CELL, HM1, HM2, H, U, V, Z, ZBC_v, ZB1, NAC, KLAS, SIDE_v, COSF, SINF, FLUX0, FLUX1, FLUX2, FLUX3};
        UpdateCell update_fn{CELL, DT, HM1, HM2, H, U, V, Z, W, ZBC_v, AREA, FNC, SIDE_v, SLCOS, SLSIN, FLUX0, FLUX1, FLUX2, FLUX3};

        // Warmup
        for (int w = 0; w < 5; w++) {
            for (int s = 0; s < steps; s++) {
                Kokkos::parallel_for("flux", nSides, flux_fn);
                Kokkos::parallel_for("update", CELL, update_fn);
            }
            Kokkos::fence();
        }

        // Timed
        std::vector<double> times;
        for (int r = 0; r < repeat; r++) {
            Kokkos::fence();
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int s = 0; s < steps; s++) {
                Kokkos::parallel_for("flux", nSides, flux_fn);
                Kokkos::parallel_for("update", CELL, update_fn);
            }
            Kokkos::fence();
            times.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count());
        }
        std::sort(times.begin(), times.end());
        printf("Kokkos F2: %d steps, median=%.3fms\n", steps, times[repeat/2]);
        printf("CSV: kokkos_f2,%d,%.3f\n", steps, times[repeat/2]);
    }
    Kokkos::finalize();
}
