/**
 * F2: Hydro-Cal OSHER — Kernel Argument Overhead A/B Test.
 * Original: 17+18 pointer args per kernel launch.
 * Optimized: 1 struct pointer + 3-4 scalar args (matches Taichi ti.field approach).
 *
 * Build: nvcc -O3 -arch=sm_90 -rdc=true hydro_osher_aligned.cu -o hydro_osher_aligned -lcudadevrt
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <chrono>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__constant__ float G_c     = 9.81f;
__constant__ float HALF_G_c = 4.905f;
__constant__ float C0_c    = 1.33f;
__constant__ float C1_C_c  = 1.7f;
__constant__ float VMIN_c  = 0.001f;
__constant__ float QLUA_c  = 0.0f;

// All 20 device pointers packed into one struct (160 bytes)
struct MeshPtrs {
    float *H, *U, *V, *Z, *W;           // mutable state (per-cell)
    float *ZBC, *ZB1, *AREA, *FNC;      // constant geometry (per-cell)
    int   *NAC;                          // topology (per-side)
    float *KLAS, *SIDE, *COSF, *SINF;   // topology (per-side)
    float *SLCOS, *SLSIN;               // topology (per-side)
    float *FLUX0, *FLUX1, *FLUX2, *FLUX3; // flux scratch (per-side)
};

// ===== Device helpers =====
__device__ __forceinline__
void QF(float h, float u, float v, float& F0, float& F1, float& F2, float& F3) {
    F0 = h * u; F1 = F0 * u; F2 = F0 * v; F3 = HALF_G_c * h * h;
}

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
    #define ADD(hh, uu, vv, ss) { float _f0,_f1,_f2,_f3; QF(hh,uu,vv,_f0,_f1,_f2,_f3); \
        R0+=_f0*(ss); R1+=_f1*(ss); R2+=_f2*(ss); R3+=_f3*(ss); }
    #define QS1(ss) ADD(QL_h, QL_u, QL_v, ss)
    #define QS2(ss) { float _U=fil/3.0f,_H=_U*_U/G_c; ADD(_H,_U,QL_v,ss) }
    #define QS3(ss) { float _ua=(fil+fir)/2.0f; float _fl=fil-_ua; float _H=_fl*_fl/(4.0f*G_c); ADD(_H,_ua,QL_v,ss) }
    #define QS5(ss) { float _ua=(fil+fir)/2.0f; float _fr=fir-_ua; float _H=_fr*_fr/(4.0f*G_c); ADD(_H,_ua,QR_v,ss) }
    #define QS6(ss) { float _U=fir/3.0f,_H=_U*_U/G_c; ADD(_H,_U,QR_v,ss) }
    #define QS7(ss) ADD(QR_h, QR_u, QR_v, ss)
    switch(K1) {
    case 1: switch(K2){case 1:QS2(1);break;case 2:QS3(1);break;case 3:QS5(1);break;case 4:QS6(1);break;} break;
    case 2: switch(K2){case 1:QS1(1);break;case 2:QS1(1);QS2(-1);QS3(1);break;case 3:QS1(1);QS2(-1);QS5(1);break;case 4:QS1(1);QS2(-1);QS6(1);break;} break;
    case 3: switch(K2){case 1:QS2(1);QS6(-1);QS7(1);break;case 2:QS3(1);QS6(-1);QS7(1);break;case 3:QS5(1);QS6(-1);QS7(1);break;case 4:QS7(1);break;} break;
    case 4: switch(K2){case 1:QS1(1);QS6(-1);QS7(1);break;case 2:QS1(1);QS2(-1);QS3(1);QS6(-1);QS7(1);break;case 3:QS1(1);QS2(-1);QS5(1);QS6(-1);QS7(1);break;case 4:QS1(1);QS2(-1);QS7(1);break;} break;
    }
    #undef ADD
    #undef QS1
    #undef QS2
    #undef QS3
    #undef QS5
    #undef QS6
    #undef QS7
}

// ===== Flux computation (shared by both kernel versions) =====
__device__ __forceinline__
void compute_flux(int idx, int CELL, float HM1, float HM2,
                  const float* H, const float* U, const float* V, const float* Z,
                  const float* ZBC, const float* ZB1,
                  const int* NAC, const float* KLAS,
                  const float* COSF, const float* SINF,
                  float& f0, float& f1, float& f2, float& f3) {
    int pos = idx / 4;
    float H1 = H[pos], U1 = U[pos], V1 = V[pos], Z1 = Z[pos];
    float v_ZB1 = ZB1[pos];
    int NC = NAC[idx] - 1;
    float KP = KLAS[idx];
    float COSJ = COSF[idx], SINJ = SINF[idx];

    float QL_h = H1, QL_u = U1*COSJ+V1*SINJ, QL_v = V1*COSJ-U1*SINJ;
    float CL = sqrtf(G_c * H1), FIL = QL_u + 2.0f*CL;
    float ZI = fmaxf(Z1, v_ZB1);

    float HC=0, BC=0, ZC=0, UC=0, VC=0;
    if (NC >= 0 && NC < CELL) {
        HC = fmaxf(H[NC], HM1); BC = ZBC[NC];
        ZC = fmaxf(BC, Z[NC]); UC = U[NC]; VC = V[NC];
    }

    f0=0; f1=0; f2=0; f3=0;
    if (KP >= 1 && KP <= 8 || KP >= 10) {
        f3 = HALF_G_c * H1 * H1;
    } else if (H1 <= HM1 && HC <= HM1) {
    } else if (ZI <= BC) {
        f0 = -C1_C_c*powf(HC,1.5f); f1 = H1*QL_u*fabsf(QL_u); f3 = HALF_G_c*H1*H1;
    } else if (ZC <= ZBC[pos]) {
        f0 = C1_C_c*powf(H1,1.5f); f1 = H1*fabsf(QL_u)*QL_u; f2 = H1*fabsf(QL_u)*QL_v;
    } else if (H1 <= HM2) {
        if (ZC > ZI) {
            float DH=fmaxf(ZC-ZBC[pos],HM1), UN=-C1_C_c*sqrtf(DH);
            f0=DH*UN; f1=f0*UN; f2=f0*(VC*COSJ-UC*SINJ); f3=HALF_G_c*H1*H1;
        } else { f0=C1_C_c*powf(H1,1.5f); f3=HALF_G_c*H1*H1; }
    } else if (HC <= HM2) {
        if (ZI > ZC) {
            float DH=fmaxf(ZI-BC,HM1), UN=C1_C_c*sqrtf(DH), HC1=ZC-ZBC[pos];
            f0=DH*UN; f1=f0*UN; f2=f0*QL_v; f3=HALF_G_c*HC1*HC1;
        } else { f0=-C1_C_c*powf(HC,1.5f); f1=H1*QL_u*QL_u; f3=HALF_G_c*H1*H1; }
    } else {
        if ((int)KP==0 && pos < NC) {
            float QR_h=fmaxf(ZC-ZBC[pos],HM1), UR=UC*COSJ+VC*SINJ;
            float ratio=fminf(HC/QR_h,1.5f), QR_u=UR*ratio;
            if (HC<=HM2||QR_h<=HM2) QR_u=copysignf(VMIN_c,UR);
            float QR_v=VC*COSJ-UC*SINJ;
            float r0,r1,r2,r3;
            osher(QL_h,QL_u,QL_v,QR_h,QR_u,QR_v,FIL,H[pos],r0,r1,r2,r3);
            f0=r0; f1=r1+(1.0f-ratio)*HC*UR*UR/2.0f; f2=r2; f3=r3;
        } else {
            float COSJ1=-COSJ, SINJ1=-SINJ;
            float L1h=H[NC], L1u=U[NC]*COSJ1+V[NC]*SINJ1, L1v=V[NC]*COSJ1-U[NC]*SINJ1;
            float CL1=sqrtf(G_c*H[NC]), FIL1=L1u+2.0f*CL1;
            float HC2=fmaxf(H1,HM1), ZC1=fmaxf(ZBC[pos],Z1);
            float R1h=fmaxf(ZC1-ZBC[NC],HM1), UR1=U1*COSJ1+V1*SINJ1;
            float ratio1=fminf(HC2/R1h,1.5f), R1u=UR1*ratio1;
            if (HC2<=HM2||R1h<=HM2) R1u=copysignf(VMIN_c,UR1);
            float R1v=V1*COSJ1-U1*SINJ1;
            float mr0,mr1,mr2,mr3;
            osher(L1h,L1u,L1v,R1h,R1u,R1v,FIL1,H[NC],mr0,mr1,mr2,mr3);
            f0=-mr0; f1=mr1+(1.0f-ratio1)*HC2*UR1*UR1/2.0f; f2=mr2;
            float ZA=sqrtf(mr3/HALF_G_c)+BC, HC3=fmaxf(ZA-ZBC[pos],0.0f);
            f3=HALF_G_c*HC3*HC3;
        }
    }
}

// ===== Update computation (shared) =====
__device__ __forceinline__
void compute_update(int pos, float DT, float HM1, float HM2,
                    float* H, float* U, float* V, float* Z, float* W,
                    const float* ZBC, const float* AREA, const float* FNC,
                    const float* SIDE, const float* SLCOS, const float* SLSIN,
                    const float* FLUX0, const float* FLUX1, const float* FLUX2, const float* FLUX3) {
    float WH=0, WU=0, WV=0;
    for (int i = 0; i < 4; i++) {
        int idx = 4*pos+i;
        float FLR1=FLUX1[idx]+FLUX3[idx], FLR2=FLUX2[idx];
        float SL=SIDE[idx], SLCA=SLCOS[idx], SLSA=SLSIN[idx];
        WH += SL*FLUX0[idx]; WU += SLCA*FLR1-SLSA*FLR2; WV += SLSA*FLR1+SLCA*FLR2;
    }
    float H1=H[pos], U1=U[pos], V1=V[pos];
    float DTA=DT/AREA[pos];
    float H2=fmaxf(H1-DTA*WH+QLUA_c, HM1), Z2=H2+ZBC[pos];
    float U2=0, V2=0;
    if (H2 > HM1) {
        if (H2 <= HM2) {
            U2=copysignf(fminf(VMIN_c,fabsf(U1)),U1);
            V2=copysignf(fminf(VMIN_c,fabsf(V1)),V1);
        } else {
            float WSF=FNC[pos]*sqrtf(U1*U1+V1*V1)/powf(H1,0.33333f);
            U2=(H1*U1-DTA*WU-DT*WSF*U1)/H2; V2=(H1*V1-DTA*WV-DT*WSF*V1)/H2;
            U2=copysignf(fminf(fabsf(U2),15.0f),U2); V2=copysignf(fminf(fabsf(V2),15.0f),V2);
        }
    }
    H[pos]=H2; U[pos]=U2; V[pos]=V2; Z[pos]=Z2; W[pos]=sqrtf(U2*U2+V2*V2);
}

// =====================================
// ORIGINAL KERNELS (17 + 18 args)
// =====================================
__global__
void calculate_flux_orig(int CELL, float HM1, float HM2,
                    const float* __restrict__ H, const float* __restrict__ U,
                    const float* __restrict__ V, const float* __restrict__ Z,
                    const float* __restrict__ ZBC, const float* __restrict__ ZB1,
                    const int* __restrict__ NAC, const float* __restrict__ KLAS,
                    const float* __restrict__ SIDE, const float* __restrict__ COSF,
                    const float* __restrict__ SINF,
                    float* __restrict__ FLUX0, float* __restrict__ FLUX1,
                    float* __restrict__ FLUX2, float* __restrict__ FLUX3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= CELL * 4) return;
    float f0, f1, f2, f3;
    compute_flux(idx, CELL, HM1, HM2, H, U, V, Z, ZBC, ZB1, NAC, KLAS, COSF, SINF, f0, f1, f2, f3);
    FLUX0[idx] = f0; FLUX1[idx] = f1; FLUX2[idx] = f2; FLUX3[idx] = f3;
}

__global__
void update_cell_orig(int CELL, float DT, float HM1, float HM2,
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
    compute_update(pos, DT, HM1, HM2, H, U, V, Z, W, ZBC, AREA, FNC, SIDE, SLCOS, SLSIN, FLUX0, FLUX1, FLUX2, FLUX3);
}

// =====================================
// OPTIMIZED KERNELS (4-5 args: 1 struct ptr + scalars)
// =====================================
__global__
void calculate_flux_opt(int CELL, float HM1, float HM2, const MeshPtrs* __restrict__ M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= CELL * 4) return;
    float f0, f1, f2, f3;
    compute_flux(idx, CELL, HM1, HM2, M->H, M->U, M->V, M->Z, M->ZBC, M->ZB1, M->NAC, M->KLAS, M->COSF, M->SINF, f0, f1, f2, f3);
    M->FLUX0[idx] = f0; M->FLUX1[idx] = f1; M->FLUX2[idx] = f2; M->FLUX3[idx] = f3;
}

__global__
void update_cell_opt(int CELL, float DT, float HM1, float HM2, MeshPtrs* __restrict__ M) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= CELL) return;
    compute_update(pos, DT, HM1, HM2, M->H, M->U, M->V, M->Z, M->W, M->ZBC, M->AREA, M->FNC, M->SIDE, M->SLCOS, M->SLSIN, M->FLUX0, M->FLUX1, M->FLUX2, M->FLUX3);
}

// =====================================
// OPTIMIZED PERSISTENT (6 args: struct ptr + 4 scalars + steps)
// =====================================
__global__
void persistent_fused_opt(int CELL, float DT, float HM1, float HM2, int steps, MeshPtrs* __restrict__ M) {
    cg::grid_group grid = cg::this_grid();
    int nSides = CELL * 4;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int step = 0; step < steps; step++) {
        for (int idx = tid; idx < nSides; idx += total_threads) {
            float f0, f1, f2, f3;
            compute_flux(idx, CELL, HM1, HM2, M->H, M->U, M->V, M->Z, M->ZBC, M->ZB1, M->NAC, M->KLAS, M->COSF, M->SINF, f0, f1, f2, f3);
            M->FLUX0[idx] = f0; M->FLUX1[idx] = f1; M->FLUX2[idx] = f2; M->FLUX3[idx] = f3;
        }
        grid.sync();
        for (int pos = tid; pos < CELL; pos += total_threads) {
            compute_update(pos, DT, HM1, HM2, M->H, M->U, M->V, M->Z, M->W, M->ZBC, M->AREA, M->FNC, M->SIDE, M->SLCOS, M->SLSIN, M->FLUX0, M->FLUX1, M->FLUX2, M->FLUX3);
        }
        if (step < steps - 1) grid.sync();
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

    printf("=== F2 Hydro-Cal: Kernel Argument Overhead A/B Test ===\n");
    printf("CELL=%d, steps=%d, repeat=%d\n", CELL, steps, repeat);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s, SMs=%d\n", prop.name, prop.multiProcessorCount);

    // ===== Host arrays =====
    std::vector<float> h_H(CELL), h_U(CELL), h_V(CELL), h_Z(CELL), h_W(CELL);
    std::vector<float> h_ZBC(CELL), h_ZB1(CELL), h_AREA(CELL), h_FNC(CELL);
    std::vector<int>   h_NAC(nSides);
    std::vector<float> h_KLAS(nSides), h_SIDE(nSides);
    std::vector<float> h_COSF(nSides), h_SINF(nSides);
    std::vector<float> h_SLCOS(nSides), h_SLSIN(nSides);

    loadBinary(binDir+"H.bin", h_H.data(), CELL);
    loadBinary(binDir+"U.bin", h_U.data(), CELL);
    loadBinary(binDir+"V.bin", h_V.data(), CELL);
    loadBinary(binDir+"Z.bin", h_Z.data(), CELL);
    loadBinary(binDir+"W.bin", h_W.data(), CELL);
    loadBinary(binDir+"ZBC.bin", h_ZBC.data(), CELL);
    loadBinary(binDir+"ZB1.bin", h_ZB1.data(), CELL);
    loadBinary(binDir+"AREA.bin", h_AREA.data(), CELL);
    loadBinary(binDir+"FNC.bin", h_FNC.data(), CELL);
    loadBinary(binDir+"NAC.bin", h_NAC.data(), nSides);
    loadBinary(binDir+"KLAS.bin", h_KLAS.data(), nSides);
    loadBinary(binDir+"SIDE.bin", h_SIDE.data(), nSides);
    loadBinary(binDir+"COSF.bin", h_COSF.data(), nSides);
    loadBinary(binDir+"SINF.bin", h_SINF.data(), nSides);
    loadBinary(binDir+"SLCOS.bin", h_SLCOS.data(), nSides);
    loadBinary(binDir+"SLSIN.bin", h_SLSIN.data(), nSides);

    // ===== Device arrays =====
    float *d_H, *d_U, *d_V, *d_Z, *d_W;
    float *d_ZBC, *d_ZB1, *d_AREA, *d_FNC;
    int   *d_NAC;
    float *d_KLAS, *d_SIDE, *d_COSF, *d_SINF, *d_SLCOS, *d_SLSIN;
    float *d_FLUX0, *d_FLUX1, *d_FLUX2, *d_FLUX3;

    CUDA_CHECK(cudaMalloc(&d_H,    CELL*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_U,    CELL*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V,    CELL*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Z,    CELL*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W,    CELL*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ZBC,  CELL*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ZB1,  CELL*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_AREA, CELL*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_FNC,  CELL*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_NAC,  nSides*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_KLAS, nSides*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_SIDE, nSides*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_COSF, nSides*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_SINF, nSides*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_SLCOS, nSides*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_SLSIN, nSides*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_FLUX0, nSides*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_FLUX1, nSides*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_FLUX2, nSides*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_FLUX3, nSides*sizeof(float)));

    // Copy constant geometry/topology
    CUDA_CHECK(cudaMemcpy(d_ZBC, h_ZBC.data(), CELL*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ZB1, h_ZB1.data(), CELL*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_AREA, h_AREA.data(), CELL*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_FNC, h_FNC.data(), CELL*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_NAC, h_NAC.data(), nSides*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_KLAS, h_KLAS.data(), nSides*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_SIDE, h_SIDE.data(), nSides*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_COSF, h_COSF.data(), nSides*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_SINF, h_SINF.data(), nSides*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_SLCOS, h_SLCOS.data(), nSides*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_SLSIN, h_SLSIN.data(), nSides*sizeof(float), cudaMemcpyHostToDevice));

    // ===== Allocate MeshPtrs struct on device =====
    MeshPtrs h_mesh = {
        d_H, d_U, d_V, d_Z, d_W,
        d_ZBC, d_ZB1, d_AREA, d_FNC,
        d_NAC, d_KLAS, d_SIDE, d_COSF, d_SINF, d_SLCOS, d_SLSIN,
        d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3
    };
    MeshPtrs* d_meshptr;
    CUDA_CHECK(cudaMalloc(&d_meshptr, sizeof(MeshPtrs)));
    CUDA_CHECK(cudaMemcpy(d_meshptr, &h_mesh, sizeof(MeshPtrs), cudaMemcpyHostToDevice));

    auto uploadState = [&]() {
        CUDA_CHECK(cudaMemcpy(d_H, h_H.data(), CELL*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), CELL*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), CELL*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Z, h_Z.data(), CELL*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), CELL*sizeof(float), cudaMemcpyHostToDevice));
    };

    int fluxThreads = 256, fluxBlocks = (nSides+255)/256;
    int updateThreads = 256, updateBlocks = (CELL+255)/256;

    printf("Original:  flux=17 args (%zu B), update=18 args (%zu B)\n", 17*8UL, 18*8UL);
    printf("Optimized: flux=4 args (%zu B), update=5 args (%zu B)\n",
           3*4UL+8UL, 4*4UL+8UL);
    printf("Struct size: %zu bytes on device\n\n", sizeof(MeshPtrs));

    // ================================================================
    printf("========== CUDA Graph A/B Test ==========\n");

    // --- [A] Original CUDA Graph ---
    printf("\n--- [A] Original (17+18 args) — CUDA Graph ---\n");
    {
        uploadState();
        cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));
        cudaGraph_t graph; cudaGraphExec_t graphExec;

        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        calculate_flux_orig<<<fluxBlocks, fluxThreads, 0, stream>>>(CELL, HM1, HM2, d_H, d_U, d_V, d_Z, d_ZBC, d_ZB1, d_NAC, d_KLAS, d_SIDE, d_COSF, d_SINF, d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3);
        update_cell_orig<<<updateBlocks, updateThreads, 0, stream>>>(CELL, DT, HM1, HM2, d_H, d_U, d_V, d_Z, d_W, d_ZBC, d_AREA, d_FNC, d_SIDE, d_SLCOS, d_SLSIN, d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

        // Warmup
        uploadState();
        for (int s = 0; s < 20; s++) CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<double> times;
        for (int r = 0; r < repeat; r++) {
            uploadState(); CUDA_CHECK(cudaStreamSynchronize(stream));
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int s = 0; s < steps; s++) CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            times.push_back(std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-t0).count());
        }
        std::sort(times.begin(), times.end());
        double median = times[repeat/2];
        printf("[A-Graph-Orig] median=%.3f ms, %.2f us/step\n", median, median/steps*1000.0);

        CUDA_CHECK(cudaGraphExecDestroy(graphExec));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    // --- [B] Optimized CUDA Graph ---
    printf("\n--- [B] Optimized (4+5 args, struct ptr) — CUDA Graph ---\n");
    {
        uploadState();
        cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));
        cudaGraph_t graph; cudaGraphExec_t graphExec;

        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        calculate_flux_opt<<<fluxBlocks, fluxThreads, 0, stream>>>(CELL, HM1, HM2, d_meshptr);
        update_cell_opt<<<updateBlocks, updateThreads, 0, stream>>>(CELL, DT, HM1, HM2, d_meshptr);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

        // Warmup
        uploadState();
        for (int s = 0; s < 20; s++) CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<double> times;
        for (int r = 0; r < repeat; r++) {
            uploadState(); CUDA_CHECK(cudaStreamSynchronize(stream));
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int s = 0; s < steps; s++) CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            times.push_back(std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-t0).count());
        }
        std::sort(times.begin(), times.end());
        double median = times[repeat/2];
        printf("[B-Graph-Opt]  median=%.3f ms, %.2f us/step\n", median, median/steps*1000.0);

        CUDA_CHECK(cudaGraphExecDestroy(graphExec));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    // ================================================================
    printf("\n========== Async Launch A/B Test ==========\n");

    // --- [C] Original Async ---
    printf("\n--- [C] Original — Async ---\n");
    {
        uploadState();
        for (int s = 0; s < 10; s++) {
            calculate_flux_orig<<<fluxBlocks, fluxThreads>>>(CELL, HM1, HM2, d_H, d_U, d_V, d_Z, d_ZBC, d_ZB1, d_NAC, d_KLAS, d_SIDE, d_COSF, d_SINF, d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3);
            update_cell_orig<<<updateBlocks, updateThreads>>>(CELL, DT, HM1, HM2, d_H, d_U, d_V, d_Z, d_W, d_ZBC, d_AREA, d_FNC, d_SIDE, d_SLCOS, d_SLSIN, d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> times;
        for (int r = 0; r < repeat; r++) {
            uploadState(); CUDA_CHECK(cudaDeviceSynchronize());
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int s = 0; s < steps; s++) {
                calculate_flux_orig<<<fluxBlocks, fluxThreads>>>(CELL, HM1, HM2, d_H, d_U, d_V, d_Z, d_ZBC, d_ZB1, d_NAC, d_KLAS, d_SIDE, d_COSF, d_SINF, d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3);
                update_cell_orig<<<updateBlocks, updateThreads>>>(CELL, DT, HM1, HM2, d_H, d_U, d_V, d_Z, d_W, d_ZBC, d_AREA, d_FNC, d_SIDE, d_SLCOS, d_SLSIN, d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            times.push_back(std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-t0).count());
        }
        std::sort(times.begin(), times.end());
        double median = times[repeat/2];
        printf("[C-Async-Orig] median=%.3f ms, %.2f us/step\n", median, median/steps*1000.0);
    }

    // --- [D] Optimized Async ---
    printf("\n--- [D] Optimized — Async ---\n");
    {
        uploadState();
        for (int s = 0; s < 10; s++) {
            calculate_flux_opt<<<fluxBlocks, fluxThreads>>>(CELL, HM1, HM2, d_meshptr);
            update_cell_opt<<<updateBlocks, updateThreads>>>(CELL, DT, HM1, HM2, d_meshptr);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> times;
        for (int r = 0; r < repeat; r++) {
            uploadState(); CUDA_CHECK(cudaDeviceSynchronize());
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int s = 0; s < steps; s++) {
                calculate_flux_opt<<<fluxBlocks, fluxThreads>>>(CELL, HM1, HM2, d_meshptr);
                update_cell_opt<<<updateBlocks, updateThreads>>>(CELL, DT, HM1, HM2, d_meshptr);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            times.push_back(std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-t0).count());
        }
        std::sort(times.begin(), times.end());
        double median = times[repeat/2];
        printf("[D-Async-Opt]  median=%.3f ms, %.2f us/step\n", median, median/steps*1000.0);
    }

    // ================================================================
    printf("\n========== Persistent Kernel ==========\n");

    // --- [E] Optimized Persistent ---
    printf("\n--- [E] Optimized Persistent (6 args, struct ptr) ---\n");
    {
        int persistentThreads = 256, maxBSm = 0;
        cudaError_t occErr = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBSm, persistent_fused_opt, persistentThreads, 0);
        if (occErr != cudaSuccess) {
            printf("[E-Persistent-Opt] SKIPPED: occupancy query failed (%s)\n", cudaGetErrorString(occErr));
            cudaGetLastError(); // clear error
        } else {
            int maxBlocks = maxBSm * prop.multiProcessorCount;
            int needed = (nSides + persistentThreads - 1) / persistentThreads;
            int pBlocks = std::min(maxBlocks, needed);
            printf("Persistent: %d blocks (max %d)\n", pBlocks, maxBlocks);

            void* kargs[] = { &CELL, &DT, &HM1, &HM2, &steps, &d_meshptr };
            uploadState();
            CUDA_CHECK(cudaLaunchCooperativeKernel((void*)persistent_fused_opt, dim3(pBlocks), dim3(persistentThreads), kargs));
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<double> times;
            for (int r = 0; r < repeat; r++) {
                uploadState(); CUDA_CHECK(cudaDeviceSynchronize());
                auto t0 = std::chrono::high_resolution_clock::now();
                CUDA_CHECK(cudaLaunchCooperativeKernel((void*)persistent_fused_opt, dim3(pBlocks), dim3(persistentThreads), kargs));
                CUDA_CHECK(cudaDeviceSynchronize());
                times.push_back(std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-t0).count());
            }
            std::sort(times.begin(), times.end());
            double median = times[repeat/2];
            printf("[E-Persistent-Opt] median=%.3f ms, %.2f us/step\n", median, median/steps*1000.0);
        }
    }

    // ================================================================
    printf("\n========== Correctness Check ==========\n");
    {
        int ck = 100;
        std::vector<float> ref_H(CELL), opt_H(CELL);

        uploadState();
        for (int s = 0; s < ck; s++) {
            calculate_flux_orig<<<fluxBlocks, fluxThreads>>>(CELL, HM1, HM2, d_H, d_U, d_V, d_Z, d_ZBC, d_ZB1, d_NAC, d_KLAS, d_SIDE, d_COSF, d_SINF, d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3);
            update_cell_orig<<<updateBlocks, updateThreads>>>(CELL, DT, HM1, HM2, d_H, d_U, d_V, d_Z, d_W, d_ZBC, d_AREA, d_FNC, d_SIDE, d_SLCOS, d_SLSIN, d_FLUX0, d_FLUX1, d_FLUX2, d_FLUX3);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaMemcpy(ref_H.data(), d_H, CELL*sizeof(float), cudaMemcpyDeviceToHost);

        uploadState();
        for (int s = 0; s < ck; s++) {
            calculate_flux_opt<<<fluxBlocks, fluxThreads>>>(CELL, HM1, HM2, d_meshptr);
            update_cell_opt<<<updateBlocks, updateThreads>>>(CELL, DT, HM1, HM2, d_meshptr);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaMemcpy(opt_H.data(), d_H, CELL*sizeof(float), cudaMemcpyDeviceToHost);

        double maxd = 0;
        for (int i = 0; i < CELL; i++) maxd = std::max(maxd, (double)fabsf(ref_H[i]-opt_H[i]));
        printf("Max |H_orig - H_opt| after %d steps: %.2e  %s\n", ck, maxd, maxd < 1e-5 ? "PASS" : "FAIL");
    }

    // ================================================================
    printf("\n========== GPU Compute Breakdown ==========\n");
    {
        uploadState();
        cudaEvent_t e0, e1, e2, e3;
        CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventCreate(&e2)); CUDA_CHECK(cudaEventCreate(&e3));

        for (int w = 0; w < 10; w++) {
            calculate_flux_opt<<<fluxBlocks, fluxThreads>>>(CELL, HM1, HM2, d_meshptr);
            update_cell_opt<<<updateBlocks, updateThreads>>>(CELL, DT, HM1, HM2, d_meshptr);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        int N = 100; float flux_ms, update_ms;
        CUDA_CHECK(cudaEventRecord(e0));
        for (int i = 0; i < N; i++) calculate_flux_opt<<<fluxBlocks, fluxThreads>>>(CELL, HM1, HM2, d_meshptr);
        CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&flux_ms, e0, e1));

        CUDA_CHECK(cudaEventRecord(e2));
        for (int i = 0; i < N; i++) update_cell_opt<<<updateBlocks, updateThreads>>>(CELL, DT, HM1, HM2, d_meshptr);
        CUDA_CHECK(cudaEventRecord(e3)); CUDA_CHECK(cudaEventSynchronize(e3));
        CUDA_CHECK(cudaEventElapsedTime(&update_ms, e2, e3));

        printf("Flux:   %.2f us/call\n", flux_ms/N*1000.0);
        printf("Update: %.2f us/call\n", update_ms/N*1000.0);
        printf("Total:  %.2f us/step\n", (flux_ms+update_ms)/N*1000.0);

        CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
        CUDA_CHECK(cudaEventDestroy(e2)); CUDA_CHECK(cudaEventDestroy(e3));
    }

    printf("\n=== Summary ===\n");
    printf("Compare [A] vs [B]: does arg count affect CUDA Graph replay?\n");
    printf("Compare [C] vs [D]: does arg count affect async launch?\n");
    printf("Compare [B] vs [E]: Graph vs Persistent for struct-ptr kernels.\n");

    cudaFree(d_H); cudaFree(d_U); cudaFree(d_V); cudaFree(d_Z); cudaFree(d_W);
    cudaFree(d_ZBC); cudaFree(d_ZB1); cudaFree(d_AREA); cudaFree(d_FNC);
    cudaFree(d_NAC); cudaFree(d_KLAS); cudaFree(d_SIDE);
    cudaFree(d_COSF); cudaFree(d_SINF); cudaFree(d_SLCOS); cudaFree(d_SLSIN);
    cudaFree(d_FLUX0); cudaFree(d_FLUX1); cudaFree(d_FLUX2); cudaFree(d_FLUX3);
    cudaFree(d_meshptr);
    return 0;
}
