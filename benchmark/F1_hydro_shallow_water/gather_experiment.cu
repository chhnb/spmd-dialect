/**
 * Experiment: Naive Gather vs Cooperative Gather Promotion
 *
 * Compares two implementations of the hydro-cal 2D shallow water kernel:
 *   A) Naive:     every neighbor read goes to global memory (current approach)
 *   B) Promoted:  cooperative load neighbors into shared memory, then compute
 *
 * Dam-break on NxN structured quad mesh.
 *
 * Compile:
 *   nvcc -O3 -arch=sm_80 gather_experiment.cu -o gather_experiment
 *
 * Run:
 *   ./gather_experiment 128 100   # N=128, 100 steps
 *   ./gather_experiment 256 100
 *   ./gather_experiment 512 50
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
#define G       9.81
#define HALF_G  4.905
#define HM1     0.001
#define HM2     0.01
#define VMIN_C  0.001
#define C1_C    0.3
#define MANNING_N 0.03

#define BLOCK_SIZE 256

// ---------------------------------------------------------------------------
// QF + Osher (shared by both kernels, device functions)
// ---------------------------------------------------------------------------
__device__ __forceinline__
void QF_d(double h, double u, double v, double &F0, double &F1, double &F2, double &F3) {
    F0 = h * u; F1 = F0 * u; F2 = F0 * v; F3 = HALF_G * h * h;
}

__device__ __forceinline__
void osher_d(double QL_h, double QL_u, double QL_v,
             double QR_h, double QR_u, double QR_v,
             double FIL_in, double H_pos,
             double &R0, double &R1, double &R2, double &R3)
{
    double CR = sqrt(G * QR_h);
    double FIR_v = QR_u - 2.0 * CR;
    double fil = FIL_in, fir = FIR_v;
    double UA = (fil + fir) / 2.0;
    double CA = fabs((fil - fir) / 4.0);
    double CL_v = sqrt(G * H_pos);
    R0 = R1 = R2 = R3 = 0.0;

    int K2 = (CA < UA) ? 1 : (UA >= 0 && UA < CA) ? 2 : (UA >= -CA && UA < 0) ? 3 : 4;
    int K1 = (QL_u < CL_v && QR_u >= -CR) ? 1 :
             (QL_u >= CL_v && QR_u >= -CR) ? 2 :
             (QL_u < CL_v && QR_u < -CR)   ? 3 : 4;

    auto add = [&](double h, double u, double v, double s) {
        double f0, f1, f2, f3; QF_d(h, u, v, f0, f1, f2, f3);
        R0 += f0*s; R1 += f1*s; R2 += f2*s; R3 += f3*s;
    };
    auto qs1 = [&](double s) { add(QL_h, QL_u, QL_v, s); };
    auto qs2 = [&](double s) { double U=fil/3, H=U*U/G; add(H, U, QL_v, s); };
    auto qs3 = [&](double s) { double ua=(fil+fir)/2; fil-=ua; double H=fil*fil/(4*G); add(H, ua, QL_v, s); };
    auto qs5 = [&](double s) { double ua=(fil+fir)/2; fir-=ua; double H=fir*fir/(4*G); add(H, ua, QR_v, s); };
    auto qs6 = [&](double s) { double U=fir/3, H=U*U/G; add(H, U, QR_v, s); };
    auto qs7 = [&](double s) { add(QR_h, QR_u, QR_v, s); };

    switch(K1) {
    case 1: switch(K2) { case 1: qs2(1); break; case 2: qs3(1); break; case 3: qs5(1); break; case 4: qs6(1); break; } break;
    case 2: switch(K2) { case 1: qs1(1); break; case 2: qs1(1);qs2(-1);qs3(1); break; case 3: qs1(1);qs2(-1);qs5(1); break; case 4: qs1(1);qs2(-1);qs6(1); break; } break;
    case 3: switch(K2) { case 1: qs2(1);qs6(-1);qs7(1); break; case 2: qs3(1);qs6(-1);qs7(1); break; case 3: qs5(1);qs6(-1);qs7(1); break; case 4: qs7(1); break; } break;
    case 4: switch(K2) { case 1: qs1(1);qs6(-1);qs7(1); break; case 2: qs1(1);qs2(-1);qs3(1);qs6(-1);qs7(1); break; case 3: qs1(1);qs2(-1);qs5(1);qs6(-1);qs7(1); break; case 4: qs1(1);qs2(-1);qs7(1); break; } break;
    }
}

// ---------------------------------------------------------------------------
// Per-cell flux + update (shared device function, parameterized by load source)
// ---------------------------------------------------------------------------
__device__ __forceinline__
void compute_cell(int pos, int CEL, double DT,
                  const int* __restrict__ NAC, const int* __restrict__ KLAS,
                  const double* __restrict__ SIDE, const double* __restrict__ COSF,
                  const double* __restrict__ SINF, const double* __restrict__ SLCOS,
                  const double* __restrict__ SLSIN, const double* __restrict__ AREA,
                  const double* __restrict__ ZBC, const double* __restrict__ FNC,
                  // Source for H/U/V/Z reads (global or shared)
                  const double* __restrict__ H_src, const double* __restrict__ U_src,
                  const double* __restrict__ V_src, const double* __restrict__ Z_src,
                  int src_offset,  // subtract from global index to get src index
                  // Always global (for writing and boundary ZBC lookups)
                  const double* __restrict__ H_pre_g, const double* __restrict__ Z_pre_g,
                  double* __restrict__ H_res, double* __restrict__ U_res,
                  double* __restrict__ V_res, double* __restrict__ Z_res,
                  double* __restrict__ W_res, int stride)
{
    double H1 = H_src[pos - src_offset], U1 = U_src[pos - src_offset];
    double V1 = V_src[pos - src_offset];
    double BI = ZBC[pos];
    double HI = fmax(H1, HM1);
    double UI = U1, VI = V1;
    if (HI <= HM2) { UI = copysign(VMIN_C, UI); VI = copysign(VMIN_C, VI); }
    double ZI = fmax(Z_src[pos - src_offset], BI);
    double WH = 0, WU = 0, WV = 0;

    for (int j = 1; j <= 4; ++j) {
        int joff = j * stride;
        int NC = NAC[joff + pos];
        int KP = KLAS[joff + pos];
        double COSJ = COSF[joff + pos], SINJ = SINF[joff + pos];
        double SL = SIDE[joff + pos];
        double SLCA = SLCOS[joff + pos], SLSA = SLSIN[joff + pos];

        double QL_h = HI, QL_u = UI*COSJ + VI*SINJ, QL_v = VI*COSJ - UI*SINJ;
        double CL_v = sqrt(G * HI);
        double FIL_v = QL_u + 2.0 * CL_v;

        double HC=0, BC=0, ZC=0, UC=0, VC=0;
        if (NC != 0) {
            HC = fmax(H_src[NC - src_offset], HM1);
            BC = ZBC[NC];
            ZC = fmax(ZBC[NC], Z_src[NC - src_offset]);
            UC = U_src[NC - src_offset];
            VC = V_src[NC - src_offset];
        }

        double f0=0, f1=0, f2=0, f3=0;

        if (KP == 4 || KP != 0) {
            f3 = HALF_G * H1 * H1;
        } else if (HI <= HM1 && HC <= HM1) {
        } else if (ZI <= BC) {
            f0 = -C1_C*pow(HC,1.5); f1 = HI*QL_u*fabs(QL_u); f3 = HALF_G*HI*HI;
        } else if (ZC <= BI) {
            f0 = C1_C*pow(HI,1.5); f1 = HI*fabs(QL_u)*QL_u; f2 = HI*fabs(QL_u)*QL_v;
        } else if (HI <= HM2) {
            if (ZC > ZI) {
                double DH = fmax(ZC-ZBC[pos], HM1), UN = -C1_C*sqrt(DH);
                f0=DH*UN; f1=f0*UN; f2=f0*(VC*COSJ-UC*SINJ); f3=HALF_G*HI*HI;
            } else { f0 = C1_C*pow(HI,1.5); f3 = HALF_G*HI*HI; }
        } else if (HC <= HM2) {
            if (ZI > ZC) {
                double DH = fmax(ZI-BC, HM1), UN = C1_C*sqrt(DH), HC1=ZC-ZBC[pos];
                f0=DH*UN; f1=f0*UN; f2=f0*QL_v; f3=HALF_G*HC1*HC1;
            } else { f0 = -C1_C*pow(HC,1.5); f1 = HI*QL_u*QL_u; f3 = HALF_G*HI*HI; }
        } else {
            if (pos < NC) {
                double QR_h = fmax(ZC-ZBC[pos], HM1);
                double UR = UC*COSJ+VC*SINJ;
                double ratio = fmin(HC/QR_h, 1.5);
                double QR_u = UR*ratio;
                if (HC<=HM2||QR_h<=HM2) QR_u=copysign(VMIN_C,UR);
                double QR_v = VC*COSJ-UC*SINJ;
                double os0,os1,os2,os3;
                osher_d(QL_h,QL_u,QL_v, QR_h,QR_u,QR_v, FIL_v, H_src[pos-src_offset], os0,os1,os2,os3);
                f0=os0; f1=os1+(1-ratio)*HC*UR*UR/2; f2=os2; f3=os3;
            } else {
                double C1=-COSJ, S1=-SINJ;
                double L1h=H_src[NC-src_offset], L1u=U_src[NC-src_offset]*C1+V_src[NC-src_offset]*S1;
                double L1v=V_src[NC-src_offset]*C1-U_src[NC-src_offset]*S1;
                double CL1=sqrt(G*L1h), FIL1=L1u+2*CL1;
                double HC2=fmax(HI,HM1), ZC1=fmax(ZBC[pos],ZI);
                double R1h=fmax(ZC1-ZBC[NC],HM1), UR1=UI*C1+VI*S1;
                double ratio1=fmin(HC2/R1h,1.5), R1u=UR1*ratio1;
                if(HC2<=HM2||R1h<=HM2) R1u=copysign(VMIN_C,UR1);
                double R1v=VI*C1-UI*S1;
                double mr0,mr1,mr2,mr3;
                osher_d(L1h,L1u,L1v, R1h,R1u,R1v, FIL1, H_src[NC-src_offset], mr0,mr1,mr2,mr3);
                f0=-mr0; f1=mr1+(1-ratio1)*HC2*UR1*UR1/2; f2=mr2;
                double ZA=sqrt(mr3/HALF_G)+BC, HC3=fmax(ZA-ZBC[pos],0.0);
                f3=HALF_G*HC3*HC3;
            }
        }

        double FLR1=f1+f3, FLR2=f2;
        WH+=SL*f0; WU+=SLCA*FLR1-SLSA*FLR2; WV+=SLSA*FLR1+SLCA*FLR2;
    }

    double DTA = DT / AREA[pos];
    double H2 = fmax(H1 - DTA*WH, HM1);
    double Z2 = H2 + BI, U2=0, V2=0;
    if (H2 > HM1) {
        if (H2 <= HM2) {
            U2=copysign(fmin(VMIN_C,fabs(U1)),U1); V2=copysign(fmin(VMIN_C,fabs(V1)),V1);
        } else {
            double WSF=G*MANNING_N*MANNING_N*sqrt(U1*U1+V1*V1)/pow(H1,0.33333);
            U2=(H1*U1-DTA*WU-DT*WSF*U1)/H2; V2=(H1*V1-DTA*WV-DT*WSF*V1)/H2;
            U2=copysign(fmin(fabs(U2),15.0),U2); V2=copysign(fmin(fabs(V2),15.0),V2);
        }
    }
    H_res[pos]=H2; U_res[pos]=U2; V_res[pos]=V2; Z_res[pos]=Z2;
    W_res[pos]=sqrt(U2*U2+V2*V2);
}


// =========================================================================
// Kernel A: NAIVE — all reads from global memory
// =========================================================================
__global__
void swe_naive(int CEL, double DT, int stride,
               const int* NAC, const int* KLAS,
               const double* SIDE, const double* COSF, const double* SINF,
               const double* SLCOS, const double* SLSIN,
               const double* AREA, const double* ZBC, const double* FNC,
               const double* H_pre, const double* U_pre,
               const double* V_pre, const double* Z_pre,
               double* H_res, double* U_res, double* V_res,
               double* Z_res, double* W_res)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (pos > CEL) return;

    compute_cell(pos, CEL, DT, NAC, KLAS, SIDE, COSF, SINF, SLCOS, SLSIN,
                 AREA, ZBC, FNC,
                 H_pre, U_pre, V_pre, Z_pre,  // source = global
                 0,                             // no offset
                 H_pre, Z_pre,                  // global refs
                 H_res, U_res, V_res, Z_res, W_res, stride);
}


// =========================================================================
// Kernel B: PROMOTED — cooperative gather into shared memory
// =========================================================================
__global__
void swe_promoted(int CEL, int N, double DT, int stride,
                  const int* NAC, const int* KLAS,
                  const double* SIDE, const double* COSF, const double* SINF,
                  const double* SLCOS, const double* SLSIN,
                  const double* AREA, const double* ZBC, const double* FNC,
                  const double* H_pre, const double* U_pre,
                  const double* V_pre, const double* Z_pre,
                  double* H_res, double* U_res, double* V_res,
                  double* Z_res, double* W_res)
{
    // This block processes cells [block_base+1 .. block_base+BLOCK_SIZE]
    int block_base = blockIdx.x * blockDim.x;  // 0-indexed base
    int tid = threadIdx.x;
    int pos = block_base + tid + 1;             // 1-indexed cell

    // Halo region: neighbors can be at pos±1 (east/west) or pos±N (north/south)
    // Extended region: [block_base + 1 - N, block_base + BLOCK_SIZE + N]
    int ext_lo = max(block_base + 1 - N, 1);
    int ext_hi = min(block_base + (int)blockDim.x + N, CEL);
    int ext_size = ext_hi - ext_lo + 1;

    // Shared memory for H, U, V, Z (4 fields)
    extern __shared__ double smem[];
    double* sh_H = smem;
    double* sh_U = smem + ext_size;
    double* sh_V = smem + ext_size * 2;
    double* sh_Z = smem + ext_size * 3;

    // Phase 1: Cooperative load — each thread loads multiple elements
    for (int i = tid; i < ext_size; i += blockDim.x) {
        int g = ext_lo + i;  // global index
        if (g >= 1 && g <= CEL) {
            sh_H[i] = H_pre[g];
            sh_U[i] = U_pre[g];
            sh_V[i] = V_pre[g];
            sh_Z[i] = Z_pre[g];
        } else {
            sh_H[i] = HM1;
            sh_U[i] = 0;
            sh_V[i] = 0;
            sh_Z[i] = 0;
        }
    }
    __syncthreads();

    // Phase 2: Compute using shared memory
    if (pos > CEL) return;

    // Check all neighbors are within the extended region
    // For structured mesh this is guaranteed; for unstructured, fall back to global
    bool all_in_range = true;
    for (int j = 1; j <= 4; ++j) {
        int nc = NAC[j * stride + pos];
        if (nc != 0 && (nc < ext_lo || nc > ext_hi)) {
            all_in_range = false;
            break;
        }
    }

    if (all_in_range) {
        // Use shared memory (fast path)
        compute_cell(pos, CEL, DT, NAC, KLAS, SIDE, COSF, SINF, SLCOS, SLSIN,
                     AREA, ZBC, FNC,
                     sh_H, sh_U, sh_V, sh_Z,  // source = shared
                     ext_lo,                     // offset: shared[g - ext_lo]
                     H_pre, Z_pre,
                     H_res, U_res, V_res, Z_res, W_res, stride);
    } else {
        // Fallback to global (safety net for unstructured)
        compute_cell(pos, CEL, DT, NAC, KLAS, SIDE, COSF, SINF, SLCOS, SLSIN,
                     AREA, ZBC, FNC,
                     H_pre, U_pre, V_pre, Z_pre,
                     0,
                     H_pre, Z_pre,
                     H_res, U_res, V_res, Z_res, W_res, stride);
    }
}


// =========================================================================
// Transfer kernel
// =========================================================================
__global__
void transfer(int CEL, double* H_pre, double* U_pre, double* V_pre,
              double* Z_pre, double* W_pre,
              const double* H_res, const double* U_res, const double* V_res,
              const double* Z_res, const double* W_res)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (pos > CEL) return;
    H_pre[pos]=H_res[pos]; U_pre[pos]=U_res[pos]; V_pre[pos]=V_res[pos];
    Z_pre[pos]=Z_res[pos]; W_pre[pos]=W_res[pos];
}


// =========================================================================
// Host: build mesh, run both kernels, compare
// =========================================================================
void init_mesh(int N, int CEL, int stride,
               int* h_NAC, int* h_KLAS, double* h_SIDE, double* h_COSF,
               double* h_SINF, double* h_SLCOS, double* h_SLSIN,
               double* h_AREA, double* h_ZBC, double* h_FNC,
               double* h_H, double* h_U, double* h_V, double* h_Z, double* h_W)
{
    double dx = 1.0;
    double edge_cos[] = {0, 0, 1, 0, -1};
    double edge_sin[] = {0, -1, 0, 1, 0};

    memset(h_NAC, 0, 5 * stride * sizeof(int));
    memset(h_KLAS, 0, 5 * stride * sizeof(int));
    memset(h_SIDE, 0, 5 * stride * sizeof(double));
    memset(h_COSF, 0, 5 * stride * sizeof(double));
    memset(h_SINF, 0, 5 * stride * sizeof(double));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int pos = i * N + j + 1;
            h_AREA[pos] = dx * dx;
            h_FNC[pos] = G * MANNING_N * MANNING_N;
            h_ZBC[pos] = 0;
            for (int e = 1; e <= 4; e++) {
                h_SIDE[e*stride+pos] = dx;
                h_COSF[e*stride+pos] = edge_cos[e];
                h_SINF[e*stride+pos] = edge_sin[e];
                h_SLCOS[e*stride+pos] = dx * edge_cos[e];
                h_SLSIN[e*stride+pos] = dx * edge_sin[e];
            }
            if (i > 0)   h_NAC[1*stride+pos] = (i-1)*N+j+1; else h_KLAS[1*stride+pos] = 4;
            if (j < N-1)  h_NAC[2*stride+pos] = i*N+(j+1)+1; else h_KLAS[2*stride+pos] = 4;
            if (i < N-1)  h_NAC[3*stride+pos] = (i+1)*N+j+1; else h_KLAS[3*stride+pos] = 4;
            if (j > 0)   h_NAC[4*stride+pos] = i*N+(j-1)+1; else h_KLAS[4*stride+pos] = 4;

            h_H[pos] = (j < N/2) ? 2.0 : 0.5;
            h_U[pos] = 0; h_V[pos] = 0;
            h_Z[pos] = h_H[pos]; h_W[pos] = 0;
        }
    }
}

double run_benchmark(bool use_promoted, int N, int CEL, int stride, double DT,
                     int steps, int warmup, int repeat,
                     int* d_NAC, int* d_KLAS,
                     double* d_SIDE, double* d_COSF, double* d_SINF,
                     double* d_SLCOS, double* d_SLSIN,
                     double* d_AREA, double* d_ZBC, double* d_FNC,
                     double* d_H_pre, double* d_U_pre, double* d_V_pre,
                     double* d_Z_pre, double* d_W_pre,
                     double* d_H_res, double* d_U_res, double* d_V_res,
                     double* d_Z_res, double* d_W_res,
                     // Host init data for reset
                     double* h_H, double* h_U, double* h_V, double* h_Z, double* h_W)
{
    int grid = (CEL + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Shared memory size for promoted kernel
    int ext_size_max = BLOCK_SIZE + 2 * N + 2;
    size_t smem_bytes = ext_size_max * 4 * sizeof(double);

    std::vector<double> times;

    for (int r = 0; r < warmup + repeat; r++) {
        // Reset state
        cudaMemcpy(d_H_pre, h_H, (CEL+1)*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_U_pre, h_U, (CEL+1)*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V_pre, h_V, (CEL+1)*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Z_pre, h_Z, (CEL+1)*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_W_pre, h_W, (CEL+1)*sizeof(double), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        auto t0 = std::chrono::high_resolution_clock::now();

        for (int s = 0; s < steps; s++) {
            if (use_promoted) {
                swe_promoted<<<grid, BLOCK_SIZE, smem_bytes>>>(
                    CEL, N, DT, stride, d_NAC, d_KLAS,
                    d_SIDE, d_COSF, d_SINF, d_SLCOS, d_SLSIN,
                    d_AREA, d_ZBC, d_FNC,
                    d_H_pre, d_U_pre, d_V_pre, d_Z_pre,
                    d_H_res, d_U_res, d_V_res, d_Z_res, d_W_res);
            } else {
                swe_naive<<<grid, BLOCK_SIZE>>>(
                    CEL, DT, stride, d_NAC, d_KLAS,
                    d_SIDE, d_COSF, d_SINF, d_SLCOS, d_SLSIN,
                    d_AREA, d_ZBC, d_FNC,
                    d_H_pre, d_U_pre, d_V_pre, d_Z_pre,
                    d_H_res, d_U_res, d_V_res, d_Z_res, d_W_res);
            }
            transfer<<<grid, BLOCK_SIZE>>>(CEL,
                d_H_pre, d_U_pre, d_V_pre, d_Z_pre, d_W_pre,
                d_H_res, d_U_res, d_V_res, d_Z_res, d_W_res);
        }
        cudaDeviceSynchronize();

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (r >= warmup)
            times.push_back(ms);
    }

    std::sort(times.begin(), times.end());
    return times[times.size() / 2];  // median
}


int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 128;
    int steps = (argc > 2) ? atoi(argv[2]) : 100;
    int warmup = 3, repeat = 10;
    int CEL = N * N;
    int stride = CEL + 1;
    double dx = 1.0;
    double DT = 0.5 * dx / (sqrt(G * 2.0) + 1e-6);

    printf("=== Gather Promotion Experiment ===\n");
    printf("N=%d, CEL=%d, steps=%d, DT=%.6f\n", N, CEL, steps, DT);
    printf("BLOCK_SIZE=%d, halo=BLOCK+2N=%d\n", BLOCK_SIZE, BLOCK_SIZE + 2*N);
    printf("Shared mem per block: %.1f KB (4 fields × %d cells × 8B)\n",
           (BLOCK_SIZE + 2*N) * 4.0 * 8.0 / 1024.0, BLOCK_SIZE + 2*N);

    // Check shared memory feasibility
    size_t smem_needed = (BLOCK_SIZE + 2*N + 2) * 4 * sizeof(double);
    if (smem_needed > 48 * 1024) {
        printf("WARNING: smem needed (%.1f KB) > 48 KB. Reduce N or BLOCK_SIZE.\n",
               smem_needed / 1024.0);
    }

    // Allocate host
    int* h_NAC = new int[5 * stride];
    int* h_KLAS = new int[5 * stride];
    double* h_SIDE = new double[5 * stride];
    double* h_COSF = new double[5 * stride];
    double* h_SINF = new double[5 * stride];
    double* h_SLCOS = new double[5 * stride];
    double* h_SLSIN = new double[5 * stride];
    double* h_AREA = new double[stride];
    double* h_ZBC = new double[stride]();
    double* h_FNC = new double[stride];
    double* h_H = new double[stride];
    double* h_U = new double[stride]();
    double* h_V = new double[stride]();
    double* h_Z = new double[stride];
    double* h_W = new double[stride]();

    init_mesh(N, CEL, stride, h_NAC, h_KLAS, h_SIDE, h_COSF, h_SINF,
              h_SLCOS, h_SLSIN, h_AREA, h_ZBC, h_FNC, h_H, h_U, h_V, h_Z, h_W);

    // Allocate device
    int *d_NAC, *d_KLAS;
    double *d_SIDE, *d_COSF, *d_SINF, *d_SLCOS, *d_SLSIN;
    double *d_AREA, *d_ZBC, *d_FNC;
    double *d_H_pre, *d_U_pre, *d_V_pre, *d_Z_pre, *d_W_pre;
    double *d_H_res, *d_U_res, *d_V_res, *d_Z_res, *d_W_res;

    #define ALLOC(ptr, sz) cudaMalloc(&ptr, (sz))
    ALLOC(d_NAC, 5*stride*sizeof(int));    ALLOC(d_KLAS, 5*stride*sizeof(int));
    ALLOC(d_SIDE, 5*stride*sizeof(double)); ALLOC(d_COSF, 5*stride*sizeof(double));
    ALLOC(d_SINF, 5*stride*sizeof(double)); ALLOC(d_SLCOS, 5*stride*sizeof(double));
    ALLOC(d_SLSIN, 5*stride*sizeof(double));
    ALLOC(d_AREA, stride*sizeof(double));  ALLOC(d_ZBC, stride*sizeof(double));
    ALLOC(d_FNC, stride*sizeof(double));
    ALLOC(d_H_pre, stride*sizeof(double)); ALLOC(d_U_pre, stride*sizeof(double));
    ALLOC(d_V_pre, stride*sizeof(double)); ALLOC(d_Z_pre, stride*sizeof(double));
    ALLOC(d_W_pre, stride*sizeof(double));
    ALLOC(d_H_res, stride*sizeof(double)); ALLOC(d_U_res, stride*sizeof(double));
    ALLOC(d_V_res, stride*sizeof(double)); ALLOC(d_Z_res, stride*sizeof(double));
    ALLOC(d_W_res, stride*sizeof(double));

    cudaMemcpy(d_NAC, h_NAC, 5*stride*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_KLAS, h_KLAS, 5*stride*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_SIDE, h_SIDE, 5*stride*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_COSF, h_COSF, 5*stride*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_SINF, h_SINF, 5*stride*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_SLCOS, h_SLCOS, 5*stride*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_SLSIN, h_SLSIN, 5*stride*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_AREA, h_AREA, stride*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ZBC, h_ZBC, stride*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_FNC, h_FNC, stride*sizeof(double), cudaMemcpyHostToDevice);

    // Run naive
    double naive_ms = run_benchmark(false, N, CEL, stride, DT, steps, warmup, repeat,
        d_NAC, d_KLAS, d_SIDE, d_COSF, d_SINF, d_SLCOS, d_SLSIN,
        d_AREA, d_ZBC, d_FNC,
        d_H_pre, d_U_pre, d_V_pre, d_Z_pre, d_W_pre,
        d_H_res, d_U_res, d_V_res, d_Z_res, d_W_res,
        h_H, h_U, h_V, h_Z, h_W);

    // Run promoted
    double promo_ms = run_benchmark(true, N, CEL, stride, DT, steps, warmup, repeat,
        d_NAC, d_KLAS, d_SIDE, d_COSF, d_SINF, d_SLCOS, d_SLSIN,
        d_AREA, d_ZBC, d_FNC,
        d_H_pre, d_U_pre, d_V_pre, d_Z_pre, d_W_pre,
        d_H_res, d_U_res, d_V_res, d_Z_res, d_W_res,
        h_H, h_U, h_V, h_Z, h_W);

    // Correctness check: run 1 step of each, compare
    cudaMemcpy(d_H_pre, h_H, stride*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U_pre, h_U, stride*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_pre, h_V, stride*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Z_pre, h_Z, stride*sizeof(double), cudaMemcpyHostToDevice);
    int grid = (CEL + BLOCK_SIZE - 1) / BLOCK_SIZE;
    swe_naive<<<grid, BLOCK_SIZE>>>(CEL, DT, stride, d_NAC, d_KLAS,
        d_SIDE, d_COSF, d_SINF, d_SLCOS, d_SLSIN, d_AREA, d_ZBC, d_FNC,
        d_H_pre, d_U_pre, d_V_pre, d_Z_pre,
        d_H_res, d_U_res, d_V_res, d_Z_res, d_W_res);
    cudaDeviceSynchronize();
    double* naive_H = new double[stride];
    cudaMemcpy(naive_H, d_H_res, stride*sizeof(double), cudaMemcpyDeviceToHost);

    cudaMemcpy(d_H_pre, h_H, stride*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U_pre, h_U, stride*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_pre, h_V, stride*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Z_pre, h_Z, stride*sizeof(double), cudaMemcpyHostToDevice);
    size_t smem = (BLOCK_SIZE + 2*N + 2) * 4 * sizeof(double);
    swe_promoted<<<grid, BLOCK_SIZE, smem>>>(CEL, N, DT, stride, d_NAC, d_KLAS,
        d_SIDE, d_COSF, d_SINF, d_SLCOS, d_SLSIN, d_AREA, d_ZBC, d_FNC,
        d_H_pre, d_U_pre, d_V_pre, d_Z_pre,
        d_H_res, d_U_res, d_V_res, d_Z_res, d_W_res);
    cudaDeviceSynchronize();
    double* promo_H = new double[stride];
    cudaMemcpy(promo_H, d_H_res, stride*sizeof(double), cudaMemcpyDeviceToHost);

    double max_diff = 0;
    for (int i = 1; i <= CEL; i++)
        max_diff = fmax(max_diff, fabs(naive_H[i] - promo_H[i]));

    printf("\n=== Results ===\n");
    printf("Naive:    %.3f ms (median, %d steps)\n", naive_ms, steps);
    printf("Promoted: %.3f ms (median, %d steps)\n", promo_ms, steps);
    printf("Speedup:  %.2fx\n", naive_ms / promo_ms);
    printf("Max |H_naive - H_promoted|: %.2e (should be ~0)\n", max_diff);
    printf("\nCSV: %d,%d,%.3f,%.3f,%.2f,%.2e\n", N, steps, naive_ms, promo_ms,
           naive_ms/promo_ms, max_diff);

    // Cleanup
    delete[] h_NAC; delete[] h_KLAS; delete[] h_SIDE; delete[] h_COSF;
    delete[] h_SINF; delete[] h_SLCOS; delete[] h_SLSIN;
    delete[] h_AREA; delete[] h_ZBC; delete[] h_FNC;
    delete[] h_H; delete[] h_U; delete[] h_V; delete[] h_Z; delete[] h_W;
    delete[] naive_H; delete[] promo_H;

    return 0;
}
