// Full Configuration Matrix: strategy × block_size × kernel × grid_size
// For PK against PERKS (ICS'23) and Kernel Batching (arXiv'25)
//
// Output: CSV for cost model training/validation
//
// Build:
//   nvcc -O3 -arch=sm_86 -rdc=true pk_matrix_benchmark.cu -o pk_matrix -lcudadevrt
//   (sm_90 for B200, sm_86 for 3060)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CHECK(call) do { auto e = (call); if(e) { printf("CUDA error %d at line %d\n", e, __LINE__); exit(1); }} while(0)

// ============================================================================
// Stencil Kernels (1D thread indexing for flexible block sizes)
// ============================================================================

// Heat2D
__global__ void k_heat2d(int N, int N2, const float* __restrict__ u, float* __restrict__ v) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N2) {
        int i = tid / N, j = tid % N;
        if (i >= 1 && i < N-1 && j >= 1 && j < N-1)
            v[tid] = u[tid] + 0.2f * (u[tid-N] + u[tid+N] + u[tid-1] + u[tid+1] - 4.0f * u[tid]);
    }
}
__global__ void k_heat2d_pers(int N, int N2, float* u, float* v, int STEPS) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / N, j = tid % N;
    for (int s = 0; s < STEPS; s++) {
        if (tid < N2 && i >= 1 && i < N-1 && j >= 1 && j < N-1)
            v[tid] = u[tid] + 0.2f * (u[tid-N] + u[tid+N] + u[tid-1] + u[tid+1] - 4.0f * u[tid]);
        cg::this_grid().sync();
        if (tid < N2 && i >= 1 && i < N-1 && j >= 1 && j < N-1) u[tid] = v[tid];
        cg::this_grid().sync();
    }
}

// Jacobi2D
__global__ void k_jacobi2d(int N, int N2, const float* __restrict__ u, float* __restrict__ v) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N2) {
        int i = tid / N, j = tid % N;
        if (i >= 1 && i < N-1 && j >= 1 && j < N-1)
            v[tid] = 0.25f * (u[tid-N] + u[tid+N] + u[tid-1] + u[tid+1]);
    }
}
__global__ void k_jacobi2d_pers(int N, int N2, float* u, float* v, int STEPS) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / N, j = tid % N;
    for (int s = 0; s < STEPS; s++) {
        if (tid < N2 && i >= 1 && i < N-1 && j >= 1 && j < N-1)
            v[tid] = 0.25f * (u[tid-N] + u[tid+N] + u[tid-1] + u[tid+1]);
        cg::this_grid().sync();
        if (tid < N2 && i >= 1 && i < N-1 && j >= 1 && j < N-1) u[tid] = v[tid];
        cg::this_grid().sync();
    }
}

// HotSpot
__global__ void k_hotspot(int N, int N2, const float* __restrict__ temp,
                          const float* __restrict__ power, float* __restrict__ result,
                          float dt_cap, float ce, float cw, float cn, float cs, float cc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N2) {
        int i = tid / N, j = tid % N;
        if (i >= 1 && i < N-1 && j >= 1 && j < N-1)
            result[tid] = cc * temp[tid] + cn * temp[tid-N] + cs * temp[tid+N]
                        + cw * temp[tid-1] + ce * temp[tid+1] + dt_cap * power[tid];
    }
}
__global__ void k_hotspot_pers(int N, int N2, float* temp, float* result,
                               const float* __restrict__ power,
                               float dt_cap, float ce, float cw, float cn, float cs, float cc,
                               int STEPS) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / N, j = tid % N;
    for (int s = 0; s < STEPS; s++) {
        if (tid < N2 && i >= 1 && i < N-1 && j >= 1 && j < N-1)
            result[tid] = cc * temp[tid] + cn * temp[tid-N] + cs * temp[tid+N]
                        + cw * temp[tid-1] + ce * temp[tid+1] + dt_cap * power[tid];
        cg::this_grid().sync();
        if (tid < N2 && i >= 1 && i < N-1 && j >= 1 && j < N-1) temp[tid] = result[tid];
        cg::this_grid().sync();
    }
}

// SRAD
#define SRAD_LAMBDA 0.5f
__global__ void k_srad(int N, int N2, const float* __restrict__ img, float* __restrict__ out, float q0sq) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N2) {
        int i = tid / N, j = tid % N;
        if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
            float Jc = img[tid];
            float dN = img[tid-N]-Jc, dS = img[tid+N]-Jc, dW = img[tid-1]-Jc, dE = img[tid+1]-Jc;
            float G2 = (dN*dN+dS*dS+dW*dW+dE*dE)/(Jc*Jc+1e-10f);
            float L = (dN+dS+dW+dE)/(Jc+1e-10f);
            float num = 0.5f*G2 - L*L/16.0f;
            float den = (1.0f+0.25f*L); den *= den;
            float qsq = num/(den+1e-10f);
            float c = 1.0f/(1.0f+(qsq-q0sq)/(q0sq*(1.0f+q0sq)+1e-10f));
            c = fmaxf(0.0f, fminf(1.0f, c));
            out[tid] = Jc + SRAD_LAMBDA*(c*dN+c*dS+c*dW+c*dE);
        }
    }
}
__global__ void k_srad_pers(int N, int N2, float* img, float* out, float q0sq, int STEPS) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / N, j = tid % N;
    for (int s = 0; s < STEPS; s++) {
        if (tid < N2 && i >= 1 && i < N-1 && j >= 1 && j < N-1) {
            float Jc = img[tid];
            float dN = img[tid-N]-Jc, dS = img[tid+N]-Jc, dW = img[tid-1]-Jc, dE = img[tid+1]-Jc;
            float G2 = (dN*dN+dS*dS+dW*dW+dE*dE)/(Jc*Jc+1e-10f);
            float L = (dN+dS+dW+dE)/(Jc+1e-10f);
            float num = 0.5f*G2 - L*L/16.0f;
            float den = (1.0f+0.25f*L); den *= den;
            float qsq = num/(den+1e-10f);
            float c = 1.0f/(1.0f+(qsq-q0sq)/(q0sq*(1.0f+q0sq)+1e-10f));
            c = fmaxf(0.0f, fminf(1.0f, c));
            out[tid] = Jc + SRAD_LAMBDA*(c*dN+c*dS+c*dW+c*dE);
        }
        cg::this_grid().sync();
        if (tid < N2 && i >= 1 && i < N-1 && j >= 1 && j < N-1) img[tid] = out[tid];
        cg::this_grid().sync();
    }
}

// Copy
__global__ void k_copy(int n, const float* __restrict__ s, float* __restrict__ d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = s[i];
}

// ============================================================================
// Benchmark one (kernel, N, B, strategy)
// ============================================================================
struct Config {
    const char* kernel;
    int N, B, steps;
    int kernel_type; // 0=heat, 1=jacobi, 2=hotspot, 3=srad
};

// HotSpot globals
static float g_dt_cap, g_ce, g_cw, g_cn, g_cs, g_cc;
static float* g_power = nullptr;

void init_hs(int N) {
    float dx = 0.001f, dt = 0.002f, cap = 3.2e6f, k_si = 100.0f;
    g_dt_cap = dt / (cap * dx * dx);
    float r = dt / (cap * dx * dx * (dx / (2.0f * k_si * dt * dx)));
    g_cc = 1.0f - 4.0f * r; g_cn = g_cs = g_ce = g_cw = r;
    int N2 = N*N;
    if (g_power) cudaFree(g_power);
    CHECK(cudaMalloc(&g_power, N2*4));
    float* h = (float*)malloc(N2*4);
    for (int i = 0; i < N2; i++) h[i] = 0.5f + 0.1f * sinf(i*0.01f);
    CHECK(cudaMemcpy(g_power, h, N2*4, cudaMemcpyHostToDevice));
    free(h);
}

float run_one(Config& c, const char* strategy) {
    int N = c.N, N2 = N*N, B = c.B, STEPS = c.steps;
    int blocks = (N2 + B - 1) / B;
    int copy_blocks = (N2 + 255) / 256;

    float *u, *v;
    CHECK(cudaMalloc(&u, N2*4)); CHECK(cudaMalloc(&v, N2*4));
    float* h = (float*)malloc(N2*4);
    for (int i = 0; i < N2; i++) h[i] = 1.0f + 0.001f*(i%1000);
    CHECK(cudaMemcpy(u, h, N2*4, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(v, 0, N2*4));
    free(h);

    if (c.kernel_type == 2) init_hs(N);

    // Select kernel function pointers
    auto launch_step = [&](cudaStream_t stream) {
        switch (c.kernel_type) {
            case 0: k_heat2d<<<blocks, B, 0, stream>>>(N, N2, u, v); break;
            case 1: k_jacobi2d<<<blocks, B, 0, stream>>>(N, N2, u, v); break;
            case 2: k_hotspot<<<blocks, B, 0, stream>>>(N, N2, u, g_power, v, g_dt_cap, g_ce, g_cw, g_cn, g_cs, g_cc); break;
            case 3: k_srad<<<blocks, B, 0, stream>>>(N, N2, u, v, 0.5f); break;
        }
    };

    // Warmup
    for (int i = 0; i < 20; i++) { launch_step(0); k_copy<<<copy_blocks,256>>>(N2,v,u); }
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    float ms, result = -1.0f;

    if (strcmp(strategy, "sync") == 0) {
        CHECK(cudaEventRecord(t0));
        for (int s = 0; s < STEPS; s++) { launch_step(0); k_copy<<<copy_blocks,256>>>(N2,v,u); CHECK(cudaDeviceSynchronize()); }
        CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1));
        CHECK(cudaEventElapsedTime(&ms, t0, t1));
        result = ms * 1000.0f / STEPS;
    }
    else if (strcmp(strategy, "async") == 0) {
        CHECK(cudaEventRecord(t0));
        for (int s = 0; s < STEPS; s++) { launch_step(0); k_copy<<<copy_blocks,256>>>(N2,v,u); }
        CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1));
        CHECK(cudaEventElapsedTime(&ms, t0, t1));
        result = ms * 1000.0f / STEPS;
    }
    else if (strcmp(strategy, "graph") == 0) {
        cudaGraph_t g; cudaGraphExec_t ge;
        cudaStream_t stream; CHECK(cudaStreamCreate(&stream));
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int s = 0; s < STEPS; s++) { launch_step(stream); k_copy<<<copy_blocks,256,0,stream>>>(N2,v,u); }
        CHECK(cudaStreamEndCapture(stream, &g));
        CHECK(cudaGraphInstantiate(&ge, g, NULL, NULL, 0));
        for (int i = 0; i < 3; i++) { cudaGraphLaunch(ge,stream); cudaStreamSynchronize(stream); }
        int REPS = 5;
        CHECK(cudaEventRecord(t0, stream));
        for (int i = 0; i < REPS; i++) cudaGraphLaunch(ge, stream);
        CHECK(cudaEventRecord(t1, stream)); CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaEventElapsedTime(&ms, t0, t1));
        result = ms * 1000.0f / (STEPS * REPS);
        cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaStreamDestroy(stream);
    }
    else if (strcmp(strategy, "persistent") == 0) {
        const void* fn = nullptr;
        switch (c.kernel_type) {
            case 0: fn = (const void*)k_heat2d_pers; break;
            case 1: fn = (const void*)k_jacobi2d_pers; break;
            case 2: fn = (const void*)k_hotspot_pers; break;
            case 3: fn = (const void*)k_srad_pers; break;
        }
        int numBSm = 0; cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm, fn, B, 0));
        int maxB = numBSm * prop.multiProcessorCount;
        if (blocks <= maxB) {
            void* args_w[7], *args_b[7];
            float q0sq = 0.5f;
            switch (c.kernel_type) {
                case 0: { int w=5; args_w[0]=&N; args_w[1]=&N2; args_w[2]=&u; args_w[3]=&v; args_w[4]=&w;
                          args_b[0]=&N; args_b[1]=&N2; args_b[2]=&u; args_b[3]=&v; args_b[4]=&STEPS; break; }
                case 1: { int w=5; args_w[0]=&N; args_w[1]=&N2; args_w[2]=&u; args_w[3]=&v; args_w[4]=&w;
                          args_b[0]=&N; args_b[1]=&N2; args_b[2]=&u; args_b[3]=&v; args_b[4]=&STEPS; break; }
                case 2: { int w=5; args_w[0]=&N; args_w[1]=&N2; args_w[2]=&u; args_w[3]=&v; args_w[4]=&g_power;
                          args_w[5]=&g_dt_cap; args_w[6]=&g_ce; // need more...
                          // Use a flat approach
                          void* a[] = {&N,&N2,&u,&v,&g_power,&g_dt_cap,&g_ce,&g_cw,&g_cn,&g_cs,&g_cc,&w};
                          cudaLaunchCooperativeKernel(fn, blocks, B, a); CHECK(cudaDeviceSynchronize());
                          void* ab[] = {&N,&N2,&u,&v,&g_power,&g_dt_cap,&g_ce,&g_cw,&g_cn,&g_cs,&g_cc,&STEPS};
                          int REPS=5; CHECK(cudaEventRecord(t0));
                          for (int i=0;i<REPS;i++) cudaLaunchCooperativeKernel(fn, blocks, B, ab);
                          CHECK(cudaEventRecord(t1)); CHECK(cudaDeviceSynchronize());
                          CHECK(cudaEventElapsedTime(&ms,t0,t1));
                          result = ms*1000.0f/(STEPS*REPS);
                          goto done; }
                case 3: { int w=5; args_w[0]=&N; args_w[1]=&N2; args_w[2]=&u; args_w[3]=&v; args_w[4]=&q0sq; args_w[5]=&w;
                          args_b[0]=&N; args_b[1]=&N2; args_b[2]=&u; args_b[3]=&v; args_b[4]=&q0sq; args_b[5]=&STEPS; break; }
            }
            if (c.kernel_type != 2) {
                int nargs = (c.kernel_type == 3) ? 6 : 5;
                cudaLaunchCooperativeKernel(fn, blocks, B, args_w); CHECK(cudaDeviceSynchronize());
                int REPS = 5; CHECK(cudaEventRecord(t0));
                for (int i = 0; i < REPS; i++) cudaLaunchCooperativeKernel(fn, blocks, B, args_b);
                CHECK(cudaEventRecord(t1)); CHECK(cudaDeviceSynchronize());
                CHECK(cudaEventElapsedTime(&ms, t0, t1));
                result = ms * 1000.0f / (STEPS * REPS);
            }
        }
        // else: result stays -1 (illegal)
    }
done:
    CHECK(cudaEventDestroy(t0)); CHECK(cudaEventDestroy(t1));
    CHECK(cudaFree(u)); CHECK(cudaFree(v));
    return result;
}

// ============================================================================
// Main: full matrix
// ============================================================================
int main() {
    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("# GPU: %s (SMs=%d)\n", prop.name, prop.multiProcessorCount);
    printf("# Full configuration matrix: kernel × N × B × strategy\n");
    printf("kernel,N,cells,B,strategy,us_per_step,speedup_vs_sync\n");

    const char* kernel_filter = getenv("PK_KERNEL");
    const char* strategy_filter = getenv("PK_STRATEGY");
    int n_filter = getenv("PK_N") ? atoi(getenv("PK_N")) : 0;
    int b_filter = getenv("PK_B") ? atoi(getenv("PK_B")) : 0;

    const char* kernels[] = {"heat2d", "jacobi2d", "hotspot", "srad"};
    int Ns[] = {64, 128, 256, 512};
    int Bs[] = {32, 64, 128, 256, 512};
    const char* strategies[] = {"sync", "async", "graph", "persistent"};
    int steps_for_N[] = {2000, 2000, 1000, 500};

    for (int ki = 0; ki < 4; ki++) {
        if (kernel_filter && strcmp(kernel_filter, kernels[ki]) != 0) continue;
        for (int ni = 0; ni < 4; ni++) {
            int N = Ns[ni];
            if (n_filter && n_filter != N) continue;
            // First get sync baseline at B=256
            Config c_base = {kernels[ki], N, 256, steps_for_N[ni], ki};
            float sync_base = run_one(c_base, "sync");

            for (int bi = 0; bi < 5; bi++) {
                int B = Bs[bi];
                if (b_filter && b_filter != B) continue;
                for (int si = 0; si < 4; si++) {
                    if (strategy_filter && strcmp(strategy_filter, strategies[si]) != 0) continue;
                    Config c = {kernels[ki], N, B, steps_for_N[ni], ki};
                    float t = run_one(c, strategies[si]);
                    float spdup = (t > 0 && sync_base > 0) ? sync_base / t : -1.0f;
                    if (t > 0)
                        printf("%s,%d,%d,%d,%s,%.2f,%.2f\n", kernels[ki], N, N*N, B, strategies[si], t, spdup);
                    else
                        printf("%s,%d,%d,%d,%s,N/A,N/A\n", kernels[ki], N, N*N, B, strategies[si]);
                    fflush(stdout);
                }
            }
        }
        fprintf(stderr, "  [%s done]\n", kernels[ki]);
    }
    return 0;
}
