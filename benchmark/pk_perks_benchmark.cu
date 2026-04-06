// PK Benchmark: 4 strategies × PERKS/Kernel-Batching kernels × multiple sizes
// Kernels: Heat2D, Jacobi2D, HotSpot, SRAD (all from PERKS/Rodinia)
// Strategies: Sync, Async, Graph, Persistent
//
// Build:
//   nvcc -O3 -arch=sm_86 -rdc=true pk_perks_benchmark.cu -o pk_perks -lcudadevrt
//   (use sm_90 for B200, sm_86 for 3060)
//
// References:
//   PERKS (ICS 2023): Jacobi2D, Heat2D, SRAD, SpMV/CG
//   Kernel Batching (arXiv 2025): HotSpot, FDTD

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CHECK(call) do { auto e = (call); if(e) { printf("CUDA error %d at %s:%d\n", e, __FILE__, __LINE__); exit(1); }} while(0)

// ============================================================================
// Kernels
// ============================================================================

// --- Heat2D (5-point stencil, fp32) ---
__global__ void heat2d_step(int N, const float* __restrict__ u, float* __restrict__ v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
        int idx = i * N + j;
        v[idx] = u[idx] + 0.2f * (u[idx-N] + u[idx+N] + u[idx-1] + u[idx+1] - 4.0f * u[idx]);
    }
}

__global__ void heat2d_persistent(int N, float* u, float* v, int STEPS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    for (int s = 0; s < STEPS; s++) {
        if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
            int idx = i * N + j;
            v[idx] = u[idx] + 0.2f * (u[idx-N] + u[idx+N] + u[idx-1] + u[idx+1] - 4.0f * u[idx]);
        }
        cg::this_grid().sync();
        if (i >= 1 && i < N-1 && j >= 1 && j < N-1) { int idx = i*N+j; u[idx] = v[idx]; }
        cg::this_grid().sync();
    }
}

// --- Jacobi2D (5-point stencil, fp32, averaging neighbors) ---
__global__ void jacobi2d_step(int N, const float* __restrict__ u, float* __restrict__ v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
        int idx = i * N + j;
        v[idx] = 0.25f * (u[idx-N] + u[idx+N] + u[idx-1] + u[idx+1]);
    }
}

__global__ void jacobi2d_persistent(int N, float* u, float* v, int STEPS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    for (int s = 0; s < STEPS; s++) {
        if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
            int idx = i * N + j;
            v[idx] = 0.25f * (u[idx-N] + u[idx+N] + u[idx-1] + u[idx+1]);
        }
        cg::this_grid().sync();
        if (i >= 1 && i < N-1 && j >= 1 && j < N-1) { int idx = i*N+j; u[idx] = v[idx]; }
        cg::this_grid().sync();
    }
}

// --- HotSpot (Rodinia, thermal simulation, 5-point stencil + self + power) ---
// Simplified: T_new = T + step * (power + neighbors - 4*T) / Cap
#define HS_STEP_DIV  0.002f
#define HS_CAP       3.2e6f
#define HS_K_SI      100.0f

__global__ void hotspot_step(int N, const float* __restrict__ temp, const float* __restrict__ power,
                             float* __restrict__ result, float dt_over_cap, float ce, float cw, float cn, float cs, float cc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
        int idx = i * N + j;
        result[idx] = cc * temp[idx] + cn * temp[idx-N] + cs * temp[idx+N]
                    + cw * temp[idx-1] + ce * temp[idx+1] + dt_over_cap * power[idx];
    }
}

__global__ void hotspot_persistent(int N, float* temp, float* result, const float* __restrict__ power,
                                   float dt_over_cap, float ce, float cw, float cn, float cs, float cc, int STEPS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    for (int s = 0; s < STEPS; s++) {
        if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
            int idx = i * N + j;
            result[idx] = cc * temp[idx] + cn * temp[idx-N] + cs * temp[idx+N]
                        + cw * temp[idx-1] + ce * temp[idx+1] + dt_over_cap * power[idx];
        }
        cg::this_grid().sync();
        if (i >= 1 && i < N-1 && j >= 1 && j < N-1) { int idx = i*N+j; temp[idx] = result[idx]; }
        cg::this_grid().sync();
    }
}

// --- SRAD (Rodinia, Speckle Reducing Anisotropic Diffusion, 5-point stencil) ---
// Simplified version: diffusion step with coefficient
#define SRAD_LAMBDA 0.5f

__global__ void srad_step(int N, const float* __restrict__ img, float* __restrict__ out, float q0sq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
        int idx = i * N + j;
        float Jc = img[idx];
        float dN = img[idx-N] - Jc;
        float dS = img[idx+N] - Jc;
        float dW = img[idx-1] - Jc;
        float dE = img[idx+1] - Jc;
        float G2 = (dN*dN + dS*dS + dW*dW + dE*dE) / (Jc * Jc + 1e-10f);
        float L = (dN + dS + dW + dE) / (Jc + 1e-10f);
        float num = (0.5f * G2) - (L * L / 16.0f);
        float den = (1.0f + 0.25f * L) * (1.0f + 0.25f * L);
        float qsq = num / (den + 1e-10f);
        float c = 1.0f / (1.0f + (qsq - q0sq) / (q0sq * (1.0f + q0sq) + 1e-10f));
        c = fmaxf(0.0f, fminf(1.0f, c));
        out[idx] = Jc + SRAD_LAMBDA * (c * dN + c * dS + c * dW + c * dE);
    }
}

__global__ void srad_persistent(int N, float* img, float* out, float q0sq, int STEPS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    for (int s = 0; s < STEPS; s++) {
        if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
            int idx = i * N + j;
            float Jc = img[idx];
            float dN = img[idx-N] - Jc;
            float dS = img[idx+N] - Jc;
            float dW = img[idx-1] - Jc;
            float dE = img[idx+1] - Jc;
            float G2 = (dN*dN + dS*dS + dW*dW + dE*dE) / (Jc * Jc + 1e-10f);
            float L = (dN + dS + dW + dE) / (Jc + 1e-10f);
            float num = (0.5f * G2) - (L * L / 16.0f);
            float den = (1.0f + 0.25f * L) * (1.0f + 0.25f * L);
            float qsq = num / (den + 1e-10f);
            float c = 1.0f / (1.0f + (qsq - q0sq) / (q0sq * (1.0f + q0sq) + 1e-10f));
            c = fmaxf(0.0f, fminf(1.0f, c));
            out[idx] = Jc + SRAD_LAMBDA * (c * dN + c * dS + c * dW + c * dE);
        }
        cg::this_grid().sync();
        if (i >= 1 && i < N-1 && j >= 1 && j < N-1) { int idx = i*N+j; img[idx] = out[idx]; }
        cg::this_grid().sync();
    }
}

// ============================================================================
// Generic copy kernel
// ============================================================================
__global__ void copy_kernel(int n, const float* __restrict__ src, float* __restrict__ dst) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// ============================================================================
// Benchmark harness (generic for any 2-array stencil)
// ============================================================================
struct Result { float sync_us, async_us, graph_us, persistent_us; };

typedef void (*StepFn)(int, const float*, float*);
typedef void (*PersistFn)(int, float*, float*, int);

// For kernels with extra args (HotSpot, SRAD), we use lambdas via wrappers below.

Result benchmark_stencil(int N, int STEPS,
                         void (*launch_step)(int N, float* u, float* v, dim3 grid, dim3 block, cudaStream_t stream),
                         void (*launch_persistent)(int N, float* u, float* v, int STEPS, dim3 grid, int maxBlocks),
                         const char* persistent_fn_name, const void* persistent_fn_ptr) {
    int N2 = N * N;
    float *u, *v;
    CHECK(cudaMalloc(&u, N2 * sizeof(float)));
    CHECK(cudaMalloc(&v, N2 * sizeof(float)));

    // Init with some values
    float* h = (float*)malloc(N2 * sizeof(float));
    for (int i = 0; i < N2; i++) h[i] = 1.0f + 0.001f * (i % 1000);
    CHECK(cudaMemcpy(u, h, N2 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(v, 0, N2 * sizeof(float)));
    free(h);

    dim3 block(16, 16), grid((N + 15) / 16, (N + 15) / 16);
    int copy_blocks = (N2 + 255) / 256;

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));

    // Warmup
    for (int i = 0; i < 20; i++) {
        launch_step(N, u, v, grid, block, 0);
        copy_kernel<<<copy_blocks, 256>>>(N2, v, u);
    }
    CHECK(cudaDeviceSynchronize());

    Result r;
    float ms;

    // --- Sync ---
    CHECK(cudaEventRecord(t0));
    for (int s = 0; s < STEPS; s++) {
        launch_step(N, u, v, grid, block, 0);
        copy_kernel<<<copy_blocks, 256>>>(N2, v, u);
        CHECK(cudaDeviceSynchronize());
    }
    CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1));
    CHECK(cudaEventElapsedTime(&ms, t0, t1));
    r.sync_us = ms * 1000.0f / STEPS;

    // --- Async ---
    CHECK(cudaEventRecord(t0));
    for (int s = 0; s < STEPS; s++) {
        launch_step(N, u, v, grid, block, 0);
        copy_kernel<<<copy_blocks, 256>>>(N2, v, u);
    }
    CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1));
    CHECK(cudaEventElapsedTime(&ms, t0, t1));
    r.async_us = ms * 1000.0f / STEPS;

    // --- Graph ---
    {
        int REPS = 5;
        cudaGraph_t graph; cudaGraphExec_t graphExec;
        cudaStream_t stream; CHECK(cudaStreamCreate(&stream));
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int s = 0; s < STEPS; s++) {
            launch_step(N, u, v, grid, block, stream);
            copy_kernel<<<copy_blocks, 256, 0, stream>>>(N2, v, u);
        }
        CHECK(cudaStreamEndCapture(stream, &graph));
        CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
        // Warmup
        for (int i = 0; i < 3; i++) { cudaGraphLaunch(graphExec, stream); cudaStreamSynchronize(stream); }
        CHECK(cudaEventRecord(t0, stream));
        for (int i = 0; i < REPS; i++) cudaGraphLaunch(graphExec, stream);
        CHECK(cudaEventRecord(t1, stream)); CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaEventElapsedTime(&ms, t0, t1));
        r.graph_us = ms * 1000.0f / (STEPS * REPS);
        cudaGraphExecDestroy(graphExec); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
    }

    // --- Persistent ---
    {
        int numBSm = 0;
        cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm, persistent_fn_ptr, 256, 0));
        int maxBlocks = numBSm * prop.multiProcessorCount;
        int neededBlocks = grid.x * grid.y;
        if (neededBlocks <= maxBlocks) {
            // Warmup
            launch_persistent(N, u, v, 5, grid, maxBlocks);
            CHECK(cudaDeviceSynchronize());
            // Benchmark
            int REPS = 5;
            CHECK(cudaEventRecord(t0));
            for (int i = 0; i < REPS; i++) {
                launch_persistent(N, u, v, STEPS, grid, maxBlocks);
            }
            CHECK(cudaEventRecord(t1)); CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventElapsedTime(&ms, t0, t1));
            r.persistent_us = ms * 1000.0f / (STEPS * REPS);
        } else {
            r.persistent_us = -1.0f;
        }
    }

    CHECK(cudaEventDestroy(t0)); CHECK(cudaEventDestroy(t1));
    CHECK(cudaFree(u)); CHECK(cudaFree(v));
    return r;
}

// ============================================================================
// Per-kernel launchers
// ============================================================================

// Heat2D
void launch_heat_step(int N, float* u, float* v, dim3 grid, dim3 block, cudaStream_t s) {
    heat2d_step<<<grid, block, 0, s>>>(N, u, v);
}
void launch_heat_persistent(int N, float* u, float* v, int STEPS, dim3 grid, int maxB) {
    void* args[] = {&N, &u, &v, &STEPS};
    cudaLaunchCooperativeKernel((void*)heat2d_persistent, grid, dim3(16,16), args);
}

// Jacobi2D
void launch_jacobi_step(int N, float* u, float* v, dim3 grid, dim3 block, cudaStream_t s) {
    jacobi2d_step<<<grid, block, 0, s>>>(N, u, v);
}
void launch_jacobi_persistent(int N, float* u, float* v, int STEPS, dim3 grid, int maxB) {
    void* args[] = {&N, &u, &v, &STEPS};
    cudaLaunchCooperativeKernel((void*)jacobi2d_persistent, grid, dim3(16,16), args);
}

// HotSpot (extra params baked in)
static float hs_dt_over_cap, hs_ce, hs_cw, hs_cn, hs_cs, hs_cc;
static float* hs_power_d = nullptr;

void init_hotspot_params(int N) {
    float dx = 0.001f; // 1mm grid
    float dt = HS_STEP_DIV;
    float Rx = dx / (2.0f * HS_K_SI * dt * dx);  // thermal resistance
    hs_dt_over_cap = dt / (HS_CAP * dx * dx);
    float r = dt / (HS_CAP * dx * dx * Rx);
    hs_cc = 1.0f - 4.0f * r;
    hs_cn = hs_cs = hs_ce = hs_cw = r;
    // Allocate power map
    int N2 = N * N;
    CHECK(cudaMalloc(&hs_power_d, N2 * sizeof(float)));
    float* p = (float*)malloc(N2 * sizeof(float));
    for (int i = 0; i < N2; i++) p[i] = 0.5f + 0.1f * sinf(i * 0.01f);
    CHECK(cudaMemcpy(hs_power_d, p, N2 * sizeof(float), cudaMemcpyHostToDevice));
    free(p);
}

void launch_hotspot_step(int N, float* u, float* v, dim3 grid, dim3 block, cudaStream_t s) {
    hotspot_step<<<grid, block, 0, s>>>(N, u, hs_power_d, v, hs_dt_over_cap, hs_ce, hs_cw, hs_cn, hs_cs, hs_cc);
}
void launch_hotspot_persistent(int N, float* u, float* v, int STEPS, dim3 grid, int maxB) {
    void* args[] = {&N, &u, &v, &hs_power_d, &hs_dt_over_cap, &hs_ce, &hs_cw, &hs_cn, &hs_cs, &hs_cc, &STEPS};
    cudaLaunchCooperativeKernel((void*)hotspot_persistent, grid, dim3(16,16), args);
}

// SRAD
static float srad_q0sq = 0.5f;

void launch_srad_step(int N, float* u, float* v, dim3 grid, dim3 block, cudaStream_t s) {
    srad_step<<<grid, block, 0, s>>>(N, u, v, srad_q0sq);
}
void launch_srad_persistent(int N, float* u, float* v, int STEPS, dim3 grid, int maxB) {
    void* args[] = {&N, &u, &v, &srad_q0sq, &STEPS};
    cudaLaunchCooperativeKernel((void*)srad_persistent, grid, dim3(16,16), args);
}

// ============================================================================
// Main
// ============================================================================
void print_row(const char* name, Result& r) {
    char ps[32], sp[16];
    if (r.persistent_us < 0) { snprintf(ps, 32, "%10s", "N/A"); snprintf(sp, 16, "%8s", "N/A"); }
    else { snprintf(ps, 32, "%10.2f", r.persistent_us); snprintf(sp, 16, "%7.1fx", r.sync_us / r.persistent_us); }
    char sa[16], sg[16];
    snprintf(sa, 16, "%7.1fx", r.sync_us / r.async_us);
    snprintf(sg, 16, "%7.1fx", r.sync_us / r.graph_us);
    printf("%-25s %10.2f %10.2f %10.2f %s  | %8s %8s %8s\n",
           name, r.sync_us, r.async_us, r.graph_us, ps, sa, sg, sp);
}

int main() {
    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("==========================================================================\n");
    printf("  PK Benchmark: PERKS / Kernel Batching kernels × 4 strategies\n");
    printf("  GPU: %s (SMs=%d)\n", prop.name, prop.multiProcessorCount);
    printf("==========================================================================\n\n");
    printf("%-25s %10s %10s %10s %10s  | Speedup over Sync\n", "Kernel", "Sync(us)", "Async(us)", "Graph(us)", "Pers(us)");
    printf("%-25s %10s %10s %10s %10s  | %8s %8s %8s\n", "", "", "", "", "", "Async", "Graph", "Persist");
    printf("%.120s\n", "---------------------------------------------------------------------------------------------------------------");

    struct { const char* name; int N; int steps; int type; } cases[] = {
        // Heat2D (PERKS benchmark)
        {"Heat2D 128",    128,  2000, 0},
        {"Heat2D 256",    256,  1000, 0},
        {"Heat2D 512",    512,  1000, 0},
        {"Heat2D 1024",  1024,   500, 0},
        // Jacobi2D (PERKS benchmark)
        {"Jacobi2D 128",  128,  2000, 1},
        {"Jacobi2D 256",  256,  1000, 1},
        {"Jacobi2D 512",  512,  1000, 1},
        {"Jacobi2D 1024",1024,   500, 1},
        // HotSpot (Kernel Batching / Rodinia)
        {"HotSpot 128",   128,  2000, 2},
        {"HotSpot 256",   256,  1000, 2},
        {"HotSpot 512",   512,  1000, 2},
        {"HotSpot 1024", 1024,   500, 2},
        // SRAD (PERKS / Rodinia)
        {"SRAD 128",      128,  2000, 3},
        {"SRAD 256",      256,  1000, 3},
        {"SRAD 512",      512,  1000, 3},
        {"SRAD 1024",    1024,   500, 3},
    };

    for (auto& c : cases) {
        Result r;
        switch (c.type) {
        case 0: // Heat2D
            r = benchmark_stencil(c.N, c.steps, launch_heat_step, launch_heat_persistent,
                                  "heat2d_persistent", (const void*)heat2d_persistent);
            break;
        case 1: // Jacobi2D
            r = benchmark_stencil(c.N, c.steps, launch_jacobi_step, launch_jacobi_persistent,
                                  "jacobi2d_persistent", (const void*)jacobi2d_persistent);
            break;
        case 2: // HotSpot
            init_hotspot_params(c.N);
            r = benchmark_stencil(c.N, c.steps, launch_hotspot_step, launch_hotspot_persistent,
                                  "hotspot_persistent", (const void*)hotspot_persistent);
            if (hs_power_d) { cudaFree(hs_power_d); hs_power_d = nullptr; }
            break;
        case 3: // SRAD
            r = benchmark_stencil(c.N, c.steps, launch_srad_step, launch_srad_persistent,
                                  "srad_persistent", (const void*)srad_persistent);
            break;
        }
        print_row(c.name, r);

        // Print separator between kernel types
        if (c.type != cases[(&c - cases) + 1 < sizeof(cases)/sizeof(cases[0]) ? (&c - cases) + 1 : (&c - cases)].type)
            printf("\n");
    }

    printf("\nPERKS (ICS 2023) reported on V100: Heat2D/Jacobi2D ~2.12x, SRAD ~2x (persistent only)\n");
    printf("Kernel Batching (2025) reported on A100: HotSpot ~1.4x (Graph only)\n");
    printf("Our model selects Persistent for small grids, Graph for large grids.\n");
    return 0;
}
