/**
 * C14: Particle-in-Cell 1D — 4-strategy benchmark.
 * 4 kernels per step: deposit, field_solve, gather_field, push.
 * Strategies: Sync loop, Async loop, CUDA Graph, Persistent Kernel.
 * Build: nvcc -O3 -arch=sm_80 -rdc=true pic1d_benchmark.cu -o pic1d_bench -lcudadevrt
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CHECK(call) do { auto e = call; if(e) { fprintf(stderr,"CUDA error %d at %s:%d\n",e,__FILE__,__LINE__); exit(1); }} while(0)

static constexpr double QM = -1.0f;   // charge/mass ratio (electron)
static constexpr double DT = 0.1f;
static constexpr double DX = 1.0f;

// --- Kernels ---

// 1. Deposit: scatter particle charges to grid via atomicAdd (CIC)
__global__ void deposit(int Np, int Ng, const float* __restrict__ x,
                        float* __restrict__ rho) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < Np) {
        float xp = x[p];
        int ic = (int)floorf(xp / DX);
        if (ic < 0) ic = 0;
        if (ic >= Ng - 1) ic = Ng - 2;
        float frac = xp / DX - ic;
        atomicAdd(&rho[ic], (1.0f - frac));
        atomicAdd(&rho[ic + 1], frac);
    }
}

// 2. Field solve: simple Gauss's law E(i) = -cumsum(rho - n0) * DX
//    Using a simple Jacobi-style relaxation for Poisson: phi''=-rho/eps0
//    For benchmark speed, do direct scan (serial on GPU, fine for small grids)
__global__ void field_solve(int Ng, const float* __restrict__ rho,
                            float* __restrict__ E, float n0) {
    // Single-thread scan — grid is small (256 or 1024)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Integrate rho to get E via Gauss's law
        float cumsum = 0.0f;
        for (int i = 0; i < Ng; i++) {
            cumsum += (rho[i] - n0) * DX;
            E[i] = -cumsum;
        }
    }
}

// 3. Gather: interpolate E-field to particle positions (CIC)
__global__ void gather_field(int Np, int Ng, const float* __restrict__ x,
                             const float* __restrict__ E,
                             float* __restrict__ Ep) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < Np) {
        float xp = x[p];
        int ic = (int)floorf(xp / DX);
        if (ic < 0) ic = 0;
        if (ic >= Ng - 1) ic = Ng - 2;
        float frac = xp / DX - ic;
        Ep[p] = (1.0f - frac) * E[ic] + frac * E[ic + 1];
    }
}

// 4. Push: leapfrog velocity + position update
__global__ void push(int Np, float Ng_len,
                     float* __restrict__ x, float* __restrict__ v,
                     const float* __restrict__ Ep) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < Np) {
        v[p] += QM * Ep[p] * DT;
        x[p] += v[p] * DT;
        // Periodic BC
        float L = Ng_len;
        if (x[p] < 0.0f) x[p] += L;
        if (x[p] >= L) x[p] -= L;
    }
}

// Zero rho array
__global__ void zero_rho(int Ng, float* __restrict__ rho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Ng) rho[i] = 0.0f;
}

// --- Persistent fused kernel ---
__global__ void pic1d_persistent(int Np, int Ng, float* x, float* v,
                                 float* rho, float* E, float* Ep,
                                 float n0, float Ng_len, int STEPS) {
    auto grid = cg::this_grid();

    for (int s = 0; s < STEPS; s++) {
        // Phase 0: zero rho
        for (int i = grid.thread_rank(); i < Ng; i += grid.size())
            rho[i] = 0.0f;
        grid.sync();

        // Phase 1: deposit
        for (int p = grid.thread_rank(); p < Np; p += grid.size()) {
            float xp = x[p];
            int ic = (int)floorf(xp / DX);
            if (ic < 0) ic = 0;
            if (ic >= Ng - 1) ic = Ng - 2;
            float frac = xp / DX - ic;
            atomicAdd(&rho[ic], (1.0f - frac));
            atomicAdd(&rho[ic + 1], frac);
        }
        grid.sync();

        // Phase 2: field solve (single thread serial scan)
        if (grid.thread_rank() == 0) {
            float cumsum = 0.0f;
            for (int i = 0; i < Ng; i++) {
                cumsum += (rho[i] - n0) * DX;
                E[i] = -cumsum;
            }
        }
        grid.sync();

        // Phase 3: gather
        for (int p = grid.thread_rank(); p < Np; p += grid.size()) {
            float xp = x[p];
            int ic = (int)floorf(xp / DX);
            if (ic < 0) ic = 0;
            if (ic >= Ng - 1) ic = Ng - 2;
            float frac = xp / DX - ic;
            Ep[p] = (1.0f - frac) * E[ic] + frac * E[ic + 1];
        }
        grid.sync();

        // Phase 4: push
        for (int p = grid.thread_rank(); p < Np; p += grid.size()) {
            v[p] += QM * Ep[p] * DT;
            x[p] += v[p] * DT;
            float L = Ng_len;
            if (x[p] < 0.0f) x[p] += L;
            if (x[p] >= L) x[p] -= L;
        }
        grid.sync();
    }
}

int main(int argc, char* argv[]) {
    int Np = (argc > 1) ? atoi(argv[1]) : 4096;
    int Ng = (argc > 2) ? atoi(argv[2]) : 256;
    int STEPS = (argc > 3) ? atoi(argv[3]) : 100;
    int REPEAT = (argc > 4) ? atoi(argv[4]) : 10;
    float Ng_len = Ng * DX;
    float n0 = (float)Np / (float)Ng;  // background density

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== C14: Particle-in-Cell 1D Benchmark ===\n");
    printf("GPU: %s, SMs=%d\n", prop.name, prop.multiProcessorCount);
    printf("Np=%d, Ng=%d, steps=%d, repeat=%d\n", Np, Ng, STEPS, REPEAT);

    float *d_x, *d_v, *d_rho, *d_E, *d_Ep;
    CHECK(cudaMalloc(&d_x, Np * sizeof(float)));
    CHECK(cudaMalloc(&d_v, Np * sizeof(float)));
    CHECK(cudaMalloc(&d_rho, Ng * sizeof(float)));
    CHECK(cudaMalloc(&d_E, Ng * sizeof(float)));
    CHECK(cudaMalloc(&d_Ep, Np * sizeof(float)));

    // Init: uniform positions + small sinusoidal perturbation, thermal velocity
    std::vector<float> h_x(Np), h_v(Np);
    for (int p = 0; p < Np; p++) {
        float base = (p + 0.5f) * Ng_len / Np;
        h_x[p] = base + 0.5f * sinf(2.0f * M_PI * base / Ng_len);
        if (h_x[p] < 0.0f) h_x[p] += Ng_len;
        if (h_x[p] >= Ng_len) h_x[p] -= Ng_len;
        h_v[p] = 0.1f * sinf(2.0f * M_PI * p / Np);  // small thermal spread
    }

    auto reset = [&]() {
        CHECK(cudaMemcpy(d_x, h_x.data(), Np * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_v, h_v.data(), Np * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemset(d_rho, 0, Ng * sizeof(float)));
        CHECK(cudaMemset(d_E, 0, Ng * sizeof(float)));
        CHECK(cudaMemset(d_Ep, 0, Np * sizeof(float)));
    };
    reset();

    int part_grid = (Np + 255) / 256;
    int rho_grid = (Ng + 255) / 256;

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    // Warmup
    for (int i = 0; i < 5; i++) {
        zero_rho<<<rho_grid, 256>>>(Ng, d_rho);
        deposit<<<part_grid, 256>>>(Np, Ng, d_x, d_rho);
        field_solve<<<1, 1>>>(Ng, d_rho, d_E, n0);
        gather_field<<<part_grid, 256>>>(Np, Ng, d_x, d_E, d_Ep);
        push<<<part_grid, 256>>>(Np, Ng_len, d_x, d_v, d_Ep);
    }
    cudaDeviceSynchronize();

    auto run_timed = [&](auto fn, const char* name) {
        std::vector<float> times;
        for (int r = 0; r < REPEAT; r++) {
            reset();
            cudaDeviceSynchronize();
            cudaEventRecord(t0);
            fn();
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            float ms;
            cudaEventElapsedTime(&ms, t0, t1);
            times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        float median = times[REPEAT / 2];
        printf("[%s] %d steps: median=%.3f ms, %.2f us/step\n",
               name, STEPS, median, median * 1000.0f / STEPS);
    };

    // Strategy 1: Sync Loop
    printf("\n--- Strategy 1: Sync Loop ---\n");
    run_timed([&]() {
        for (int s = 0; s < STEPS; s++) {
            zero_rho<<<rho_grid, 256>>>(Ng, d_rho);
            deposit<<<part_grid, 256>>>(Np, Ng, d_x, d_rho);
            field_solve<<<1, 1>>>(Ng, d_rho, d_E, n0);
            gather_field<<<part_grid, 256>>>(Np, Ng, d_x, d_E, d_Ep);
            push<<<part_grid, 256>>>(Np, Ng_len, d_x, d_v, d_Ep);
            cudaDeviceSynchronize();
        }
    }, "Sync Loop");

    // Strategy 2: Async Loop
    printf("\n--- Strategy 2: Async Loop ---\n");
    run_timed([&]() {
        for (int s = 0; s < STEPS; s++) {
            zero_rho<<<rho_grid, 256>>>(Ng, d_rho);
            deposit<<<part_grid, 256>>>(Np, Ng, d_x, d_rho);
            field_solve<<<1, 1>>>(Ng, d_rho, d_E, n0);
            gather_field<<<part_grid, 256>>>(Np, Ng, d_x, d_E, d_Ep);
            push<<<part_grid, 256>>>(Np, Ng_len, d_x, d_v, d_Ep);
        }
        cudaDeviceSynchronize();
    }, "Async Loop");

    // Strategy 3: CUDA Graph
    printf("\n--- Strategy 3: CUDA Graph ---\n");
    {
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int s = 0; s < STEPS; s++) {
            zero_rho<<<rho_grid, 256, 0, stream>>>(Ng, d_rho);
            deposit<<<part_grid, 256, 0, stream>>>(Np, Ng, d_x, d_rho);
            field_solve<<<1, 1, 0, stream>>>(Ng, d_rho, d_E, n0);
            gather_field<<<part_grid, 256, 0, stream>>>(Np, Ng, d_x, d_E, d_Ep);
            push<<<part_grid, 256, 0, stream>>>(Np, Ng_len, d_x, d_v, d_Ep);
        }
        CHECK(cudaStreamEndCapture(stream, &graph));
        CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

        run_timed([&]() {
            cudaGraphLaunch(graphExec, stream);
            cudaStreamSynchronize(stream);
        }, "CUDA Graph");

        CHECK(cudaGraphExecDestroy(graphExec));
        CHECK(cudaGraphDestroy(graph));
        CHECK(cudaStreamDestroy(stream));
    }

    // Strategy 4: Persistent Kernel
    printf("\n--- Strategy 4: Persistent Kernel ---\n");
    {
        int blockSize = 256;
        int numBlocks;
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
              pic1d_persistent, blockSize, 0));
        int maxBlocks = numBlocks * prop.multiProcessorCount;
        int needed = (Np + blockSize - 1) / blockSize;
        int launchBlocks = (needed <= maxBlocks) ? needed : maxBlocks;

        if (launchBlocks > 0) {
            printf("Persistent: %d blocks (need %d, max %d)\n",
                   launchBlocks, needed, maxBlocks);
            void* args[] = {&Np, &Ng, &d_x, &d_v, &d_rho, &d_E, &d_Ep,
                            &n0, &Ng_len, &STEPS};
            run_timed([&]() {
                cudaLaunchCooperativeKernel((void*)pic1d_persistent,
                    dim3(launchBlocks), dim3(blockSize), args);
                cudaDeviceSynchronize();
            }, "Persistent");
        } else {
            printf("Persistent: N/A (max blocks = 0)\n");
        }
    }

    // Overhead breakdown
    printf("\n--- Overhead Breakdown ---\n");
    {
        float ms;
        cudaDeviceSynchronize();
        cudaEventRecord(t0);
        for (int i = 0; i < 100; i++) {
            zero_rho<<<rho_grid, 256>>>(Ng, d_rho);
            deposit<<<part_grid, 256>>>(Np, Ng, d_x, d_rho);
            field_solve<<<1, 1>>>(Ng, d_rho, d_E, n0);
            gather_field<<<part_grid, 256>>>(Np, Ng, d_x, d_E, d_Ep);
            push<<<part_grid, 256>>>(Np, Ng_len, d_x, d_v, d_Ep);
        }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms, t0, t1);
        printf("GPU compute (5 kernels): %.2f us/step\n", ms * 10.0f);
    }

    printf("\n=== CSV: pic1d,%d,%d,%d,", Np, Ng, STEPS);
    printf("A100\n");

    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_v));
    CHECK(cudaFree(d_rho));
    CHECK(cudaFree(d_E));
    CHECK(cudaFree(d_Ep));
    return 0;
}
