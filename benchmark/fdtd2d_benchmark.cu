/**
 * C11: FDTD 2D Maxwell — 4-strategy benchmark.
 * 3 kernels per step: update_ey, update_ex, update_hz (staggered Yee grid).
 * Strategies: Sync loop, Async loop, CUDA Graph, Persistent Kernel.
 * Build: nvcc -O3 -arch=sm_80 -rdc=true fdtd2d_benchmark.cu -o fdtd2d_bench -lcudadevrt
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

// Physical constants (normalized)
static constexpr float COURANT = 0.5f;  // dt/(dx*mu0*eps0)^0.5

// --- Kernels ---
// Ey(i,j) += COURANT * (Hz(i,j) - Hz(i-1,j))
__global__ void update_ey(int Nx, int Ny, float* __restrict__ ey,
                          const float* __restrict__ hz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= 1 && i < Nx && j < Ny) {
        int idx = i * Ny + j;
        ey[idx] += COURANT * (hz[idx] - hz[(i - 1) * Ny + j]);
    }
}

// Ex(i,j) -= COURANT * (Hz(i,j) - Hz(i,j-1))
__global__ void update_ex(int Nx, int Ny, float* __restrict__ ex,
                          const float* __restrict__ hz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < Nx && j >= 1 && j < Ny) {
        int idx = i * Ny + j;
        ex[idx] -= COURANT * (hz[idx] - hz[i * Ny + (j - 1)]);
    }
}

// Hz(i,j) -= COURANT * (Ex(i,j+1) - Ex(i,j) + Ey(i+1,j) - Ey(i,j))
__global__ void update_hz(int Nx, int Ny, float* __restrict__ hz,
                          const float* __restrict__ ex,
                          const float* __restrict__ ey) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < Nx - 1 && j < Ny - 1) {
        int idx = i * Ny + j;
        hz[idx] -= COURANT * (ex[i * Ny + (j + 1)] - ex[idx]
                             + ey[(i + 1) * Ny + j] - ey[idx]);
    }
}

// --- Persistent fused kernel ---
__global__ void fdtd2d_persistent(int Nx, int Ny, float* ex, float* ey,
                                  float* hz, int STEPS) {
    auto grid = cg::this_grid();
    int total = Nx * Ny;

    for (int s = 0; s < STEPS; s++) {
        // Phase 1: update_ey  (i in [1,Nx), j in [0,Ny))
        for (int tid = grid.thread_rank(); tid < total; tid += grid.size()) {
            int i = tid / Ny;
            int j = tid % Ny;
            if (i >= 1 && i < Nx && j < Ny) {
                int idx = i * Ny + j;
                ey[idx] += COURANT * (hz[idx] - hz[(i - 1) * Ny + j]);
            }
        }
        grid.sync();

        // Phase 2: update_ex  (i in [0,Nx), j in [1,Ny))
        for (int tid = grid.thread_rank(); tid < total; tid += grid.size()) {
            int i = tid / Ny;
            int j = tid % Ny;
            if (i < Nx && j >= 1 && j < Ny) {
                int idx = i * Ny + j;
                ex[idx] -= COURANT * (hz[idx] - hz[i * Ny + (j - 1)]);
            }
        }
        grid.sync();

        // Phase 3: update_hz  (i in [0,Nx-1), j in [0,Ny-1))
        for (int tid = grid.thread_rank(); tid < total; tid += grid.size()) {
            int i = tid / Ny;
            int j = tid % Ny;
            if (i < Nx - 1 && j < Ny - 1) {
                int idx = i * Ny + j;
                hz[idx] -= COURANT * (ex[i * Ny + (j + 1)] - ex[idx]
                                     + ey[(i + 1) * Ny + j] - ey[idx]);
            }
        }
        grid.sync();
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int STEPS = (argc > 2) ? atoi(argv[2]) : 100;
    int REPEAT = (argc > 3) ? atoi(argv[3]) : 10;
    int Nx = N, Ny = N;
    int total = Nx * Ny;

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== C11: FDTD 2D Maxwell Benchmark ===\n");
    printf("GPU: %s, SMs=%d\n", prop.name, prop.multiProcessorCount);
    printf("N=%d (%d cells), steps=%d, repeat=%d\n", N, total, STEPS, REPEAT);

    float *ex, *ey, *hz;
    CHECK(cudaMalloc(&ex, total * sizeof(float)));
    CHECK(cudaMalloc(&ey, total * sizeof(float)));
    CHECK(cudaMalloc(&hz, total * sizeof(float)));

    // Init: zero fields, small Hz pulse at center
    std::vector<float> h_zero(total, 0.0f);
    std::vector<float> h_hz(total, 0.0f);
    h_hz[(Nx / 2) * Ny + Ny / 2] = 1.0f;  // point source

    auto reset = [&]() {
        CHECK(cudaMemcpy(ex, h_zero.data(), total * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(ey, h_zero.data(), total * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(hz, h_hz.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    };
    reset();

    dim3 block(16, 16);
    dim3 grid((Nx + 15) / 16, (Ny + 15) / 16);

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    // Warmup
    for (int i = 0; i < 5; i++) {
        update_ey<<<grid, block>>>(Nx, Ny, ey, hz);
        update_ex<<<grid, block>>>(Nx, Ny, ex, hz);
        update_hz<<<grid, block>>>(Nx, Ny, hz, ex, ey);
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
            update_ey<<<grid, block>>>(Nx, Ny, ey, hz);
            update_ex<<<grid, block>>>(Nx, Ny, ex, hz);
            update_hz<<<grid, block>>>(Nx, Ny, hz, ex, ey);
            cudaDeviceSynchronize();
        }
    }, "Sync Loop");

    // Strategy 2: Async Loop
    printf("\n--- Strategy 2: Async Loop ---\n");
    run_timed([&]() {
        for (int s = 0; s < STEPS; s++) {
            update_ey<<<grid, block>>>(Nx, Ny, ey, hz);
            update_ex<<<grid, block>>>(Nx, Ny, ex, hz);
            update_hz<<<grid, block>>>(Nx, Ny, hz, ex, ey);
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
            update_ey<<<grid, block, 0, stream>>>(Nx, Ny, ey, hz);
            update_ex<<<grid, block, 0, stream>>>(Nx, Ny, ex, hz);
            update_hz<<<grid, block, 0, stream>>>(Nx, Ny, hz, ex, ey);
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
              fdtd2d_persistent, blockSize, 0));
        int maxBlocks = numBlocks * prop.multiProcessorCount;
        // Use 1D launch for persistent
        int needed = (total + blockSize - 1) / blockSize;
        int launchBlocks = (needed <= maxBlocks) ? needed : maxBlocks;

        if (launchBlocks > 0) {
            printf("Persistent: %d blocks (need %d, max %d)\n",
                   launchBlocks, needed, maxBlocks);
            void* args[] = {&Nx, &Ny, &ex, &ey, &hz, &STEPS};
            run_timed([&]() {
                cudaLaunchCooperativeKernel((void*)fdtd2d_persistent,
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
            update_ey<<<grid, block>>>(Nx, Ny, ey, hz);
            update_ex<<<grid, block>>>(Nx, Ny, ex, hz);
            update_hz<<<grid, block>>>(Nx, Ny, hz, ex, ey);
        }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms, t0, t1);
        printf("GPU compute (3 kernels): %.2f us/step\n", ms * 10.0f);
    }

    printf("\n=== CSV: fdtd2d,%d,%d,", N, STEPS);
    printf("A100\n");

    CHECK(cudaFree(ex));
    CHECK(cudaFree(ey));
    CHECK(cudaFree(hz));
    return 0;
}
