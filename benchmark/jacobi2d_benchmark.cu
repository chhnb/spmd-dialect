/**
 * C1: Jacobi 2D 5-point stencil — 4-strategy benchmark.
 * Strategies: Sync loop, Async loop, CUDA Graph, Persistent Kernel.
 * Build: nvcc -O3 -arch=sm_80 -rdc=true jacobi2d_benchmark.cu -o jacobi2d_bench -lcudadevrt
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

// --- Kernels ---
__global__ void jacobi_step(int N, const float* __restrict__ u, float* __restrict__ v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
        int idx = i * N + j;
        v[idx] = 0.25f * (u[idx-N] + u[idx+N] + u[idx-1] + u[idx+1]);
    }
}

__global__ void copy_kernel(int N2, const float* __restrict__ src, float* __restrict__ dst) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < N2) dst[i] = src[i];
}

__global__ void jacobi_persistent(int N, float* u, float* v, int STEPS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    for (int s = 0; s < STEPS; s++) {
        if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
            int idx = i * N + j;
            v[idx] = 0.25f * (u[idx-N] + u[idx+N] + u[idx-1] + u[idx+1]);
        }
        cg::this_grid().sync();
        if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
            int idx = i * N + j;
            u[idx] = v[idx];
        }
        cg::this_grid().sync();
    }
}

// Grid-stride persistent: uses maxCooperativeBlocks, never N/A
__global__ void jacobi_persistent_stride(int N, float* u, float* v, int STEPS) {
    auto grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int N2 = N * N;
    for (int s = 0; s < STEPS; s++) {
        for (int idx = tid; idx < N2; idx += stride) {
            int i = idx / N, j = idx % N;
            if (i >= 1 && i < N-1 && j >= 1 && j < N-1)
                v[idx] = 0.25f * (u[idx-N] + u[idx+N] + u[idx-1] + u[idx+1]);
        }
        grid.sync();
        for (int idx = tid; idx < N2; idx += stride)
            u[idx] = v[idx];
        grid.sync();
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 4096;
    int STEPS = (argc > 2) ? atoi(argv[2]) : 100;
    int REPEAT = (argc > 3) ? atoi(argv[3]) : 10;
    int N2 = N * N;

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== C1: Jacobi 2D 5pt Benchmark ===\n");
    printf("GPU: %s, SMs=%d\n", prop.name, prop.multiProcessorCount);
    printf("N=%d (%d cells), steps=%d, repeat=%d\n", N, N2, STEPS, REPEAT);

    float *u, *v;
    CHECK(cudaMalloc(&u, N2 * sizeof(float)));
    CHECK(cudaMalloc(&v, N2 * sizeof(float)));

    // Init: boundary = 1.0, interior = 0.0
    std::vector<float> h_u(N2, 0.0f);
    for (int j = 0; j < N; j++) h_u[j] = 1.0f; // top row
    CHECK(cudaMemcpy(u, h_u.data(), N2 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(v, h_u.data(), N2 * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    int copy_grid = (N2 + 255) / 256;

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    // Warmup
    for (int i = 0; i < 5; i++) {
        jacobi_step<<<grid, block>>>(N, u, v);
        copy_kernel<<<copy_grid, 256>>>(N2, v, u);
    }
    cudaDeviceSynchronize();

    auto run_timed = [&](auto fn, const char* name) {
        std::vector<float> times;
        for (int r = 0; r < REPEAT; r++) {
            CHECK(cudaMemcpy(u, h_u.data(), N2 * sizeof(float), cudaMemcpyHostToDevice));
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
            jacobi_step<<<grid, block>>>(N, u, v);
            copy_kernel<<<copy_grid, 256>>>(N2, v, u);
            cudaDeviceSynchronize();
        }
    }, "Sync Loop");

    // Strategy 2: Async Loop
    printf("\n--- Strategy 2: Async Loop ---\n");
    run_timed([&]() {
        for (int s = 0; s < STEPS; s++) {
            jacobi_step<<<grid, block>>>(N, u, v);
            copy_kernel<<<copy_grid, 256>>>(N2, v, u);
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
            jacobi_step<<<grid, block, 0, stream>>>(N, u, v);
            copy_kernel<<<copy_grid, 256, 0, stream>>>(N2, v, u);
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
        int numBlocks;
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
              jacobi_persistent, block.x * block.y, 0));
        int maxBlocks = numBlocks * prop.multiProcessorCount;
        int needed = grid.x * grid.y;

        if (needed <= maxBlocks) {
            printf("Persistent: %dx%d blocks (need %d, max %d)\n",
                   grid.x, grid.y, needed, maxBlocks);
            void* args[] = {&N, &u, &v, &STEPS};
            run_timed([&]() {
                cudaLaunchCooperativeKernel((void*)jacobi_persistent,
                    grid, block, args);
                cudaDeviceSynchronize();
            }, "Persistent");
        } else {
            printf("Persistent: N/A (need %d blocks, max %d)\n", needed, maxBlocks);
        }
    }

    // Strategy 5: Grid-Stride Persistent (never N/A)
    printf("\n--- Strategy 5: Grid-Stride Persistent ---\n");
    {
        int gsBSm = 0;
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&gsBSm,
              jacobi_persistent_stride, 256, 0));
        int gsMax = gsBSm * prop.multiProcessorCount;
        printf("Grid-stride: %d blocks (always fits)\n", gsMax);
        void* gsArgs[] = {&N, &u, &v, &STEPS};
        run_timed([&]() {
            CHECK(cudaMemcpy(u, h_u.data(), N2 * sizeof(float), cudaMemcpyHostToDevice));
            CHECK(cudaMemset(v, 0, N2 * sizeof(float)));
            cudaLaunchCooperativeKernel((void*)jacobi_persistent_stride,
                dim3(gsMax), dim3(256), gsArgs);
            cudaDeviceSynchronize();
        }, "GridStride");
    }

    // Overhead breakdown
    printf("\n--- Overhead Breakdown ---\n");
    {
        float ms;
        cudaDeviceSynchronize();
        cudaEventRecord(t0);
        for (int i = 0; i < 100; i++) {
            jacobi_step<<<grid, block>>>(N, u, v);
            copy_kernel<<<copy_grid, 256>>>(N2, v, u);
        }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms, t0, t1);
        printf("GPU compute (2 kernels): %.2f us/step\n", ms * 10.0f);
    }

    printf("\n=== CSV: jacobi2d,%d,%d,", N, STEPS);
    printf("A100\n");

    CHECK(cudaFree(u));
    CHECK(cudaFree(v));
    return 0;
}
