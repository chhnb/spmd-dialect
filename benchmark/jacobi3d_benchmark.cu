/**
 * C2: Jacobi 3D 7-point stencil — 4-strategy benchmark.
 * Strategies: Sync loop, Async loop, CUDA Graph, Persistent Kernel.
 * Build: nvcc -O3 -arch=sm_80 -rdc=true jacobi3d_benchmark.cu -o jacobi3d_bench -lcudadevrt
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

// --- Device Graph tail launch support ---
__device__ cudaGraphExec_t d_graph_exec;
__global__ void tail_launch_kernel(int* steps_remaining) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int rem = atomicSub(steps_remaining, 1);
        if (rem > 1) cudaGraphLaunch(d_graph_exec, cudaStreamGraphTailLaunch);
    }
}

// --- Kernels ---
__global__ void jacobi3d_step(int NX, int NY, int NZ,
                              const float* __restrict__ u, float* __restrict__ v) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = NX * NY * NZ;
    if (tid >= total) return;
    int i = tid / (NY * NZ);
    int j = (tid / NZ) % NY;
    int k = tid % NZ;
    if (i >= 1 && i < NX-1 && j >= 1 && j < NY-1 && k >= 1 && k < NZ-1) {
        int idx = i * NY * NZ + j * NZ + k;
        v[idx] = (u[idx - NY*NZ] + u[idx + NY*NZ] +
                  u[idx - NZ]    + u[idx + NZ] +
                  u[idx - 1]     + u[idx + 1]) / 6.0f;
    }
}

__global__ void copy_kernel(int N, const float* __restrict__ src, float* __restrict__ dst) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < N) dst[i] = src[i];
}

__global__ void jacobi3d_persistent(int NX, int NY, int NZ, int total,
                                    float* u, float* v, int STEPS) {
    int NYNZ = NY * NZ;
    for (int s = 0; s < STEPS; s++) {
        for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total;
             tid += gridDim.x * blockDim.x) {
            int i = tid / NYNZ;
            int j = (tid / NZ) % NY;
            int k = tid % NZ;
            if (i >= 1 && i < NX-1 && j >= 1 && j < NY-1 && k >= 1 && k < NZ-1) {
                v[tid] = (u[tid - NYNZ] + u[tid + NYNZ] +
                          u[tid - NZ]   + u[tid + NZ] +
                          u[tid - 1]    + u[tid + 1]) / 6.0f;
            }
        }
        cg::this_grid().sync();
        for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total;
             tid += gridDim.x * blockDim.x) {
            int i = tid / NYNZ;
            int j = (tid / NZ) % NY;
            int k = tid % NZ;
            if (i >= 1 && i < NX-1 && j >= 1 && j < NY-1 && k >= 1 && k < NZ-1) {
                u[tid] = v[tid];
            }
        }
        cg::this_grid().sync();
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 64;
    int STEPS = (argc > 2) ? atoi(argv[2]) : 100;
    int REPEAT = (argc > 3) ? atoi(argv[3]) : 10;
    int NX = N, NY = N, NZ = N;
    int total = NX * NY * NZ;

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== C2: Jacobi 3D 7pt Benchmark ===\n");
    printf("GPU: %s, SMs=%d\n", prop.name, prop.multiProcessorCount);
    printf("N=%d (%d^3 = %d cells), steps=%d, repeat=%d\n", N, N, total, STEPS, REPEAT);

    float *u, *v;
    CHECK(cudaMalloc(&u, total * sizeof(float)));
    CHECK(cudaMalloc(&v, total * sizeof(float)));

    // Init: boundary face k=0 = 1.0, rest = 0.0
    std::vector<float> h_u(total, 0.0f);
    for (int i = 0; i < NX; i++)
        for (int j = 0; j < NY; j++)
            h_u[i * NY * NZ + j * NZ + 0] = 1.0f;
    CHECK(cudaMemcpy(u, h_u.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(v, h_u.data(), total * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    int copyGrid = (total + 255) / 256;

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    // Warmup
    for (int i = 0; i < 5; i++) {
        jacobi3d_step<<<gridSize, blockSize>>>(NX, NY, NZ, u, v);
        copy_kernel<<<copyGrid, 256>>>(total, v, u);
    }
    cudaDeviceSynchronize();

    auto run_timed = [&](auto fn, const char* name) {
        std::vector<float> times;
        for (int r = 0; r < REPEAT; r++) {
            CHECK(cudaMemcpy(u, h_u.data(), total * sizeof(float), cudaMemcpyHostToDevice));
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
            jacobi3d_step<<<gridSize, blockSize>>>(NX, NY, NZ, u, v);
            copy_kernel<<<copyGrid, 256>>>(total, v, u);
            cudaDeviceSynchronize();
        }
    }, "Sync Loop");

    // Strategy 2: Async Loop
    printf("\n--- Strategy 2: Async Loop ---\n");
    run_timed([&]() {
        for (int s = 0; s < STEPS; s++) {
            jacobi3d_step<<<gridSize, blockSize>>>(NX, NY, NZ, u, v);
            copy_kernel<<<copyGrid, 256>>>(total, v, u);
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
            jacobi3d_step<<<gridSize, blockSize, 0, stream>>>(NX, NY, NZ, u, v);
            copy_kernel<<<copyGrid, 256, 0, stream>>>(total, v, u);
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

    // Strategy 3b: Device Graph (tail launch, same kernels, no fusion)
    printf("\n--- Strategy 3b: Device Graph (tail launch) ---\n");
    {
        int *d_steps;
        CHECK(cudaMalloc(&d_steps, sizeof(int)));
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));

        // Capture ONE step: jacobi3d_step + copy_kernel
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        jacobi3d_step<<<gridSize, blockSize, 0, stream>>>(NX, NY, NZ, u, v);
        copy_kernel<<<copyGrid, 256, 0, stream>>>(total, v, u);
        tail_launch_kernel<<<1, 1, 0, stream>>>(d_steps);
        CHECK(cudaStreamEndCapture(stream, &graph));

        // Instantiate for device-side launch
        CHECK(cudaGraphInstantiateWithFlags(&graphExec, graph,
              cudaGraphInstantiateFlagDeviceLaunch));
        CHECK(cudaGraphUpload(graphExec, stream));

        // Copy graph exec handle to device symbol
        cudaGraphExec_t* d_sym_ptr;
        CHECK(cudaGetSymbolAddress((void**)&d_sym_ptr, d_graph_exec));
        CHECK(cudaMemcpy(d_sym_ptr, &graphExec, sizeof(cudaGraphExec_t),
              cudaMemcpyHostToDevice));

        // Warmup
        for (int w = 0; w < 5; w++) {
            int sv = STEPS;
            CHECK(cudaMemcpy(d_steps, &sv, sizeof(int), cudaMemcpyHostToDevice));
            CHECK(cudaGraphLaunch(graphExec, stream));
            CHECK(cudaStreamSynchronize(stream));
        }

        run_timed([&]() {
            int sv = STEPS;
            CHECK(cudaMemcpy(d_steps, &sv, sizeof(int), cudaMemcpyHostToDevice));
            CHECK(cudaGraphLaunch(graphExec, stream));
            cudaStreamSynchronize(stream);
        }, "DevGraph");

        CHECK(cudaGraphExecDestroy(graphExec));
        CHECK(cudaGraphDestroy(graph));
        CHECK(cudaStreamDestroy(stream));
        CHECK(cudaFree(d_steps));
    }

    // Strategy 4: Persistent Kernel (grid-stride loop)
    printf("\n--- Strategy 4: Persistent Kernel ---\n");
    {
        int numBlocks;
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
              jacobi3d_persistent, blockSize, 0));
        int maxBlocks = numBlocks * prop.multiProcessorCount;
        int needed = gridSize;
        int pGridSize = (needed <= maxBlocks) ? needed : maxBlocks;

        printf("Persistent: %d blocks (need %d, max %d, launch %d)\n",
               pGridSize, needed, maxBlocks, pGridSize);
        void* args[] = {&NX, &NY, &NZ, &total, &u, &v, &STEPS};
        run_timed([&]() {
            cudaLaunchCooperativeKernel((void*)jacobi3d_persistent,
                dim3(pGridSize), dim3(blockSize), args);
            cudaDeviceSynchronize();
        }, "Persistent");
    }

    printf("\n=== CSV: jacobi3d,%d,%d,A100 ===\n", N, STEPS);

    CHECK(cudaFree(u));
    CHECK(cudaFree(v));
    return 0;
}
