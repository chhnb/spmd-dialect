/**
 * C12: MacCormack 3D advection — 4-strategy benchmark.
 * 3 kernels per step: predictor, corrector, copy.
 * Strategies: Sync loop, Async loop, CUDA Graph, Persistent Kernel.
 * Build: nvcc -O3 -arch=sm_80 -rdc=true maccormack3d_benchmark.cu -o maccormack3d_bench -lcudadevrt
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

// Constant velocity field
static constexpr float U = 1.0f, V = 0.5f, W = 0.25f;
static constexpr float DT = 0.001f, DX = 1.0f;

// 3D index
__device__ __forceinline__ int idx3(int i, int j, int k, int Ny, int Nz) {
    return (i * Ny + j) * Nz + k;
}

// Predictor: forward differences
__global__ void predictor(int Nx, int Ny, int Nz,
                          const float* __restrict__ q,
                          float* __restrict__ qp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1 && k >= 1 && k < Nz - 1) {
        float c = q[idx3(i, j, k, Ny, Nz)];
        float dqdx = (q[idx3(i + 1, j, k, Ny, Nz)] - c) / DX;
        float dqdy = (q[idx3(i, j + 1, k, Ny, Nz)] - c) / DX;
        float dqdz = (q[idx3(i, j, k + 1, Ny, Nz)] - c) / DX;
        qp[idx3(i, j, k, Ny, Nz)] = c - DT * (U * dqdx + V * dqdy + W * dqdz);
    }
}

// Corrector: backward differences + average with original
__global__ void corrector(int Nx, int Ny, int Nz,
                          const float* __restrict__ q,
                          const float* __restrict__ qp,
                          float* __restrict__ qnew) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1 && k >= 1 && k < Nz - 1) {
        int id = idx3(i, j, k, Ny, Nz);
        float cp = qp[id];
        float dqdx = (cp - qp[idx3(i - 1, j, k, Ny, Nz)]) / DX;
        float dqdy = (cp - qp[idx3(i, j - 1, k, Ny, Nz)]) / DX;
        float dqdz = (cp - qp[idx3(i, j, k - 1, Ny, Nz)]) / DX;
        float qbar = cp - DT * (U * dqdx + V * dqdy + W * dqdz);
        qnew[id] = 0.5f * (q[id] + qbar);
    }
}

// Copy result back to state array
__global__ void copy_kernel(int total, const float* __restrict__ src,
                            float* __restrict__ dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) dst[i] = src[i];
}

// --- Persistent fused kernel ---
__global__ void maccormack3d_persistent(int Nx, int Ny, int Nz,
                                        float* q, float* qp, float* qnew,
                                        int STEPS) {
    auto grid = cg::this_grid();
    int total = Nx * Ny * Nz;

    for (int s = 0; s < STEPS; s++) {
        // Phase 1: predictor
        for (int tid = grid.thread_rank(); tid < total; tid += grid.size()) {
            int i = tid / (Ny * Nz);
            int j = (tid / Nz) % Ny;
            int k = tid % Nz;
            if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1 && k >= 1 && k < Nz - 1) {
                float c = q[tid];
                float dqdx = (q[idx3(i + 1, j, k, Ny, Nz)] - c) / DX;
                float dqdy = (q[idx3(i, j + 1, k, Ny, Nz)] - c) / DX;
                float dqdz = (q[idx3(i, j, k + 1, Ny, Nz)] - c) / DX;
                qp[tid] = c - DT * (U * dqdx + V * dqdy + W * dqdz);
            }
        }
        grid.sync();

        // Phase 2: corrector
        for (int tid = grid.thread_rank(); tid < total; tid += grid.size()) {
            int i = tid / (Ny * Nz);
            int j = (tid / Nz) % Ny;
            int k = tid % Nz;
            if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1 && k >= 1 && k < Nz - 1) {
                float cp = qp[tid];
                float dqdx = (cp - qp[idx3(i - 1, j, k, Ny, Nz)]) / DX;
                float dqdy = (cp - qp[idx3(i, j - 1, k, Ny, Nz)]) / DX;
                float dqdz = (cp - qp[idx3(i, j, k - 1, Ny, Nz)]) / DX;
                float qbar = cp - DT * (U * dqdx + V * dqdy + W * dqdz);
                qnew[tid] = 0.5f * (q[tid] + qbar);
            }
        }
        grid.sync();

        // Phase 3: copy qnew -> q
        for (int tid = grid.thread_rank(); tid < total; tid += grid.size()) {
            q[tid] = qnew[tid];
        }
        grid.sync();
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 64;
    int STEPS = (argc > 2) ? atoi(argv[2]) : 100;
    int REPEAT = (argc > 3) ? atoi(argv[3]) : 10;
    int Nx = N, Ny = N, Nz = N;
    int total = Nx * Ny * Nz;

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== C12: MacCormack 3D Advection Benchmark ===\n");
    printf("GPU: %s, SMs=%d\n", prop.name, prop.multiProcessorCount);
    printf("N=%d (%d cells), steps=%d, repeat=%d\n", N, total, STEPS, REPEAT);

    float *q, *qp, *qnew;
    CHECK(cudaMalloc(&q, total * sizeof(float)));
    CHECK(cudaMalloc(&qp, total * sizeof(float)));
    CHECK(cudaMalloc(&qnew, total * sizeof(float)));

    // Init: Gaussian blob at center
    std::vector<float> h_q(total, 0.0f);
    float cx = Nx / 2.0f, cy = Ny / 2.0f, cz = Nz / 2.0f;
    float sigma2 = (N / 8.0f) * (N / 8.0f);
    for (int i = 0; i < Nx; i++)
        for (int j = 0; j < Ny; j++)
            for (int k = 0; k < Nz; k++) {
                float r2 = (i - cx) * (i - cx) + (j - cy) * (j - cy) + (k - cz) * (k - cz);
                h_q[(i * Ny + j) * Nz + k] = expf(-r2 / (2.0f * sigma2));
            }

    auto reset = [&]() {
        CHECK(cudaMemcpy(q, h_q.data(), total * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemset(qp, 0, total * sizeof(float)));
        CHECK(cudaMemset(qnew, 0, total * sizeof(float)));
    };
    reset();

    dim3 block(8, 8, 4);
    dim3 grid3d((Nx + 7) / 8, (Ny + 7) / 8, (Nz + 3) / 4);
    int copy_grid = (total + 255) / 256;

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    // Warmup
    for (int i = 0; i < 5; i++) {
        predictor<<<grid3d, block>>>(Nx, Ny, Nz, q, qp);
        corrector<<<grid3d, block>>>(Nx, Ny, Nz, q, qp, qnew);
        copy_kernel<<<copy_grid, 256>>>(total, qnew, q);
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
            predictor<<<grid3d, block>>>(Nx, Ny, Nz, q, qp);
            corrector<<<grid3d, block>>>(Nx, Ny, Nz, q, qp, qnew);
            copy_kernel<<<copy_grid, 256>>>(total, qnew, q);
            cudaDeviceSynchronize();
        }
    }, "Sync Loop");

    // Strategy 2: Async Loop
    printf("\n--- Strategy 2: Async Loop ---\n");
    run_timed([&]() {
        for (int s = 0; s < STEPS; s++) {
            predictor<<<grid3d, block>>>(Nx, Ny, Nz, q, qp);
            corrector<<<grid3d, block>>>(Nx, Ny, Nz, q, qp, qnew);
            copy_kernel<<<copy_grid, 256>>>(total, qnew, q);
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
            predictor<<<grid3d, block, 0, stream>>>(Nx, Ny, Nz, q, qp);
            corrector<<<grid3d, block, 0, stream>>>(Nx, Ny, Nz, q, qp, qnew);
            copy_kernel<<<copy_grid, 256, 0, stream>>>(total, qnew, q);
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
              maccormack3d_persistent, blockSize, 0));
        int maxBlocks = numBlocks * prop.multiProcessorCount;
        int needed = (total + blockSize - 1) / blockSize;
        int launchBlocks = (needed <= maxBlocks) ? needed : maxBlocks;

        if (launchBlocks > 0) {
            printf("Persistent: %d blocks (need %d, max %d)\n",
                   launchBlocks, needed, maxBlocks);
            void* args[] = {&Nx, &Ny, &Nz, &q, &qp, &qnew, &STEPS};
            run_timed([&]() {
                cudaLaunchCooperativeKernel((void*)maccormack3d_persistent,
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
            predictor<<<grid3d, block>>>(Nx, Ny, Nz, q, qp);
            corrector<<<grid3d, block>>>(Nx, Ny, Nz, q, qp, qnew);
            copy_kernel<<<copy_grid, 256>>>(total, qnew, q);
        }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms, t0, t1);
        printf("GPU compute (3 kernels): %.2f us/step\n", ms * 10.0f);
    }

    printf("\n=== CSV: maccormack3d,%d,%d,", N, STEPS);
    printf("A100\n");

    CHECK(cudaFree(q));
    CHECK(cudaFree(qp));
    CHECK(cudaFree(qnew));
    return 0;
}
