/**
 * C19: LU Decomposition (PolyBench-style) — 4-strategy benchmark.
 * N-1 kernel launches per step (one per pivot row).
 * Each kernel processes remaining rows below the pivot.
 * Build: nvcc -O3 -arch=sm_80 -rdc=true lu_benchmark.cu -o lu_bench -lcudadevrt
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

// LU kernel for a single pivot k: update rows i > k
__global__ void lu_step_kernel(int N, int k, float* __restrict__ A) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + k + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + k + 1;
    if (i < N && j < N) {
        // First column of the submatrix (compute L)
        if (j == k + 1) {
            A[i * N + k] /= A[k * N + k];
        }
    }
}

// Separate update kernel after L column is computed
__global__ void lu_update_kernel(int N, int k, float* __restrict__ A) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + k + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + k + 1;
    if (i < N && j < N) {
        A[i * N + j] -= A[i * N + k] * A[k * N + j];
    }
}

// Combined LU kernel per pivot (factor column + update submatrix)
__global__ void lu_pivot_kernel(int N, int k, float* __restrict__ A) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int remaining = N - k - 1;
    // Phase 1: compute L column
    if (tid < remaining) {
        int i = tid + k + 1;
        A[i * N + k] /= A[k * N + k];
    }
    __syncthreads();
    // Phase 2: update submatrix
    // Use 2D indexing within 1D thread layout
    for (int i = k + 1 + (int)(blockIdx.x); i < N; i += gridDim.x) {
        if (tid < remaining) {
            int j = tid + k + 1;
            A[i * N + j] -= A[i * N + k] * A[k * N + j];
        }
    }
}

// Persistent: all pivots in one cooperative kernel
// NOTE: this is challenging because each pivot depends on the previous.
// We use grid_sync between pivots.
__global__ void lu_persistent(int N, float* __restrict__ A) {
    auto grid = cg::this_grid();
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int k = 0; k < N - 1; k++) {
        // Factor L column
        int i = tid_x + k + 1;
        if (i < N && tid_y == 0) {
            A[i * N + k] /= A[k * N + k];
        }
        grid.sync();

        // Update submatrix
        i = tid_x + k + 1;
        int j = tid_y + k + 1;
        if (i < N && j < N) {
            A[i * N + j] -= A[i * N + k] * A[k * N + j];
        }
        grid.sync();
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 512;
    int STEPS = (argc > 2) ? atoi(argv[2]) : 3;
    int REPEAT = (argc > 3) ? atoi(argv[3]) : 10;
    long N2 = (long)N * N;
    int launches_per_step = 2 * (N - 1); // factor + update per pivot

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== C19: LU Decomposition Benchmark ===\n");
    printf("GPU: %s, SMs=%d\n", prop.name, prop.multiProcessorCount);
    printf("N=%d (%ld elements), steps=%d, repeat=%d\n", N, N2, STEPS, REPEAT);
    printf("Launches/step: %d\n", launches_per_step);

    float *A;
    CHECK(cudaMalloc(&A, N2 * sizeof(float)));

    // Init: diagonally dominant matrix for stability
    std::vector<float> h_A(N2);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            h_A[i*N+j] = (i == j) ? (float)N + 1.0f : sinf((float)(i*N+j) * 0.01f) * 0.5f;

    auto reset = [&]() {
        CHECK(cudaMemcpy(A, h_A.data(), N2*sizeof(float), cudaMemcpyHostToDevice));
    };
    reset();

    dim3 block(16, 16);
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    // Warmup
    {
        dim3 g((N+15)/16, (N+15)/16);
        lu_step_kernel<<<g, block>>>(N, 0, A);
        lu_update_kernel<<<g, block>>>(N, 0, A);
        cudaDeviceSynchronize();
    }

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
        float total_launches = (float)STEPS * launches_per_step;
        printf("[%s] %d steps: median=%.3f ms, %.2f us/step, %.2f us/launch\n",
               name, STEPS, median, median * 1000.0f / STEPS,
               median * 1000.0f / total_launches);
    };

    // Strategy 1: Sync
    printf("\n--- Strategy 1: Sync ---\n");
    run_timed([&]() {
        for (int s = 0; s < STEPS; s++) {
            reset();
            for (int k = 0; k < N - 1; k++) {
                int rem = N - k - 1;
                dim3 g((rem+15)/16, (rem+15)/16);
                lu_step_kernel<<<g, block>>>(N, k, A);
                cudaDeviceSynchronize();
                lu_update_kernel<<<g, block>>>(N, k, A);
                cudaDeviceSynchronize();
            }
        }
    }, "Sync");

    // Strategy 2: Async
    printf("\n--- Strategy 2: Async ---\n");
    run_timed([&]() {
        for (int s = 0; s < STEPS; s++) {
            reset();
            for (int k = 0; k < N - 1; k++) {
                int rem = N - k - 1;
                dim3 g((rem+15)/16, (rem+15)/16);
                lu_step_kernel<<<g, block>>>(N, k, A);
                lu_update_kernel<<<g, block>>>(N, k, A);
                // Must sync between pivots (data dependency)
                cudaDeviceSynchronize();
            }
        }
    }, "Async");

    // Strategy 3: CUDA Graph
    printf("\n--- Strategy 3: CUDA Graph ---\n");
    printf("Note: LU has variable grid sizes per pivot => CUDA Graph captures fixed topology.\n");
    printf("We capture with max grid size; early pivots waste threads but overhead is amortized.\n");
    {
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));

        cudaEvent_t gi0, gi1;
        CHECK(cudaEventCreate(&gi0));
        CHECK(cudaEventCreate(&gi1));

        // For graph, use max grid for all pivots (simpler, captures correctly)
        dim3 max_grid((N+15)/16, (N+15)/16);

        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int k = 0; k < N - 1; k++) {
            lu_step_kernel<<<max_grid, block, 0, stream>>>(N, k, A);
            lu_update_kernel<<<max_grid, block, 0, stream>>>(N, k, A);
        }
        CHECK(cudaStreamEndCapture(stream, &graph));

        size_t numNodes;
        CHECK(cudaGraphGetNodes(graph, nullptr, &numNodes));
        printf("Graph nodes: %zu\n", numNodes);

        cudaEventRecord(gi0);
        CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
        cudaEventRecord(gi1);
        cudaEventSynchronize(gi1);
        float inst_ms;
        cudaEventElapsedTime(&inst_ms, gi0, gi1);
        printf("Graph instantiation: %.3f ms\n", inst_ms);

        run_timed([&]() {
            for (int s = 0; s < STEPS; s++) {
                reset();
                cudaGraphLaunch(graphExec, stream);
                cudaStreamSynchronize(stream);
            }
        }, "Graph");

        CHECK(cudaGraphExecDestroy(graphExec));
        CHECK(cudaGraphDestroy(graph));
        CHECK(cudaStreamDestroy(stream));
        CHECK(cudaEventDestroy(gi0));
        CHECK(cudaEventDestroy(gi1));
    }

    // Strategy 4: Persistent Kernel
    printf("\n--- Strategy 4: Persistent Kernel ---\n");
    {
        dim3 pk_grid((N+15)/16, (N+15)/16);
        int numBlocks;
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
              lu_persistent, block.x * block.y, 0));
        int maxBlocks = numBlocks * prop.multiProcessorCount;
        int needed = pk_grid.x * pk_grid.y;

        if (needed <= maxBlocks) {
            printf("Persistent: %d blocks (max %d)\n", needed, maxBlocks);
            void* args[] = {&N, &A};
            run_timed([&]() {
                for (int s = 0; s < STEPS; s++) {
                    reset();
                    cudaLaunchCooperativeKernel((void*)lu_persistent,
                        pk_grid, block, args);
                    cudaDeviceSynchronize();
                }
            }, "Persistent");
        } else {
            printf("Persistent: N/A (need %d blocks, max %d) — grid too large for cooperative launch\n",
                   needed, maxBlocks);
            printf("  LU requires N*N threads to cover full matrix; with N=%d, need %d blocks.\n",
                   N, needed);
        }
    }

    CHECK(cudaFree(A));
    return 0;
}
