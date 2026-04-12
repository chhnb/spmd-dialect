/**
 * C4: Wave 2D equation — 4-strategy benchmark.
 * u_new = 2*u - u_old + c^2*dt^2*laplacian(u)
 * Strategies: Sync loop, Async loop, CUDA Graph, Persistent Kernel.
 * Build: nvcc -O3 -arch=sm_80 -rdc=true wave2d_benchmark.cu -o wave2d_bench -lcudadevrt
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

static const float C = 1.0f;
static const float DT = 0.1f;
static const float DX = 1.0f;
static const float C2DT2 = C * C * DT * DT / (DX * DX);

// 1 kernel/step: compute u_new from u and u_old, then caller swaps pointers
__global__ void wave2d_step(int N, const float* __restrict__ u,
                            const float* __restrict__ u_old,
                            float* __restrict__ u_new) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
        int idx = i * N + j;
        float lap = u[idx-N] + u[idx+N] + u[idx-1] + u[idx+1] - 4.0f * u[idx];
        u_new[idx] = 2.0f * u[idx] - u_old[idx] + C2DT2 * lap;
    }
}

// Persistent: 1D thread mapping, handles pointer rotation internally
__global__ void wave2d_persistent(int N, int N2, float* buf0, float* buf1, float* buf2, int STEPS) {
    // 1D indexing to control block count for cooperative launch
    float* u_old_p = buf0;
    float* u_p = buf1;
    float* u_new_p = buf2;

    for (int s = 0; s < STEPS; s++) {
        for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N2;
             tid += gridDim.x * blockDim.x) {
            int i = tid / N;
            int j = tid % N;
            if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
                float lap = u_p[tid-N] + u_p[tid+N] + u_p[tid-1] + u_p[tid+1] - 4.0f * u_p[tid];
                u_new_p[tid] = 2.0f * u_p[tid] - u_old_p[tid] + C2DT2 * lap;
            }
        }
        cg::this_grid().sync();
        // rotate: u_old <- u, u <- u_new, u_new <- u_old
        float* tmp = u_old_p;
        u_old_p = u_p;
        u_p = u_new_p;
        u_new_p = tmp;
        cg::this_grid().sync();
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 512;
    int STEPS = (argc > 2) ? atoi(argv[2]) : 100;
    int REPEAT = (argc > 3) ? atoi(argv[3]) : 10;
    int N2 = N * N;

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== C4: Wave 2D Benchmark ===\n");
    printf("GPU: %s, SMs=%d\n", prop.name, prop.multiProcessorCount);
    printf("N=%d (%d cells), steps=%d, repeat=%d\n", N, N2, STEPS, REPEAT);
    printf("c=%.1f, dt=%.2f, c2dt2/dx2=%.4f\n", C, DT, C2DT2);

    float *d_u_old, *d_u, *d_u_new;
    CHECK(cudaMalloc(&d_u_old, N2 * sizeof(float)));
    CHECK(cudaMalloc(&d_u, N2 * sizeof(float)));
    CHECK(cudaMalloc(&d_u_new, N2 * sizeof(float)));

    // Init: Gaussian pulse in center
    std::vector<float> h_u(N2, 0.0f);
    int cx = N / 2, cy = N / 2;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float dx = (float)(i - cx), dy = (float)(j - cy);
            h_u[i * N + j] = expf(-(dx*dx + dy*dy) / (2.0f * 100.0f));
        }

    auto reset = [&]() {
        CHECK(cudaMemcpy(d_u_old, h_u.data(), N2 * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_u, h_u.data(), N2 * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemset(d_u_new, 0, N2 * sizeof(float)));
    };
    reset();

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    // Warmup
    for (int i = 0; i < 5; i++) {
        wave2d_step<<<grid, block>>>(N, d_u, d_u_old, d_u_new);
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

    // Strategy 1: Sync Loop (1 kernel/step + pointer swap)
    printf("\n--- Strategy 1: Sync Loop ---\n");
    run_timed([&]() {
        float *uo = d_u_old, *uc = d_u, *un = d_u_new;
        for (int s = 0; s < STEPS; s++) {
            wave2d_step<<<grid, block>>>(N, uc, uo, un);
            cudaDeviceSynchronize();
            float* tmp = uo; uo = uc; uc = un; un = tmp;
        }
    }, "Sync Loop");

    // Strategy 2: Async Loop
    // Note: with pointer rotation we must unroll or use a fixed pattern.
    // Since we swap 3 buffers, pattern repeats every 3 steps.
    printf("\n--- Strategy 2: Async Loop ---\n");
    run_timed([&]() {
        float *uo = d_u_old, *uc = d_u, *un = d_u_new;
        for (int s = 0; s < STEPS; s++) {
            wave2d_step<<<grid, block>>>(N, uc, uo, un);
            float* tmp = uo; uo = uc; uc = un; un = tmp;
        }
        cudaDeviceSynchronize();
    }, "Async Loop");

    // Strategy 3: CUDA Graph
    // Capture 3 steps (one full rotation cycle), replay STEPS/3 times
    printf("\n--- Strategy 3: CUDA Graph ---\n");
    {
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));

        // Capture 3 steps = one full pointer rotation cycle
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        wave2d_step<<<grid, block, 0, stream>>>(N, d_u, d_u_old, d_u_new);       // step 0: old=buf0,cur=buf1,new=buf2
        wave2d_step<<<grid, block, 0, stream>>>(N, d_u_new, d_u, d_u_old);       // step 1: old=buf1,cur=buf2,new=buf0
        wave2d_step<<<grid, block, 0, stream>>>(N, d_u_old, d_u_new, d_u);       // step 2: old=buf2,cur=buf0,new=buf1
        CHECK(cudaStreamEndCapture(stream, &graph));
        CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

        int tripleSteps = STEPS / 3;
        int remainder = STEPS % 3;

        run_timed([&]() {
            for (int t = 0; t < tripleSteps; t++) {
                cudaGraphLaunch(graphExec, stream);
            }
            // remainder steps
            float *uo = d_u_old, *uc = d_u, *un = d_u_new;
            for (int s = 0; s < remainder; s++) {
                wave2d_step<<<grid, block, 0, stream>>>(N, uc, uo, un);
                float* tmp = uo; uo = uc; uc = un; un = tmp;
            }
            cudaStreamSynchronize(stream);
        }, "CUDA Graph");

        CHECK(cudaGraphExecDestroy(graphExec));
        CHECK(cudaGraphDestroy(graph));
        CHECK(cudaStreamDestroy(stream));
    }

    // Strategy 4: Persistent Kernel (1D thread mapping)
    printf("\n--- Strategy 4: Persistent Kernel ---\n");
    {
        int pBlockSize = 256;
        int numBlocks;
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
              wave2d_persistent, pBlockSize, 0));
        int maxBlocks = numBlocks * prop.multiProcessorCount;
        int needed = (N2 + pBlockSize - 1) / pBlockSize;
        // Use min(needed, maxBlocks) for grid-stride loop
        int pGridSize = (needed <= maxBlocks) ? needed : maxBlocks;

        printf("Persistent: %d blocks (need %d, max %d, launch %d)\n",
               pGridSize, needed, maxBlocks, pGridSize);
        void* args[] = {(void*)&N, (void*)&N2, &d_u_old, &d_u, &d_u_new, &STEPS};
        run_timed([&]() {
            cudaLaunchCooperativeKernel((void*)wave2d_persistent,
                dim3(pGridSize), dim3(pBlockSize), args);
            cudaDeviceSynchronize();
        }, "Persistent");
    }

    printf("\n=== CSV: wave2d,%d,%d,A100 ===\n", N, STEPS);

    CHECK(cudaFree(d_u_old));
    CHECK(cudaFree(d_u));
    CHECK(cudaFree(d_u_new));
    return 0;
}
