/**
 * Common CUDA 4-strategy benchmark harness.
 *
 * Each kernel file only needs to define:
 *   1. Step kernel(s) — normal __global__ functions
 *   2. Persistent kernel — __global__ with for-loop + grid.sync()
 *   3. A launch wrapper using the macros below
 *
 * Usage pattern in each *_cuda_4strategy.cu:
 *   #include "../common/cuda_harness.cuh"
 *   // define kernels...
 *   // define test function using BENCH_* helpers
 *   int main() { ... }
 */
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CHECK_CUDA(call) do { \
    cudaError_t e = (call); \
    if(e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %d (%s) at %s:%d\n", e, cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

struct BenchResult {
    float sync_us;
    float async_us;
    float graph_us;
    float persistent_us;  // -1 if not feasible
};

// Copy kernel (used by all stencil-like benchmarks)
__global__ void harness_copy_f(int N, const float* src, float* dst) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < N) dst[i] = src[i];
}

// Print one result line
inline void print_result(const char* name, BenchResult r) {
    char ps[32], sp[16];
    if (r.persistent_us < 0) {
        snprintf(ps, 32, "%10s", "N/A");
        snprintf(sp, 16, "%8s", "N/A");
    } else {
        snprintf(ps, 32, "%10.2f", r.persistent_us);
        snprintf(sp, 16, "%7.1fx", r.sync_us / r.persistent_us);
    }
    char sa[16], sg[16];
    snprintf(sa, 16, "%7.1fx", r.sync_us / r.async_us);
    snprintf(sg, 16, "%7.1fx", r.sync_us / r.graph_us);
    printf("%-32s %10.2f %10.2f %10.2f %s  | %8s %8s %8s\n",
           name, r.sync_us, r.async_us, r.graph_us, ps, sa, sg, sp);
}

inline void print_header() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SMs=%d, Compute %d.%d)\n\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    printf("%-32s %10s %10s %10s %10s  | Speedups over Sync\n",
           "Kernel", "Sync(us)", "Async(us)", "Graph(us)", "Persist(us)");
    printf("%-32s %10s %10s %10s %10s  | %8s %8s %8s\n",
           "", "", "", "", "", "Async", "Graph", "Persist");
    printf("%.32s %.10s %.10s %.10s %.10s  | %.8s %.8s %.8s\n",
           "--------------------------------", "----------", "----------",
           "----------", "----------", "--------", "--------", "--------");
}

/**
 * Generic 2D stencil benchmark: step_kernel(N, u, v) + copy_back
 * persist_kernel(N, u, v, STEPS) with grid.sync()
 *
 * Template parameters avoid function pointer overhead.
 */
template<typename StepFn, typename PersistFn>
BenchResult bench_2d(StepFn step_fn, PersistFn persist_fn,
                     int N, int STEPS, bool has_persist = true) {
    int N2 = N * N;
    float *u, *v;
    CHECK_CUDA(cudaMalloc(&u, N2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&v, N2 * sizeof(float)));
    // Init with small nonzero values
    float* h = (float*)malloc(N2 * sizeof(float));
    for (int i = 0; i < N2; i++) h[i] = 1.0f + 0.01f * (i % 100);
    cudaMemcpy(u, h, N2 * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
    CHECK_CUDA(cudaMemset(v, 0, N2 * sizeof(float)));

    dim3 block(16, 16), grid((N + 15) / 16, (N + 15) / 16);
    int cg_blocks = (N2 + 255) / 256;
    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));

    // Warmup
    for (int i = 0; i < 20; i++) {
        step_fn(N, u, v, grid, block, 0);
        harness_copy_f<<<cg_blocks, 256>>>(N2, v, u);
    }
    cudaDeviceSynchronize();

    BenchResult r;
    float ms;

    // Sync
    cudaEventRecord(t0);
    for (int s = 0; s < STEPS; s++) {
        step_fn(N, u, v, grid, block, 0);
        harness_copy_f<<<cg_blocks, 256>>>(N2, v, u);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    r.sync_us = ms * 1000.0f / STEPS;

    // Async
    cudaEventRecord(t0);
    for (int s = 0; s < STEPS; s++) {
        step_fn(N, u, v, grid, block, 0);
        harness_copy_f<<<cg_blocks, 256>>>(N2, v, u);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    r.async_us = ms * 1000.0f / STEPS;

    // Graph
    {
        int REPS = 5;
        cudaGraph_t g; cudaGraphExec_t ge;
        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));
        CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int s = 0; s < STEPS; s++) {
            step_fn(N, u, v, grid, block, stream);
            harness_copy_f<<<cg_blocks, 256, 0, stream>>>(N2, v, u);
        }
        CHECK_CUDA(cudaStreamEndCapture(stream, &g));
        CHECK_CUDA(cudaGraphInstantiate(&ge, g, NULL, NULL, 0));
        for (int i = 0; i < 3; i++) { cudaGraphLaunch(ge, stream); cudaStreamSynchronize(stream); }
        cudaEventRecord(t0, stream);
        for (int i = 0; i < REPS; i++) cudaGraphLaunch(ge, stream);
        cudaEventRecord(t1, stream); cudaStreamSynchronize(stream);
        cudaEventElapsedTime(&ms, t0, t1);
        r.graph_us = ms * 1000.0f / (STEPS * REPS);
        cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaStreamDestroy(stream);
    }

    // Persistent
    r.persistent_us = -1;
    if (has_persist) {
        int numBSm = 0;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm, (void*)persist_fn, 256, 0);
        int maxB = numBSm * prop.multiProcessorCount;
        if ((int)(grid.x * grid.y) <= maxB) {
            void* args[] = {(void*)&N, (void*)&u, (void*)&v, (void*)&STEPS};
            cudaLaunchCooperativeKernel((void*)persist_fn, grid, block, args);
            cudaDeviceSynchronize();
            int REPS = 5;
            cudaEventRecord(t0);
            for (int i = 0; i < REPS; i++)
                cudaLaunchCooperativeKernel((void*)persist_fn, grid, block, args);
            cudaEventRecord(t1); cudaDeviceSynchronize();
            cudaEventElapsedTime(&ms, t0, t1);
            r.persistent_us = ms * 1000.0f / (STEPS * REPS);
        }
    }

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(u); cudaFree(v);
    return r;
}

/**
 * Generic 1D benchmark: step_kernel(N, u, v) + copy_back
 */
template<typename StepFn, typename PersistFn>
BenchResult bench_1d(StepFn step_fn, PersistFn persist_fn,
                     int N, int STEPS, bool has_persist = true) {
    float *u, *v;
    CHECK_CUDA(cudaMalloc(&u, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&v, N * sizeof(float)));
    float* h = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h[i] = 1.0f + 0.01f * (i % 100);
    cudaMemcpy(u, h, N * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
    CHECK_CUDA(cudaMemset(v, 0, N * sizeof(float)));

    int blocks = (N + 255) / 256;
    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));

    for (int i = 0; i < 20; i++) {
        step_fn(N, u, v, dim3(blocks), dim3(256), 0);
        harness_copy_f<<<blocks, 256>>>(N, v, u);
    }
    cudaDeviceSynchronize();

    BenchResult r;
    float ms;

    // Sync
    cudaEventRecord(t0);
    for (int s = 0; s < STEPS; s++) {
        step_fn(N, u, v, dim3(blocks), dim3(256), 0);
        harness_copy_f<<<blocks, 256>>>(N, v, u);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    r.sync_us = ms * 1000.0f / STEPS;

    // Async
    cudaEventRecord(t0);
    for (int s = 0; s < STEPS; s++) {
        step_fn(N, u, v, dim3(blocks), dim3(256), 0);
        harness_copy_f<<<blocks, 256>>>(N, v, u);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    r.async_us = ms * 1000.0f / STEPS;

    // Graph
    {
        int REPS = 5;
        cudaGraph_t g; cudaGraphExec_t ge;
        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));
        CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int s = 0; s < STEPS; s++) {
            step_fn(N, u, v, dim3(blocks), dim3(256), stream);
            harness_copy_f<<<blocks, 256, 0, stream>>>(N, v, u);
        }
        CHECK_CUDA(cudaStreamEndCapture(stream, &g));
        CHECK_CUDA(cudaGraphInstantiate(&ge, g, NULL, NULL, 0));
        for (int i = 0; i < 3; i++) { cudaGraphLaunch(ge, stream); cudaStreamSynchronize(stream); }
        cudaEventRecord(t0, stream);
        for (int i = 0; i < REPS; i++) cudaGraphLaunch(ge, stream);
        cudaEventRecord(t1, stream); cudaStreamSynchronize(stream);
        cudaEventElapsedTime(&ms, t0, t1);
        r.graph_us = ms * 1000.0f / (STEPS * REPS);
        cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaStreamDestroy(stream);
    }

    // Persistent
    r.persistent_us = -1;
    if (has_persist) {
        int numBSm = 0;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm, (void*)persist_fn, 256, 0);
        int maxB = numBSm * prop.multiProcessorCount;
        if (blocks <= maxB) {
            void* args[] = {(void*)&N, (void*)&u, (void*)&v, (void*)&STEPS};
            cudaLaunchCooperativeKernel((void*)persist_fn, dim3(blocks), dim3(256), args);
            cudaDeviceSynchronize();
            int REPS = 5;
            cudaEventRecord(t0);
            for (int i = 0; i < REPS; i++)
                cudaLaunchCooperativeKernel((void*)persist_fn, dim3(blocks), dim3(256), args);
            cudaEventRecord(t1); cudaDeviceSynchronize();
            cudaEventElapsedTime(&ms, t0, t1);
            r.persistent_us = ms * 1000.0f / (STEPS * REPS);
        }
    }

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(u); cudaFree(v);
    return r;
}
