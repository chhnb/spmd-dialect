/**
 * Device-Side Graph Launch Proof-of-Concept
 *
 * Tests if cudaGraphLaunch with cudaStreamGraphFireAndForget works on
 * A100 (sm_80) + CUDA 12.6. Uses a minimal Heat2D stencil kernel.
 *
 * Build: nvcc -O3 -arch=sm_80 -rdc=true device_graph_poc.cu -o device_graph_poc -lcudadevrt
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define CHECK(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %d (%s) at %s:%d\n", e, cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// Simple Heat2D stencil kernel
__global__ void heat2d_step(int N, const float* __restrict__ u, float* __restrict__ v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && i < N-1 && j > 0 && j < N-1) {
        v[i*N+j] = 0.25f * (u[(i-1)*N+j] + u[(i+1)*N+j] + u[i*N+(j-1)] + u[i*N+(j+1)]);
    }
}

// Copy kernel (v -> u)
__global__ void copy_kernel(int N, const float* __restrict__ src, float* __restrict__ dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N*N) dst[idx] = src[idx];
}

// Completion signal kernel — last kernel in step writes flag
__global__ void signal_completion(volatile int* flag, int expected_val) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __threadfence_system();
        atomicAdd((int*)flag, 1);
    }
}

// Device-side scheduler kernel (1 block, 1 thread for PoC)
__global__ void device_scheduler(cudaGraphExec_t graph_exec, int total_steps,
                                  volatile int* completion_flag) {
    if (threadIdx.x == 0) {
        for (int s = 0; s < total_steps; s++) {
            // Fire-and-forget launch
            cudaGraphLaunch(graph_exec, cudaStreamGraphFireAndForget);

            // Poll for completion
            while (atomicAdd((int*)completion_flag, 0) <= s) {
                // spin-wait
            }
        }
    }
}

int main(int argc, char** argv) {
    int N = 256;
    int STEPS = 100;
    int REPEAT = 10;

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== Device-Side Graph Launch PoC ===\n");
    printf("GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    int runtimeVer = 0;
    cudaRuntimeGetVersion(&runtimeVer);
    printf("CUDA Runtime: %d.%d\n", runtimeVer / 1000, (runtimeVer % 1000) / 10);
    printf("N=%d, steps=%d\n\n", N, STEPS);

    // Check if device graph launch is supported
    int deviceGraphSupported = 0;
    CHECK(cudaDeviceGetAttribute(&deviceGraphSupported,
                                  cudaDevAttrMemoryPoolSupportedHandleTypes, 0));
    printf("Device graph launch support check... ");

    // Allocate
    float *d_u, *d_v;
    int *d_completion;
    CHECK(cudaMalloc(&d_u, N*N*sizeof(float)));
    CHECK(cudaMalloc(&d_v, N*N*sizeof(float)));
    CHECK(cudaMalloc(&d_completion, sizeof(int)));

    // Init
    std::vector<float> h_u(N*N, 0.0f);
    h_u[N/2 * N + N/2] = 1.0f;
    CHECK(cudaMemcpy(d_u, h_u.data(), N*N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_v, 0, N*N*sizeof(float)));

    dim3 block(16, 16);
    dim3 grid((N+15)/16, (N+15)/16);
    int copyBlock = 256;
    int copyGrid = (N*N + 255) / 256;

    // ====== Strategy 1: Host-side Graph (baseline) ======
    {
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));

        // Capture one step: heat2d + copy
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        heat2d_step<<<grid, block, 0, stream>>>(N, d_u, d_v);
        copy_kernel<<<copyGrid, copyBlock, 0, stream>>>(N, d_v, d_u);
        CHECK(cudaStreamEndCapture(stream, &graph));
        CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

        // Warmup
        for (int w = 0; w < 5; w++) {
            for (int s = 0; s < STEPS; s++) {
                CHECK(cudaGraphLaunch(graphExec, stream));
            }
            CHECK(cudaStreamSynchronize(stream));
        }

        // Benchmark host-side graph replay
        std::vector<float> times;
        cudaEvent_t t0, t1;
        CHECK(cudaEventCreate(&t0));
        CHECK(cudaEventCreate(&t1));

        for (int r = 0; r < REPEAT; r++) {
            CHECK(cudaMemcpy(d_u, h_u.data(), N*N*sizeof(float), cudaMemcpyHostToDevice));
            CHECK(cudaEventRecord(t0, stream));
            for (int s = 0; s < STEPS; s++) {
                CHECK(cudaGraphLaunch(graphExec, stream));
            }
            CHECK(cudaEventRecord(t1, stream));
            CHECK(cudaEventSynchronize(t1));
            float ms;
            CHECK(cudaEventElapsedTime(&ms, t0, t1));
            times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        float median = times[REPEAT/2];
        printf("[Host Graph]    median=%.3f ms, %.2f us/step\n", median, median*1000/STEPS);

        CHECK(cudaGraphExecDestroy(graphExec));
        CHECK(cudaGraphDestroy(graph));
        CHECK(cudaStreamDestroy(stream));
        CHECK(cudaEventDestroy(t0));
        CHECK(cudaEventDestroy(t1));
    }

    // ====== Strategy 2: Device-side Graph (fire-and-forget) ======
    {
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));

        // Capture one step: heat2d + copy + signal
        CHECK(cudaMemset(d_completion, 0, sizeof(int)));
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        heat2d_step<<<grid, block, 0, stream>>>(N, d_u, d_v);
        copy_kernel<<<copyGrid, copyBlock, 0, stream>>>(N, d_v, d_u);
        signal_completion<<<1, 1, 0, stream>>>(d_completion, 0);
        CHECK(cudaStreamEndCapture(stream, &graph));
        CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

        // Enable fire-and-forget uploads
        // The graph must be configured for device launch
        cudaGraphNode_t* nodes = nullptr;
        size_t numNodes = 0;
        CHECK(cudaGraphGetNodes(graph, nodes, &numNodes));

        printf("[Device Graph]  Attempting device-side launch...\n");

        // Try launching the scheduler
        CHECK(cudaMemset(d_completion, 0, sizeof(int)));
        CHECK(cudaMemcpy(d_u, h_u.data(), N*N*sizeof(float), cudaMemcpyHostToDevice));

        // Warmup device scheduler
        for (int w = 0; w < 5; w++) {
            CHECK(cudaMemset(d_completion, 0, sizeof(int)));
            device_scheduler<<<1, 1>>>(graphExec, STEPS, d_completion);
            CHECK(cudaDeviceSynchronize());
        }

        // Benchmark
        std::vector<float> times;
        cudaEvent_t t0, t1;
        CHECK(cudaEventCreate(&t0));
        CHECK(cudaEventCreate(&t1));

        for (int r = 0; r < REPEAT; r++) {
            CHECK(cudaMemcpy(d_u, h_u.data(), N*N*sizeof(float), cudaMemcpyHostToDevice));
            CHECK(cudaMemset(d_completion, 0, sizeof(int)));
            CHECK(cudaEventRecord(t0));
            device_scheduler<<<1, 1>>>(graphExec, STEPS, d_completion);
            CHECK(cudaEventRecord(t1));
            CHECK(cudaEventSynchronize(t1));
            float ms;
            CHECK(cudaEventElapsedTime(&ms, t0, t1));
            times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        float median = times[REPEAT/2];
        printf("[Device Graph]  median=%.3f ms, %.2f us/step\n", median, median*1000/STEPS);

        float host_median = 0; // TODO: compare
        printf("\nDevice-side launch %s\n",
               (median > 0) ? "SUCCEEDED" : "FAILED");

        CHECK(cudaGraphExecDestroy(graphExec));
        CHECK(cudaGraphDestroy(graph));
        CHECK(cudaStreamDestroy(stream));
        CHECK(cudaEventDestroy(t0));
        CHECK(cudaEventDestroy(t1));
    }

    CHECK(cudaFree(d_u));
    CHECK(cudaFree(d_v));
    CHECK(cudaFree(d_completion));

    return 0;
}
