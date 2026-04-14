/**
 * Device-Side Graph Launch PoC — Self-restarting tail launch
 *
 * Uses __device__ global variable to store the graph exec handle,
 * avoiding fragile cudaGraphExecKernelNodeSetParams.
 *
 * Pattern: capture step graph with tail_launcher as last node.
 * tail_launcher reads the graph exec from device memory and
 * tail-launches it if more steps remain.
 *
 * Build: nvcc -O3 -arch=sm_80 -rdc=true device_graph_poc.cu -o device_graph_poc -lcudadevrt
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %d (%s) at %s:%d\n", \
                e, cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// Device-visible storage for the graph exec handle
__device__ cudaGraphExec_t d_graph_exec;

__global__ void heat2d_step(int N, const float* __restrict__ u, float* __restrict__ v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && i < N-1 && j > 0 && j < N-1) {
        v[i*N+j] = 0.25f * (u[(i-1)*N+j] + u[(i+1)*N+j] + u[i*N+(j-1)] + u[i*N+(j+1)]);
    }
}

__global__ void copy_field(int N2, const float* __restrict__ src, float* __restrict__ dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N2) dst[idx] = src[idx];
}

// Last kernel in step graph: tail-launches self if steps remain
__global__ void tail_launcher(int* steps_remaining) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int rem = atomicSub(steps_remaining, 1);
        if (rem > 1) {
            // Tail launch: replaces current graph execution, guaranteeing
            // the current step completes before the next one starts
            cudaGraphLaunch(d_graph_exec, cudaStreamGraphTailLaunch);
        }
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 256;
    int STEPS = (argc > 2) ? atoi(argv[2]) : 100;
    int REPEAT = (argc > 3) ? atoi(argv[3]) : 10;

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int rtVer = 0; cudaRuntimeGetVersion(&rtVer);
    printf("=== Device-Side Graph Launch PoC (Tail Launch v2) ===\n");
    printf("GPU: %s (sm_%d%d), CUDA %d.%d\n",
           prop.name, prop.major, prop.minor, rtVer/1000, (rtVer%1000)/10);
    printf("N=%d, steps=%d, repeat=%d\n\n", N, STEPS, REPEAT);

    float *d_u, *d_v;
    int *d_steps;
    CHECK(cudaMalloc(&d_u, N*N*sizeof(float)));
    CHECK(cudaMalloc(&d_v, N*N*sizeof(float)));
    CHECK(cudaMalloc(&d_steps, sizeof(int)));

    std::vector<float> h_u(N*N, 0.0f);
    h_u[N/2*N + N/2] = 0.01f;

    auto reset_data = [&]() {
        CHECK(cudaMemcpy(d_u, h_u.data(), N*N*sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemset(d_v, 0, N*N*sizeof(float)));
    };
    auto reset_steps = [&]() {
        CHECK(cudaMemcpy(d_steps, &STEPS, sizeof(int), cudaMemcpyHostToDevice));
    };

    dim3 block(16, 16), grid((N+15)/16, (N+15)/16);
    int cpBlk = 256, cpGrd = (N*N+255)/256;

    // ====== Sync baseline (for reference) ======
    float sync_median = 0;
    {
        cudaEvent_t t0, t1;
        CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
        std::vector<float> times;
        for (int w = 0; w < 5; w++) {
            reset_data();
            for (int s = 0; s < STEPS; s++) {
                heat2d_step<<<grid, block>>>(N, d_u, d_v);
                copy_field<<<cpGrd, cpBlk>>>(N*N, d_v, d_u);
                CHECK(cudaDeviceSynchronize());
            }
        }
        for (int r = 0; r < REPEAT; r++) {
            reset_data();
            CHECK(cudaEventRecord(t0));
            for (int s = 0; s < STEPS; s++) {
                heat2d_step<<<grid, block>>>(N, d_u, d_v);
                copy_field<<<cpGrd, cpBlk>>>(N*N, d_v, d_u);
                CHECK(cudaDeviceSynchronize());
            }
            CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1));
            float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1));
            times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        sync_median = times[REPEAT/2];
        printf("[Sync]           median=%.3f ms, %.2f us/step\n",
               sync_median, sync_median*1000.0f/STEPS);
        CHECK(cudaEventDestroy(t0)); CHECK(cudaEventDestroy(t1));
    }

    // ====== Host-side Graph ======
    float host_median = 0;
    std::vector<float> h_host_out(N*N);
    {
        cudaGraph_t g; cudaGraphExec_t ge; cudaStream_t s;
        CHECK(cudaStreamCreate(&s));
        CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        heat2d_step<<<grid, block, 0, s>>>(N, d_u, d_v);
        copy_field<<<cpGrd, cpBlk, 0, s>>>(N*N, d_v, d_u);
        CHECK(cudaStreamEndCapture(s, &g));
        CHECK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));

        for (int w = 0; w < 5; w++) {
            reset_data();
            for (int st = 0; st < STEPS; st++) CHECK(cudaGraphLaunch(ge, s));
            CHECK(cudaStreamSynchronize(s));
        }
        cudaEvent_t t0, t1;
        CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
        std::vector<float> times;
        for (int r = 0; r < REPEAT; r++) {
            reset_data();
            CHECK(cudaEventRecord(t0, s));
            for (int st = 0; st < STEPS; st++) CHECK(cudaGraphLaunch(ge, s));
            CHECK(cudaEventRecord(t1, s));
            CHECK(cudaEventSynchronize(t1));
            float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1));
            times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        host_median = times[REPEAT/2];
        printf("[Host Graph]     median=%.3f ms, %.2f us/step\n",
               host_median, host_median*1000.0f/STEPS);

        // Save reference output
        reset_data();
        for (int st = 0; st < STEPS; st++) CHECK(cudaGraphLaunch(ge, s));
        CHECK(cudaStreamSynchronize(s));
        CHECK(cudaMemcpy(h_host_out.data(), d_u, N*N*sizeof(float), cudaMemcpyDeviceToHost));

        CHECK(cudaGraphExecDestroy(ge)); CHECK(cudaGraphDestroy(g));
        CHECK(cudaStreamDestroy(s));
        CHECK(cudaEventDestroy(t0)); CHECK(cudaEventDestroy(t1));
    }

    // ====== Device-side Graph (tail launch) ======
    float device_median = 0;
    {
        cudaGraph_t g; cudaGraphExec_t ge; cudaStream_t s;
        CHECK(cudaStreamCreate(&s));

        // Capture: stencil + copy + tail_launcher
        CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        heat2d_step<<<grid, block, 0, s>>>(N, d_u, d_v);
        copy_field<<<cpGrd, cpBlk, 0, s>>>(N*N, d_v, d_u);
        tail_launcher<<<1, 1, 0, s>>>(d_steps);
        CHECK(cudaStreamEndCapture(s, &g));
        // Instantiate with device-launch flag (required for device-side cudaGraphLaunch)
        CHECK(cudaGraphInstantiateWithFlags(&ge, g, cudaGraphInstantiateFlagDeviceLaunch));
        // Upload graph to device for device-side launch
        CHECK(cudaGraphUpload(ge, s));

        // Store graph exec handle in device memory so tail_launcher can read it
        CHECK(cudaMemcpyToSymbol(d_graph_exec, &ge, sizeof(cudaGraphExec_t)));

        // Warmup
        for (int w = 0; w < 5; w++) {
            reset_data(); reset_steps();
            CHECK(cudaGraphLaunch(ge, s));  // One host launch; GPU chains the rest
            CHECK(cudaStreamSynchronize(s));
        }

        // Timed runs
        cudaEvent_t t0, t1;
        CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
        std::vector<float> times;
        for (int r = 0; r < REPEAT; r++) {
            reset_data(); reset_steps();
            CHECK(cudaEventRecord(t0, s));
            CHECK(cudaGraphLaunch(ge, s));
            CHECK(cudaEventRecord(t1, s));
            CHECK(cudaEventSynchronize(t1));
            float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1));
            times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        device_median = times[REPEAT/2];
        printf("[Device Graph]   median=%.3f ms, %.2f us/step\n",
               device_median, device_median*1000.0f/STEPS);

        // Verify output
        reset_data(); reset_steps();
        CHECK(cudaGraphLaunch(ge, s));
        CHECK(cudaStreamSynchronize(s));
        std::vector<float> h_dev_out(N*N);
        CHECK(cudaMemcpy(h_dev_out.data(), d_u, N*N*sizeof(float), cudaMemcpyDeviceToHost));
        float max_diff = 0, max_val = 0;
        for (int i = 0; i < N*N; i++) {
            float d = fabsf(h_host_out[i] - h_dev_out[i]);
            if (d > max_diff) max_diff = d;
            if (fabsf(h_host_out[i]) > max_val) max_val = fabsf(h_host_out[i]);
        }
        float rel_err = max_val > 0 ? max_diff / max_val : max_diff;
        printf("\n--- Output Verification ---\n");
        printf("max_abs_diff=%.2e, max_field=%.2e, rel_err=%.2e\n", max_diff, max_val, rel_err);
        printf("Correctness: %s\n", rel_err < 0.05f ? "MATCH" : "MISMATCH");

        CHECK(cudaGraphExecDestroy(ge)); CHECK(cudaGraphDestroy(g));
        CHECK(cudaStreamDestroy(s));
        CHECK(cudaEventDestroy(t0)); CHECK(cudaEventDestroy(t1));
    }

    // Summary
    printf("\n--- Summary ---\n");
    printf("Sync:         %.2f us/step\n", sync_median*1000.0f/STEPS);
    printf("Host Graph:   %.2f us/step\n", host_median*1000.0f/STEPS);
    printf("Device Graph: %.2f us/step\n", device_median*1000.0f/STEPS);
    printf("Speedup (Device vs Host Graph): %.2fx\n",
           device_median > 0 ? host_median / device_median : 0);
    printf("Speedup (Device vs Sync):       %.2fx\n",
           device_median > 0 ? sync_median / device_median : 0);

    CHECK(cudaFree(d_u)); CHECK(cudaFree(d_v)); CHECK(cudaFree(d_steps));
    return 0;
}
