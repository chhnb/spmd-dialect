/**
 * Conditional Graph Loop PoC — GPU-side while loop via CUDA 12.4+ conditional nodes
 *
 * One host launch → GPU internally loops N times → done.
 * Zero per-step host involvement, no grid size limit.
 *
 * Build: nvcc -O3 -arch=sm_80 -rdc=true conditional_graph_poc.cu -o conditional_graph_poc -lcudadevrt
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %d (%s) at %s:%d\n", \
                e, cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

__global__ void heat2d_step(int N, const float* __restrict__ u, float* __restrict__ v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && i < N-1 && j > 0 && j < N-1) {
        int idx = i*N+j;
        v[idx] = u[idx] + 0.2f*(u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-4*u[idx]);
    }
}

__global__ void copy_field(int N2, const float* __restrict__ src, float* __restrict__ dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N2) dst[idx] = src[idx];
}

__global__ void decrement_and_check(int* counter, cudaGraphConditionalHandle handle) {
    if (threadIdx.x == 0) {
        int rem = atomicSub(counter, 1);
        cudaGraphSetConditional(handle, (rem > 1) ? 1 : 0);
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 256;
    int STEPS = (argc > 2) ? atoi(argv[2]) : 100;
    int REPEAT = (argc > 3) ? atoi(argv[3]) : 10;

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int rtVer = 0; cudaRuntimeGetVersion(&rtVer);
    printf("=== Conditional Graph Loop PoC ===\n");
    printf("GPU: %s (sm_%d%d), CUDA %d.%d\n",
           prop.name, prop.major, prop.minor, rtVer/1000, (rtVer%1000)/10);
    printf("N=%d, steps=%d, repeat=%d\n\n", N, STEPS, REPEAT);

    float *d_u, *d_v;
    int *d_counter;
    CHECK(cudaMalloc(&d_u, N*N*sizeof(float)));
    CHECK(cudaMalloc(&d_v, N*N*sizeof(float)));
    CHECK(cudaMalloc(&d_counter, sizeof(int)));

    std::vector<float> h_u(N*N, 0.0f);
    h_u[N/2*N + N/2] = 0.01f;

    auto reset = [&]() {
        CHECK(cudaMemcpy(d_u, h_u.data(), N*N*sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemset(d_v, 0, N*N*sizeof(float)));
    };

    dim3 block(16, 16), grid((N+15)/16, (N+15)/16);
    int cpBlk = 256, cpGrd = (N*N+255)/256;

    // ====== Reference: Host Graph (full capture) ======
    float fullgraph_med = 0;
    std::vector<float> h_ref(N*N);
    {
        cudaStream_t s; CHECK(cudaStreamCreate(&s));
        cudaGraph_t g; cudaGraphExec_t ge;
        reset();
        CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        for (int st = 0; st < STEPS; st++) {
            heat2d_step<<<grid, block, 0, s>>>(N, d_u, d_v);
            copy_field<<<cpGrd, cpBlk, 0, s>>>(N*N, d_v, d_u);
        }
        CHECK(cudaStreamEndCapture(s, &g));
        CHECK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        for (int w = 0; w < 5; w++) { reset(); cudaGraphLaunch(ge, s); cudaStreamSynchronize(s); }
        cudaEvent_t t0, t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
        std::vector<float> times;
        for (int r = 0; r < REPEAT; r++) {
            reset();
            CHECK(cudaEventRecord(t0, s)); CHECK(cudaGraphLaunch(ge, s));
            CHECK(cudaEventRecord(t1, s)); CHECK(cudaEventSynchronize(t1));
            float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1)); times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        fullgraph_med = times[REPEAT/2];
        printf("[Graph Full]        %.3f ms = %.2f us/step\n", fullgraph_med, fullgraph_med*1000.f/STEPS);
        reset(); cudaGraphLaunch(ge, s); cudaStreamSynchronize(s);
        CHECK(cudaMemcpy(h_ref.data(), d_u, N*N*sizeof(float), cudaMemcpyDeviceToHost));
        cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaStreamDestroy(s);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }

    // ====== Per-step host graph replay (fair baseline) ======
    float perstep_med = 0;
    {
        cudaStream_t s; CHECK(cudaStreamCreate(&s));
        cudaGraph_t g; cudaGraphExec_t ge;
        CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        heat2d_step<<<grid, block, 0, s>>>(N, d_u, d_v);
        copy_field<<<cpGrd, cpBlk, 0, s>>>(N*N, d_v, d_u);
        CHECK(cudaStreamEndCapture(s, &g));
        CHECK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        for (int w = 0; w < 5; w++) {
            reset(); for (int st = 0; st < STEPS; st++) cudaGraphLaunch(ge, s);
            cudaStreamSynchronize(s);
        }
        cudaEvent_t t0, t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
        std::vector<float> times;
        for (int r = 0; r < REPEAT; r++) {
            reset(); CHECK(cudaEventRecord(t0, s));
            for (int st = 0; st < STEPS; st++) CHECK(cudaGraphLaunch(ge, s));
            CHECK(cudaEventRecord(t1, s)); CHECK(cudaEventSynchronize(t1));
            float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1)); times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        perstep_med = times[REPEAT/2];
        printf("[Graph Per-Step]    %.3f ms = %.2f us/step\n", perstep_med, perstep_med*1000.f/STEPS);
        cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaStreamDestroy(s);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }

    // ====== Conditional Graph While Loop ======
    float condloop_med = 0;
    {
        cudaStream_t s; CHECK(cudaStreamCreate(&s));

        // Step 1: Create main graph with a while-conditional node
        cudaGraph_t mainGraph;
        CHECK(cudaGraphCreate(&mainGraph, 0));

        // Create conditional handle (default=1 means "enter loop")
        cudaGraphConditionalHandle whileHandle;
        CHECK(cudaGraphConditionalHandleCreate(&whileHandle, mainGraph,
                                                1, cudaGraphCondAssignDefault));

        // Add conditional while node
        cudaGraph_t bodyGraph = nullptr;
        cudaConditionalNodeParams cp;
        memset(&cp, 0, sizeof(cp));
        cp.handle = whileHandle;
        cp.type = cudaGraphCondTypeWhile;
        cp.size = 1;
        cp.phGraph_out = &bodyGraph;

        // Must zero-init all reserved fields
        char npbuf[sizeof(cudaGraphNodeParams)];
        memset(npbuf, 0, sizeof(npbuf));
        cudaGraphNodeParams* np = (cudaGraphNodeParams*)npbuf;
        np->type = cudaGraphNodeTypeConditional;
        np->conditional = cp;

        cudaGraphNode_t whileNode;
        cudaError_t addErr = cudaGraphAddNode(&whileNode, mainGraph, nullptr, 0, np);
        if (addErr != cudaSuccess) {
            printf("[Cond While Loop]   NOT SUPPORTED: cudaGraphAddNode returned %d (%s)\n",
                   addErr, cudaGetErrorString(addErr));
            printf("  Conditional graph nodes may require sm_90+ (H100)\n");
            condloop_med = -1;
            cudaGraphDestroy(mainGraph); cudaStreamDestroy(s);
        } else {

        // phGraph_out is written by cudaGraphAddNode into the node params
        bodyGraph = np->conditional.phGraph_out[0];
        printf("  bodyGraph = %p (from np->conditional.phGraph_out)\n", (void*)bodyGraph);

        // bodyGraph is now set by cudaGraphAddNode (via phGraph_out)
        // Step 2: Populate body graph using stream capture
        CHECK(cudaStreamBeginCaptureToGraph(s, bodyGraph, nullptr, nullptr, 0,
                                             cudaStreamCaptureModeGlobal));
        heat2d_step<<<grid, block, 0, s>>>(N, d_u, d_v);
        copy_field<<<cpGrd, cpBlk, 0, s>>>(N*N, d_v, d_u);
        decrement_and_check<<<1, 1, 0, s>>>(d_counter, whileHandle);
        CHECK(cudaStreamEndCapture(s, &bodyGraph));

        // Step 3: Instantiate
        cudaGraphExec_t ge;
        // Conditional nodes require device launch capability
        // Try without device launch flag first (conditional nodes may not need it)
        cudaError_t instErr = cudaGraphInstantiateWithFlags(&ge, mainGraph, 0);
        if (instErr != cudaSuccess) {
            printf("[Cond While Loop]   Instantiation failed: %s\n", cudaGetErrorString(instErr));
            // Try with device launch flag
            instErr = cudaGraphInstantiateWithFlags(&ge, mainGraph, cudaGraphInstantiateFlagDeviceLaunch);
            if (instErr != cudaSuccess) {
                printf("[Cond While Loop]   Also failed with device flag: %s\n", cudaGetErrorString(instErr));
                condloop_med = -1;
                cudaGraphDestroy(mainGraph); cudaStreamDestroy(s);
            } else {
                CHECK(cudaGraphUpload(ge, s));
            }
        }
        if (instErr == cudaSuccess) {

        // Warmup
        for (int w = 0; w < 5; w++) {
            reset();
            int sv = STEPS; CHECK(cudaMemcpy(d_counter, &sv, sizeof(int), cudaMemcpyHostToDevice));
            CHECK(cudaGraphLaunch(ge, s)); CHECK(cudaStreamSynchronize(s));
        }

        // Timed runs
        cudaEvent_t t0, t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
        std::vector<float> times;
        for (int r = 0; r < REPEAT; r++) {
            reset();
            int sv = STEPS; CHECK(cudaMemcpy(d_counter, &sv, sizeof(int), cudaMemcpyHostToDevice));
            CHECK(cudaEventRecord(t0, s));
            CHECK(cudaGraphLaunch(ge, s));
            CHECK(cudaEventRecord(t1, s)); CHECK(cudaEventSynchronize(t1));
            float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1)); times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        condloop_med = times[REPEAT/2];
        printf("[Cond While Loop]   %.3f ms = %.2f us/step\n", condloop_med, condloop_med*1000.f/STEPS);

        // Verify correctness
        reset();
        int sv = STEPS; CHECK(cudaMemcpy(d_counter, &sv, sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaGraphLaunch(ge, s)); CHECK(cudaStreamSynchronize(s));
        std::vector<float> h_out(N*N);
        CHECK(cudaMemcpy(h_out.data(), d_u, N*N*sizeof(float), cudaMemcpyDeviceToHost));
        float maxdiff = 0;
        for (int i = 0; i < N*N; i++) {
            float d = fabsf(h_ref[i] - h_out[i]);
            if (d > maxdiff) maxdiff = d;
        }
        printf("  Correctness: max_diff=%.2e (%s)\n", maxdiff, maxdiff < 1e-5f ? "MATCH" : "MISMATCH");

        cudaGraphExecDestroy(ge); cudaGraphDestroy(mainGraph);
        cudaStreamDestroy(s); cudaEventDestroy(t0); cudaEventDestroy(t1);
        } // end if (instErr == cudaSuccess)
        } // end else (cudaGraphAddNode succeeded)
    }

    // Summary
    printf("\n--- N=%d, %d steps ---\n", N, STEPS);
    printf("Graph Full:       %.2f us/step (optimal: entire loop in one graph)\n", fullgraph_med*1000.f/STEPS);
    printf("Graph Per-Step:   %.2f us/step (fair baseline: per-step host replay)\n", perstep_med*1000.f/STEPS);
    printf("Cond While Loop:  %.2f us/step (one launch, GPU loops internally)\n", condloop_med*1000.f/STEPS);
    if (condloop_med > 0 && perstep_med > 0)
        printf("\nCond Loop vs Per-Step: %.2fx\n", perstep_med / condloop_med);
    if (condloop_med > 0 && fullgraph_med > 0)
        printf("Cond Loop vs Full Graph: %.2fx\n", fullgraph_med / condloop_med);

    cudaFree(d_u); cudaFree(d_v); cudaFree(d_counter);
    return 0;
}
