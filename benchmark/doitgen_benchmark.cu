/**
 * C18: DOITGEN multi-dimensional contraction (PolyBench-style) — 4-strategy benchmark.
 * Host launches one kernel per r-index: NR launches per step.
 * A(p,q,r) = sum_s A(p,q,s) * C4(s,r)
 * Build: nvcc -O3 -arch=sm_80 -rdc=true doitgen_benchmark.cu -o doitgen_bench -lcudadevrt
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

// Kernel: for a given r, compute A_out(p,q,r) = sum_s A_in(p,q,s) * C4(s,r)
__global__ void doitgen_slice_kernel(int NP, int NQ, int NR,
                                      int r,
                                      const float* __restrict__ A,
                                      const float* __restrict__ C4,
                                      float* __restrict__ A_out) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int q = blockIdx.y * blockDim.y + threadIdx.y;
    if (p < NP && q < NQ) {
        float sum = 0.0f;
        for (int s = 0; s < NR; s++) {
            sum += A[p * NQ * NR + q * NR + s] * C4[s * NR + r];
        }
        A_out[p * NQ * NR + q * NR + r] = sum;
    }
}

// Persistent: process all r-indices in one kernel
// All r-indices are independent (read A, write A_out) — no barrier needed
__global__ void doitgen_persistent(int NP, int NQ, int NR,
                                    const float* __restrict__ A,
                                    const float* __restrict__ C4,
                                    float* __restrict__ A_out) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int q = blockIdx.y * blockDim.y + threadIdx.y;

    for (int r = 0; r < NR; r++) {
        if (p < NP && q < NQ) {
            float sum = 0.0f;
            for (int s = 0; s < NR; s++) {
                sum += A[p * NQ * NR + q * NR + s] * C4[s * NR + r];
            }
            A_out[p * NQ * NR + q * NR + r] = sum;
        }
    }
}

// Grid-stride persistent: flatten all (p,q,r), 0 barrier per step
// All r-indices are independent (read A, write A_out), no sync needed
__global__ void doitgen_persistent_stride(int NP, int NQ, int NR,
                                           const float* __restrict__ A,
                                           const float* __restrict__ C4,
                                           float* __restrict__ A_out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int total = NP * NQ * NR;

    for (int idx = tid; idx < total; idx += stride) {
        int tmp = idx;
        int r = tmp % NR; tmp /= NR;
        int q = tmp % NQ;
        int p = tmp / NQ;
        float sum = 0.0f;
        for (int s = 0; s < NR; s++)
            sum += A[p * NQ * NR + q * NR + s] * C4[s * NR + r];
        A_out[p * NQ * NR + q * NR + r] = sum;
    }
}

int main(int argc, char* argv[]) {
    int NP = (argc > 1) ? atoi(argv[1]) : 128;
    int NQ = NP, NR = NP;
    int STEPS = (argc > 2) ? atoi(argv[2]) : 3;
    int REPEAT = (argc > 3) ? atoi(argv[3]) : 10;
    long N3 = (long)NP * NQ * NR;
    int launches_per_step = NR;

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== C18: DOITGEN Benchmark ===\n");
    printf("GPU: %s, SMs=%d\n", prop.name, prop.multiProcessorCount);
    printf("NP=NQ=NR=%d (%ld elements), steps=%d, repeat=%d\n", NP, N3, STEPS, REPEAT);
    printf("Launches/step: %d\n", launches_per_step);

    float *A, *A_out, *C4;
    CHECK(cudaMalloc(&A, N3 * sizeof(float)));
    CHECK(cudaMalloc(&A_out, N3 * sizeof(float)));
    CHECK(cudaMalloc(&C4, (long)NR * NR * sizeof(float)));

    std::vector<float> h_A(N3), h_C4((long)NR * NR);
    for (long i = 0; i < N3; i++) h_A[i] = sinf((float)i * 0.001f);
    for (long i = 0; i < (long)NR*NR; i++) h_C4[i] = cosf((float)i * 0.002f) / NR;  // scaled for stability

    auto reset = [&]() {
        CHECK(cudaMemcpy(A, h_A.data(), N3*sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(C4, h_C4.data(), (long)NR*NR*sizeof(float), cudaMemcpyHostToDevice));
    };
    reset();

    dim3 block(16, 16);
    dim3 grid2d((NP + 15) / 16, (NQ + 15) / 16);

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    // Warmup
    doitgen_slice_kernel<<<grid2d, block>>>(NP, NQ, NR, 0, A, C4, A_out);
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
        float total_launches = (float)STEPS * launches_per_step;
        printf("[%s] %d steps: median=%.3f ms, %.2f us/step, %.2f us/launch\n",
               name, STEPS, median, median * 1000.0f / STEPS,
               median * 1000.0f / total_launches);
    };

    // Strategy 1: Sync
    printf("\n--- Strategy 1: Sync ---\n");
    run_timed([&]() {
        for (int s = 0; s < STEPS; s++) {
            for (int r = 0; r < NR; r++) {
                doitgen_slice_kernel<<<grid2d, block>>>(NP, NQ, NR, r, A, C4, A_out);
                cudaDeviceSynchronize();
            }
            // Copy result back for next step
            cudaMemcpy(A, A_out, N3*sizeof(float), cudaMemcpyDeviceToDevice);
        }
    }, "Sync");

    // Strategy 2: Async
    printf("\n--- Strategy 2: Async ---\n");
    run_timed([&]() {
        for (int s = 0; s < STEPS; s++) {
            for (int r = 0; r < NR; r++) {
                doitgen_slice_kernel<<<grid2d, block>>>(NP, NQ, NR, r, A, C4, A_out);
            }
            cudaDeviceSynchronize();
            cudaMemcpy(A, A_out, N3*sizeof(float), cudaMemcpyDeviceToDevice);
        }
    }, "Async");

    // Strategy 3: CUDA Graph
    printf("\n--- Strategy 3: CUDA Graph ---\n");
    {
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));

        cudaEvent_t gi0, gi1;
        CHECK(cudaEventCreate(&gi0));
        CHECK(cudaEventCreate(&gi1));

        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int s = 0; s < STEPS; s++) {
            for (int r = 0; r < NR; r++) {
                doitgen_slice_kernel<<<grid2d, block, 0, stream>>>(NP, NQ, NR, r, A, C4, A_out);
            }
            cudaMemcpyAsync(A, A_out, N3*sizeof(float), cudaMemcpyDeviceToDevice, stream);
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
            cudaGraphLaunch(graphExec, stream);
            cudaStreamSynchronize(stream);
        }, "Graph");

        CHECK(cudaGraphExecDestroy(graphExec));
        CHECK(cudaGraphDestroy(graph));
        CHECK(cudaStreamDestroy(stream));
        CHECK(cudaEventDestroy(gi0));
        CHECK(cudaEventDestroy(gi1));
    }

    // Strategy 3b: Device Graph (tail launch)
    printf("\n--- Strategy 3b: Device Graph (tail launch) ---\n");
    printf("[DevGraph] N/A (requires host memcpy between steps)\n");

    // Strategy 4: Persistent Kernel
    printf("\n--- Strategy 4: Persistent Kernel ---\n");
    {
        // No cooperative launch needed — r-indices are independent, no barrier
        int needed = grid2d.x * grid2d.y;
        printf("Persistent: %d blocks (no barrier, no grid limit)\n", needed);
        run_timed([&]() {
            for (int s = 0; s < STEPS; s++) {
                doitgen_persistent<<<dim3(needed, 1, 1), block>>>(NP, NQ, NR, A, C4, A_out);
                cudaDeviceSynchronize();
                cudaMemcpy(A, A_out, N3*sizeof(float), cudaMemcpyDeviceToDevice);
            }
        }, "Persistent");
    }

    // Strategy 5: Fused kernel (all (p,q,r) in one launch, 0 barriers)
    printf("\n--- Strategy 5: Fused (0 barrier, 1 launch) ---\n");
    {
        int total = NP * NQ * NR;
        int fusedBlocks = (total + 255) / 256;
        printf("Fused: %d blocks x 256 threads (total %d elements)\n", fusedBlocks, total);
        run_timed([&]() {
            doitgen_persistent_stride<<<fusedBlocks, 256>>>(NP, NQ, NR, A, C4, A_out);
            cudaDeviceSynchronize();
        }, "Fused");
    }

    CHECK(cudaFree(A));
    CHECK(cudaFree(A_out));
    CHECK(cudaFree(C4));
    return 0;
}
