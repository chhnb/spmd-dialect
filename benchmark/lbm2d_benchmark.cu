/**
 * C5: LBM D2Q9 — 4-strategy benchmark.
 * Stream + collide fused in one kernel per step.
 * Strategies: Sync loop, Async loop, CUDA Graph, Persistent Kernel.
 * Build: nvcc -O3 -arch=sm_80 -rdc=true lbm2d_benchmark.cu -o lbm2d_bench -lcudadevrt
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

// D2Q9 lattice velocities and weights
__constant__ int d_ex[9] = {0, 1, 0, -1,  0, 1, -1, -1,  1};
__constant__ int d_ey[9] = {0, 0, 1,  0, -1, 1,  1, -1, -1};
__constant__ float d_w[9] = {4.0f/9.0f,
    1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f};

// Stream + Collide fused: read from f_in (with streaming offset), write to f_out
__global__ void lbm_stream_collide(int NX, int NY, float omega,
                                   const float* __restrict__ f_in,
                                   float* __restrict__ f_out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    int idx = x * NY + y;
    float f[9];

    // Stream: pull from neighbors
    for (int q = 0; q < 9; q++) {
        int sx = (x - d_ex[q] + NX) % NX;
        int sy = (y - d_ey[q] + NY) % NY;
        f[q] = f_in[q * NX * NY + sx * NY + sy];
    }

    // Compute macroscopic quantities
    float rho = 0.0f, ux = 0.0f, uy = 0.0f;
    for (int q = 0; q < 9; q++) {
        rho += f[q];
        ux += f[q] * d_ex[q];
        uy += f[q] * d_ey[q];
    }
    if (rho > 1e-10f) { ux /= rho; uy /= rho; }

    // Collide: BGK
    float u2 = ux * ux + uy * uy;
    for (int q = 0; q < 9; q++) {
        float eu = d_ex[q] * ux + d_ey[q] * uy;
        float feq = d_w[q] * rho * (1.0f + 3.0f * eu + 4.5f * eu * eu - 1.5f * u2);
        f_out[q * NX * NY + idx] = f[q] + omega * (feq - f[q]);
    }
}

// Persistent version: 1D thread mapping with grid-stride loop
__global__ void lbm_persistent(int NX, int NY, int cells, float omega,
                               float* f0, float* f1, int STEPS) {
    for (int s = 0; s < STEPS; s++) {
        float* f_in  = (s % 2 == 0) ? f0 : f1;
        float* f_out = (s % 2 == 0) ? f1 : f0;

        for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < cells;
             tid += gridDim.x * blockDim.x) {
            int x = tid / NY;
            int y = tid % NY;
            float f[9];
            for (int q = 0; q < 9; q++) {
                int sx = (x - d_ex[q] + NX) % NX;
                int sy = (y - d_ey[q] + NY) % NY;
                f[q] = f_in[q * cells + sx * NY + sy];
            }
            float rho = 0.0f, ux = 0.0f, uy = 0.0f;
            for (int q = 0; q < 9; q++) {
                rho += f[q]; ux += f[q] * d_ex[q]; uy += f[q] * d_ey[q];
            }
            if (rho > 1e-10f) { ux /= rho; uy /= rho; }
            float u2 = ux * ux + uy * uy;
            for (int q = 0; q < 9; q++) {
                float eu = d_ex[q] * ux + d_ey[q] * uy;
                float feq = d_w[q] * rho * (1.0f + 3.0f * eu + 4.5f * eu * eu - 1.5f * u2);
                f_out[q * cells + tid] = f[q] + omega * (feq - f[q]);
            }
        }
        cg::this_grid().sync();
    }
}

int main(int argc, char* argv[]) {
    int NX = (argc > 1) ? atoi(argv[1]) : 512;
    int NY = (argc > 2) ? atoi(argv[2]) : 256;
    int STEPS = (argc > 3) ? atoi(argv[3]) : 100;
    int REPEAT = (argc > 4) ? atoi(argv[4]) : 10;
    int cells = NX * NY;
    int totalF = 9 * cells;  // 9 distribution functions per cell

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== C5: LBM D2Q9 Benchmark ===\n");
    printf("GPU: %s, SMs=%d\n", prop.name, prop.multiProcessorCount);
    printf("NX=%d, NY=%d (%d cells), steps=%d, repeat=%d\n", NX, NY, cells, STEPS, REPEAT);

    float *f0, *f1;
    CHECK(cudaMalloc(&f0, totalF * sizeof(float)));
    CHECK(cudaMalloc(&f1, totalF * sizeof(float)));

    // Init: equilibrium at rho=1, u=0
    std::vector<float> h_f(totalF);
    float w_h[9] = {4.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
                     1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f};
    for (int q = 0; q < 9; q++)
        for (int c = 0; c < cells; c++)
            h_f[q * cells + c] = w_h[q];

    auto reset = [&]() {
        CHECK(cudaMemcpy(f0, h_f.data(), totalF * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(f1, h_f.data(), totalF * sizeof(float), cudaMemcpyHostToDevice));
    };
    reset();

    float omega = 1.0f;
    dim3 block(16, 16);
    dim3 grid((NX + 15) / 16, (NY + 15) / 16);

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    // Warmup
    for (int i = 0; i < 5; i++) {
        lbm_stream_collide<<<grid, block>>>(NX, NY, omega, f0, f1);
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

    // Strategy 1: Sync Loop (1 kernel/step, ping-pong buffers)
    printf("\n--- Strategy 1: Sync Loop ---\n");
    run_timed([&]() {
        for (int s = 0; s < STEPS; s++) {
            if (s % 2 == 0)
                lbm_stream_collide<<<grid, block>>>(NX, NY, omega, f0, f1);
            else
                lbm_stream_collide<<<grid, block>>>(NX, NY, omega, f1, f0);
            cudaDeviceSynchronize();
        }
    }, "Sync Loop");

    // Strategy 2: Async Loop
    printf("\n--- Strategy 2: Async Loop ---\n");
    run_timed([&]() {
        for (int s = 0; s < STEPS; s++) {
            if (s % 2 == 0)
                lbm_stream_collide<<<grid, block>>>(NX, NY, omega, f0, f1);
            else
                lbm_stream_collide<<<grid, block>>>(NX, NY, omega, f1, f0);
        }
        cudaDeviceSynchronize();
    }, "Async Loop");

    // Strategy 3: CUDA Graph (capture 2 steps = one ping-pong cycle)
    printf("\n--- Strategy 3: CUDA Graph ---\n");
    {
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        lbm_stream_collide<<<grid, block, 0, stream>>>(NX, NY, omega, f0, f1);
        lbm_stream_collide<<<grid, block, 0, stream>>>(NX, NY, omega, f1, f0);
        CHECK(cudaStreamEndCapture(stream, &graph));
        CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

        int pairs = STEPS / 2;
        int remainder = STEPS % 2;

        run_timed([&]() {
            for (int t = 0; t < pairs; t++) {
                cudaGraphLaunch(graphExec, stream);
            }
            if (remainder) {
                lbm_stream_collide<<<grid, block, 0, stream>>>(NX, NY, omega, f0, f1);
            }
            cudaStreamSynchronize(stream);
        }, "CUDA Graph");

        CHECK(cudaGraphExecDestroy(graphExec));
        CHECK(cudaGraphDestroy(graph));
        CHECK(cudaStreamDestroy(stream));
    }

    // Strategy 3b: Device Graph (tail launch)
    printf("\n--- Strategy 3b: Device Graph (tail launch) ---\n");
    printf("[DevGraph] N/A (requires host buffer swap between steps)\n");

    // Strategy 4: Persistent Kernel (1D thread mapping with grid-stride)
    printf("\n--- Strategy 4: Persistent Kernel ---\n");
    {
        int pBlockSize = 256;
        int numBlocks;
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
              lbm_persistent, pBlockSize, 0));
        int maxBlocks = numBlocks * prop.multiProcessorCount;
        int needed = (cells + pBlockSize - 1) / pBlockSize;
        int pGridSize = (needed <= maxBlocks) ? needed : maxBlocks;

        printf("Persistent: %d blocks (need %d, max %d, launch %d)\n",
               pGridSize, needed, maxBlocks, pGridSize);
        void* args[] = {(void*)&NX, (void*)&NY, (void*)&cells, &omega, &f0, &f1, &STEPS};
        run_timed([&]() {
            cudaLaunchCooperativeKernel((void*)lbm_persistent,
                dim3(pGridSize), dim3(pBlockSize), args);
            cudaDeviceSynchronize();
        }, "Persistent");
    }

    printf("\n=== CSV: lbm2d,%d,%d,%d,A100 ===\n", NX, NY, STEPS);

    CHECK(cudaFree(f0));
    CHECK(cudaFree(f1));
    return 0;
}
