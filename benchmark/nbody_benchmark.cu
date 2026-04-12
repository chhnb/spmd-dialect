/**
 * C6: N-body all-pairs — 4-strategy benchmark.
 * 2 kernels/step: compute_forces (shared mem tiling) + integrate.
 * Strategies: Sync loop, Async loop, CUDA Graph, Persistent Kernel.
 * Build: nvcc -O3 -arch=sm_80 -rdc=true nbody_benchmark.cu -o nbody_bench -lcudadevrt
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

static const float DT = 0.001f;
static const float EPS2 = 0.01f;  // softening
static const int TILE_SIZE = 256;

// Shared memory tiled force computation
__global__ void compute_forces(int N, const float4* __restrict__ pos,
                               const float* __restrict__ mass,
                               float3* __restrict__ acc) {
    extern __shared__ float4 spos[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float3 ai = {0.0f, 0.0f, 0.0f};
    float4 pi;
    if (i < N) pi = pos[i];

    for (int tile = 0; tile < (N + blockDim.x - 1) / blockDim.x; tile++) {
        int j = tile * blockDim.x + threadIdx.x;
        if (j < N) {
            spos[threadIdx.x] = pos[j];
            // pack mass into w
            spos[threadIdx.x].w = mass[j];
        } else {
            spos[threadIdx.x] = {0.0f, 0.0f, 0.0f, 0.0f};
        }
        __syncthreads();

        if (i < N) {
            for (int k = 0; k < blockDim.x; k++) {
                float dx = spos[k].x - pi.x;
                float dy = spos[k].y - pi.y;
                float dz = spos[k].z - pi.z;
                float r2 = dx*dx + dy*dy + dz*dz + EPS2;
                float inv_r3 = rsqrtf(r2) / r2;
                float mj = spos[k].w;
                ai.x += mj * dx * inv_r3;
                ai.y += mj * dy * inv_r3;
                ai.z += mj * dz * inv_r3;
            }
        }
        __syncthreads();
    }

    if (i < N) acc[i] = ai;
}

// Leapfrog integration
__global__ void integrate(int N, float4* __restrict__ pos,
                          float3* __restrict__ vel,
                          const float3* __restrict__ acc, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        vel[i].x += acc[i].x * dt;
        vel[i].y += acc[i].y * dt;
        vel[i].z += acc[i].z * dt;
        pos[i].x += vel[i].x * dt;
        pos[i].y += vel[i].y * dt;
        pos[i].z += vel[i].z * dt;
    }
}

// Persistent: fused force + integrate
__global__ void nbody_persistent(int N, float4* pos, float3* vel,
                                 float* mass, float3* acc, float dt, int STEPS) {
    extern __shared__ float4 spos[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int s = 0; s < STEPS; s++) {
        // Compute forces
        float3 ai = {0.0f, 0.0f, 0.0f};
        float4 pi;
        if (i < N) pi = pos[i];

        for (int tile = 0; tile < (N + blockDim.x - 1) / blockDim.x; tile++) {
            int j = tile * blockDim.x + threadIdx.x;
            if (j < N) {
                spos[threadIdx.x] = pos[j];
                spos[threadIdx.x].w = mass[j];
            } else {
                spos[threadIdx.x] = {0.0f, 0.0f, 0.0f, 0.0f};
            }
            __syncthreads();

            if (i < N) {
                for (int k = 0; k < blockDim.x; k++) {
                    float dx = spos[k].x - pi.x;
                    float dy = spos[k].y - pi.y;
                    float dz = spos[k].z - pi.z;
                    float r2 = dx*dx + dy*dy + dz*dz + EPS2;
                    float inv_r3 = rsqrtf(r2) / r2;
                    float mj = spos[k].w;
                    ai.x += mj * dx * inv_r3;
                    ai.y += mj * dy * inv_r3;
                    ai.z += mj * dz * inv_r3;
                }
            }
            __syncthreads();
        }

        if (i < N) acc[i] = ai;

        cg::this_grid().sync();

        // Integrate
        if (i < N) {
            vel[i].x += acc[i].x * dt;
            vel[i].y += acc[i].y * dt;
            vel[i].z += acc[i].z * dt;
            pos[i].x += vel[i].x * dt;
            pos[i].y += vel[i].y * dt;
            pos[i].z += vel[i].z * dt;
        }

        cg::this_grid().sync();
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 4096;
    int STEPS = (argc > 2) ? atoi(argv[2]) : 100;
    int REPEAT = (argc > 3) ? atoi(argv[3]) : 10;

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== C6: N-body All-Pairs Benchmark ===\n");
    printf("GPU: %s, SMs=%d\n", prop.name, prop.multiProcessorCount);
    printf("N=%d particles, steps=%d, repeat=%d\n", N, STEPS, REPEAT);

    float4 *d_pos;
    float3 *d_vel, *d_acc;
    float *d_mass;
    CHECK(cudaMalloc(&d_pos, N * sizeof(float4)));
    CHECK(cudaMalloc(&d_vel, N * sizeof(float3)));
    CHECK(cudaMalloc(&d_acc, N * sizeof(float3)));
    CHECK(cudaMalloc(&d_mass, N * sizeof(float)));

    // Init: random positions in [-1, 1]^3, zero velocity, unit mass
    std::vector<float4> h_pos(N);
    std::vector<float3> h_vel(N, {0.0f, 0.0f, 0.0f});
    std::vector<float> h_mass(N, 1.0f);
    srand(42);
    for (int i = 0; i < N; i++) {
        h_pos[i] = {(float)rand()/RAND_MAX*2-1, (float)rand()/RAND_MAX*2-1,
                     (float)rand()/RAND_MAX*2-1, 0.0f};
    }

    auto reset = [&]() {
        CHECK(cudaMemcpy(d_pos, h_pos.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_vel, h_vel.data(), N * sizeof(float3), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_mass, h_mass.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemset(d_acc, 0, N * sizeof(float3)));
    };
    reset();

    int blockSize = TILE_SIZE;
    int gridSize = (N + blockSize - 1) / blockSize;
    size_t sharedMem = blockSize * sizeof(float4);

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    // Warmup
    for (int i = 0; i < 3; i++) {
        compute_forces<<<gridSize, blockSize, sharedMem>>>(N, d_pos, d_mass, d_acc);
        integrate<<<gridSize, blockSize>>>(N, d_pos, d_vel, d_acc, DT);
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
            compute_forces<<<gridSize, blockSize, sharedMem>>>(N, d_pos, d_mass, d_acc);
            integrate<<<gridSize, blockSize>>>(N, d_pos, d_vel, d_acc, DT);
            cudaDeviceSynchronize();
        }
    }, "Sync Loop");

    // Strategy 2: Async Loop
    printf("\n--- Strategy 2: Async Loop ---\n");
    run_timed([&]() {
        for (int s = 0; s < STEPS; s++) {
            compute_forces<<<gridSize, blockSize, sharedMem>>>(N, d_pos, d_mass, d_acc);
            integrate<<<gridSize, blockSize>>>(N, d_pos, d_vel, d_acc, DT);
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
            compute_forces<<<gridSize, blockSize, sharedMem, stream>>>(N, d_pos, d_mass, d_acc);
            integrate<<<gridSize, blockSize, 0, stream>>>(N, d_pos, d_vel, d_acc, DT);
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
        int numBlocks;
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
              nbody_persistent, blockSize, sharedMem));
        int maxBlocks = numBlocks * prop.multiProcessorCount;
        int needed = gridSize;

        if (needed <= maxBlocks) {
            printf("Persistent: %d blocks (need %d, max %d)\n",
                   needed, needed, maxBlocks);
            float dt = DT;
            void* args[] = {(void*)&N, &d_pos, &d_vel, &d_mass, &d_acc, &dt, &STEPS};
            run_timed([&]() {
                cudaLaunchCooperativeKernel((void*)nbody_persistent,
                    dim3(needed), dim3(blockSize), args, sharedMem);
                cudaDeviceSynchronize();
            }, "Persistent");
        } else {
            printf("Persistent: N/A (need %d blocks, max %d)\n", needed, maxBlocks);
        }
    }

    printf("\n=== CSV: nbody,%d,%d,A100 ===\n", N, STEPS);

    CHECK(cudaFree(d_pos));
    CHECK(cudaFree(d_vel));
    CHECK(cudaFree(d_acc));
    CHECK(cudaFree(d_mass));
    return 0;
}
