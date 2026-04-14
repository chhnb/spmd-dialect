/**
 * C17: 3D Convolution (PolyBench-style) — 4-strategy benchmark.
 * Host launches one kernel per z-slice. NZ slices => NZ launches per step.
 * Build: nvcc -O3 -arch=sm_80 -rdc=true conv3d_benchmark.cu -o conv3d_bench -lcudadevrt
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

// 3D convolution kernel for a single z-slice
// 3x3x3 stencil with uniform weights (1/27)
__global__ void conv3d_slice_kernel(int NX, int NY, int NZ, int z,
                                     const float* __restrict__ A,
                                     float* __restrict__ B) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (x < NX-1 && y < NY-1 && z >= 1 && z < NZ-1) {
        float sum = 0.0f;
        for (int dz = -1; dz <= 1; dz++)
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                    sum += A[(x+dx)*NY*NZ + (y+dy)*NZ + (z+dz)];
        B[x*NY*NZ + y*NZ + z] = sum / 27.0f;
    }
}

// Persistent: process all z-slices in one kernel using grid_sync
__global__ void conv3d_persistent(int NX, int NY, int NZ,
                                   const float* __restrict__ A,
                                   float* __restrict__ B) {
    auto grid = cg::this_grid();
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y + 1;

    // Process all interior z-slices; since each slice is independent for convolution,
    // we just loop and sync between slices for consistent memory view
    for (int z = 1; z < NZ - 1; z++) {
        if (tid_x < NX-1 && tid_y < NY-1) {
            float sum = 0.0f;
            for (int dz = -1; dz <= 1; dz++)
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                        sum += A[(tid_x+dx)*NY*NZ + (tid_y+dy)*NZ + (z+dz)];
            B[tid_x*NY*NZ + tid_y*NZ + z] = sum / 27.0f;
        }
        grid.sync();
    }
}

// Grid-stride persistent: never N/A
__global__ void conv3d_persistent_stride(int NX, int NY, int NZ,
                                          const float* __restrict__ A,
                                          float* __restrict__ B) {
    auto grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int slice_size = (NX-2) * (NY-2);  // interior xy points per z-slice

    for (int z = 1; z < NZ - 1; z++) {
        for (int idx = tid; idx < slice_size; idx += stride) {
            int ix = idx / (NY-2) + 1;
            int iy = idx % (NY-2) + 1;
            float sum = 0.0f;
            for (int dz = -1; dz <= 1; dz++)
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                        sum += A[(ix+dx)*NY*NZ + (iy+dy)*NZ + (z+dz)];
            B[ix*NY*NZ + iy*NZ + z] = sum / 27.0f;
        }
        grid.sync();
    }
}

int main(int argc, char* argv[]) {
    int NX = (argc > 1) ? atoi(argv[1]) : 128;
    int NY = NX, NZ = NX;
    int STEPS = (argc > 2) ? atoi(argv[2]) : 5;
    int REPEAT = (argc > 3) ? atoi(argv[3]) : 10;
    long N3 = (long)NX * NY * NZ;
    int launches_per_step = NZ - 2; // interior z-slices

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== C17: 3D Convolution Benchmark ===\n");
    printf("GPU: %s, SMs=%d\n", prop.name, prop.multiProcessorCount);
    printf("NX=NY=NZ=%d (%ld cells), steps=%d, repeat=%d\n", NX, N3, STEPS, REPEAT);
    printf("Launches/step: %d\n", launches_per_step);

    float *A, *B;
    CHECK(cudaMalloc(&A, N3 * sizeof(float)));
    CHECK(cudaMalloc(&B, N3 * sizeof(float)));

    // Init with synthetic data
    std::vector<float> h_A(N3);
    for (long i = 0; i < N3; i++) h_A[i] = sinf((float)i * 0.001f);
    CHECK(cudaMemcpy(A, h_A.data(), N3*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(B, 0, N3*sizeof(float)));

    dim3 block(16, 16);
    dim3 slice_grid((NX - 2 + 15) / 16, (NY - 2 + 15) / 16);

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    // Warmup
    conv3d_slice_kernel<<<slice_grid, block>>>(NX, NY, NZ, 1, A, B);
    cudaDeviceSynchronize();

    auto run_timed = [&](auto fn, const char* name) {
        std::vector<float> times;
        for (int r = 0; r < REPEAT; r++) {
            CHECK(cudaMemcpy(A, h_A.data(), N3*sizeof(float), cudaMemcpyHostToDevice));
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
            for (int z = 1; z < NZ - 1; z++) {
                conv3d_slice_kernel<<<slice_grid, block>>>(NX, NY, NZ, z, A, B);
                cudaDeviceSynchronize();
            }
            // Swap A,B for next step
            float* tmp = A; A = B; B = tmp;
        }
    }, "Sync");

    // Strategy 2: Async
    printf("\n--- Strategy 2: Async ---\n");
    run_timed([&]() {
        for (int s = 0; s < STEPS; s++) {
            for (int z = 1; z < NZ - 1; z++) {
                conv3d_slice_kernel<<<slice_grid, block>>>(NX, NY, NZ, z, A, B);
            }
            cudaDeviceSynchronize();
            float* tmp = A; A = B; B = tmp;
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
            if (s % 2 == 0) {
                for (int z = 1; z < NZ - 1; z++)
                    conv3d_slice_kernel<<<slice_grid, block, 0, stream>>>(NX, NY, NZ, z, A, B);
            } else {
                for (int z = 1; z < NZ - 1; z++)
                    conv3d_slice_kernel<<<slice_grid, block, 0, stream>>>(NX, NY, NZ, z, B, A);
            }
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

    // Strategy 4: Persistent Kernel
    printf("\n--- Strategy 4: Persistent Kernel ---\n");
    {
        int numBlocks;
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
              conv3d_persistent, block.x * block.y, 0));
        int maxBlocks = numBlocks * prop.multiProcessorCount;
        int needed = slice_grid.x * slice_grid.y;

        if (needed <= maxBlocks) {
            printf("Persistent: %d blocks (max %d)\n", needed, maxBlocks);
            // Persistent version does all z-slices internally
            void* args[] = {&NX, &NY, &NZ, &A, &B};
            run_timed([&]() {
                for (int s = 0; s < STEPS; s++) {
                    cudaLaunchCooperativeKernel((void*)conv3d_persistent,
                        dim3(needed, 1, 1), block, args);
                    cudaDeviceSynchronize();
                    float* tmp = A; A = B; B = tmp;
                    // Update args for swapped pointers
                    args[3] = &A; args[4] = &B;
                }
            }, "Persistent");
        } else {
            printf("Persistent: N/A (need %d blocks, max %d)\n", needed, maxBlocks);
        }
    }

    // Strategy 5: Grid-Stride Persistent (never N/A)
    printf("\n--- Strategy 5: Grid-Stride Persistent ---\n");
    {
        int gsBSm = 0;
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&gsBSm,
              conv3d_persistent_stride, 256, 0));
        int gsMax = gsBSm * prop.multiProcessorCount;
        printf("Grid-stride: %d blocks (always fits)\n", gsMax);
        void* gsArgs[] = {&NX, &NY, &NZ, &A, &B};
        run_timed([&]() {
            CHECK(cudaMemcpy(A, h_A.data(), N3*sizeof(float), cudaMemcpyHostToDevice));
            cudaLaunchCooperativeKernel((void*)conv3d_persistent_stride,
                dim3(gsMax), dim3(256), gsArgs);
            cudaDeviceSynchronize();
        }, "GridStride");
    }

    CHECK(cudaFree(A));
    CHECK(cudaFree(B));
    return 0;
}
