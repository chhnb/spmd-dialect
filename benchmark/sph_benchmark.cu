/**
 * C7: SPH density computation — 4-strategy benchmark.
 * 2 kernels/step: build_grid (uniform grid) + compute_density.
 * Strategies: Sync loop, Async loop, CUDA Graph, Persistent Kernel.
 * Build: nvcc -O3 -arch=sm_80 -rdc=true sph_benchmark.cu -o sph_bench -lcudadevrt
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

// --- Device Graph tail launch support ---
__device__ cudaGraphExec_t d_graph_exec;
__global__ void tail_launch_kernel(int* steps_remaining) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int rem = atomicSub(steps_remaining, 1);
        if (rem > 1) cudaGraphLaunch(d_graph_exec, cudaStreamGraphTailLaunch);
    }
}

// SPH parameters
static const float H = 0.05f;       // smoothing radius
static const float H2 = H * H;
static const float DOMAIN = 1.0f;   // domain [0, DOMAIN]^2
static const float MASS = 1.0f;
static const float DT_SPH = 0.0001f;

// Grid parameters (computed at runtime)
struct GridParams {
    int gridX, gridY;
    float cellSize;
    int maxPerCell;
    int totalCells;
};

// Poly6 kernel (unnormalized, for density)
__device__ float poly6(float r2, float h2) {
    if (r2 >= h2) return 0.0f;
    float d = h2 - r2;
    return d * d * d;  // will normalize with constant
}

// Build uniform grid: count + fill
__global__ void build_grid(int N, const float2* __restrict__ pos,
                           int* __restrict__ cellCount,
                           int* __restrict__ cellParticles,
                           int gridX, int gridY, float cellSize, int maxPerCell) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float2 p = pos[i];
    int cx = min(max((int)(p.x / cellSize), 0), gridX - 1);
    int cy = min(max((int)(p.y / cellSize), 0), gridY - 1);
    int cellIdx = cx * gridY + cy;

    int slot = atomicAdd(&cellCount[cellIdx], 1);
    if (slot < maxPerCell) {
        cellParticles[cellIdx * maxPerCell + slot] = i;
    }
}

// Compute density using neighbor search
__global__ void compute_density(int N, const float2* __restrict__ pos,
                                float* __restrict__ density,
                                const int* __restrict__ cellCount,
                                const int* __restrict__ cellParticles,
                                int gridX, int gridY, float cellSize,
                                int maxPerCell, float h2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float2 pi = pos[i];
    int cx = min(max((int)(pi.x / cellSize), 0), gridX - 1);
    int cy = min(max((int)(pi.y / cellSize), 0), gridY - 1);

    float rho = 0.0f;

    // Search 3x3 neighborhood
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            int nx = cx + dx, ny = cy + dy;
            if (nx < 0 || nx >= gridX || ny < 0 || ny >= gridY) continue;
            int cellIdx = nx * gridY + ny;
            int count = min(cellCount[cellIdx], maxPerCell);
            for (int s = 0; s < count; s++) {
                int j = cellParticles[cellIdx * maxPerCell + s];
                float2 pj = pos[j];
                float ddx = pi.x - pj.x, ddy = pi.y - pj.y;
                float r2 = ddx * ddx + ddy * ddy;
                rho += MASS * poly6(r2, h2);
            }
        }
    }

    // Normalize with poly6 constant: 4/(pi*h^8)
    float h8 = h2 * h2 * h2 * h2;
    density[i] = rho * 4.0f / (3.14159265f * h8);
}

// Simple position jitter to simulate motion (for multi-step benchmark)
__global__ void jitter_positions(int N, float2* __restrict__ pos, int step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }
    // tiny deterministic perturbation
    float dx = sinf((float)(i + step * 7)) * DT_SPH;
    float dy = cosf((float)(i + step * 13)) * DT_SPH;
    pos[i].x = fminf(fmaxf(pos[i].x + dx, 0.001f), DOMAIN - 0.001f);
    pos[i].y = fminf(fmaxf(pos[i].y + dy, 0.001f), DOMAIN - 0.001f);
}

// Persistent kernel: fused build_grid + compute_density
__global__ void sph_persistent(int N, float2* pos, float* density,
                               int* cellCount, int* cellParticles,
                               int gridX, int gridY, float cellSize,
                               int maxPerCell, float h2, int totalCells,
                               int STEPS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int s = 0; s < STEPS; s++) {
        // Clear grid (all threads help)
        for (int c = i; c < totalCells; c += gridDim.x * blockDim.x) {
            cellCount[c] = 0;
        }
        cg::this_grid().sync();

        // Build grid
        if (i < N) {
            float2 p = pos[i];
            int cx = min(max((int)(p.x / cellSize), 0), gridX - 1);
            int cy = min(max((int)(p.y / cellSize), 0), gridY - 1);
            int cellIdx = cx * gridY + cy;
            int slot = atomicAdd(&cellCount[cellIdx], 1);
            if (slot < maxPerCell) {
                cellParticles[cellIdx * maxPerCell + slot] = i;
            }
        }
        cg::this_grid().sync();

        // Compute density
        if (i < N) {
            float2 pi = pos[i];
            int cx = min(max((int)(pi.x / cellSize), 0), gridX - 1);
            int cy = min(max((int)(pi.y / cellSize), 0), gridY - 1);
            float rho = 0.0f;
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    int nx = cx + dx, ny = cy + dy;
                    if (nx < 0 || nx >= gridX || ny < 0 || ny >= gridY) continue;
                    int cellIdx = nx * gridY + ny;
                    int count = min(cellCount[cellIdx], maxPerCell);
                    for (int ss = 0; ss < count; ss++) {
                        int j = cellParticles[cellIdx * maxPerCell + ss];
                        float2 pj = pos[j];
                        float ddx = pi.x - pj.x, ddy = pi.y - pj.y;
                        float r2 = ddx * ddx + ddy * ddy;
                        rho += MASS * poly6(r2, h2);
                    }
                }
            }
            float h8 = h2 * h2 * h2 * h2;
            density[i] = rho * 4.0f / (3.14159265f * h8);
        }

        // Jitter positions
        if (i < N) {
            float dx = sinf((float)(i + s * 7)) * DT_SPH;
            float dy = cosf((float)(i + s * 13)) * DT_SPH;
            pos[i].x = fminf(fmaxf(pos[i].x + dx, 0.001f), DOMAIN - 0.001f);
            pos[i].y = fminf(fmaxf(pos[i].y + dy, 0.001f), DOMAIN - 0.001f);
        }
        cg::this_grid().sync();
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 8192;
    int STEPS = (argc > 2) ? atoi(argv[2]) : 100;
    int REPEAT = (argc > 3) ? atoi(argv[3]) : 10;

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== C7: SPH Density Benchmark ===\n");
    printf("GPU: %s, SMs=%d\n", prop.name, prop.multiProcessorCount);
    printf("N=%d particles, h=%.3f, steps=%d, repeat=%d\n", N, H, STEPS, REPEAT);

    // Grid setup
    GridParams gp;
    gp.cellSize = H;
    gp.gridX = (int)(DOMAIN / gp.cellSize) + 1;
    gp.gridY = (int)(DOMAIN / gp.cellSize) + 1;
    gp.maxPerCell = 64;
    gp.totalCells = gp.gridX * gp.gridY;
    printf("Grid: %d x %d cells, maxPerCell=%d\n", gp.gridX, gp.gridY, gp.maxPerCell);

    float2 *d_pos;
    float *d_density;
    int *d_cellCount, *d_cellParticles;
    CHECK(cudaMalloc(&d_pos, N * sizeof(float2)));
    CHECK(cudaMalloc(&d_density, N * sizeof(float)));
    CHECK(cudaMalloc(&d_cellCount, gp.totalCells * sizeof(int)));
    CHECK(cudaMalloc(&d_cellParticles, gp.totalCells * gp.maxPerCell * sizeof(int)));

    // Init: random positions in [0, DOMAIN]^2
    std::vector<float2> h_pos(N);
    srand(42);
    for (int i = 0; i < N; i++) {
        h_pos[i] = {(float)rand()/RAND_MAX * DOMAIN, (float)rand()/RAND_MAX * DOMAIN};
    }

    auto reset = [&]() {
        CHECK(cudaMemcpy(d_pos, h_pos.data(), N * sizeof(float2), cudaMemcpyHostToDevice));
        CHECK(cudaMemset(d_density, 0, N * sizeof(float)));
        CHECK(cudaMemset(d_cellCount, 0, gp.totalCells * sizeof(int)));
    };
    reset();

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    // Warmup
    for (int i = 0; i < 3; i++) {
        cudaMemset(d_cellCount, 0, gp.totalCells * sizeof(int));
        build_grid<<<gridSize, blockSize>>>(N, d_pos, d_cellCount, d_cellParticles,
                                            gp.gridX, gp.gridY, gp.cellSize, gp.maxPerCell);
        compute_density<<<gridSize, blockSize>>>(N, d_pos, d_density, d_cellCount,
                                                  d_cellParticles, gp.gridX, gp.gridY,
                                                  gp.cellSize, gp.maxPerCell, H2);
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
            cudaMemset(d_cellCount, 0, gp.totalCells * sizeof(int));
            build_grid<<<gridSize, blockSize>>>(N, d_pos, d_cellCount, d_cellParticles,
                                                gp.gridX, gp.gridY, gp.cellSize, gp.maxPerCell);
            compute_density<<<gridSize, blockSize>>>(N, d_pos, d_density, d_cellCount,
                                                      d_cellParticles, gp.gridX, gp.gridY,
                                                      gp.cellSize, gp.maxPerCell, H2);
            jitter_positions<<<gridSize, blockSize>>>(N, d_pos, s);
            cudaDeviceSynchronize();
        }
    }, "Sync Loop");

    // Strategy 2: Async Loop
    printf("\n--- Strategy 2: Async Loop ---\n");
    run_timed([&]() {
        for (int s = 0; s < STEPS; s++) {
            cudaMemset(d_cellCount, 0, gp.totalCells * sizeof(int));
            build_grid<<<gridSize, blockSize>>>(N, d_pos, d_cellCount, d_cellParticles,
                                                gp.gridX, gp.gridY, gp.cellSize, gp.maxPerCell);
            compute_density<<<gridSize, blockSize>>>(N, d_pos, d_density, d_cellCount,
                                                      d_cellParticles, gp.gridX, gp.gridY,
                                                      gp.cellSize, gp.maxPerCell, H2);
            jitter_positions<<<gridSize, blockSize>>>(N, d_pos, s);
        }
        cudaDeviceSynchronize();
    }, "Async Loop");

    // Strategy 3: CUDA Graph
    // Note: cudaMemset is capturable in CUDA graph
    printf("\n--- Strategy 3: CUDA Graph ---\n");
    {
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int s = 0; s < STEPS; s++) {
            cudaMemsetAsync(d_cellCount, 0, gp.totalCells * sizeof(int), stream);
            build_grid<<<gridSize, blockSize, 0, stream>>>(N, d_pos, d_cellCount, d_cellParticles,
                                                            gp.gridX, gp.gridY, gp.cellSize, gp.maxPerCell);
            compute_density<<<gridSize, blockSize, 0, stream>>>(N, d_pos, d_density, d_cellCount,
                                                                  d_cellParticles, gp.gridX, gp.gridY,
                                                                  gp.cellSize, gp.maxPerCell, H2);
            jitter_positions<<<gridSize, blockSize, 0, stream>>>(N, d_pos, s);
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

    // Strategy 3b: Device Graph (tail launch, same kernels, no fusion)
    printf("\n--- Strategy 3b: Device Graph (tail launch) ---\n");
    {
        int *d_steps_dg;
        CHECK(cudaMalloc(&d_steps_dg, sizeof(int)));
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));

        // Capture ONE step: memset + build_grid + compute_density + jitter
        // Note: jitter uses step=0 in capture (fixed in graph), acceptable for benchmark
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        cudaMemsetAsync(d_cellCount, 0, gp.totalCells * sizeof(int), stream);
        build_grid<<<gridSize, blockSize, 0, stream>>>(N, d_pos, d_cellCount, d_cellParticles,
                                                        gp.gridX, gp.gridY, gp.cellSize, gp.maxPerCell);
        compute_density<<<gridSize, blockSize, 0, stream>>>(N, d_pos, d_density, d_cellCount,
                                                              d_cellParticles, gp.gridX, gp.gridY,
                                                              gp.cellSize, gp.maxPerCell, H2);
        jitter_positions<<<gridSize, blockSize, 0, stream>>>(N, d_pos, 0);
        tail_launch_kernel<<<1, 1, 0, stream>>>(d_steps_dg);
        CHECK(cudaStreamEndCapture(stream, &graph));

        // Instantiate for device-side launch
        CHECK(cudaGraphInstantiateWithFlags(&graphExec, graph,
              cudaGraphInstantiateFlagDeviceLaunch));
        CHECK(cudaGraphUpload(graphExec, stream));

        // Copy graph exec handle to device symbol
        cudaGraphExec_t* d_sym_ptr;
        CHECK(cudaGetSymbolAddress((void**)&d_sym_ptr, d_graph_exec));
        CHECK(cudaMemcpy(d_sym_ptr, &graphExec, sizeof(cudaGraphExec_t),
              cudaMemcpyHostToDevice));

        // Warmup
        for (int w = 0; w < 5; w++) {
            int sv = STEPS;
            CHECK(cudaMemcpy(d_steps_dg, &sv, sizeof(int), cudaMemcpyHostToDevice));
            CHECK(cudaGraphLaunch(graphExec, stream));
            CHECK(cudaStreamSynchronize(stream));
        }

        run_timed([&]() {
            int sv = STEPS;
            CHECK(cudaMemcpy(d_steps_dg, &sv, sizeof(int), cudaMemcpyHostToDevice));
            CHECK(cudaGraphLaunch(graphExec, stream));
            cudaStreamSynchronize(stream);
        }, "DevGraph");

        CHECK(cudaGraphExecDestroy(graphExec));
        CHECK(cudaGraphDestroy(graph));
        CHECK(cudaStreamDestroy(stream));
        CHECK(cudaFree(d_steps_dg));
    }

    // Strategy 4: Persistent Kernel
    printf("\n--- Strategy 4: Persistent Kernel ---\n");
    {
        int numBlocks;
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
              sph_persistent, blockSize, 0));
        int maxBlocks = numBlocks * prop.multiProcessorCount;
        int needed = gridSize;

        if (needed <= maxBlocks) {
            printf("Persistent: %d blocks (need %d, max %d)\n",
                   needed, needed, maxBlocks);
            float h2 = H2;
            void* args[] = {(void*)&N, &d_pos, &d_density, &d_cellCount, &d_cellParticles,
                            (void*)&gp.gridX, (void*)&gp.gridY, &gp.cellSize,
                            (void*)&gp.maxPerCell, &h2, (void*)&gp.totalCells, &STEPS};
            run_timed([&]() {
                cudaLaunchCooperativeKernel((void*)sph_persistent,
                    dim3(needed), dim3(blockSize), args);
                cudaDeviceSynchronize();
            }, "Persistent");
        } else {
            printf("Persistent: N/A (need %d blocks, max %d)\n", needed, maxBlocks);
        }
    }

    printf("\n=== CSV: sph,%d,%d,A100 ===\n", N, STEPS);

    CHECK(cudaFree(d_pos));
    CHECK(cudaFree(d_density));
    CHECK(cudaFree(d_cellCount));
    CHECK(cudaFree(d_cellParticles));
    return 0;
}
