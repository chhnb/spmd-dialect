/**
 * C16: Stable Fluids (Stam 1999) — 4-strategy benchmark.
 * Per step: advect + divergence + N*(jacobi_pressure + copy) + project
 * Default jacobi_iters=50 => 102 kernel launches per step.
 * Build: nvcc -O3 -arch=sm_80 -rdc=true stable_fluids_benchmark.cu -o stable_fluids_bench -lcudadevrt
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

// --- Kernels: Stable Fluids phases ---

// Advection: semi-Lagrangian backtrace
__global__ void advect_kernel(int N, const float* __restrict__ vx, const float* __restrict__ vy,
                               const float* __restrict__ field_in, float* __restrict__ field_out, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
        int idx = i * N + j;
        float x = (float)i - dt * vx[idx];
        float y = (float)j - dt * vy[idx];
        x = fmaxf(0.5f, fminf((float)(N-1) - 0.5f, x));
        y = fmaxf(0.5f, fminf((float)(N-1) - 0.5f, y));
        int i0 = (int)x, j0 = (int)y;
        float s = x - i0, t = y - j0;
        int i1 = min(i0+1, N-1), j1 = min(j0+1, N-1);
        field_out[idx] = (1-s)*(1-t)*field_in[i0*N+j0] + s*(1-t)*field_in[i1*N+j0]
                       + (1-s)*t*field_in[i0*N+j1] + s*t*field_in[i1*N+j1];
    }
}

// Divergence of velocity
__global__ void divergence_kernel(int N, const float* __restrict__ vx, const float* __restrict__ vy,
                                   float* __restrict__ div) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
        int idx = i * N + j;
        div[idx] = -0.5f * ((vx[(i+1)*N+j] - vx[(i-1)*N+j]) + (vy[i*N+j+1] - vy[i*N+j-1]));
    }
}

// Jacobi iteration for pressure solve
__global__ void jacobi_pressure_kernel(int N, const float* __restrict__ p, const float* __restrict__ div,
                                        float* __restrict__ p_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
        int idx = i * N + j;
        p_out[idx] = 0.25f * (p[(i-1)*N+j] + p[(i+1)*N+j] + p[i*N+j-1] + p[i*N+j+1] + div[idx]);
    }
}

// Copy kernel
__global__ void copy_kernel(int N2, const float* __restrict__ src, float* __restrict__ dst) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < N2) dst[i] = src[i];
}

// Pressure projection
__global__ void project_kernel(int N, float* __restrict__ vx, float* __restrict__ vy,
                                const float* __restrict__ p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
        int idx = i * N + j;
        vx[idx] -= 0.5f * (p[(i+1)*N+j] - p[(i-1)*N+j]);
        vy[idx] -= 0.5f * (p[i*N+j+1] - p[i*N+j-1]);
    }
}

// --- Persistent: all phases fused ---
__global__ void stable_fluids_persistent(int N, float* vx, float* vy,
                                          float* vx_tmp, float* vy_tmp,
                                          float* p, float* p_tmp, float* div,
                                          int jacobi_iters, int STEPS, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    auto grid = cg::this_grid();
    int N2 = N * N;

    for (int step = 0; step < STEPS; step++) {
        // Phase 1: Advect vx
        if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
            int idx = i * N + j;
            float x = (float)i - dt * vx[idx];
            float y = (float)j - dt * vy[idx];
            x = fmaxf(0.5f, fminf((float)(N-1) - 0.5f, x));
            y = fmaxf(0.5f, fminf((float)(N-1) - 0.5f, y));
            int i0 = (int)x, j0 = (int)y;
            float s = x - i0, t_ = y - j0;
            int i1 = min(i0+1, N-1), j1 = min(j0+1, N-1);
            vx_tmp[idx] = (1-s)*(1-t_)*vx[i0*N+j0] + s*(1-t_)*vx[i1*N+j0]
                         + (1-s)*t_*vx[i0*N+j1] + s*t_*vx[i1*N+j1];
            vy_tmp[idx] = (1-s)*(1-t_)*vy[i0*N+j0] + s*(1-t_)*vy[i1*N+j0]
                         + (1-s)*t_*vy[i0*N+j1] + s*t_*vy[i1*N+j1];
        }
        grid.sync();

        // Copy back
        if (i * N + j < N2) { vx[i*N+j] = vx_tmp[i*N+j]; vy[i*N+j] = vy_tmp[i*N+j]; }
        grid.sync();

        // Phase 2: Divergence
        if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
            int idx = i * N + j;
            div[idx] = -0.5f * ((vx[(i+1)*N+j] - vx[(i-1)*N+j]) + (vy[i*N+j+1] - vy[i*N+j-1]));
            p[idx] = 0.0f;
        }
        grid.sync();

        // Phase 3: Jacobi pressure iterations
        for (int it = 0; it < jacobi_iters; it++) {
            if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
                int idx = i * N + j;
                p_tmp[idx] = 0.25f * (p[(i-1)*N+j] + p[(i+1)*N+j] + p[i*N+j-1] + p[i*N+j+1] + div[idx]);
            }
            grid.sync();
            if (i * N + j < N2) p[i*N+j] = p_tmp[i*N+j];
            grid.sync();
        }

        // Phase 4: Project
        if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
            int idx = i * N + j;
            vx[idx] -= 0.5f * (p[(i+1)*N+j] - p[(i-1)*N+j]);
            vy[idx] -= 0.5f * (p[i*N+j+1] - p[i*N+j-1]);
        }
        grid.sync();
    }
}

// Grid-stride persistent: same algorithm, uses maxCooperativeBlocks
__global__ void stable_fluids_persistent_stride(int N, float* vx, float* vy,
                                                 float* vx_tmp, float* vy_tmp,
                                                 float* p, float* p_tmp, float* div,
                                                 int jacobi_iters, int STEPS, float dt) {
    auto grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int N2 = N * N;

    for (int step = 0; step < STEPS; step++) {
        // Phase 1: Advect
        for (int idx = tid; idx < N2; idx += stride) {
            int i = idx / N, j = idx % N;
            if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
                float x = (float)i - dt * vx[idx];
                float y = (float)j - dt * vy[idx];
                x = fmaxf(0.5f, fminf((float)(N-1)-0.5f, x));
                y = fmaxf(0.5f, fminf((float)(N-1)-0.5f, y));
                int i0=(int)x, j0=(int)y;
                float s=x-i0, t_=y-j0;
                int i1=min(i0+1,N-1), j1=min(j0+1,N-1);
                vx_tmp[idx]=(1-s)*(1-t_)*vx[i0*N+j0]+s*(1-t_)*vx[i1*N+j0]+(1-s)*t_*vx[i0*N+j1]+s*t_*vx[i1*N+j1];
                vy_tmp[idx]=(1-s)*(1-t_)*vy[i0*N+j0]+s*(1-t_)*vy[i1*N+j0]+(1-s)*t_*vy[i0*N+j1]+s*t_*vy[i1*N+j1];
            }
        }
        grid.sync();
        for (int idx = tid; idx < N2; idx += stride) { vx[idx]=vx_tmp[idx]; vy[idx]=vy_tmp[idx]; }
        grid.sync();

        // Phase 2: Divergence
        for (int idx = tid; idx < N2; idx += stride) {
            int i = idx / N, j = idx % N;
            if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
                div[idx]=-0.5f*((vx[(i+1)*N+j]-vx[(i-1)*N+j])+(vy[i*N+j+1]-vy[i*N+j-1]));
                p[idx]=0.0f;
            }
        }
        grid.sync();

        // Phase 3: Jacobi
        for (int it = 0; it < jacobi_iters; it++) {
            for (int idx = tid; idx < N2; idx += stride) {
                int i = idx / N, j = idx % N;
                if (i >= 1 && i < N-1 && j >= 1 && j < N-1)
                    p_tmp[idx]=0.25f*(p[(i-1)*N+j]+p[(i+1)*N+j]+p[i*N+j-1]+p[i*N+j+1]+div[idx]);
            }
            grid.sync();
            for (int idx = tid; idx < N2; idx += stride) p[idx]=p_tmp[idx];
            grid.sync();
        }

        // Phase 4: Project
        for (int idx = tid; idx < N2; idx += stride) {
            int i = idx / N, j = idx % N;
            if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
                vx[idx]-=0.5f*(p[(i+1)*N+j]-p[(i-1)*N+j]);
                vy[idx]-=0.5f*(p[i*N+j+1]-p[i*N+j-1]);
            }
        }
        grid.sync();
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 256;
    int STEPS = (argc > 2) ? atoi(argv[2]) : 20;
    int REPEAT = (argc > 3) ? atoi(argv[3]) : 10;
    int jacobi_iters = 50;
    int N2 = N * N;
    int launches_per_step = 2 + 2 * jacobi_iters; // advect + div + 50*(jacobi+copy) + project = 102

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== C16: Stable Fluids Benchmark ===\n");
    printf("GPU: %s, SMs=%d\n", prop.name, prop.multiProcessorCount);
    printf("N=%d (%d cells), steps=%d, repeat=%d, jacobi_iters=%d\n", N, N2, STEPS, REPEAT, jacobi_iters);
    printf("Launches/step: %d\n", launches_per_step);

    float *vx, *vy, *vx_tmp, *vy_tmp, *p, *p_tmp, *div_buf;
    CHECK(cudaMalloc(&vx, N2 * sizeof(float)));
    CHECK(cudaMalloc(&vy, N2 * sizeof(float)));
    CHECK(cudaMalloc(&vx_tmp, N2 * sizeof(float)));
    CHECK(cudaMalloc(&vy_tmp, N2 * sizeof(float)));
    CHECK(cudaMalloc(&p, N2 * sizeof(float)));
    CHECK(cudaMalloc(&p_tmp, N2 * sizeof(float)));
    CHECK(cudaMalloc(&div_buf, N2 * sizeof(float)));

    // Init: vortex velocity field
    std::vector<float> h_vx(N2), h_vy(N2);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float cx = (float)i / N - 0.5f, cy = (float)j / N - 0.5f;
            h_vx[i*N+j] = -cy * 10.0f;
            h_vy[i*N+j] =  cx * 10.0f;
        }

    auto reset = [&]() {
        CHECK(cudaMemcpy(vx, h_vx.data(), N2*sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(vy, h_vy.data(), N2*sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemset(p, 0, N2*sizeof(float)));
        CHECK(cudaMemset(p_tmp, 0, N2*sizeof(float)));
        CHECK(cudaMemset(div_buf, 0, N2*sizeof(float)));
    };
    reset();

    dim3 block(16, 16);
    dim3 grid2d((N + 15) / 16, (N + 15) / 16);
    int copy_grid = (N2 + 255) / 256;
    float dt = 0.1f;

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    // Warmup
    advect_kernel<<<grid2d, block>>>(N, vx, vy, vx, vx_tmp, dt);
    divergence_kernel<<<grid2d, block>>>(N, vx, vy, div_buf);
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
            advect_kernel<<<grid2d, block>>>(N, vx, vy, vx, vx_tmp, dt);
            cudaDeviceSynchronize();
            copy_kernel<<<copy_grid, 256>>>(N2, vx_tmp, vx);
            cudaDeviceSynchronize();
            advect_kernel<<<grid2d, block>>>(N, vx, vy, vy, vy_tmp, dt);
            cudaDeviceSynchronize();
            copy_kernel<<<copy_grid, 256>>>(N2, vy_tmp, vy);
            cudaDeviceSynchronize();
            divergence_kernel<<<grid2d, block>>>(N, vx, vy, div_buf);
            cudaDeviceSynchronize();
            cudaMemset(p, 0, N2*sizeof(float));
            for (int it = 0; it < jacobi_iters; it++) {
                jacobi_pressure_kernel<<<grid2d, block>>>(N, p, div_buf, p_tmp);
                cudaDeviceSynchronize();
                copy_kernel<<<copy_grid, 256>>>(N2, p_tmp, p);
                cudaDeviceSynchronize();
            }
            project_kernel<<<grid2d, block>>>(N, vx, vy, p);
            cudaDeviceSynchronize();
        }
    }, "Sync");

    // Strategy 2: Async
    printf("\n--- Strategy 2: Async ---\n");
    run_timed([&]() {
        for (int s = 0; s < STEPS; s++) {
            advect_kernel<<<grid2d, block>>>(N, vx, vy, vx, vx_tmp, dt);
            copy_kernel<<<copy_grid, 256>>>(N2, vx_tmp, vx);
            advect_kernel<<<grid2d, block>>>(N, vx, vy, vy, vy_tmp, dt);
            copy_kernel<<<copy_grid, 256>>>(N2, vy_tmp, vy);
            divergence_kernel<<<grid2d, block>>>(N, vx, vy, div_buf);
            cudaMemset(p, 0, N2*sizeof(float));
            for (int it = 0; it < jacobi_iters; it++) {
                jacobi_pressure_kernel<<<grid2d, block>>>(N, p, div_buf, p_tmp);
                copy_kernel<<<copy_grid, 256>>>(N2, p_tmp, p);
            }
            project_kernel<<<grid2d, block>>>(N, vx, vy, p);
        }
        cudaDeviceSynchronize();
    }, "Async");

    // Strategy 3: CUDA Graph
    printf("\n--- Strategy 3: CUDA Graph ---\n");
    {
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));

        // Measure instantiation cost
        cudaEvent_t gi0, gi1;
        CHECK(cudaEventCreate(&gi0));
        CHECK(cudaEventCreate(&gi1));

        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int s = 0; s < STEPS; s++) {
            advect_kernel<<<grid2d, block, 0, stream>>>(N, vx, vy, vx, vx_tmp, dt);
            copy_kernel<<<copy_grid, 256, 0, stream>>>(N2, vx_tmp, vx);
            advect_kernel<<<grid2d, block, 0, stream>>>(N, vx, vy, vy, vy_tmp, dt);
            copy_kernel<<<copy_grid, 256, 0, stream>>>(N2, vy_tmp, vy);
            divergence_kernel<<<grid2d, block, 0, stream>>>(N, vx, vy, div_buf);
            cudaMemsetAsync(p, 0, N2*sizeof(float), stream);
            for (int it = 0; it < jacobi_iters; it++) {
                jacobi_pressure_kernel<<<grid2d, block, 0, stream>>>(N, p, div_buf, p_tmp);
                copy_kernel<<<copy_grid, 256, 0, stream>>>(N2, p_tmp, p);
            }
            project_kernel<<<grid2d, block, 0, stream>>>(N, vx, vy, p);
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
              stable_fluids_persistent, block.x * block.y, 0));
        int maxBlocks = numBlocks * prop.multiProcessorCount;
        int needed = grid2d.x * grid2d.y;

        if (needed <= maxBlocks) {
            printf("Persistent: %dx%d = %d blocks (max %d)\n",
                   grid2d.x, grid2d.y, needed, maxBlocks);
            void* args[] = {&N, &vx, &vy, &vx_tmp, &vy_tmp, &p, &p_tmp, &div_buf,
                           &jacobi_iters, &STEPS, &dt};
            run_timed([&]() {
                cudaLaunchCooperativeKernel((void*)stable_fluids_persistent,
                    grid2d, block, args);
                cudaDeviceSynchronize();
            }, "Persistent");
        } else {
            printf("Persistent: N/A (need %d blocks, max %d)\n", needed, maxBlocks);
        }
    }

    // Strategy 5: Grid-Stride Persistent (NEVER N/A)
    printf("\n--- Strategy 5: Grid-Stride Persistent ---\n");
    {
        int gsBSm = 0;
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&gsBSm,
              stable_fluids_persistent_stride, 256, 0));
        int gsMax = gsBSm * prop.multiProcessorCount;
        printf("Grid-stride: %d blocks (always fits)\n", gsMax);
        float dt = 0.1f;
        void* gsArgs[] = {&N, &vx, &vy, &vx_tmp, &vy_tmp, &p, &p_tmp, &div_buf,
                          &jacobi_iters, &STEPS, &dt};
        run_timed([&]() {
            reset();
            cudaLaunchCooperativeKernel((void*)stable_fluids_persistent_stride,
                dim3(gsMax), dim3(256), gsArgs);
            cudaDeviceSynchronize();
        }, "GridStride");
    }

    CHECK(cudaFree(vx)); CHECK(cudaFree(vy));
    CHECK(cudaFree(vx_tmp)); CHECK(cudaFree(vy_tmp));
    CHECK(cudaFree(p)); CHECK(cudaFree(p_tmp)); CHECK(cudaFree(div_buf));
    return 0;
}
