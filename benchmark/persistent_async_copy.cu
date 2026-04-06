// Test: can cudaMemcpyAsync overlap with a cooperative (persistent) kernel?
// If yes → persistent kernel can save data every N steps without exiting
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cooperative_groups.h>
#include <thread>
#include <atomic>
#include <chrono>

// Persistent kernel: heat2d, writes to double buffer, signals via mapped flag
__global__ void heat2d_persistent_save(
    int N, float* u, float* v,
    float* save_buf0, float* save_buf1,  // double buffer for saving
    volatile int* flag,                   // mapped memory flag (host-visible)
    int STEPS, int save_interval
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = i * N + j;
    int N2 = N * N;

    for (int s = 0; s < STEPS; s++) {
        // Compute
        if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
            int idx = i*N + j;
            v[idx] = u[idx] + 0.2f*(u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-4*u[idx]);
        }
        cooperative_groups::this_grid().sync();

        // Copy back
        if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
            u[i*N+j] = v[i*N+j];
        }
        cooperative_groups::this_grid().sync();

        // Save to output buffer every save_interval steps
        if (s % save_interval == 0 && s > 0) {
            float* save_buf = ((s / save_interval) % 2 == 0) ? save_buf0 : save_buf1;
            // Each thread copies its portion
            if (tid < N2) {
                save_buf[tid] = u[tid];
            }
            cooperative_groups::this_grid().sync();

            // Thread 0 signals host
            if (tid == 0) {
                __threadfence_system();  // ensure writes visible to host
                *flag = s;               // mapped memory: host can see this
            }
        }
    }
    // Final signal
    if (tid == 0) {
        __threadfence_system();
        *flag = STEPS;
    }
}

// Simple persistent kernel WITHOUT saving (baseline)
__global__ void heat2d_persistent_nosave(int N, float* u, float* v, int STEPS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    for (int s = 0; s < STEPS; s++) {
        if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
            int idx = i*N + j;
            v[idx] = u[idx] + 0.2f*(u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-4*u[idx]);
        }
        cooperative_groups::this_grid().sync();
        if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
            u[i*N+j] = v[i*N+j];
        }
        cooperative_groups::this_grid().sync();
    }
}

#define CHECK(call) { auto e = call; if(e) { printf("CUDA error %d (%s) at line %d\n", e, cudaGetErrorString(e), __LINE__); exit(1); }}

int main() {
    int N = 256;
    int STEPS = 2000;
    int SAVE_INTERVAL = 100;  // save every 100 steps

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SMs=%d)\n", prop.name, prop.multiProcessorCount);

    dim3 block(16, 16), grid;

    // Fall back to the largest multiple-of-16 N that fits cooperative launch.
    int numBSm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm, heat2d_persistent_save, 256, 0);
    int maxBlocks = numBSm * prop.multiProcessorCount;
    int requestedN = N;
    while (N >= 16) {
        grid = dim3((N + 15) / 16, (N + 15) / 16);
        int needBlocks = grid.x * grid.y;
        if (needBlocks <= maxBlocks) break;
        N -= 16;
    }
    if (N < 16) {
        printf("No feasible cooperative-launch grid found for this GPU\n");
        return 1;
    }

    int N2 = N * N;
    printf("Test: N=%d, %d steps, save every %d steps", N, STEPS, SAVE_INTERVAL);
    if (N != requestedN) {
        printf("  (requested %d, reduced for coop launch)", requestedN);
    }
    printf("\n\n");

    float *u, *v, *save_buf0, *save_buf1;
    CHECK(cudaMalloc(&u, N2*4)); CHECK(cudaMalloc(&v, N2*4));
    CHECK(cudaMalloc(&save_buf0, N2*4)); CHECK(cudaMalloc(&save_buf1, N2*4));
    CHECK(cudaMemset(u, 0, N2*4)); CHECK(cudaMemset(v, 0, N2*4));

    // Mapped memory for flag (visible to both host and device)
    volatile int* flag_host;
    volatile int* flag_dev;
    CHECK(cudaHostAlloc((void**)&flag_host, sizeof(int), cudaHostAllocMapped));
    CHECK(cudaHostGetDevicePointer((void**)&flag_dev, (void*)flag_host, 0));
    *flag_host = -1;

    // Host-side buffer for saved data
    float* host_saved;
    CHECK(cudaHostAlloc((void**)&host_saved, N2*4, cudaHostAllocDefault));

    // Copy stream for async D2H
    cudaStream_t copy_stream;
    CHECK(cudaStreamCreate(&copy_stream));

        // Check cooperative launch feasibility
    int needBlocks = grid.x * grid.y;
    printf("Need %d blocks, max %d (occ=%d/SM × %d SMs)\n", needBlocks, maxBlocks, numBSm, prop.multiProcessorCount);
    if (needBlocks > maxBlocks) { printf("Grid too large for cooperative launch\n"); return 1; }

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));

    // ============================================================
    // Test 1: Persistent kernel WITHOUT saving (baseline)
    // ============================================================
    {
        CHECK(cudaMemset(u, 0, N2*4)); CHECK(cudaMemset(v, 0, N2*4));
        void* args[] = {(void*)&N, (void*)&u, (void*)&v, (void*)&STEPS};
        // warmup
        cudaLaunchCooperativeKernel((void*)heat2d_persistent_nosave, grid, block, args);
        cudaDeviceSynchronize();

        cudaEventRecord(t0);
        cudaLaunchCooperativeKernel((void*)heat2d_persistent_nosave, grid, block, args);
        cudaEventRecord(t1); cudaDeviceSynchronize();
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        printf("\n[1] Persistent (no save):      %.2f ms total, %.2f μs/step\n", ms, ms*1000/STEPS);
    }

    // ============================================================
    // Test 2: Persistent kernel + async D2H copy on separate stream
    // ============================================================
    {
        CHECK(cudaMemset(u, 0, N2*4)); CHECK(cudaMemset(v, 0, N2*4));
        *flag_host = -1;

        int saves_done = 0;
        int total_saves = STEPS / SAVE_INTERVAL;

        void* args[] = {
            (void*)&N, (void*)&u, (void*)&v,
            (void*)&save_buf0, (void*)&save_buf1,
            (void*)&flag_dev, (void*)&STEPS, (void*)&SAVE_INTERVAL
        };

        cudaEventRecord(t0);

        // Launch persistent kernel on default stream
        cudaLaunchCooperativeKernel((void*)heat2d_persistent_save, grid, block, args);

        // Host thread: poll flag and issue async copies
        while (saves_done < total_saves) {
            int current_flag = *flag_host;
            if (current_flag >= (saves_done + 1) * SAVE_INTERVAL) {
                // Kernel has completed this save point
                float* src = ((saves_done + 1) % 2 == 0) ? save_buf0 : save_buf1;
                cudaMemcpyAsync(host_saved, src, N2*4, cudaMemcpyDeviceToHost, copy_stream);
                saves_done++;
            }
            // Don't spin too aggressively
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }

        cudaDeviceSynchronize();
        cudaStreamSynchronize(copy_stream);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        // Note: t1 is recorded after sync, so we time differently
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        printf("[2] Persistent + async copy:   %.2f ms total, %.2f μs/step  (%d saves done)\n",
               ms, ms*1000/STEPS, saves_done);
    }

    // ============================================================
    // Test 3: Sync loop + sync copy every N steps (Taichi-like)
    // ============================================================
    {
        CHECK(cudaMemset(u, 0, N2*4)); CHECK(cudaMemset(v, 0, N2*4));
        int cg = (N2+255)/256;

        // Simple kernels for sync loop
        extern __global__ void heat2d_compute(int N, const float* u, float* v);
        extern __global__ void copy_kernel(int N2, const float* s, float* d);

        cudaEventRecord(t0);
        for (int s = 0; s < STEPS; s++) {
            // Compute
            heat2d_persistent_nosave<<<grid, block>>>(N, u, v, 1);  // hack: 1 step
            cudaDeviceSynchronize();
            // Save every SAVE_INTERVAL steps
            if (s % SAVE_INTERVAL == 0 && s > 0) {
                cudaMemcpy(host_saved, u, N2*4, cudaMemcpyDeviceToHost);
            }
        }
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        printf("[3] Sync loop + sync save:     %.2f ms total, %.2f μs/step\n", ms, ms*1000/STEPS);
    }

    // ============================================================
    // Test 4: CUDA Graph (break at save points)
    // ============================================================
    {
        CHECK(cudaMemset(u, 0, N2*4)); CHECK(cudaMemset(v, 0, N2*4));

        // Capture SAVE_INTERVAL steps as a graph
        cudaGraph_t g; cudaGraphExec_t ge;
        cudaStream_t cap_stream;
        CHECK(cudaStreamCreate(&cap_stream));

        int nosave_steps = SAVE_INTERVAL;
        void* args_nosave[] = {(void*)&N, (void*)&u, (void*)&v, (void*)&nosave_steps};

        CHECK(cudaStreamBeginCapture(cap_stream, cudaStreamCaptureModeGlobal));
        cudaLaunchCooperativeKernel((void*)heat2d_persistent_nosave, grid, block, args_nosave, 0, cap_stream);
        CHECK(cudaStreamEndCapture(cap_stream, &g));
        CHECK(cudaGraphInstantiate(&ge, g, NULL, NULL, 0));

        // warmup
        cudaGraphLaunch(ge, cap_stream); cudaStreamSynchronize(cap_stream);

        cudaEventRecord(t0);
        int num_chunks = STEPS / SAVE_INTERVAL;
        for (int c = 0; c < num_chunks; c++) {
            cudaGraphLaunch(ge, cap_stream);
            cudaStreamSynchronize(cap_stream);
            if (c > 0) {
                cudaMemcpy(host_saved, u, N2*4, cudaMemcpyDeviceToHost);
            }
        }
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        printf("[4] Graph (break at save):     %.2f ms total, %.2f μs/step\n", ms, ms*1000/STEPS);

        cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaStreamDestroy(cap_stream);
    }

    printf("\nKey question: does Test 2 match Test 1?\n");
    printf("  If yes → async copy overlaps with persistent kernel (zero overhead saving)\n");
    printf("  If no  → async copy blocks (need to break out to save)\n");

    cudaFree(u); cudaFree(v); cudaFree(save_buf0); cudaFree(save_buf1);
    cudaFreeHost((void*)flag_host); cudaFreeHost(host_saved);
    cudaStreamDestroy(copy_stream);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}
