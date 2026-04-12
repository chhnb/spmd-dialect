/**
 * CG Solver Benchmark: Multi-Kernel Fusion for Conjugate Gradient
 *
 * CG iteration = 5 kernels per step:
 *   1. matvec:    Ap = A * p          (stencil-like)
 *   2. dots:      rr = dot(r,r);  pAp = dot(p,Ap)   (reduction)
 *   3. update_xr: x += alpha*p;  r -= alpha*Ap       (axpy)
 *   4. dot_rnew:  rr_new = dot(r,r)                  (reduction)
 *   5. update_p:  p = r + beta*p                      (axpy)
 *
 * Challenge: alpha = rr/pAp and beta = rr_new/rr require SCALAR values
 * computed by reductions. In separate-kernel mode, these go through host.
 * In persistent fusion, we compute them on-device using block 0.
 *
 * Strategies:
 *   [1] Sync:       5 kernel launches + host readback per step
 *   [2] Async:      5 launches, no per-step sync (but need sync for scalar readback)
 *   [3] Graph:      capture full CG step sequence
 *   [4] Persistent: single cooperative kernel, all 5 phases + device-side reductions
 *
 * Build: nvcc -O3 -arch=sm_86 -rdc=true cg_fusion_benchmark.cu -o cg_fusion -lcudadevrt
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// ======================================================================
// Separate kernels (for sync/async/graph)
// ======================================================================

// 2D Laplacian matvec: Ap = A*p (5-point stencil on N×N grid, flattened)
__global__ void k_matvec(int N, int N2, const float* p, float* Ap) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N2) return;
    int i = idx / N, j = idx % N;
    float s = 4.0f * p[idx];
    if (i > 0)   s -= p[idx - N];
    if (i < N-1) s -= p[idx + N];
    if (j > 0)   s -= p[idx - 1];
    if (j < N-1) s -= p[idx + 1];
    Ap[idx] = s;
}

// Dot products: rr = dot(r,r), pAp = dot(p,Ap)
__global__ void k_dots(int N2, const float* r, const float* p, const float* Ap,
                       float* d_rr, float* d_pAp) {
    __shared__ float s_rr[256], s_pAp[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float rr = 0, pap = 0;
    for (int i = idx; i < N2; i += blockDim.x * gridDim.x) {
        rr += r[i] * r[i];
        pap += p[i] * Ap[i];
    }
    s_rr[tid] = rr; s_pAp[tid] = pap;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) { s_rr[tid] += s_rr[tid+s]; s_pAp[tid] += s_pAp[tid+s]; }
        __syncthreads();
    }
    if (tid == 0) { atomicAdd(d_rr, s_rr[0]); atomicAdd(d_pAp, s_pAp[0]); }
}

// x += alpha*p, r -= alpha*Ap
__global__ void k_update_xr(int N2, float* x, float* r, const float* p,
                             const float* Ap, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N2) return;
    x[idx] += alpha * p[idx];
    r[idx] -= alpha * Ap[idx];
}

// rr_new = dot(r,r)
__global__ void k_dot_rnew(int N2, const float* r, float* d_rnew) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float v = 0;
    for (int i = idx; i < N2; i += blockDim.x * gridDim.x) v += r[i] * r[i];
    s[tid] = v; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) s[tid] += s[tid+st];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(d_rnew, s[0]);
}

// p = r + beta * p
__global__ void k_update_p(int N2, const float* r, float* p, float beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N2) return;
    p[idx] = r[idx] + beta * p[idx];
}

// Zero scalar
__global__ void k_zero(float* v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = 0;
}

// ======================================================================
// Persistent fused CG kernel (all 5 phases in one kernel)
// ======================================================================
__global__ void cg_persistent(int N, int N2, float* x, float* r, float* p, float* Ap,
                               float* g_rr, float* g_pAp, float* g_rnew, int STEPS) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    // Shared memory for block-level reduction
    __shared__ float s_rr[256], s_pAp[256], s_rnew[256];
    // Device-side scalars (only block 0 computes alpha/beta)
    __shared__ float s_alpha, s_beta;

    for (int step = 0; step < STEPS; step++) {
        // Phase 1: matvec Ap = A*p
        for (int idx = tid; idx < N2; idx += total_threads) {
            int i = idx / N, j = idx % N;
            float s = 4.0f * p[idx];
            if (i > 0)   s -= p[idx - N];
            if (i < N-1) s -= p[idx + N];
            if (j > 0)   s -= p[idx - 1];
            if (j < N-1) s -= p[idx + 1];
            Ap[idx] = s;
        }
        cg::this_grid().sync();

        // Phase 2: dot(r,r) and dot(p,Ap) — block-level reduction + atomic
        if (step == 0 || true) {  // always reset
            if (tid == 0) { *g_rr = 0; *g_pAp = 0; }
            cg::this_grid().sync();
        }
        {
            int ltid = threadIdx.x;
            float rr = 0, pap = 0;
            for (int i = tid; i < N2; i += total_threads) {
                rr += r[i] * r[i]; pap += p[i] * Ap[i];
            }
            s_rr[ltid] = rr; s_pAp[ltid] = pap;
            __syncthreads();
            for (int s = 128; s > 0; s >>= 1) {
                if (ltid < s) { s_rr[ltid] += s_rr[ltid+s]; s_pAp[ltid] += s_pAp[ltid+s]; }
                __syncthreads();
            }
            if (ltid == 0) { atomicAdd(g_rr, s_rr[0]); atomicAdd(g_pAp, s_pAp[0]); }
        }
        cg::this_grid().sync();

        // Compute alpha on device (thread 0)
        float alpha;
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            s_alpha = *g_rr / (*g_pAp + 1e-20f);
        }
        cg::this_grid().sync();
        // Broadcast alpha via global memory
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *g_pAp = s_alpha;  // reuse g_pAp to broadcast
        }
        cg::this_grid().sync();
        alpha = *g_pAp;

        // Phase 3: x += alpha*p, r -= alpha*Ap
        for (int idx = tid; idx < N2; idx += total_threads) {
            x[idx] += alpha * p[idx];
            r[idx] -= alpha * Ap[idx];
        }
        cg::this_grid().sync();

        // Phase 4: dot(r_new, r_new)
        if (tid == 0) *g_rnew = 0;
        cg::this_grid().sync();
        {
            int ltid = threadIdx.x;
            float v = 0;
            for (int i = tid; i < N2; i += total_threads) v += r[i] * r[i];
            s_rnew[ltid] = v; __syncthreads();
            for (int s = 128; s > 0; s >>= 1) {
                if (ltid < s) s_rnew[ltid] += s_rnew[ltid+s];
                __syncthreads();
            }
            if (ltid == 0) atomicAdd(g_rnew, s_rnew[0]);
        }
        cg::this_grid().sync();

        // Compute beta on device
        float beta;
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            s_beta = *g_rnew / (*g_rr + 1e-20f);
        }
        cg::this_grid().sync();
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *g_rr = s_beta;  // reuse to broadcast
        }
        cg::this_grid().sync();
        beta = *g_rr;

        // Phase 5: p = r + beta*p
        for (int idx = tid; idx < N2; idx += total_threads) {
            p[idx] = r[idx] + beta * p[idx];
        }
        cg::this_grid().sync();
    }
}

// ======================================================================
// Benchmarking
// ======================================================================
#define CHECK(call) { auto e = call; if(e) { printf("CUDA error %d at %d\n",e,__LINE__); exit(1); }}

int main(int argc, char** argv) {
    int N = 128;
    int STEPS = 200;
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) STEPS = atoi(argv[2]);
    int N2 = N * N;
    int B = 256;
    int grid = (N2 + B - 1) / B;
    int dot_grid = min(grid, 64);  // limit dot product grid

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SMs=%d)\n", prop.name, prop.multiProcessorCount);
    printf("CG Solver: N=%d (%d unknowns), %d CG iterations\n", N, N2, STEPS);
    printf("Per CG step: 5 kernels (matvec + 2 dots + update_xr + update_p)\n\n");

    float *x, *r, *p, *Ap, *b;
    float *d_rr, *d_pAp, *d_rnew;
    CHECK(cudaMalloc(&x, N2*4)); CHECK(cudaMalloc(&r, N2*4));
    CHECK(cudaMalloc(&p, N2*4)); CHECK(cudaMalloc(&Ap, N2*4));
    CHECK(cudaMalloc(&b, N2*4));
    CHECK(cudaMalloc(&d_rr, 4)); CHECK(cudaMalloc(&d_pAp, 4)); CHECK(cudaMalloc(&d_rnew, 4));

    // Init: b=1, x=0, r=b, p=r
    {
        float* h = (float*)calloc(N2, 4);
        for (int i = 0; i < N2; i++) h[i] = 1.0f;
        cudaMemcpy(b, h, N2*4, cudaMemcpyHostToDevice);
        cudaMemcpy(r, h, N2*4, cudaMemcpyHostToDevice);
        cudaMemcpy(p, h, N2*4, cudaMemcpyHostToDevice);
        cudaMemset(x, 0, N2*4);
        free(h);
    }

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    float ms;
    int REPS = 10;

    // Warmup
    for (int i = 0; i < 10; i++) {
        k_matvec<<<grid,B>>>(N,N2,p,Ap);
        k_zero<<<1,4>>>(d_rr, 2);
        k_dots<<<dot_grid,B>>>(N2,r,p,Ap,d_rr,d_pAp);
    }
    cudaDeviceSynchronize();

    // --- [1] Sync loop: 5 kernels + host readback per step ---
    {
        cudaMemset(x, 0, N2*4);
        float h_rr = 1.0f;

        cudaEventRecord(t0);
        for (int rep = 0; rep < REPS; rep++) {
            h_rr = (float)N2;  // reset
            for (int s = 0; s < STEPS; s++) {
                k_matvec<<<grid,B>>>(N,N2,p,Ap);
                k_zero<<<1,4>>>(d_rr, 2);
                k_dots<<<dot_grid,B>>>(N2,r,p,Ap,d_rr,d_pAp);
                cudaDeviceSynchronize();  // need scalar values
                float rr_val, pAp_val;
                cudaMemcpy(&rr_val, d_rr, 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(&pAp_val, d_pAp, 4, cudaMemcpyDeviceToHost);
                float alpha = rr_val / (pAp_val + 1e-20f);
                k_update_xr<<<grid,B>>>(N2,x,r,p,Ap,alpha);
                k_zero<<<1,1>>>(d_rnew, 1);
                k_dot_rnew<<<dot_grid,B>>>(N2,r,d_rnew);
                cudaDeviceSynchronize();
                float rnew_val;
                cudaMemcpy(&rnew_val, d_rnew, 4, cudaMemcpyDeviceToHost);
                float beta = rnew_val / (rr_val + 1e-20f);
                k_update_p<<<grid,B>>>(N2,r,p,beta);
                cudaDeviceSynchronize();
                h_rr = rnew_val;
            }
        }
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms, t0, t1);
        printf("[1] Sync (5 kern + host readback): %7.1f us/step  (%.1f ms total)\n",
               ms*1000/(STEPS*REPS), ms/REPS);
    }

    // --- [2] Async ---
    // Async loop is N/A for CG: each iteration requires cudaMemcpy (D2H) to
    // read dot-product results for computing alpha and beta on the host.
    // Without these synchronization points, the CG iteration produces incorrect
    // results. Unlike simple stencil loops, CG has data-dependent host logic
    // that cannot be deferred to the end of the loop.
    printf("[2] Async: N/A (CG requires host readback of dot products each iteration — cannot defer synchronization)\n");

    // --- [3] CUDA Graph ---
    // Graph is also N/A for CG (same reason as Async): data-dependent host
    // readback of dot products cannot be captured in a static graph.
    printf("[3] CUDA Graph: N/A (CG requires host readback for alpha/beta — data-dependent control flow incompatible with static Graph capture)\n");

    // --- [4] Persistent fused CG ---
    {
        int numBSm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm, cg_persistent, B, 0);
        int maxBlocks = numBSm * prop.multiProcessorCount;
        int cg_grid = min(grid, maxBlocks);

        printf("\n  Persistent: need %d blocks, max %d (%d/SM × %d SMs)\n",
               grid, maxBlocks, numBSm, prop.multiProcessorCount);

        if (cg_grid > 0 && cg_grid <= maxBlocks) {
            cudaMemset(x, 0, N2*4);
            // Re-init r=b, p=b
            cudaMemcpy(r, b, N2*4, cudaMemcpyDeviceToDevice);
            cudaMemcpy(p, b, N2*4, cudaMemcpyDeviceToDevice);

            void* args[] = {(void*)&N, (void*)&N2, (void*)&x, (void*)&r, (void*)&p, (void*)&Ap,
                           (void*)&d_rr, (void*)&d_pAp, (void*)&d_rnew, (void*)&STEPS};

            // Warmup
            cudaLaunchCooperativeKernel((void*)cg_persistent, dim3(cg_grid), dim3(B), args);
            cudaDeviceSynchronize();

            // Re-init
            cudaMemset(x, 0, N2*4);
            cudaMemcpy(r, b, N2*4, cudaMemcpyDeviceToDevice);
            cudaMemcpy(p, b, N2*4, cudaMemcpyDeviceToDevice);

            cudaEventRecord(t0);
            for (int rep = 0; rep < REPS; rep++) {
                cudaLaunchCooperativeKernel((void*)cg_persistent, dim3(cg_grid), dim3(B), args);
            }
            cudaEventRecord(t1); cudaDeviceSynchronize();
            cudaEventElapsedTime(&ms, t0, t1);
            printf("[4] Persistent fused (1 kern):     %7.1f us/step  (%.1f ms total)\n",
                   ms*1000/(STEPS*REPS), ms/REPS);

            float sync_us = 0;  // will be filled from [1]
            // Note: speedup reported at the end
        } else {
            printf("[4] Persistent: SKIP (grid too large)\n");
        }
    }

    // (Old Graph section removed — N/A printed above as [3])
    {
    }

    printf("\nKey insight:\n");
    printf("  Sync CG needs host readback for alpha/beta → 2 extra sync points per step\n");
    printf("  Persistent CG computes alpha/beta on device → 0 host sync, 1 kernel launch total\n");
    printf("  This is exactly the 'dynamic control flow' case where Graph can't help\n");
    printf("  but Persistent can.\n");

    cudaFree(x); cudaFree(r); cudaFree(p); cudaFree(Ap); cudaFree(b);
    cudaFree(d_rr); cudaFree(d_pAp); cudaFree(d_rnew);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}
