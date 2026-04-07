/**
 * LULESH-like Fusion Benchmark: 3 kernels/step → persistent fused
 *
 * Simplified 2D Lagrangian hydro (Sedov blast):
 *   1. CalcForces: element pressure → node forces (scatter)
 *   2. UpdateNodes: F/m → acceleration → velocity → position
 *   3. UpdateEOS: new volume → density → pressure (equation of state)
 *
 * Build: nvcc -O3 -arch=sm_90 -rdc=true lulesh_fusion_benchmark.cu -o lulesh_fusion -lcudadevrt
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// ======================================================================
// Separate kernels
// ======================================================================

// Forces: pressure gradient → scatter to nodes (element-parallel)
__global__ void k_forces(int NE, int NP1, const float* p_el, const float* vol,
                          const float* xn_x, const float* xn_y,
                          float* fn_x, float* fn_y, int N) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= NE) return;
    int i = e / N, j = e % N;
    int n0=i*(N+1)+j, n1=(i+1)*(N+1)+j, n2=(i+1)*(N+1)+j+1, n3=i*(N+1)+j+1;
    float pr = p_el[e] * vol[e] * 0.25f;
    atomicAdd(&fn_x[n0], -pr); atomicAdd(&fn_y[n0], -pr);
    atomicAdd(&fn_x[n1],  pr); atomicAdd(&fn_y[n1], -pr);
    atomicAdd(&fn_x[n2],  pr); atomicAdd(&fn_y[n2],  pr);
    atomicAdd(&fn_x[n3], -pr); atomicAdd(&fn_y[n3],  pr);
}

// Reset forces
__global__ void k_reset_forces(int NN, float* fn_x, float* fn_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NN) { fn_x[i] = 0; fn_y[i] = 0; }
}

// Update nodes: velocity += F/m * dt, position += v * dt
__global__ void k_update_nodes(int NN, float* vn_x, float* vn_y,
                                float* xn_x, float* xn_y,
                                const float* fn_x, const float* fn_y,
                                const float* mass, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NN) return;
    if (mass[i] > 0) {
        vn_x[i] += fn_x[i] / mass[i] * dt;
        vn_y[i] += fn_y[i] / mass[i] * dt;
    }
    xn_x[i] += vn_x[i] * dt;
    xn_y[i] += vn_y[i] * dt;
}

// EOS: recompute volume, density, pressure from node positions
__global__ void k_eos(int NE, const float* xn_x, const float* xn_y,
                       float* vol, float* rho, float* p_el, const float* e_el,
                       const float* mass_el, int N) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= NE) return;
    int i = e / N, j = e % N;
    int n0=i*(N+1)+j, n1=(i+1)*(N+1)+j, n2=(i+1)*(N+1)+j+1, n3=i*(N+1)+j+1;
    // Shoelace area
    float x0=xn_x[n0],y0=xn_y[n0], x1=xn_x[n1],y1=xn_y[n1];
    float x2=xn_x[n2],y2=xn_y[n2], x3=xn_x[n3],y3=xn_y[n3];
    float new_vol = 0.5f * fabsf((x1-x3)*(y2-y0) - (x2-x0)*(y1-y3));
    new_vol = fmaxf(new_vol, 1e-10f);
    vol[e] = new_vol;
    rho[e] = mass_el[e] / new_vol;
    p_el[e] = (1.4f - 1.0f) * rho[e] * e_el[e];  // ideal gas EOS
}

// ======================================================================
// Persistent fused kernel (all 3 phases + force reset)
// ======================================================================
__global__ void lulesh_persistent(int NE, int NN, int N, float dt, int STEPS,
    float* p_el, float* vol, float* rho, const float* e_el, const float* mass_el,
    float* xn_x, float* xn_y, float* vn_x, float* vn_y,
    float* fn_x, float* fn_y, const float* mass_n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = blockDim.x * gridDim.x;

    for (int step = 0; step < STEPS; step++) {
        // Phase 0: Reset forces
        for (int i = tid; i < NN; i += total) { fn_x[i] = 0; fn_y[i] = 0; }
        cg::this_grid().sync();

        // Phase 1: CalcForces (element-parallel, scatter to nodes)
        for (int e = tid; e < NE; e += total) {
            int i = e / N, j = e % N;
            int n0=i*(N+1)+j, n1=(i+1)*(N+1)+j, n2=(i+1)*(N+1)+j+1, n3=i*(N+1)+j+1;
            float pr = p_el[e] * vol[e] * 0.25f;
            atomicAdd(&fn_x[n0], -pr); atomicAdd(&fn_y[n0], -pr);
            atomicAdd(&fn_x[n1],  pr); atomicAdd(&fn_y[n1], -pr);
            atomicAdd(&fn_x[n2],  pr); atomicAdd(&fn_y[n2],  pr);
            atomicAdd(&fn_x[n3], -pr); atomicAdd(&fn_y[n3],  pr);
        }
        cg::this_grid().sync();

        // Phase 2: UpdateNodes
        for (int i = tid; i < NN; i += total) {
            if (mass_n[i] > 0) {
                vn_x[i] += fn_x[i] / mass_n[i] * dt;
                vn_y[i] += fn_y[i] / mass_n[i] * dt;
            }
            xn_x[i] += vn_x[i] * dt;
            xn_y[i] += vn_y[i] * dt;
        }
        cg::this_grid().sync();

        // Phase 3: EOS
        for (int e = tid; e < NE; e += total) {
            int i = e / N, j = e % N;
            int n0=i*(N+1)+j, n1=(i+1)*(N+1)+j, n2=(i+1)*(N+1)+j+1, n3=i*(N+1)+j+1;
            float x0=xn_x[n0],y0=xn_y[n0], x1=xn_x[n1],y1=xn_y[n1];
            float x2=xn_x[n2],y2=xn_y[n2], x3=xn_x[n3],y3=xn_y[n3];
            float nv = 0.5f*fabsf((x1-x3)*(y2-y0)-(x2-x0)*(y1-y3));
            nv = fmaxf(nv, 1e-10f);
            vol[e] = nv; rho[e] = mass_el[e] / nv;
            p_el[e] = 0.4f * rho[e] * e_el[e];
        }
        cg::this_grid().sync();
    }
}

#define CHECK(call) { auto e = call; if(e) { printf("CUDA error %d at %d\n",e,__LINE__); exit(1); }}

int main(int argc, char** argv) {
    int N = 64;
    int STEPS = 500;
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) STEPS = atoi(argv[2]);
    int NE = N * N;
    int NN = (N+1) * (N+1);
    int B = 256;
    int grid_e = (NE + B - 1) / B;
    int grid_n = (NN + B - 1) / B;
    float dt = 0.0001f;

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SMs=%d)\n", prop.name, prop.multiProcessorCount);
    printf("LULESH-like: N=%d (%d elements, %d nodes), %d steps\n", N, NE, NN, STEPS);
    printf("Per step: reset_forces + calc_forces + update_nodes + EOS = 4 kernels\n\n");

    // Allocate
    float *p_el, *vol, *rho, *e_el, *mass_el;
    float *xn_x, *xn_y, *vn_x, *vn_y, *fn_x, *fn_y, *mass_n;
    CHECK(cudaMalloc(&p_el, NE*4)); CHECK(cudaMalloc(&vol, NE*4));
    CHECK(cudaMalloc(&rho, NE*4)); CHECK(cudaMalloc(&e_el, NE*4));
    CHECK(cudaMalloc(&mass_el, NE*4));
    CHECK(cudaMalloc(&xn_x, NN*4)); CHECK(cudaMalloc(&xn_y, NN*4));
    CHECK(cudaMalloc(&vn_x, NN*4)); CHECK(cudaMalloc(&vn_y, NN*4));
    CHECK(cudaMalloc(&fn_x, NN*4)); CHECK(cudaMalloc(&fn_y, NN*4));
    CHECK(cudaMalloc(&mass_n, NN*4));

    // Init on host
    {
        float* h = (float*)calloc(max(NE,NN), 4);
        // Node positions: grid
        float* hx = (float*)malloc(NN*4); float* hy = (float*)malloc(NN*4);
        float* hm = (float*)malloc(NN*4);
        for (int i = 0; i <= N; i++) for (int j = 0; j <= N; j++) {
            int id = i*(N+1)+j;
            hx[id] = (float)i/N; hy[id] = (float)j/N; hm[id] = 1.0f/NN;
        }
        cudaMemcpy(xn_x, hx, NN*4, cudaMemcpyHostToDevice);
        cudaMemcpy(xn_y, hy, NN*4, cudaMemcpyHostToDevice);
        cudaMemcpy(mass_n, hm, NN*4, cudaMemcpyHostToDevice);
        cudaMemset(vn_x, 0, NN*4); cudaMemset(vn_y, 0, NN*4);

        // Element data
        float* he = (float*)malloc(NE*4); float* hv = (float*)malloc(NE*4);
        float* hme = (float*)malloc(NE*4);
        for (int i = 0; i < NE; i++) {
            he[i] = (i < 9) ? 1.0f : 0.0f;  // Sedov energy deposit in corner
            hv[i] = 1.0f / NE;
            hme[i] = 1.0f / NE;
        }
        cudaMemcpy(e_el, he, NE*4, cudaMemcpyHostToDevice);
        cudaMemcpy(vol, hv, NE*4, cudaMemcpyHostToDevice);
        cudaMemcpy(mass_el, hme, NE*4, cudaMemcpyHostToDevice);
        // Init pressure from EOS
        for (int i = 0; i < NE; i++) h[i] = 0.4f * (1.0f/NE/hv[i]) * he[i];
        cudaMemcpy(p_el, h, NE*4, cudaMemcpyHostToDevice);
        cudaMemcpy(rho, hme, NE*4, cudaMemcpyHostToDevice);  // initial rho = mass/vol = 1

        free(h); free(hx); free(hy); free(hm); free(he); free(hv); free(hme);
    }

    cudaEvent_t t0, t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    float ms; int REPS = 3;

    // Warmup
    for (int i = 0; i < 20; i++) {
        k_reset_forces<<<grid_n,B>>>(NN,fn_x,fn_y);
        k_forces<<<grid_e,B>>>(NE,NN,p_el,vol,xn_x,xn_y,fn_x,fn_y,N);
        k_update_nodes<<<grid_n,B>>>(NN,vn_x,vn_y,xn_x,xn_y,fn_x,fn_y,mass_n,dt);
        k_eos<<<grid_e,B>>>(NE,xn_x,xn_y,vol,rho,p_el,e_el,mass_el,N);
    }
    cudaDeviceSynchronize();

    // [1] Sync loop
    cudaEventRecord(t0);
    for (int rep = 0; rep < REPS; rep++)
    for (int s = 0; s < STEPS; s++) {
        k_reset_forces<<<grid_n,B>>>(NN,fn_x,fn_y);
        k_forces<<<grid_e,B>>>(NE,NN,p_el,vol,xn_x,xn_y,fn_x,fn_y,N);
        k_update_nodes<<<grid_n,B>>>(NN,vn_x,vn_y,xn_x,xn_y,fn_x,fn_y,mass_n,dt);
        k_eos<<<grid_e,B>>>(NE,xn_x,xn_y,vol,rho,p_el,e_el,mass_el,N);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    float sync_us = ms*1000/(STEPS*REPS);
    printf("[1] Sync (4 kernels/step):     %7.1f us/step\n", sync_us);

    // [2] Async loop
    cudaEventRecord(t0);
    for (int rep = 0; rep < REPS; rep++)
    for (int s = 0; s < STEPS; s++) {
        k_reset_forces<<<grid_n,B>>>(NN,fn_x,fn_y);
        k_forces<<<grid_e,B>>>(NE,NN,p_el,vol,xn_x,xn_y,fn_x,fn_y,N);
        k_update_nodes<<<grid_n,B>>>(NN,vn_x,vn_y,xn_x,xn_y,fn_x,fn_y,mass_n,dt);
        k_eos<<<grid_e,B>>>(NE,xn_x,xn_y,vol,rho,p_el,e_el,mass_el,N);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    float async_us = ms*1000/(STEPS*REPS);
    printf("[2] Async (4 kernels, no sync):%7.1f us/step\n", async_us);

    // [3] CUDA Graph
    {
        cudaGraph_t g; cudaGraphExec_t ge;
        cudaStream_t stream; CHECK(cudaStreamCreate(&stream));
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int s = 0; s < STEPS; s++) {
            k_reset_forces<<<grid_n,B,0,stream>>>(NN,fn_x,fn_y);
            k_forces<<<grid_e,B,0,stream>>>(NE,NN,p_el,vol,xn_x,xn_y,fn_x,fn_y,N);
            k_update_nodes<<<grid_n,B,0,stream>>>(NN,vn_x,vn_y,xn_x,xn_y,fn_x,fn_y,mass_n,dt);
            k_eos<<<grid_e,B,0,stream>>>(NE,xn_x,xn_y,vol,rho,p_el,e_el,mass_el,N);
        }
        CHECK(cudaStreamEndCapture(stream, &g));
        CHECK(cudaGraphInstantiate(&ge, g, NULL, NULL, 0));
        cudaGraphLaunch(ge, stream); cudaStreamSynchronize(stream);  // warmup

        cudaEventRecord(t0, stream);
        for (int rep = 0; rep < REPS; rep++) cudaGraphLaunch(ge, stream);
        cudaEventRecord(t1, stream); cudaStreamSynchronize(stream);
        cudaEventElapsedTime(&ms, t0, t1);
        float graph_us = ms*1000/(STEPS*REPS);
        printf("[3] CUDA Graph (4 kern batch): %7.1f us/step\n", graph_us);
        cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaStreamDestroy(stream);
    }

    // [4] Persistent fused
    {
        int numBSm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm, lulesh_persistent, B, 0);
        int maxBlocks = numBSm * prop.multiProcessorCount;
        int need = max(grid_e, grid_n);
        printf("\n  Persistent: need %d blocks, max %d (%d/SM × %d SMs)\n",
               need, maxBlocks, numBSm, prop.multiProcessorCount);

        if (need <= maxBlocks) {
            void* args[] = {(void*)&NE, (void*)&NN, (void*)&N, (void*)&dt, (void*)&STEPS,
                (void*)&p_el, (void*)&vol, (void*)&rho, (void*)&e_el, (void*)&mass_el,
                (void*)&xn_x, (void*)&xn_y, (void*)&vn_x, (void*)&vn_y,
                (void*)&fn_x, (void*)&fn_y, (void*)&mass_n};
            // Warmup
            cudaLaunchCooperativeKernel((void*)lulesh_persistent, dim3(need), dim3(B), args);
            cudaDeviceSynchronize();

            cudaEventRecord(t0);
            for (int rep = 0; rep < REPS; rep++)
                cudaLaunchCooperativeKernel((void*)lulesh_persistent, dim3(need), dim3(B), args);
            cudaEventRecord(t1); cudaDeviceSynchronize();
            cudaEventElapsedTime(&ms, t0, t1);
            float persist_us = ms*1000/(STEPS*REPS);
            printf("[4] Persistent fused (1 kern): %7.1f us/step\n", persist_us);
            printf("\nSpeedups vs Sync: Async=%.1fx Graph=%.1fx Persistent=%.1fx\n",
                   sync_us/async_us, sync_us/(ms*1000/(STEPS*REPS)), sync_us/persist_us);
        } else {
            printf("[4] Persistent: SKIP (grid too large)\n");
        }
    }

    cudaFree(p_el); cudaFree(vol); cudaFree(rho); cudaFree(e_el); cudaFree(mass_el);
    cudaFree(xn_x); cudaFree(xn_y); cudaFree(vn_x); cudaFree(vn_y);
    cudaFree(fn_x); cudaFree(fn_y); cudaFree(mass_n);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}
