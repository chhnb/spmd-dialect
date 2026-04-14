/**
 * Grid-Stride Persistent Kernel PoC
 *
 * Solves the cooperative launch grid limit by using a grid-stride loop.
 * Each thread handles MULTIPLE cells instead of one fixed cell.
 * Grid size = maxCooperativeBlocks (always fits), not problem-size-dependent.
 *
 * Build: nvcc -O3 -arch=sm_80 -rdc=true gridstride_persistent_poc.cu -o gridstride_persistent_poc -lcudadevrt
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CHECK(call) do { cudaError_t e=(call); if(e){fprintf(stderr,"CUDA %d (%s) at %s:%d\n",e,cudaGetErrorString(e),__FILE__,__LINE__);exit(1);}} while(0)

__global__ void heat2d_step(int N, const float* u, float* v) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    if(i>0&&i<N-1&&j>0&&j<N-1){int idx=i*N+j;v[idx]=u[idx]+0.2f*(u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-4*u[idx]);}
}
__global__ void copy_f(int N2, const float* s, float* d) {
    int i=blockIdx.x*256+threadIdx.x; if(i<N2) d[i]=s[i];
}

// Original persistent (fixed grid — may N/A for large N)
__global__ void heat2d_persistent_fixed(int N, float* u, float* v, int STEPS) {
    cg::grid_group grid = cg::this_grid();
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    for(int s=0;s<STEPS;s++){
        if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;v[idx]=u[idx]+0.2f*(u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-4*u[idx]);}
        grid.sync();
        if(i<N&&j<N){int idx=i*N+j;u[idx]=v[idx];}
        grid.sync();
    }
}

// NEW: Grid-stride persistent (always fits — no grid limit!)
__global__ void heat2d_persistent_stride(int N, float* u, float* v, int STEPS) {
    cg::grid_group grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int N2 = N * N;

    for (int s = 0; s < STEPS; s++) {
        // Stencil phase: grid-stride loop over all cells
        for (int idx = tid; idx < N2; idx += stride) {
            int i = idx / N, j = idx % N;
            if (i > 0 && i < N-1 && j > 0 && j < N-1)
                v[idx] = u[idx] + 0.2f*(u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-4*u[idx]);
        }
        grid.sync();
        // Copy phase: grid-stride
        for (int idx = tid; idx < N2; idx += stride)
            u[idx] = v[idx];
        grid.sync();
    }
}

int main(int argc, char** argv) {
    int N = (argc>1) ? atoi(argv[1]) : 256;
    int STEPS = (argc>2) ? atoi(argv[2]) : 100;
    int REPEAT = (argc>3) ? atoi(argv[3]) : 10;

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== Grid-Stride Persistent Kernel PoC ===\n");
    printf("GPU: %s, SMs=%d\n", prop.name, prop.multiProcessorCount);
    printf("N=%d, steps=%d\n\n", N, STEPS);

    float *d_u, *d_v;
    int N2 = N*N;
    CHECK(cudaMalloc(&d_u, N2*sizeof(float)));
    CHECK(cudaMalloc(&d_v, N2*sizeof(float)));
    std::vector<float> h_u(N2, 0.0f);
    h_u[N/2*N+N/2] = 0.01f;

    auto reset = [&](){
        CHECK(cudaMemcpy(d_u,h_u.data(),N2*4,cudaMemcpyHostToDevice));
        CHECK(cudaMemset(d_v,0,N2*4));
    };

    dim3 block2d(16,16), grid2d((N+15)/16,(N+15)/16);
    int cpBlk=256, cpGrd=(N2+255)/256;

    // Compute maxCooperativeBlocks for grid-stride persistent
    int maxBlocksPerSM = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM,
          heat2d_persistent_stride, 256, 0));
    int maxCoopBlocks = maxBlocksPerSM * prop.multiProcessorCount;
    printf("Max cooperative blocks: %d (%d per SM × %d SMs)\n",
           maxCoopBlocks, maxBlocksPerSM, prop.multiProcessorCount);
    printf("Fixed grid needs: %d blocks\n", (int)(grid2d.x * grid2d.y));
    printf("Grid-stride uses: %d blocks (always fits!)\n\n", maxCoopBlocks);

    // ====== Host Graph Full Capture (reference) ======
    float graph_med = 0;
    std::vector<float> h_ref(N2);
    {
        cudaStream_t s; CHECK(cudaStreamCreate(&s));
        cudaGraph_t g; cudaGraphExec_t ge;
        reset();
        CHECK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
        for(int st=0;st<STEPS;st++){heat2d_step<<<grid2d,block2d,0,s>>>(N,d_u,d_v);copy_f<<<cpGrd,cpBlk,0,s>>>(N2,d_v,d_u);}
        CHECK(cudaStreamEndCapture(s,&g));
        CHECK(cudaGraphInstantiate(&ge,g,nullptr,nullptr,0));
        for(int w=0;w<5;w++){reset();cudaGraphLaunch(ge,s);cudaStreamSynchronize(s);}
        cudaEvent_t t0,t1;CHECK(cudaEventCreate(&t0));CHECK(cudaEventCreate(&t1));
        std::vector<float> times;
        for(int r=0;r<REPEAT;r++){
            reset();cudaEventRecord(t0,s);cudaGraphLaunch(ge,s);
            cudaEventRecord(t1,s);cudaEventSynchronize(t1);
            float ms;cudaEventElapsedTime(&ms,t0,t1);times.push_back(ms);
        }
        std::sort(times.begin(),times.end());
        graph_med=times[REPEAT/2];
        printf("[Graph Full]         %.3f ms = %.2f us/step\n",graph_med,graph_med*1000.f/STEPS);
        reset();cudaGraphLaunch(ge,s);cudaStreamSynchronize(s);
        CHECK(cudaMemcpy(h_ref.data(),d_u,N2*4,cudaMemcpyDeviceToHost));
        cudaGraphExecDestroy(ge);cudaGraphDestroy(g);cudaStreamDestroy(s);
        cudaEventDestroy(t0);cudaEventDestroy(t1);
    }

    // ====== Fixed-grid Persistent (may N/A) ======
    float fixed_med = -1;
    {
        int fixedBlocks = (int)(grid2d.x * grid2d.y);
        int numBSm = 0;
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm,
              heat2d_persistent_fixed, 256, 0));
        int maxB = numBSm * prop.multiProcessorCount;

        if (fixedBlocks <= maxB) {
            void* args[] = {(void*)&N, (void*)&d_u, (void*)&d_v, (void*)&STEPS};
            // Warmup
            for(int w=0;w<5;w++){
                reset();
                cudaLaunchCooperativeKernel((void*)heat2d_persistent_fixed,grid2d,block2d,args);
                cudaDeviceSynchronize();
            }
            cudaEvent_t t0,t1;CHECK(cudaEventCreate(&t0));CHECK(cudaEventCreate(&t1));
            std::vector<float> times;
            for(int r=0;r<REPEAT;r++){
                reset();cudaEventRecord(t0);
                cudaLaunchCooperativeKernel((void*)heat2d_persistent_fixed,grid2d,block2d,args);
                cudaEventRecord(t1);cudaDeviceSynchronize();
                float ms;cudaEventElapsedTime(&ms,t0,t1);times.push_back(ms);
            }
            std::sort(times.begin(),times.end());
            fixed_med=times[REPEAT/2];
            printf("[Persistent Fixed]   %.3f ms = %.2f us/step\n",fixed_med,fixed_med*1000.f/STEPS);
            cudaEventDestroy(t0);cudaEventDestroy(t1);
        } else {
            printf("[Persistent Fixed]   N/A (need %d blocks, max %d)\n", fixedBlocks, maxB);
        }
    }

    // ====== Grid-Stride Persistent (NEVER N/A!) ======
    float stride_med = 0;
    {
        dim3 strideGrid(maxCoopBlocks), strideBlock(256);
        void* args[] = {(void*)&N, (void*)&d_u, (void*)&d_v, (void*)&STEPS};

        for(int w=0;w<5;w++){
            reset();
            cudaLaunchCooperativeKernel((void*)heat2d_persistent_stride,strideGrid,strideBlock,args);
            cudaDeviceSynchronize();
        }
        cudaEvent_t t0,t1;CHECK(cudaEventCreate(&t0));CHECK(cudaEventCreate(&t1));
        std::vector<float> times;
        for(int r=0;r<REPEAT;r++){
            reset();cudaEventRecord(t0);
            cudaLaunchCooperativeKernel((void*)heat2d_persistent_stride,strideGrid,strideBlock,args);
            cudaEventRecord(t1);cudaDeviceSynchronize();
            float ms;cudaEventElapsedTime(&ms,t0,t1);times.push_back(ms);
        }
        std::sort(times.begin(),times.end());
        stride_med=times[REPEAT/2];
        printf("[Persistent Stride]  %.3f ms = %.2f us/step\n",stride_med,stride_med*1000.f/STEPS);

        // Verify correctness
        reset();
        cudaLaunchCooperativeKernel((void*)heat2d_persistent_stride,strideGrid,strideBlock,args);
        cudaDeviceSynchronize();
        std::vector<float> h_out(N2);
        CHECK(cudaMemcpy(h_out.data(),d_u,N2*4,cudaMemcpyDeviceToHost));
        float maxdiff=0;
        for(int i=0;i<N2;i++){float d=fabsf(h_ref[i]-h_out[i]);if(d>maxdiff)maxdiff=d;}
        printf("  Correctness: max_diff=%.2e (%s)\n",maxdiff,maxdiff<1e-5f?"MATCH":"MISMATCH");

        cudaEventDestroy(t0);cudaEventDestroy(t1);
    }

    // Summary
    printf("\n--- N=%d, %d steps ---\n", N, STEPS);
    printf("Graph Full:        %.2f us/step\n", graph_med*1000.f/STEPS);
    if(fixed_med>0) printf("Persistent Fixed:  %.2f us/step\n", fixed_med*1000.f/STEPS);
    else printf("Persistent Fixed:  N/A\n");
    printf("Persistent Stride: %.2f us/step  ← NEVER N/A\n", stride_med*1000.f/STEPS);

    CHECK(cudaFree(d_u));CHECK(cudaFree(d_v));
    return 0;
}
