/**
 * C20: ADI — 4-strategy benchmark. ~2*(N-2) launches per timestep.
 * Build: nvcc -O3 -arch=sm_80 -rdc=true adi_benchmark.cu -o adi_bench -lcudadevrt
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cooperative_groups.h>
#define CHECK(call) do{auto e=call;if(e){fprintf(stderr,"err %d L%d\n",e,__LINE__);exit(1);}}while(0)

__global__ void adi_row(int N, float* u, float* v, int row, float a, float b) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=1&&j<N-1){int idx=row*N+j; v[idx]=a*u[idx-N]+b*u[idx]+a*u[idx+N];}
}
__global__ void adi_col(int N, float* u, float* v, int col, float a, float b) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=1&&i<N-1){int idx=i*N+col; u[idx]=a*v[idx-1]+b*v[idx]+a*v[idx+1];}
}

int main(int argc,char**argv){
    int N=(argc>1)?atoi(argv[1]):512, TS=(argc>2)?atoi(argv[2]):5, REP=(argc>3)?atoi(argv[3]):10;
    int lps=2*(N-2);
    cudaDeviceProp p; CHECK(cudaGetDeviceProperties(&p,0));
    printf("=== C20: ADI ===\nGPU: %s, N=%d, tsteps=%d, launches/step=%d\n",p.name,N,TS,lps);
    float *u,*v; int N2=N*N;
    CHECK(cudaMalloc(&u,N2*4)); CHECK(cudaMalloc(&v,N2*4));
    CHECK(cudaMemset(u,0,N2*4)); CHECK(cudaMemset(v,0,N2*4));
    float a=0.1f,b=0.8f; int thr=256,blk=(N+255)/256;
    cudaEvent_t t0,t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    for(int i=1;i<N-1;i++) adi_row<<<blk,thr>>>(N,u,v,i,a,b);
    cudaDeviceSynchronize();

    auto step=[&](cudaStream_t s=0){
        for(int i=1;i<N-1;i++) adi_row<<<blk,thr,0,s>>>(N,u,v,i,a,b);
        for(int j=1;j<N-1;j++) adi_col<<<blk,thr,0,s>>>(N,u,v,j,a,b);
    };
    auto bench=[&](auto fn,const char*nm){
        std::vector<float>ts; for(int r=0;r<REP;r++){
            cudaEventRecord(t0); fn(); cudaEventRecord(t1); cudaEventSynchronize(t1);
            float ms; cudaEventElapsedTime(&ms,t0,t1); ts.push_back(ms);
        } std::sort(ts.begin(),ts.end()); float med=ts[REP/2];
        printf("[%s] %d tsteps: median=%.3f ms, %.2f us/step, %.2f us/launch\n",nm,TS,med,med*1000/TS,med*1000/TS/lps);
    };

    bench([&]{for(int t=0;t<TS;t++){step();cudaDeviceSynchronize();}},"Sync");
    bench([&]{for(int t=0;t<TS;t++)step();cudaDeviceSynchronize();},"Async");

    // Graph
    {cudaGraph_t g;cudaGraphExec_t ge;cudaStream_t s;CHECK(cudaStreamCreate(&s));
     CHECK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
     for(int t=0;t<TS;t++){for(int i=1;i<N-1;i++)adi_row<<<blk,thr,0,s>>>(N,u,v,i,a,b);
       for(int j=1;j<N-1;j++)adi_col<<<blk,thr,0,s>>>(N,u,v,j,a,b);}
     CHECK(cudaStreamEndCapture(s,&g)); CHECK(cudaGraphInstantiate(&ge,g,NULL,NULL,0));
     bench([&]{cudaGraphLaunch(ge,s);cudaStreamSynchronize(s);},"Graph");
     CHECK(cudaGraphExecDestroy(ge));CHECK(cudaGraphDestroy(g));CHECK(cudaStreamDestroy(s));}

    // Persistent is N/A: ADI launches one kernel per row/column with different
    // row/col index parameters. A cooperative persistent kernel would require all
    // rows/columns to be processed by the same grid, but each row kernel only
    // processes N elements (1 block), while persistent requires the grid to be
    // fixed at launch. The total grid would need N-2 concurrent blocks per row
    // direction, exceeding cooperative limits for large N. Additionally, the
    // row-sweep and column-sweep phases have a sequential dependency between them.
    printf("[Persistent] N/A (each launch processes a different row/col with 1-block grid; cooperative launch requires fixed grid size, and row↔column phase has sequential dependency)\n");
    CHECK(cudaFree(u));CHECK(cudaFree(v)); return 0;
}
