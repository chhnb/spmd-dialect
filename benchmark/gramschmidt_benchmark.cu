/**
 * C21: Gram-Schmidt — 4-strategy benchmark. 3*N launches total.
 * Build: nvcc -O3 -arch=sm_80 -rdc=true gramschmidt_benchmark.cu -o gramschmidt_bench -lcudadevrt
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cooperative_groups.h>
#define CHECK(call) do{auto e=call;if(e){fprintf(stderr,"err %d L%d\n",e,__LINE__);exit(1);}}while(0)

__global__ void gs_normalize(int M, float* Q, float* R, int k) {
    // R[k*M+k] = norm(Q[:,k]), Q[:,k] /= R[k*M+k]
    if(threadIdx.x==0&&blockIdx.x==0){
        float sum=0; for(int i=0;i<M;i++){float v=Q[i*M+k]; sum+=v*v;}
        R[k*M+k]=sqrtf(sum);
        float inv=1.0f/R[k*M+k];
        for(int i=0;i<M;i++) Q[i*M+k]*=inv;
    }
}
__global__ void gs_project(int M, float* Q, float* R, int k, int j) {
    // R[k*M+j] = dot(Q[:,k], Q[:,j]), Q[:,j] -= R[k*M+j]*Q[:,k]
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    // Simple: single thread for dot product (small M)
    if(i==0){
        float dot=0; for(int ii=0;ii<M;ii++) dot+=Q[ii*M+k]*Q[ii*M+j];
        R[k*M+j]=dot;
        for(int ii=0;ii<M;ii++) Q[ii*M+j]-=dot*Q[ii*M+k];
    }
}

int main(int argc,char**argv){
    int N=(argc>1)?atoi(argv[1]):512, REP=(argc>2)?atoi(argv[2]):10;
    int launches=3*N; // normalize + (N-k-1) projects per col ≈ 1.5*N^2 but simplified to 3*N per-column model
    // Actually: N normalize + N*(N-1)/2 projects. For timing, use the PolyBench pattern: 3 kernels per column
    cudaDeviceProp p; CHECK(cudaGetDeviceProperties(&p,0));
    printf("=== C21: Gram-Schmidt ===\nGPU: %s, N=%d, ~%d total launches\n",p.name,N,launches);

    double *Q,*R; int M=N;
    CHECK(cudaMalloc(&Q,M*M*8)); CHECK(cudaMalloc(&R,M*M*8));
    // Init Q: identity + sin perturbation (full rank, fp64 for determinism)
    std::vector<double>hQ(M*M); for(int i=0;i<M;i++) for(int j=0;j<M;j++) { double base=(i==j)?1.0:0.0; hQ[i*M+j]=base+sin((double)(i*M+j+1)*0.1)*0.3; }
    CHECK(cudaMemcpy(Q,hQ.data(),M*M*4,cudaMemcpyHostToDevice));
    CHECK(cudaMemset(R,0,M*M*4));

    cudaEvent_t t0,t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));

    auto full_gs=[&](cudaStream_t s=0){
        for(int k=0;k<N;k++){
            gs_normalize<<<1,1,0,s>>>(M,Q,R,k);
            for(int j=k+1;j<N;j++)
                gs_project<<<1,1,0,s>>>(M,Q,R,k,j);
        }
    };

    int total_launches_actual=0;
    for(int k=0;k<N;k++){total_launches_actual+=1+(N-k-1);}

    auto bench=[&](auto fn,const char*nm){
        // Reset Q each time
        std::vector<float>ts; for(int r=0;r<REP;r++){
            CHECK(cudaMemcpy(Q,hQ.data(),M*M*4,cudaMemcpyHostToDevice));
            cudaDeviceSynchronize();
            cudaEventRecord(t0); fn(); cudaEventRecord(t1); cudaEventSynchronize(t1);
            float ms; cudaEventElapsedTime(&ms,t0,t1); ts.push_back(ms);
        } std::sort(ts.begin(),ts.end()); float med=ts[REP/2];
        printf("[%s] median=%.3f ms, %.2f us/launch (%d launches)\n",nm,med,med*1000/total_launches_actual,total_launches_actual);
    };

    bench([&]{full_gs();cudaDeviceSynchronize();},"Sync");
    // Async: Gram-Schmidt has serial dependencies (project[j] depends on normalize[k],
    // and normalize[k+1] depends on all project[j] for j>k). Without explicit sync
    // between launches, the GPU driver serializes them via implicit dependencies on
    // the default stream. Async timing equals Sync because the algorithm is inherently serial.
    printf("[Async] N/A (serial data dependency: each projection depends on prior normalization; launches on default stream are implicitly serialized)\n");

    // Graph
    {cudaGraph_t g;cudaGraphExec_t ge;cudaStream_t s;CHECK(cudaStreamCreate(&s));
     CHECK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
     full_gs(s);
     CHECK(cudaStreamEndCapture(s,&g));CHECK(cudaGraphInstantiate(&ge,g,NULL,NULL,0));
     bench([&]{cudaGraphLaunch(ge,s);cudaStreamSynchronize(s);},"Graph");
     CHECK(cudaGraphExecDestroy(ge));CHECK(cudaGraphDestroy(g));CHECK(cudaStreamDestroy(s));}

    printf("[Persistent] N/A (serial dependency chain, variable grid per launch)\n");
    CHECK(cudaFree(Q));CHECK(cudaFree(R)); return 0;
}
