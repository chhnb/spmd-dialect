// Overhead elimination: 4 solutions × 2 kernels × 5 sizes
// Solutions: Sync loop, Async loop, CUDA Graph, Persistent Kernel
// Build: nvcc -O3 -arch=sm_90 -rdc=true overhead_solutions.cu -o overhead_solutions -lcudadevrt
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cooperative_groups.h>

__global__ void heat2d(int N, const float* u, float* v) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;v[idx]=u[idx]+0.2f*(u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-4*u[idx]);}
}
__global__ void copy_f(int N2, const float* s, float* d) {
    int i=blockIdx.x*256+threadIdx.x; if(i<N2) d[i]=s[i];
}
__global__ void heat2d_persistent(int N, float* u, float* v, int STEPS) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    for(int s=0;s<STEPS;s++){
        if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;v[idx]=u[idx]+0.2f*(u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-4*u[idx]);}
        cooperative_groups::this_grid().sync();
        if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;u[idx]=v[idx];}
        cooperative_groups::this_grid().sync();
    }
}

__global__ void gs_step(int N, const float* gu, const float* gv, float* gu2, float* gv2) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=1&&i<N-1&&j>=1&&j<N-1){
        int idx=i*N+j;
        float lu=gu[idx-N]+gu[idx+N]+gu[idx-1]+gu[idx+1]-4*gu[idx];
        float lv=gv[idx-N]+gv[idx+N]+gv[idx-1]+gv[idx+1]-4*gv[idx];
        float uvv=gu[idx]*gv[idx]*gv[idx];
        gu2[idx]=gu[idx]+0.16f*lu-uvv+0.06f*(1.0f-gu[idx]);
        gv2[idx]=gv[idx]+0.08f*lv+uvv-0.122f*gv[idx];
    }
}
__global__ void gs_persistent(int N, float* gu, float* gv, float* gu2, float* gv2, int STEPS) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    for(int s=0;s<STEPS;s++){
        if(i>=1&&i<N-1&&j>=1&&j<N-1){
            int idx=i*N+j;
            float lu=gu[idx-N]+gu[idx+N]+gu[idx-1]+gu[idx+1]-4*gu[idx];
            float lv=gv[idx-N]+gv[idx+N]+gv[idx-1]+gv[idx+1]-4*gv[idx];
            float uvv=gu[idx]*gv[idx]*gv[idx];
            gu2[idx]=gu[idx]+0.16f*lu-uvv+0.06f*(1.0f-gu[idx]);
            gv2[idx]=gv[idx]+0.08f*lv+uvv-0.122f*gv[idx];
        }
        cooperative_groups::this_grid().sync();
        if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;gu[idx]=gu2[idx];gv[idx]=gv2[idx];}
        cooperative_groups::this_grid().sync();
    }
}

#define CHECK(call) { auto e = call; if(e) { printf("CUDA error %d at line %d\n", e, __LINE__); exit(1); }}

struct Result { float sync_us, async_us, graph_us, persistent_us; };

Result test_heat(int N, int STEPS) {
    int N2=N*N; float *u,*v;
    CHECK(cudaMalloc(&u,N2*4)); CHECK(cudaMalloc(&v,N2*4));
    CHECK(cudaMemset(u,0,N2*4)); CHECK(cudaMemset(v,0,N2*4));
    dim3 block(16,16),grid((N+15)/16,(N+15)/16); int cg=(N2+255)/256;
    cudaEvent_t t0,t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    for(int i=0;i<20;i++){heat2d<<<grid,block>>>(N,u,v);copy_f<<<cg,256>>>(N2,v,u);}
    cudaDeviceSynchronize();
    Result r; float ms;

    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){heat2d<<<grid,block>>>(N,u,v);copy_f<<<cg,256>>>(N2,v,u);cudaDeviceSynchronize();}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1); r.sync_us=ms*1000/STEPS;

    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){heat2d<<<grid,block>>>(N,u,v);copy_f<<<cg,256>>>(N2,v,u);}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1); r.async_us=ms*1000/STEPS;

    {int REPS=5;cudaGraph_t g;cudaGraphExec_t ge;
     cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
     CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
     for(int s=0;s<STEPS;s++){heat2d<<<grid,block,0,stream>>>(N,u,v);copy_f<<<cg,256,0,stream>>>(N2,v,u);}
     CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&ge,g,NULL,NULL,0));
     for(int i=0;i<3;i++){cudaGraphLaunch(ge,stream);cudaStreamSynchronize(stream);}
     cudaEventRecord(t0,stream);for(int i=0;i<REPS;i++)cudaGraphLaunch(ge,stream);
     cudaEventRecord(t1,stream);cudaStreamSynchronize(stream);
     cudaEventElapsedTime(&ms,t0,t1);r.graph_us=ms*1000/(STEPS*REPS);
     cudaGraphExecDestroy(ge);cudaGraphDestroy(g);cudaStreamDestroy(stream);}

    {int numBSm=0;cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
     cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm,heat2d_persistent,256,0);
     int maxB=numBSm*prop.multiProcessorCount;
     if((int)(grid.x*grid.y)<=maxB){
         void*args[]={(void*)&N,(void*)&u,(void*)&v,(void*)&STEPS};
         cudaLaunchCooperativeKernel((void*)heat2d_persistent,grid,block,args);cudaDeviceSynchronize();
         int REPS=5;cudaEventRecord(t0);
         for(int i=0;i<REPS;i++)cudaLaunchCooperativeKernel((void*)heat2d_persistent,grid,block,args);
         cudaEventRecord(t1);cudaDeviceSynchronize();
         cudaEventElapsedTime(&ms,t0,t1);r.persistent_us=ms*1000/(STEPS*REPS);
     } else r.persistent_us=-1;}

    cudaEventDestroy(t0);cudaEventDestroy(t1);cudaFree(u);cudaFree(v);
    return r;
}

Result test_gs(int N, int STEPS) {
    int N2=N*N; float *gu,*gv,*gu2,*gv2;
    CHECK(cudaMalloc(&gu,N2*4));CHECK(cudaMalloc(&gv,N2*4));
    CHECK(cudaMalloc(&gu2,N2*4));CHECK(cudaMalloc(&gv2,N2*4));
    CHECK(cudaMemset(gu,0,N2*4));CHECK(cudaMemset(gv,0,N2*4));
    dim3 block(16,16),grid((N+15)/16,(N+15)/16); int cg=(N2+255)/256;
    cudaEvent_t t0,t1;CHECK(cudaEventCreate(&t0));CHECK(cudaEventCreate(&t1));
    for(int i=0;i<20;i++){gs_step<<<grid,block>>>(N,gu,gv,gu2,gv2);copy_f<<<cg,256>>>(N2,gu2,gu);copy_f<<<cg,256>>>(N2,gv2,gv);}
    cudaDeviceSynchronize();
    Result r; float ms;

    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){gs_step<<<grid,block>>>(N,gu,gv,gu2,gv2);copy_f<<<cg,256>>>(N2,gu2,gu);copy_f<<<cg,256>>>(N2,gv2,gv);cudaDeviceSynchronize();}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1);r.sync_us=ms*1000/STEPS;

    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){gs_step<<<grid,block>>>(N,gu,gv,gu2,gv2);copy_f<<<cg,256>>>(N2,gu2,gu);copy_f<<<cg,256>>>(N2,gv2,gv);}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1);r.async_us=ms*1000/STEPS;

    {int REPS=5;cudaGraph_t g;cudaGraphExec_t ge;
     cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
     CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
     for(int s=0;s<STEPS;s++){gs_step<<<grid,block,0,stream>>>(N,gu,gv,gu2,gv2);copy_f<<<cg,256,0,stream>>>(N2,gu2,gu);copy_f<<<cg,256,0,stream>>>(N2,gv2,gv);}
     CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&ge,g,NULL,NULL,0));
     for(int i=0;i<3;i++){cudaGraphLaunch(ge,stream);cudaStreamSynchronize(stream);}
     cudaEventRecord(t0,stream);for(int i=0;i<REPS;i++)cudaGraphLaunch(ge,stream);
     cudaEventRecord(t1,stream);cudaStreamSynchronize(stream);
     cudaEventElapsedTime(&ms,t0,t1);r.graph_us=ms*1000/(STEPS*REPS);
     cudaGraphExecDestroy(ge);cudaGraphDestroy(g);cudaStreamDestroy(stream);}

    {int numBSm=0;cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
     cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm,gs_persistent,256,0);
     int maxB=numBSm*prop.multiProcessorCount;
     if((int)(grid.x*grid.y)<=maxB){
         void*args[]={(void*)&N,(void*)&gu,(void*)&gv,(void*)&gu2,(void*)&gv2,(void*)&STEPS};
         cudaLaunchCooperativeKernel((void*)gs_persistent,grid,block,args);cudaDeviceSynchronize();
         int REPS=5;cudaEventRecord(t0);
         for(int i=0;i<REPS;i++)cudaLaunchCooperativeKernel((void*)gs_persistent,grid,block,args);
         cudaEventRecord(t1);cudaDeviceSynchronize();
         cudaEventElapsedTime(&ms,t0,t1);r.persistent_us=ms*1000/(STEPS*REPS);
     } else r.persistent_us=-1;}

    cudaEventDestroy(t0);cudaEventDestroy(t1);
    cudaFree(gu);cudaFree(gv);cudaFree(gu2);cudaFree(gv2);
    return r;
}

int main(){
    cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
    printf("GPU: %s (SMs=%d, Compute %d.%d)\n\n",prop.name,prop.multiProcessorCount,prop.major,prop.minor);
    printf("%-25s %10s %10s %10s %10s  | Speedups over Sync\n","Kernel","Sync","Async","Graph","Persist");
    printf("%-25s %10s %10s %10s %10s  | %8s %8s %8s\n","","(us)","(us)","(us)","(us)","Async","Graph","Persist");
    printf("%.25s %.10s %.10s %.10s %.10s  | %.8s %.8s %.8s\n",
           "-------------------------","----------","----------","----------","----------","--------","--------","--------");

    struct {const char*name;int N;int steps;int type;} cases[]={
        {"Heat2D 128sq",128,2000,0},{"Heat2D 256sq",256,1000,0},{"Heat2D 512sq",512,1000,0},
        {"Heat2D 1024sq",1024,500,0},{"Heat2D 2048sq",2048,200,0},
        {"GrayScott 128sq",128,2000,1},{"GrayScott 256sq",256,1000,1},
        {"GrayScott 512sq",512,500,1},{"GrayScott 1024sq",1024,200,1},
    };
    for(auto&c:cases){
        Result r=(c.type==0)?test_heat(c.N,c.steps):test_gs(c.N,c.steps);
        char ps[32],sa[16],sg[16],sp[16];
        if(r.persistent_us<0){snprintf(ps,32,"%10s","N/A");snprintf(sp,16,"%8s","N/A");}
        else{snprintf(ps,32,"%10.2f",r.persistent_us);snprintf(sp,16,"%7.1fx",r.sync_us/r.persistent_us);}
        snprintf(sa,16,"%7.1fx",r.sync_us/r.async_us);
        snprintf(sg,16,"%7.1fx",r.sync_us/r.graph_us);
        printf("%-25s %10.2f %10.2f %10.2f %s  | %8s %8s %8s\n",
               c.name,r.sync_us,r.async_us,r.graph_us,ps,sa,sg,sp);
    }
}
