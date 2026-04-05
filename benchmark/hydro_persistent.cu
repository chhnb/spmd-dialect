// Hydro-cal F2: Persistent kernel vs CUDA Graph vs sync loop
// Real workload: 6675 cells, 26700 edges, 2 kernels/step (flux + update)
// Grid = 105 blocks << 148 SMs → perfect for cooperative launch
// Build: nvcc -O3 -arch=sm_90 -rdc=true hydro_persistent.cu -o hydro_persistent -lcudadevrt
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cooperative_groups.h>

constexpr float G = 9.81f;
constexpr float HALF_G = 4.905f;
constexpr float HM1 = 0.001f;

__global__ void calculate_flux(int NEDGE, const float* H, const float* U, const float* V,
                                const int* NAC, const float* KLAS,
                                const float* COSF, const float* SINF, const float* SIDE,
                                float* FLUX0, float* FLUX1, float* FLUX2, float* FLUX3, int CELL) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= NEDGE) return;
    int i = e / 4;
    if (KLAS[e] == 0.0f) {
        int nc = NAC[e] - 1;
        if (nc < 0 || nc >= CELL) { FLUX0[e]=FLUX1[e]=FLUX2[e]=FLUX3[e]=0; return; }
        float hi=H[i],ui=U[i],vi=V[i], hn=H[nc],un=U[nc],vn=V[nc];
        float cf=COSF[e],sf=SINF[e],sl=SIDE[e];
        float ul=ui*cf+vi*sf, vl=-ui*sf+vi*cf, ur=un*cf+vn*sf, vr=-un*sf+vn*cf;
        float cl=sqrtf(G*hi),cr=sqrtf(G*hn);
        float smax=fmaxf(fabsf(ul)+cl,fabsf(ur)+cr);
        float f0=0.5f*sl*(hi*ul+hn*ur-smax*(hn-hi));
        float f1=0.5f*sl*(hi*ul*ul+HALF_G*hi*hi+hn*ur*ur+HALF_G*hn*hn-smax*(hn*ur-hi*ul));
        float f2=0.5f*sl*(hi*ul*vl+hn*ur*vr-smax*(hn*vr-hi*vl));
        FLUX0[e]=f0; FLUX1[e]=f1*cf-f2*sf; FLUX2[e]=f1*sf+f2*cf;
        FLUX3[e]=HALF_G*0.5f*sl*(hi*hi+hn*hn);
    } else {
        float hi=H[i],cf=COSF[e],sf=SINF[e],sl=SIDE[e];
        FLUX0[e]=0; FLUX1[e]=HALF_G*hi*hi*sl*cf; FLUX2[e]=HALF_G*hi*hi*sl*sf; FLUX3[e]=0;
    }
}

__global__ void update_cell(int CELL, float* H, float* U, float* V, float* Z,
                             const float* FLUX0, const float* FLUX1, const float* FLUX2,
                             const float* AREA, const float* ZBC, const float* FNC, float DT) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= CELL || AREA[i] <= 0) return;
    float f0=0,f1=0,f2=0;
    for(int j=0;j<4;j++){int e=4*i+j; f0+=FLUX0[e]; f1+=FLUX1[e]; f2+=FLUX2[e];}
    float inv_a = DT/AREA[i];
    float h_new = fmaxf(H[i]-f0*inv_a, HM1);
    float hu=H[i]*U[i]-f1*inv_a, hv=H[i]*V[i]-f2*inv_a;
    float w=sqrtf(U[i]*U[i]+V[i]*V[i]);
    float fric=1.0f+DT*FNC[i]*w/(h_new*h_new);
    H[i]=h_new; U[i]=hu/(h_new*fric); V[i]=hv/(h_new*fric); Z[i]=h_new+ZBC[i];
}

__global__ void hydro_persistent(int CELL, int NEDGE,
    float* H, float* U, float* V, float* Z,
    const int* NAC, const float* KLAS,
    const float* COSF, const float* SINF, const float* SIDE,
    float* FLUX0, float* FLUX1, float* FLUX2, float* FLUX3,
    const float* AREA, const float* ZBC, const float* FNC,
    float DT, int STEPS) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int step = 0; step < STEPS; step++) {
        if (tid < NEDGE) {
            int e=tid, i=e/4;
            if (KLAS[e]==0.0f) {
                int nc=NAC[e]-1;
                if(nc>=0&&nc<CELL){
                    float hi=H[i],ui=U[i],vi=V[i],hn=H[nc],un=U[nc],vn=V[nc];
                    float cf=COSF[e],sf=SINF[e],sl=SIDE[e];
                    float ul=ui*cf+vi*sf,vl=-ui*sf+vi*cf,ur=un*cf+vn*sf,vr=-un*sf+vn*cf;
                    float cl=sqrtf(G*hi),cr=sqrtf(G*hn);
                    float smax=fmaxf(fabsf(ul)+cl,fabsf(ur)+cr);
                    float f0=0.5f*sl*(hi*ul+hn*ur-smax*(hn-hi));
                    float f1=0.5f*sl*(hi*ul*ul+HALF_G*hi*hi+hn*ur*ur+HALF_G*hn*hn-smax*(hn*ur-hi*ul));
                    float f2=0.5f*sl*(hi*ul*vl+hn*ur*vr-smax*(hn*vr-hi*vl));
                    FLUX0[e]=f0;FLUX1[e]=f1*cf-f2*sf;FLUX2[e]=f1*sf+f2*cf;
                    FLUX3[e]=HALF_G*0.5f*sl*(hi*hi+hn*hn);
                } else {FLUX0[e]=FLUX1[e]=FLUX2[e]=FLUX3[e]=0;}
            } else {
                float hi=H[e/4],cf=COSF[e],sf=SINF[e],sl=SIDE[e];
                FLUX0[e]=0;FLUX1[e]=HALF_G*hi*hi*sl*cf;FLUX2[e]=HALF_G*hi*hi*sl*sf;FLUX3[e]=0;
            }
        }
        cooperative_groups::this_grid().sync();
        if (tid < CELL && AREA[tid] > 0) {
            int i=tid;
            float f0=0,f1=0,f2=0;
            for(int j=0;j<4;j++){int e=4*i+j;f0+=FLUX0[e];f1+=FLUX1[e];f2+=FLUX2[e];}
            float inv_a=DT/AREA[i];
            float h_new=fmaxf(H[i]-f0*inv_a,HM1);
            float hu=H[i]*U[i]-f1*inv_a,hv=H[i]*V[i]-f2*inv_a;
            float w=sqrtf(U[i]*U[i]+V[i]*V[i]);
            float fric=1.0f+DT*FNC[i]*w/(h_new*h_new);
            H[i]=h_new;U[i]=hu/(h_new*fric);V[i]=hv/(h_new*fric);Z[i]=h_new+ZBC[i];
        }
        cooperative_groups::this_grid().sync();
    }
}

#define CHECK(call) { auto e = call; if(e) { printf("CUDA error %d at line %d\n", e, __LINE__); exit(1); }}

int main() {
    int CELL=6675, NEDGE=4*CELL; float DT=4.0f;
    int SPD=900, NDAYS=10, TOTAL=SPD*NDAYS;

    float *H,*U,*V,*Z,*COSF,*SINF,*SIDE,*KLAS_f,*AREA,*ZBC,*FNC;
    float *FLUX0,*FLUX1,*FLUX2,*FLUX3; int *NAC;
    CHECK(cudaMalloc(&H,CELL*4));CHECK(cudaMalloc(&U,CELL*4));CHECK(cudaMalloc(&V,CELL*4));CHECK(cudaMalloc(&Z,CELL*4));
    CHECK(cudaMalloc(&AREA,CELL*4));CHECK(cudaMalloc(&ZBC,CELL*4));CHECK(cudaMalloc(&FNC,CELL*4));
    CHECK(cudaMalloc(&NAC,NEDGE*4));CHECK(cudaMalloc(&KLAS_f,NEDGE*4));
    CHECK(cudaMalloc(&COSF,NEDGE*4));CHECK(cudaMalloc(&SINF,NEDGE*4));CHECK(cudaMalloc(&SIDE,NEDGE*4));
    CHECK(cudaMalloc(&FLUX0,NEDGE*4));CHECK(cudaMalloc(&FLUX1,NEDGE*4));
    CHECK(cudaMalloc(&FLUX2,NEDGE*4));CHECK(cudaMalloc(&FLUX3,NEDGE*4));

    {
        std::vector<float> h(CELL,1.0f),a(CELL,100.0f),z(CELL,0.0f),fnc(CELL,0.01f);
        std::vector<int> nac(NEDGE); std::vector<float> klas(NEDGE,0.0f);
        std::vector<float> cf(NEDGE),sf(NEDGE),sd(NEDGE,10.0f);
        srand(42);
        for(int i=0;i<CELL;i++) for(int j=0;j<4;j++){
            int e=4*i+j; nac[e]=(rand()%CELL)+1;
            cf[e]=(j==0)?1.0f:(j==2)?-1.0f:0.0f;
            sf[e]=(j==1)?1.0f:(j==3)?-1.0f:0.0f;
            if(rand()%5==0) klas[e]=4.0f;
        }
        cudaMemcpy(H,h.data(),CELL*4,cudaMemcpyHostToDevice);
        cudaMemcpy(AREA,a.data(),CELL*4,cudaMemcpyHostToDevice);
        cudaMemcpy(ZBC,z.data(),CELL*4,cudaMemcpyHostToDevice);
        cudaMemcpy(FNC,fnc.data(),CELL*4,cudaMemcpyHostToDevice);
        cudaMemcpy(NAC,nac.data(),NEDGE*4,cudaMemcpyHostToDevice);
        cudaMemcpy(KLAS_f,klas.data(),NEDGE*4,cudaMemcpyHostToDevice);
        cudaMemcpy(COSF,cf.data(),NEDGE*4,cudaMemcpyHostToDevice);
        cudaMemcpy(SINF,sf.data(),NEDGE*4,cudaMemcpyHostToDevice);
        cudaMemcpy(SIDE,sd.data(),NEDGE*4,cudaMemcpyHostToDevice);
        cudaMemset(U,0,CELL*4);cudaMemset(V,0,CELL*4);cudaMemset(Z,0,CELL*4);
        cudaMemset(FLUX0,0,NEDGE*4);cudaMemset(FLUX1,0,NEDGE*4);
        cudaMemset(FLUX2,0,NEDGE*4);cudaMemset(FLUX3,0,NEDGE*4);
    }

    cudaEvent_t t0,t1;CHECK(cudaEventCreate(&t0));CHECK(cudaEventCreate(&t1));
    int fb=(NEDGE+255)/256,ub=(CELL+255)/256;
    cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
    printf("GPU: %s (SMs=%d)\nHydro-cal: %d cells, %d edges, %d steps\n\n",
           prop.name,prop.multiProcessorCount,CELL,NEDGE,TOTAL);

    for(int i=0;i<50;i++){
        calculate_flux<<<fb,256>>>(NEDGE,H,U,V,NAC,KLAS_f,COSF,SINF,SIDE,FLUX0,FLUX1,FLUX2,FLUX3,CELL);
        update_cell<<<ub,256>>>(CELL,H,U,V,Z,FLUX0,FLUX1,FLUX2,AREA,ZBC,FNC,DT);
    }
    cudaDeviceSynchronize();

    float ms; int REPS=3;

    // [1] Sync
    cudaEventRecord(t0);
    for(int r=0;r<REPS;r++)for(int s=0;s<TOTAL;s++){
        calculate_flux<<<fb,256>>>(NEDGE,H,U,V,NAC,KLAS_f,COSF,SINF,SIDE,FLUX0,FLUX1,FLUX2,FLUX3,CELL);
        update_cell<<<ub,256>>>(CELL,H,U,V,Z,FLUX0,FLUX1,FLUX2,AREA,ZBC,FNC,DT);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1);
    float sync_us=ms*1000/(TOTAL*REPS);
    printf("[1] Sync loop (Taichi-like):  %7.2f us/step  (%6.1f ms/day)\n",sync_us,ms/(NDAYS*REPS));

    // [2] Async
    cudaEventRecord(t0);
    for(int r=0;r<REPS;r++)for(int s=0;s<TOTAL;s++){
        calculate_flux<<<fb,256>>>(NEDGE,H,U,V,NAC,KLAS_f,COSF,SINF,SIDE,FLUX0,FLUX1,FLUX2,FLUX3,CELL);
        update_cell<<<ub,256>>>(CELL,H,U,V,Z,FLUX0,FLUX1,FLUX2,AREA,ZBC,FNC,DT);
    }
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1);
    float async_us=ms*1000/(TOTAL*REPS);
    printf("[2] Async loop (Kokkos):      %7.2f us/step  (%6.1f ms/day)\n",async_us,ms/(NDAYS*REPS));

    // [3] Graph
    float graph_us;
    {cudaGraph_t g;cudaGraphExec_t ge;
     cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
     CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
     for(int s=0;s<SPD;s++){
         calculate_flux<<<fb,256,0,stream>>>(NEDGE,H,U,V,NAC,KLAS_f,COSF,SINF,SIDE,FLUX0,FLUX1,FLUX2,FLUX3,CELL);
         update_cell<<<ub,256,0,stream>>>(CELL,H,U,V,Z,FLUX0,FLUX1,FLUX2,AREA,ZBC,FNC,DT);
     }
     CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&ge,g,NULL,NULL,0));
     for(int i=0;i<3;i++){cudaGraphLaunch(ge,stream);cudaStreamSynchronize(stream);}
     cudaEventRecord(t0,stream);
     for(int d=0;d<NDAYS*REPS;d++)cudaGraphLaunch(ge,stream);
     cudaEventRecord(t1,stream);cudaStreamSynchronize(stream);
     cudaEventElapsedTime(&ms,t0,t1);graph_us=ms*1000/(TOTAL*REPS);
     printf("[3] CUDA Graph (900-step):    %7.2f us/step  (%6.1f ms/day)\n",graph_us,ms/(NDAYS*REPS));
     cudaGraphExecDestroy(ge);cudaGraphDestroy(g);cudaStreamDestroy(stream);}

    // [4] Persistent
    float persist_us=-1;
    {int numBSm=0;
     cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm,hydro_persistent,256,0);
     int maxB=numBSm*prop.multiProcessorCount,needB=(NEDGE+255)/256;
     printf("\n  Persistent: need %d blocks, max %d (%d blk/SM x %d SMs)\n",needB,maxB,numBSm,prop.multiProcessorCount);
     if(needB<=maxB){
         dim3 pg(needB),pb(256);
         {int ws=100;void*a[]={(void*)&CELL,(void*)&NEDGE,(void*)&H,(void*)&U,(void*)&V,(void*)&Z,
             (void*)&NAC,(void*)&KLAS_f,(void*)&COSF,(void*)&SINF,(void*)&SIDE,
             (void*)&FLUX0,(void*)&FLUX1,(void*)&FLUX2,(void*)&FLUX3,
             (void*)&AREA,(void*)&ZBC,(void*)&FNC,(void*)&DT,&ws};
          cudaLaunchCooperativeKernel((void*)hydro_persistent,pg,pb,a);cudaDeviceSynchronize();}
         cudaEventRecord(t0);
         for(int r=0;r<REPS;r++)for(int d=0;d<NDAYS;d++){
             void*a[]={(void*)&CELL,(void*)&NEDGE,(void*)&H,(void*)&U,(void*)&V,(void*)&Z,
                 (void*)&NAC,(void*)&KLAS_f,(void*)&COSF,(void*)&SINF,(void*)&SIDE,
                 (void*)&FLUX0,(void*)&FLUX1,(void*)&FLUX2,(void*)&FLUX3,
                 (void*)&AREA,(void*)&ZBC,(void*)&FNC,(void*)&DT,&SPD};
             cudaLaunchCooperativeKernel((void*)hydro_persistent,pg,pb,a);
         }
         cudaEventRecord(t1);cudaDeviceSynchronize();
         cudaEventElapsedTime(&ms,t0,t1);persist_us=ms*1000/(TOTAL*REPS);
         printf("[4] Persistent kernel:        %7.2f us/step  (%6.1f ms/day)\n",persist_us,ms/(NDAYS*REPS));
     } else printf("[4] Persistent: SKIP\n");}

    printf("\n=== Speedup vs Sync ===\n");
    printf("  Async:      %.2fx\n",sync_us/async_us);
    printf("  Graph:      %.2fx\n",sync_us/graph_us);
    if(persist_us>0)printf("  Persistent: %.2fx\n",sync_us/persist_us);
    printf("\n=== Overhead breakdown ===\n");
    printf("  GPU compute:  ~%.1f us (Graph lower bound)\n",graph_us);
    printf("  Launch OH:     %.1f us\n",async_us-graph_us);
    printf("  Sync OH:       %.1f us\n",sync_us-async_us);
    printf("  Total OH:      %.1f us (%.0f%% of Sync)\n",sync_us-graph_us,(sync_us-graph_us)/sync_us*100);

    cudaFree(H);cudaFree(U);cudaFree(V);cudaFree(Z);
    cudaFree(NAC);cudaFree(KLAS_f);cudaFree(COSF);cudaFree(SINF);cudaFree(SIDE);
    cudaFree(FLUX0);cudaFree(FLUX1);cudaFree(FLUX2);cudaFree(FLUX3);
    cudaFree(AREA);cudaFree(ZBC);cudaFree(FNC);
    cudaEventDestroy(t0);cudaEventDestroy(t1);
}
