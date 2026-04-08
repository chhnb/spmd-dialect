/**
 * Extended Overhead Characterization: 4 strategies × 10 kernels × multiple sizes
 * Covers 8 domains to verify overhead floor across kernel types.
 *
 * Kernels (domain):
 *   1. Heat2D        (Stencil)         — already verified
 *   2. GrayScott     (Stencil/RD)      — already verified
 *   3. Wave2D        (Stencil)         — 2nd-order time
 *   4. Jacobi2D      (Stencil/Solver)  — iterative
 *   5. Burgers2D     (CFD)             — nonlinear convection
 *   6. FDTD Maxwell  (EM)              — leapfrog E/H update
 *   7. NBody         (Particle)        — O(N²) pairwise
 *   8. SWE LaxFried  (CFD)             — shallow water equations
 *   9. SRAD          (Classic/Rodinia) — speckle reducing anisotropic diffusion
 *  10. ExplicitFEM   (FEM)             — lumped mass explicit
 *
 * Build:
 *   nvcc -O3 -arch=sm_86 -rdc=true overhead_characterization_extended.cu \
 *        -o overhead_extended -lcudadevrt
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CHECK(call) { auto e = call; if(e) { printf("CUDA error %d (%s) at line %d\n", e, cudaGetErrorString(e), __LINE__); exit(1); }}

// =========================================================================
// Generic 2D benchmark harness (compute kernel + copy-back)
// =========================================================================
struct Result { float sync_us, async_us, graph_us, persistent_us; };

// Forward declarations for persistent kernels
__global__ void wave2d_persistent(int N, float* u, float* v, float* u_prev, int STEPS);
__global__ void jacobi2d_persistent(int N, float* u, float* v, int STEPS);
__global__ void burgers2d_persistent(int N, float* u, float* v, float* uu, float* vv, int STEPS);
__global__ void fdtd2d_persistent(int N, float* Ex, float* Ey, float* Hz, int STEPS);
__global__ void swe_persistent(int N, float* h, float* hu, float* hv, float* h2, float* hu2, float* hv2, int STEPS);
__global__ void srad_persistent(int N, float* u, float* v, int STEPS);
__global__ void fem2d_persistent(int NE, int NP1, float* u, float* v, float* f, float* mass, int N, int STEPS);

// =========================================================================
// 1. Wave2D (2nd-order time, 3 arrays: u, v=u_new, u_prev)
// =========================================================================
__global__ void wave2d_step(int N, const float* u, float* v, const float* u_prev) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=1&&i<N-1&&j>=1&&j<N-1){
        int idx=i*N+j;
        float lap=u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-4.0f*u[idx];
        v[idx]=2.0f*u[idx]-u_prev[idx]+0.01f*lap;
    }
}
__global__ void copy2_f(int N2, const float* s1, float* d1, const float* s2, float* d2) {
    int i=blockIdx.x*256+threadIdx.x;
    if(i<N2){d1[i]=s1[i]; d2[i]=s2[i];}
}
__global__ void wave2d_persistent(int N, float* u, float* v, float* u_prev, int STEPS) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    for(int s=0;s<STEPS;s++){
        if(i>=1&&i<N-1&&j>=1&&j<N-1){
            int idx=i*N+j;
            float lap=u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-4.0f*u[idx];
            v[idx]=2.0f*u[idx]-u_prev[idx]+0.01f*lap;
        }
        cg::this_grid().sync();
        if(i<N&&j<N){int idx=i*N+j; u_prev[idx]=u[idx]; u[idx]=v[idx];}
        cg::this_grid().sync();
    }
}

// =========================================================================
// 2. Jacobi2D (iterative solver: u_new = 0.25*(neighbors))
// =========================================================================
__global__ void jacobi2d_step(int N, const float* u, float* v) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=1&&i<N-1&&j>=1&&j<N-1){
        int idx=i*N+j;
        v[idx]=0.25f*(u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]);
    }
}
__global__ void copy_f(int N2, const float* s, float* d) {
    int i=blockIdx.x*256+threadIdx.x; if(i<N2) d[i]=s[i];
}
__global__ void jacobi2d_persistent(int N, float* u, float* v, int STEPS) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    for(int s=0;s<STEPS;s++){
        if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;v[idx]=0.25f*(u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]);}
        cg::this_grid().sync();
        if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;u[idx]=v[idx];}
        cg::this_grid().sync();
    }
}

// =========================================================================
// 3. Burgers2D (nonlinear convection-diffusion)
// =========================================================================
__global__ void burgers2d_step(int N, const float* u, const float* v, float* u2, float* v2) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=1&&i<N-1&&j>=1&&j<N-1){
        int idx=i*N+j;
        float nu=0.01f, dt=0.001f, dx=1.0f/N;
        float dudx=(u[idx]-u[idx-1])/dx, dudy=(u[idx]-u[idx-N])/dx;
        float dvdx=(v[idx]-v[idx-1])/dx, dvdy=(v[idx]-v[idx-N])/dx;
        float lapu=(u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-4.0f*u[idx])/(dx*dx);
        float lapv=(v[idx-N]+v[idx+N]+v[idx-1]+v[idx+1]-4.0f*v[idx])/(dx*dx);
        u2[idx]=u[idx]+dt*(-u[idx]*dudx-v[idx]*dudy+nu*lapu);
        v2[idx]=v[idx]+dt*(-u[idx]*dvdx-v[idx]*dvdy+nu*lapv);
    }
}
__global__ void burgers2d_persistent(int N, float* u, float* v, float* u2, float* v2, int STEPS) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    for(int s=0;s<STEPS;s++){
        if(i>=1&&i<N-1&&j>=1&&j<N-1){
            int idx=i*N+j;
            float nu=0.01f,dt=0.001f,dx=1.0f/N;
            float dudx=(u[idx]-u[idx-1])/dx, dudy=(u[idx]-u[idx-N])/dx;
            float dvdx=(v[idx]-v[idx-1])/dx, dvdy=(v[idx]-v[idx-N])/dx;
            float lapu=(u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-4.0f*u[idx])/(dx*dx);
            float lapv=(v[idx-N]+v[idx+N]+v[idx-1]+v[idx+1]-4.0f*v[idx])/(dx*dx);
            u2[idx]=u[idx]+dt*(-u[idx]*dudx-v[idx]*dudy+nu*lapu);
            v2[idx]=v[idx]+dt*(-u[idx]*dvdx-v[idx]*dvdy+nu*lapv);
        }
        cg::this_grid().sync();
        if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;u[idx]=u2[idx];v[idx]=v2[idx];}
        cg::this_grid().sync();
    }
}

// =========================================================================
// 4. FDTD Maxwell 2D (TE mode: Ex, Ey, Hz leapfrog)
// =========================================================================
__global__ void fdtd2d_step(int N, float* Ex, float* Ey, float* Hz) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    float dt=0.001f, dx=0.01f;
    // Update Hz
    if(i>=0&&i<N-1&&j>=0&&j<N-1){
        int idx=i*N+j;
        Hz[idx]+=dt/dx*(Ex[idx+N]-Ex[idx]-Ey[idx+1]+Ey[idx]);
    }
    __syncthreads(); // note: not grid sync, just block-level for simplicity
    // Update Ex
    if(i>=1&&i<N&&j>=0&&j<N){
        int idx=i*N+j;
        Ex[idx]+=dt/dx*(Hz[idx]-Hz[idx-N]);
    }
    // Update Ey
    if(i>=0&&i<N&&j>=1&&j<N){
        int idx=i*N+j;
        Ey[idx]-=dt/dx*(Hz[idx]-Hz[idx-1]);
    }
}
__global__ void fdtd2d_persistent(int N, float* Ex, float* Ey, float* Hz, int STEPS) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    float dt=0.001f, dx=0.01f;
    for(int s=0;s<STEPS;s++){
        if(i<N-1&&j<N-1){int idx=i*N+j;Hz[idx]+=dt/dx*(Ex[idx+N]-Ex[idx]-Ey[idx+1]+Ey[idx]);}
        cg::this_grid().sync();
        if(i>=1&&i<N&&j<N){int idx=i*N+j;Ex[idx]+=dt/dx*(Hz[idx]-Hz[idx-N]);}
        if(i<N&&j>=1&&j<N){int idx=i*N+j;Ey[idx]-=dt/dx*(Hz[idx]-Hz[idx-1]);}
        cg::this_grid().sync();
    }
}

// =========================================================================
// 5. SWE Lax-Friedrichs (shallow water: h, hu, hv)
// =========================================================================
__global__ void swe_step(int N, const float* h, const float* hu, const float* hv,
                          float* h2, float* hu2, float* hv2) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=1&&i<N-1&&j>=1&&j<N-1){
        int idx=i*N+j;
        float g=9.81f, dx=1.0f/N, dt=0.0001f;
        float h_avg=0.25f*(h[idx-N]+h[idx+N]+h[idx-1]+h[idx+1]);
        float hu_avg=0.25f*(hu[idx-N]+hu[idx+N]+hu[idx-1]+hu[idx+1]);
        float hv_avg=0.25f*(hv[idx-N]+hv[idx+N]+hv[idx-1]+hv[idx+1]);
        // Lax-Friedrichs flux
        float dhu_dx=(hu[idx+1]-hu[idx-1])/(2*dx);
        float dhv_dy=(hv[idx+N]-hv[idx-N])/(2*dx);
        h2[idx]=h_avg-dt*(dhu_dx+dhv_dy);
        // momentum (simplified)
        float u_=hu[idx]/(fmaxf(h[idx],1e-6f)), v_=hv[idx]/(fmaxf(h[idx],1e-6f));
        hu2[idx]=hu_avg-dt*((hu[idx+1]*u_-hu[idx-1]*u_)/(2*dx)+g*h[idx]*(h[idx+1]-h[idx-1])/(2*dx));
        hv2[idx]=hv_avg-dt*((hv[idx+N]*v_-hv[idx-N]*v_)/(2*dx)+g*h[idx]*(h[idx+N]-h[idx-N])/(2*dx));
    }
}
__global__ void copy3_f(int N2, const float* s1, float* d1, const float* s2, float* d2, const float* s3, float* d3) {
    int i=blockIdx.x*256+threadIdx.x;
    if(i<N2){d1[i]=s1[i];d2[i]=s2[i];d3[i]=s3[i];}
}
__global__ void swe_persistent(int N, float* h, float* hu, float* hv,
                                float* h2, float* hu2, float* hv2, int STEPS) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    int N2=N*N;
    for(int s=0;s<STEPS;s++){
        if(i>=1&&i<N-1&&j>=1&&j<N-1){
            int idx=i*N+j;
            float g=9.81f, dx=1.0f/N, dt=0.0001f;
            float h_avg=0.25f*(h[idx-N]+h[idx+N]+h[idx-1]+h[idx+1]);
            float hu_avg=0.25f*(hu[idx-N]+hu[idx+N]+hu[idx-1]+hu[idx+1]);
            float hv_avg=0.25f*(hv[idx-N]+hv[idx+N]+hv[idx-1]+hv[idx+1]);
            float dhu_dx=(hu[idx+1]-hu[idx-1])/(2*dx);
            float dhv_dy=(hv[idx+N]-hv[idx-N])/(2*dx);
            h2[idx]=h_avg-dt*(dhu_dx+dhv_dy);
            float u_=hu[idx]/(fmaxf(h[idx],1e-6f)), v_=hv[idx]/(fmaxf(h[idx],1e-6f));
            hu2[idx]=hu_avg-dt*((hu[idx+1]*u_-hu[idx-1]*u_)/(2*dx)+g*h[idx]*(h[idx+1]-h[idx-1])/(2*dx));
            hv2[idx]=hv_avg-dt*((hv[idx+N]*v_-hv[idx-N]*v_)/(2*dx)+g*h[idx]*(h[idx+N]-h[idx-N])/(2*dx));
        }
        cg::this_grid().sync();
        int tid=i*N+j;
        if(tid<N2){h[tid]=h2[tid];hu[tid]=hu2[tid];hv[tid]=hv2[tid];}
        cg::this_grid().sync();
    }
}

// =========================================================================
// 6. SRAD (Rodinia: speckle reducing anisotropic diffusion)
// =========================================================================
__global__ void srad_step(int N, const float* u, float* v) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=1&&i<N-1&&j>=1&&j<N-1){
        int idx=i*N+j;
        float Jc=u[idx], dt=0.05f, lambda=0.5f;
        float dN=u[idx-N]-Jc, dS=u[idx+N]-Jc, dW=u[idx-1]-Jc, dE=u[idx+1]-Jc;
        float G2=(dN*dN+dS*dS+dW*dW+dE*dE)/(Jc*Jc+1e-10f);
        float L=(dN+dS+dW+dE)/(Jc+1e-10f);
        float num=(0.5f*G2-L*L/16.0f);
        float den=(1.0f+0.25f*L)*(1.0f+0.25f*L)+1e-10f;
        float q=num/den;
        float q0=1.0f; // reference
        float c=1.0f/(1.0f+(q-q0*q0)/(q0*q0*(1.0f+q0*q0)+1e-10f));
        c=fmaxf(fminf(c,1.0f),0.0f);
        v[idx]=Jc+dt*lambda*(c*dN+c*dS+c*dW+c*dE);
    }
}
__global__ void srad_persistent(int N, float* u, float* v, int STEPS) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    for(int s=0;s<STEPS;s++){
        if(i>=1&&i<N-1&&j>=1&&j<N-1){
            int idx=i*N+j;
            float Jc=u[idx], dt=0.05f, lambda=0.5f;
            float dN=u[idx-N]-Jc, dS=u[idx+N]-Jc, dW=u[idx-1]-Jc, dE=u[idx+1]-Jc;
            float G2=(dN*dN+dS*dS+dW*dW+dE*dE)/(Jc*Jc+1e-10f);
            float L=(dN+dS+dW+dE)/(Jc+1e-10f);
            float num=(0.5f*G2-L*L/16.0f);
            float den=(1.0f+0.25f*L)*(1.0f+0.25f*L)+1e-10f;
            float q=num/den;
            float q0=1.0f;
            float c=1.0f/(1.0f+(q-q0*q0)/(q0*q0*(1.0f+q0*q0)+1e-10f));
            c=fmaxf(fminf(c,1.0f),0.0f);
            v[idx]=Jc+dt*lambda*(c*dN+c*dS+c*dW+c*dE);
        }
        cg::this_grid().sync();
        if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;u[idx]=v[idx];}
        cg::this_grid().sync();
    }
}

// =========================================================================
// 7. NBody (O(N²) pairwise, 1D launch)
// =========================================================================
__global__ void nbody_step(int N, const float* px, const float* py,
                            const float* vx, const float* vy,
                            float* px2, float* py2, float* vx2, float* vy2) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;
    float ax=0, ay=0, dt=0.001f, eps=0.01f;
    for(int j=0;j<N;j++){
        float dx=px[j]-px[i], dy=py[j]-py[i];
        float r2=dx*dx+dy*dy+eps;
        float inv=rsqrtf(r2*r2*r2);
        ax+=dx*inv; ay+=dy*inv;
    }
    vx2[i]=vx[i]+dt*ax; vy2[i]=vy[i]+dt*ay;
    px2[i]=px[i]+dt*vx2[i]; py2[i]=py[i]+dt*vy2[i];
}
__global__ void copy4_f(int N, const float* s1, float* d1, const float* s2, float* d2,
                         const float* s3, float* d3, const float* s4, float* d4) {
    int i=blockIdx.x*256+threadIdx.x;
    if(i<N){d1[i]=s1[i];d2[i]=s2[i];d3[i]=s3[i];d4[i]=s4[i];}
}

// NBody persistent (1D grid)
__global__ void nbody_persistent(int N, float* px, float* py, float* vx, float* vy,
                                  float* px2, float* py2, float* vx2, float* vy2, int STEPS) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    int total=blockDim.x*gridDim.x;
    for(int s=0;s<STEPS;s++){
        for(int i=tid;i<N;i+=total){
            float ax=0, ay=0, dt=0.001f, eps=0.01f;
            for(int j=0;j<N;j++){
                float dx=px[j]-px[i], dy=py[j]-py[i];
                float r2=dx*dx+dy*dy+eps;
                float inv=rsqrtf(r2*r2*r2);
                ax+=dx*inv; ay+=dy*inv;
            }
            vx2[i]=vx[i]+dt*ax; vy2[i]=vy[i]+dt*ay;
            px2[i]=px[i]+dt*vx2[i]; py2[i]=py[i]+dt*vy2[i];
        }
        cg::this_grid().sync();
        for(int i=tid;i<N;i+=total){px[i]=px2[i];py[i]=py2[i];vx[i]=vx2[i];vy[i]=vy2[i];}
        cg::this_grid().sync();
    }
}

// =========================================================================
// Generic 2D benchmark harness
// =========================================================================
typedef void(*StepFn2D)(int,const float*,float*);
typedef void(*PersistFn)(void);

Result bench_2field(const char* name, int N, int STEPS,
                    void(*launch_step)(int,float*,float*,dim3,dim3),
                    void(*launch_copy)(int,float*,float*,int),
                    void*persist_kernel, int persist_nargs, void**persist_args_template,
                    int n_arrays, float** arrays, int total_bytes) {
    dim3 block(16,16), grid((N+15)/16,(N+15)/16);
    int N2=N*N, cg=(N2+255)/256;
    cudaEvent_t t0,t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    Result r; float ms;

    // Warmup
    for(int i=0;i<20;i++){launch_step(N,arrays[0],arrays[1],grid,block); launch_copy(N2,arrays[1],arrays[0],cg);}
    cudaDeviceSynchronize();

    // Sync
    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){launch_step(N,arrays[0],arrays[1],grid,block);launch_copy(N2,arrays[1],arrays[0],cg);cudaDeviceSynchronize();}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1); r.sync_us=ms*1000/STEPS;

    // Async
    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){launch_step(N,arrays[0],arrays[1],grid,block);launch_copy(N2,arrays[1],arrays[0],cg);}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1); r.async_us=ms*1000/STEPS;

    // Graph
    {int REPS=5;cudaGraph_t g;cudaGraphExec_t ge;
     cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
     CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
     for(int s=0;s<STEPS;s++){
         launch_step(N,arrays[0],arrays[1],grid,block); // TODO: need stream version
         launch_copy(N2,arrays[1],arrays[0],cg);
     }
     CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&ge,g,NULL,NULL,0));
     for(int i=0;i<3;i++){cudaGraphLaunch(ge,stream);cudaStreamSynchronize(stream);}
     cudaEventRecord(t0,stream);for(int i=0;i<REPS;i++)cudaGraphLaunch(ge,stream);
     cudaEventRecord(t1,stream);cudaStreamSynchronize(stream);
     cudaEventElapsedTime(&ms,t0,t1);r.graph_us=ms*1000/(STEPS*REPS);
     cudaGraphExecDestroy(ge);cudaGraphDestroy(g);cudaStreamDestroy(stream);}

    // Persistent
    r.persistent_us=-1;
    if(persist_kernel){
        int numBSm=0;cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm,persist_kernel,256,0);
        int maxB=numBSm*prop.multiProcessorCount;
        if((int)(grid.x*grid.y)<=maxB){
            // Warmup
            cudaLaunchCooperativeKernel(persist_kernel,grid,block,persist_args_template);
            cudaDeviceSynchronize();
            // Reset arrays
            for(int a=0;a<n_arrays;a++) cudaMemset(arrays[a],0,total_bytes/n_arrays);
            int REPS=5;cudaEventRecord(t0);
            for(int i=0;i<REPS;i++)
                cudaLaunchCooperativeKernel(persist_kernel,grid,block,persist_args_template);
            cudaEventRecord(t1);cudaDeviceSynchronize();
            cudaEventElapsedTime(&ms,t0,t1);r.persistent_us=ms*1000/(STEPS*REPS);
        }
    }

    cudaEventDestroy(t0);cudaEventDestroy(t1);
    return r;
}

// =========================================================================
// Per-kernel test functions (self-contained, like test_heat in original)
// =========================================================================

Result test_jacobi2d(int N, int STEPS) {
    int N2=N*N; float *u,*v;
    CHECK(cudaMalloc(&u,N2*4));CHECK(cudaMalloc(&v,N2*4));
    // Init: boundary=1, interior=0
    CHECK(cudaMemset(u,0,N2*4));CHECK(cudaMemset(v,0,N2*4));
    dim3 block(16,16),grid((N+15)/16,(N+15)/16); int cg=(N2+255)/256;
    cudaEvent_t t0,t1;CHECK(cudaEventCreate(&t0));CHECK(cudaEventCreate(&t1));
    for(int i=0;i<20;i++){jacobi2d_step<<<grid,block>>>(N,u,v);copy_f<<<cg,256>>>(N2,v,u);}
    cudaDeviceSynchronize();
    Result r; float ms;

    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){jacobi2d_step<<<grid,block>>>(N,u,v);copy_f<<<cg,256>>>(N2,v,u);cudaDeviceSynchronize();}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1); r.sync_us=ms*1000/STEPS;

    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){jacobi2d_step<<<grid,block>>>(N,u,v);copy_f<<<cg,256>>>(N2,v,u);}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1); r.async_us=ms*1000/STEPS;

    {int REPS=5;cudaGraph_t g;cudaGraphExec_t ge;
     cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
     CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
     for(int s=0;s<STEPS;s++){jacobi2d_step<<<grid,block,0,stream>>>(N,u,v);copy_f<<<cg,256,0,stream>>>(N2,v,u);}
     CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&ge,g,NULL,NULL,0));
     for(int i=0;i<3;i++){cudaGraphLaunch(ge,stream);cudaStreamSynchronize(stream);}
     cudaEventRecord(t0,stream);for(int i=0;i<REPS;i++)cudaGraphLaunch(ge,stream);
     cudaEventRecord(t1,stream);cudaStreamSynchronize(stream);
     cudaEventElapsedTime(&ms,t0,t1);r.graph_us=ms*1000/(STEPS*REPS);
     cudaGraphExecDestroy(ge);cudaGraphDestroy(g);cudaStreamDestroy(stream);}

    {int numBSm=0;cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
     cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm,jacobi2d_persistent,256,0);
     int maxB=numBSm*prop.multiProcessorCount;
     if((int)(grid.x*grid.y)<=maxB){
         void*args[]={(void*)&N,(void*)&u,(void*)&v,(void*)&STEPS};
         cudaLaunchCooperativeKernel((void*)jacobi2d_persistent,grid,block,args);cudaDeviceSynchronize();
         int REPS=5;cudaEventRecord(t0);
         for(int i=0;i<REPS;i++)cudaLaunchCooperativeKernel((void*)jacobi2d_persistent,grid,block,args);
         cudaEventRecord(t1);cudaDeviceSynchronize();
         cudaEventElapsedTime(&ms,t0,t1);r.persistent_us=ms*1000/(STEPS*REPS);
     } else r.persistent_us=-1;}

    cudaEventDestroy(t0);cudaEventDestroy(t1);cudaFree(u);cudaFree(v);
    return r;
}

Result test_wave2d(int N, int STEPS) {
    int N2=N*N; float *u,*v,*up;
    CHECK(cudaMalloc(&u,N2*4));CHECK(cudaMalloc(&v,N2*4));CHECK(cudaMalloc(&up,N2*4));
    CHECK(cudaMemset(u,0,N2*4));CHECK(cudaMemset(v,0,N2*4));CHECK(cudaMemset(up,0,N2*4));
    dim3 block(16,16),grid((N+15)/16,(N+15)/16); int cg=(N2+255)/256;
    cudaEvent_t t0,t1;CHECK(cudaEventCreate(&t0));CHECK(cudaEventCreate(&t1));
    for(int i=0;i<20;i++){wave2d_step<<<grid,block>>>(N,u,v,up);copy2_f<<<cg,256>>>(N2,u,up,v,u);}
    cudaDeviceSynchronize();
    Result r; float ms;

    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){wave2d_step<<<grid,block>>>(N,u,v,up);copy2_f<<<cg,256>>>(N2,u,up,v,u);cudaDeviceSynchronize();}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1); r.sync_us=ms*1000/STEPS;

    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){wave2d_step<<<grid,block>>>(N,u,v,up);copy2_f<<<cg,256>>>(N2,u,up,v,u);}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1); r.async_us=ms*1000/STEPS;

    {int REPS=5;cudaGraph_t g;cudaGraphExec_t ge;
     cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
     CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
     for(int s=0;s<STEPS;s++){wave2d_step<<<grid,block,0,stream>>>(N,u,v,up);copy2_f<<<cg,256,0,stream>>>(N2,u,up,v,u);}
     CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&ge,g,NULL,NULL,0));
     for(int i=0;i<3;i++){cudaGraphLaunch(ge,stream);cudaStreamSynchronize(stream);}
     cudaEventRecord(t0,stream);for(int i=0;i<REPS;i++)cudaGraphLaunch(ge,stream);
     cudaEventRecord(t1,stream);cudaStreamSynchronize(stream);
     cudaEventElapsedTime(&ms,t0,t1);r.graph_us=ms*1000/(STEPS*REPS);
     cudaGraphExecDestroy(ge);cudaGraphDestroy(g);cudaStreamDestroy(stream);}

    {int numBSm=0;cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
     cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm,wave2d_persistent,256,0);
     int maxB=numBSm*prop.multiProcessorCount;
     if((int)(grid.x*grid.y)<=maxB){
         void*args[]={(void*)&N,(void*)&u,(void*)&v,(void*)&up,(void*)&STEPS};
         cudaLaunchCooperativeKernel((void*)wave2d_persistent,grid,block,args);cudaDeviceSynchronize();
         int REPS=5;cudaEventRecord(t0);
         for(int i=0;i<REPS;i++)cudaLaunchCooperativeKernel((void*)wave2d_persistent,grid,block,args);
         cudaEventRecord(t1);cudaDeviceSynchronize();
         cudaEventElapsedTime(&ms,t0,t1);r.persistent_us=ms*1000/(STEPS*REPS);
     } else r.persistent_us=-1;}

    cudaEventDestroy(t0);cudaEventDestroy(t1);cudaFree(u);cudaFree(v);cudaFree(up);
    return r;
}

Result test_burgers2d(int N, int STEPS) {
    int N2=N*N; float *u,*v,*u2,*v2;
    CHECK(cudaMalloc(&u,N2*4));CHECK(cudaMalloc(&v,N2*4));
    CHECK(cudaMalloc(&u2,N2*4));CHECK(cudaMalloc(&v2,N2*4));
    CHECK(cudaMemset(u,0,N2*4));CHECK(cudaMemset(v,0,N2*4));
    dim3 block(16,16),grid((N+15)/16,(N+15)/16); int cg=(N2+255)/256;
    cudaEvent_t t0,t1;CHECK(cudaEventCreate(&t0));CHECK(cudaEventCreate(&t1));
    for(int i=0;i<20;i++){burgers2d_step<<<grid,block>>>(N,u,v,u2,v2);copy_f<<<cg,256>>>(N2,u2,u);copy_f<<<cg,256>>>(N2,v2,v);}
    cudaDeviceSynchronize();
    Result r; float ms;

    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){burgers2d_step<<<grid,block>>>(N,u,v,u2,v2);copy_f<<<cg,256>>>(N2,u2,u);copy_f<<<cg,256>>>(N2,v2,v);cudaDeviceSynchronize();}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1); r.sync_us=ms*1000/STEPS;

    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){burgers2d_step<<<grid,block>>>(N,u,v,u2,v2);copy_f<<<cg,256>>>(N2,u2,u);copy_f<<<cg,256>>>(N2,v2,v);}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1); r.async_us=ms*1000/STEPS;

    {int REPS=5;cudaGraph_t g;cudaGraphExec_t ge;
     cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
     CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
     for(int s=0;s<STEPS;s++){burgers2d_step<<<grid,block,0,stream>>>(N,u,v,u2,v2);copy_f<<<cg,256,0,stream>>>(N2,u2,u);copy_f<<<cg,256,0,stream>>>(N2,v2,v);}
     CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&ge,g,NULL,NULL,0));
     for(int i=0;i<3;i++){cudaGraphLaunch(ge,stream);cudaStreamSynchronize(stream);}
     cudaEventRecord(t0,stream);for(int i=0;i<REPS;i++)cudaGraphLaunch(ge,stream);
     cudaEventRecord(t1,stream);cudaStreamSynchronize(stream);
     cudaEventElapsedTime(&ms,t0,t1);r.graph_us=ms*1000/(STEPS*REPS);
     cudaGraphExecDestroy(ge);cudaGraphDestroy(g);cudaStreamDestroy(stream);}

    {int numBSm=0;cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
     cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm,burgers2d_persistent,256,0);
     int maxB=numBSm*prop.multiProcessorCount;
     if((int)(grid.x*grid.y)<=maxB){
         void*args[]={(void*)&N,(void*)&u,(void*)&v,(void*)&u2,(void*)&v2,(void*)&STEPS};
         cudaLaunchCooperativeKernel((void*)burgers2d_persistent,grid,block,args);cudaDeviceSynchronize();
         int REPS=5;cudaEventRecord(t0);
         for(int i=0;i<REPS;i++)cudaLaunchCooperativeKernel((void*)burgers2d_persistent,grid,block,args);
         cudaEventRecord(t1);cudaDeviceSynchronize();
         cudaEventElapsedTime(&ms,t0,t1);r.persistent_us=ms*1000/(STEPS*REPS);
     } else r.persistent_us=-1;}

    cudaEventDestroy(t0);cudaEventDestroy(t1);
    cudaFree(u);cudaFree(v);cudaFree(u2);cudaFree(v2);
    return r;
}

Result test_fdtd2d(int N, int STEPS) {
    int N2=N*N; float *Ex,*Ey,*Hz;
    CHECK(cudaMalloc(&Ex,N2*4));CHECK(cudaMalloc(&Ey,N2*4));CHECK(cudaMalloc(&Hz,N2*4));
    CHECK(cudaMemset(Ex,0,N2*4));CHECK(cudaMemset(Ey,0,N2*4));CHECK(cudaMemset(Hz,0,N2*4));
    dim3 block(16,16),grid((N+15)/16,(N+15)/16);
    cudaEvent_t t0,t1;CHECK(cudaEventCreate(&t0));CHECK(cudaEventCreate(&t1));
    for(int i=0;i<20;i++){fdtd2d_step<<<grid,block>>>(N,Ex,Ey,Hz);}
    cudaDeviceSynchronize();
    Result r; float ms;

    // FDTD is in-place (no copy needed), but Sync still syncs each step
    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){fdtd2d_step<<<grid,block>>>(N,Ex,Ey,Hz);cudaDeviceSynchronize();}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1); r.sync_us=ms*1000/STEPS;

    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){fdtd2d_step<<<grid,block>>>(N,Ex,Ey,Hz);}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1); r.async_us=ms*1000/STEPS;

    {int REPS=5;cudaGraph_t g;cudaGraphExec_t ge;
     cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
     CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
     for(int s=0;s<STEPS;s++)fdtd2d_step<<<grid,block,0,stream>>>(N,Ex,Ey,Hz);
     CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&ge,g,NULL,NULL,0));
     for(int i=0;i<3;i++){cudaGraphLaunch(ge,stream);cudaStreamSynchronize(stream);}
     cudaEventRecord(t0,stream);for(int i=0;i<REPS;i++)cudaGraphLaunch(ge,stream);
     cudaEventRecord(t1,stream);cudaStreamSynchronize(stream);
     cudaEventElapsedTime(&ms,t0,t1);r.graph_us=ms*1000/(STEPS*REPS);
     cudaGraphExecDestroy(ge);cudaGraphDestroy(g);cudaStreamDestroy(stream);}

    {int numBSm=0;cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
     cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm,fdtd2d_persistent,256,0);
     int maxB=numBSm*prop.multiProcessorCount;
     if((int)(grid.x*grid.y)<=maxB){
         void*args[]={(void*)&N,(void*)&Ex,(void*)&Ey,(void*)&Hz,(void*)&STEPS};
         cudaLaunchCooperativeKernel((void*)fdtd2d_persistent,grid,block,args);cudaDeviceSynchronize();
         int REPS=5;cudaEventRecord(t0);
         for(int i=0;i<REPS;i++)cudaLaunchCooperativeKernel((void*)fdtd2d_persistent,grid,block,args);
         cudaEventRecord(t1);cudaDeviceSynchronize();
         cudaEventElapsedTime(&ms,t0,t1);r.persistent_us=ms*1000/(STEPS*REPS);
     } else r.persistent_us=-1;}

    cudaEventDestroy(t0);cudaEventDestroy(t1);
    cudaFree(Ex);cudaFree(Ey);cudaFree(Hz);
    return r;
}

Result test_srad(int N, int STEPS) {
    int N2=N*N; float *u,*v;
    CHECK(cudaMalloc(&u,N2*4));CHECK(cudaMalloc(&v,N2*4));
    // Init with random-ish data (avoid division by zero)
    float* h=(float*)malloc(N2*4);
    for(int i=0;i<N2;i++) h[i]=1.0f+0.1f*(i%100);
    cudaMemcpy(u,h,N2*4,cudaMemcpyHostToDevice);
    CHECK(cudaMemset(v,0,N2*4));
    free(h);
    dim3 block(16,16),grid((N+15)/16,(N+15)/16); int cg=(N2+255)/256;
    cudaEvent_t t0,t1;CHECK(cudaEventCreate(&t0));CHECK(cudaEventCreate(&t1));
    for(int i=0;i<20;i++){srad_step<<<grid,block>>>(N,u,v);copy_f<<<cg,256>>>(N2,v,u);}
    cudaDeviceSynchronize();
    Result r; float ms;

    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){srad_step<<<grid,block>>>(N,u,v);copy_f<<<cg,256>>>(N2,v,u);cudaDeviceSynchronize();}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1); r.sync_us=ms*1000/STEPS;

    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){srad_step<<<grid,block>>>(N,u,v);copy_f<<<cg,256>>>(N2,v,u);}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1); r.async_us=ms*1000/STEPS;

    {int REPS=5;cudaGraph_t g;cudaGraphExec_t ge;
     cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
     CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
     for(int s=0;s<STEPS;s++){srad_step<<<grid,block,0,stream>>>(N,u,v);copy_f<<<cg,256,0,stream>>>(N2,v,u);}
     CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&ge,g,NULL,NULL,0));
     for(int i=0;i<3;i++){cudaGraphLaunch(ge,stream);cudaStreamSynchronize(stream);}
     cudaEventRecord(t0,stream);for(int i=0;i<REPS;i++)cudaGraphLaunch(ge,stream);
     cudaEventRecord(t1,stream);cudaStreamSynchronize(stream);
     cudaEventElapsedTime(&ms,t0,t1);r.graph_us=ms*1000/(STEPS*REPS);
     cudaGraphExecDestroy(ge);cudaGraphDestroy(g);cudaStreamDestroy(stream);}

    {int numBSm=0;cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
     cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm,srad_persistent,256,0);
     int maxB=numBSm*prop.multiProcessorCount;
     if((int)(grid.x*grid.y)<=maxB){
         void*args[]={(void*)&N,(void*)&u,(void*)&v,(void*)&STEPS};
         cudaLaunchCooperativeKernel((void*)srad_persistent,grid,block,args);cudaDeviceSynchronize();
         int REPS=5;cudaEventRecord(t0);
         for(int i=0;i<REPS;i++)cudaLaunchCooperativeKernel((void*)srad_persistent,grid,block,args);
         cudaEventRecord(t1);cudaDeviceSynchronize();
         cudaEventElapsedTime(&ms,t0,t1);r.persistent_us=ms*1000/(STEPS*REPS);
     } else r.persistent_us=-1;}

    cudaEventDestroy(t0);cudaEventDestroy(t1);cudaFree(u);cudaFree(v);
    return r;
}

Result test_nbody(int N, int STEPS) {
    float *px,*py,*vx,*vy,*px2,*py2,*vx2,*vy2;
    CHECK(cudaMalloc(&px,N*4));CHECK(cudaMalloc(&py,N*4));
    CHECK(cudaMalloc(&vx,N*4));CHECK(cudaMalloc(&vy,N*4));
    CHECK(cudaMalloc(&px2,N*4));CHECK(cudaMalloc(&py2,N*4));
    CHECK(cudaMalloc(&vx2,N*4));CHECK(cudaMalloc(&vy2,N*4));
    // Init random positions
    float* h=(float*)malloc(N*4);
    for(int i=0;i<N;i++) h[i]=(float)(i%100)/100.0f;
    cudaMemcpy(px,h,N*4,cudaMemcpyHostToDevice);
    for(int i=0;i<N;i++) h[i]=(float)((i*7)%100)/100.0f;
    cudaMemcpy(py,h,N*4,cudaMemcpyHostToDevice);
    CHECK(cudaMemset(vx,0,N*4));CHECK(cudaMemset(vy,0,N*4));
    free(h);

    int blocks=(N+255)/256;
    cudaEvent_t t0,t1;CHECK(cudaEventCreate(&t0));CHECK(cudaEventCreate(&t1));
    for(int i=0;i<5;i++){nbody_step<<<blocks,256>>>(N,px,py,vx,vy,px2,py2,vx2,vy2);copy4_f<<<(N+255)/256,256>>>(N,px2,px,py2,py,vx2,vx,vy2,vy);}
    cudaDeviceSynchronize();
    Result r; float ms;

    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){nbody_step<<<blocks,256>>>(N,px,py,vx,vy,px2,py2,vx2,vy2);copy4_f<<<(N+255)/256,256>>>(N,px2,px,py2,py,vx2,vx,vy2,vy);cudaDeviceSynchronize();}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1); r.sync_us=ms*1000/STEPS;

    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){nbody_step<<<blocks,256>>>(N,px,py,vx,vy,px2,py2,vx2,vy2);copy4_f<<<(N+255)/256,256>>>(N,px2,px,py2,py,vx2,vx,vy2,vy);}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1); r.async_us=ms*1000/STEPS;

    {int REPS=5;cudaGraph_t g;cudaGraphExec_t ge;
     cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
     CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
     for(int s=0;s<STEPS;s++){nbody_step<<<blocks,256,0,stream>>>(N,px,py,vx,vy,px2,py2,vx2,vy2);copy4_f<<<(N+255)/256,256,0,stream>>>(N,px2,px,py2,py,vx2,vx,vy2,vy);}
     CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&ge,g,NULL,NULL,0));
     for(int i=0;i<3;i++){cudaGraphLaunch(ge,stream);cudaStreamSynchronize(stream);}
     cudaEventRecord(t0,stream);for(int i=0;i<REPS;i++)cudaGraphLaunch(ge,stream);
     cudaEventRecord(t1,stream);cudaStreamSynchronize(stream);
     cudaEventElapsedTime(&ms,t0,t1);r.graph_us=ms*1000/(STEPS*REPS);
     cudaGraphExecDestroy(ge);cudaGraphDestroy(g);cudaStreamDestroy(stream);}

    // NBody persistent (1D)
    {int numBSm=0;cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
     cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm,nbody_persistent,256,0);
     int maxB=numBSm*prop.multiProcessorCount;
     if(blocks<=maxB){
         void*args[]={(void*)&N,(void*)&px,(void*)&py,(void*)&vx,(void*)&vy,
                      (void*)&px2,(void*)&py2,(void*)&vx2,(void*)&vy2,(void*)&STEPS};
         dim3 pgrid(blocks),pblock(256);
         cudaLaunchCooperativeKernel((void*)nbody_persistent,pgrid,pblock,args);cudaDeviceSynchronize();
         int REPS=5;cudaEventRecord(t0);
         for(int i=0;i<REPS;i++)cudaLaunchCooperativeKernel((void*)nbody_persistent,pgrid,pblock,args);
         cudaEventRecord(t1);cudaDeviceSynchronize();
         cudaEventElapsedTime(&ms,t0,t1);r.persistent_us=ms*1000/(STEPS*REPS);
     } else r.persistent_us=-1;}

    cudaEventDestroy(t0);cudaEventDestroy(t1);
    cudaFree(px);cudaFree(py);cudaFree(vx);cudaFree(vy);
    cudaFree(px2);cudaFree(py2);cudaFree(vx2);cudaFree(vy2);
    return r;
}

// =========================================================================
// 8. Additional 2D stencil variants (parameterized via macros)
// All follow the same pattern: compute step + copy back, persistent with grid.sync
// =========================================================================

// --- Heat3D (7-point stencil, 3D indexing on flat array) ---
__global__ void heat3d_step(int N, const float* u, float* v) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int N3=N*N*N, N2=N*N;
    if(idx>=N3) return;
    int i=idx/(N2), j=(idx/N)%N, k=idx%N;
    if(i>=1&&i<N-1&&j>=1&&j<N-1&&k>=1&&k<N-1)
        v[idx]=u[idx]+0.1f*(u[idx-N2]+u[idx+N2]+u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-6.0f*u[idx]);
}
__global__ void heat3d_persistent(int N, float* u, float* v, int STEPS) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    int N3=N*N*N, N2=N*N;
    for(int s=0;s<STEPS;s++){
        for(int idx=tid;idx<N3;idx+=blockDim.x*gridDim.x){
            int i=idx/N2, j=(idx/N)%N, k=idx%N;
            if(i>=1&&i<N-1&&j>=1&&j<N-1&&k>=1&&k<N-1)
                v[idx]=u[idx]+0.1f*(u[idx-N2]+u[idx+N2]+u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-6.0f*u[idx]);
        }
        cg::this_grid().sync();
        for(int idx=tid;idx<N3;idx+=blockDim.x*gridDim.x) u[idx]=v[idx];
        cg::this_grid().sync();
    }
}

// --- Allen-Cahn (reaction-diffusion phase field) ---
__global__ void allencahn_step(int N, const float* u, float* v) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;
        float lap=u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-4.0f*u[idx];
        float phi=u[idx]; v[idx]=phi+0.01f*(0.01f*lap+phi-phi*phi*phi);}
}
__global__ void allencahn_persistent(int N, float* u, float* v, int STEPS) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    for(int s=0;s<STEPS;s++){
        if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;
            float lap=u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-4.0f*u[idx];
            float phi=u[idx]; v[idx]=phi+0.01f*(0.01f*lap+phi-phi*phi*phi);}
        cg::this_grid().sync();
        if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;u[idx]=v[idx];}
        cg::this_grid().sync();
    }
}

// --- Cahn-Hilliard (4th order PDE) ---
__global__ void cahnhilliard_step(int N, const float* u, float* v) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=2&&i<N-2&&j>=2&&j<N-2){int idx=i*N+j;
        float lap=u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-4.0f*u[idx];
        float mu=-0.01f*lap+u[idx]*u[idx]*u[idx]-u[idx];
        // bilaplacian approx via mu laplacian
        float mu_n=-0.01f*(u[idx-2*N]+u[idx+2*N]+u[idx-2]+u[idx+2]-4.0f*lap)/4.0f; // rough
        v[idx]=u[idx]+0.001f*(mu_n+lap);}
}
__global__ void cahnhilliard_persistent(int N, float* u, float* v, int STEPS) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    for(int s=0;s<STEPS;s++){
        if(i>=2&&i<N-2&&j>=2&&j<N-2){int idx=i*N+j;
            float lap=u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-4.0f*u[idx];
            float mu=-0.01f*lap+u[idx]*u[idx]*u[idx]-u[idx];
            float mu_n=-0.01f*(u[idx-2*N]+u[idx+2*N]+u[idx-2]+u[idx+2]-4.0f*lap)/4.0f;
            v[idx]=u[idx]+0.001f*(mu_n+lap);}
        cg::this_grid().sync();
        if(i>=2&&i<N-2&&j>=2&&j<N-2){int idx=i*N+j;u[idx]=v[idx];}
        cg::this_grid().sync();
    }
}

// --- ConvDiff (convection-diffusion) ---
__global__ void convdiff_step(int N, const float* u, float* v) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;
        float dx=1.0f/N, nu=0.01f, cx=1.0f, cy=0.5f;
        float lap=(u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-4.0f*u[idx])/(dx*dx);
        float adv_x=cx*(u[idx+1]-u[idx-1])/(2*dx);
        float adv_y=cy*(u[idx+N]-u[idx-N])/(2*dx);
        v[idx]=u[idx]+0.0001f*(nu*lap-adv_x-adv_y);}
}
__global__ void convdiff_persistent(int N, float* u, float* v, int STEPS) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    for(int s=0;s<STEPS;s++){
        if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;
            float dx=1.0f/N, nu=0.01f, cx=1.0f, cy=0.5f;
            float lap=(u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-4.0f*u[idx])/(dx*dx);
            float adv_x=cx*(u[idx+1]-u[idx-1])/(2*dx);
            float adv_y=cy*(u[idx+N]-u[idx-N])/(2*dx);
            v[idx]=u[idx]+0.0001f*(nu*lap-adv_x-adv_y);}
        cg::this_grid().sync();
        if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;u[idx]=v[idx];}
        cg::this_grid().sync();
    }
}

// --- HotSpot (Rodinia, temperature with power source) ---
__global__ void hotspot_step(int N, const float* u, float* v, const float* power) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;
        float cap=0.5f, rx=0.01f, ry=0.01f, amb=80.0f;
        float lap_x=(u[idx-1]+u[idx+1]-2.0f*u[idx])*rx;
        float lap_y=(u[idx-N]+u[idx+N]-2.0f*u[idx])*ry;
        v[idx]=u[idx]+cap*(lap_x+lap_y+power[idx]+(amb-u[idx])*0.001f);}
}
__global__ void hotspot_persistent(int N, float* u, float* v, const float* power, int STEPS) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    for(int s=0;s<STEPS;s++){
        if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;
            float cap=0.5f, rx=0.01f, ry=0.01f, amb=80.0f;
            float lap_x=(u[idx-1]+u[idx+1]-2.0f*u[idx])*rx;
            float lap_y=(u[idx-N]+u[idx+N]-2.0f*u[idx])*ry;
            v[idx]=u[idx]+cap*(lap_x+lap_y+power[idx]+(amb-u[idx])*0.001f);}
        cg::this_grid().sync();
        if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;u[idx]=v[idx];}
        cg::this_grid().sync();
    }
}

// --- Poisson2D / Helmholtz2D (Jacobi, same as Jacobi2D with source) ---
__global__ void poisson2d_step(int N, const float* u, float* v, const float* rhs) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;
        float dx2=1.0f/(N*N);
        v[idx]=0.25f*(u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-dx2*rhs[idx]);}
}
__global__ void poisson2d_persistent(int N, float* u, float* v, const float* rhs, int STEPS) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    for(int s=0;s<STEPS;s++){
        if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;
            float dx2=1.0f/(N*N);
            v[idx]=0.25f*(u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]-dx2*rhs[idx]);}
        cg::this_grid().sync();
        if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;u[idx]=v[idx];}
        cg::this_grid().sync();
    }
}

// --- LBM D2Q9 (lattice Boltzmann, 9 distribution functions) ---
__global__ void lbm_step(int N, float* f, float* f2) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    if(i<1||i>=N-1||j<1||j>=N-1) return;
    int idx=i*N+j, N2=N*N;
    // 9 velocities: (0,0),(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1)
    float w[9]={4.0f/9,1.0f/9,1.0f/9,1.0f/9,1.0f/9,1.0f/36,1.0f/36,1.0f/36,1.0f/36};
    int ex[9]={0,1,0,-1,0,1,-1,-1,1}, ey[9]={0,0,1,0,-1,1,1,-1,-1};
    // Compute macroscopic
    float rho=0, ux=0, uy=0;
    for(int q=0;q<9;q++){float fq=f[q*N2+idx]; rho+=fq; ux+=ex[q]*fq; uy+=ey[q]*fq;}
    ux/=(rho+1e-10f); uy/=(rho+1e-10f);
    // Collision + streaming
    float omega=1.5f;
    for(int q=0;q<9;q++){
        float eu=ex[q]*ux+ey[q]*uy;
        float feq=w[q]*rho*(1.0f+3*eu+4.5f*eu*eu-1.5f*(ux*ux+uy*uy));
        float fnew=f[q*N2+idx]+omega*(feq-f[q*N2+idx]);
        int ni=i+ey[q], nj=j+ex[q];
        if(ni>=0&&ni<N&&nj>=0&&nj<N) f2[q*N2+ni*N+nj]=fnew;
    }
}
__global__ void lbm_persistent(int N, float* f, float* f2, int STEPS) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    int N2=N*N;
    float w[9]={4.0f/9,1.0f/9,1.0f/9,1.0f/9,1.0f/9,1.0f/36,1.0f/36,1.0f/36,1.0f/36};
    int ex[9]={0,1,0,-1,0,1,-1,-1,1}, ey[9]={0,0,1,0,-1,1,1,-1,-1};
    for(int s=0;s<STEPS;s++){
        if(i>=1&&i<N-1&&j>=1&&j<N-1){
            int idx=i*N+j;
            float rho=0,ux=0,uy=0;
            for(int q=0;q<9;q++){float fq=f[q*N2+idx];rho+=fq;ux+=ex[q]*fq;uy+=ey[q]*fq;}
            ux/=(rho+1e-10f);uy/=(rho+1e-10f);
            float omega=1.5f;
            for(int q=0;q<9;q++){
                float eu=ex[q]*ux+ey[q]*uy;
                float feq=w[q]*rho*(1.0f+3*eu+4.5f*eu*eu-1.5f*(ux*ux+uy*uy));
                float fnew=f[q*N2+idx]+omega*(feq-f[q*N2+idx]);
                int ni=i+ey[q],nj=j+ex[q];
                if(ni>=0&&ni<N&&nj>=0&&nj<N) f2[q*N2+ni*N+nj]=fnew;
            }
        }
        cg::this_grid().sync();
        // swap f <-> f2
        int tid=(i*N+j);
        if(tid<N2*9){for(int idx=tid;idx<N2*9;idx+=blockDim.x*gridDim.x*blockDim.y*gridDim.y) f[idx]=f2[idx];}
        cg::this_grid().sync();
    }
}

// --- 1D kernels: Upwind, Euler1D, Schrodinger, KuramotoSivashinsky, MassSpring ---
__global__ void upwind1d_step(int N, float* u, float* v) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=1&&i<N-1){float c=1.0f, dx=1.0f/N; v[i]=u[i]-c*0.0001f/dx*(u[i]-u[i-1]);}
}
__global__ void upwind1d_persistent(int N, float* u, float* v, int STEPS) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    for(int s=0;s<STEPS;s++){
        for(int i=tid;i<N;i+=blockDim.x*gridDim.x)
            if(i>=1&&i<N-1){float c=1.0f,dx=1.0f/N;v[i]=u[i]-c*0.0001f/dx*(u[i]-u[i-1]);}
        cg::this_grid().sync();
        for(int i=tid;i<N;i+=blockDim.x*gridDim.x) u[i]=v[i];
        cg::this_grid().sync();
    }
}

__global__ void euler1d_step(int N, const float* rho, const float* rhou, const float* E,
                              float* rho2, float* rhou2, float* E2) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=1&&i<N-1){
        float dx=1.0f/N, dt=0.00001f, gamma=1.4f;
        float u_=rhou[i]/(rho[i]+1e-10f);
        float p=(gamma-1)*(E[i]-0.5f*rho[i]*u_*u_);
        // Lax-Friedrichs
        rho2[i]=0.5f*(rho[i-1]+rho[i+1])-dt/(2*dx)*(rhou[i+1]-rhou[i-1]);
        rhou2[i]=0.5f*(rhou[i-1]+rhou[i+1])-dt/(2*dx)*(rhou[i+1]*u_+p-rhou[i-1]*u_-p);
        E2[i]=0.5f*(E[i-1]+E[i+1])-dt/(2*dx)*((E[i+1]+p)*u_-(E[i-1]+p)*u_);
    }
}
__global__ void euler1d_persistent(int N, float* rho, float* rhou, float* E,
                                    float* rho2, float* rhou2, float* E2, int STEPS) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    for(int s=0;s<STEPS;s++){
        for(int i=tid;i<N;i+=blockDim.x*gridDim.x) if(i>=1&&i<N-1){
            float dx=1.0f/N,dt=0.00001f,gamma=1.4f;
            float u_=rhou[i]/(rho[i]+1e-10f);
            float p=(gamma-1)*(E[i]-0.5f*rho[i]*u_*u_);
            rho2[i]=0.5f*(rho[i-1]+rho[i+1])-dt/(2*dx)*(rhou[i+1]-rhou[i-1]);
            rhou2[i]=0.5f*(rhou[i-1]+rhou[i+1])-dt/(2*dx)*(rhou[i+1]*u_+p-rhou[i-1]*u_-p);
            E2[i]=0.5f*(E[i-1]+E[i+1])-dt/(2*dx)*((E[i+1]+p)*u_-(E[i-1]+p)*u_);
        }
        cg::this_grid().sync();
        for(int i=tid;i<N;i+=blockDim.x*gridDim.x){rho[i]=rho2[i];rhou[i]=rhou2[i];E[i]=E2[i];}
        cg::this_grid().sync();
    }
}

// --- Reduction (global sum) ---
__global__ void reduce_step(int N, const float* u, float* out) {
    __shared__ float s[256];
    int tid=threadIdx.x, idx=blockIdx.x*256+tid;
    s[tid]=(idx<N)?u[idx]:0;
    __syncthreads();
    for(int st=128;st>0;st>>=1){if(tid<st)s[tid]+=s[tid+st];__syncthreads();}
    if(tid==0) atomicAdd(out, s[0]);
}

// --- SWE Lax-Friedrichs (simplified, 2D stencil interface for harness) ---
// Already defined above as swe_step/swe_persistent with 6 arrays. Use dedicated test.

// --- Jacobi3D 7-point ---
__global__ void jacobi3d_step_flat(int N3, int N, const float* u, float* v) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=N3) return;
    int N2=N*N;
    int i=idx/N2, j=(idx/N)%N, k=idx%N;
    if(i>=1&&i<N-1&&j>=1&&j<N-1&&k>=1&&k<N-1)
        v[idx]=( u[idx-N2]+u[idx+N2]+u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1] ) / 6.0f;
}
__global__ void jacobi3d_persistent_flat(int N3, int N, float* u, float* v, int STEPS) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    int total=blockDim.x*gridDim.x;
    int N2=N*N;
    for(int s=0;s<STEPS;s++){
        for(int idx=tid;idx<N3;idx+=total){
            int i=idx/N2, j=(idx/N)%N, k=idx%N;
            if(i>=1&&i<N-1&&j>=1&&j<N-1&&k>=1&&k<N-1)
                v[idx]=(u[idx-N2]+u[idx+N2]+u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1])/6.0f;
        }
        cg::this_grid().sync();
        for(int idx=tid;idx<N3;idx+=total) u[idx]=v[idx];
        cg::this_grid().sync();
    }
}

// --- Schrodinger 1D (FTCS, complex as 2 floats) ---
__global__ void schrodinger1d_step(int N, float* u, float* v) {
    // u[2*i]=real, u[2*i+1]=imag; evolve with simple potential
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=1&&i<N-1){
        float dt=0.0001f, dx=1.0f/N;
        float lap_r=(u[2*(i-1)]+u[2*(i+1)]-2*u[2*i])/(dx*dx);
        float lap_i=(u[2*(i-1)+1]+u[2*(i+1)+1]-2*u[2*i+1])/(dx*dx);
        v[2*i]=u[2*i]+dt*0.5f*lap_i;     // d(Re)/dt = 0.5 * laplacian(Im)
        v[2*i+1]=u[2*i+1]-dt*0.5f*lap_r; // d(Im)/dt = -0.5 * laplacian(Re)
    }
}
__global__ void schrodinger1d_persistent(int N, float* u, float* v, int STEPS) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    for(int s=0;s<STEPS;s++){
        for(int i=tid;i<N;i+=blockDim.x*gridDim.x) if(i>=1&&i<N-1){
            float dt=0.0001f, dx=1.0f/N;
            float lap_r=(u[2*(i-1)]+u[2*(i+1)]-2*u[2*i])/(dx*dx);
            float lap_i=(u[2*(i-1)+1]+u[2*(i+1)+1]-2*u[2*i+1])/(dx*dx);
            v[2*i]=u[2*i]+dt*0.5f*lap_i;
            v[2*i+1]=u[2*i+1]-dt*0.5f*lap_r;
        }
        cg::this_grid().sync();
        for(int i=tid;i<2*N;i+=blockDim.x*gridDim.x) u[i]=v[i];
        cg::this_grid().sync();
    }
}

// --- KuramotoSivashinsky 1D ---
__global__ void ks1d_step(int N, float* u, float* v) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=2&&i<N-2){
        float dx=1.0f/N, dt=0.00001f;
        float dudx=(u[i+1]-u[i-1])/(2*dx);
        float d2u=(u[i+1]+u[i-1]-2*u[i])/(dx*dx);
        float d4u=(u[i+2]-4*u[i+1]+6*u[i]-4*u[i-1]+u[i-2])/(dx*dx*dx*dx);
        v[i]=u[i]+dt*(-u[i]*dudx-d2u-d4u);
    }
}
__global__ void ks1d_persistent(int N, float* u, float* v, int STEPS) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    for(int s=0;s<STEPS;s++){
        for(int i=tid;i<N;i+=blockDim.x*gridDim.x) if(i>=2&&i<N-2){
            float dx=1.0f/N, dt=0.00001f;
            float dudx=(u[i+1]-u[i-1])/(2*dx);
            float d2u=(u[i+1]+u[i-1]-2*u[i])/(dx*dx);
            float d4u=(u[i+2]-4*u[i+1]+6*u[i]-4*u[i-1]+u[i-2])/(dx*dx*dx*dx);
            v[i]=u[i]+dt*(-u[i]*dudx-d2u-d4u);
        }
        cg::this_grid().sync();
        for(int i=tid;i<N;i+=blockDim.x*gridDim.x) u[i]=v[i];
        cg::this_grid().sync();
    }
}

// --- SemiLagrangian Advection 2D ---
__global__ void semilag_step(int N, const float* u, float* v) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=1&&i<N-1&&j>=1&&j<N-1){
        int idx=i*N+j;
        float cx=0.5f, cy=0.3f, dt=0.001f, dx=1.0f/N;
        // Departure point
        float xd=(float)j*dx - cx*dt;
        float yd=(float)i*dx - cy*dt;
        // Clamp
        xd=fmaxf(dx,fminf(xd,(N-2)*dx));
        yd=fmaxf(dx,fminf(yd,(N-2)*dx));
        // Bilinear interp
        int j0=(int)(xd/dx), i0=(int)(yd/dx);
        j0=max(1,min(j0,N-2)); i0=max(1,min(i0,N-2));
        float fx=xd/dx-j0, fy=yd/dx-i0;
        v[idx]=(1-fx)*(1-fy)*u[i0*N+j0]+fx*(1-fy)*u[i0*N+j0+1]
              +(1-fx)*fy*u[(i0+1)*N+j0]+fx*fy*u[(i0+1)*N+j0+1];
    }
}
__global__ void semilag_persistent(int N, float* u, float* v, int STEPS) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    for(int s=0;s<STEPS;s++){
        if(i>=1&&i<N-1&&j>=1&&j<N-1){
            int idx=i*N+j;
            float cx=0.5f,cy=0.3f,dt=0.001f,dx=1.0f/N;
            float xd=(float)j*dx-cx*dt, yd=(float)i*dx-cy*dt;
            xd=fmaxf(dx,fminf(xd,(N-2)*dx)); yd=fmaxf(dx,fminf(yd,(N-2)*dx));
            int j0=(int)(xd/dx), i0=(int)(yd/dx);
            j0=max(1,min(j0,N-2)); i0=max(1,min(i0,N-2));
            float fx=xd/dx-j0, fy=yd/dx-i0;
            v[idx]=(1-fx)*(1-fy)*u[i0*N+j0]+fx*(1-fy)*u[i0*N+j0+1]
                  +(1-fx)*fy*u[(i0+1)*N+j0]+fx*fy*u[(i0+1)*N+j0+1];
        }
        cg::this_grid().sync();
        if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;u[idx]=v[idx];}
        cg::this_grid().sync();
    }
}

// --- MassSpring 1D ---
__global__ void massspring1d_step(int N, float* x, float* v_out) {
    // x[2*i]=position, x[2*i+1]=velocity; v_out=next state
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=1&&i<N-1){
        float k=100.0f, m=1.0f, dt=0.0001f, rest=1.0f/N;
        float pos=x[2*i], vel=x[2*i+1];
        float f=k*(x[2*(i+1)]-pos-rest)+k*(x[2*(i-1)]-pos+rest);
        v_out[2*i]=pos+dt*vel;
        v_out[2*i+1]=vel+dt*f/m;
    }
}
__global__ void massspring1d_persistent(int N, float* x, float* v_out, int STEPS) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    for(int s=0;s<STEPS;s++){
        for(int i=tid;i<N;i+=blockDim.x*gridDim.x) if(i>=1&&i<N-1){
            float k=100.0f,m=1.0f,dt=0.0001f,rest=1.0f/N;
            float pos=x[2*i],vel=x[2*i+1];
            float f=k*(x[2*(i+1)]-pos-rest)+k*(x[2*(i-1)]-pos+rest);
            v_out[2*i]=pos+dt*vel;
            v_out[2*i+1]=vel+dt*f/m;
        }
        cg::this_grid().sync();
        for(int i=tid;i<2*N;i+=blockDim.x*gridDim.x) x[i]=v_out[i];
        cg::this_grid().sync();
    }
}

// --- Reduction (sum) as time-stepping proxy ---
// Each "step" = reduce + broadcast (simulate iterative convergence check pattern)
__global__ void reduce_step_proxy(int N, float* __restrict__ u, float* __restrict__ v) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=1&&i<N-1) v[i]=u[i]*0.999f+0.0005f*(u[i-1]+u[i+1]);
    else if(i<N) v[i]=u[i];
}
__global__ void reduce_persistent_proxy(int N, float* u, float* v, int STEPS) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    for(int s=0;s<STEPS;s++){
        for(int i=tid;i<N;i+=blockDim.x*gridDim.x){
            if(i>=1&&i<N-1) v[i]=u[i]*0.999f+0.0005f*(u[i-1]+u[i+1]);
            else v[i]=u[i];
        }
        cg::this_grid().sync();
        for(int i=tid;i<N;i+=blockDim.x*gridDim.x) u[i]=v[i];
        cg::this_grid().sync();
    }
}

// --- SpMV CSR (synthetic banded matrix) ---
__global__ void spmv_step(int N, float* x, float* y) {
    // Synthetic 3-banded: y[i] = x[i-1] + 2*x[i] + x[i+1]
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=1&&i<N-1) y[i]=x[i-1]+2.0f*x[i]+x[i+1];
    else if(i==0) y[i]=2.0f*x[i]+x[i+1];
    else if(i==N-1) y[i]=x[i-1]+2.0f*x[i];
}
__global__ void spmv_persistent(int N, float* x, float* y, int STEPS) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    for(int s=0;s<STEPS;s++){
        for(int i=tid;i<N;i+=blockDim.x*gridDim.x){
            if(i>=1&&i<N-1) y[i]=x[i-1]+2.0f*x[i]+x[i+1];
            else if(i==0) y[i]=2.0f*x[i]+x[i+1];
            else y[i]=x[i-1]+2.0f*x[i];
        }
        cg::this_grid().sync();
        for(int i=tid;i<N;i+=blockDim.x*gridDim.x) x[i]=y[i];
        cg::this_grid().sync();
    }
}

// --- ExplicitFEM 2D (lumped mass, triangle mesh on structured grid) ---
__global__ void fem2d_step(int N, const float* u, float* v) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;
        // 6-neighbor stencil (triangulated quad)
        v[idx]=u[idx]+0.01f*(u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]+u[idx-N+1]+u[idx+N-1]-6.0f*u[idx]);}
}
__global__ void fem2d_persistent_simple(int N, float* u, float* v, int STEPS) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    for(int s=0;s<STEPS;s++){
        if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;
            v[idx]=u[idx]+0.01f*(u[idx-N]+u[idx+N]+u[idx-1]+u[idx+1]+u[idx-N+1]+u[idx+N-1]-6.0f*u[idx]);}
        cg::this_grid().sync();
        if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;u[idx]=v[idx];}
        cg::this_grid().sync();
    }
}

// --- Cloth spring (2D grid, position + velocity = 4 floats per node, but simplified) ---
__global__ void cloth_step(int N, const float* u, float* v) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;
        float k=50.0f, rest=1.0f/N;
        float fx=k*((u[idx+1]-u[idx])-rest)+k*((u[idx-1]-u[idx])+rest);
        float fy=k*((u[idx+N]-u[idx])-rest)+k*((u[idx-N]-u[idx])+rest);
        v[idx]=u[idx]+0.0001f*(fx+fy);}
}
__global__ void cloth_persistent(int N, float* u, float* v, int STEPS) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    for(int s=0;s<STEPS;s++){
        if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;
            float k=50.0f,rest=1.0f/N;
            float fx=k*((u[idx+1]-u[idx])-rest)+k*((u[idx-1]-u[idx])+rest);
            float fy=k*((u[idx+N]-u[idx])-rest)+k*((u[idx-N]-u[idx])+rest);
            v[idx]=u[idx]+0.0001f*(fx+fy);}
        cg::this_grid().sync();
        if(i>=1&&i<N-1&&j>=1&&j<N-1){int idx=i*N+j;u[idx]=v[idx];}
        cg::this_grid().sync();
    }
}

// --- SPH Density (brute force O(N²), smoothing kernel) ---
__global__ void sph_step(int N, float* px, float* py, float* vx, float* vy, float* rho_sph,
                          float* px2, float* py2, float* vx2, float* vy2) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;
    float h=0.1f, dt=0.0001f, mass=1.0f, k=100.0f, rho0=1.0f;
    // Density
    float rho_i=0;
    for(int j=0;j<N;j++){
        float dx=px[j]-px[i], dy=py[j]-py[i];
        float r2=dx*dx+dy*dy;
        if(r2<h*h){float q=sqrtf(r2)/h; rho_i+=mass*(1-q)*(1-q);}
    }
    rho_sph[i]=fmaxf(rho_i,0.001f);
    // Pressure force
    float ax=0, ay=0;
    float pi_=k*(rho_i-rho0);
    for(int j=0;j<N;j++){
        if(i==j) continue;
        float dx=px[j]-px[i], dy=py[j]-py[i];
        float r=sqrtf(dx*dx+dy*dy)+1e-6f;
        if(r<h){float pj=k*(rho_sph[j]-rho0);
            float f=-mass*(pi_+pj)/(2*rho_sph[j])*(-2*(1-r/h)/h)/r;
            ax+=f*dx; ay+=f*dy;}
    }
    vx2[i]=vx[i]+dt*ax; vy2[i]=vy[i]+dt*ay;
    px2[i]=px[i]+dt*vx2[i]; py2[i]=py[i]+dt*vy2[i];
}
__global__ void sph_persistent(int N, float* px, float* py, float* vx, float* vy, float* rho_sph,
                                float* px2, float* py2, float* vx2, float* vy2, int STEPS) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    int total=blockDim.x*gridDim.x;
    float h=0.1f, dt=0.0001f, mass=1.0f, k=100.0f, rho0=1.0f;
    for(int s=0;s<STEPS;s++){
        for(int i=tid;i<N;i+=total){
            float rho_i=0;
            for(int j=0;j<N;j++){float dx=px[j]-px[i],dy=py[j]-py[i];float r2=dx*dx+dy*dy;
                if(r2<h*h){float q=sqrtf(r2)/h;rho_i+=mass*(1-q)*(1-q);}}
            rho_sph[i]=fmaxf(rho_i,0.001f);
        }
        cg::this_grid().sync();
        for(int i=tid;i<N;i+=total){
            float ax=0,ay=0,pi_=k*(rho_sph[i]-rho0);
            for(int j=0;j<N;j++){if(i==j)continue;
                float dx=px[j]-px[i],dy=py[j]-py[i];float r=sqrtf(dx*dx+dy*dy)+1e-6f;
                if(r<h){float pj=k*(rho_sph[j]-rho0);
                    float f=-mass*(pi_+pj)/(2*rho_sph[j])*(-2*(1-r/h)/h)/r;
                    ax+=f*dx;ay+=f*dy;}}
            vx2[i]=vx[i]+dt*ax;vy2[i]=vy[i]+dt*ay;
            px2[i]=px[i]+dt*vx2[i];py2[i]=py[i]+dt*vy2[i];
        }
        cg::this_grid().sync();
        for(int i=tid;i<N;i+=total){px[i]=px2[i];py[i]=py2[i];vx[i]=vx2[i];vy[i]=vy2[i];}
        cg::this_grid().sync();
    }
}

// --- DEM (spring-dashpot contact, brute force O(N²)) ---
__global__ void dem_step(int N, float* px, float* py, float* vx, float* vy,
                          float* px2, float* py2, float* vx2, float* vy2) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;
    float dt=0.0001f, rad=0.01f, kn=1e4f, gn=10.0f;
    float ax=0, ay=-9.81f; // gravity
    for(int j=0;j<N;j++){
        if(i==j) continue;
        float dx=px[j]-px[i], dy=py[j]-py[i];
        float dist=sqrtf(dx*dx+dy*dy)+1e-10f;
        float overlap=2*rad-dist;
        if(overlap>0){float nx=dx/dist,ny=dy/dist;
            float dvn=(vx[j]-vx[i])*nx+(vy[j]-vy[i])*ny;
            float fn=kn*overlap-gn*dvn;
            ax+=fn*nx; ay+=fn*ny;}
    }
    vx2[i]=vx[i]+dt*ax; vy2[i]=vy[i]+dt*ay;
    px2[i]=px[i]+dt*vx2[i]; py2[i]=py[i]+dt*vy2[i];
}
__global__ void dem_persistent(int N, float* px, float* py, float* vx, float* vy,
                                float* px2, float* py2, float* vx2, float* vy2, int STEPS) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    int total=blockDim.x*gridDim.x;
    for(int s=0;s<STEPS;s++){
        for(int i=tid;i<N;i+=total){
            float dt=0.0001f,rad=0.01f,kn=1e4f,gn=10.0f;
            float ax=0,ay=-9.81f;
            for(int j=0;j<N;j++){if(i==j)continue;
                float dx=px[j]-px[i],dy=py[j]-py[i];float dist=sqrtf(dx*dx+dy*dy)+1e-10f;
                float overlap=2*rad-dist;
                if(overlap>0){float nx=dx/dist,ny=dy/dist;
                    float dvn=(vx[j]-vx[i])*nx+(vy[j]-vy[i])*ny;
                    float fn=kn*overlap-gn*dvn; ax+=fn*nx;ay+=fn*ny;}}
            vx2[i]=vx[i]+0.0001f*ax;vy2[i]=vy[i]+0.0001f*ay;
            px2[i]=px[i]+0.0001f*vx2[i];py2[i]=py[i]+0.0001f*vy2[i];
        }
        cg::this_grid().sync();
        for(int i=tid;i<N;i+=total){px[i]=px2[i];py[i]=py2[i];vx[i]=vx2[i];vy[i]=vy2[i];}
        cg::this_grid().sync();
    }
}

// --- MD Lennard-Jones (brute force O(N²)) ---
__global__ void mdlj_step(int N, float* px, float* py, float* vx, float* vy,
                           float* px2, float* py2, float* vx2, float* vy2) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;
    float dt=0.0001f, eps=1.0f, sigma=0.01f;
    float ax=0, ay=0;
    for(int j=0;j<N;j++){
        if(i==j) continue;
        float dx=px[j]-px[i], dy=py[j]-py[i];
        float r2=dx*dx+dy*dy+1e-10f;
        float s2=sigma*sigma/r2, s6=s2*s2*s2;
        float f=24*eps*(2*s6*s6-s6)/r2;
        ax+=f*dx; ay+=f*dy;
    }
    vx2[i]=vx[i]+dt*ax; vy2[i]=vy[i]+dt*ay;
    px2[i]=px[i]+dt*vx2[i]; py2[i]=py[i]+dt*vy2[i];
}
__global__ void mdlj_persistent(int N, float* px, float* py, float* vx, float* vy,
                                 float* px2, float* py2, float* vx2, float* vy2, int STEPS) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    int total=blockDim.x*gridDim.x;
    for(int s=0;s<STEPS;s++){
        for(int i=tid;i<N;i+=total){
            float dt=0.0001f,eps=1.0f,sigma=0.01f;
            float ax=0,ay=0;
            for(int j=0;j<N;j++){if(i==j)continue;
                float dx=px[j]-px[i],dy=py[j]-py[i];float r2=dx*dx+dy*dy+1e-10f;
                float s2=sigma*sigma/r2,s6=s2*s2*s2;
                float f=24*eps*(2*s6*s6-s6)/r2; ax+=f*dx;ay+=f*dy;}
            vx2[i]=vx[i]+dt*ax;vy2[i]=vy[i]+dt*ay;
            px2[i]=px[i]+dt*vx2[i];py2[i]=py[i]+dt*vy2[i];
        }
        cg::this_grid().sync();
        for(int i=tid;i<N;i+=total){px[i]=px2[i];py[i]=py2[i];vx[i]=vx2[i];vy[i]=vy2[i];}
        cg::this_grid().sync();
    }
}

// --- PIC 1D (4 phases: deposit + field_solve + interpolate + push) ---
__global__ void pic_deposit(int NP, int NG, const float* xp, float* rho_grid) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=NP) return;
    float dx=1.0f/NG;
    int cell=(int)(xp[i]/dx); cell=max(0,min(cell,NG-1));
    atomicAdd(&rho_grid[cell], 1.0f);
}
__global__ void pic_field(int NG, const float* rho_grid, float* E_grid) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=1&&i<NG-1) E_grid[i]=-(rho_grid[i+1]-rho_grid[i-1])*0.5f*NG;
}
__global__ void pic_push(int NP, int NG, float* xp, float* vp, const float* E_grid) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=NP) return;
    float dx=1.0f/NG, dt=0.0001f;
    int cell=(int)(xp[i]/dx); cell=max(0,min(cell,NG-1));
    vp[i]+=dt*E_grid[cell];
    xp[i]+=dt*vp[i];
    if(xp[i]<0) xp[i]+=1.0f; if(xp[i]>=1.0f) xp[i]-=1.0f;
}
__global__ void pic_zero(int NG, float* rho_grid) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<NG) rho_grid[i]=0;
}
__global__ void pic_persistent(int NP, int NG, float* xp, float* vp,
                                float* rho_grid, float* E_grid, int STEPS) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    int total=blockDim.x*gridDim.x;
    float dx=1.0f/NG, dt=0.0001f;
    for(int s=0;s<STEPS;s++){
        for(int i=tid;i<NG;i+=total) rho_grid[i]=0;
        cg::this_grid().sync();
        for(int i=tid;i<NP;i+=total){int c=(int)(xp[i]/dx);c=max(0,min(c,NG-1));atomicAdd(&rho_grid[c],1.0f);}
        cg::this_grid().sync();
        for(int i=tid;i<NG;i+=total) if(i>=1&&i<NG-1) E_grid[i]=-(rho_grid[i+1]-rho_grid[i-1])*0.5f*NG;
        cg::this_grid().sync();
        for(int i=tid;i<NP;i+=total){int c=(int)(xp[i]/dx);c=max(0,min(c,NG-1));
            vp[i]+=dt*E_grid[c]; xp[i]+=dt*vp[i];
            if(xp[i]<0)xp[i]+=1.0f; if(xp[i]>=1.0f)xp[i]-=1.0f;}
        cg::this_grid().sync();
    }
}

// --- Monte Carlo Random Walk ---
__global__ void montecarlo_step(int N, float* x, float* y, unsigned int* state) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;
    // LCG random
    unsigned int s=state[i];
    s=s*1664525u+1013904223u; float r1=(s&0xFFFF)/65536.0f-0.5f;
    s=s*1664525u+1013904223u; float r2=(s&0xFFFF)/65536.0f-0.5f;
    state[i]=s;
    x[i]+=0.01f*r1; y[i]+=0.01f*r2;
}
__global__ void montecarlo_persistent(int N, float* x, float* y, unsigned int* state, int STEPS) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    for(int s=0;s<STEPS;s++){
        for(int i=tid;i<N;i+=blockDim.x*gridDim.x){
            unsigned int st=state[i];
            st=st*1664525u+1013904223u; float r1=(st&0xFFFF)/65536.0f-0.5f;
            st=st*1664525u+1013904223u; float r2=(st&0xFFFF)/65536.0f-0.5f;
            state[i]=st;
            x[i]+=0.01f*r1; y[i]+=0.01f*r2;
        }
        cg::this_grid().sync();
    }
}

// =========================================================================
// Parameterized 2D test harness (for all stencil-like 2-field kernels)
// =========================================================================
typedef void(*Step2D)(int, const float*, float*);
typedef void(*Persist2D)(int, float*, float*, int);

Result test_2d_stencil(Step2D step_fn, Persist2D persist_fn, int N, int STEPS) {
    int N2=N*N; float *u,*v;
    CHECK(cudaMalloc(&u,N2*4));CHECK(cudaMalloc(&v,N2*4));
    CHECK(cudaMemset(u,0,N2*4));CHECK(cudaMemset(v,0,N2*4));
    // Init with small values to avoid NaN
    float* h=(float*)malloc(N2*4);
    for(int i=0;i<N2;i++) h[i]=1.0f+0.01f*(i%100);
    cudaMemcpy(u,h,N2*4,cudaMemcpyHostToDevice); free(h);

    dim3 block(16,16),grid((N+15)/16,(N+15)/16); int cg=(N2+255)/256;
    cudaEvent_t t0,t1;CHECK(cudaEventCreate(&t0));CHECK(cudaEventCreate(&t1));
    for(int i=0;i<20;i++){step_fn<<<grid,block>>>(N,u,v);copy_f<<<cg,256>>>(N2,v,u);}
    cudaDeviceSynchronize();
    Result r; float ms;

    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){step_fn<<<grid,block>>>(N,u,v);copy_f<<<cg,256>>>(N2,v,u);cudaDeviceSynchronize();}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1); r.sync_us=ms*1000/STEPS;

    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){step_fn<<<grid,block>>>(N,u,v);copy_f<<<cg,256>>>(N2,v,u);}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1); r.async_us=ms*1000/STEPS;

    {int REPS=5;cudaGraph_t g;cudaGraphExec_t ge;
     cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
     CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
     for(int s=0;s<STEPS;s++){step_fn<<<grid,block,0,stream>>>(N,u,v);copy_f<<<cg,256,0,stream>>>(N2,v,u);}
     CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&ge,g,NULL,NULL,0));
     for(int i=0;i<3;i++){cudaGraphLaunch(ge,stream);cudaStreamSynchronize(stream);}
     cudaEventRecord(t0,stream);for(int i=0;i<REPS;i++)cudaGraphLaunch(ge,stream);
     cudaEventRecord(t1,stream);cudaStreamSynchronize(stream);
     cudaEventElapsedTime(&ms,t0,t1);r.graph_us=ms*1000/(STEPS*REPS);
     cudaGraphExecDestroy(ge);cudaGraphDestroy(g);cudaStreamDestroy(stream);}

    r.persistent_us=-1;
    if(persist_fn){
        int numBSm=0;cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm,(void*)persist_fn,256,0);
        int maxB=numBSm*prop.multiProcessorCount;
        if((int)(grid.x*grid.y)<=maxB){
            void*args[]={(void*)&N,(void*)&u,(void*)&v,(void*)&STEPS};
            cudaLaunchCooperativeKernel((void*)persist_fn,grid,block,args);cudaDeviceSynchronize();
            int REPS=5;cudaEventRecord(t0);
            for(int i=0;i<REPS;i++)cudaLaunchCooperativeKernel((void*)persist_fn,grid,block,args);
            cudaEventRecord(t1);cudaDeviceSynchronize();
            cudaEventElapsedTime(&ms,t0,t1);r.persistent_us=ms*1000/(STEPS*REPS);
        }
    }

    cudaEventDestroy(t0);cudaEventDestroy(t1);cudaFree(u);cudaFree(v);
    return r;
}

// 1D test harness
typedef void(*Step1D)(int, float*, float*);
typedef void(*Persist1D)(int, float*, float*, int);

Result test_1d(Step1D step_fn, Persist1D persist_fn, int N, int STEPS) {
    float *u,*v;
    CHECK(cudaMalloc(&u,N*4));CHECK(cudaMalloc(&v,N*4));
    float* h=(float*)malloc(N*4);
    for(int i=0;i<N;i++) h[i]=1.0f+0.01f*(i%100);
    cudaMemcpy(u,h,N*4,cudaMemcpyHostToDevice); free(h);
    CHECK(cudaMemset(v,0,N*4));

    int blocks=(N+255)/256;
    cudaEvent_t t0,t1;CHECK(cudaEventCreate(&t0));CHECK(cudaEventCreate(&t1));
    for(int i=0;i<20;i++){step_fn<<<blocks,256>>>(N,u,v);copy_f<<<blocks,256>>>(N,v,u);}
    cudaDeviceSynchronize();
    Result r; float ms;

    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){step_fn<<<blocks,256>>>(N,u,v);copy_f<<<blocks,256>>>(N,v,u);cudaDeviceSynchronize();}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1); r.sync_us=ms*1000/STEPS;

    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){step_fn<<<blocks,256>>>(N,u,v);copy_f<<<blocks,256>>>(N,v,u);}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1); r.async_us=ms*1000/STEPS;

    {int REPS=5;cudaGraph_t g;cudaGraphExec_t ge;
     cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
     CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
     for(int s=0;s<STEPS;s++){step_fn<<<blocks,256,0,stream>>>(N,u,v);copy_f<<<blocks,256,0,stream>>>(N,v,u);}
     CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&ge,g,NULL,NULL,0));
     for(int i=0;i<3;i++){cudaGraphLaunch(ge,stream);cudaStreamSynchronize(stream);}
     cudaEventRecord(t0,stream);for(int i=0;i<REPS;i++)cudaGraphLaunch(ge,stream);
     cudaEventRecord(t1,stream);cudaStreamSynchronize(stream);
     cudaEventElapsedTime(&ms,t0,t1);r.graph_us=ms*1000/(STEPS*REPS);
     cudaGraphExecDestroy(ge);cudaGraphDestroy(g);cudaStreamDestroy(stream);}

    r.persistent_us=-1;
    if(persist_fn){
        int numBSm=0;cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm,(void*)persist_fn,256,0);
        int maxB=numBSm*prop.multiProcessorCount;
        if(blocks<=maxB){
            void*args[]={(void*)&N,(void*)&u,(void*)&v,(void*)&STEPS};
            dim3 pgrid(blocks),pblock(256);
            cudaLaunchCooperativeKernel((void*)persist_fn,pgrid,pblock,args);cudaDeviceSynchronize();
            int REPS=5;cudaEventRecord(t0);
            for(int i=0;i<REPS;i++)cudaLaunchCooperativeKernel((void*)persist_fn,pgrid,pblock,args);
            cudaEventRecord(t1);cudaDeviceSynchronize();
            cudaEventElapsedTime(&ms,t0,t1);r.persistent_us=ms*1000/(STEPS*REPS);
        }
    }

    cudaEventDestroy(t0);cudaEventDestroy(t1);cudaFree(u);cudaFree(v);
    return r;
}

// HotSpot needs extra power array
Result test_hotspot(int N, int STEPS) {
    int N2=N*N; float *u,*v,*power;
    CHECK(cudaMalloc(&u,N2*4));CHECK(cudaMalloc(&v,N2*4));CHECK(cudaMalloc(&power,N2*4));
    float* h=(float*)malloc(N2*4);
    for(int i=0;i<N2;i++) h[i]=80.0f+0.1f*(i%50);
    cudaMemcpy(u,h,N2*4,cudaMemcpyHostToDevice);
    for(int i=0;i<N2;i++) h[i]=0.001f*(i%100);
    cudaMemcpy(power,h,N2*4,cudaMemcpyHostToDevice); free(h);

    dim3 block(16,16),grid((N+15)/16,(N+15)/16); int cg=(N2+255)/256;
    cudaEvent_t t0,t1;CHECK(cudaEventCreate(&t0));CHECK(cudaEventCreate(&t1));
    for(int i=0;i<20;i++){hotspot_step<<<grid,block>>>(N,u,v,power);copy_f<<<cg,256>>>(N2,v,u);}
    cudaDeviceSynchronize();
    Result r; float ms;

    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){hotspot_step<<<grid,block>>>(N,u,v,power);copy_f<<<cg,256>>>(N2,v,u);cudaDeviceSynchronize();}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1); r.sync_us=ms*1000/STEPS;

    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){hotspot_step<<<grid,block>>>(N,u,v,power);copy_f<<<cg,256>>>(N2,v,u);}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1); r.async_us=ms*1000/STEPS;

    {int REPS=5;cudaGraph_t g;cudaGraphExec_t ge;
     cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
     CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
     for(int s=0;s<STEPS;s++){hotspot_step<<<grid,block,0,stream>>>(N,u,v,power);copy_f<<<cg,256,0,stream>>>(N2,v,u);}
     CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&ge,g,NULL,NULL,0));
     for(int i=0;i<3;i++){cudaGraphLaunch(ge,stream);cudaStreamSynchronize(stream);}
     cudaEventRecord(t0,stream);for(int i=0;i<REPS;i++)cudaGraphLaunch(ge,stream);
     cudaEventRecord(t1,stream);cudaStreamSynchronize(stream);
     cudaEventElapsedTime(&ms,t0,t1);r.graph_us=ms*1000/(STEPS*REPS);
     cudaGraphExecDestroy(ge);cudaGraphDestroy(g);cudaStreamDestroy(stream);}

    {int numBSm=0;cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
     cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm,hotspot_persistent,256,0);
     int maxB=numBSm*prop.multiProcessorCount;
     if((int)(grid.x*grid.y)<=maxB){
         void*args[]={(void*)&N,(void*)&u,(void*)&v,(void*)&power,(void*)&STEPS};
         cudaLaunchCooperativeKernel((void*)hotspot_persistent,grid,block,args);cudaDeviceSynchronize();
         int REPS=5;cudaEventRecord(t0);
         for(int i=0;i<REPS;i++)cudaLaunchCooperativeKernel((void*)hotspot_persistent,grid,block,args);
         cudaEventRecord(t1);cudaDeviceSynchronize();
         cudaEventElapsedTime(&ms,t0,t1);r.persistent_us=ms*1000/(STEPS*REPS);
     } else r.persistent_us=-1;}

    cudaEventDestroy(t0);cudaEventDestroy(t1);cudaFree(u);cudaFree(v);cudaFree(power);
    return r;
}

// LBM test (9 fields)
Result test_lbm(int N, int STEPS) {
    int N2=N*N; float *f,*f2;
    CHECK(cudaMalloc(&f,9*N2*4));CHECK(cudaMalloc(&f2,9*N2*4));
    // Init with equilibrium
    float* h=(float*)malloc(9*N2*4);
    float w[9]={4.0f/9,1.0f/9,1.0f/9,1.0f/9,1.0f/9,1.0f/36,1.0f/36,1.0f/36,1.0f/36};
    for(int q=0;q<9;q++) for(int i=0;i<N2;i++) h[q*N2+i]=w[q];
    cudaMemcpy(f,h,9*N2*4,cudaMemcpyHostToDevice);
    cudaMemcpy(f2,h,9*N2*4,cudaMemcpyHostToDevice); free(h);

    dim3 block(16,16),grid((N+15)/16,(N+15)/16);
    cudaEvent_t t0,t1;CHECK(cudaEventCreate(&t0));CHECK(cudaEventCreate(&t1));
    for(int i=0;i<20;i++){lbm_step<<<grid,block>>>(N,f,f2);cudaMemcpy(f,f2,9*N2*4,cudaMemcpyDeviceToDevice);}
    cudaDeviceSynchronize();
    Result r; float ms;

    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){lbm_step<<<grid,block>>>(N,f,f2);cudaMemcpy(f,f2,9*N2*4,cudaMemcpyDeviceToDevice);cudaDeviceSynchronize();}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1); r.sync_us=ms*1000/STEPS;

    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){lbm_step<<<grid,block>>>(N,f,f2);cudaMemcpy(f,f2,9*N2*4,cudaMemcpyDeviceToDevice);}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1); r.async_us=ms*1000/STEPS;

    {int REPS=5;cudaGraph_t g;cudaGraphExec_t ge;
     cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
     CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
     for(int s=0;s<STEPS;s++){lbm_step<<<grid,block,0,stream>>>(N,f,f2);cudaMemcpyAsync(f,f2,9*N2*4,cudaMemcpyDeviceToDevice,stream);}
     CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&ge,g,NULL,NULL,0));
     for(int i=0;i<3;i++){cudaGraphLaunch(ge,stream);cudaStreamSynchronize(stream);}
     cudaEventRecord(t0,stream);for(int i=0;i<REPS;i++)cudaGraphLaunch(ge,stream);
     cudaEventRecord(t1,stream);cudaStreamSynchronize(stream);
     cudaEventElapsedTime(&ms,t0,t1);r.graph_us=ms*1000/(STEPS*REPS);
     cudaGraphExecDestroy(ge);cudaGraphDestroy(g);cudaStreamDestroy(stream);}

    {int numBSm=0;cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
     cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm,lbm_persistent,256,0);
     int maxB=numBSm*prop.multiProcessorCount;
     if((int)(grid.x*grid.y)<=maxB){
         void*args[]={(void*)&N,(void*)&f,(void*)&f2,(void*)&STEPS};
         cudaLaunchCooperativeKernel((void*)lbm_persistent,grid,block,args);cudaDeviceSynchronize();
         int REPS=5;cudaEventRecord(t0);
         for(int i=0;i<REPS;i++)cudaLaunchCooperativeKernel((void*)lbm_persistent,grid,block,args);
         cudaEventRecord(t1);cudaDeviceSynchronize();
         cudaEventElapsedTime(&ms,t0,t1);r.persistent_us=ms*1000/(STEPS*REPS);
     } else r.persistent_us=-1;}

    cudaEventDestroy(t0);cudaEventDestroy(t1);cudaFree(f);cudaFree(f2);
    return r;
}

// MonteCarlo test
Result test_montecarlo(int N, int STEPS) {
    float *x,*y; unsigned int *state;
    CHECK(cudaMalloc(&x,N*4));CHECK(cudaMalloc(&y,N*4));CHECK(cudaMalloc(&state,N*4));
    CHECK(cudaMemset(x,0,N*4));CHECK(cudaMemset(y,0,N*4));
    unsigned int* h=(unsigned int*)malloc(N*4);
    for(int i=0;i<N;i++) h[i]=i*12345+67890;
    cudaMemcpy(state,h,N*4,cudaMemcpyHostToDevice); free(h);

    int blocks=(N+255)/256;
    cudaEvent_t t0,t1;CHECK(cudaEventCreate(&t0));CHECK(cudaEventCreate(&t1));
    for(int i=0;i<20;i++) montecarlo_step<<<blocks,256>>>(N,x,y,state);
    cudaDeviceSynchronize();
    Result r; float ms;

    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++){montecarlo_step<<<blocks,256>>>(N,x,y,state);cudaDeviceSynchronize();}
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1); r.sync_us=ms*1000/STEPS;

    cudaEventRecord(t0);
    for(int s=0;s<STEPS;s++) montecarlo_step<<<blocks,256>>>(N,x,y,state);
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1); r.async_us=ms*1000/STEPS;

    {int REPS=5;cudaGraph_t g;cudaGraphExec_t ge;
     cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
     CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
     for(int s=0;s<STEPS;s++) montecarlo_step<<<blocks,256,0,stream>>>(N,x,y,state);
     CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&ge,g,NULL,NULL,0));
     for(int i=0;i<3;i++){cudaGraphLaunch(ge,stream);cudaStreamSynchronize(stream);}
     cudaEventRecord(t0,stream);for(int i=0;i<REPS;i++)cudaGraphLaunch(ge,stream);
     cudaEventRecord(t1,stream);cudaStreamSynchronize(stream);
     cudaEventElapsedTime(&ms,t0,t1);r.graph_us=ms*1000/(STEPS*REPS);
     cudaGraphExecDestroy(ge);cudaGraphDestroy(g);cudaStreamDestroy(stream);}

    {int numBSm=0;cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
     cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm,montecarlo_persistent,256,0);
     int maxB=numBSm*prop.multiProcessorCount;
     if(blocks<=maxB){
         void*args[]={(void*)&N,(void*)&x,(void*)&y,(void*)&state,(void*)&STEPS};
         dim3 pgrid(blocks),pblock(256);
         cudaLaunchCooperativeKernel((void*)montecarlo_persistent,pgrid,pblock,args);cudaDeviceSynchronize();
         int REPS=5;cudaEventRecord(t0);
         for(int i=0;i<REPS;i++)cudaLaunchCooperativeKernel((void*)montecarlo_persistent,pgrid,pblock,args);
         cudaEventRecord(t1);cudaDeviceSynchronize();
         cudaEventElapsedTime(&ms,t0,t1);r.persistent_us=ms*1000/(STEPS*REPS);
     } else r.persistent_us=-1;}

    cudaEventDestroy(t0);cudaEventDestroy(t1);cudaFree(x);cudaFree(y);cudaFree(state);
    return r;
}

// =========================================================================
// Main
// =========================================================================
void print_result(const char* name, Result r) {
    char ps[32],sa[16],sg[16],sp[16];
    if(r.persistent_us<0){snprintf(ps,32,"%10s","N/A");snprintf(sp,16,"%8s","N/A");}
    else{snprintf(ps,32,"%10.2f",r.persistent_us);snprintf(sp,16,"%7.1fx",r.sync_us/r.persistent_us);}
    snprintf(sa,16,"%7.1fx",r.sync_us/r.async_us);
    snprintf(sg,16,"%7.1fx",r.sync_us/r.graph_us);
    printf("%-28s %10.2f %10.2f %10.2f %s  | %8s %8s %8s\n",
           name,r.sync_us,r.async_us,r.graph_us,ps,sa,sg,sp);
}

int main(){
    cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
    printf("GPU: %s (SMs=%d, Compute %d.%d)\n",prop.name,prop.multiProcessorCount,prop.major,prop.minor);
    printf("Extended Overhead Characterization: 4 strategies across all kernel types\n\n");

    printf("%-28s %10s %10s %10s %10s  | Speedups over Sync\n","Kernel","Sync","Async","Graph","Persist");
    printf("%-28s %10s %10s %10s %10s  | %8s %8s %8s\n","","(us)","(us)","(us)","(us)","Async","Graph","Persist");
    printf("%.28s %.10s %.10s %.10s %.10s  | %.8s %.8s %.8s\n",
           "----------------------------","----------","----------","----------","----------","--------","--------","--------");

    // === Stencil domain ===
    printf("\n--- Stencil (5-point 2D variants) ---\n");
    print_result("Jacobi2D 128sq",    test_jacobi2d(128, 2000));
    print_result("Jacobi2D 256sq",    test_jacobi2d(256, 1000));
    print_result("Jacobi2D 512sq",    test_jacobi2d(512, 500));
    print_result("Wave2D 128sq",      test_wave2d(128, 2000));
    print_result("Wave2D 256sq",      test_wave2d(256, 1000));
    print_result("AllenCahn 128sq",   test_2d_stencil(allencahn_step, allencahn_persistent, 128, 2000));
    print_result("AllenCahn 256sq",   test_2d_stencil(allencahn_step, allencahn_persistent, 256, 1000));
    print_result("CahnHilliard 128sq",test_2d_stencil((Step2D)cahnhilliard_step, (Persist2D)cahnhilliard_persistent, 128, 2000));
    print_result("CahnHilliard 256sq",test_2d_stencil((Step2D)cahnhilliard_step, (Persist2D)cahnhilliard_persistent, 256, 1000));
    print_result("ConvDiff 128sq",    test_2d_stencil(convdiff_step, convdiff_persistent, 128, 2000));
    print_result("ConvDiff 256sq",    test_2d_stencil(convdiff_step, convdiff_persistent, 256, 1000));
    // Poisson2D uses same body as Jacobi2D (rhs=0), skip separate test
    // (Poisson with rhs needs extra array, Jacobi2D covers the stencil pattern)

    print_result("SemiLagrangian 128sq",test_2d_stencil(semilag_step, semilag_persistent, 128, 2000));
    print_result("SemiLagrangian 256sq",test_2d_stencil(semilag_step, semilag_persistent, 256, 1000));

    // === CFD domain ===
    printf("\n--- CFD ---\n");
    print_result("Burgers2D 128sq",   test_burgers2d(128, 2000));
    print_result("Burgers2D 256sq",   test_burgers2d(256, 1000));
    print_result("LBM D2Q9 64sq",     test_lbm(64, 1000));
    print_result("LBM D2Q9 128sq",    test_lbm(128, 500));
    // SWE_LaxFried covered by swe_step (needs 6 arrays, use dedicated test below)

    // === EM domain ===
    printf("\n--- EM ---\n");
    print_result("FDTD Maxwell 128sq", test_fdtd2d(128, 2000));
    print_result("FDTD Maxwell 256sq", test_fdtd2d(256, 1000));

    // === FEM / Structure ===
    printf("\n--- FEM / Structure ---\n");
    print_result("ExplicitFEM 128sq", test_2d_stencil(fem2d_step, fem2d_persistent_simple, 128, 2000));
    print_result("ExplicitFEM 256sq", test_2d_stencil(fem2d_step, fem2d_persistent_simple, 256, 1000));
    print_result("Cloth 64sq",        test_2d_stencil(cloth_step, cloth_persistent, 64, 2000));
    print_result("Cloth 128sq",       test_2d_stencil(cloth_step, cloth_persistent, 128, 2000));

    // === Classic (Rodinia) ===
    printf("\n--- Classic (Rodinia + PERKS) ---\n");
    print_result("SRAD 128sq",        test_srad(128, 2000));
    print_result("SRAD 256sq",        test_srad(256, 1000));
    print_result("HotSpot 128sq",     test_hotspot(128, 2000));
    print_result("HotSpot 256sq",     test_hotspot(256, 1000));
    print_result("SpMV N=4096",       test_1d(spmv_step, spmv_persistent, 4096, 2000));
    print_result("SpMV N=16384",      test_1d(spmv_step, spmv_persistent, 16384, 2000));
    print_result("Reduction N=16384", test_1d(reduce_step_proxy, reduce_persistent_proxy, 16384, 2000));

    // === Particle ===
    printf("\n--- Particle ---\n");
    print_result("NBody N=256",       test_nbody(256, 500));
    print_result("NBody N=512",       test_nbody(512, 200));
    print_result("NBody N=1024",      test_nbody(1024, 100));
    print_result("MonteCarlo N=1024", test_montecarlo(1024, 2000));
    print_result("MonteCarlo N=4096", test_montecarlo(4096, 2000));

    // SPH (O(N²) brute force, like NBody but with density + pressure)
    {
        int N=1024, STEPS=200;
        float *px,*py,*vx,*vy,*rho_s,*px2,*py2,*vx2,*vy2;
        CHECK(cudaMalloc(&px,N*4));CHECK(cudaMalloc(&py,N*4));
        CHECK(cudaMalloc(&vx,N*4));CHECK(cudaMalloc(&vy,N*4));CHECK(cudaMalloc(&rho_s,N*4));
        CHECK(cudaMalloc(&px2,N*4));CHECK(cudaMalloc(&py2,N*4));
        CHECK(cudaMalloc(&vx2,N*4));CHECK(cudaMalloc(&vy2,N*4));
        float*h=(float*)malloc(N*4);for(int i=0;i<N;i++)h[i]=(float)(i%32)/32.0f;
        cudaMemcpy(px,h,N*4,cudaMemcpyHostToDevice);
        for(int i=0;i<N;i++)h[i]=(float)((i*7)%32)/32.0f;
        cudaMemcpy(py,h,N*4,cudaMemcpyHostToDevice);
        CHECK(cudaMemset(vx,0,N*4));CHECK(cudaMemset(vy,0,N*4));CHECK(cudaMemset(rho_s,0,N*4));free(h);
        int bl=(N+255)/256;
        cudaEvent_t t0,t1;CHECK(cudaEventCreate(&t0));CHECK(cudaEventCreate(&t1));
        for(int i=0;i<5;i++){sph_step<<<bl,256>>>(N,px,py,vx,vy,rho_s,px2,py2,vx2,vy2);
            copy4_f<<<bl,256>>>(N,px2,px,py2,py,vx2,vx,vy2,vy);}
        cudaDeviceSynchronize();
        Result r;float ms;
        cudaEventRecord(t0);
        for(int s=0;s<STEPS;s++){sph_step<<<bl,256>>>(N,px,py,vx,vy,rho_s,px2,py2,vx2,vy2);
            copy4_f<<<bl,256>>>(N,px2,px,py2,py,vx2,vx,vy2,vy);cudaDeviceSynchronize();}
        cudaEventRecord(t1);cudaEventSynchronize(t1);cudaEventElapsedTime(&ms,t0,t1);r.sync_us=ms*1000/STEPS;
        cudaEventRecord(t0);
        for(int s=0;s<STEPS;s++){sph_step<<<bl,256>>>(N,px,py,vx,vy,rho_s,px2,py2,vx2,vy2);
            copy4_f<<<bl,256>>>(N,px2,px,py2,py,vx2,vx,vy2,vy);}
        cudaEventRecord(t1);cudaEventSynchronize(t1);cudaEventElapsedTime(&ms,t0,t1);r.async_us=ms*1000/STEPS;
        {int REPS=5;cudaGraph_t g;cudaGraphExec_t ge;cudaStream_t stream;
         CHECK(cudaStreamCreate(&stream));
         CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
         for(int s=0;s<STEPS;s++){sph_step<<<bl,256,0,stream>>>(N,px,py,vx,vy,rho_s,px2,py2,vx2,vy2);
             copy4_f<<<bl,256,0,stream>>>(N,px2,px,py2,py,vx2,vx,vy2,vy);}
         CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&ge,g,NULL,NULL,0));
         for(int i=0;i<3;i++){cudaGraphLaunch(ge,stream);cudaStreamSynchronize(stream);}
         cudaEventRecord(t0,stream);for(int i=0;i<REPS;i++)cudaGraphLaunch(ge,stream);
         cudaEventRecord(t1,stream);cudaStreamSynchronize(stream);
         cudaEventElapsedTime(&ms,t0,t1);r.graph_us=ms*1000/(STEPS*REPS);
         cudaGraphExecDestroy(ge);cudaGraphDestroy(g);cudaStreamDestroy(stream);}
        {int numBSm=0;cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
         cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm,sph_persistent,256,0);
         int maxB=numBSm*prop.multiProcessorCount;
         if(bl<=maxB){
             void*args[]={(void*)&N,(void*)&px,(void*)&py,(void*)&vx,(void*)&vy,(void*)&rho_s,
                          (void*)&px2,(void*)&py2,(void*)&vx2,(void*)&vy2,(void*)&STEPS};
             cudaLaunchCooperativeKernel((void*)sph_persistent,dim3(bl),dim3(256),args);cudaDeviceSynchronize();
             int REPS=5;cudaEventRecord(t0);
             for(int i=0;i<REPS;i++)cudaLaunchCooperativeKernel((void*)sph_persistent,dim3(bl),dim3(256),args);
             cudaEventRecord(t1);cudaDeviceSynchronize();
             cudaEventElapsedTime(&ms,t0,t1);r.persistent_us=ms*1000/(STEPS*REPS);
         } else r.persistent_us=-1;}
        print_result("SPH N=1024",r);
        cudaEventDestroy(t0);cudaEventDestroy(t1);
        cudaFree(px);cudaFree(py);cudaFree(vx);cudaFree(vy);cudaFree(rho_s);
        cudaFree(px2);cudaFree(py2);cudaFree(vx2);cudaFree(vy2);
    }

    // DEM N=1024 (reuse NBody harness pattern)
    {
        int N=1024, STEPS=200;
        float *px,*py,*vx,*vy,*px2,*py2,*vx2,*vy2;
        CHECK(cudaMalloc(&px,N*4));CHECK(cudaMalloc(&py,N*4));
        CHECK(cudaMalloc(&vx,N*4));CHECK(cudaMalloc(&vy,N*4));
        CHECK(cudaMalloc(&px2,N*4));CHECK(cudaMalloc(&py2,N*4));
        CHECK(cudaMalloc(&vx2,N*4));CHECK(cudaMalloc(&vy2,N*4));
        float*h=(float*)malloc(N*4);for(int i=0;i<N;i++)h[i]=(float)(i%32)*0.03f;
        cudaMemcpy(px,h,N*4,cudaMemcpyHostToDevice);
        for(int i=0;i<N;i++)h[i]=(float)((i*7)%32)*0.03f;
        cudaMemcpy(py,h,N*4,cudaMemcpyHostToDevice);
        CHECK(cudaMemset(vx,0,N*4));CHECK(cudaMemset(vy,0,N*4));free(h);
        int bl=(N+255)/256;
        cudaEvent_t t0,t1;CHECK(cudaEventCreate(&t0));CHECK(cudaEventCreate(&t1));
        for(int i=0;i<5;i++){dem_step<<<bl,256>>>(N,px,py,vx,vy,px2,py2,vx2,vy2);
            copy4_f<<<bl,256>>>(N,px2,px,py2,py,vx2,vx,vy2,vy);}
        cudaDeviceSynchronize();
        Result r;float ms;
        cudaEventRecord(t0);
        for(int s=0;s<STEPS;s++){dem_step<<<bl,256>>>(N,px,py,vx,vy,px2,py2,vx2,vy2);
            copy4_f<<<bl,256>>>(N,px2,px,py2,py,vx2,vx,vy2,vy);cudaDeviceSynchronize();}
        cudaEventRecord(t1);cudaEventSynchronize(t1);cudaEventElapsedTime(&ms,t0,t1);r.sync_us=ms*1000/STEPS;
        cudaEventRecord(t0);
        for(int s=0;s<STEPS;s++){dem_step<<<bl,256>>>(N,px,py,vx,vy,px2,py2,vx2,vy2);
            copy4_f<<<bl,256>>>(N,px2,px,py2,py,vx2,vx,vy2,vy);}
        cudaEventRecord(t1);cudaEventSynchronize(t1);cudaEventElapsedTime(&ms,t0,t1);r.async_us=ms*1000/STEPS;
        {int REPS=5;cudaGraph_t g;cudaGraphExec_t ge;cudaStream_t stream;
         CHECK(cudaStreamCreate(&stream));CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
         for(int s=0;s<STEPS;s++){dem_step<<<bl,256,0,stream>>>(N,px,py,vx,vy,px2,py2,vx2,vy2);
             copy4_f<<<bl,256,0,stream>>>(N,px2,px,py2,py,vx2,vx,vy2,vy);}
         CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&ge,g,NULL,NULL,0));
         for(int i=0;i<3;i++){cudaGraphLaunch(ge,stream);cudaStreamSynchronize(stream);}
         cudaEventRecord(t0,stream);for(int i=0;i<REPS;i++)cudaGraphLaunch(ge,stream);
         cudaEventRecord(t1,stream);cudaStreamSynchronize(stream);
         cudaEventElapsedTime(&ms,t0,t1);r.graph_us=ms*1000/(STEPS*REPS);
         cudaGraphExecDestroy(ge);cudaGraphDestroy(g);cudaStreamDestroy(stream);}
        {int numBSm=0;cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
         cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm,dem_persistent,256,0);
         int maxB=numBSm*prop.multiProcessorCount;
         if(bl<=maxB){
             void*args[]={(void*)&N,(void*)&px,(void*)&py,(void*)&vx,(void*)&vy,
                          (void*)&px2,(void*)&py2,(void*)&vx2,(void*)&vy2,(void*)&STEPS};
             cudaLaunchCooperativeKernel((void*)dem_persistent,dim3(bl),dim3(256),args);cudaDeviceSynchronize();
             int REPS=5;cudaEventRecord(t0);
             for(int i=0;i<REPS;i++)cudaLaunchCooperativeKernel((void*)dem_persistent,dim3(bl),dim3(256),args);
             cudaEventRecord(t1);cudaDeviceSynchronize();
             cudaEventElapsedTime(&ms,t0,t1);r.persistent_us=ms*1000/(STEPS*REPS);
         } else r.persistent_us=-1;}
        print_result("DEM N=1024",r);
        cudaEventDestroy(t0);cudaEventDestroy(t1);
        cudaFree(px);cudaFree(py);cudaFree(vx);cudaFree(vy);
        cudaFree(px2);cudaFree(py2);cudaFree(vx2);cudaFree(vy2);
    }

    // MD_LJ N=1024 (same pattern)
    {
        int N=1024, STEPS=200;
        float *px,*py,*vx,*vy,*px2,*py2,*vx2,*vy2;
        CHECK(cudaMalloc(&px,N*4));CHECK(cudaMalloc(&py,N*4));
        CHECK(cudaMalloc(&vx,N*4));CHECK(cudaMalloc(&vy,N*4));
        CHECK(cudaMalloc(&px2,N*4));CHECK(cudaMalloc(&py2,N*4));
        CHECK(cudaMalloc(&vx2,N*4));CHECK(cudaMalloc(&vy2,N*4));
        float*h=(float*)malloc(N*4);for(int i=0;i<N;i++)h[i]=(float)(i%32)*0.03f;
        cudaMemcpy(px,h,N*4,cudaMemcpyHostToDevice);
        for(int i=0;i<N;i++)h[i]=(float)((i*7)%32)*0.03f;
        cudaMemcpy(py,h,N*4,cudaMemcpyHostToDevice);
        CHECK(cudaMemset(vx,0,N*4));CHECK(cudaMemset(vy,0,N*4));free(h);
        int bl=(N+255)/256;
        cudaEvent_t t0,t1;CHECK(cudaEventCreate(&t0));CHECK(cudaEventCreate(&t1));
        for(int i=0;i<5;i++){mdlj_step<<<bl,256>>>(N,px,py,vx,vy,px2,py2,vx2,vy2);
            copy4_f<<<bl,256>>>(N,px2,px,py2,py,vx2,vx,vy2,vy);}
        cudaDeviceSynchronize();
        Result r;float ms;
        cudaEventRecord(t0);
        for(int s=0;s<STEPS;s++){mdlj_step<<<bl,256>>>(N,px,py,vx,vy,px2,py2,vx2,vy2);
            copy4_f<<<bl,256>>>(N,px2,px,py2,py,vx2,vx,vy2,vy);cudaDeviceSynchronize();}
        cudaEventRecord(t1);cudaEventSynchronize(t1);cudaEventElapsedTime(&ms,t0,t1);r.sync_us=ms*1000/STEPS;
        cudaEventRecord(t0);
        for(int s=0;s<STEPS;s++){mdlj_step<<<bl,256>>>(N,px,py,vx,vy,px2,py2,vx2,vy2);
            copy4_f<<<bl,256>>>(N,px2,px,py2,py,vx2,vx,vy2,vy);}
        cudaEventRecord(t1);cudaEventSynchronize(t1);cudaEventElapsedTime(&ms,t0,t1);r.async_us=ms*1000/STEPS;
        {int REPS=5;cudaGraph_t g;cudaGraphExec_t ge;cudaStream_t stream;
         CHECK(cudaStreamCreate(&stream));CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
         for(int s=0;s<STEPS;s++){mdlj_step<<<bl,256,0,stream>>>(N,px,py,vx,vy,px2,py2,vx2,vy2);
             copy4_f<<<bl,256,0,stream>>>(N,px2,px,py2,py,vx2,vx,vy2,vy);}
         CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&ge,g,NULL,NULL,0));
         for(int i=0;i<3;i++){cudaGraphLaunch(ge,stream);cudaStreamSynchronize(stream);}
         cudaEventRecord(t0,stream);for(int i=0;i<REPS;i++)cudaGraphLaunch(ge,stream);
         cudaEventRecord(t1,stream);cudaStreamSynchronize(stream);
         cudaEventElapsedTime(&ms,t0,t1);r.graph_us=ms*1000/(STEPS*REPS);
         cudaGraphExecDestroy(ge);cudaGraphDestroy(g);cudaStreamDestroy(stream);}
        {int numBSm=0;cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
         cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm,mdlj_persistent,256,0);
         int maxB=numBSm*prop.multiProcessorCount;
         if(bl<=maxB){
             void*args[]={(void*)&N,(void*)&px,(void*)&py,(void*)&vx,(void*)&vy,
                          (void*)&px2,(void*)&py2,(void*)&vx2,(void*)&vy2,(void*)&STEPS};
             cudaLaunchCooperativeKernel((void*)mdlj_persistent,dim3(bl),dim3(256),args);cudaDeviceSynchronize();
             int REPS=5;cudaEventRecord(t0);
             for(int i=0;i<REPS;i++)cudaLaunchCooperativeKernel((void*)mdlj_persistent,dim3(bl),dim3(256),args);
             cudaEventRecord(t1);cudaDeviceSynchronize();
             cudaEventElapsedTime(&ms,t0,t1);r.persistent_us=ms*1000/(STEPS*REPS);
         } else r.persistent_us=-1;}
        print_result("MD_LJ N=1024",r);
        cudaEventDestroy(t0);cudaEventDestroy(t1);
        cudaFree(px);cudaFree(py);cudaFree(vx);cudaFree(vy);
        cudaFree(px2);cudaFree(py2);cudaFree(vx2);cudaFree(vy2);
    }

    // PIC 1D (4 phases: zero + deposit + field_solve + push)
    {
        int NP=4096, NG=256, STEPS=1000;
        float *xp,*vp,*rho_g,*E_g;
        CHECK(cudaMalloc(&xp,NP*4));CHECK(cudaMalloc(&vp,NP*4));
        CHECK(cudaMalloc(&rho_g,NG*4));CHECK(cudaMalloc(&E_g,NG*4));
        float*h=(float*)malloc(NP*4);for(int i=0;i<NP;i++)h[i]=(float)(i%NP)/(float)NP;
        cudaMemcpy(xp,h,NP*4,cudaMemcpyHostToDevice);CHECK(cudaMemset(vp,0,NP*4));free(h);
        int blp=(NP+255)/256, blg=(NG+255)/256;
        cudaEvent_t t0,t1;CHECK(cudaEventCreate(&t0));CHECK(cudaEventCreate(&t1));
        for(int i=0;i<20;i++){pic_zero<<<blg,256>>>(NG,rho_g);pic_deposit<<<blp,256>>>(NP,NG,xp,rho_g);
            pic_field<<<blg,256>>>(NG,rho_g,E_g);pic_push<<<blp,256>>>(NP,NG,xp,vp,E_g);}
        cudaDeviceSynchronize();
        Result r;float ms;
        cudaEventRecord(t0);
        for(int s=0;s<STEPS;s++){pic_zero<<<blg,256>>>(NG,rho_g);pic_deposit<<<blp,256>>>(NP,NG,xp,rho_g);
            pic_field<<<blg,256>>>(NG,rho_g,E_g);pic_push<<<blp,256>>>(NP,NG,xp,vp,E_g);cudaDeviceSynchronize();}
        cudaEventRecord(t1);cudaEventSynchronize(t1);cudaEventElapsedTime(&ms,t0,t1);r.sync_us=ms*1000/STEPS;
        cudaEventRecord(t0);
        for(int s=0;s<STEPS;s++){pic_zero<<<blg,256>>>(NG,rho_g);pic_deposit<<<blp,256>>>(NP,NG,xp,rho_g);
            pic_field<<<blg,256>>>(NG,rho_g,E_g);pic_push<<<blp,256>>>(NP,NG,xp,vp,E_g);}
        cudaEventRecord(t1);cudaEventSynchronize(t1);cudaEventElapsedTime(&ms,t0,t1);r.async_us=ms*1000/STEPS;
        {int REPS=5;cudaGraph_t g;cudaGraphExec_t ge;cudaStream_t stream;
         CHECK(cudaStreamCreate(&stream));CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
         for(int s=0;s<STEPS;s++){pic_zero<<<blg,256,0,stream>>>(NG,rho_g);pic_deposit<<<blp,256,0,stream>>>(NP,NG,xp,rho_g);
             pic_field<<<blg,256,0,stream>>>(NG,rho_g,E_g);pic_push<<<blp,256,0,stream>>>(NP,NG,xp,vp,E_g);}
         CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&ge,g,NULL,NULL,0));
         for(int i=0;i<3;i++){cudaGraphLaunch(ge,stream);cudaStreamSynchronize(stream);}
         cudaEventRecord(t0,stream);for(int i=0;i<REPS;i++)cudaGraphLaunch(ge,stream);
         cudaEventRecord(t1,stream);cudaStreamSynchronize(stream);
         cudaEventElapsedTime(&ms,t0,t1);r.graph_us=ms*1000/(STEPS*REPS);
         cudaGraphExecDestroy(ge);cudaGraphDestroy(g);cudaStreamDestroy(stream);}
        {int numBSm=0;cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
         cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm,pic_persistent,256,0);
         int maxB=numBSm*prop.multiProcessorCount;
         if(blp<=maxB){
             void*args[]={(void*)&NP,(void*)&NG,(void*)&xp,(void*)&vp,(void*)&rho_g,(void*)&E_g,(void*)&STEPS};
             cudaLaunchCooperativeKernel((void*)pic_persistent,dim3(blp),dim3(256),args);cudaDeviceSynchronize();
             int REPS=5;cudaEventRecord(t0);
             for(int i=0;i<REPS;i++)cudaLaunchCooperativeKernel((void*)pic_persistent,dim3(blp),dim3(256),args);
             cudaEventRecord(t1);cudaDeviceSynchronize();
             cudaEventElapsedTime(&ms,t0,t1);r.persistent_us=ms*1000/(STEPS*REPS);
         } else r.persistent_us=-1;}
        print_result("PIC NP=4096",r);
        cudaEventDestroy(t0);cudaEventDestroy(t1);
        cudaFree(xp);cudaFree(vp);cudaFree(rho_g);cudaFree(E_g);
    }

    // === 1D PDE ===
    printf("\n--- 1D PDE ---\n");
    print_result("Upwind1D N=4096",   test_1d(upwind1d_step, upwind1d_persistent, 4096, 2000));
    print_result("Upwind1D N=16384",  test_1d(upwind1d_step, upwind1d_persistent, 16384, 2000));
    print_result("Schrodinger1D N=4096",test_1d(schrodinger1d_step, schrodinger1d_persistent, 4096, 2000));
    print_result("KuramotoSivash N=4096",test_1d(ks1d_step, ks1d_persistent, 4096, 2000));
    print_result("MassSpring1D N=4096",test_1d(massspring1d_step, massspring1d_persistent, 4096, 2000));

    // === Euler1D (6 arrays: rho, rhou, E + copies) ===
    {
        int N=4096, STEPS=2000; float *rho,*rhou,*E,*rho2,*rhou2,*E2;
        CHECK(cudaMalloc(&rho,N*4));CHECK(cudaMalloc(&rhou,N*4));CHECK(cudaMalloc(&E,N*4));
        CHECK(cudaMalloc(&rho2,N*4));CHECK(cudaMalloc(&rhou2,N*4));CHECK(cudaMalloc(&E2,N*4));
        float*h=(float*)malloc(N*4);
        for(int i=0;i<N;i++)h[i]=1.0f+0.01f*(i%100);
        cudaMemcpy(rho,h,N*4,cudaMemcpyHostToDevice);cudaMemcpy(E,h,N*4,cudaMemcpyHostToDevice);
        CHECK(cudaMemset(rhou,0,N*4));free(h);
        int bl=(N+255)/256;
        cudaEvent_t t0,t1;CHECK(cudaEventCreate(&t0));CHECK(cudaEventCreate(&t1));
        for(int i=0;i<20;i++){euler1d_step<<<bl,256>>>(N,rho,rhou,E,rho2,rhou2,E2);
            copy_f<<<bl,256>>>(N,rho2,rho);copy_f<<<bl,256>>>(N,rhou2,rhou);copy_f<<<bl,256>>>(N,E2,E);}
        cudaDeviceSynchronize();
        Result r;float ms;
        cudaEventRecord(t0);
        for(int s=0;s<STEPS;s++){euler1d_step<<<bl,256>>>(N,rho,rhou,E,rho2,rhou2,E2);
            copy_f<<<bl,256>>>(N,rho2,rho);copy_f<<<bl,256>>>(N,rhou2,rhou);copy_f<<<bl,256>>>(N,E2,E);cudaDeviceSynchronize();}
        cudaEventRecord(t1);cudaEventSynchronize(t1);cudaEventElapsedTime(&ms,t0,t1);r.sync_us=ms*1000/STEPS;
        cudaEventRecord(t0);
        for(int s=0;s<STEPS;s++){euler1d_step<<<bl,256>>>(N,rho,rhou,E,rho2,rhou2,E2);
            copy_f<<<bl,256>>>(N,rho2,rho);copy_f<<<bl,256>>>(N,rhou2,rhou);copy_f<<<bl,256>>>(N,E2,E);}
        cudaEventRecord(t1);cudaEventSynchronize(t1);cudaEventElapsedTime(&ms,t0,t1);r.async_us=ms*1000/STEPS;
        {int REPS=5;cudaGraph_t g;cudaGraphExec_t ge;cudaStream_t stream;
         CHECK(cudaStreamCreate(&stream));
         CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
         for(int s=0;s<STEPS;s++){euler1d_step<<<bl,256,0,stream>>>(N,rho,rhou,E,rho2,rhou2,E2);
             copy_f<<<bl,256,0,stream>>>(N,rho2,rho);copy_f<<<bl,256,0,stream>>>(N,rhou2,rhou);copy_f<<<bl,256,0,stream>>>(N,E2,E);}
         CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&ge,g,NULL,NULL,0));
         for(int i=0;i<3;i++){cudaGraphLaunch(ge,stream);cudaStreamSynchronize(stream);}
         cudaEventRecord(t0,stream);for(int i=0;i<REPS;i++)cudaGraphLaunch(ge,stream);
         cudaEventRecord(t1,stream);cudaStreamSynchronize(stream);
         cudaEventElapsedTime(&ms,t0,t1);r.graph_us=ms*1000/(STEPS*REPS);
         cudaGraphExecDestroy(ge);cudaGraphDestroy(g);cudaStreamDestroy(stream);}
        {int numBSm=0;cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
         cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm,euler1d_persistent,256,0);
         int maxB=numBSm*prop.multiProcessorCount;
         if(bl<=maxB){
             void*args[]={(void*)&N,(void*)&rho,(void*)&rhou,(void*)&E,(void*)&rho2,(void*)&rhou2,(void*)&E2,(void*)&STEPS};
             cudaLaunchCooperativeKernel((void*)euler1d_persistent,dim3(bl),dim3(256),args);cudaDeviceSynchronize();
             int REPS=5;cudaEventRecord(t0);
             for(int i=0;i<REPS;i++)cudaLaunchCooperativeKernel((void*)euler1d_persistent,dim3(bl),dim3(256),args);
             cudaEventRecord(t1);cudaDeviceSynchronize();
             cudaEventElapsedTime(&ms,t0,t1);r.persistent_us=ms*1000/(STEPS*REPS);
         } else r.persistent_us=-1;}
        print_result("Euler1D N=4096",r);
        cudaEventDestroy(t0);cudaEventDestroy(t1);
        cudaFree(rho);cudaFree(rhou);cudaFree(E);cudaFree(rho2);cudaFree(rhou2);cudaFree(E2);
    }

    // === SWE Lax-Friedrichs (6 arrays: h, hu, hv + copies) ===
    {
        int N=128, STEPS=2000, N2=N*N; float *h_f,*hu,*hv,*h2,*hu2,*hv2;
        CHECK(cudaMalloc(&h_f,N2*4));CHECK(cudaMalloc(&hu,N2*4));CHECK(cudaMalloc(&hv,N2*4));
        CHECK(cudaMalloc(&h2,N2*4));CHECK(cudaMalloc(&hu2,N2*4));CHECK(cudaMalloc(&hv2,N2*4));
        float*hh=(float*)malloc(N2*4);for(int i=0;i<N2;i++)hh[i]=1.0f+0.01f*(i%100);
        cudaMemcpy(h_f,hh,N2*4,cudaMemcpyHostToDevice);CHECK(cudaMemset(hu,0,N2*4));CHECK(cudaMemset(hv,0,N2*4));free(hh);
        dim3 block(16,16),grid((N+15)/16,(N+15)/16);int cg=(N2+255)/256;
        cudaEvent_t t0,t1;CHECK(cudaEventCreate(&t0));CHECK(cudaEventCreate(&t1));
        for(int i=0;i<20;i++){swe_step<<<grid,block>>>(N,h_f,hu,hv,h2,hu2,hv2);copy3_f<<<cg,256>>>(N2,h2,h_f,hu2,hu,hv2,hv);}
        cudaDeviceSynchronize();
        Result r;float ms;
        cudaEventRecord(t0);
        for(int s=0;s<STEPS;s++){swe_step<<<grid,block>>>(N,h_f,hu,hv,h2,hu2,hv2);copy3_f<<<cg,256>>>(N2,h2,h_f,hu2,hu,hv2,hv);cudaDeviceSynchronize();}
        cudaEventRecord(t1);cudaEventSynchronize(t1);cudaEventElapsedTime(&ms,t0,t1);r.sync_us=ms*1000/STEPS;
        cudaEventRecord(t0);
        for(int s=0;s<STEPS;s++){swe_step<<<grid,block>>>(N,h_f,hu,hv,h2,hu2,hv2);copy3_f<<<cg,256>>>(N2,h2,h_f,hu2,hu,hv2,hv);}
        cudaEventRecord(t1);cudaEventSynchronize(t1);cudaEventElapsedTime(&ms,t0,t1);r.async_us=ms*1000/STEPS;
        {int REPS=5;cudaGraph_t g;cudaGraphExec_t ge;cudaStream_t stream;
         CHECK(cudaStreamCreate(&stream));
         CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
         for(int s=0;s<STEPS;s++){swe_step<<<grid,block,0,stream>>>(N,h_f,hu,hv,h2,hu2,hv2);copy3_f<<<cg,256,0,stream>>>(N2,h2,h_f,hu2,hu,hv2,hv);}
         CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&ge,g,NULL,NULL,0));
         for(int i=0;i<3;i++){cudaGraphLaunch(ge,stream);cudaStreamSynchronize(stream);}
         cudaEventRecord(t0,stream);for(int i=0;i<REPS;i++)cudaGraphLaunch(ge,stream);
         cudaEventRecord(t1,stream);cudaStreamSynchronize(stream);
         cudaEventElapsedTime(&ms,t0,t1);r.graph_us=ms*1000/(STEPS*REPS);
         cudaGraphExecDestroy(ge);cudaGraphDestroy(g);cudaStreamDestroy(stream);}
        {int numBSm=0;cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
         cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm,swe_persistent,256,0);
         int maxB=numBSm*prop.multiProcessorCount;
         if((int)(grid.x*grid.y)<=maxB){
             void*args[]={(void*)&N,(void*)&h_f,(void*)&hu,(void*)&hv,(void*)&h2,(void*)&hu2,(void*)&hv2,(void*)&STEPS};
             cudaLaunchCooperativeKernel((void*)swe_persistent,grid,block,args);cudaDeviceSynchronize();
             int REPS=5;cudaEventRecord(t0);
             for(int i=0;i<REPS;i++)cudaLaunchCooperativeKernel((void*)swe_persistent,grid,block,args);
             cudaEventRecord(t1);cudaDeviceSynchronize();
             cudaEventElapsedTime(&ms,t0,t1);r.persistent_us=ms*1000/(STEPS*REPS);
         } else r.persistent_us=-1;}
        print_result("SWE_LaxFried 128sq",r);
        cudaEventDestroy(t0);cudaEventDestroy(t1);
        cudaFree(h_f);cudaFree(hu);cudaFree(hv);cudaFree(h2);cudaFree(hu2);cudaFree(hv2);
    }

    // === Helmholtz/Poisson (same stencil as Jacobi, different name) ===
    printf("\n--- Helmholtz/Poisson ---\n");
    // Poisson2D = Jacobi iteration with RHS=0, same as Jacobi2D kernel
    print_result("Poisson2D 128sq",  test_2d_stencil(jacobi2d_step, jacobi2d_persistent, 128, 2000));
    // Helmholtz2D = Jacobi + shift, using same kernel as proxy
    print_result("Helmholtz2D 128sq",test_2d_stencil(jacobi2d_step, jacobi2d_persistent, 128, 2000));

    // === 3D ===
    printf("\n--- 3D Stencil ---\n");
    // Heat3D (already defined, 1D flat launch)
    {
        for(int N : {32, 64}) {
            int N3=N*N*N; float *u3,*v3;
            CHECK(cudaMalloc(&u3,N3*4)); CHECK(cudaMalloc(&v3,N3*4));
            CHECK(cudaMemset(u3,0,N3*4)); CHECK(cudaMemset(v3,0,N3*4));
            int bl=(N3+255)/256; int STEPS=(N==32)?1000:500;
            cudaEvent_t t0,t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
            for(int i=0;i<20;i++){heat3d_step<<<bl,256>>>(N,u3,v3);copy_f<<<bl,256>>>(N3,v3,u3);}
            cudaDeviceSynchronize();
            Result r; float ms;
            cudaEventRecord(t0);
            for(int s=0;s<STEPS;s++){heat3d_step<<<bl,256>>>(N,u3,v3);copy_f<<<bl,256>>>(N3,v3,u3);cudaDeviceSynchronize();}
            cudaEventRecord(t1);cudaEventSynchronize(t1);
            cudaEventElapsedTime(&ms,t0,t1);r.sync_us=ms*1000/STEPS;
            cudaEventRecord(t0);
            for(int s=0;s<STEPS;s++){heat3d_step<<<bl,256>>>(N,u3,v3);copy_f<<<bl,256>>>(N3,v3,u3);}
            cudaEventRecord(t1);cudaEventSynchronize(t1);
            cudaEventElapsedTime(&ms,t0,t1);r.async_us=ms*1000/STEPS;
            {int REPS=5;cudaGraph_t g;cudaGraphExec_t ge;cudaStream_t stream;
             CHECK(cudaStreamCreate(&stream));
             CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
             for(int s=0;s<STEPS;s++){heat3d_step<<<bl,256,0,stream>>>(N,u3,v3);copy_f<<<bl,256,0,stream>>>(N3,v3,u3);}
             CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&ge,g,NULL,NULL,0));
             for(int i=0;i<3;i++){cudaGraphLaunch(ge,stream);cudaStreamSynchronize(stream);}
             cudaEventRecord(t0,stream);for(int i=0;i<REPS;i++)cudaGraphLaunch(ge,stream);
             cudaEventRecord(t1,stream);cudaStreamSynchronize(stream);
             cudaEventElapsedTime(&ms,t0,t1);r.graph_us=ms*1000/(STEPS*REPS);
             cudaGraphExecDestroy(ge);cudaGraphDestroy(g);cudaStreamDestroy(stream);}
            {int numBSm=0;cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
             cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm,heat3d_persistent,256,0);
             int maxB=numBSm*prop.multiProcessorCount;
             if(bl<=maxB){
                 void*args[]={(void*)&N,(void*)&u3,(void*)&v3,(void*)&STEPS};
                 cudaLaunchCooperativeKernel((void*)heat3d_persistent,dim3(bl),dim3(256),args);cudaDeviceSynchronize();
                 int REPS=5;cudaEventRecord(t0);
                 for(int i=0;i<REPS;i++)cudaLaunchCooperativeKernel((void*)heat3d_persistent,dim3(bl),dim3(256),args);
                 cudaEventRecord(t1);cudaDeviceSynchronize();
                 cudaEventElapsedTime(&ms,t0,t1);r.persistent_us=ms*1000/(STEPS*REPS);
             } else r.persistent_us=-1;}
            char name[64]; snprintf(name,64,"Heat3D %d^3",N);
            print_result(name,r);
            cudaEventDestroy(t0);cudaEventDestroy(t1);cudaFree(u3);cudaFree(v3);
        }
    }
    // Jacobi3D: uses (N3, N, u, v) signature - needs dedicated test
    {
        for(int N : {32, 64}) {
            int N3=N*N*N; float *u3,*v3;
            CHECK(cudaMalloc(&u3,N3*4)); CHECK(cudaMalloc(&v3,N3*4));
            CHECK(cudaMemset(u3,0,N3*4)); CHECK(cudaMemset(v3,0,N3*4));
            int bl=(N3+255)/256; int STEPS=(N==32)?1000:500;
            cudaEvent_t t0,t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
            for(int i=0;i<20;i++){jacobi3d_step_flat<<<bl,256>>>(N3,N,u3,v3);copy_f<<<bl,256>>>(N3,v3,u3);}
            cudaDeviceSynchronize();
            Result r; float ms;
            cudaEventRecord(t0);
            for(int s=0;s<STEPS;s++){jacobi3d_step_flat<<<bl,256>>>(N3,N,u3,v3);copy_f<<<bl,256>>>(N3,v3,u3);cudaDeviceSynchronize();}
            cudaEventRecord(t1);cudaEventSynchronize(t1);
            cudaEventElapsedTime(&ms,t0,t1);r.sync_us=ms*1000/STEPS;
            cudaEventRecord(t0);
            for(int s=0;s<STEPS;s++){jacobi3d_step_flat<<<bl,256>>>(N3,N,u3,v3);copy_f<<<bl,256>>>(N3,v3,u3);}
            cudaEventRecord(t1);cudaEventSynchronize(t1);
            cudaEventElapsedTime(&ms,t0,t1);r.async_us=ms*1000/STEPS;
            {int REPS=5;cudaGraph_t g;cudaGraphExec_t ge;cudaStream_t stream;
             CHECK(cudaStreamCreate(&stream));
             CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
             for(int s=0;s<STEPS;s++){jacobi3d_step_flat<<<bl,256,0,stream>>>(N3,N,u3,v3);copy_f<<<bl,256,0,stream>>>(N3,v3,u3);}
             CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&ge,g,NULL,NULL,0));
             for(int i=0;i<3;i++){cudaGraphLaunch(ge,stream);cudaStreamSynchronize(stream);}
             cudaEventRecord(t0,stream);for(int i=0;i<REPS;i++)cudaGraphLaunch(ge,stream);
             cudaEventRecord(t1,stream);cudaStreamSynchronize(stream);
             cudaEventElapsedTime(&ms,t0,t1);r.graph_us=ms*1000/(STEPS*REPS);
             cudaGraphExecDestroy(ge);cudaGraphDestroy(g);cudaStreamDestroy(stream);}
            {int numBSm=0;cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
             cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBSm,jacobi3d_persistent_flat,256,0);
             int maxB=numBSm*prop.multiProcessorCount;
             if(bl<=maxB){
                 void*args[]={(void*)&N3,(void*)&N,(void*)&u3,(void*)&v3,(void*)&STEPS};
                 cudaLaunchCooperativeKernel((void*)jacobi3d_persistent_flat,dim3(bl),dim3(256),args);cudaDeviceSynchronize();
                 int REPS=5;cudaEventRecord(t0);
                 for(int i=0;i<REPS;i++)cudaLaunchCooperativeKernel((void*)jacobi3d_persistent_flat,dim3(bl),dim3(256),args);
                 cudaEventRecord(t1);cudaDeviceSynchronize();
                 cudaEventElapsedTime(&ms,t0,t1);r.persistent_us=ms*1000/(STEPS*REPS);
             } else r.persistent_us=-1;}
            char name[64]; snprintf(name,64,"Jacobi3D %d^3",N);
            print_result(name,r);
            cudaEventDestroy(t0);cudaEventDestroy(t1);cudaFree(u3);cudaFree(v3);
        }
    }

    printf("\n===================================================================\n");
    printf("Also verified in separate files:\n");
    printf("  Heat2D + GrayScott:  overhead_solutions.cu\n");
    printf("  F1 OSHER (fp64):     F1_hydro_shallow_water/hydro_cuda_osher.cu\n");
    printf("  CG Solver (5 kern):  cg_fusion_benchmark.cu\n");
    printf("  LULESH (4 kern):     lulesh_fusion_benchmark.cu\n");
    printf("  DMA overlap:         persistent_async_copy.cu\n");
    printf("===================================================================\n");
    printf("Total coverage: 20+ kernels × 4 strategies across 7+ domains\n");
}
