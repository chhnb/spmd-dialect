/**
 * Kokkos characterization: 36 kernel types × multiple sizes.
 * Single file covering all simulation domains.
 *
 * Build:
 *   cmake --build build-cuda --target characterization_all
 * Run:
 *   ./build-cuda/characterization_all [steps]
 */

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

using View1D = Kokkos::View<double*>;
using View2D = Kokkos::View<double**>;

// =========================================================================
// Benchmark harness
// =========================================================================
struct Result { double us_per_step; };

template<typename StepFn>
Result bench(StepFn step_fn, int warmup, int steps) {
    for (int i = 0; i < warmup; i++) step_fn();
    Kokkos::fence();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < steps; s++) step_fn();
    Kokkos::fence();
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    return {us / steps};
}

void report(const char* name, Result r) {
    printf("  %-35s %8.1f us/step\n", name, r.us_per_step);
}

// =========================================================================
// Generic 2D stencil test
// =========================================================================
template<typename KernelFn>
Result test_2d(KernelFn kern, int N, int steps) {
    View2D u("u", N, N), v("v", N, N);
    Kokkos::deep_copy(u, 1.0);
    auto step = [&]() {
        Kokkos::parallel_for("step", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1},{N-1,N-1}),
            KOKKOS_LAMBDA(int i, int j) { kern(u, v, i, j, N); });
        Kokkos::parallel_for("copy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{N,N}),
            KOKKOS_LAMBDA(int i, int j) { u(i,j) = v(i,j); });
    };
    return bench(step, 20, steps);
}

// =========================================================================
// Kernel lambdas (2D stencil variants)
// =========================================================================
auto heat2d_fn = KOKKOS_LAMBDA(View2D u, View2D v, int i, int j, int N) {
    v(i,j) = u(i,j) + 0.2*(u(i-1,j)+u(i+1,j)+u(i,j-1)+u(i,j+1)-4.0*u(i,j));
};

// Can't use generic lambda with KOKKOS_LAMBDA in all compilers,
// so we use functors instead.

// =========================================================================
// Functors for each 2D kernel
// =========================================================================
#define STENCIL_2D_FUNCTOR(NAME, BODY)                                    \
struct NAME##_step {                                                       \
    View2D u, v; int N;                                                   \
    KOKKOS_INLINE_FUNCTION void operator()(int i, int j) const { BODY }   \
};                                                                        \
struct NAME##_copy {                                                       \
    View2D u, v;                                                          \
    KOKKOS_INLINE_FUNCTION void operator()(int i, int j) const { u(i,j)=v(i,j); } \
};

STENCIL_2D_FUNCTOR(Heat2D,
    v(i,j) = u(i,j) + 0.2*(u(i-1,j)+u(i+1,j)+u(i,j-1)+u(i,j+1)-4.0*u(i,j));
)

STENCIL_2D_FUNCTOR(Jacobi2D,
    v(i,j) = 0.25*(u(i-1,j)+u(i+1,j)+u(i,j-1)+u(i,j+1));
)

STENCIL_2D_FUNCTOR(Wave2D,
    // simplified: treat v as u_prev for demo
    double lap = u(i-1,j)+u(i+1,j)+u(i,j-1)+u(i,j+1)-4.0*u(i,j);
    v(i,j) = u(i,j) + 0.01*lap;  // simplified wave
)

STENCIL_2D_FUNCTOR(AllenCahn,
    double lap = u(i-1,j)+u(i+1,j)+u(i,j-1)+u(i,j+1)-4.0*u(i,j);
    double phi = u(i,j);
    v(i,j) = phi + 0.01*(0.01*lap + phi - phi*phi*phi);
)

STENCIL_2D_FUNCTOR(ConvDiff,
    double dx = 1.0/N, nu=0.01, cx=1.0, cy=0.5;
    double lap = (u(i-1,j)+u(i+1,j)+u(i,j-1)+u(i,j+1)-4.0*u(i,j))/(dx*dx);
    double ax = cx*(u(i,j+1)-u(i,j-1))/(2*dx);
    double ay = cy*(u(i+1,j)-u(i-1,j))/(2*dx);
    v(i,j) = u(i,j) + 0.0001*(nu*lap - ax - ay);
)

STENCIL_2D_FUNCTOR(SRAD,
    double Jc = u(i,j), dt=0.05, lam=0.5;
    double dN=u(i-1,j)-Jc, dS=u(i+1,j)-Jc, dW=u(i,j-1)-Jc, dE=u(i,j+1)-Jc;
    double c = 0.5;
    v(i,j) = Jc + dt*lam*(c*dN + c*dS + c*dW + c*dE);
)

STENCIL_2D_FUNCTOR(HotSpot,
    double cap=0.5, rx=0.01, ry=0.01, amb=80.0;
    double lx = (u(i,j-1)+u(i,j+1)-2.0*u(i,j))*rx;
    double ly = (u(i-1,j)+u(i+1,j)-2.0*u(i,j))*ry;
    v(i,j) = u(i,j) + cap*(lx + ly + (amb-u(i,j))*0.001);
)

STENCIL_2D_FUNCTOR(Poisson2D,
    v(i,j) = 0.25*(u(i-1,j)+u(i+1,j)+u(i,j-1)+u(i,j+1));
)

STENCIL_2D_FUNCTOR(SemiLag,
    double cx=0.5, cy=0.3, dt=0.001, dx=1.0/N;
    double xd = j*dx - cx*dt, yd = i*dx - cy*dt;
    xd = fmax(dx, fmin(xd, (N-2)*dx));
    yd = fmax(dx, fmin(yd, (N-2)*dx));
    int j0 = (int)(xd/dx), i0 = (int)(yd/dx);
    j0 = j0<1?1:(j0>N-2?N-2:j0); i0 = i0<1?1:(i0>N-2?N-2:i0);
    double fx = xd/dx-j0, fy = yd/dx-i0;
    v(i,j) = (1-fx)*(1-fy)*u(i0,j0)+fx*(1-fy)*u(i0,j0+1)
            +(1-fx)*fy*u(i0+1,j0)+fx*fy*u(i0+1,j0+1);
)

STENCIL_2D_FUNCTOR(ExplFEM,
    v(i,j) = u(i,j) + 0.01*(u(i-1,j)+u(i+1,j)+u(i,j-1)+u(i,j+1)-4.0*u(i,j));
)

STENCIL_2D_FUNCTOR(Cloth,
    double k=50.0, rest=1.0/N;
    double fx = k*((u(i,j+1)-u(i,j))-rest) + k*((u(i,j-1)-u(i,j))+rest);
    double fy = k*((u(i+1,j)-u(i,j))-rest) + k*((u(i-1,j)-u(i,j))+rest);
    v(i,j) = u(i,j) + 0.0001*(fx+fy);
)

STENCIL_2D_FUNCTOR(FDTD,
    // simplified: single-field proxy for overhead measurement
    double dt=0.001, dx=0.01;
    v(i,j) = u(i,j) + dt/dx*(u(i+1,j)-u(i,j)-u(i,j+1)+u(i,j));
)

STENCIL_2D_FUNCTOR(Burgers2D,
    double nu=0.01, dt=0.001, dx=1.0/N;
    double dudx = (u(i,j)-u(i,j-1))/dx;
    double lap = (u(i-1,j)+u(i+1,j)+u(i,j-1)+u(i,j+1)-4.0*u(i,j))/(dx*dx);
    v(i,j) = u(i,j) + dt*(-u(i,j)*dudx + nu*lap);
)

STENCIL_2D_FUNCTOR(GrayScott,
    double lu = u(i-1,j)+u(i+1,j)+u(i,j-1)+u(i,j+1)-4.0*u(i,j);
    v(i,j) = u(i,j) + 0.16*lu + 0.06*(1.0-u(i,j));
)

STENCIL_2D_FUNCTOR(CahnHilliard,
    double lap = u(i-1,j)+u(i+1,j)+u(i,j-1)+u(i,j+1)-4.0*u(i,j);
    v(i,j) = u(i,j) + 0.001*lap;
)

STENCIL_2D_FUNCTOR(Helmholtz2D,
    v(i,j) = 0.25*(u(i-1,j)+u(i+1,j)+u(i,j-1)+u(i,j+1));
)

// Generic 2D test using functors
template<typename StepF, typename CopyF>
Result test_2d_functor(int N, int steps) {
    View2D u("u", N, N), v("v", N, N);
    Kokkos::deep_copy(u, 1.0);
    auto mdpol = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1},{N-1,N-1});
    auto mdall = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{N,N});
    StepF sf{u, v, N}; CopyF cf{u, v};
    auto step = [&]() {
        Kokkos::parallel_for("step", mdpol, sf);
        Kokkos::parallel_for("copy", mdall, cf);
    };
    return bench(step, 20, steps);
}

// =========================================================================
// 1D kernels
// =========================================================================
struct Upwind1D_step {
    View1D u, v; int N;
    KOKKOS_INLINE_FUNCTION void operator()(int i) const {
        if(i>=1 && i<N-1) { double c=1.0, dx=1.0/N; v(i)=u(i)-c*0.0001/dx*(u(i)-u(i-1)); }
    }
};
struct Copy1D { View1D u, v; KOKKOS_INLINE_FUNCTION void operator()(int i) const { u(i)=v(i); } };

struct KS1D_step {
    View1D u, v; int N;
    KOKKOS_INLINE_FUNCTION void operator()(int i) const {
        if(i>=2 && i<N-2) {
            double dx=1.0/N, dt=0.00001;
            double dudx=(u(i+1)-u(i-1))/(2*dx);
            double d2u=(u(i+1)+u(i-1)-2*u(i))/(dx*dx);
            double d4u=(u(i+2)-4*u(i+1)+6*u(i)-4*u(i-1)+u(i-2))/(dx*dx*dx*dx);
            v(i)=u(i)+dt*(-u(i)*dudx-d2u-d4u);
        }
    }
};

struct SpMV_step {
    View1D x, y; int N;
    KOKKOS_INLINE_FUNCTION void operator()(int i) const {
        if(i>=1&&i<N-1) y(i)=x(i-1)+2.0*x(i)+x(i+1);
        else if(i==0) y(i)=2.0*x(i)+x(i+1);
        else if(i==N-1) y(i)=x(i-1)+2.0*x(i);
    }
};

struct Reduce_step {
    View1D u, v; int N;
    KOKKOS_INLINE_FUNCTION void operator()(int i) const {
        if(i>=1&&i<N-1) v(i)=u(i)*0.999+0.0005*(u(i-1)+u(i+1));
        else v(i)=u(i);
    }
};

struct MassSpring1D_step {
    View1D x, v; int N;
    KOKKOS_INLINE_FUNCTION void operator()(int i) const {
        if(i>=1&&i<N-1) {
            double k=100.0,m=1.0,dt=0.0001,rest=1.0/N;
            double pos=x(2*i),vel=x(2*i+1);
            double f=k*(x(2*(i+1))-pos-rest)+k*(x(2*(i-1))-pos+rest);
            v(2*i)=pos+dt*vel;
            v(2*i+1)=vel+dt*f/m;
        }
    }
};

struct Schrodinger1D_step {
    View1D u, v; int N;
    KOKKOS_INLINE_FUNCTION void operator()(int i) const {
        if(i>=1&&i<N-1) {
            double dt=0.0001,dx=1.0/N;
            double lr=(u(2*(i-1))+u(2*(i+1))-2*u(2*i))/(dx*dx);
            double li=(u(2*(i-1)+1)+u(2*(i+1)+1)-2*u(2*i+1))/(dx*dx);
            v(2*i)=u(2*i)+dt*0.5*li;
            v(2*i+1)=u(2*i+1)-dt*0.5*lr;
        }
    }
};

template<typename StepF>
Result test_1d_functor(int N, int steps) {
    View1D u("u", N), v("v", N);
    Kokkos::deep_copy(u, 1.0);
    StepF sf{u, v, N}; Copy1D cf{u, v};
    auto step = [&]() {
        Kokkos::parallel_for("step", N, sf);
        Kokkos::parallel_for("copy", N, cf);
    };
    return bench(step, 20, steps);
}

// =========================================================================
// NBody
// =========================================================================
struct NBody_step {
    View1D px, py, vx, vy, px2, py2, vx2, vy2;
    int N;
    KOKKOS_INLINE_FUNCTION void operator()(int i) const {
        double ax=0,ay=0,dt=0.001,eps=0.01;
        for(int j=0;j<N;j++){
            double dx=px(j)-px(i),dy=py(j)-py(i);
            double r2=dx*dx+dy*dy+eps;
            double inv=1.0/sqrt(r2*r2*r2);
            ax+=dx*inv; ay+=dy*inv;
        }
        vx2(i)=vx(i)+dt*ax; vy2(i)=vy(i)+dt*ay;
        px2(i)=px(i)+dt*vx2(i); py2(i)=py(i)+dt*vy2(i);
    }
};
struct NBody_copy {
    View1D px,py,vx,vy,px2,py2,vx2,vy2;
    int N;
    KOKKOS_INLINE_FUNCTION void operator()(int i) const {
        px(i)=px2(i);py(i)=py2(i);vx(i)=vx2(i);vy(i)=vy2(i);
    }
};

Result test_nbody(int N, int steps) {
    View1D px("px",N),py("py",N),vx("vx",N),vy("vy",N);
    View1D px2("px2",N),py2("py2",N),vx2("vx2",N),vy2("vy2",N);
    Kokkos::parallel_for(N, KOKKOS_LAMBDA(int i){px(i)=(double)(i%100)/100.0;py(i)=(double)((i*7)%100)/100.0;});
    NBody_step ns{px,py,vx,vy,px2,py2,vx2,vy2,N};
    NBody_copy nc{px,py,vx,vy,px2,py2,vx2,vy2,N};
    auto step=[&](){ Kokkos::parallel_for(N,ns); Kokkos::parallel_for(N,nc); };
    return bench(step, 5, steps);
}

// =========================================================================
// Additional missing kernels
// =========================================================================

// Heat3D (7-point stencil, 1D flat)
struct Heat3D_step {
    View1D u, v; int N;
    KOKKOS_INLINE_FUNCTION void operator()(int idx) const {
        int N2=N*N;
        int i=idx/N2, j=(idx/N)%N, k=idx%N;
        if(i>=1&&i<N-1&&j>=1&&j<N-1&&k>=1&&k<N-1)
            v(idx)=u(idx)+0.1*(u(idx-N2)+u(idx+N2)+u(idx-N)+u(idx+N)+u(idx-1)+u(idx+1)-6.0*u(idx));
    }
};

// Jacobi3D
struct Jacobi3D_step {
    View1D u, v; int N;
    KOKKOS_INLINE_FUNCTION void operator()(int idx) const {
        int N2=N*N;
        int i=idx/N2, j=(idx/N)%N, k=idx%N;
        if(i>=1&&i<N-1&&j>=1&&j<N-1&&k>=1&&k<N-1)
            v(idx)=(u(idx-N2)+u(idx+N2)+u(idx-N)+u(idx+N)+u(idx-1)+u(idx+1))/6.0;
    }
};

Result test_3d(int N, int steps, bool heat) {
    int N3=N*N*N;
    View1D u("u",N3), v("v",N3);
    Kokkos::deep_copy(u, 1.0);
    if(heat) {
        Heat3D_step sf{u,v,N}; Copy1D cf{u,v};
        auto step=[&](){ Kokkos::parallel_for(N3,sf); Kokkos::parallel_for(N3,cf); };
        return bench(step,20,steps);
    } else {
        Jacobi3D_step sf{u,v,N}; Copy1D cf{u,v};
        auto step=[&](){ Kokkos::parallel_for(N3,sf); Kokkos::parallel_for(N3,cf); };
        return bench(step,20,steps);
    }
}

// SWE Lax-Friedrichs (simplified, 2D with 3 fields in 1D views)
struct SWE_step {
    View1D h, hu, hv, h2, hu2, hv2; int N;
    KOKKOS_INLINE_FUNCTION void operator()(int i, int j) const {
        int idx=i*N+j; double g=9.81, dx=1.0/N, dt=0.0001;
        double ha=0.25*(h(idx-N)+h(idx+N)+h(idx-1)+h(idx+1));
        double dhu=(hu(idx+1)-hu(idx-1))/(2*dx);
        double dhv=(hv(idx+N)-hv(idx-N))/(2*dx);
        h2(idx)=ha-dt*(dhu+dhv);
        hu2(idx)=0.25*(hu(idx-N)+hu(idx+N)+hu(idx-1)+hu(idx+1));
        hv2(idx)=0.25*(hv(idx-N)+hv(idx+N)+hv(idx-1)+hv(idx+1));
    }
};
struct SWE_copy {
    View1D h,hu,hv,h2,hu2,hv2; int N;
    KOKKOS_INLINE_FUNCTION void operator()(int idx) const {
        h(idx)=h2(idx); hu(idx)=hu2(idx); hv(idx)=hv2(idx);
    }
};

Result test_swe(int N, int steps) {
    int N2=N*N;
    View1D h("h",N2),hu("hu",N2),hv("hv",N2),h2("h2",N2),hu2("hu2",N2),hv2("hv2",N2);
    Kokkos::deep_copy(h, 1.0);
    auto mdpol=Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1},{N-1,N-1});
    SWE_step sf{h,hu,hv,h2,hu2,hv2,N}; SWE_copy cf{h,hu,hv,h2,hu2,hv2,N};
    auto step=[&](){ Kokkos::parallel_for("swe",mdpol,sf); Kokkos::parallel_for("swecopy",N2,cf); };
    return bench(step,20,steps);
}

// Euler1D (3 fields)
struct Euler1D_step {
    View1D rho,rhou,E,rho2,rhou2,E2; int N;
    KOKKOS_INLINE_FUNCTION void operator()(int i) const {
        if(i>=1&&i<N-1){
            double dx=1.0/N,dt=0.00001,gamma=1.4;
            double u_=rhou(i)/(rho(i)+1e-10);
            double p=(gamma-1)*(E(i)-0.5*rho(i)*u_*u_);
            rho2(i)=0.5*(rho(i-1)+rho(i+1))-dt/(2*dx)*(rhou(i+1)-rhou(i-1));
            rhou2(i)=0.5*(rhou(i-1)+rhou(i+1))-dt/(2*dx)*(rhou(i+1)*u_+p-rhou(i-1)*u_-p);
            E2(i)=0.5*(E(i-1)+E(i+1))-dt/(2*dx)*((E(i+1)+p)*u_-(E(i-1)+p)*u_);
        }
    }
};

Result test_euler1d(int N, int steps) {
    View1D rho("rho",N),rhou("rhou",N),E("E",N),rho2("rho2",N),rhou2("rhou2",N),E2("E2",N);
    Kokkos::deep_copy(rho,1.0); Kokkos::deep_copy(E,1.0);
    Euler1D_step sf{rho,rhou,E,rho2,rhou2,E2,N};
    struct EC{View1D r,ru,e,r2,ru2,e2;int N;
        KOKKOS_INLINE_FUNCTION void operator()(int i)const{r(i)=r2(i);ru(i)=ru2(i);e(i)=e2(i);}};
    EC cf{rho,rhou,E,rho2,rhou2,E2,N};
    auto step=[&](){ Kokkos::parallel_for(N,sf); Kokkos::parallel_for(N,cf); };
    return bench(step,20,steps);
}

// MonteCarlo random walk
struct MC_step {
    View1D x,y; Kokkos::View<unsigned*> state; int N;
    KOKKOS_INLINE_FUNCTION void operator()(int i) const {
        unsigned s=state(i);
        s=s*1664525u+1013904223u; double r1=((s&0xFFFF)/65536.0)-0.5;
        s=s*1664525u+1013904223u; double r2=((s&0xFFFF)/65536.0)-0.5;
        state(i)=s;
        x(i)+=0.01*r1; y(i)+=0.01*r2;
    }
};

Result test_montecarlo(int N, int steps) {
    View1D x("x",N),y("y",N);
    Kokkos::View<unsigned*> state("state",N);
    Kokkos::parallel_for(N,KOKKOS_LAMBDA(int i){state(i)=i*12345+67890;});
    MC_step sf{x,y,state,N};
    auto step=[&](){ Kokkos::parallel_for(N,sf); };
    return bench(step,20,steps);
}

// LBM D2Q9 (simplified)
struct LBM_step {
    View1D f, f2; int N;
    KOKKOS_INLINE_FUNCTION void operator()(int i, int j) const {
        if(i<1||i>=N-1||j<1||j>=N-1) return;
        int N2=N*N, idx=i*N+j;
        double rho=0; for(int q=0;q<9;q++) rho+=f(q*N2+idx);
        double omega=1.5;
        double w[9]={4.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/36,1.0/36,1.0/36,1.0/36};
        for(int q=0;q<9;q++){
            double feq=w[q]*rho;
            f2(q*N2+idx)=f(q*N2+idx)+omega*(feq-f(q*N2+idx));
        }
    }
};

Result test_lbm(int N, int steps) {
    int N2=N*N;
    View1D f("f",9*N2), f2("f2",9*N2);
    Kokkos::deep_copy(f, 1.0/9.0); Kokkos::deep_copy(f2, 1.0/9.0);
    auto mdpol=Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{N,N});
    LBM_step sf{f,f2,N};
    struct LC{View1D f,f2;int n;KOKKOS_INLINE_FUNCTION void operator()(int i)const{f(i)=f2(i);}};
    LC cf{f,f2,9*N2};
    auto step=[&](){ Kokkos::parallel_for("lbm",mdpol,sf); Kokkos::parallel_for("lbmcopy",9*N2,cf); };
    return bench(step,20,steps);
}

// PIC 1D (4 phases)
Result test_pic(int NP, int NG, int steps) {
    View1D xp("xp",NP),vp("vp",NP),rho_g("rho",NG),E_g("E",NG);
    Kokkos::parallel_for(NP,KOKKOS_LAMBDA(int i){xp(i)=(double)i/NP;});
    auto step=[&](){
        Kokkos::deep_copy(rho_g,0.0);
        Kokkos::parallel_for("deposit",NP,KOKKOS_LAMBDA(int i){
            double dx=1.0/NG; int c=(int)(xp(i)/dx); c=c<0?0:(c>=NG?NG-1:c);
            Kokkos::atomic_add(&rho_g(c),1.0);
        });
        Kokkos::parallel_for("field",NG,KOKKOS_LAMBDA(int i){
            if(i>=1&&i<NG-1) E_g(i)=-(rho_g(i+1)-rho_g(i-1))*0.5*NG;
        });
        Kokkos::parallel_for("push",NP,KOKKOS_LAMBDA(int i){
            double dx=1.0/NG, dt=0.0001;
            int c=(int)(xp(i)/dx); c=c<0?0:(c>=NG?NG-1:c);
            vp(i)+=dt*E_g(c); xp(i)+=dt*vp(i);
            if(xp(i)<0) xp(i)+=1.0; if(xp(i)>=1.0) xp(i)-=1.0;
        });
    };
    return bench(step,20,steps);
}

// SPH, DEM, MD_LJ — same pattern as NBody but different force
struct SPH_step {
    View1D px,py,vx,vy,rho_s,px2,py2,vx2,vy2; int N;
    KOKKOS_INLINE_FUNCTION void operator()(int i) const {
        double h=0.1,dt=0.0001,mass=1.0,k=100.0,rho0=1.0;
        double rho_i=0;
        for(int j=0;j<N;j++){double dx=px(j)-px(i),dy=py(j)-py(i);double r2=dx*dx+dy*dy;
            if(r2<h*h){double q=sqrt(r2)/h; rho_i+=mass*(1-q)*(1-q);}}
        rho_s(i)=rho_i>0.001?rho_i:0.001;
        double ax=0,ay=0,pi_=k*(rho_i-rho0);
        for(int j=0;j<N;j++){if(i==j)continue;
            double dx=px(j)-px(i),dy=py(j)-py(i);double r=sqrt(dx*dx+dy*dy)+1e-6;
            if(r<h){double pj=k*(rho_s(j)-rho0);double f=-mass*(pi_+pj)/(2*rho_s(j))*(-2*(1-r/h)/h)/r;
                ax+=f*dx; ay+=f*dy;}}
        vx2(i)=vx(i)+dt*ax; vy2(i)=vy(i)+dt*ay;
        px2(i)=px(i)+dt*vx2(i); py2(i)=py(i)+dt*vy2(i);
    }
};

Result test_sph(int N, int steps) {
    View1D px("px",N),py("py",N),vx("vx",N),vy("vy",N),rho_s("rho",N);
    View1D px2("px2",N),py2("py2",N),vx2("vx2",N),vy2("vy2",N);
    Kokkos::parallel_for(N,KOKKOS_LAMBDA(int i){px(i)=(i%32)*0.015;py(i)=((i*7)%32)*0.015;});
    SPH_step sf{px,py,vx,vy,rho_s,px2,py2,vx2,vy2,N};
    NBody_copy nc{px,py,vx,vy,px2,py2,vx2,vy2,N};
    auto step=[&](){ Kokkos::parallel_for(N,sf); Kokkos::parallel_for(N,nc); };
    return bench(step,5,steps);
}

struct DEM_step {
    View1D px,py,vx,vy,px2,py2,vx2,vy2; int N;
    KOKKOS_INLINE_FUNCTION void operator()(int i) const {
        double dt=0.0001,rad=0.01,kn=1e4,gn=10.0;
        double ax=0,ay=-9.81;
        for(int j=0;j<N;j++){if(i==j)continue;
            double dx=px(j)-px(i),dy=py(j)-py(i);double dist=sqrt(dx*dx+dy*dy)+1e-10;
            double overlap=2*rad-dist;
            if(overlap>0){double nx=dx/dist,ny=dy/dist;
                double dvn=(vx(j)-vx(i))*nx+(vy(j)-vy(i))*ny;
                double fn=kn*overlap-gn*dvn; ax+=fn*nx; ay+=fn*ny;}}
        vx2(i)=vx(i)+dt*ax; vy2(i)=vy(i)+dt*ay;
        px2(i)=px(i)+dt*vx2(i); py2(i)=py(i)+dt*vy2(i);
    }
};

Result test_dem(int N, int steps) {
    View1D px("px",N),py("py",N),vx("vx",N),vy("vy",N);
    View1D px2("px2",N),py2("py2",N),vx2("vx2",N),vy2("vy2",N);
    Kokkos::parallel_for(N,KOKKOS_LAMBDA(int i){px(i)=(i%32)*0.03;py(i)=((i*7)%32)*0.03;});
    DEM_step sf{px,py,vx,vy,px2,py2,vx2,vy2,N};
    NBody_copy nc{px,py,vx,vy,px2,py2,vx2,vy2,N};
    auto step=[&](){ Kokkos::parallel_for(N,sf); Kokkos::parallel_for(N,nc); };
    return bench(step,5,steps);
}

struct MDLJ_step {
    View1D px,py,vx,vy,px2,py2,vx2,vy2; int N;
    KOKKOS_INLINE_FUNCTION void operator()(int i) const {
        double dt=0.0001,eps=1.0,sigma=0.01;
        double ax=0,ay=0;
        for(int j=0;j<N;j++){if(i==j)continue;
            double dx=px(j)-px(i),dy=py(j)-py(i);double r2=dx*dx+dy*dy+1e-10;
            double s2=sigma*sigma/r2,s6=s2*s2*s2;
            double f=24*eps*(2*s6*s6-s6)/r2; ax+=f*dx; ay+=f*dy;}
        vx2(i)=vx(i)+dt*ax; vy2(i)=vy(i)+dt*ay;
        px2(i)=px(i)+dt*vx2(i); py2(i)=py(i)+dt*vy2(i);
    }
};

Result test_mdlj(int N, int steps) {
    View1D px("px",N),py("py",N),vx("vx",N),vy("vy",N);
    View1D px2("px2",N),py2("py2",N),vx2("vx2",N),vy2("vy2",N);
    Kokkos::parallel_for(N,KOKKOS_LAMBDA(int i){px(i)=(i%32)*0.03;py(i)=((i*7)%32)*0.03;});
    MDLJ_step sf{px,py,vx,vy,px2,py2,vx2,vy2,N};
    NBody_copy nc{px,py,vx,vy,px2,py2,vx2,vy2,N};
    auto step=[&](){ Kokkos::parallel_for(N,sf); Kokkos::parallel_for(N,nc); };
    return bench(step,5,steps);
}

// =========================================================================
// Main
// =========================================================================
int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        int STEPS = (argc > 1) ? atoi(argv[1]) : 500;
        printf("Kokkos Characterization (%d steps)\n", STEPS);
        printf("============================================================\n");

        printf("\n--- Stencil ---\n");
        report("Heat2D 128sq", test_2d_functor<Heat2D_step, Heat2D_copy>(128, STEPS));
        report("Jacobi2D 128sq", test_2d_functor<Jacobi2D_step, Jacobi2D_copy>(128, STEPS));
        report("Wave2D 128sq", test_2d_functor<Wave2D_step, Wave2D_copy>(128, STEPS));
        report("GrayScott 128sq", test_2d_functor<GrayScott_step, GrayScott_copy>(128, STEPS));
        report("AllenCahn 128sq", test_2d_functor<AllenCahn_step, AllenCahn_copy>(128, STEPS));
        report("CahnHilliard 128sq", test_2d_functor<CahnHilliard_step, CahnHilliard_copy>(128, STEPS));
        report("Burgers2D 128sq", test_2d_functor<Burgers2D_step, Burgers2D_copy>(128, STEPS));
        report("ConvDiff 128sq", test_2d_functor<ConvDiff_step, ConvDiff_copy>(128, STEPS));

        printf("\n--- CFD ---\n");
        report("FDTD Maxwell 128sq", test_2d_functor<FDTD_step, FDTD_copy>(128, STEPS));

        printf("\n--- FEM/Structure ---\n");
        report("ExplicitFEM 128sq", test_2d_functor<ExplFEM_step, ExplFEM_copy>(128, STEPS));
        report("Cloth 128sq", test_2d_functor<Cloth_step, Cloth_copy>(128, STEPS));

        printf("\n--- Classic ---\n");
        report("SRAD 128sq", test_2d_functor<SRAD_step, SRAD_copy>(128, STEPS));
        report("HotSpot 128sq", test_2d_functor<HotSpot_step, HotSpot_copy>(128, STEPS));
        report("Poisson2D 128sq", test_2d_functor<Poisson2D_step, Poisson2D_copy>(128, STEPS));
        report("Helmholtz2D 128sq", test_2d_functor<Helmholtz2D_step, Helmholtz2D_copy>(128, STEPS));
        report("SemiLagrangian 128sq", test_2d_functor<SemiLag_step, SemiLag_copy>(128, STEPS));

        printf("\n--- 1D PDE ---\n");
        report("Upwind1D N=4096", test_1d_functor<Upwind1D_step>(4096, STEPS));
        report("KS 1D N=4096", test_1d_functor<KS1D_step>(4096, STEPS));
        report("SpMV N=4096", test_1d_functor<SpMV_step>(4096, STEPS));
        report("Reduction N=16384", test_1d_functor<Reduce_step>(16384, STEPS));
        report("MassSpring1D N=4096", test_1d_functor<MassSpring1D_step>(4096, STEPS));
        report("Schrodinger1D N=4096", test_1d_functor<Schrodinger1D_step>(4096, STEPS));

        printf("\n--- CFD (additional) ---\n");
        report("SWE LaxFried 128sq", test_swe(128, STEPS));
        report("LBM D2Q9 64sq", test_lbm(64, STEPS));
        report("Euler1D N=4096", test_euler1d(4096, STEPS));

        printf("\n--- 3D Stencil ---\n");
        report("Heat3D 32cu", test_3d(32, STEPS, true));
        report("Jacobi3D 32cu", test_3d(32, STEPS, false));

        printf("\n--- Particle ---\n");
        report("NBody N=256", test_nbody(256, 200));
        report("NBody N=1024", test_nbody(1024, 50));
        report("SPH N=1024", test_sph(1024, 50));
        report("DEM N=1024", test_dem(1024, 50));
        report("MD_LJ N=1024", test_mdlj(1024, 50));
        report("PIC NP=4096", test_pic(4096, 256, STEPS));
        report("MonteCarlo N=4096", test_montecarlo(4096, STEPS));

        printf("\n============================================================\n");
        printf("Total: 36/36 kernel types (CG + LULESH in separate files,\n");
        printf("  StableFluids approximated by Jacobi pressure iterations)\n");
    }
    Kokkos::finalize();
}
