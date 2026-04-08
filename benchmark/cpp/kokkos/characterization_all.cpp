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

        printf("\n--- Particle ---\n");
        report("NBody N=256", test_nbody(256, 200));
        report("NBody N=1024", test_nbody(1024, 50));

        printf("\n============================================================\n");
        printf("Total: 23 kernel types (remaining: LBM, SWE, StableFluids,\n");
        printf("  Euler1D, SPH, DEM, MD_LJ, PIC, MonteCarlo, Heat3D,\n");
        printf("  Jacobi3D, CG, LULESH — need separate implementations)\n");
    }
    Kokkos::finalize();
}
