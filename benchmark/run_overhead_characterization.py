"""
Overhead Characterization: 25+ kernel types × multiple sizes
Measures Taichi per-step time to quantify launch overhead fraction.
Output: kernel name, domain, size, elements, μs/step, estimated overhead %

Usage: python run_overhead_characterization.py
"""
import time
import sys

# Estimated fixed overhead (will be calibrated per-GPU)
PYTHON_OH = 15.0  # μs baseline, updated after first measurement

def bench(step_fn, n_steps=200):
    """Return μs per step."""
    import taichi as ti
    for _ in range(10): step_fn()
    ti.sync()
    t0 = time.perf_counter()
    for _ in range(n_steps): step_fn()
    ti.sync()
    return (time.perf_counter() - t0) * 1e6 / n_steps

results = []

def run_and_record(name, domain, size_str, n_elem, step_fn, n_steps=200):
    us = bench(step_fn, n_steps)
    results.append((name, domain, size_str, n_elem, us))
    oh_pct = min(PYTHON_OH / us * 100, 100) if us > 0 else 0
    tag = "OH-DOM" if oh_pct > 50 else "COMPUTE" if oh_pct < 20 else "TRANS"
    print(f"  {name:<30} {size_str:>10} {us:>8.1f} us  OH~{oh_pct:>3.0f}%  [{tag}]")

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

import taichi as ti

# ============================
# 1. Heat Equation 2D
# ============================
section("1. Heat Equation 2D")
for N in [64, 128, 256, 512, 1024, 2048]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    u=ti.field(ti.f32,(N,N)); v=ti.field(ti.f32,(N,N))
    @ti.kernel
    def heat2d():
        for i,j in ti.ndrange((1,N-1),(1,N-1)):
            v[i,j]=u[i,j]+0.2*(u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1]-4*u[i,j])
        for i,j in u: u[i,j]=v[i,j]
    steps=500 if N<=512 else 200 if N<=1024 else 100
    run_and_record("Heat2D",  "Thermal", f"{N}^2", N*N, heat2d, steps)
    ti.reset()

# ============================
# 2. Wave Equation 2D
# ============================
section("2. Wave Equation 2D")
for N in [128, 256, 512, 1024, 2048]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    hp=ti.field(ti.f32,(N,N)); hc=ti.field(ti.f32,(N,N))
    @ti.kernel
    def wave2d():
        for i,j in ti.ndrange((1,N-1),(1,N-1)):
            lap=hc[i-1,j]+hc[i+1,j]+hc[i,j-1]+hc[i,j+1]-4*hc[i,j]
            h_new=2*hc[i,j]-hp[i,j]+0.01*lap
            hp[i,j]=hc[i,j]; hc[i,j]=h_new
    steps=500 if N<=512 else 200
    run_and_record("Wave2D", "Acoustics", f"{N}^2", N*N, wave2d, steps)
    ti.reset()

# ============================
# 3. Jacobi 2D
# ============================
section("3. Jacobi 2D")
for N in [64, 256, 512, 1024, 2048]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    a=ti.field(ti.f32,(N,N));b=ti.field(ti.f32,(N,N))
    @ti.kernel
    def jacobi():
        for i,j in ti.ndrange((1,N-1),(1,N-1)):
            b[i,j]=0.25*(a[i-1,j]+a[i+1,j]+a[i,j-1]+a[i,j+1])
        for i,j in a: a[i,j]=b[i,j]
    steps=500 if N<=512 else 200
    run_and_record("Jacobi2D", "LinearSolve", f"{N}^2", N*N, jacobi, steps)
    ti.reset()

# ============================
# 4. Gray-Scott Reaction-Diffusion
# ============================
section("4. Gray-Scott RD")
for N in [128, 256, 512, 1024]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    gu=ti.field(ti.f32,(N,N));gv=ti.field(ti.f32,(N,N))
    gu2=ti.field(ti.f32,(N,N));gv2=ti.field(ti.f32,(N,N))
    @ti.kernel
    def grayscott():
        for i,j in ti.ndrange((1,N-1),(1,N-1)):
            lu=gu[i-1,j]+gu[i+1,j]+gu[i,j-1]+gu[i,j+1]-4*gu[i,j]
            lv=gv[i-1,j]+gv[i+1,j]+gv[i,j-1]+gv[i,j+1]-4*gv[i,j]
            uvv=gu[i,j]*gv[i,j]*gv[i,j]
            gu2[i,j]=gu[i,j]+0.16*lu-uvv+0.06*(1-gu[i,j])
            gv2[i,j]=gv[i,j]+0.08*lv+uvv-0.122*gv[i,j]
        for i,j in gu: gu[i,j]=gu2[i,j]; gv[i,j]=gv2[i,j]
    steps=500 if N<=512 else 200
    run_and_record("GrayScott", "Chemistry", f"{N}^2", N*N, grayscott, steps)
    ti.reset()

# ============================
# 5. Allen-Cahn Phase Field
# ============================
section("5. Allen-Cahn")
for N in [128, 256, 512, 1024]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    phi=ti.field(ti.f32,(N,N));phi2=ti.field(ti.f32,(N,N))
    @ti.kernel
    def allencahn():
        for i,j in ti.ndrange((1,N-1),(1,N-1)):
            lap=phi[i-1,j]+phi[i+1,j]+phi[i,j-1]+phi[i,j+1]-4*phi[i,j]
            p=phi[i,j]; phi2[i,j]=p+0.001*(0.01*lap-p*(p*p-1))
        for i,j in phi: phi[i,j]=phi2[i,j]
    run_and_record("AllenCahn", "Materials", f"{N}^2", N*N, allencahn, 500)
    ti.reset()

# ============================
# 6. FDTD Maxwell 2D
# ============================
section("6. FDTD Maxwell 2D")
for N in [128, 256, 512, 1024, 2048]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    Ez=ti.field(ti.f32,(N,N));Hx=ti.field(ti.f32,(N,N));Hy=ti.field(ti.f32,(N,N))
    @ti.kernel
    def fdtd():
        for i,j in ti.ndrange((0,N-1),(0,N-1)):
            Hx[i,j]-=0.5*(Ez[i,j+1]-Ez[i,j]); Hy[i,j]+=0.5*(Ez[i+1,j]-Ez[i,j])
        for i,j in ti.ndrange((1,N),(1,N)):
            Ez[i,j]+=0.5*(Hy[i,j]-Hy[i-1,j]-Hx[i,j]+Hx[i,j-1])
    steps=500 if N<=512 else 200
    run_and_record("FDTD_Maxwell", "EM", f"{N}^2", N*N, fdtd, steps)
    ti.reset()

# ============================
# 7. Burgers 2D
# ============================
section("7. Burgers 2D")
for N in [128, 256, 512, 1024]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    bu=ti.field(ti.f32,(N,N));bv=ti.field(ti.f32,(N,N))
    bu2=ti.field(ti.f32,(N,N));bv2=ti.field(ti.f32,(N,N))
    @ti.kernel
    def burgers():
        nu=0.01; dt=0.001
        for i,j in ti.ndrange((1,N-1),(1,N-1)):
            ux=(bu[i+1,j]-bu[i-1,j])*0.5; uy=(bu[i,j+1]-bu[i,j-1])*0.5
            vx=(bv[i+1,j]-bv[i-1,j])*0.5; vy=(bv[i,j+1]-bv[i,j-1])*0.5
            lu=bu[i-1,j]+bu[i+1,j]+bu[i,j-1]+bu[i,j+1]-4*bu[i,j]
            lv=bv[i-1,j]+bv[i+1,j]+bv[i,j-1]+bv[i,j+1]-4*bv[i,j]
            bu2[i,j]=bu[i,j]+dt*(-bu[i,j]*ux-bv[i,j]*uy+nu*lu)
            bv2[i,j]=bv[i,j]+dt*(-bu[i,j]*vx-bv[i,j]*vy+nu*lv)
        for i,j in bu: bu[i,j]=bu2[i,j]; bv[i,j]=bv2[i,j]
    run_and_record("Burgers2D", "Fluids", f"{N}^2", N*N, burgers, 500)
    ti.reset()

# ============================
# 8. Convection-Diffusion
# ============================
section("8. Convection-Diffusion")
for N in [128, 256, 512, 1024]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    cd=ti.field(ti.f32,(N,N));cd2=ti.field(ti.f32,(N,N))
    @ti.kernel
    def convdiff():
        for i,j in ti.ndrange((1,N-1),(1,N-1)):
            lap=cd[i-1,j]+cd[i+1,j]+cd[i,j-1]+cd[i,j+1]-4*cd[i,j]
            adv=0.5*(cd[i+1,j]-cd[i-1,j])+0.3*(cd[i,j+1]-cd[i,j-1])
            cd2[i,j]=cd[i,j]+0.001*(0.01*lap-adv)
        for i,j in cd: cd[i,j]=cd2[i,j]
    run_and_record("ConvDiff", "Transport", f"{N}^2", N*N, convdiff, 500)
    ti.reset()

# ============================
# 9. SWE (Lax-Friedrichs)
# ============================
section("9. Shallow Water Eqn")
for N in [128, 256, 512, 1024]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    h_sw=ti.field(ti.f32,(N,N));hu=ti.field(ti.f32,(N,N));hv=ti.field(ti.f32,(N,N))
    h2=ti.field(ti.f32,(N,N));hu2=ti.field(ti.f32,(N,N));hv2=ti.field(ti.f32,(N,N))
    @ti.kernel
    def swe_lf():
        g=9.81; dt=0.001
        for i,j in ti.ndrange((1,N-1),(1,N-1)):
            h_avg=0.25*(h_sw[i-1,j]+h_sw[i+1,j]+h_sw[i,j-1]+h_sw[i,j+1])
            fx=(hu[i+1,j]-hu[i-1,j])*0.5; fy=(hv[i,j+1]-hv[i,j-1])*0.5
            h2[i,j]=h_avg-dt*(fx+fy)
            hu2[i,j]=hu[i,j]-dt*g*h_sw[i,j]*(h_sw[i+1,j]-h_sw[i-1,j])*0.5
            hv2[i,j]=hv[i,j]-dt*g*h_sw[i,j]*(h_sw[i,j+1]-h_sw[i,j-1])*0.5
        for i,j in h_sw: h_sw[i,j]=h2[i,j];hu[i,j]=hu2[i,j];hv[i,j]=hv2[i,j]
    @ti.kernel
    def swe_init():
        for i,j in h_sw: h_sw[i,j]=1.0+(0.5 if i<N//2 else 0.0)
    swe_init()
    run_and_record("SWE_LaxFried", "CFD", f"{N}^2", N*N, swe_lf, 500)
    ti.reset()

# ============================
# 10. Cloth (mass-spring)
# ============================
section("10. Cloth Mass-Spring")
for N in [32, 64, 128, 256]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    xc=ti.Vector.field(3,ti.f32,(N,N));vc=ti.Vector.field(3,ti.f32,(N,N))
    @ti.kernel
    def cloth():
        for i,j in vc:
            vc[i,j][1]-=0.001
            if i>0: vc[i,j]+=(xc[i-1,j]-xc[i,j])*0.5
            if i<N-1: vc[i,j]+=(xc[i+1,j]-xc[i,j])*0.5
            if j>0: vc[i,j]+=(xc[i,j-1]-xc[i,j])*0.5
            if j<N-1: vc[i,j]+=(xc[i,j+1]-xc[i,j])*0.5
        for i,j in xc: xc[i,j]+=vc[i,j]*0.01
    @ti.kernel
    def cloth_init():
        for i,j in xc: xc[i,j]=[float(i)/N,1.0,float(j)/N]
    cloth_init()
    run_and_record("Cloth_Spring", "Textile", f"{N}^2", N*N, cloth, 500)
    ti.reset()

# ============================
# 11. N-body (brute force, limited)
# ============================
section("11. N-body")
for N in [256, 1024, 4096]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    pos=ti.Vector.field(3,ti.f32,N);vel=ti.Vector.field(3,ti.f32,N)
    @ti.kernel
    def nbody_init():
        for i in pos: pos[i]=[ti.random(),ti.random(),ti.random()]
    @ti.kernel
    def nbody():
        dt=0.0001
        for i in range(N):
            acc=ti.Vector([0.0,0.0,0.0])
            for j in range(N):
                if i!=j:
                    r=pos[j]-pos[i]; d=r.norm()+0.01
                    acc+=r/(d*d*d)
            vel[i]+=acc*dt; pos[i]+=vel[i]*dt
    nbody_init()
    steps=100 if N<=1024 else 20
    run_and_record("NBody", "Particle", f"N={N}", N, nbody, steps)
    ti.reset()

# ============================
# 12. SPH Density
# ============================
section("12. SPH Density")
for N in [1024, 4096, 16384]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    xp=ti.Vector.field(2,ti.f32,N);rho=ti.field(ti.f32,N)
    @ti.kernel
    def sph_init():
        s=int(ti.sqrt(float(N)))
        for i in xp: xp[i]=[float(i%s)/s,float(i//s)/s]
    @ti.kernel
    def sph_density():
        for i in range(N):
            d=0.0
            for j in range(ti.min(N,256)):
                r=(xp[i]-xp[j]).norm()
                if r<0.05: d+=(0.05-r)**3
            rho[i]=d
    sph_init()
    steps=100 if N<=4096 else 20
    run_and_record("SPH_Density", "Particle", f"N={N}", N, sph_density, steps)
    ti.reset()

# ============================
# 13. Heat 3D
# ============================
section("13. Heat 3D")
for N in [16, 32, 64, 128]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    u3=ti.field(ti.f32,(N,N,N));v3=ti.field(ti.f32,(N,N,N))
    @ti.kernel
    def heat3d():
        for i,j,k in ti.ndrange((1,N-1),(1,N-1),(1,N-1)):
            v3[i,j,k]=u3[i,j,k]+0.1*(u3[i-1,j,k]+u3[i+1,j,k]+u3[i,j-1,k]+u3[i,j+1,k]+u3[i,j,k-1]+u3[i,j,k+1]-6*u3[i,j,k])
        for i,j,k in u3: u3[i,j,k]=v3[i,j,k]
    steps=200 if N<=64 else 50
    run_and_record("Heat3D", "Thermal", f"{N}^3", N**3, heat3d, steps)
    ti.reset()

# ============================
# 14. Semi-Lagrangian Advection
# ============================
section("14. Semi-Lagrangian Advection")
for N in [128, 256, 512, 1024]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    q=ti.field(ti.f32,(N,N));q2=ti.field(ti.f32,(N,N))
    @ti.kernel
    def advect():
        for i,j in ti.ndrange((1,N-1),(1,N-1)):
            x=float(i)-0.5; y=float(j)-0.3
            x=ti.max(0.5,ti.min(float(N)-1.5,x));y=ti.max(0.5,ti.min(float(N)-1.5,y))
            i0=int(x);j0=int(y);s=x-i0;t=y-j0
            q2[i,j]=(1-s)*(1-t)*q[i0,j0]+s*(1-t)*q[i0+1,j0]+(1-s)*t*q[i0,j0+1]+s*t*q[i0+1,j0+1]
        for i,j in q: q[i,j]=q2[i,j]
    run_and_record("SemiLagrangian", "Transport", f"{N}^2", N*N, advect, 500)
    ti.reset()

# ============================
# 15. Cahn-Hilliard
# ============================
section("15. Cahn-Hilliard")
for N in [128, 256, 512]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    ch=ti.field(ti.f32,(N,N));ch2=ti.field(ti.f32,(N,N));mu=ti.field(ti.f32,(N,N))
    @ti.kernel
    def cahnhilliard():
        for i,j in ti.ndrange((1,N-1),(1,N-1)):
            lap=ch[i-1,j]+ch[i+1,j]+ch[i,j-1]+ch[i,j+1]-4*ch[i,j]
            mu[i,j]=ch[i,j]**3-ch[i,j]-0.01*lap
        for i,j in ti.ndrange((1,N-1),(1,N-1)):
            lap_mu=mu[i-1,j]+mu[i+1,j]+mu[i,j-1]+mu[i,j+1]-4*mu[i,j]
            ch2[i,j]=ch[i,j]+0.001*lap_mu
        for i,j in ch: ch[i,j]=ch2[i,j]
    run_and_record("CahnHilliard", "Materials", f"{N}^2", N*N, cahnhilliard, 500)
    ti.reset()

# ============================
# Summary
# ============================
print(f"\n{'='*80}")
print(f"  SUMMARY: {len(results)} configurations")
print(f"{'='*80}")
print(f"{'Kernel':<30} {'Domain':<12} {'Size':>10} {'us/step':>10} {'Class':>10}")
print("-"*75)

oh_dom = 0; compute_dom = 0; trans = 0
for name, domain, size, nelem, us in results:
    oh = min(PYTHON_OH / us * 100, 100) if us > 0 else 0
    cls = "OH-DOM" if oh > 50 else "COMPUTE" if oh < 20 else "TRANS"
    if oh > 50: oh_dom += 1
    elif oh < 20: compute_dom += 1
    else: trans += 1
    print(f"{name:<30} {domain:<12} {size:>10} {us:>9.1f} {cls:>10}")

print(f"\nClassification: OH-dominated={oh_dom}  Transitional={trans}  Compute-dominated={compute_dom}")
print(f"Total: {len(results)} configurations across {15} kernel types")
