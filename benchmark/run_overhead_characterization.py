"""
Overhead Characterization: 30 kernel types × multiple sizes
Measures Taichi per-step time to quantify launch overhead fraction.

Kernel categories:
  Stencil (9):    Heat2D/3D, Wave2D, Jacobi, Burgers, ConvDiff, AllenCahn, CahnHilliard, GrayScott
  CFD (4):        SWE, LBM, StableFluids (pressure project), Euler1D
  Particle (5):   NBody, SPH, DEM, PIC1D, MolecularDynamics(LJ)
  EM (2):         FDTD Maxwell, Helmholtz
  FEM (2):        ExplicitFEM, MassSpring/Cloth
  Transport (2):  SemiLagrangian, Advection1D-WENO
  PDE misc (3):   Poisson(Jacobi iter), Schrodinger, KuramotoSivashinsky
  Other (3):      Reduction, Prefix-sum pattern, MonteCarlo

Total: 30 distinct kernel types × 2-5 sizes each = ~100 configurations

Usage: python run_overhead_characterization.py
"""
import time
import sys
import os

# Suppress Taichi verbose output
os.environ.setdefault('TI_LOG_LEVEL', 'warn')

PYTHON_OH = 15.0  # μs estimated baseline

def bench(step_fn, n_steps=200):
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

# =====================================================================
# STENCIL METHODS (9 types)
# =====================================================================

# --- 1. Heat 2D ---
section("1. Heat 2D")
for N in [64, 128, 256, 512, 1024, 2048]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    u=ti.field(ti.f32,(N,N));v=ti.field(ti.f32,(N,N))
    @ti.kernel
    def heat2d():
        for i,j in ti.ndrange((1,N-1),(1,N-1)):
            v[i,j]=u[i,j]+0.2*(u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1]-4*u[i,j])
        for i,j in u: u[i,j]=v[i,j]
    run_and_record("Heat2D","Stencil",f"{N}^2",N*N,heat2d,500 if N<=512 else 100)
    ti.reset()

# --- 2. Heat 3D ---
section("2. Heat 3D")
for N in [16, 32, 64, 128]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    u3=ti.field(ti.f32,(N,N,N));v3=ti.field(ti.f32,(N,N,N))
    @ti.kernel
    def heat3d():
        for i,j,k in ti.ndrange((1,N-1),(1,N-1),(1,N-1)):
            v3[i,j,k]=u3[i,j,k]+0.1*(u3[i-1,j,k]+u3[i+1,j,k]+u3[i,j-1,k]+u3[i,j+1,k]+u3[i,j,k-1]+u3[i,j,k+1]-6*u3[i,j,k])
        for i,j,k in u3: u3[i,j,k]=v3[i,j,k]
    run_and_record("Heat3D","Stencil",f"{N}^3",N**3,heat3d,200 if N<=64 else 50)
    ti.reset()

# --- 3. Wave 2D ---
section("3. Wave 2D")
for N in [128, 256, 512, 1024]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    hp=ti.field(ti.f32,(N,N));hc=ti.field(ti.f32,(N,N))
    @ti.kernel
    def wave2d():
        for i,j in ti.ndrange((1,N-1),(1,N-1)):
            lap=hc[i-1,j]+hc[i+1,j]+hc[i,j-1]+hc[i,j+1]-4*hc[i,j]
            h_new=2*hc[i,j]-hp[i,j]+0.01*lap
            hp[i,j]=hc[i,j]; hc[i,j]=h_new
    run_and_record("Wave2D","Stencil",f"{N}^2",N*N,wave2d,500 if N<=512 else 100)
    ti.reset()

# --- 4. Jacobi 2D ---
section("4. Jacobi 2D")
for N in [64, 256, 512, 1024]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    a=ti.field(ti.f32,(N,N));b=ti.field(ti.f32,(N,N))
    @ti.kernel
    def jacobi():
        for i,j in ti.ndrange((1,N-1),(1,N-1)):
            b[i,j]=0.25*(a[i-1,j]+a[i+1,j]+a[i,j-1]+a[i,j+1])
        for i,j in a: a[i,j]=b[i,j]
    run_and_record("Jacobi2D","Stencil",f"{N}^2",N*N,jacobi,500)
    ti.reset()

# --- 5. Gray-Scott ---
section("5. Gray-Scott RD")
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
    run_and_record("GrayScott","Stencil",f"{N}^2",N*N,grayscott,500 if N<=512 else 100)
    ti.reset()

# --- 6. Allen-Cahn ---
section("6. Allen-Cahn")
for N in [128, 256, 512, 1024]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    phi=ti.field(ti.f32,(N,N));phi2=ti.field(ti.f32,(N,N))
    @ti.kernel
    def allencahn():
        for i,j in ti.ndrange((1,N-1),(1,N-1)):
            lap=phi[i-1,j]+phi[i+1,j]+phi[i,j-1]+phi[i,j+1]-4*phi[i,j]
            p=phi[i,j]; phi2[i,j]=p+0.001*(0.01*lap-p*(p*p-1))
        for i,j in phi: phi[i,j]=phi2[i,j]
    run_and_record("AllenCahn","Stencil",f"{N}^2",N*N,allencahn,500)
    ti.reset()

# --- 7. Cahn-Hilliard ---
section("7. Cahn-Hilliard")
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
    run_and_record("CahnHilliard","Stencil",f"{N}^2",N*N,cahnhilliard,500)
    ti.reset()

# --- 8. Burgers 2D ---
section("8. Burgers 2D")
for N in [128, 256, 512, 1024]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    bu=ti.field(ti.f32,(N,N));bv=ti.field(ti.f32,(N,N))
    bu2=ti.field(ti.f32,(N,N));bv2=ti.field(ti.f32,(N,N))
    @ti.kernel
    def burgers():
        for i,j in ti.ndrange((1,N-1),(1,N-1)):
            ux=(bu[i+1,j]-bu[i-1,j])*0.5;uy=(bu[i,j+1]-bu[i,j-1])*0.5
            vx=(bv[i+1,j]-bv[i-1,j])*0.5;vy=(bv[i,j+1]-bv[i,j-1])*0.5
            lu=bu[i-1,j]+bu[i+1,j]+bu[i,j-1]+bu[i,j+1]-4*bu[i,j]
            lv=bv[i-1,j]+bv[i+1,j]+bv[i,j-1]+bv[i,j+1]-4*bv[i,j]
            bu2[i,j]=bu[i,j]+0.001*(-bu[i,j]*ux-bv[i,j]*uy+0.01*lu)
            bv2[i,j]=bv[i,j]+0.001*(-bu[i,j]*vx-bv[i,j]*vy+0.01*lv)
        for i,j in bu: bu[i,j]=bu2[i,j]; bv[i,j]=bv2[i,j]
    run_and_record("Burgers2D","Stencil",f"{N}^2",N*N,burgers,500)
    ti.reset()

# --- 9. Convection-Diffusion ---
section("9. Convection-Diffusion")
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
    run_and_record("ConvDiff","Stencil",f"{N}^2",N*N,convdiff,500)
    ti.reset()

# =====================================================================
# CFD (4 types)
# =====================================================================

# --- 10. SWE Lax-Friedrichs ---
section("10. SWE Lax-Friedrichs")
for N in [128, 256, 512, 1024]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    h_sw=ti.field(ti.f32,(N,N));hu=ti.field(ti.f32,(N,N));hv=ti.field(ti.f32,(N,N))
    h2=ti.field(ti.f32,(N,N));hu2=ti.field(ti.f32,(N,N));hv2=ti.field(ti.f32,(N,N))
    @ti.kernel
    def swe_lf():
        for i,j in ti.ndrange((1,N-1),(1,N-1)):
            h_avg=0.25*(h_sw[i-1,j]+h_sw[i+1,j]+h_sw[i,j-1]+h_sw[i,j+1])
            fx=(hu[i+1,j]-hu[i-1,j])*0.5;fy=(hv[i,j+1]-hv[i,j-1])*0.5
            h2[i,j]=h_avg-0.001*(fx+fy)
            hu2[i,j]=hu[i,j]-0.001*9.81*h_sw[i,j]*(h_sw[i+1,j]-h_sw[i-1,j])*0.5
            hv2[i,j]=hv[i,j]-0.001*9.81*h_sw[i,j]*(h_sw[i,j+1]-h_sw[i,j-1])*0.5
        for i,j in h_sw: h_sw[i,j]=h2[i,j];hu[i,j]=hu2[i,j];hv[i,j]=hv2[i,j]
    @ti.kernel
    def swe_init():
        for i,j in h_sw: h_sw[i,j]=1.0+(0.5 if i<N//2 else 0.0)
    swe_init()
    run_and_record("SWE_LaxFried","CFD",f"{N}^2",N*N,swe_lf,500)
    ti.reset()

# --- 11. LBM D2Q9 ---
section("11. LBM D2Q9")
for N in [64, 128, 256, 512]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    f_old=ti.field(ti.f32,(N,N,9));f_new=ti.field(ti.f32,(N,N,9))
    rho=ti.field(ti.f32,(N,N));ux=ti.field(ti.f32,(N,N));uy=ti.field(ti.f32,(N,N))
    ex=ti.field(ti.i32,9);ey=ti.field(ti.i32,9);w=ti.field(ti.f32,9)
    @ti.kernel
    def lbm_init():
        ex_v=[0,1,0,-1,0,1,-1,-1,1]; ey_v=[0,0,1,0,-1,1,1,-1,-1]
        w_v=[4.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/36,1.0/36,1.0/36,1.0/36]
        for k in range(9): ex[k]=ex_v[k];ey[k]=ey_v[k];w[k]=w_v[k]
        for i,j in rho:
            rho[i,j]=1.0; ux[i,j]=0.0; uy[i,j]=0.0
            for k in range(9): f_old[i,j,k]=w[k]
    @ti.kernel
    def lbm_step():
        omega=1.0
        for i,j in ti.ndrange(N,N):
            r=0.0;u=0.0;v=0.0
            for k in range(9): r+=f_old[i,j,k];u+=f_old[i,j,k]*ex[k];v+=f_old[i,j,k]*ey[k]
            if r>0: u/=r;v/=r
            rho[i,j]=r;ux[i,j]=u;uy[i,j]=v
            for k in range(9):
                eu=float(ex[k])*u+float(ey[k])*v; usq=u*u+v*v
                feq=w[k]*r*(1+3*eu+4.5*eu*eu-1.5*usq)
                f_coll=f_old[i,j,k]+omega*(feq-f_old[i,j,k])
                ni=(i+ex[k]+N)%N; nj=(j+ey[k]+N)%N
                f_new[ni,nj,k]=f_coll
        for i,j in ti.ndrange(N,N):
            for k in range(9): f_old[i,j,k]=f_new[i,j,k]
    lbm_init()
    run_and_record("LBM_D2Q9","CFD",f"{N}^2",N*N,lbm_step,200 if N<=256 else 50)
    ti.reset()

# --- 12. Stable Fluids (pressure projection) ---
section("12. Stable Fluids (Jacobi pressure)")
for N in [64, 128, 256, 512]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    vx=ti.field(ti.f32,(N,N));vy=ti.field(ti.f32,(N,N))
    p=ti.field(ti.f32,(N,N));div=ti.field(ti.f32,(N,N));p2=ti.field(ti.f32,(N,N))
    @ti.kernel
    def divergence():
        for i,j in ti.ndrange((1,N-1),(1,N-1)):
            div[i,j]=-0.5*(vx[i+1,j]-vx[i-1,j]+vy[i,j+1]-vy[i,j-1])
    @ti.kernel
    def pressure_jacobi():
        for i,j in ti.ndrange((1,N-1),(1,N-1)):
            p2[i,j]=0.25*(p[i-1,j]+p[i+1,j]+p[i,j-1]+p[i,j+1]+div[i,j])
        for i,j in p: p[i,j]=p2[i,j]
    @ti.kernel
    def project():
        for i,j in ti.ndrange((1,N-1),(1,N-1)):
            vx[i,j]-=0.5*(p[i+1,j]-p[i-1,j])
            vy[i,j]-=0.5*(p[i,j+1]-p[i,j-1])
    def fluid_step():
        divergence()
        for _ in range(20): pressure_jacobi()  # 20 Jacobi iterations
        project()
    run_and_record("StableFluids","CFD",f"{N}^2",N*N,fluid_step,100)
    ti.reset()

# --- 13. Euler 1D (compressible) ---
section("13. Euler 1D (compressible, Lax-Friedrichs)")
for N in [1024, 4096, 16384, 65536]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    rho_e=ti.field(ti.f32,N);mom=ti.field(ti.f32,N);E=ti.field(ti.f32,N)
    rho2=ti.field(ti.f32,N);mom2=ti.field(ti.f32,N);E2=ti.field(ti.f32,N)
    @ti.kernel
    def euler1d_init():
        for i in rho_e:
            if i<N//2: rho_e[i]=1.0;mom[i]=0.0;E[i]=2.5
            else: rho_e[i]=0.125;mom[i]=0.0;E[i]=0.25
    @ti.kernel
    def euler1d():
        gamma=1.4; dt=0.0001
        for i in ti.ndrange((1,N-1)):
            # Lax-Friedrichs
            rho2[i]=0.5*(rho_e[i-1]+rho_e[i+1])-dt*0.5*(mom[i+1]-mom[i-1])
            u_l=mom[i-1]/(rho_e[i-1]+1e-10);u_r=mom[i+1]/(rho_e[i+1]+1e-10)
            p_l=(gamma-1)*(E[i-1]-0.5*rho_e[i-1]*u_l*u_l)
            p_r=(gamma-1)*(E[i+1]-0.5*rho_e[i+1]*u_r*u_r)
            mom2[i]=0.5*(mom[i-1]+mom[i+1])-dt*0.5*(mom[i+1]*u_r+p_r-mom[i-1]*u_l-p_l)
            E2[i]=0.5*(E[i-1]+E[i+1])-dt*0.5*((E[i+1]+p_r)*u_r-(E[i-1]+p_l)*u_l)
        for i in rho_e: rho_e[i]=rho2[i];mom[i]=mom2[i];E[i]=E2[i]
    euler1d_init()
    run_and_record("Euler1D","CFD",f"N={N}",N,euler1d,500)
    ti.reset()

# =====================================================================
# PARTICLE METHODS (5 types)
# =====================================================================

# --- 14. N-body ---
section("14. N-body (brute force)")
for N in [256, 1024, 4096]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    pos=ti.Vector.field(3,ti.f32,N);vel=ti.Vector.field(3,ti.f32,N)
    @ti.kernel
    def nbody_init():
        for i in pos: pos[i]=[ti.random(),ti.random(),ti.random()]
    @ti.kernel
    def nbody():
        for i in range(N):
            acc=ti.Vector([0.0,0.0,0.0])
            for j in range(N):
                if i!=j:
                    r=pos[j]-pos[i];d=r.norm()+0.01
                    acc+=r/(d*d*d)
            vel[i]+=acc*0.0001;pos[i]+=vel[i]*0.0001
    nbody_init()
    run_and_record("NBody","Particle",f"N={N}",N,nbody,100 if N<=1024 else 20)
    ti.reset()

# --- 15. SPH Density ---
section("15. SPH Density")
for N in [1024, 4096, 16384]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    xp=ti.Vector.field(2,ti.f32,N);rho_s=ti.field(ti.f32,N)
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
            rho_s[i]=d
    sph_init()
    run_and_record("SPH_Density","Particle",f"N={N}",N,sph_density,100 if N<=4096 else 20)
    ti.reset()

# --- 16. DEM (Discrete Element Method) ---
section("16. DEM (Discrete Element, spring-dashpot)")
for N in [256, 1024, 4096]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    xd=ti.Vector.field(2,ti.f32,N);vd=ti.Vector.field(2,ti.f32,N)
    rad=ti.field(ti.f32,N)
    @ti.kernel
    def dem_init():
        s=int(ti.sqrt(float(N)))
        for i in xd:
            xd[i]=[float(i%s)/s*2,float(i//s)/s*2]
            vd[i]=[0.0,0.0]; rad[i]=0.02
    @ti.kernel
    def dem_step():
        kn=1000.0;cn=10.0;dt=0.0001
        for i in range(N):
            f=ti.Vector([0.0,-9.8])
            for j in range(ti.min(N,128)):
                if i!=j:
                    d=xd[i]-xd[j];dist=d.norm();overlap=rad[i]+rad[j]-dist
                    if overlap>0 and dist>1e-6:
                        n=d/dist;vrel=(vd[i]-vd[j]).dot(n)
                        f+=(kn*overlap-cn*vrel)*n
            # floor
            if xd[i][1]<rad[i]:
                f[1]+=kn*(rad[i]-xd[i][1])-cn*vd[i][1]
            vd[i]+=f*dt;xd[i]+=vd[i]*dt
    dem_init()
    run_and_record("DEM","Particle",f"N={N}",N,dem_step,100 if N<=1024 else 20)
    ti.reset()

# --- 17. Molecular Dynamics (Lennard-Jones) ---
section("17. MD Lennard-Jones")
for N in [256, 1024, 4096]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    xm=ti.Vector.field(3,ti.f32,N);vm=ti.Vector.field(3,ti.f32,N)
    @ti.kernel
    def md_init():
        s=int(ti.pow(float(N),1.0/3.0))+1
        for i in xm:
            xm[i]=[float(i%s)/s,float((i//s)%s)/s,float(i//(s*s))/s]
            vm[i]=[0.0,0.0,0.0]
    @ti.kernel
    def md_step():
        eps=1.0;sigma=0.1;dt=0.001;rc=0.3
        for i in range(N):
            f=ti.Vector([0.0,0.0,0.0])
            for j in range(ti.min(N,256)):
                if i!=j:
                    r=xm[j]-xm[i];d=r.norm()
                    if d<rc and d>0.01:
                        sr6=(sigma/d)**6
                        fmag=24*eps*(2*sr6*sr6-sr6)/(d*d)
                        f+=fmag*r
            vm[i]+=f*dt;xm[i]+=vm[i]*dt
    md_init()
    run_and_record("MD_LJ","Particle",f"N={N}",N,md_step,100 if N<=1024 else 20)
    ti.reset()

# --- 18. PIC 1D (Particle-in-Cell) ---
section("18. PIC 1D")
for NP in [1024, 4096, 16384]:
    NG=256  # grid cells
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    xp_pic=ti.field(ti.f32,NP);vp=ti.field(ti.f32,NP)
    rho_g=ti.field(ti.f32,NG);phi=ti.field(ti.f32,NG);Ef=ti.field(ti.f32,NG)
    phi2_pic=ti.field(ti.f32,NG)
    @ti.kernel
    def pic_init():
        for i in xp_pic: xp_pic[i]=ti.random()*float(NG); vp[i]=(ti.random()-0.5)*0.1
    @ti.kernel
    def pic_deposit():
        for i in rho_g: rho_g[i]=0.0
        for i in range(NP):
            idx=int(xp_pic[i])%NG
            ti.atomic_add(rho_g[idx],1.0/NP*NG)
    @ti.kernel
    def pic_poisson():
        for i in ti.ndrange((1,NG-1)):
            phi2_pic[i]=0.5*(phi[i-1]+phi[i+1]+rho_g[i])
        for i in phi: phi[i]=phi2_pic[i]
    @ti.kernel
    def pic_efield():
        for i in ti.ndrange((1,NG-1)):
            Ef[i]=-0.5*(phi[i+1]-phi[i-1])
    @ti.kernel
    def pic_push():
        dt=0.01
        for i in range(NP):
            idx=int(xp_pic[i])%NG
            vp[i]+=Ef[idx]*dt
            xp_pic[i]+=vp[i]*dt
            xp_pic[i]=xp_pic[i]%float(NG)
    def pic_step():
        pic_deposit()
        for _ in range(10): pic_poisson()
        pic_efield()
        pic_push()
    pic_init()
    run_and_record("PIC_1D","Particle",f"NP={NP}",NP,pic_step,100)
    ti.reset()

# =====================================================================
# EM (2 types)
# =====================================================================

# --- 19. FDTD Maxwell 2D ---
section("19. FDTD Maxwell 2D")
for N in [128, 256, 512, 1024]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    Ez=ti.field(ti.f32,(N,N));Hx=ti.field(ti.f32,(N,N));Hy=ti.field(ti.f32,(N,N))
    @ti.kernel
    def fdtd():
        for i,j in ti.ndrange((0,N-1),(0,N-1)):
            Hx[i,j]-=0.5*(Ez[i,j+1]-Ez[i,j]);Hy[i,j]+=0.5*(Ez[i+1,j]-Ez[i,j])
        for i,j in ti.ndrange((1,N),(1,N)):
            Ez[i,j]+=0.5*(Hy[i,j]-Hy[i-1,j]-Hx[i,j]+Hx[i,j-1])
    run_and_record("FDTD_Maxwell","EM",f"{N}^2",N*N,fdtd,500 if N<=512 else 100)
    ti.reset()

# --- 20. Helmholtz 2D (iterative) ---
section("20. Helmholtz 2D (Jacobi iterative)")
for N in [128, 256, 512]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    uh=ti.field(ti.f32,(N,N));uh2=ti.field(ti.f32,(N,N));fh=ti.field(ti.f32,(N,N))
    @ti.kernel
    def helmholtz_init():
        for i,j in fh: fh[i,j]=1.0 if (i==N//2 and j==N//2) else 0.0
    @ti.kernel
    def helmholtz():
        k2=1.0; h2=1.0/(N*N)
        for i,j in ti.ndrange((1,N-1),(1,N-1)):
            lap=uh[i-1,j]+uh[i+1,j]+uh[i,j-1]+uh[i,j+1]-4*uh[i,j]
            uh2[i,j]=(lap/h2+fh[i,j])/(4.0/h2+k2)
        for i,j in uh: uh[i,j]=uh2[i,j]
    helmholtz_init()
    run_and_record("Helmholtz2D","EM",f"{N}^2",N*N,helmholtz,500)
    ti.reset()

# =====================================================================
# FEM / STRUCTURE (3 types)
# =====================================================================

# --- 21. Explicit FEM (2D triangles) ---
section("21. Explicit FEM 2D")
for N in [32, 64, 128]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    nn=(N+1)*(N+1);ne=N*N*2
    xf=ti.Vector.field(2,ti.f32,nn);vf=ti.Vector.field(2,ti.f32,nn);ff=ti.Vector.field(2,ti.f32,nn)
    el=ti.Vector.field(3,int,ne)
    @ti.kernel
    def fem_init():
        for i,j in ti.ndrange(N+1,N+1): xf[i*(N+1)+j]=[float(i)/N,float(j)/N]
        for i,j in ti.ndrange(N,N):
            n0=i*(N+1)+j;n1=(i+1)*(N+1)+j;n2=i*(N+1)+j+1;n3=(i+1)*(N+1)+j+1
            el[(i*N+j)*2]=[n0,n1,n2]; el[(i*N+j)*2+1]=[n1,n3,n2]
    @ti.kernel
    def fem_step():
        for i in ff: ff[i]=[0.0,-9.8]
        for e in range(ne):
            a,b,c=el[e][0],el[e][1],el[e][2]
            D=ti.Matrix.cols([xf[b]-xf[a],xf[c]-xf[a]]); ar=0.5*abs(D.determinant())
            f_e=ar*ti.Vector([0.0,-0.1])
            ff[a]+=f_e; ff[b]+=f_e; ff[c]+=f_e
        for i in vf: vf[i]+=0.0001*ff[i]; xf[i]+=0.0001*vf[i]
    fem_init()
    run_and_record("ExplicitFEM","FEM",f"{ne}elem",ne,fem_step,200)
    ti.reset()

# --- 22. Cloth (mass-spring) ---
section("22. Cloth Mass-Spring")
for N in [32, 64, 128, 256]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    xc=ti.Vector.field(3,ti.f32,(N,N));vc=ti.Vector.field(3,ti.f32,(N,N))
    @ti.kernel
    def cloth_init():
        for i,j in xc: xc[i,j]=[float(i)/N,1.0,float(j)/N]
    @ti.kernel
    def cloth():
        for i,j in vc:
            vc[i,j][1]-=0.001
            if i>0: vc[i,j]+=(xc[i-1,j]-xc[i,j])*0.5
            if i<N-1: vc[i,j]+=(xc[i+1,j]-xc[i,j])*0.5
            if j>0: vc[i,j]+=(xc[i,j-1]-xc[i,j])*0.5
            if j<N-1: vc[i,j]+=(xc[i,j+1]-xc[i,j])*0.5
        for i,j in xc: xc[i,j]+=vc[i,j]*0.01
    cloth_init()
    run_and_record("Cloth","Structure",f"{N}^2",N*N,cloth,500)
    ti.reset()

# --- 23. Mass-Spring 1D chain ---
section("23. Mass-Spring 1D")
for N in [256, 1024, 4096, 16384]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    xs=ti.field(ti.f32,N);vs_f=ti.field(ti.f32,N)
    @ti.kernel
    def spring_init():
        for i in xs: xs[i]=float(i)*0.01
    @ti.kernel
    def spring_step():
        k=100.0;dt=0.0001
        for i in range(1,N-1):
            f=k*(xs[i+1]-xs[i])-k*(xs[i]-xs[i-1])-0.1*vs_f[i]
            vs_f[i]+=f*dt; xs[i]+=vs_f[i]*dt
    spring_init()
    run_and_record("MassSpring1D","Structure",f"N={N}",N,spring_step,500)
    ti.reset()

# =====================================================================
# TRANSPORT (2 types)
# =====================================================================

# --- 24. Semi-Lagrangian Advection ---
section("24. Semi-Lagrangian Advection")
for N in [128, 256, 512, 1024]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    q=ti.field(ti.f32,(N,N));q2=ti.field(ti.f32,(N,N))
    @ti.kernel
    def advect():
        for i,j in ti.ndrange((1,N-1),(1,N-1)):
            x=float(i)-0.5;y=float(j)-0.3
            x=ti.max(0.5,ti.min(float(N)-1.5,x));y=ti.max(0.5,ti.min(float(N)-1.5,y))
            i0=int(x);j0=int(y);s=x-i0;t=y-j0
            q2[i,j]=(1-s)*(1-t)*q[i0,j0]+s*(1-t)*q[i0+1,j0]+(1-s)*t*q[i0,j0+1]+s*t*q[i0+1,j0+1]
        for i,j in q: q[i,j]=q2[i,j]
    run_and_record("SemiLagrangian","Transport",f"{N}^2",N*N,advect,500)
    ti.reset()

# --- 25. Upwind Advection 1D ---
section("25. Upwind Advection 1D")
for N in [1024, 4096, 16384, 65536]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    ua=ti.field(ti.f32,N);ua2=ti.field(ti.f32,N)
    @ti.kernel
    def upwind_init():
        for i in ua: ua[i]=1.0 if (N//4<i<N//2) else 0.0
    @ti.kernel
    def upwind():
        c=0.5;dt=0.001
        for i in ti.ndrange((1,N)):
            ua2[i]=ua[i]-c*dt*(ua[i]-ua[i-1])
        for i in ua: ua[i]=ua2[i]
    upwind_init()
    run_and_record("Upwind1D","Transport",f"N={N}",N,upwind,500)
    ti.reset()

# =====================================================================
# PDE MISC (3 types)
# =====================================================================

# --- 26. Poisson (Jacobi iterative) ---
section("26. Poisson 2D (Jacobi)")
for N in [128, 256, 512, 1024]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    up=ti.field(ti.f32,(N,N));up2=ti.field(ti.f32,(N,N));fp=ti.field(ti.f32,(N,N))
    @ti.kernel
    def poisson_init():
        for i,j in fp: fp[i,j]=1.0 if (i==N//2 and j==N//2) else 0.0
    @ti.kernel
    def poisson():
        h2=1.0/(N*N)
        for i,j in ti.ndrange((1,N-1),(1,N-1)):
            up2[i,j]=0.25*(up[i-1,j]+up[i+1,j]+up[i,j-1]+up[i,j+1]-h2*fp[i,j])
        for i,j in up: up[i,j]=up2[i,j]
    poisson_init()
    run_and_record("Poisson2D","PDE",f"{N}^2",N*N,poisson,500)
    ti.reset()

# --- 27. Schrodinger (split-step, real part) ---
section("27. Schrodinger 1D (FTCS)")
for N in [1024, 4096, 16384]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    re=ti.field(ti.f32,N);im=ti.field(ti.f32,N)
    re2=ti.field(ti.f32,N);im2=ti.field(ti.f32,N)
    @ti.kernel
    def schrodinger_init():
        for i in re:
            x=float(i)/N-0.5
            re[i]=ti.exp(-50*x*x)*ti.cos(20*x)
            im[i]=ti.exp(-50*x*x)*ti.sin(20*x)
    @ti.kernel
    def schrodinger():
        dt=0.00001;dx2=1.0/(N*N)
        for i in ti.ndrange((1,N-1)):
            lap_re=re[i-1]+re[i+1]-2*re[i]
            lap_im=im[i-1]+im[i+1]-2*im[i]
            re2[i]=re[i]+dt*0.5*lap_im/dx2
            im2[i]=im[i]-dt*0.5*lap_re/dx2
        for i in re: re[i]=re2[i]; im[i]=im2[i]
    schrodinger_init()
    run_and_record("Schrodinger1D","PDE",f"N={N}",N,schrodinger,500)
    ti.reset()

# --- 28. Kuramoto-Sivashinsky ---
section("28. Kuramoto-Sivashinsky 1D")
for N in [256, 1024, 4096]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    uk=ti.field(ti.f32,N);uk2=ti.field(ti.f32,N)
    @ti.kernel
    def ks_init():
        for i in uk: uk[i]=ti.cos(2*3.14159*float(i)/N)*(1+0.1*ti.sin(4*3.14159*float(i)/N))
    @ti.kernel
    def ks_step():
        dt=0.01; dx=float(N)/100
        for i in ti.ndrange((2,N-2)):
            u=uk[i]
            ux=(uk[i+1]-uk[i-1])/(2*dx)
            uxx=(uk[i+1]-2*u+uk[i-1])/(dx*dx)
            uxxxx=(uk[i+2]-4*uk[i+1]+6*u-4*uk[i-1]+uk[i-2])/(dx*dx*dx*dx)
            uk2[i]=u+dt*(-u*ux-uxx-uxxxx)
        for i in uk: uk[i]=uk2[i]
    ks_init()
    run_and_record("KuramotoSivash","PDE",f"N={N}",N,ks_step,500)
    ti.reset()

# =====================================================================
# OTHER (2 types)
# =====================================================================

# --- 29. Reduction ---
section("29. Reduction (sum)")
for N in [1024, 16384, 262144, 1048576]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    arr=ti.field(ti.f32,N); result=ti.field(ti.f32,())
    @ti.kernel
    def red_init():
        for i in arr: arr[i]=1.0
    @ti.kernel
    def reduce_sum():
        s=0.0
        for i in arr: s+=arr[i]
        result[None]=s
    red_init()
    run_and_record("Reduction","Other",f"N={N}",N,reduce_sum,500)
    ti.reset()

# --- 30. Monte Carlo (random walk diffusion) ---
section("30. Monte Carlo Random Walk")
for N in [1024, 4096, 16384]:
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    xmc=ti.field(ti.f32,N);ymc=ti.field(ti.f32,N)
    @ti.kernel
    def mc_init():
        for i in xmc: xmc[i]=0.0; ymc[i]=0.0
    @ti.kernel
    def mc_step():
        for i in range(N):
            xmc[i]+=(ti.random()-0.5)*0.1
            ymc[i]+=(ti.random()-0.5)*0.1
    mc_init()
    run_and_record("MonteCarlo_RW","Other",f"N={N}",N,mc_step,500)
    ti.reset()

# =====================================================================
# SUMMARY
# =====================================================================
print(f"\n{'='*80}")
print(f"  SUMMARY: {len(results)} configurations across 30 kernel types")
print(f"{'='*80}")
print(f"{'Kernel':<30} {'Domain':<12} {'Size':>10} {'us/step':>10} {'Class':>10}")
print("-"*75)

oh_dom=0;compute_dom=0;trans=0
for name,domain,size,nelem,us in results:
    oh=min(PYTHON_OH/us*100,100) if us>0 else 0
    cls="OH-DOM" if oh>50 else "COMPUTE" if oh<20 else "TRANS"
    if oh>50: oh_dom+=1
    elif oh<20: compute_dom+=1
    else: trans+=1
    print(f"{name:<30} {domain:<12} {size:>10} {us:>9.1f} {cls:>10}")

print(f"\nClassification: OH-dominated={oh_dom}  Transitional={trans}  Compute-dominated={compute_dom}")
print(f"Total: {len(results)} configs across 30 kernel types")
print(f"\nKernel types by domain:")
print(f"  Stencil:   Heat2D/3D, Wave2D, Jacobi, GrayScott, AllenCahn, CahnHilliard, Burgers, ConvDiff (9)")
print(f"  CFD:       SWE, LBM, StableFluids, Euler1D (4)")
print(f"  Particle:  NBody, SPH, DEM, MD_LJ, PIC (5)")
print(f"  EM:        FDTD, Helmholtz (2)")
print(f"  FEM/Struc: ExplicitFEM, Cloth, MassSpring1D (3)")
print(f"  Transport: SemiLagrangian, Upwind1D (2)")
print(f"  PDE:       Poisson, Schrodinger, KuramotoSivashinsky (3)")
print(f"  Other:     Reduction, MonteCarlo (2)")
