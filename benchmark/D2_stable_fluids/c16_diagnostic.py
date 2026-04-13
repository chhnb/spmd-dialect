"""C16 Stable Fluids phase-by-phase divergence diagnostic.
Runs Taichi CUDA, Taichi CPU, and Warp step-by-step, comparing
output after each phase (advect, divergence, jacobi, project)."""
import subprocess, tempfile, numpy as np, sys, os
PYTHON = "/home/scratch.huanhuanc_gpu/spmd/spmd-venv/bin/python"
BD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_phase(backend, framework, phase_stop, N=64, steps=1):
    """Run C16 stopping after a specific phase and return output array."""
    npy = tempfile.mktemp(suffix=".npy")
    if framework == "taichi":
        code = f"""
import sys; sys.path.insert(0,"D2_stable_fluids"); sys.path.insert(0,".")
import taichi as ti
ti.init(arch=ti.{"cuda" if backend=="cuda" else "cpu"}, default_fp=ti.f32)
N={N}; dt=0.03
u=ti.Vector.field(2,dtype=ti.f32,shape=(N,N))
u_new=ti.Vector.field(2,dtype=ti.f32,shape=(N,N))
p=ti.field(dtype=ti.f32,shape=(N,N))
p_new=ti.field(dtype=ti.f32,shape=(N,N))
div=ti.field(dtype=ti.f32,shape=(N,N))
@ti.func
def sample_v(field:ti.template(),x:float,y:float)->ti.math.vec2:
    i=ti.max(0,ti.min(ti.cast(x,ti.i32),N-1))
    j=ti.max(0,ti.min(ti.cast(y,ti.i32),N-1))
    return field[i,j]
@ti.kernel
def advect():
    for i,j in u:
        coord=ti.Vector([float(i),float(j)])-dt*N*u[i,j]
        u_new[i,j]=sample_v(u,coord[0],coord[1])
@ti.kernel
def divergence_step():
    for i,j in u_new:
        vl=sample_v(u_new,float(i-1),float(j))[0]
        vr=sample_v(u_new,float(i+1),float(j))[0]
        vb=sample_v(u_new,float(i),float(j-1))[1]
        vt=sample_v(u_new,float(i),float(j+1))[1]
        div[i,j]=0.5*(vr-vl+vt-vb)
@ti.kernel
def jacobi_step():
    for i,j in p:
        il=ti.max(i-1,0);ir=ti.min(i+1,N-1)
        jl=ti.max(j-1,0);jr=ti.min(j+1,N-1)
        p_new[i,j]=0.25*(p[il,j]+p[ir,j]+p[i,jl]+p[i,jr]-div[i,j])
@ti.kernel
def copy_p():
    for i,j in p: p[i,j]=p_new[i,j]
@ti.kernel
def project():
    for i,j in u_new:
        il=ti.max(i-1,0);ir=ti.min(i+1,N-1)
        jl=ti.max(j-1,0);jr=ti.min(j+1,N-1)
        u[i,j]=u_new[i,j]-0.5*ti.Vector([p[ir,j]-p[il,j],p[i,jr]-p[i,jl]])
@ti.kernel
def init():
    for i,j in u:
        cx,cy=N*0.5,N*0.5
        u[i,j]=ti.Vector([-(float(j)-cy)*0.01,(float(i)-cx)*0.01])
init()
for _ in range({steps}):
    advect(); ti.sync()
    if "{phase_stop}"=="advect": break
    divergence_step(); ti.sync()
    if "{phase_stop}"=="divergence": break
    for _j in range(50):
        jacobi_step(); copy_p()
    ti.sync()
    if "{phase_stop}"=="jacobi": break
    project(); ti.sync()
ti.sync()
import numpy as np
a=u.to_numpy().flatten().astype(np.float64)
np.save("{npy}",a)
print(f"n={{len(a)}}")
"""
    else:  # warp
        code = f"""
import sys,os; os.environ["WARP_CACHE_PATH"]="/home/scratch.huanhuanc_gpu/spmd/.warp_cache"
sys.path.insert(0,"D2_stable_fluids"); sys.path.insert(0,".")
import warp as wp; wp.init(); import numpy as np
from fluid_warp import FluidMesh, advect_kernel, divergence_kernel, jacobi_kernel, copy_kernel, project_kernel, clamp_idx
N={N}; dt=0.03; cx,cy=N*0.5,N*0.5
u_np=np.zeros((N,N,2),dtype=np.float32)
for i in range(N):
    for j in range(N):
        u_np[i,j,0]=-(j-cy)*0.01; u_np[i,j,1]=(i-cx)*0.01
mesh=FluidMesh()
mesh.u=wp.array(u_np,dtype=wp.vec2,device="cuda")
mesh.u_new=wp.zeros((N,N),dtype=wp.vec2,device="cuda")
mesh.p=wp.zeros((N,N),dtype=float,device="cuda")
mesh.p_new=wp.zeros((N,N),dtype=float,device="cuda")
mesh.div=wp.zeros((N,N),dtype=float,device="cuda")
mesh.N=N; mesh.dt=dt
dim=(N,N)
for _ in range({steps}):
    wp.launch(advect_kernel,dim=dim,inputs=[mesh],device="cuda"); wp.synchronize()
    if "{phase_stop}"=="advect": break
    wp.launch(divergence_kernel,dim=dim,inputs=[mesh],device="cuda"); wp.synchronize()
    if "{phase_stop}"=="divergence": break
    for _j in range(50):
        wp.launch(jacobi_kernel,dim=dim,inputs=[mesh],device="cuda")
        wp.launch(copy_kernel,dim=dim,inputs=[mesh],device="cuda")
    wp.synchronize()
    if "{phase_stop}"=="jacobi": break
    wp.launch(project_kernel,dim=dim,inputs=[mesh],device="cuda"); wp.synchronize()
wp.synchronize()
a=mesh.u.numpy().flatten().astype(np.float64)
np.save("{npy}",a)
print(f"n={{len(a)}}")
"""
    r = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True, timeout=120, cwd=BD)
    if os.path.exists(npy):
        return np.load(npy)
    return None

if __name__ == "__main__":
    for phase in ["advect", "divergence", "jacobi", "project"]:
        tc = run_phase("cuda", "taichi", phase)
        tw = run_phase("cuda", "warp", phase)
        if tc is not None and tw is not None and tc.shape == tw.shape:
            rel = np.abs(tc - tw) / (np.maximum(np.abs(tc), np.abs(tw)) + 1e-12)
            print(f"Phase {phase:12s}: max_rel={rel.max():.6e}  {'OK' if rel.max()<=0.05 else 'DIVERGED'}")
            if rel.max() > 0.05:
                idx = np.argmax(rel)
                print(f"  First mismatch at flat index {idx}: taichi={tc[idx]:.10f} warp={tw[idx]:.10f}")
                break
        else:
            print(f"Phase {phase:12s}: ERROR (missing output)")
