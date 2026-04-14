"""Test whether Taichi kernel calls are sync or async from Python."""
import taichi as ti
import time

ti.init(arch=ti.cuda, default_fp=ti.f32)

N = 4096
u = ti.field(ti.f32, shape=(N, N))
v = ti.field(ti.f32, shape=(N, N))

@ti.kernel
def stencil():
    for i, j in u:
        if i >= 1 and i < N-1 and j >= 1 and j < N-1:
            v[i,j] = 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])

@ti.kernel
def trivial():
    for i, j in u:
        pass  # almost no work

# Warmup
for _ in range(10):
    stencil()
    trivial()
ti.sync()

print(f"N={N}, stencil on {N}x{N} grid")
print()

# Test 1: single stencil call WITHOUT sync — does Python block?
times = []
for _ in range(20):
    ti.sync()
    t0 = time.perf_counter()
    stencil()
    t1 = time.perf_counter()  # no ti.sync() — is Python already back?
    times.append((t1-t0)*1e6)
times.sort()
print(f"stencil() no sync:  {times[10]:.1f} us  (if async, this = Python dispatch only)")

# Test 2: single stencil call WITH sync
times = []
for _ in range(20):
    ti.sync()
    t0 = time.perf_counter()
    stencil()
    ti.sync()
    t1 = time.perf_counter()
    times.append((t1-t0)*1e6)
times.sort()
print(f"stencil() + sync:   {times[10]:.1f} us  (dispatch + GPU compute + sync)")

# Test 3: trivial kernel WITHOUT sync
times = []
for _ in range(20):
    ti.sync()
    t0 = time.perf_counter()
    trivial()
    t1 = time.perf_counter()
    times.append((t1-t0)*1e6)
times.sort()
print(f"trivial() no sync:  {times[10]:.1f} us  (if async, should be same as stencil no-sync)")

# Test 4: 100 stencils back-to-back, then sync
ti.sync()
t0 = time.perf_counter()
for _ in range(100):
    stencil()
t1 = time.perf_counter()
ti.sync()
t2 = time.perf_counter()
print()
print(f"100x stencil() launch time:  {(t1-t0)*1e6:.1f} us  ({(t1-t0)*1e6/100:.1f} us/call)")
print(f"100x stencil() + final sync: {(t2-t0)*1e6:.1f} us  ({(t2-t0)*1e6/100:.1f} us/call)")
print()

# Interpretation
launch_only = (t1-t0)*1e6/100
with_sync = (t2-t0)*1e6/100
if with_sync > launch_only * 1.5:
    print(f"ASYNC: launch={launch_only:.1f} us/call, total={with_sync:.1f} us/call")
    print("  → GPU work happens AFTER Python returns (async dispatch)")
else:
    print(f"SYNC: launch={launch_only:.1f} us/call, total={with_sync:.1f} us/call")
    print("  → Python blocks until GPU finishes (sync dispatch)")
