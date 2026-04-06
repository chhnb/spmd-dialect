"""
F1 Hydro SWE: Cross-framework comparison with complete OSHER solver.
Runs Taichi, Warp, Kokkos on same structured mesh (dam-break) at multiple sizes.

FIXED: properly separate JIT compilation from benchmark timing.

Usage:
  cd benchmark/F1_hydro_shallow_water
  python run_all_frameworks.py

For CUDA (Sync/Async/Graph/Persistent), compile and run separately:
  nvcc -O3 -arch=sm_86 -rdc=true hydro_cuda_osher.cu -o hydro_cuda_osher -lcudadevrt
  ./hydro_cuda_osher
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('TI_LOG_LEVEL', 'warn')

SIZES = [32, 64, 128]
WARMUP = 20
STEPS = 500  # timed steps

print("=" * 80)
print("F1 Hydro SWE — Cross-Framework Comparison (Complete OSHER Solver)")
print(f"Warmup={WARMUP}, Timed steps={STEPS}")
print("=" * 80)

results = {}

# --- Taichi ---
print("\n--- Taichi (CUDA, fp64) ---")
try:
    import taichi as ti
    from hydro_taichi import run as taichi_run
    for N in SIZES:
        # run() returns (step_fn, sync_fn, H_field)
        # step_fn() internally loops `steps` times
        # Use steps=1 so each call = 1 timestep
        step_fn, sync_fn, _ = taichi_run(N, steps=1, backend="cuda")

        # Warmup (JIT compile happens on first call, amortized here)
        for _ in range(WARMUP):
            step_fn()
        sync_fn()

        # Benchmark
        sync_fn()
        t0 = time.perf_counter()
        for _ in range(STEPS):
            step_fn()
        sync_fn()
        elapsed = time.perf_counter() - t0
        us = elapsed * 1e6 / STEPS
        results[('Taichi', N)] = us
        print(f"  N={N:>4} ({N*N:>6} cells): {us:>8.1f} μs/step")
        ti.reset()
except Exception as e:
    print(f"  Taichi failed: {e}")

# --- Warp ---
print("\n--- Warp (CUDA, fp64) ---")
try:
    import warp as wp
    from hydro_warp import run as warp_run
    for N in SIZES:
        step_fn, sync_fn, _ = warp_run(N, steps=1, backend="cuda")

        for _ in range(WARMUP):
            step_fn()
        sync_fn()

        sync_fn()
        t0 = time.perf_counter()
        for _ in range(STEPS):
            step_fn()
        sync_fn()
        elapsed = time.perf_counter() - t0
        us = elapsed * 1e6 / STEPS
        results[('Warp', N)] = us
        print(f"  N={N:>4} ({N*N:>6} cells): {us:>8.1f} μs/step")
except Exception as e:
    print(f"  Warp failed: {e}")

# --- Kokkos (if binary exists) ---
print("\n--- Kokkos (CUDA) ---")
kokkos_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "cpp", "kokkos", "build-cuda", "hydro_swe_kokkos")
if not os.path.exists(kokkos_bin):
    # Try alternative paths
    for alt in ["hydro_swe_kokkos", "../cpp/kokkos/hydro_swe_kokkos",
                os.path.expanduser("~/spmd-dialect/benchmark/cpp/kokkos/build-cuda/hydro_swe_kokkos")]:
        if os.path.exists(alt):
            kokkos_bin = alt
            break

if os.path.exists(kokkos_bin):
    import subprocess
    for N in SIZES:
        try:
            # Kokkos binary typically takes: N steps repeat
            result = subprocess.run(
                [kokkos_bin, str(N), str(STEPS), "1"],
                capture_output=True, text=True, timeout=60
            )
            print(f"  N={N:>4}: {result.stdout.strip()}")
        except Exception as e:
            print(f"  N={N:>4}: Kokkos failed: {e}")
else:
    print(f"  Kokkos binary not found. Build with:")
    print(f"    cd benchmark/cpp/kokkos && cmake --build build-cuda --target hydro_swe_kokkos")
    print(f"  Or specify path to binary.")

# --- NumPy (CPU baseline, small sizes only) ---
print("\n--- NumPy (CPU) ---")
try:
    from hydro_numpy import run as numpy_run
    for N in SIZES:
        if N > 64:
            print(f"  N={N:>4}: skipped (CPU too slow)")
            continue
        step_fn, sync_fn, _ = numpy_run(N, steps=1, backend="cpu")
        # Warmup
        for _ in range(3):
            step_fn()
        t0 = time.perf_counter()
        n_steps = 50
        for _ in range(n_steps):
            step_fn()
        elapsed = time.perf_counter() - t0
        us = elapsed * 1e6 / n_steps
        results[('NumPy', N)] = us
        print(f"  N={N:>4} ({N*N:>6} cells): {us:>8.1f} μs/step")
except Exception as e:
    print(f"  NumPy failed: {e}")

# --- Summary table ---
print(f"\n{'='*80}")
print(f"  SUMMARY (μs/step, RTX 3060, complete OSHER fp64)")
print(f"{'='*80}")
print(f"{'Framework':<20} {'32²':>10} {'64²':>10} {'128²':>10}")
print("-" * 55)
for fw in ['Taichi', 'Warp', 'NumPy']:
    vals = []
    for N in SIZES:
        v = results.get((fw, N))
        vals.append(f"{v:>9.1f}" if v else f"{'—':>9}")
    print(f"{fw:<20} {''.join(vals)}")

print(f"\nFor CUDA (Sync/Async/Graph/Persistent), run: ./hydro_cuda_osher")
print(f"For Kokkos, build and run: ./hydro_swe_kokkos N STEPS REPEAT")
