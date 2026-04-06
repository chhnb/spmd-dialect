"""
F1 Hydro SWE: Cross-framework comparison with complete OSHER solver.
Runs Taichi, Warp on same structured mesh (dam-break) at multiple sizes.
Sizes chosen so persistent kernel works on RTX 3060 (≤120 blocks).

Usage:
  python benchmark/F1_hydro_shallow_water/run_all_frameworks.py

For CUDA comparison (Sync/Async/Graph/Persistent), compile and run:
  nvcc -O3 -arch=sm_86 -rdc=true benchmark/F1_hydro_shallow_water/hydro_cuda_osher.cu \
       -o hydro_cuda_osher -lcudadevrt
  ./hydro_cuda_osher
"""
import subprocess
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

SIZES = [32, 64, 128]  # N values; 128² = 16384 cells = 64 blocks (fits 3060 persistent)
STEPS = 200

print("=" * 80)
print("F1 Hydro SWE — Cross-Framework Comparison (Complete OSHER Solver)")
print("=" * 80)

# --- Taichi ---
print("\n--- Taichi (CUDA) ---")
try:
    import taichi as ti
    from hydro_taichi import run as run_taichi
    for N in SIZES:
        # Warmup
        run_taichi(N, steps=5, backend="cuda")
        ti.sync()
        # Benchmark
        t0 = time.perf_counter()
        run_taichi(N, steps=STEPS, backend="cuda")
        ti.sync()
        elapsed = time.perf_counter() - t0
        us_step = elapsed * 1e6 / STEPS
        print(f"  N={N:>4} ({N*N:>6} cells): {us_step:>8.1f} μs/step  ({elapsed*1000:.1f} ms total)")
        ti.reset()
except Exception as e:
    print(f"  Taichi failed: {e}")

# --- Warp ---
print("\n--- Warp (CUDA) ---")
try:
    import warp as wp
    from hydro_warp import run as run_warp
    for N in SIZES:
        run_warp(N, steps=5, backend="cuda")
        wp.synchronize()
        t0 = time.perf_counter()
        run_warp(N, steps=STEPS, backend="cuda")
        wp.synchronize()
        elapsed = time.perf_counter() - t0
        us_step = elapsed * 1e6 / STEPS
        print(f"  N={N:>4} ({N*N:>6} cells): {us_step:>8.1f} μs/step  ({elapsed*1000:.1f} ms total)")
except Exception as e:
    print(f"  Warp failed: {e}")

# --- NumPy (CPU baseline) ---
print("\n--- NumPy (CPU) ---")
try:
    from hydro_numpy import run as run_numpy
    for N in SIZES:
        if N > 64:
            print(f"  N={N:>4}: skipped (too slow for CPU)")
            continue
        t0 = time.perf_counter()
        run_numpy(N, steps=min(STEPS, 50), backend="cpu")
        elapsed = time.perf_counter() - t0
        us_step = elapsed * 1e6 / min(STEPS, 50)
        print(f"  N={N:>4} ({N*N:>6} cells): {us_step:>8.1f} μs/step  ({elapsed*1000:.1f} ms total)")
except Exception as e:
    print(f"  NumPy failed: {e}")

print("\n--- Summary ---")
print("For CUDA (Sync/Async/Graph/Persistent) comparison, run:")
print("  ./hydro_cuda_osher")
print("\nKey: all frameworks use the SAME complete OSHER Riemann solver")
print("     on the SAME structured dam-break mesh.")
