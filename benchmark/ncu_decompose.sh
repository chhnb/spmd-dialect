#!/bin/bash
# NCU Overhead Decomposition: measure pure kernel body time for each strategy
# Then compare with wall-clock timing to extract launch/sync/barrier overhead
export PATH=/home/scratch.huanhuanc_gpu/spmd/cuda-toolkit/bin:$PATH
export LD_LIBRARY_PATH=/home/scratch.huanhuanc_gpu/spmd/cuda-toolkit/lib64:$LD_LIBRARY_PATH

METRICS="gpu__time_duration.sum,launch__waves_per_multiprocessor,sm__warps_active.avg.pct_of_peak_sustained_active"
OUT="results/ncu_overhead_decomposition.txt"

echo "=== NCU Overhead Decomposition (A100, sm_80) ===" | tee "$OUT"
echo "Date: $(date)" | tee -a "$OUT"
nvidia-smi --query-gpu=name,compute_cap --format=csv | tee -a "$OUT"
echo "" | tee -a "$OUT"

###############################################################################
# 1. Jacobi2D: simple 2-kernel/step (jacobi_step + copy_kernel)
###############################################################################
echo "=== C1: Jacobi2D ===" | tee -a "$OUT"
for N in 256 1024 4096; do
    echo "--- N=$N ---" | tee -a "$OUT"

    # Single jacobi_step body time
    echo "[jacobi_step body]" | tee -a "$OUT"
    ncu --metrics $METRICS --kernel-name jacobi_step --launch-skip 5 --launch-count 3 \
        ./jacobi2d_bench $N 10 3 2>&1 | grep "gpu__time_duration" | tee -a "$OUT"

    # Single copy_kernel body time
    echo "[copy_kernel body]" | tee -a "$OUT"
    ncu --metrics $METRICS --kernel-name copy_kernel --launch-skip 5 --launch-count 3 \
        ./jacobi2d_bench $N 10 3 2>&1 | grep "gpu__time_duration" | tee -a "$OUT"

    # Persistent kernel total (10 steps fused)
    echo "[persistent total (10 steps)]" | tee -a "$OUT"
    ncu --metrics $METRICS --kernel-name jacobi_persistent --launch-skip 0 --launch-count 1 \
        ./jacobi2d_bench $N 10 3 2>&1 | grep "gpu__time_duration" | tee -a "$OUT"

    # Grid-stride persistent total
    echo "[grid-stride persistent total (10 steps)]" | tee -a "$OUT"
    ncu --metrics $METRICS --kernel-name jacobi_persistent_stride --launch-skip 0 --launch-count 1 \
        ./jacobi2d_bench $N 10 3 2>&1 | grep "gpu__time_duration" | tee -a "$OUT"

    # Wall clock timing
    echo "[wall-clock timing]" | tee -a "$OUT"
    ./jacobi2d_bench $N 10 5 2>&1 | grep -E "^\[" | tee -a "$OUT"
    echo "" | tee -a "$OUT"
done

###############################################################################
# 2. HydroF2 OSHER: real-world 2-kernel/step
###############################################################################
echo "=== F2: HydroF2 OSHER (CELL=24020) ===" | tee -a "$OUT"

echo "[calculate_flux body]" | tee -a "$OUT"
ncu --metrics $METRICS --kernel-name calculate_flux --launch-skip 5 --launch-count 3 \
    ./hydro_osher_a100 100 3 2>&1 | grep "gpu__time_duration" | tee -a "$OUT"

echo "[update_cell body]" | tee -a "$OUT"
ncu --metrics $METRICS --kernel-name update_cell --launch-skip 5 --launch-count 3 \
    ./hydro_osher_a100 100 3 2>&1 | grep "gpu__time_duration" | tee -a "$OUT"

echo "[persistent_fused total (100 steps)]" | tee -a "$OUT"
ncu --metrics $METRICS --kernel-name persistent_fused --launch-skip 0 --launch-count 1 \
    ./hydro_osher_a100 100 3 2>&1 | grep "gpu__time_duration" | tee -a "$OUT"

echo "[wall-clock timing]" | tee -a "$OUT"
./hydro_osher_a100 100 5 2>&1 | grep -E "^\[|Flux kernel|Update kernel|GPU total" | tee -a "$OUT"
echo "" | tee -a "$OUT"

###############################################################################
# 3. Wave2D: simple 1-kernel/step
###############################################################################
echo "=== C4: Wave2D ===" | tee -a "$OUT"
for N in 512 4096; do
    echo "--- N=$N ---" | tee -a "$OUT"
    echo "[wave_step body]" | tee -a "$OUT"
    ncu --metrics $METRICS --kernel-name wave_step --launch-skip 5 --launch-count 3 \
        ./wave2d_bench $N 100 3 2>&1 | grep "gpu__time_duration" | tee -a "$OUT"
    echo "[wall-clock]" | tee -a "$OUT"
    ./wave2d_bench $N 100 3 2>&1 | grep -E "^\[" | tee -a "$OUT"
    echo "" | tee -a "$OUT"
done

###############################################################################
# 4. LBM D2Q9: medium compute kernel
###############################################################################
echo "=== C5: LBM D2Q9 ===" | tee -a "$OUT"
for SIZE in "512 256" "2048 1024"; do
    echo "--- $SIZE ---" | tee -a "$OUT"
    echo "[lbm_stream_collide body]" | tee -a "$OUT"
    ncu --metrics $METRICS --kernel-name lbm --launch-skip 5 --launch-count 3 \
        ./lbm2d_bench $SIZE 100 3 2>&1 | grep "gpu__time_duration" | tee -a "$OUT"
    echo "[wall-clock]" | tee -a "$OUT"
    ./lbm2d_bench $SIZE 100 3 2>&1 | grep -E "^\[" | tee -a "$OUT"
    echo "" | tee -a "$OUT"
done

###############################################################################
# 5. Heat2D + GrayScott from overhead_solutions
###############################################################################
echo "=== C3/C10: Heat2D + GrayScott ===" | tee -a "$OUT"
echo "[wall-clock]" | tee -a "$OUT"
./overhead_solutions_a100 2>&1 | head -40 | tee -a "$OUT"
echo "" | tee -a "$OUT"

echo "=== DONE ===" | tee -a "$OUT"
