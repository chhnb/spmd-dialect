#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

mkdir -p benchmark/results

nvcc -O3 -arch=sm_86 -rdc=true benchmark/pk_matrix_benchmark.cu -o benchmark/pk_matrix_prof -lcudadevrt
nvcc -O3 -arch=sm_86 -rdc=true benchmark/F1_hydro_shallow_water/hydro_cuda_osher.cu -o benchmark/hydro_cuda_osher_prof -lcudadevrt

ncu --csv \
  --kernel-name regex:k_heat2d \
  --launch-count 1 \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__bytes_write.sum,smsp__sass_thread_inst_executed_op_fp32_pred_on.sum \
  env PK_KERNEL=heat2d PK_N=128 PK_B=256 PK_STRATEGY=graph ./benchmark/pk_matrix_prof \
  > benchmark/results/ncu_heat2d_128.csv

ncu --csv \
  --kernel-name regex:persistent_kernel \
  --launch-count 1 \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__bytes_write.sum,smsp__sass_thread_inst_executed_op_fp64_pred_on.sum \
  ./benchmark/hydro_cuda_osher_prof --n 64 --steps 10 --mode persistent \
  > benchmark/results/ncu_osher_64_persistent.csv

ncu --csv \
  --kernel-name regex:shallow_water_step \
  --launch-count 1 \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__bytes_write.sum,smsp__sass_thread_inst_executed_op_fp64_pred_on.sum \
  ./benchmark/hydro_cuda_osher_prof --n 128 --steps 10 --mode graph \
  > benchmark/results/ncu_osher_128_graph.csv

./benchmark/summarize_ncu_metrics.py
