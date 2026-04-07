# 3060 执行指令 v2

## 概览

3 组实验，按优先级排序。所有命令都在 `spmd-dialect/` 目录下执行。

---

## 实验 A: 完整 OSHER Solver 4 策略对比 [最重要]

这是证明"即使重 kernel 也有显著 overhead"的核心数据。

```bash
# 编译 (arch=sm_86 for 3060)
nvcc -O3 -arch=sm_86 -rdc=true \
  benchmark/F1_hydro_shallow_water/hydro_cuda_osher.cu \
  -o hydro_cuda_osher -lcudadevrt

# 跑多个 grid size
for N in 32 64 128 256; do
  echo "=== OSHER N=$N ==="
  ./hydro_cuda_osher --n $N --steps 500 --mode all
done 2>&1 | tee benchmark/results/3060_osher_all.txt

# 如果支持 persistent (小 grid 才行):
./hydro_cuda_osher --n 64 --steps 500 --mode persistent 2>&1 | tee -a benchmark/results/3060_osher_all.txt
```

**看什么**: 3060 上完整 OSHER 的 overhead 占比。B200 上是 56%，3060 可能更高。

---

## 实验 B: NCU Profiling 更多 kernel [重要 — cost model 需要]

当前只有 heat2d 和 osher 的 NCU 数据。需要覆盖更多 kernel 来拟合 cost model。

```bash
# 先确认 ncu 可用
ncu --version

# 编译 pk_matrix_benchmark
nvcc -O3 -arch=sm_86 -rdc=true benchmark/pk_matrix_benchmark.cu \
  -o pk_matrix_prof -lcudadevrt

# 编译 OSHER
nvcc -O3 -arch=sm_86 -rdc=true \
  benchmark/F1_hydro_shallow_water/hydro_cuda_osher.cu \
  -o hydro_cuda_osher_prof -lcudadevrt

# NCU metrics 定义
METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
dram__bytes_read.sum,dram__bytes_write.sum,\
launch__registers_per_thread,\
smsp__sass_thread_inst_executed_op_fp32_pred_on.sum"

METRICS_FP64="sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
dram__bytes_read.sum,dram__bytes_write.sum,\
launch__registers_per_thread,\
smsp__sass_thread_inst_executed_op_fp64_pred_on.sum"

# --- Heat2D at multiple sizes ---
for N in 64 128 256 512; do
  echo "=== NCU heat2d N=$N ==="
  ncu --csv --kernel-name regex:k_heat2d --launch-count 1 \
    --metrics $METRICS \
    env PK_KERNEL=heat2d PK_N=$N PK_B=256 PK_STRATEGY=graph ./pk_matrix_prof \
    > benchmark/results/ncu_heat2d_${N}.csv
done

# --- HotSpot at multiple sizes ---
for N in 128 256 512; do
  echo "=== NCU hotspot N=$N ==="
  ncu --csv --kernel-name regex:k_hotspot --launch-count 1 \
    --metrics $METRICS \
    env PK_KERNEL=hotspot PK_N=$N PK_B=256 PK_STRATEGY=graph ./pk_matrix_prof \
    > benchmark/results/ncu_hotspot_${N}.csv
done

# --- Jacobi2D ---
for N in 128 256 512; do
  echo "=== NCU jacobi2d N=$N ==="
  ncu --csv --kernel-name regex:k_jacobi --launch-count 1 \
    --metrics $METRICS \
    env PK_KERNEL=jacobi2d PK_N=$N PK_B=256 PK_STRATEGY=graph ./pk_matrix_prof \
    > benchmark/results/ncu_jacobi2d_${N}.csv
done

# --- SRAD ---
for N in 128 256 512; do
  echo "=== NCU srad N=$N ==="
  ncu --csv --kernel-name regex:k_srad --launch-count 1 \
    --metrics $METRICS \
    env PK_KERNEL=srad PK_N=$N PK_B=256 PK_STRATEGY=graph ./pk_matrix_prof \
    > benchmark/results/ncu_srad_${N}.csv
done

# --- OSHER at multiple sizes (fp64 metrics) ---
for N in 32 64 128; do
  echo "=== NCU osher N=$N ==="
  ncu --csv --kernel-name regex:shallow_water_step --launch-count 1 \
    --metrics $METRICS_FP64 \
    ./hydro_cuda_osher_prof --n $N --steps 5 --mode graph \
    > benchmark/results/ncu_osher_${N}_graph.csv
done

echo "=== All NCU profiling done ==="
```

**看什么**: 每个 kernel 的寄存器数、SM throughput、occupancy、DRAM bytes。这些是 cost model 的输入特征。

---

## 实验 C: 补充 pk_matrix 数据 [如果实验 A/B 跑完还有时间]

如果 `pk_matrix_benchmark.cu` 支持更多 kernel（检查代码），补充跑一些没覆盖的配置：

```bash
# 查看支持哪些 kernel
grep -o 'PK_KERNEL.*==.*"[a-z]*"' benchmark/pk_matrix_benchmark.cu

# 跑全量 matrix (已有 3060_pk_matrix.csv，跑这个只是更新)
./pk_matrix_prof 2>&1 | tee benchmark/results/3060_pk_matrix_v2.csv
```

---

## 实验完成后

```bash
# 把所有结果 commit + push
cd spmd-dialect
git add benchmark/results/3060_osher_all.txt benchmark/results/ncu_*.csv
git commit -m "3060: OSHER full solver + NCU profiling for 5 kernel types"
git push origin main
```

---

## 各实验的预期结果

### 实验 A 预期 (OSHER on 3060):
```
N=64 (4096 cells, 16 blocks):
  Sync: ~80-100 μs   Graph: ~20-30 μs   Persistent: ~15-25 μs (可能可以跑)
  OH: 60-70%

N=128 (16384 cells, 64 blocks):
  Sync: ~150-200 μs   Graph: ~80-120 μs   Persistent: 可能超限
  OH: 40-50%

N=256 (65536 cells, 256 blocks):
  Sync: ~500+ μs   Graph: ~400+ μs   Persistent: N/A
  OH: <20% (compute dominated)
```

### 实验 B 预期 (NCU):
每个 kernel 得到: registers_per_thread, SM throughput %, achieved occupancy %, DRAM R/W bytes, FP instructions。
这些数据用来拟合 cost model v2 的参数。

---

## B200 已有对照 (OSHER 完整 solver)

```
B200, 24020 cells (F2 refactored):
  Sync: 22.3 μs   Async: 11.4 μs   Graph: 8.2 μs   Persistent: 9.6 μs
  Compute: ~9.5 μs   OH: 56%
```
