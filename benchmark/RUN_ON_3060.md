# 3060 Benchmark Execution Plan

## 目标

在 RTX 3060 (28 SMs, sm_86) 上复现 B200 的全部实验，收集：
1. 每个 kernel 的 overhead fraction (对比 B200，验证代际趋势)
2. Persistent kernel 的 cooperative launch limit (3060 更小)
3. CUDA Graph vs Persistent vs Async 三种方案的对比
4. Compute-Communication overlap (persistent + DMA) 验证
5. Register tuning 在 3060 上的效果

## 环境准备

```bash
# 1. 确认 GPU
nvidia-smi  # 应该显示 RTX 3060

# 2. 安装 Python 环境 (如果还没有)
# 复用 spmd-venv 或者创建新的
python -m venv spmd-venv
pip install taichi numpy
pip install warp-lang  # 或 nvidia-warp

# 3. 编译 CUDA 实验 (需要 nvcc, arch 改成 sm_86)
nvcc -O3 -arch=sm_86 -rdc=true benchmark/overhead_solutions.cu -o benchmark/overhead_solutions -lcudadevrt
nvcc -O3 -arch=sm_86 -rdc=true benchmark/hydro_persistent.cu -o benchmark/hydro_persistent -lcudadevrt
nvcc -O3 -arch=sm_86 -rdc=true benchmark/persistent_async_copy.cu -o benchmark/persistent_async_copy -lcudadevrt
```

## 实验 1: 60+ Kernel Overhead Characterization (Taichi)

```bash
# 运行所有 Taichi benchmark kernels
# 输出: 每个 kernel 的 μs/step
python benchmark/run_overhead_characterization.py 2>&1 | tee results/3060_characterization.txt
```

这会跑 25+ 种 kernel × 多种 size，测量 Taichi 的 per-step overhead。

## 实验 2: 4 方案对比 (CUDA)

```bash
# Sync / Async / Graph / Persistent 对比 (Heat2D, GrayScott, 多种 size)
./benchmark/overhead_solutions 2>&1 | tee results/3060_overhead_solutions.txt
```

**注意**: 3060 上 persistent kernel 可能在较大 grid 时不可用 (cooperative limit ~56-84 blocks)。这正好说明 strategy selection 的必要性。

## 实验 3: Hydro-cal 实际案例

```bash
# hydro-cal: 6675 cells = 105 blocks
# 3060 cooperative limit 可能不够! 如果报错就记录 "persistent N/A on 3060"
./benchmark/hydro_persistent 2>&1 | tee results/3060_hydro.txt
```

## 实验 4: Compute-Communication Overlap

```bash
# persistent kernel + async DMA copy
./benchmark/persistent_async_copy 2>&1 | tee results/3060_async_copy.txt
```

## 实验 5: 跨框架对比 (Taichi vs Warp vs Kokkos)

```bash
# 各个 benchmark 目录
python benchmark/A1_jacobi_2d/run.py 2>&1 | tee results/3060_jacobi.txt
python benchmark/A3_wave_equation/run.py 2>&1 | tee results/3060_wave.txt
python benchmark/B1_nbody/run.py 2>&1 | tee results/3060_nbody.txt
python benchmark/D2_stable_fluids/run.py 2>&1 | tee results/3060_fluids.txt
python benchmark/F1_hydro_shallow_water/run.py 2>&1 | tee results/3060_hydro_swe.txt
```

## 收集数据后的分析

所有结果保存在 `results/` 目录。需要回答的问题:

1. **代际趋势验证**: 3060 上 overhead fraction 是否比 B200 低 (预期 30-40% vs 65%)?
2. **Persistent 可用性**: 3060 的 cooperative limit 能跑多大的 grid?
3. **Strategy selection**: 哪些 kernel 在 3060 上 Graph 最优? 哪些 Persistent 最优?
4. **Async DMA**: 在 3060 上 overlap 是否同样 work?

## B200 已有数据 (对照)

```
B200 (148 SMs):
  Heat2D 256²:  Sync=12.9  Async=8.1  Graph=3.3  Persistent=4.8  μs/step
  GrayScott 256²: Sync=17.5  Async=12.3  Graph=5.7  Persistent=6.5
  Hydro-cal:    Sync=15.2  Async=8.2  Graph=5.3  Persistent=5.7
  Persistent+DMA: 4.99 μs (vs 4.74 no-save = 5.3% overhead)
  Taichi floor: ~15 μs regardless of problem size
  Warp floor: ~15-18 μs
```
