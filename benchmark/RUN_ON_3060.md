# RTX 3060 Benchmark Execution Plan

## 硬件信息
- RTX 3060: Ampere (sm_86), 28 SMs, 12GB GDDR6
- Cooperative launch limit: ~56-84 blocks (需实测)
- 对照: B200 (148 SMs, cooperative limit ~444 blocks)

## 环境准备

```bash
# 1. 确认 GPU
nvidia-smi

# 2. Python 环境
# 如果有已有 venv 就复用，否则新建
python3 -m venv spmd-venv
source spmd-venv/bin/activate
pip install taichi numpy

# 3. 编译 CUDA 实验 (arch=sm_86 for 3060)
nvcc -O3 -arch=sm_86 -rdc=true benchmark/overhead_solutions.cu \
     -o benchmark/build/overhead_solutions -lcudadevrt
nvcc -O3 -arch=sm_86 -rdc=true benchmark/hydro_persistent.cu \
     -o benchmark/build/hydro_persistent -lcudadevrt
nvcc -O3 -arch=sm_86 -rdc=true benchmark/persistent_async_copy.cu \
     -o benchmark/build/persistent_async_copy -lcudadevrt

# 4. 创建输出目录
mkdir -p benchmark/results
```

---

## 实验列表 (按优先级排序)

### 实验 1 [最重要]: 36 Kernel Overhead Characterization

**目的**: 测量每种 kernel 的 per-step 时间，计算 overhead fraction，验证代际趋势。

```bash
cd spmd-dialect
python benchmark/run_overhead_characterization.py 2>&1 | tee benchmark/results/3060_characterization.txt
```

**预计运行时间**: 15-30 分钟
**输出**: 每个 kernel 的 μs/step + OH-DOM/TRANS/COMPUTE 分类
**关键看的**:
- Taichi 的 per-step floor 是多少（B200 上是 ~15μs）
- 多少比例是 OH-dominated（B200 上是 90%）
- 3060 上 overhead fraction 是否更低（预期，因为 compute 更慢）

---

### 实验 2 [重要]: 4 策略对比 (Sync/Async/Graph/Persistent)

**目的**: 在 3060 上对比四种 overhead 消除策略。

```bash
benchmark/build/overhead_solutions 2>&1 | tee benchmark/results/3060_solutions.txt
```

**预计运行时间**: 5-10 分钟
**关键看的**:
- Persistent kernel 在哪个 grid size 开始不可用（cooperative limit）
- Graph vs Persistent 哪个更快
- Async 的 speedup 是否和 B200 一致 (~1.9x)

---

### 实验 3 [重要]: Hydro-cal 真实案例

**目的**: 真实非结构化网格 (6675 cells, 105 blocks) 上测试。

```bash
benchmark/build/hydro_persistent 2>&1 | tee benchmark/results/3060_hydro.txt
```

**关键看的**:
- 105 blocks 是否超出 3060 cooperative limit（可能!）
- 如果超出 → 记录 "persistent N/A on 3060"，这证明了 strategy selection 的必要性
- Graph 和 Async 的 speedup

---

### 实验 4 [重要]: Compute-Communication Overlap

**目的**: 验证 persistent kernel + async DMA 在 3060 上是否同样 work。

```bash
benchmark/build/persistent_async_copy 2>&1 | tee benchmark/results/3060_async_copy.txt
```

**关键看的**:
- persistent+save vs persistent+no-save 的差值（B200 上是 5.3%，接近零）
- 如果 3060 上也接近零 → 证明 DMA overlap 是跨 GPU 代通用的

---

### 实验 5 [补充]: 跨框架对比

**目的**: 收集 Taichi vs Warp 在多个 kernel 上的数据。

```bash
# 如果装了 warp (pip install warp-lang)
python benchmark/A1_jacobi_2d/run.py 2>&1 | tee benchmark/results/3060_jacobi.txt
python benchmark/A3_wave_equation/run.py 2>&1 | tee benchmark/results/3060_wave.txt
python benchmark/B1_nbody/run.py 2>&1 | tee benchmark/results/3060_nbody.txt
python benchmark/D2_stable_fluids/run.py 2>&1 | tee benchmark/results/3060_fluids.txt
```

---

### 实验 6 [补充]: Multi-kernel/step Cases (Fusion 价值验证)

**目的**: 测量多 kernel/step 的 case 的每步总开销，证明 fusion 的价值。

这些数据已包含在实验 1 中 (run_overhead_characterization.py)，重点关注:

| Case | Kernels/Step | 看什么 |
|---|---|---|
| **CG Solver** | 5 | per-step 时间 vs 单 kernel 的 5 倍 → 量化多 kernel overhead |
| **Stable Fluids** | 22 | 22 kernel/step 的 overhead 应极其显著 |
| **PIC 1D** | 4 | particle-grid 耦合的 overhead |
| **LULESH-like** | 3 | 经典 proxy app 的 overhead |

---

## 结果整理

跑完后，所有数据在 `benchmark/results/`。需要回答的核心问题:

### Q1: 代际趋势验证
```
B200 (148 SMs): overhead fraction ~65%, GPU utilization ~35%
3060 (28 SMs):  overhead fraction = ?%, GPU utilization = ?%
```
预期 3060 的 overhead fraction 更低 (30-40%)，因为 compute 更慢 (fewer SMs)。
但这正好证明："越强的 GPU，overhead 占比越高"。

### Q2: Cooperative Launch Limit
```
3060: 28 SMs × occupancy blocks/SM = ? max blocks
哪些 kernel 的 grid 超过这个 limit？
```
超限的 kernel → 只能用 Graph 或 Async（不能 persistent）

### Q3: Multi-kernel 的 overhead 倍数
```
CG (5 kernels/step): overhead = 5 × single-kernel overhead?
StableFluids (22 kernels/step): overhead = 22 × ?
```
如果 overhead 确实和 kernel 数成正比 → fusion 的理论收益 = kernel 数倍

### Q4: Async DMA Overlap
```
persistent + save: ? μs/step
persistent no-save: ? μs/step
差值 / no-save = ?%
```
如果 <10% → 证明 DMA overlap 跨 GPU 通用

---

## B200 已有数据 (对照)

```
B200 (148 SMs):
  Heat2D 256²:  Sync=12.9  Async=8.1  Graph=3.3  Persistent=4.8  μs/step
  GrayScott 256²: Sync=17.5  Async=12.3  Graph=5.7  Persistent=6.5
  Hydro-cal 6675: Sync=15.2  Async=8.2  Graph=5.3  Persistent=5.7
  Persistent+DMA: 4.99 μs (vs 4.74 no-save = 5.3% overhead)
  Taichi floor: ~15 μs (all kernels, regardless of size)
  Warp floor:  ~15-18 μs
  Cooperative limit: ~444 blocks (occ=3/SM × 148 SMs)
  OH-dominated: 54/60 configs at typical mesh sizes
```
