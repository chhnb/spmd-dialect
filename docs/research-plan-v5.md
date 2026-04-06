# Research Plan v5: Joint Strategy–Geometry Optimization

**Version:** 5.0
**Date:** 2026-04-06
**Status:** Draft — refines v4 model based on RTX 3060 data
**Supersedes:** research-plan-v4 (strategy selection with fixed block size)

---

## 0. Why v5: What the Data Broke in v4

v4 的模型把 block size B 当作已知输入 (G_i = ceil(N/B) 是固定的)。
3060 数据证明这是错误的：

| 变量 | 变化范围 | 对 Persistent 32² 的影响 |
|---|---|---|
| **Block size** (128 vs 256) | 2x | **1.86x** (71.8→38.5 μs) |
| Register limit (32 vs default) | ~1.6x regs | 1.16x (71.8→60.1 μs) |
| Strategy (Sync→Persistent, 固定 B=256) | — | 1.91x (137.4→71.8 μs) |

Block size 对 32² 的影响 (1.86x) 几乎和 strategy 选择本身 (1.91x) 一样大。
更关键的是，**B 和 strategy 是耦合的**：

- 减小 B → 更多 blocks → 更好的 SM 利用率 → 但可能超出 cooperative limit → persistent 不可用
- 增大 B → 更少 blocks → 每 block 更多寄存器 → persistent 可用 → 但 SM 空闲

**v4 无法捕捉这个耦合。v5 把 B 从输入变为决策变量。**

---

## 1. 问题重建

### 1.1 决策空间

给定一个 time-stepping loop (K kernels/step, N cells)，同时选择：

```
决策变量:
  strategy ∈ {Sync, Async, Graph, Persistent}
  B        ∈ {32, 64, 128, 256, 512}        // threads per block

目标: minimize T_step(strategy, B, N)
```

这是一个 **4 × 5 = 20 个候选点** 的离散优化——很小，但关键是需要准确预测每个点的 T_step。

### 1.2 问题分解

```
T_step(strategy, B, N) = T_compute(B, N) + T_overhead(strategy, B, N)
                          ^^^^^^^^^^^        ^^^^^^^^^^^^^^^^^^^^^^^
                          GPU 上的纯计算      launch / sync / barrier
```

v4 把 T_compute 当作常数 T_i，与 B 无关。实际上 T_compute 强烈依赖 B（通过 occupancy 和 wave count）。

---

## 2. 模型公式 (v5)

### 2.1 输入参数

```
Per-kernel (从 nvcc 编译 + ncu profiling 获取):
  R_i(B)   — kernel i 在 block size B 时的寄存器数 (编译器分配，B 不同可能不同)
  bytes_i  — kernel i 的 global memory 访问量 (bytes/cell)
  flops_i  — kernel i 的浮点运算量 (FLOPs/cell)

Hardware (查手册 / cudaGetDeviceProperties):
  S        — SM count (3060 = 28, B200 = 148)
  RF       — registers per SM (Ampere = 65536)
  W_max    — max warps per SM (Ampere = 48)
  BW_peak  — peak memory bandwidth (GB/s)
  B_max_SM — max blocks per SM (Ampere = 16)
  SM_cap   — shared memory per SM (bytes)

Overhead constants (一次性标定):
  T_launch — CUDA kernel launch overhead (μs)
  T_sync   — cudaDeviceSynchronize overhead (μs)
  T_graph  — CUDA Graph replay overhead per step (μs)
  T_gsync  — cooperative grid.sync() overhead (μs)
```

### 2.2 Occupancy 子模型

给定 block size B 和 register count R：

```
warps_per_block(B) = ceil(B / 32)

max_blocks_per_SM(B, R) = min(
    floor(RF / (R × B)),                    // register limit
    floor(W_max / warps_per_block(B)),       // warp limit
    B_max_SM,                                // hardware block limit
    floor(SM_cap / shared_per_block)         // shared memory limit (if used)
)

occupancy(B, R) = warps_per_block(B) × max_blocks_per_SM(B, R) / W_max
```

**cooperative launch limit:**
```
coop_limit(B, R) = max_blocks_per_SM(B, R) × S
```

这是 `cudaOccupancyMaxActiveBlocksPerMultiprocessor()` 返回的值 × SM 数。

### 2.3 Compute 子模型

```
blocks_needed(B, N) = ceil(N / B)

waves(B, R, N) = ceil(blocks_needed(B, N) / (max_blocks_per_SM(B, R) × S))

// 对 memory-bound kernel (大多数 simulation kernel):
T_compute_i(B, N) = waves(B, R_i, N) × (bytes_i × B) / BW_effective(occupancy)

// 对 compute-bound kernel:
T_compute_i(B, N) = waves(B, R_i, N) × (flops_i × B) / compute_peak
```

其中 `BW_effective(occ)` 是关键的经验函数：
```
BW_effective(occ) = BW_peak × α(occ)

α(occ) 的经验拟合:
  当 occ ≈ 100%: α ≈ 0.7-0.85 (受 cache/bank conflict 限制)
  当 occ < 25%:  α ↓ 急剧下降 (latency hiding 不足)
  α(occ) ≈ min(1, β × occ^γ)   // 幂律拟合，β, γ 从实测标定
```

**简化版 (实用):**
不单独建模 BW_effective，而是直接用 **ncu 测的 T_kernel(B)** 作为基准，然后用 wave scaling 做外推：
```
T_compute_i(B, N) = T_kernel_i(B, N_ref) × waves(B, R_i, N) / waves(B, R_i, N_ref)
```
即：在参考 N_ref 上实测一次，其他 N 通过 wave count 线性外推。

### 2.4 Overhead 子模型

每一步的 overhead（与计算无关的固定开销）：

```
            ┌ Sync:       K × (T_launch + T_sync)
            │
T_overhead =┤ Async:      K × T_launch
            │
            │ Graph:      T_graph_replay           // ≈ 1-3 μs (nearly constant)
            │
            └ Persistent: (K-1) × T_gsync          // grid.sync() between phases
```

注意：Persistent 的 T_overhead 不包含 launch（只 launch 一次），但包含 K-1 次 grid.sync。

### 2.5 Strategy 约束

```
Persistent 可用 iff:
  blocks_needed(B, N) ≤ coop_limit(B, R_fused)
  且 R_fused = max(R_1, ..., R_K)           // 融合 kernel 取最大寄存器

Graph 可用 iff:
  loop 无 dynamic control flow (无 convergence check, 无 adaptive dt)

Async/Sync:
  始终可用
```

### 2.6 联合优化

```
minimize   T_step(strategy, B) = Σ_i T_compute_i(B, N) + T_overhead(strategy, B, N)

over       strategy ∈ {Sync, Async, Graph, Persistent}
           B ∈ {32, 64, 128, 256, 512}

subject to:
           feasibility(strategy, B, N, R_fused)
```

20 个候选点，直接枚举取 min。

---

## 3. 模型验证：用已有数据回测

### 3.1 Case: F1 OSHER, 32² (1024 cells), 3060

已知：K = 2 (compute + transfer), R ≈ 106 regs, S = 28 SMs

**B = 256:**
```
blocks = ceil(1024/256) = 4
warps/block = 8
max_blocks/SM = min(floor(65536/(106×256)), floor(48/8), 16) = min(2, 6, 16) = 2
coop_limit = 2 × 28 = 56 ≥ 4 → Persistent OK
active_SMs = 4 blocks / (最多 2 blocks/SM) = 2..4 SMs (只用了 4/28 = 14% SMs)
waves = ceil(4 / (2×28)) = 1 wave, 但只有 4 blocks 在跑 → 大量 SM 空闲
```

**B = 128:**
```
blocks = ceil(1024/128) = 8
warps/block = 4
max_blocks/SM = min(floor(65536/(106×128)), floor(48/4), 16) = min(4, 12, 16) = 4
coop_limit = 4 × 28 = 112 ≥ 8 → Persistent OK
active_SMs = 8 blocks → 至少 8 SMs active (2..4 blocks/SM) → 28% SMs
waves = ceil(8 / (4×28)) = 1 wave, 但 8 blocks 比 4 blocks 多 2x 并行度
```

**预测 speedup = 8/4 = 2x (SM utilization ratio)**
**实测 speedup = 71.8/38.5 = 1.86x** → 误差 7.5%，模型方向完全正确！

### 3.2 Case: F1 OSHER, 64² (4096 cells), 3060

**B = 256:**
```
blocks = 16, max_blocks/SM = 2, active = min(16, 2×28) = 16
waves = ceil(16/56) = 1, 16 blocks 在 28 SMs 上 → 57% SM 利用率
```

**B = 128:**
```
blocks = 32, max_blocks/SM = 4, active = min(32, 4×28) = 32
waves = ceil(32/112) = 1, 32 blocks 在 28 SMs 上 → 但 register pressure ↑
```

**预测**: B=128 blocks 数增加 2x，但 register pressure 增加。net effect 不确定。
**实测**: B=256 (51.0 μs) < B=128 (56.1 μs) → **B=256 反而更好！**

这说明 64² 时 16 blocks 已经足够覆盖多数 SMs，继续增加 blocks 的收益被 register pressure 抵消。**模型需要同时考虑 wave utilization 和 per-thread efficiency 的 tradeoff。**

### 3.3 Key Insight from Validation

```
32² (极少 blocks): block count 主导 → 减小 B 大幅提升 SM 利用率 → 1.86x
64² (中等 blocks): block count 已够 → 增大 B 保护 register → 反而更好
128² (大量 blocks): SM 全满 → B 选择影响较小 → 策略差异收敛
```

**存在一个 crossover point**：当 blocks_needed ≈ S 时，B 的选择最敏感。
这个 crossover 恰好是 block size 和 strategy 耦合最强的区域。

---

## 4. v5 vs v4 差异

| 维度 | v4 模型 | v5 模型 |
|---|---|---|
| **决策变量** | strategy only | strategy **+ block size B** |
| **T_compute** | 常数 T_i | f(B, occupancy, wave count) |
| **Occupancy** | 只在 persistent 公式中出现 | 所有策略都受 B 影响 |
| **Cooperative limit** | 固定 G_limit | coop_limit(B, R) — 随 B 变化 |
| **Register** | R_fused = max(R_i) | R_i(B) — 编译器对不同 B 分配不同 regs |
| **搜索空间** | 4 个 strategy | 4 × 5 = 20 个 (strategy, B) pair |
| **核心发现** | strategy 选择 | **B 和 strategy 耦合**；block count < S 时 B 选择可给 ~2x |

---

## 5. 对贡献的影响

### C1 升级: Joint Strategy–Geometry Model

不再是 "what strategy to use"，而是 **"what (strategy, block_size) pair to use"**。

这有更强的学术价值：
- 前人 (PERKS, PyGraph, Kernel Batching) 都只选 strategy，不调 B
- 我们展示 B 的影响和 strategy 一样大 (1.86x vs 1.91x at 32²)
- **coupled decision** 比 independent decision 更有趣（reviewer 更认可）

### 模型复杂度仍然可控

20 个候选点的枚举 vs v4 的 4 个候选点 — 复杂度只增加 5x，仍然是 O(1) 的 lookup table。

在编译器/runtime 中实现：
```python
def select_best(N, K, kernels, gpu):
    best_time = float('inf')
    best_config = None
    for B in [32, 64, 128, 256, 512]:
        for strategy in ['sync', 'async', 'graph', 'persistent']:
            if not feasible(strategy, B, N, kernels, gpu):
                continue
            t = predict_T_step(strategy, B, N, kernels, gpu)
            if t < best_time:
                best_time = t
                best_config = (strategy, B)
    return best_config
```

### 新的实验设计

在 v4 实验基础上增加：

| 实验 | 目的 |
|---|---|
| **Block size sweep** × strategy × grid size | 建立 T_compute(B, N) 的经验数据 |
| **α(occ) 标定** | 测量 BW_effective vs occupancy 的关系 |
| **Crossover point 验证** | 验证 blocks ≈ S 时 B 选择最敏感 |
| **Joint vs Independent** | 对比 (一起选 B+strategy) vs (先选 B 再选 strategy) 的差异 |

---

## 6. 下一步

1. **在 3060 上对 F1 OSHER 做完整的 B × strategy 矩阵** (5 × 4 = 20 configurations × 3 grid sizes)
2. **用 ncu 收集 R(B)**, 验证不同 B 下编译器分配的寄存器数
3. **标定 α(occ)**: 跑 Heat2D 等简单 kernel 在不同 occupancy 下的 BW 利用率
4. **验证 crossover point**: 扫描 N 从 16² 到 256²，找到 B 选择敏感性转折点
5. **更新 advisor report** 用 v5 模型框架解释所有现有数据

---

## Appendix: 3060 Data Summary (v5 relevant)

### A. Block Size Impact (Persistent, F1 OSHER fp64)

| B (threads) | 32² blocks | 32² μs | 64² blocks | 64² μs | 128² blocks | 128² μs |
|---|---|---|---|---|---|---|
| 128 | 8 | **38.5** | 32 | 56.1 | 128 | 153.4 |
| 256 | 4 | 71.8 | 16 | **51.0** | 64 | **149.0** |

### B. Register Limit Impact (Persistent, B=256, F1 OSHER fp64)

| maxrregcount | 32² μs | 64² μs | 128² μs |
|---|---|---|---|
| default (~106) | 71.8 | 51.0 | 149.0 |
| 64 | 72.1 | 51.2 | 151.0 |
| 48 | 61.6 | 53.5 | 151.1 |
| 32 | 60.1 | 51.7 | 155.8 |

### C. Cross-Framework (fixed B=128 for 32², B=256 for 64²/128²)

| 框架 | 32² | 64² | 128² |
|---|---|---|---|
| CUDA Persistent (adaptive) | **38.5** | **56.1** | 153.4 |
| CUDA Graph (adaptive) | 42.2 | 69.4 | **144.4** |
| Kokkos | 52.5 | 57.0 | 154.7 |
| Warp | 131.4 | 128.1 | 129.3 |
| Taichi | 189.7 | 138.1 | 160.9 |
| CUDA Sync | 137.4 | 149.2 | 212.4 |
