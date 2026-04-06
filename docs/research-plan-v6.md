# Research Plan v6: GPU Utilization Model for Simulation Time-Stepping

**Date:** 2026-04-06
**Status:** Draft for advisor discussion
**Supersedes:** v4 (strategy selection), v5 (+ block size)

---

## 1. Research Problem

### 1.1 现象

Python GPU simulation DSLs (Taichi, Warp) 让用户用 50 行 Python 就能写出 GPU 加速的 PDE solver。但 GPU 实际利用率极低：

| 工况 (3060, fp64) | T_measured | T_compute_best | **GPU 利用率** |
|---|---|---|---|
| F1 OSHER 32² (1024 cells) | 137.4 μs | 38.5 μs | **28%** |
| F1 OSHER 64² (4096 cells) | 149.2 μs | 51.0 μs | **34%** |
| F2 Hydro 6675 cells (简化) | 84.3 μs | 6.1 μs | **7%** |
| Heat2D 128² (16K cells) | 73.7 μs | 3.5 μs | **5%** |

**超过 60% 的 GPU 周期在做无用功。** 而且这个问题随 GPU 代际恶化：

| GPU | Compute (μs) | Overhead (μs) | 利用率 |
|---|---|---|---|
| V100 (2017) | ~26 | ~10 | 73% |
| 3060 (2021) | ~7 | ~77 | 8% |
| B200 (2024) | ~5 | ~10 | 35% |
| 2028 (推测) | ~1.3 | ~10 | 12% |

### 1.2 根本原因：不是单一瓶颈

直觉上可能以为 "overhead" 就是 launch 开销。但数据表明 **利用率损失来自多个层级，而且层级之间有耦合**。

以 F1 OSHER 32² 为例：

```
T_sync_default = 137.4 μs 分解:

  ┌─ L1 Temporal:     65.6 μs (48%)  GPU 在等 launch/sync
  │    → 优化方法: Persistent/Graph 消除
  │
  ├─ L2 Spatial:      33.3 μs (24%)  28 个 SM 只有 4 个在跑
  │    → 优化方法: 减小 block size, 增加 block 数
  │
  ├─ L3 Occupancy:    ~15 μs (11%)  每个 SM latency hiding 不足
  │    → 优化方法: 调 register limit, shared memory
  │
  ├─ L4 Memory:       ~10 μs (7%)   冗余 global memory 访问
  │    → 优化方法: register caching, 消除 transfer phase
  │
  └─ L5 Compute:      ~13 μs (10%)  真正的有效计算 (下界)
       → divergence, fp64 半速 — 不在优化范围
```

### 1.3 核心问题

> **GPU simulation time-stepping loop 的性能受多层级因素影响，这些因素有独立的优化旋钮但相互耦合。如何建立统一模型来理解、预测和自动优化？**

这不是 "用 Persistent 还是 Graph" 的问题。这是 **"在一个多维耦合空间里找最优配置"** 的问题。

---

## 2. 五层 GPU 利用率模型

### 2.1 模型结构

```
T_step = T_overhead + T_compute

T_overhead = f(strategy, K)                    ← L1 Temporal
T_compute  = T_ideal / (η_S × η_O × η_M × η_C)  ← L2-L5

总利用率 = T_ideal / T_step = T_ideal / (T_overhead + T_ideal / (η_S × η_O × η_M × η_C))
```

### 2.2 各层级定义

#### L1: Temporal Efficiency — "GPU 有没有在跑？"

GPU 在 kernel launch 之间、synchronization 时完全空闲。

| 来源 | 开销 (3060 实测) |
|---|---|
| cudaLaunchKernel | 3-10 μs/次 |
| cudaDeviceSynchronize | 5-15 μs/次 |
| Python runtime (Taichi/Warp) | 5-20 μs/步 |
| CUDA driver scheduling | 2-5 μs/次 |

**数据支撑**:
- F2 简化 (2 kernels/step): Sync = 84.3 μs, Persistent = 6.1 μs → **78.2 μs (93%) 是 temporal loss**
- F1 OSHER 64²: Sync = 149.2 μs, Persistent = 51.0 μs → **98.2 μs (66%)**
- Heat2D 128²: Sync = 73.7 μs, Graph = 4.2 μs → **69.5 μs (94%)**

#### L2: Spatial Efficiency — "多少 SM 在工作？"

当 block 数 < SM 数 × blocks_per_SM 时，部分 SM 空闲。

```
η_S = min(1, blocks_active / (max_blocks_per_SM × S))
blocks_active = ceil(N / (B × C))
```

**数据支撑**:
- F1 32², B=256: 4 blocks / 28 SMs → **η_S ≈ 0.14**
- F1 32², B=128: 8 blocks / 28 SMs → **η_S ≈ 0.29** → 1.86x faster
- F1 128², B=256: 64 blocks / 28 SMs → **η_S ≈ 1.0** → B 选择不再敏感

#### L3: Occupancy Efficiency — "每个 SM 跑得满吗？"

occupancy 不足 → active warps 少 → memory latency 无法 hiding → SM stall。

```
η_O = α(occupancy)
occupancy = active_warps_per_SM / max_warps_per_SM
active_warps = warps_per_block × max_blocks_per_SM(B, R)
```

**数据支撑**:
- Register sweep (F1 OSHER Persistent): default → reg32: **最多 16% 改善**
- Kokkos 112 regs vs CUDA 106 regs → 性能差异主要不来自 occupancy 而是 launch geometry

#### L4: Memory Traffic Efficiency — "多少 memory 访问是必要的？"

每步 simulation 中，部分 global memory 访问是 **冗余的**：中间结果被写出又立即读回。

```
η_M = useful_bytes / total_bytes
```

**估算 (F1 OSHER)**:
- 每 cell 每步: 320 bytes total traffic
- 其中 own-cell round-trip (write→read→write→read): 160 bytes → **50% 冗余**
- Register caching 可消除 ~120 bytes → η_M 从 0.5 提升到 0.75

**关键约束**: register caching **仅在 Persistent kernel 中可行** (thread 跨 timestep 存活)。Separate launch / Graph 无法做到。

#### L5: Compute Efficiency — "每条指令在做有用功吗？"

| 来源 | 影响 |
|---|---|
| Warp divergence | OSHER solver 有分支: wet/dry, flow direction → ~10-20% loss |
| fp64 半速 | 3060: fp64 = 1/64 fp32 throughput → 常数因子 |
| Instruction overhead | 寻址计算、边界检查 |

这一层主要由**算法决定**，不在我们的自动优化范围内。但需要在模型中量化，作为性能下界。

### 2.3 层间耦合

```
          L1 Temporal              L2 Spatial           L3 Occupancy        L4 Memory
          ───────────              ──────────           ────────────        ─────────
旋钮:     strategy                 B (block size)       reg_limit           reg_caching
                                   C (coarsening)       shared_mem_config   time_tiling

耦合:
  B ↓ ───→ blocks ↑ ──→ η_S ↑ (好)
     └──→ 可能超 coop_limit ──→ Persistent 不可用 ──→ η_T 受限 (坏)
     └──→ warps/block ↓ ──→ η_O 可能 ↓ (坏)

  reg_limit ↓ ──→ blocks/SM ↑ ──→ η_O ↑ (好)
            └──→ spill to local mem ──→ η_M ↓ (坏)

  Persistent ──→ 使能 register caching ──→ η_M ↑ (好)
            └──→ 需要 blocks ≤ coop_limit ──→ 约束 B 和 N 的范围 (限制)

  Graph ──→ η_T ↑ (消除 launch, 好)
       └──→ 不支持 dynamic control flow (限制)
       └──→ 不支持 register caching (η_M 没变)
```

**核心 insight**: 不存在一个旋钮能优化所有层级。每个优化都有 tradeoff。模型的价值在于预测这些 tradeoff 在什么条件下如何平衡。

---

## 3. 完整调优空间

### 3.1 所有旋钮

| 类别 | 旋钮 | 取值空间 | 影响层级 |
|---|---|---|---|
| **Inter-kernel** | Execution strategy | {Sync, Async, Graph, Persistent} | L1 |
| | Fusion scope | {none, partial, full} | L1, L4 |
| **Cross-timestep** | Register caching | {off, own-cell} | L4 |
| | Time tiling depth | {1, 2, 4} | L4 |
| | Double buffering | {off, on} | L4 |
| | Async DMA | {off, periodic} | L1 (for save) |
| **Intra-kernel** | Block size B | {32, 64, 128, 256, 512} | L2, L3 |
| | Thread coarsening C | {1, 2, 4} | L2, L3 |
| | Register limit | {default, 32, 48, 64} | L3 |
| | L1/shared split | {prefer_L1, prefer_shared} | L3 |
| **Data** | Precision | {fp32, fp64, mixed} | L3, L5 |
| | Layout | {AoS, SoA} | L3 |

### 3.2 约束

```
Persistent ⟹ blocks(B,C,N) ≤ coop_limit(B, R, S)
Graph      ⟹ 无 dynamic control flow
Register caching ⟹ Persistent (thread 必须跨步存活)
Time tiling      ⟹ Persistent
B × C ≤ 1024 (hardware thread limit)
R_fused = max(R_i) for full fusion
```

### 3.3 搜索空间估算

理论: 4 × 3 × 2 × 3 × 2 × 2 × 5 × 3 × 4 × 2 × 2 × 2 ≈ 200K
去掉约束违反后: **~2000-5000 可行配置**
去掉不重要的旋钮 (只留 Top-3: strategy, B, register_caching): **4 × 5 × 2 = 40**

**这远小于 TVM 的 10⁹ 搜索空间 → 不需要 ML cost model，解析模型 + 枚举即可。**

### 3.4 与 TVM/Halide 的关系

| 维度 | TVM/Halide | 我们 |
|---|---|---|
| **优化对象** | 单个 tensor operator | Time-stepping loop (多 kernel 循环) |
| **搜索空间** | Intra-kernel: tile, bind, vectorize, ... | **Inter-kernel + cross-timestep + intra-kernel** |
| **搜索规模** | ~10⁹ (需要 ML) | ~10³ (解析模型 + 枚举) |
| **Cost model** | 学习型 (XGBoost/NN) | 解析型 (5 层分解) |
| **适用领域** | ML (matmul, conv, attention) | Simulation (stencil, CFD, particle, FEM) |
| **时间结构** | DAG (单次执行) | **循环 (重复执行同一 kernel 序列)** |

**关键差异**: TVM 把每个 op 独立优化。我们的问题有 **时间循环结构**，使得 inter-kernel 和 cross-timestep 优化成为可能 (Graph, Persistent, register caching)。这是 simulation workload 独有的，ML 没有。

---

## 4. Related Work & Positioning

### 4.1 Gap Analysis

| 工作 | 做了什么 | 没做什么 |
|---|---|---|
| **TVM** (OSDI 2018) | Intra-kernel auto-tune | 不管 kernel 之间；不管 time-stepping |
| **Halide** (PLDI 2013) | Algorithm-schedule 分离 | 同 TVM |
| **PERKS** (ICS 2023) | 单 kernel persistent + register cache | 不做 multi-kernel；不做 strategy selection；不建模 |
| **PyGraph** (arXiv 2025) | Auto CUDA Graph, Graph vs Stream cost model | 不考虑 Persistent；只做 ML |
| **AsyncTaichi** (arXiv 2020) | SFG + megakernel fusion, 1.87x | **废弃** (sparse SNode 追踪太复杂)；无 cost model |
| **HFuse** (CGO 2022) | Horizontal fusion + register analysis | ML DAG；不是 time-stepping |
| **FLUDA** (AIAA 2023) | Manual CFD kernel fusion, 4x | 手动、单应用、无通用模型 |
| **Kernel Batching** (arXiv 2025) | Graph batch size model | 只做 Graph；不做 Persistent/register caching |
| **FreeStencil** (ICPP 2024) | JIT stencil fusion, 3.29x | 只做结构化 stencil |
| **Leonid** (ICS 2025) | Kokkos auto fusion, 1.26x | 不做 persistent；不做 register caching |
| **Roofline** (CACM 2009) | Compute vs memory bandwidth | 不建模 launch overhead；不建模 SM utilization |
| **Top-Down CPU** (ISPASS 2014) | CPU pipeline slot 4 层分解 (500+ citations) | CPU 专属；无 GPU 对应物 |

### 4.2 我们的定位

**"Top-Down Analysis for GPU Simulation"**

类比 Intel Top-Down 把 CPU 性能损失分解为 Front-end / Back-end / Bad Speculation / Retiring 四层，我们把 **GPU simulation time-stepping 的性能损失分解为 Temporal / Spatial / Occupancy / Memory / Compute 五层**。

Top-Down CPU (2014) 有 500+ citations，被集成进 Intel VTune。如果我们的 GPU 五层分解对 simulation 同样有效，它有潜力成为 GPU simulation 性能分析的标准框架。

### 4.3 核心贡献定位

| Contribution | 具体内容 | vs 前人 |
|---|---|---|
| **C1: 五层利用率模型** | Temporal + Spatial + Occupancy + Memory + Compute 分解 | 首次统一建模 kernel 间和 kernel 内的性能损失 |
| **C2: 调优空间定义** | Inter-kernel + cross-timestep + intra-kernel 三类旋钮 | TVM 只覆盖 intra-kernel |
| **C3: 解析 cost model** | 预测每层 η 和 T_step(config) | 不需要 ML，可解释，可跨 GPU 迁移 |
| **C4: 自动配置选择** | 枚举可行配置，model 预测，选最优 | 首次对 simulation loop 做 auto-tuning |
| **C5: 全面 characterization** | 60+ kernels × 多框架 × 2 GPUs 的五层分解数据 | 首次系统量化 simulation workload 的 GPU 利用率 |

---

## 5. Evaluation Plan

### 5.1 实验一：五层分解验证 (C1, C5)

**目标**: 对代表性 kernel 做完整五层分解，验证模型的分解能力。

**方法**:
- 选 6-8 个 kernel 覆盖不同类型: Heat2D (简单 stencil), Jacobi (iterative), OSHER (复杂 CFD), N-body (particle), LBM (lattice), CG (多 kernel)
- 每个 kernel 跑 3+ grid sizes (small/medium/large)
- 用 ncu + nsys + cudaEvent 测量每层:
  - L1: nsys timeline → GPU idle time
  - L2: ncu SM utilization
  - L3: ncu achieved occupancy + memory throughput / peak
  - L4: ncu DRAM read/write bytes → 算 redundancy
  - L5: ncu warp execution efficiency

**预期结果**: 展示不同 kernel × grid size 下，五层分解的比例变化规律。验证小 grid 以 L1+L2 为主，大 grid 以 L5 为主。

### 5.2 实验二：调优空间探索 (C2, C4)

**目标**: 对 F1 OSHER 做完整的配置矩阵，找到最优点，验证 cost model 预测。

**方法**:

Phase A — 核心旋钮矩阵 (40 配置):
```
strategy × B × register_caching
= {Sync, Async, Graph, Persistent} × {32, 64, 128, 256, 512} × {off, on}
× 3 grid sizes (32², 64², 128²)
= 40 × 3 = 120 实验点
```

Phase B — 次要旋钮 refinement (在 Phase A 最优附近):
```
最优 (strategy, B) ± {reg_limit, coarsening, L1/shared}
≈ 50 额外实验点
```

**测量**: 每个配置的 T_step (median of 5 repeats, 500 steps each)

**预期结果**:
- 实测最优配置 (oracle)
- Cost model 预测的最优配置
- 报告: model accuracy (MAPE), selection accuracy (model vs oracle 匹配率)

### 5.3 实验三：Register Caching 验证 (C1 中 L4 层)

**目标**: 验证 register caching 对 memory traffic 和性能的影响。

**方法**:
- 在 F1 OSHER persistent kernel 中实现 register caching (消除 transfer phase, own-cell data 留寄存器)
- 用 ncu 测量 global memory traffic (bytes) before/after
- 测量 T_step before/after

**预期结果**:
- Memory traffic 减少 ~30-40%
- T_step 改善 ~1.3-1.5x
- 验证 L4 层的估算

### 5.4 实验四：跨 GPU 泛化 (C3)

**目标**: 验证 cost model 在不同 GPU 上的泛化能力。

**方法**:
- 在 3060 上标定模型参数 (T_launch, T_sync, α(occ), ...)
- 用标定后的模型预测 B200 上的最优配置
- 在 B200 上实测验证

**预期结果**: 模型跨 GPU 预测误差 < 20%。

### 5.5 实验五：跨框架对比 (C5)

**目标**: 展示不同框架在五层模型下的差异。

**方法**:
- 同一个 kernel (F1 OSHER) 在 Taichi, Warp, Kokkos, CUDA 上测量
- 对每个框架做五层分解
- 分析各框架的利用率瓶颈

**已有数据 (3060, F1 OSHER 32²)**:
| 框架 | T_step (μs) | 主要瓶颈层 |
|---|---|---|
| CUDA Persistent (adaptive) | 38.5 | L2 Spatial (SM 利用率仍只 29%) |
| Kokkos | 52.5 | L1 Temporal (C++ async 但仍有 launch) |
| Warp | 131.4 | L1 Temporal (per-step launch) |
| Taichi | 189.7 | L1 Temporal (per-step launch + sync) |
| CUDA Sync | 137.4 | L1 Temporal (48%) + L2 Spatial (24%) |

### 5.6 Benchmark Suite

| Category | Kernels | Count |
|---|---|---|
| Structured stencil | Heat2D, Wave2D, Jacobi2D, Burgers, Conv-Diff | 10+ |
| Phase field | Allen-Cahn, Cahn-Hilliard, Gray-Scott | 6+ |
| CFD | SWE (OSHER), LBM, Stable Fluids, Hydro-cal F2 | 8+ |
| Particle | N-body, SPH, DEM | 6+ |
| FEM / Structure | Explicit FEM, Cloth, Mass-Spring | 6+ |
| Multi-kernel | CG Solver, LULESH, MacCormack 3D | 6+ |
| EM / Other | FDTD Maxwell, Monte Carlo, PIC | 10+ |

**Total: 60+ kernels, 130+ configurations**

---

## 6. Implementation Plan

### Phase 1: 数据收集 + 五层标定 (4 weeks)

- [ ] 对 6-8 个代表性 kernel 做完整 ncu/nsys profiling
- [ ] 提取每层参数: T_launch, T_sync, R(B), occupancy(B), DRAM traffic
- [ ] 标定解析模型常数: α(occ), T_graph_replay, T_grid_sync
- [ ] 在 3060 + B200 上各做一轮
- **产出**: 五层分解数据集 + 标定参数

### Phase 2: Cost Model + 验证 (4 weeks)

- [ ] 实现解析 cost model (Python): 输入 (kernel_info, hardware, config) → 输出 predicted T_step
- [ ] 用 F1 OSHER 的 120 配置矩阵验证模型精度
- [ ] 用 30 个 kernel 拟合, 30 个 kernel 验证 (train/test split)
- [ ] Sensitivity analysis: 哪个参数对模型预测影响最大
- **产出**: cost model 代码 + 验证结果 + MAPE 报告

### Phase 3: Register Caching + Auto-selection (4 weeks)

- [ ] 在 OSHER persistent kernel 中实现 register caching
- [ ] 实现 auto-configuration selector: 枚举 + model 预测 + 选最优
- [ ] 对 10+ kernels 验证 auto-selection vs oracle
- [ ] 实现 compute-communication overlap (async DMA)
- **产出**: register caching 实现 + auto-selector 工具 + 评估

### Phase 4: 论文 (4 weeks)

- [ ] 论文撰写 (结构见 Section 7)
- [ ] 开源 benchmark suite + cost model + auto-selector
- **产出**: 论文 draft

**Total: ~16 weeks (4 months)**

---

## 7. Paper Outline

**Target venue**: ICS / PPoPP / CGO / SC

### Title Candidates

1. "The GPU Utilization Wall: A Five-Level Performance Model for Simulation Time-Stepping"
2. "Beyond Single-Kernel Tuning: Auto-Configuring GPU Simulation Loops"
3. "Where Do the Cycles Go? Decomposing GPU Utilization Loss in Simulation DSLs"

### Structure

**1. Introduction** (1.5p)
- GPU simulation DSLs 利用率只有 5-35%
- 不是单一瓶颈: launch overhead, SM starvation, occupancy, memory traffic 同时作用
- 现有优化各解一层: TVM (intra-kernel), PERKS (persistent), PyGraph (Graph)
- 没有人统一建模 → 无法自动选择最优配置
- 我们: 五层分解模型 + 调优空间定义 + 自动配置选择

**2. Background & Motivation** (1.5p)
- Taichi/Warp 执行模型 + 为什么产生 overhead
- GPU 硬件: SM, warp, register, memory hierarchy
- 代际趋势: overhead 不变而 compute 越来越快 → 利用率持续恶化

**3. The Five-Level GPU Utilization Model** (3p) — **核心贡献**
- 3.1 模型定义: T_step = T_overhead + T_ideal / (η_S × η_O × η_M × η_C)
- 3.2 L1 Temporal: overhead 来源 + 测量方法
- 3.3 L2 Spatial: block starvation 模型
- 3.4 L3 Occupancy: latency hiding 模型
- 3.5 L4 Memory Traffic: redundancy 分析 + register caching
- 3.6 L5 Compute: divergence + precision
- 3.7 层间耦合分析

**4. Tuning Space & Auto-Configuration** (2p)
- 4.1 调优旋钮定义 (inter-kernel, cross-timestep, intra-kernel)
- 4.2 约束 (cooperative limit, dynamic flow, ...)
- 4.3 Cost model for T_step prediction
- 4.4 Enumeration-based auto-configuration algorithm
- 4.5 与 TVM 的关系和区别

**5. Evaluation** (3p)
- 5.1 五层分解: 60+ kernels 的分解结果 + 规律
- 5.2 Model accuracy: MAPE across kernels and grid sizes
- 5.3 Auto-configuration accuracy: model vs oracle
- 5.4 End-to-end speedup: before (default) vs after (auto-configured)
- 5.5 Register caching: memory traffic reduction + speedup
- 5.6 Cross-GPU generalization: 3060 → B200
- 5.7 Cross-framework: Taichi vs Warp vs Kokkos vs CUDA

**6. Discussion** (0.5p)
- Limitations: 不处理 irregular workload, multi-GPU, dynamic mesh
- DSL 框架应该怎么集成这个模型
- 未来: 编译器自动应用 (MLIR pass)

**7. Related Work** (1p)

---

## 8. Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| 模型精度不够 (MAPE > 30%) | C3 弱化 | 用更多 kernel 标定; 加经验修正项; 退化为 profiling-guided |
| Register caching 实现太复杂 | C1 的 L4 层缺乏验证 | 先对 Heat2D (简单 stencil) 做 proof-of-concept |
| 搜索空间太小，reviewer 说 "trivial" | C2/C4 弱化 | 强调跨层耦合使最优点非直觉; 展示 oracle 和默认配置的差距 |
| Reviewer: "just use CUDA Graph" | — | (1) Graph 不支持 dynamic flow; (2) 不支持 register caching; (3) 模型可以预测什么时候 Graph 最优 |
| Reviewer: "incremental over PERKS" | — | PERKS 单 kernel + 无模型; 我们多层模型 + multi-kernel + auto-select |
| Reviewer: "TVM already solves this" | — | TVM 不覆盖 inter-kernel / cross-timestep; 我们的搜索空间和 TVM 正交 |
| 只有 2 个 GPU (3060, B200) | 泛化性不足 | 尝试借用 A100/H100; 即使只有 2 个 GPU，SM 数差 5x，仍有说服力 |

---

## 9. Expected Impact

### 对社区
- **五层分解成为 GPU simulation 性能分析的标准方法** (类比 Top-Down for CPU)
- **benchmark suite** 开源，成为 simulation overhead 研究的共用基准
- **cost model** 可被 Taichi/Warp 等框架直接集成

### 对论文
- **C1 (五层模型)**: 概念贡献，适合顶会
- **C5 (characterization)**: 数据贡献，60+ kernel 的系统测量
- **C3 (cost model) + C4 (auto-config)**: 实用贡献，可验证
- **C2 (调优空间)**: 定义贡献，bridging TVM 和 simulation

---

## Appendix: Key Data Points

### A. Overhead 分解实测 (3060)

| Kernel | Grid | T_sync (μs) | T_best (μs) | OH% | 最优策略 |
|---|---|---|---|---|---|
| Heat2D | 128² | 73.7 | 3.5 (Persistent) | 95% | Persistent |
| Heat2D | 256² | 72.5 | 6.7 (Graph) | 91% | Graph (超 coop limit) |
| Heat2D | 1024² | 150.7 | 83.2 (Graph) | 45% | Graph |
| F1 OSHER | 32² | 137.4 | 38.5 (Pers+B128) | 72% | Persistent + adaptive B |
| F1 OSHER | 64² | 149.2 | 51.0 (Persistent) | 66% | Persistent |
| F1 OSHER | 128² | 212.4 | 144.4 (Graph) | 32% | Graph |
| F2 Hydro (简化) | 6675 cells | 84.3 | 6.1 (Persistent) | 93% | Persistent |
| F2 Hydro + DMA | 6675 cells | — | 6.6 (Pers+DMA) | — | Persistent + async DMA |

### B. Block Size Impact (3060, F1 OSHER Persistent)

| B | 32² μs | 64² μs | 128² μs |
|---|---|---|---|
| 128 | **38.5** | 56.1 | 153.4 |
| 256 | 71.8 | **51.0** | **149.0** |
| 最优 B | 128 | 256 | 256 |
| Speedup vs wrong B | 1.86x | 1.10x | 1.03x |

### C. Cross-Framework (3060, F1 OSHER fp64, adaptive B)

| Framework | 32² μs | 64² μs | 128² μs |
|---|---|---|---|
| CUDA Persistent (adaptive) | 38.5 | 56.1 | 153.4 |
| CUDA Graph (adaptive) | 42.2 | 69.4 | 144.4 |
| Kokkos | 52.5 | 57.0 | 154.7 |
| Warp | 131.4 | 128.1 | 129.3 |
| Taichi | 189.7 | 138.1 | 160.9 |
| CUDA Sync | 137.4 | 149.2 | 212.4 |
