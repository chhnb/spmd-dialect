# 研究方案完整综述（供外部评估）

## 一、研究问题

Python GPU simulation DSLs（Taichi, Warp）让用户用几十行 Python 就能写出 GPU 仿真代码。但我们的实测发现，**end-to-end step time 中 60-95% 是浪费的**——不是在做有效计算，而是在等 kernel launch、sync、driver overhead，或因为 SM 没被充分利用。

具体数据（RTX 3060 Laptop, 30 SMs）：

| 工况 | 实测时间 | 最优可达 | **利用率** |
|---|---|---|---|
| Heat2D 128² (简单 stencil) | 73.7 μs | 3.5 μs | **5%** |
| OSHER 32² (完整 Riemann solver, fp64, 106 regs) | 137.4 μs | 38.5 μs | **28%** |
| OSHER 64² (4096 cells) | 149.2 μs | 51.0 μs | **34%** |
| OSHER 128² (16K cells) | 212.4 μs | 144.4 μs | **68%** |

**问题不限于某一个框架。** Taichi、Warp、甚至 C++ Kokkos 都存在不同程度的效率损失。问题也不限于某一种 GPU——在 B200（数据中心, 148 SMs）上同样存在。

## 二、为什么不是一个已解决的问题

### 已有方案及其局限

| 已有方案 | 做了什么 | 局限 |
|---|---|---|
| **CUDA Graph** (NVIDIA 2019) | 录制 kernel launch 序列，GPU 回放 | 无 grid size 限制，但不能做跨步数据复用 |
| **Persistent Kernel / PERKS** (ICS 2023) | 单 kernel 内部时间循环 + grid sync | 只处理单 kernel；有 cooperative launch grid size 限制 |
| **AsyncTaichi** (Hu et al. 2020, **已废弃**) | 通用计算图 (State-Flow Graph) + megakernel fusion | 过于复杂（4360 行，70% 处理稀疏数据结构），维护不住被废弃 |
| **PyGraph** (2025 预印本) | 自动 CUDA Graph for PyTorch | 只针对 ML 推理，不考虑 persistent kernel |

**核心空白**：没有人建模过 "什么时候用哪种策略"。最优配置随 (kernel 复杂度, grid size, GPU 型号) **共同变化**，不存在万能方案：

| 工况 | 3060 最优 | B200 最优 | 原因 |
|---|---|---|---|
| Jacobi2D 128² | Persistent (29.6x) | **Graph** (4.3x) | 3060 SM 少，persistent 的 grid sync 更高效 |
| OSHER 128² | **Graph** (1.75x) | Graph (1.6x) | OSHER 寄存器多(106), persistent 超 cooperative limit |
| Heat2D 128² | Persistent (20.8x) | Graph (4.4x) | 寄存器少，persistent 可行 |
| OSHER 32² + B=128 | Persistent+小block (3.57x) | Persistent (1.7x) | 小 grid 需要调 block size 增加 SM 覆盖 |

**最优策略在不同条件下发生翻转——这不是人能手动判断的。**

## 三、我们的方案：Legality-Aware Loop-Level Step-Time Model

### 3.1 核心思路

把 time-stepping simulation loop 的 end-to-end step time 分解为 **5 个层级的损失**，每层有不同的优化旋钮。通过一个 legality-aware 的 cost model，自动选择最优的 (execution strategy, block size, register caching) 配置。

### 3.2 五层损失分解

以 F1 OSHER 32² 在 3060 上为例（默认配置 = Sync loop, 256 threads/block）：

```
T_measured = 137.4 μs 分解:

  L1 Temporal:     65.6 μs (48%)  — GPU 在等 kernel launch/sync
     → 优化: Persistent kernel / CUDA Graph
  
  L2 Spatial:      33.3 μs (24%)  — 28 个 SM 只有 4 个在跑 (4 blocks < 28 SMs)
     → 优化: 减小 block size (256→128) → block 数翻倍 → 更多 SM 工作
  
  L3 Scheduler:    ~15 μs (11%)   — 每个 SM 内 latency hiding 不足
     → 优化: register limit → 增加 occupancy
  
  L4 Memory:       ~10 μs (7%)    — 冗余 global memory 读写
     → 优化: register caching（仅 persistent kernel 可做）
  
  L5 Compute:      ~13 μs (10%)   — 真正的有效计算（不可优化）
```

**不同 grid size 下瓶颈层级不同**：

| Grid | L1 (Temporal) | L2 (Spatial) | L3-L5 (Compute) | 主要瓶颈 |
|---|---|---|---|---|
| 32² (1024 cells) | 48% | 24% | 28% | L1 + L2 |
| 64² (4096 cells) | 66% | ~3% | 31% | **L1 主导** |
| 128² (16K cells) | 32% | ~0% | 68% | **Compute 主导** |

→ 小 grid 需要同时优化 L1 + L2；大 grid 优化 L1 收益递减。

### 3.3 Legality Filter

很多 (strategy, block_size, grid_size) 组合**根本不合法**：

```
Persistent kernel:  总 block 数 ≤ cooperative_launch_limit (SM数 × blocks_per_SM)
  → 3060 OSHER: limit ≈ 120 blocks → 128²(64 blocks) ✅, 256²(256 blocks) ❌
  → B200: limit ≈ 444 blocks → 更宽松

Block size:  B × components ≤ 1024 (hardware thread limit)

Register caching:  仅 persistent kernel 可做（thread 必须跨 timestep 存活）
```

Legality filter 把搜索空间从 ~200K 种配置砍到 ~2000-5000 种可行配置。

### 3.4 Cost Model

```
对每个可行配置 (strategy, B, reg_cache) 预测 step time:

T_step = T_frontend + T_device + T_exposed_copy

T_frontend = a_strategy × K + b_strategy        (L1: per-strategy affine model)
T_device   = waves × t_wave + t_tail            (L2-L5: wave model)

其中:
  waves = ceil(n_blocks / (resident_per_SM × num_SMs))
  n_blocks = ceil(grid_cells / block_size)
  resident_per_SM = f(block_size, register_count)   // occupancy calculator
```

模型自动选出使 T_step 最小的配置。

### 3.5 关键发现：层间耦合

各层的优化旋钮**互相影响**，这是模型的核心 insight：

```
减小 block size:
  → blocks ↑ → SM 覆盖率 ↑        (好: L2 改善)
  → 但可能超 cooperative limit     (坏: persistent 变不合法)
  → warps/block ↓ → latency hiding ↓ (坏: L3 恶化)

限制寄存器数:
  → blocks/SM ↑ → occupancy ↑      (好: L3 改善)  
  → 但 spill to local memory       (坏: L4 恶化)

Persistent kernel:
  → 消除 launch/sync overhead      (好: L1 归零)
  → 使能 register caching          (好: L4 改善)
  → 但 grid ≤ cooperative limit    (约束: 限制 N 和 B 的范围)
```

**举例**：OSHER 32² 上，Persistent kernel (消除 L1) + 小 block size (改善 L2) 联合优化给 3.57x，比任何单一优化都大。但 64² 上，小 block size 反而更慢（blocks 已经够多，小 B 的 register pressure 抵消了收益）。

## 四、Benchmark 基础设施

### 4.1 Kernel 覆盖

- **36 种 kernel types**（15 种独立计算模式），覆盖 9 个领域
- ~130 个测试配置
- 包含经典 benchmark：Jacobi3D (Parboil/PERKS), HotSpot/SRAD (Rodinia), SpMV, CG, LULESH (ECP)
- **两个真实工程案例**：hydro-cal F1 (完整 OSHER Riemann solver, fp64) 和 F2 (非结构化网格)

**Multi-kernel/step 案例**（fusion 价值最高）：
- CG Solver: 5 kernels/step (matvec + 2 dots + 2 updates)
- Stable Fluids: 22 kernels/step
- LULESH-like: 3 kernels/step
- PIC: 4 kernels/step

### 4.2 测试平台

| GPU | SMs | 架构 | 定位 |
|---|---|---|---|
| RTX 3060 Laptop | 30 | Ampere (sm_86) | 消费级，科研用户主力 |
| B200 | 148 | Blackwell (sm_90) | 数据中心 |

### 4.3 对比框架

| 框架 | 语言 | 执行模式 |
|---|---|---|
| Taichi | Python | per-step sync (类似 CUDA Sync) |
| Warp | Python | per-step sync (overhead 略低于 Taichi) |
| Kokkos | C++ | async launch (类似 CUDA Async) |
| CUDA (4 策略) | C/CUDA | Sync / Async / Graph / Persistent |
| NumPy | Python | CPU baseline |

### 4.4 对齐实验（关键）

CUDA 的 OSHER benchmark (`hydro_osher_benchmark.cu`) 使用和 Taichi/Kokkos **完全相同的** OSHER Riemann solver 和网格数据，确保苹果对苹果比较。早期简化版 (`hydro_persistent.cu`) 用 Rusanov flux，数据不可直接比较。

## 五、已有实验结果

### 5.1 核心数据：OSHER Solver 4 策略对比 (3060)

| N | Cells | Sync (μs) | Async (μs) | Graph (μs) | Persistent (μs) | OH% |
|---|---|---|---|---|---|---|
| 32 | 1024 | 109.4 | 49.5 | 40.7 | **38.4 (2.85x)** | 63% |
| 64 | 4096 | 157.9 | 68.8 | 100.7 | **69.2 (2.28x)** | 56% |
| 128 | 16384 | 257.5 | 155.1 | **147.5 (1.75x)** | 148.9 | 43% |

即使完整 OSHER solver (106 registers, fp64)，小/中网格上 overhead 仍占 43-63%。

### 5.2 联合优化 (strategy + block size)

| 配置 | OSHER 32² (μs) | vs 默认 |
|---|---|---|
| 默认 (Sync, B=256) | 137.4 | 1.0x |
| 只改 strategy (Persistent, B=256) | 71.8 | 1.91x |
| 只改 block size (Sync, B=128) | — | ~1x |
| **联合** (Persistent, B=128) | **38.5** | **3.57x** |

联合 > 单独之和，因为 L1 和 L2 是**乘性关系**。

### 5.3 Compute-Communication Overlap

| 方案 | μs/step | 说明 |
|---|---|---|
| Persistent (不保存) | 5.85 | 纯计算 |
| Persistent + async DMA | 6.58 (+12.5%) | DMA 引擎在 persistent kernel 跑着时同时传数据 |
| Sync + sync save | 76.69 | 传统阻塞式 |

**GPU 的 Compute Engine 和 DMA Copy Engine 是独立硬件**，persistent kernel 中可以并行传数据，几乎零开销。PERKS 和 AsyncTaichi 都没有做过这个。

### 5.4 跨框架验证 (F1 OSHER, 3060, fp64)

| 框架 | 32² | 64² | 128² |
|---|---|---|---|
| CUDA Persistent (adaptive B) | **38.5** | **56.1** | 153.4 |
| CUDA Graph (adaptive B) | 42.2 | 69.4 | **144.4** |
| Kokkos (C++, fp64) | 52.5 | 57.0 | 154.7 |
| CUDA Sync | 137.4 | 149.2 | 212.4 |
| Taichi | 189.7 | 138.1 | 160.9 |

- Persistent (adaptive B) 在 32² 上比 Kokkos 快 27%——消除 temporal + spatial loss 后可超越 C++ 原生框架
- 128² 时所有方案收敛（compute 主导）
- Taichi ≈ CUDA Sync（说明 Taichi 的主要 overhead 来自 per-step sync 模式）

### 5.5 Cost Model 初步结果

| 指标 | 当前值 |
|---|---|
| 时间预测误差 (MAPE) | 59.4% |
| 策略选择准确率 | 64% (9/14) |

模型能正确区分 regime（OH-dominated vs compute-dominated），但精度需要改进。

### 5.6 NCU Profiling (3060, 5 kernel types)

| Kernel | Regs/Thread | SM Throughput% (128²) | Occupancy% | 类型 |
|---|---|---|---|---|
| jacobi2d | 16 | 7.7% | 33.3% | 轻量 stencil |
| heat2d | 18 | 7.7% | 33.1% | 轻量 stencil |
| hotspot | 20 | 7.9% | 33.2% | 轻量 stencil |
| srad | 26 | 14.5% | 32.0% | 中等 stencil |
| **osher** | **106** | **60.6%** | **31.3%** | **重量 solver** |

## 六、与前人工作的精确定位

| 前人 | Venue | 做了什么 | 我们的差异 |
|---|---|---|---|
| **AsyncTaichi** | arXiv 2020 (废弃) | 通用 SFG + megakernel fusion, 1.87x | 分析其废弃原因；我们聚焦 time-stepping + 建模 |
| **PERKS** | ICS 2023 | 单 kernel persistent + register cache, 2.12x stencil | 我们: multi-kernel, + DMA overlap, + strategy selection, + block size tuning |
| **PyGraph** | arXiv 2025 | 自动 CUDA Graph for PyTorch, 12% avg | 我们: + persistent 选项, + legality model, + simulation domain |
| **HFuse** | CGO 2022 | Horizontal fusion + register analysis for ML | 不同 domain (DAG vs time-stepping) |
| **FLUDA** | NASA 2023 | 手动 CFD fusion, 4x | 单应用, 无模型, 无 strategy selection |
| **Kernel Batching** | arXiv 2025 | CUDA Graph batch size model | 只建模 Graph, 无 persistent |
| **FreeStencil** | ICPP 2024 | JIT fusion for structured stencils, 3.29x | 只限结构化 stencil |

**我们的独特定位**：
1. 首个建模 persistent kernel fusion 的 cost-benefit tradeoff (occupancy penalty, cooperative limit, block size 耦合)
2. 首个统一建模 Graph vs Persistent vs Async 的 legality-aware selector
3. 首个验证 DMA overlap 在 persistent kernel 中的可行性
4. 最全面的 simulation overhead characterization (36 kernels × 2 GPUs × 5 frameworks)

## 七、论文目标

**标题方向**: "Legality-Aware Loop-Level Step-Time Model for GPU Simulation: When to Fuse, How to Launch, and What to Cache"

**投稿目标**: ICS / PPoPP / CGO (systems + compiler)

**核心贡献**:
1. (C1) 五层损失分解 + 层间耦合分析
2. (C2) Legality-aware cost model + automatic strategy/block-size selector
3. (C3) Compute-communication overlap (persistent + DMA)
4. (C4) 36-kernel × 2-GPU × 5-framework characterization

**待完成**: Cost model 精度从 64% → 目标 85%+，论文撰写

## 八、关于 nsys/ncu 可视化

### 建议做的 nsys 图

**nsys timeline** 最直观地展示 overhead：

1. **Taichi Sync loop**: 可以看到 kernel 之间有大量空白（CPU 在做 Python → driver → sync）
2. **CUDA Graph replay**: 连续的 kernel 执行，几乎无间隙
3. **Persistent kernel**: 一整条长 bar，中间无断开

三张 nsys timeline 并排 → 一目了然地展示"overhead 在哪"。

```bash
# 在 3060 上跑
nsys profile -o sync_timeline python benchmark/A1_jacobi_2d/jacobi_taichi.py
nsys profile -o graph_timeline ./hydro_osher --strategy graph --N 64
nsys profile -o persistent_timeline ./hydro_osher --strategy persistent --N 32
```

### 建议做的 ncu 图

你已经有 ncu 数据了（`results/ncu_*.csv`），可以出：
1. **Register count vs SM throughput** 散点图（证明 OSHER 106 regs 是 compute-heavy）
2. **Occupancy vs achieved throughput**（证明 occupancy 不是唯一指标）

### 优先级

**nsys timeline > ncu 详细数据**。nsys 的可视化效果对汇报最有冲击力——一张图就能说清楚 "为什么 65% 时间在浪费"。ncu 数据更适合论文正文的细节分析。
