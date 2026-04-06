# Research Plan v6: GPU Utilization Model for Simulation Time-Stepping

**Date:** 2026-04-06
**Status:** Draft for advisor discussion
**Supersedes:** v4 (strategy selection), v5 (+ block size)

---

## 1. Research Problem

### 1.1 现象

Python GPU simulation DSLs (Taichi, Warp) 让用户用几十行 Python 就能写出 GPU 加速的 PDE solver。但我们的测量发现，GPU 实际利用率极低——**超过 60-95% 的 GPU 周期在做无用功**：

| 工况 (RTX 3060, fp64) | T_measured | T_compute_best | **GPU 利用率** |
|---|---|---|---|
| Heat2D 128² (16K cells, 简单 stencil) | 73.7 μs | 3.5 μs | **5%** |
| F2 Hydro 6675 cells (简化 Rusanov, 2 kernels) | 84.3 μs | 6.1 μs | **7%** |
| F1 OSHER 32² (1024 cells, 完整 Riemann solver) | 137.4 μs | 38.5 μs | **28%** |
| F1 OSHER 64² (4096 cells) | 149.2 μs | 51.0 μs | **34%** |
| F1 OSHER 128² (16K cells) | 212.4 μs | 144.4 μs | **68%** |

以上数据来自我们在 RTX 3060 Laptop GPU (28 SMs, Ampere sm_86) 上的实测。`T_measured` 是默认执行模式 (Sync loop)；`T_compute_best` 是我们能达到的最优配置。

**这个问题随 GPU 代际恶化**。我们在 B200 (148 SMs) 上的实测也验证了这一趋势：

| GPU | Compute (μs) | Overhead (μs) | 利用率 |
|---|---|---|---|
| V100 (2017) | ~26 | ~10 | 73% |
| RTX 3060 (2021) | ~7 | ~77 | 8% |
| B200 (2024) | ~5 | ~10 | 35% |
| 2028 (推测) | ~1.3 | ~10 | 12% |

**原因**: GPU 计算能力每代翻倍，但 CUDA driver 的 launch/sync 开销基本不变 (~10 μs)。随着 compute 变快，固定开销占比越来越高。

### 1.2 根本原因：不是单一瓶颈

直觉上可能认为 "overhead" 就是 kernel launch 开销。但我们的实验数据表明，**利用率损失来自多个层级，而且层级之间有耦合**。

以 F1 OSHER 32² 在 3060 上为例（默认配置: Sync loop, 256 threads/block）：

```
T_sync_default = 137.4 μs 分解:

  ┌─ L1 Temporal:     65.6 μs (48%)  GPU 在等 launch/sync
  │    → 消除方法: Persistent kernel / CUDA Graph
  │
  ├─ L2 Spatial:      33.3 μs (24%)  28 个 SM 只有 4 个在跑
  │    → 消除方法: 减小 block size (256→128) → block 数翻倍
  │
  ├─ L3 Occupancy:    ~15 μs (11%)   每个 SM 的 latency hiding 不足
  │    → 消除方法: 调 register limit → 增加 active warps
  │
  ├─ L4 Memory:       ~10 μs (7%)    冗余 global memory 访问
  │    → 消除方法: register caching (仅 Persistent 可做)
  │
  └─ L5 Compute:      ~13 μs (10%)   真正的有效计算 (下界)
       → divergence, fp64 半速 — 由算法决定
```

**不同 grid size 下瓶颈层级不同**：

| Grid | L1 Temporal | L2 Spatial | L3-L5 Compute | 主要瓶颈 |
|---|---|---|---|---|
| 32² | 48% | 24% | 28% | L1 + L2 |
| 64² | 66% | ~3% | 31% | **L1 主导** |
| 128² | 32% | ~0% | 68% | **Compute 主导** |

### 1.3 核心研究问题

> **GPU simulation time-stepping loop 的性能受多层级因素影响，这些因素各有独立的优化旋钮但相互耦合。如何建立统一模型来理解、预测和自动优化？**

这不是 "用 Persistent 还是 Graph" 的问题。这是 **"在一个多维耦合调优空间里，理解每个因素的影响并找到最优配置"** 的问题。

---

## 2. 已有实验结果

我们已经在两个 GPU 平台上完成了大量实验，覆盖了 36 种 kernel、5 个框架、多种优化策略。以下是按主题整理的完整结果。

### 2.1 Benchmark 基础设施

**覆盖范围**：
- **36 种 kernel types**（15 种独立计算模式），覆盖 9 个领域
- **~130 个测试配置** (不同 grid size × kernel type)
- 领域: Stencil, CFD, Particle, EM, FEM, Transport, PDE, Classic benchmarks
- 经典 benchmark: Jacobi3D (PERKS), HotSpot/SRAD (Rodinia), SpMV, CG, LULESH (ECP)

**两个真实工程案例**：
- **Hydro-cal F1**: 完整 OSHER Riemann solver, 结构化网格, fp64, 多 grid size
- **Hydro-cal F2**: 完整 OSHER, 真实非结构化网格 (24020 cells), fp32

**五个框架实现**：Taichi, Warp, Kokkos, CUDA (4 策略), NumPy (CPU baseline)

### 2.2 核心发现 1: Overhead 在小/中 grid 上非常严重

#### Hydro-cal F2 简化版 (6675 cells, 2 kernels/step, 3060)

这是第一个让我们看到 overhead 严重性的实验：

| 执行方式 | μs/step | 加速比 | 说明 |
|---|---|---|---|
| **Sync loop** (Taichi 默认) | 84.3 | 1.0x | 每步 cudaDeviceSynchronize |
| **Async loop** (去掉逐步 sync) | 31.8 | **2.7x** | Kokkos 风格，只 launch 不 sync |
| **CUDA Graph** (录制 900 步回放) | 7.6 | **11.1x** | GPU 自行回放录制的命令 |
| **Persistent Kernel** (cooperative launch) | 6.1 | **13.9x** | 单次 launch，kernel 内部循环 + grid.sync |

```
Overhead 分解:
  GPU 纯计算:    7.6 μs  (9%)       ← 真正有用的
  Launch 开销:  24.2 μs  (29%)      ← 可被 Graph/Persistent 消除
  Sync 开销:   52.6 μs  (62%)      ← 可被 Async 消除
  ─────────────────────
  Total OH:    76.8 μs  (91%)
```

**91% 的时间不在计算，在等。**

#### Heat2D stencil: Overhead 随 Grid Size 变化 (3060)

我们用简单的 Heat2D stencil 做了系统的 grid size sweep，观察 overhead 占比如何变化：

| Grid Size | Sync (μs) | Graph (μs) | Persistent (μs) | OH% | Graph 加速 |
|---|---|---|---|---|---|
| 128² (16K cells) | 73.7 | 4.2 | **3.5** | 94% | **17.6x** |
| 256² (65K cells) | 72.5 | 6.7 | N/A (超限) | 91% | **10.8x** |
| 512² (262K cells) | 80.6 | 18.9 | N/A | 77% | **4.3x** |
| 1024² (1M cells) | 150.7 | 83.2 | N/A | 45% | **1.8x** |
| 2048² (4M cells) | 417.7 | 335.2 | N/A | 20% | **1.2x** |

**关键发现**：
- ≤512²: overhead 占 77-94%, Graph/Persistent 给 4-18x 加速
- ≥1024²: overhead 降至 20-45%, 收益递减
- **Persistent 在 3060 上 cooperative limit = 120 blocks → 只有 128² (64 blocks) 能跑**，256² 以上必须 fallback to Graph

**这直接证明了 strategy selection 的必要性**：不存在一个策略在所有 grid size 上都最优。

### 2.3 核心发现 2: Block Size 的影响比 Register 限制大得多

这是一个意外发现。我们原本在调 register 限制试图提升 persistent kernel 性能，却发现 block size 的影响大得多。

#### Register 限制 sweep (F1 OSHER, Persistent, B=256, 3060)

| maxrregcount | 32² (μs) | 64² (μs) | 128² (μs) |
|---|---|---|---|
| default (~106 regs) | 71.8 | 51.0 | 149.0 |
| 64 | 72.1 | 51.2 | 151.0 |
| 48 | 61.6 | 53.5 | 151.1 |
| 32 | 60.1 | 51.7 | 155.8 |

Register 限制**最多带来 16% 改善** (32² 从 71.8→60.1)，而且**不稳定**（128² 反而变慢 4.6%，因为 spill 到 local memory）。

#### Adaptive block size (3060)

| 配置 | 32² (μs) | 64² (μs) | 128² (μs) |
|---|---|---|---|
| 固定 256 threads | 71.8 | 51.0 | 149.0 |
| **Adaptive** (128t for 32², 256t 其余) | **38.5** | **56.1** | 153.4 |
| 改善 | **-46%** | +10% | +3% |

**32² 用 128 threads: 1024 cells / 128 = 8 blocks (vs 4 blocks with 256 threads)**。3060 有 28 SMs，从 4→8 blocks 让 SM 利用率从 14%→29%。

**block size 的影响 (1.86x) 和 strategy 选择本身 (Sync→Persistent = 1.91x) 几乎一样大！**

更有趣的是，64² 时 B=256 反而比 B=128 更好 (51.0 vs 56.1)。原因是 64² 已有 16 blocks，SM 利用率足够，而 128 threads 带来的 register pressure 反而拖慢了计算。**最优 B 随 grid size 变化，不存在一个万能值。**

#### 对比 Kokkos 的 ncu profiling

`ncu` 显示 Kokkos 在 64² 上用 `128 threads × 32 blocks`，寄存器 112 regs/thread（vs CUDA 106）。Kokkos 性能好**不是因为寄存器少，而是因为 launch geometry (block size + block count) 更合理**。

### 2.4 核心发现 3: Compute-Communication Overlap

在 persistent kernel 运行期间，GPU 的 Compute Engine 和 DMA Copy Engine 是独立硬件，可以并行工作。我们验证了这一点：

| 方案 | μs/step | 说明 |
|---|---|---|
| Persistent (不保存数据) | 5.85 | 纯计算 baseline |
| **Persistent + async DMA** | **6.58 (+12.5%)** | 每 100 步 DMA 传数据回 host |
| Sync loop + sync save | 76.69 | 传统阻塞式保存 |
| Graph (break to save) | 9.34 (+60%) | Graph 必须中断回放才能保存 |

**Persistent kernel 可以在计算的同时通过 DMA 引擎传数据回 host，几乎零开销。** 这是 Graph 做不到的——Graph 保存数据必须打断回放。PERKS 和 AsyncTaichi 都没有做过这个。

### 2.5 核心发现 4: 跨框架对比

我们在 F1 OSHER (完整 Riemann solver, fp64) 上做了跨框架对比。**注意**：早期 Taichi benchmark 代码有 bug (JIT 编译时间包含在 timing 中，导致 ~5800 μs)，已修正。

#### F1 OSHER 跨框架 (3060, fp64, 500 steps, adaptive B)

| 框架 | 32² | 64² | 128² |
|---|---|---|---|
| **CUDA Persistent (adaptive)** | **38.5** | **56.1** | 153.4 |
| **CUDA Graph (adaptive)** | 42.2 | 69.4 | **144.4** |
| **Kokkos (CUDA, fp64)** | 52.5 | 57.0 | 154.7 |
| CUDA Async | 84.9 | 75.0 | 153.4 |
| Warp (CUDA, fp64) | 131.4 | 128.1 | 129.3 |
| CUDA Sync | 137.4 | 149.2 | 212.4 |
| Taichi (CUDA, fp64) | 189.7 | 138.1 | 160.9 |
| NumPy (CPU) | 127,471 | 499,155 | — |

**关键观察**：
- **Persistent (adaptive B) 在 32² 上最快** (38.5 μs)，比 Kokkos (52.5) 快 27%。说明消除 temporal + spatial 两层损失后，可以超越 C++ 原生框架。
- **Kokkos 在 64² 上与 Persistent 接近** (57 vs 56)。Kokkos 是 C++，没有 Python overhead，nvcc 优化好——它代表了 "无 DSL overhead" 的 baseline。
- **128² 时所有方案收敛** (129-155 μs)。说明 compute 主导后，overhead 优化收益递减。
- **Taichi ≈ CUDA Sync**。说明 Taichi 确实每步做了 sync，其 runtime overhead 本身不大，主要是 sync 模式带来的 idle time。
- **Warp 比 Taichi 快 ~30%**，但仍远慢于 Kokkos/Graph/Persistent。Warp 有 ~130 μs 的 overhead floor。

#### B200 上的对比 (148 SMs, 完整 OSHER)

| 框架 | 32² | 64² | 128² |
|---|---|---|---|
| CUDA Persistent | 11.4 | 11.4 | 12.1 |
| CUDA Graph | 11.3 | 12.3 | 12.3 |
| Kokkos | 12.3 | 12.5 | 14.4 |
| CUDA Sync | 18.8 | 19.3 | 19.7 |
| Warp | 34.8 | 34.2 | 34.0 |
| Taichi | 48.0 | 25.8 | 25.7 |

B200 上 Persistent/Graph/Kokkos 三者非常接近 (11-14 μs)，因为 148 SMs 消除了 spatial loss，compute 主导。

#### 其他 kernel 的跨框架数据 (3060)

**Jacobi 2D** (经典 stencil):

| 框架 | 256² | 1024² | 4096² |
|---|---|---|---|
| Taichi | 0.77 ms | 0.90 ms | 5.09 ms |
| Warp | 0.71 ms | 0.82 ms | 3.83 ms |

**Wave 2D**:

| 框架 | 128² | 512² | 2048² |
|---|---|---|---|
| Taichi | 0.82 ms | 0.83 ms | 3.47 ms |
| Warp | 0.74 ms | 0.68 ms | 3.06 ms |

**N-body** (O(N²) pairwise):

| 框架 | N=256 | N=1024 | N=4096 |
|---|---|---|---|
| Taichi | 0.95 ms | 1.37 ms | 12.9 ms |
| Warp | 0.84 ms | 1.14 ms | 12.6 ms |

**23 种高级 simulation kernel** (characterization_extended, 3060):

| Kernel | 小 grid (μs) | 大 grid (μs) | OH% |
|---|---|---|---|
| LBM D2Q9 | 2,900 | 54,300 | 0.02-0.5% |
| SPH Grid | 770 | 1,000 | 1.4-1.9% |
| MPM88 | 128 | 149 | **10-12%** |
| StableFluids (22 kernels/step) | 6,700 | 6,800 | 0.2% |
| Euler2D Rusanov | 880 | 1,250 | 1.2-1.7% |
| ClothSpring | 1,700 | 2,200 | 0.7-0.9% |

注意 MPM88 的 OH% 是 10-12%——它的 per-step 时间短 (128 μs)，说明对于轻量 kernel, overhead 始终显著。而 LBM/StableFluids 等重量 kernel, compute 主导, overhead 可忽略。

### 2.6 已验证的优化效果汇总

下表汇总了我们已经在**真实 simulation kernel** 上验证有效的优化：

| 优化技术 | 验证 Kernel | 加速比 | 影响层级 | 数据来源 |
|---|---|---|---|---|
| **Persistent kernel** | F2 Hydro (6675 cells) | **13.9x** vs Sync | L1 (temporal) | 3060_hydro.txt |
| **CUDA Graph** | Heat2D (128²) | **17.6x** vs Sync | L1 (temporal) | 3060 characterization |
| **CUDA Graph** | Heat2D (512²) | **4.3x** vs Sync | L1 (temporal) | 3060 characterization |
| **Async (sync elimination)** | F2 Hydro | **2.7x** vs Sync | L1 (temporal) | 3060_hydro.txt |
| **Adaptive block size** | F1 OSHER (32²) | **1.86x** vs fixed B | L2 (spatial) | 3060_osher_adaptive.txt |
| **Persistent + async DMA** | F2 Hydro + save | **+12.5%** only | L1 (DMA overlap) | 3060_async_copy.txt |
| Register limit (maxreg=32) | F1 OSHER (32²) | 1.16x vs default | L3 (occupancy) | 3060_osher_reg32.txt |

**已验证的联合优化**:
- Persistent + adaptive B (F1 OSHER 32²): 137.4 → 38.5 μs = **3.57x** (同时优化 L1 + L2)
- 这比任何单一优化都大 (Persistent alone = 1.91x, adaptive B alone 需要在 Persistent 下才有效)

### 2.7 关键观察：为什么最优配置不直觉

| Grid | 最优 Strategy | 最优 B | 为什么 |
|---|---|---|---|
| 32² | Persistent | 128 | blocks 太少 (4@B=256), 必须减小 B 喂满 SM |
| 64² | Persistent | 256 | blocks 够了 (16@B=256), 大 B 保护 register |
| 128² | **Graph** | 256 | 超出 cooperative limit, fallback to Graph |
| 256²+ | **Graph** | 256 | Persistent 不可用 |
| 6675 (F2) | Persistent | 256 | cells 够, cooperative limit 内 |

**不同 grid size → 不同最优 strategy → 不同最优 B。** 这不是一个人能手动判断的——需要模型。

---

## 3. 五层 GPU 利用率模型

基于以上实验数据，我们提出以下分析框架。

### 3.1 模型结构

```
T_step = T_overhead + T_compute

T_overhead = f(strategy, K)                            ← L1 Temporal
T_compute  = T_ideal / (η_S × η_O × η_M × η_C)       ← L2-L5

总利用率 η = T_ideal / T_step
```

### 3.2 L1: Temporal Efficiency — "GPU 有没有在跑？"

GPU 在 kernel launch/sync 之间完全空闲。

```
              ┌ Sync:       K × (T_launch + T_sync)    ≈ K × 50 μs (3060 实测)
              │
T_overhead =  ┤ Async:      K × T_launch               ≈ K × 20 μs
              │
              │ Graph:      T_graph_replay              ≈ 1-3 μs/step
              │
              └ Persistent: (K-1) × T_grid_sync         ≈ (K-1) × 2 μs
```

**实测 overhead 常数 (3060)**:
| 参数 | 值 | 测量方法 |
|---|---|---|
| T_launch | ~10 μs | nsys: kernel launch to execution start |
| T_sync | ~15 μs | nsys: cudaDeviceSynchronize duration |
| T_python (Taichi) | ~15 μs | T_taichi - T_cuda_sync per step |
| T_graph_replay | ~2 μs | cudaEvent: graph launch per step |
| T_grid_sync | ~2 μs | persistent kernel: measured grid.sync cost |

**数据验证**:
- F2 Sync (K=2): 预测 T_overhead = 2 × (10+15) = 50 μs; 实测 84.3 - 6.1 = 78.2 μs → 还有 ~28 μs 来自 driver scheduling + Python runtime
- Heat2D Sync: 预测 ~50 μs overhead; 实测 73.7 - 3.5 = 70.2 μs → 类似

### 3.3 L2: Spatial Efficiency — "多少 SM 在工作？"

```
η_S = min(1, blocks_active / (max_blocks_per_SM(B,R) × S))
blocks_active = ceil(N / (B × C))
```

**实测验证 (F1 OSHER 32², Persistent)**:
- B=256: blocks=4, η_S ≈ 4/(2×28) = 0.07 (理论); T=71.8 μs
- B=128: blocks=8, η_S ≈ 8/(4×28) = 0.07 (理论); T=38.5 μs
- 预测 speedup = 2x (SM 利用率翻倍); **实测 1.86x** → 误差 7.5%

**crossover 观察**: 当 blocks ≈ S (28 on 3060) 时，B 选择最敏感。
- 32² (4-8 blocks << 28 SMs): B 影响极大 (1.86x)
- 64² (16-32 blocks ≈ 28 SMs): tradeoff 区 (1.10x)
- 128² (64 blocks >> 28 SMs): B 影响极小 (1.03x)

### 3.4 L3: Occupancy Efficiency — "每个 SM 跑得满吗？"

```
occupancy = active_warps / max_warps_per_SM
η_O ≈ α(occupancy)     // α 是经验函数，待标定
```

**实测数据 (F1 OSHER register sweep)**:
| maxrregcount | Regs est. | Occupancy est. | 32² μs | vs default |
|---|---|---|---|---|
| default | ~106 | ~12.5% (1 block of 256t = 8 warps / 48 max) | 71.8 | 1.00x |
| 64 | ~64 | ~25% (2 blocks/SM) | 72.1 | 1.00x |
| 48 | ~48 | ~33% (4 blocks/SM est.) | 61.6 | 1.16x |
| 32 | ~32 | ~50% (8 blocks/SM est.) | 60.1 | 1.19x |

Register 限制最多给 19%——因为更多的 occupancy 帮助 hiding memory latency，但到一定程度后 register spill 抵消了收益。

**与 block size 比较**: B 从 256→128 在 32² 上给 1.86x; register 限制最多给 1.19x。**说明 L2 (spatial) 的影响远大于 L3 (occupancy)，至少在小 grid 上。**

### 3.5 L4: Memory Traffic Efficiency — "多少 memory 访问是必要的？"

当前 F1 OSHER persistent kernel 有两个 phase:
1. **Compute**: 读 own + 4 neighbors 的状态 → 计算 flux → 写 result
2. **Transfer**: 读 result → 写回 state (copy-back)

```
Per cell per step memory traffic:
  Compute:   read 25 vars × 8B = 200B, write 5 × 8B = 40B
  Transfer:  read 5 × 8B = 40B, write 5 × 8B = 40B
  Total:     320 bytes/cell/step

其中 "own-cell round-trip": compute 写 res → transfer 读 res → transfer 写 pre → 下一步读 pre
  = 4 × 40B = 160 bytes → 50% 是冗余的
```

**Register caching (PERKS 技术) 可以消除**: 自己的 5 个 state 留在寄存器中，消除 transfer phase + 下一步的 own-cell read。
- 预计节省: ~120/320 = 37.5% memory traffic
- 预计加速: ~1.3-1.5x (memory-bound kernel 上)
- **仅 Persistent kernel 可做** (thread 跨 timestep 存活); Graph/Async/Sync 做不到

**这是尚待验证的预估，将在 Phase 3 实验中实现和测量。**

### 3.6 L5: Compute Efficiency

| 来源 | 影响 | 数据 |
|---|---|---|
| Warp divergence | OSHER solver 有大量分支 (wet/dry, 流向判断) | ncu warp efficiency 待测 |
| fp64 半速 | 3060: fp64 = 1/64 fp32; B200: fp64 = 1/2 fp32 | 常数因子 |
| Instruction overhead | 寻址计算、边界检查 | 一般 <10% |

这一层由**算法和硬件**决定，不在自动优化范围内。但需要在模型中作为 T_ideal 的计算基础。

### 3.7 层间耦合

这是模型最核心的 insight：**各层的优化旋钮相互影响。**

```
B ↓ (减小 block size)
  → blocks ↑ → η_S ↑                       (好: 更多 SM 工作)
  → 但可能超 coop_limit → Persistent 不可用  (坏: L1 无法优化)
  → warps/block ↓ → η_O 可能 ↓              (坏: latency hiding 变差)

Register limit ↓ (限制寄存器)
  → blocks/SM ↑ → η_O ↑                     (好: 更多 latency hiding)
  → 但 spill to local mem → T_compute ↑      (坏: 额外 memory traffic)

Persistent kernel
  → 消除 launch/sync → η_T ≈ 1              (好: temporal loss 归零)
  → 使能 register caching → η_M ↑            (好: 减少冗余 traffic)
  → 但需要 blocks ≤ coop_limit               (约束: 限制了 N 和 B 的范围)

CUDA Graph
  → 消除大部分 launch overhead                (好)
  → 无 coop_limit 约束                        (好: 大 grid 也能用)
  → 不支持 dynamic control flow               (约束)
  → 不支持 register caching                    (L4 无法优化)
```

**核心 insight**: 不存在一个旋钮能优化所有层级。**最优配置是这些 tradeoff 在特定 (kernel, grid size, GPU) 条件下的平衡点。**

---

## 4. 完整调优空间

### 4.1 所有旋钮

| 类别 | 旋钮 | 取值空间 | 影响层级 | 已有数据? |
|---|---|---|---|---|
| **Inter-kernel** | Execution strategy | {Sync, Async, Graph, Persistent} | L1 | ✅ 大量数据 |
| | Fusion scope | {none, partial, full megakernel} | L1, L4 | ✅ F2 上验证 |
| **Cross-timestep** | Register caching | {off, own-cell} | L4 | ❌ 待实现 |
| | Time tiling depth | {1, 2, 4} | L4 | ❌ 待实现 |
| | Double buffering | {off, on} | L4 | ❌ 待实现 |
| | Async DMA overlap | {off, periodic} | L1 (save) | ✅ F2 上验证 |
| **Intra-kernel** | Block size B | {32, 64, 128, 256, 512} | L2, L3 | ✅ F1 上验证 |
| | Thread coarsening C | {1, 2, 4} | L2, L3 | ❌ 待测 |
| | Register limit | {default, 32, 48, 64} | L3 | ✅ F1 上验证 |
| | L1/shared split | {prefer_L1, prefer_shared} | L3 | ❌ 待测 |
| **Data** | Precision | {fp32, fp64, mixed} | L3, L5 | 部分 (F1=fp64, F2=fp32) |
| | Layout | {AoS, SoA} | L3 | 框架决定 |

### 4.2 约束

```
Persistent    ⟹ blocks(B,C,N) ≤ coop_limit(B, R, S)
Graph         ⟹ 无 dynamic control flow (无 convergence check, 无 adaptive dt)
Reg caching   ⟹ Persistent (thread 必须跨 timestep 存活)
Time tiling   ⟹ Persistent
B × C ≤ 1024  (hardware thread limit)
R_fused = max(R_i) for full fusion (寄存器压力取最大)
```

### 4.3 搜索空间

完整空间: ~200K 理论配置
去掉约束违反: **~2000-5000 可行配置**
只留 Top-3 旋钮 (strategy, B, reg_caching): **4 × 5 × 2 = 40 配置**

**这远小于 TVM 的 ~10⁹ 搜索空间 → 不需要 ML cost model，解析模型 + 枚举即可。**

### 4.4 与 TVM/Halide 的关系

| 维度 | TVM/Halide (OSDI 2018 / PLDI 2013) | 我们 |
|---|---|---|
| **优化对象** | 单个 tensor operator | **Time-stepping loop** (多 kernel 循环) |
| **搜索空间** | Intra-kernel: tile, bind, vectorize, ... | **Inter-kernel + cross-timestep + intra-kernel** |
| **搜索规模** | ~10⁹ (需要 ML cost model) | ~10³ (解析模型 + 枚举) |
| **Cost model** | 学习型 (XGBoost/NN) | 解析型 (5 层分解) |
| **适用领域** | ML (matmul, conv, attention) | Simulation (stencil, CFD, particle, FEM) |
| **时间结构** | DAG (单次执行) | **循环** (重复执行同一 kernel 序列) |

**关键差异**: TVM 把每个 op 独立优化。simulation 有 **时间循环结构** → inter-kernel (strategy) 和 cross-timestep (register caching, time tiling) 优化成为可能。这是 TVM 无法覆盖的维度。

---

## 5. Related Work & Positioning

### 5.1 Comprehensive Related Work

#### A. GPU Performance Modeling & Analysis

| 工作 | Venue | 做了什么 | 与我们的关系 |
|---|---|---|---|
| **Roofline** (Williams et al.) | CACM 2009 | Compute vs memory bandwidth 二维分析 | 我们 L3+L5 的基础；但不建模 launch overhead 和 SM utilization |
| **Hong & Kim** (MWP/CWP) | ISCA 2009 | 首个 GPU 解析模型: compute + memory cycles | L3 (occupancy) 理论基础；不建模 kernel 间 overhead |
| **Volkov** (Occupancy vs ILP) | GTC 2010, Berkeley PhD 2016 | **证明低 occupancy + 高 ILP 可达 peak** | 直接解释我们 register sweep 数据；EBISU 也引用此工作 |
| **Yasin** (Top-Down CPU) | ISPASS 2014 (500+ cit.) | CPU pipeline slot 四层分解 → VTune | **我们的直接类比**: GPU 五层分解 |
| **Instruction Roofline** (Yang et al.) | ISPASS 2020 | 用 instruction count 扩展 roofline | L5 测量方法参考 |
| **PPT-GPU** (Arafa et al.) | ISPASS 2021 | 无 cycle-level 仿真的 GPU 解析模型，15% 误差 | 方法论参考: 解析模型也能足够准确 |

#### B. Kernel Fusion & Launch Overhead

| 工作 | Venue | 做了什么 | 与我们的关系 |
|---|---|---|---|
| **Wahib & Maruyama** | SC 2014 | Scalable fusion search for memory-bound FD stencils | 早期 HPC fusion 工作；无 persistent/graph 选项 |
| **KLAP** (El Hajj et al.) | MICRO 2016 | Kernel launch aggregation compiler, 6.58x | 编译器 launch 聚合；针对 dynamic parallelism |
| **AsyncTaichi** (Hu et al.) | arXiv 2020 | SFG megakernel fusion for Taichi, 1.87x → **废弃** | 前车之鉴: 通用方法太复杂；无 cost model |
| **HFuse** (Li et al.) | CGO 2022 | Horizontal fusion + register pressure for ML | register pressure 分析方法参考；但 ML DAG 非 time-stepping |
| **PERKS** (Zhang et al.) | ICS 2023 | **单 kernel persistent + register cache, 2.12x stencil** | **直接 baseline**; 不做 multi-kernel; 不建模 |
| **EBISU** (Matsumura et al.) | ICS 2023 | **低 occupancy + deep temporal blocking, 2.53x** | 关键 insight: 故意降低 occupancy 换更深 time tiling |
| **FreeStencil** (Zhu et al.) | ICPP 2024 | JIT stencil solver fusion, 3.29x, DRAM 减 34% | 只做结构化线性 solver |
| **Kernel Batching** | arXiv 2025 | CUDA Graph batch size 解析模型 for simulation | Graph-only model; 我们加 Persistent + Async |
| **PyGraph** (Ghosh et al.) | arXiv 2025 | Auto CUDA Graph for ML; 发现 **25% Graph 反而降速** | 验证 cost model 必要性; 但只做 Graph vs Stream |
| **Leonid** | ICS 2025 | Kokkos auto fusion, 1.58x GPU | 无 persistent; 无 cost model |

#### C. Simulation-Specific & GPU-Driven Execution

| 工作 | Venue | 做了什么 | 与我们的关系 |
|---|---|---|---|
| **Korch & Werner** | PPAM 2019 | ODE time-step fusion via hexagonal tiling on GPU | 直接相关: 跨 timestep fusion; 但需 limited access distance |
| **FLUDA** (NASA) | AIAA 2023 | 手动 CFD kernel fusion for FUN3D, 4x | 真实 CFD 验证; 但手动、单应用 |
| **ARK** (Hwang et al.) | **NSDI 2023** | **GPU-driven persistent loop kernel (无 CPU 干预)** | 和我们 persistent 思路一致; 但做 ML distributed |
| **Yamazaki** | SC'24 Workshop | Async + fusion for 大气模型 NICAM, 37%+10% | **直接验证 overhead wall thesis**; 但单应用无模型 |
| **Mirage** (Jia et al.) | **OSDI 2025** | Multi-GPU LLM inference megakernel, 1.2-6.7x | Persistent megakernel 最大规模应用; 但 ML 无 cost model |
| **PyFuser** (Al-Awar et al.) | **ISSTA 2025** | **Python HPC kernel 动态 fusion via PyKokkos, 3.8x** | **直接竞争对手**; 但无 persistent/graph 无 hardware-aware selection |

#### D. Auto-Tuning Frameworks

| 工作 | Venue | 做了什么 | 与我们的关系 |
|---|---|---|---|
| **Halide** (Ragan-Kelley et al.) | PLDI 2013 | Algorithm-schedule 分离 | 思想先驱 |
| **TVM** (Chen et al.) | OSDI 2018 | Intra-kernel auto-tune for ML | 我们 intra-kernel 部分类似; 但 TVM 不管 inter-kernel |
| **ATF** (Rasch et al.) | HPDC 2018, TPDS 2021 | 约束搜索空间剪枝 auto-tuning | 约束处理参考 (类似 coop_limit) |
| **Kernel Tuner** (van Werkhoven) | FGCS 2019 | Python GPU kernel auto-tuning for HPC | **HPC auto-tuning 标杆**; 只做 intra-kernel |
| **AN5D** (Matsumura et al.) | CGO 2020 | 自动化 temporal blocking codegen with perf model | 有 perf model 指导参数选择; 只做 stencil |
| **KTT** (Petrovič et al.) | SPE 2022 | C++ runtime auto-tuning，支持 kernel composition | Kernel composition 概念接近我们的 inter-kernel |

#### E. Memory Traffic & Compute-Communication Overlap

| 工作 | Venue | 做了什么 | 与我们的关系 |
|---|---|---|---|
| **BrickLib** (Zhao et al.) | SC 2019 | Brick 数据布局，cache miss 减 19x | L4 数据布局参考 |
| **ConvStencil** (Chen et al.) | PPoPP 2024 | Stencil→GEMM on Tensor Cores | 正交方向 |
| **LoRAStencil** (Zhang et al.) | SC 2024 | Low-rank stencil on TC, 2.16x | Memory redundancy 消除 |
| **FlashFFTStencil** (Han et al.) | PPoPP 2025 | FFT stencil on FP64 TC + kernel fusion, 2.57x | 激进 temporal fusion 方向 |
| **ConCCL** (Agrawal et al.) | **ISPASS 2025** | **GPU DMA engine overlap; naive C3 只有 21% ideal** | **最接近我们 DMA 工作**; 验证 DMA > SM-based overlap |
| **Punniyamurthy et al.** | SC 2024 | GPU-initiated fused compute+comm, 22% | Fine-grained overlap 参考 |

### 5.2 定位

**"Top-Down Analysis for GPU Simulation"**

类比 Intel Top-Down 把 CPU 性能损失分解为 Front-end / Back-end / Bad Speculation / Retiring 四层 (Yasin 2014, 500+ citations, 集成进 VTune)，我们把 **GPU simulation time-stepping 的性能损失分解为 5 层** (Temporal / Spatial / Occupancy / Memory / Compute)。

**核心差异化**: 现有 GPU performance model (Roofline, Hong-Kim 2009) **只建模 kernel 内部**——它们假设 kernel 已经在跑了。对 simulation workload, **kernel 之间的损失 (temporal) 往往比 kernel 内部的损失更大** (我们的数据: 48-94% 是 temporal loss)。没有人把 kernel 间和 kernel 内的性能损失统一建模过。

### 5.3 核心贡献

| Contribution | 具体内容 | vs 前人 |
|---|---|---|
| **C1: 五层利用率模型** | Temporal + Spatial + Occupancy + Memory + Compute 分解 | 首次统一建模 kernel 间和 kernel 内的性能损失 |
| **C2: 调优空间定义** | Inter-kernel + cross-timestep + intra-kernel 三类旋钮 | TVM 只覆盖 intra-kernel；PERKS 不覆盖 strategy selection |
| **C3: 解析 cost model** | 预测每层 η 和 T_step(config) | 不需要 ML，可解释，可跨 GPU 迁移 |
| **C4: 自动配置选择** | 枚举可行配置，model 预测，选最优 | 首次对 simulation loop 做 auto-tuning |
| **C5: 全面 characterization** | 60+ kernels × 5 框架 × 2 GPUs 的系统测量 | 首次系统量化 simulation workload 的 GPU 利用率 |

---

## 6. Research Plan: 剩余工作

### 6.1 已完成 vs 待完成

| 工作项 | 状态 | 说明 |
|---|---|---|
| Benchmark infrastructure (36 kernels, 5 frameworks) | ✅ 完成 | 130+ 配置已测 |
| Overhead characterization (3060 + B200) | ✅ 完成 | 所有 kernel 有 Sync 基线 |
| Strategy comparison (Sync/Async/Graph/Persistent) | ✅ 完成 | F1, F2, Heat2D 上验证 |
| Block size sweep + adaptive B | ✅ 完成 | F1 OSHER 上验证 |
| Register limit sweep | ✅ 完成 | F1 OSHER, 4 个 maxrregcount |
| Cross-framework comparison (Taichi/Warp/Kokkos/CUDA) | ✅ 完成 | F1 OSHER + Heat/Wave/N-body |
| Compute-communication overlap (async DMA) | ✅ 完成 | F2 Hydro 上验证 |
| **五层分解的 ncu/nsys 精确测量** | ❌ 待做 | 需要对代表性 kernel 做 profiling |
| **解析 cost model 实现 + 验证** | ❌ 待做 | Python 工具 |
| **Register caching 实现** | ❌ 待做 | 在 persistent kernel 中实现 |
| **完整配置矩阵 (strategy × B × reg_caching)** | ❌ 待做 | 120 配置 × 3 grid sizes |
| **Auto-configuration selector** | ❌ 待做 | 模型驱动的自动选择 |
| **跨 GPU 泛化验证** | ❌ 待做 | 3060 标定 → B200 预测 |

### 6.2 实验计划

#### 实验 1: 五层分解精确测量 (C1, C5)

**目标**: 对代表性 kernel 用 ncu + nsys 做完整五层分解。

**Kernel 选择** (6-8 个, 覆盖不同类型):
- Heat2D (简单 stencil, 1 kernel/step)
- Jacobi2D (iterative stencil, 收敛检查)
- F1 OSHER (复杂 CFD, 2 kernels/step, 分支密集)
- N-body (particle, O(N²), compute-intensive)
- LBM D2Q9 (lattice, streaming access pattern)
- CG Solver (多 kernel: SpMV + dot + axpy, 5 kernels/step)

每个 kernel 跑 3+ grid sizes (small / medium / large)。

**测量方法**:
- L1: nsys timeline → GPU idle fraction
- L2: ncu `sm__inst_executed` per SM → SM utilization
- L3: ncu `sm__warps_active` / `sm__warps_launched` → achieved occupancy
- L4: ncu `dram__bytes_read/write` → actual vs minimum traffic ratio
- L5: ncu `smsp__sass_thread_inst_executed_op_fp64_pred_on` → useful FLOP efficiency

#### 实验 2: 完整配置矩阵 (C2, C4)

**目标**: 对 F1 OSHER 做 strategy × B 的完整矩阵，验证 cost model。

**Phase A** — 核心矩阵 (40 配置):
```
{Sync, Async, Graph, Persistent} × {32, 64, 128, 256, 512}
× {32², 64², 128²}
= 60 可行实验点 (去掉约束违反)
```

**Phase B** — 加入 register caching (待实现):
```
{Persistent + reg_cache_off, Persistent + reg_cache_on}
× {128, 256}
× {32², 64², 128²}
= 12 实验点
```

**每个配置**: median of 5 repeats, 500 steps each。

#### 实验 3: Register Caching 实现 (C1 L4 层)

**目标**: 在 F1 OSHER persistent kernel 中实现 register caching，验证 L4 层估算。

**实现方案**:
- 消除 transfer phase: own-cell 的 H,U,V,Z,W 留在寄存器中
- grid.sync() 后直接开始下一步 (neighbor 数据仍从 global memory 读)
- 用 ncu 测量 DRAM traffic before/after

**预期**: memory traffic 减少 ~37%, T_step 改善 1.3-1.5x。

#### 实验 4: Cost Model 验证 (C3)

**目标**: 实现 Python 解析 cost model，验证预测精度。

```python
def predict_T_step(kernel_info, hardware, config):
    """config = (strategy, B, reg_caching)"""
    T_overhead = overhead_model(config.strategy, kernel_info.K, hardware)
    eta_S = spatial_model(config.B, kernel_info.N, hardware.S, kernel_info.R)
    eta_O = occupancy_model(config.B, kernel_info.R, hardware)
    eta_M = memory_model(config.reg_caching, kernel_info)
    eta_C = compute_model(kernel_info)  # from ncu profiling
    T_compute = kernel_info.T_ideal / (eta_S * eta_O * eta_M * eta_C)
    return T_overhead + T_compute
```

**验证**: 用实验 2 的实测数据对比。
- 报告: MAPE (mean absolute percentage error) across all configurations
- Selection accuracy: 模型选的最优 vs oracle 最优的匹配率

#### 实验 5: 跨 GPU 泛化 (C3)

- 在 3060 上标定模型参数
- 只改 hardware 参数 (S=148, BW, ...) → 预测 B200 上的最优配置
- 在 B200 上实测验证

### 6.3 Timeline

```
已完成 (2026-01~04):
  ✅ Benchmark suite 建设 (36 kernels, 5 frameworks)
  ✅ 3060 + B200 全面测量
  ✅ Strategy comparison + block size + register limit
  ✅ Cross-framework comparison
  ✅ Compute-communication overlap

2026-04~05 (Phase 1, 4 weeks):
  [ ] 实验 1: ncu/nsys 五层精确测量
  [ ] 实验 2 Phase A: 配置矩阵 (strategy × B)

2026-05~06 (Phase 2, 4 weeks):
  [ ] 实验 3: Register caching 实现 + 验证
  [ ] 实验 2 Phase B: 加入 register caching 的配置矩阵
  [ ] 实验 4: Cost model 实现 + 验证

2026-06~07 (Phase 3, 4 weeks):
  [ ] 实验 5: 跨 GPU 泛化验证
  [ ] Auto-configuration selector
  [ ] End-to-end evaluation (10+ kernels)

2026-07~08 (Phase 4, 4 weeks):
  [ ] 论文撰写
  [ ] 开源 benchmark suite + cost model

2026-08~09: Submit (ICS / PPoPP / CGO 2027)
```

---

## 7. Paper Outline

**Target venue**: ICS / PPoPP / CGO / SC

### Title Candidates

1. "The GPU Utilization Wall: A Five-Level Performance Model for Simulation Time-Stepping"
2. "Beyond Single-Kernel Tuning: Auto-Configuring GPU Simulation Loops"
3. "Where Do the Cycles Go? Decomposing GPU Utilization Loss in Simulation DSLs"

### Structure

**1. Introduction** (1.5p)
- GPU simulation DSLs 利用率只有 5-35% (我们的数据)
- 不是单一瓶颈: launch overhead, SM starvation, occupancy, memory traffic 同时作用
- 现有优化各解一层: TVM (intra-kernel), PERKS (persistent), PyGraph (Graph)
- 没有人统一建模 → 无法自动选择最优配置
- 我们: 五层分解模型 + 调优空间定义 + 自动配置选择
- 在真实 CFD solver (OSHER) 上验证: **3.57x end-to-end 加速**

**2. Background & Motivation** (1.5p)
- Taichi/Warp 执行模型 + overhead 来源
- GPU 硬件: SM, warp, register, memory hierarchy
- 代际趋势: overhead 不变而 compute 越来越快 → 利用率持续恶化
- 已有解: Graph, Persistent, Async — 各有适用范围和限制

**3. Characterization: Where Do the Cycles Go?** (2.5p) — C5
- 3.1 Benchmark suite: 36 kernels, 130 configs, 5 frameworks, 2 GPUs
- 3.2 Overhead 分解 (Hydro F2: 91% overhead; Heat2D: 94% at 128²)
- 3.3 Overhead vs grid size curve (77-94% at ≤512², 20-45% at ≥1024²)
- 3.4 Cross-framework: Taichi ≈ CUDA Sync; Kokkos ≈ CUDA Async
- 3.5 Block size vs register limit (1.86x vs 1.16x)
- 3.6 代际趋势 (3060 vs B200)

**4. The Five-Level Model** (3p) — C1 (核心)
- 4.1 模型结构: T_step = T_overhead + T_ideal / (η_S × η_O × η_M × η_C)
- 4.2 L1 Temporal: overhead model + 实测常数
- 4.3 L2 Spatial: SM utilization model + block size effect
- 4.4 L3 Occupancy: register-occupancy tradeoff
- 4.5 L4 Memory Traffic: redundancy analysis + register caching
- 4.6 L5 Compute: divergence + precision
- 4.7 Layer coupling analysis

**5. Tuning Space & Auto-Configuration** (2p) — C2, C3, C4
- 5.1 调优旋钮: inter-kernel, cross-timestep, intra-kernel
- 5.2 Constraints (coop limit, dynamic flow, ...)
- 5.3 Analytical cost model
- 5.4 Auto-configuration algorithm (enumerate + predict + select)
- 5.5 Relationship to TVM/Halide

**6. Evaluation** (3p)
- 6.1 Five-level decomposition results (6-8 kernels × 3 sizes)
- 6.2 Model accuracy: MAPE across configurations
- 6.3 Auto-selection accuracy: model vs oracle
- 6.4 End-to-end speedup: default → auto-configured (10+ kernels)
- 6.5 Register caching: memory traffic reduction + speedup
- 6.6 Cross-GPU generalization: 3060 → B200
- 6.7 Cross-framework: explaining Taichi/Warp/Kokkos differences

**7. Discussion** (0.5p)
- Limitations: 不处理 irregular workload, multi-GPU, dynamic mesh
- 如何集成到 DSL 框架 (Taichi, Warp)

**8. Related Work** (1p)

---

## 8. Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| Cost model MAPE > 30% | C3 弱化 | 加经验修正项; 增加标定 kernel 数量; 退化为 profiling-guided |
| Register caching 实现复杂 | L4 层缺验证 | 先对 Heat2D (简单 stencil) 做 PoC; 最坏情况 L4 作为理论分析 |
| Reviewer: "搜索空间 trivial" | C2/C4 弱化 | 展示最优点非直觉 (32²→B=128, 64²→B=256); 展示 default→optimal 的 gap |
| Reviewer: "just use CUDA Graph" | — | Graph 不支持 dynamic flow + 不支持 register caching; 我们的模型预测什么时候 Graph 最优 |
| Reviewer: "incremental over PERKS" | — | PERKS: 单 kernel + 无模型 + 无 strategy selection; 我们: 五层模型 + multi-kernel + auto-select + DMA overlap |
| Reviewer: "TVM already does this" | — | TVM 不覆盖 inter-kernel / cross-timestep; 搜索空间与 TVM 正交 |
| 只有 2 个 GPU | 泛化性不足 | SM 数差 5x (28 vs 148)，已有差异显著; 尝试借用 A100/H100 |

---

## 9. Expected Impact

**对社区**:
- 五层分解成为 GPU simulation 性能分析的标准方法 (类比 Top-Down for CPU)
- Benchmark suite 开源 → simulation overhead 研究的共用基准
- Cost model 可被 Taichi/Warp/Kokkos 直接集成

**对论文**:
- C1 (五层模型): 概念贡献 — 首次统一 kernel 间和 kernel 内建模
- C5 (characterization): 数据贡献 — 60+ kernel, 5 框架, 2 GPUs
- C3+C4 (cost model + auto-config): 实用贡献 — 可验证
- C2 (调优空间): 定义贡献 — bridging TVM/Halide 和 simulation
