# 研究进展汇报：GPU Simulation DSL Launch Overhead

## 1. 研究问题

Python GPU simulation DSLs（Taichi, Warp）每个 timestep 经过 Python → CUDA driver → GPU launch → sync 的完整路径，产生固定 overhead。当 GPU 计算时间很短时，overhead 成为性能瓶颈。

**核心问题**：overhead 有多大？什么时候它是瓶颈？能不能自动消除？

## 2. 实验设计

### Benchmark Suite
- **36 种 kernel types**（15 种独立计算模式），覆盖 9 个领域
- 包含经典 benchmark：Jacobi3D (Parboil/PERKS), HotSpot/SRAD (Rodinia), SpMV, CG, LULESH (ECP)
- 共 **130 个配置**

### 测试平台
| GPU | SMs | 定位 |
|---|---|---|
| RTX 3060 Laptop | 30 | 消费级 |
| B200 | 148 | 数据中心 |

### 对比方案
| 方案 | 原理 |
|---|---|
| Sync loop | 每步 sync（Taichi/Warp 默认）|
| Async loop | 去掉逐步 sync（Kokkos 模式）|
| CUDA Graph | 录制 launch 序列，GPU 回放 |
| Persistent Kernel | 单次 launch，kernel 内部循环 + grid sync |
| Persistent + DMA | Compute Engine ∥ Copy Engine 并行保存数据 |

## 3. 实验结果

### 3.1 完整 OSHER Solver（真实工程 kernel, fp64）

**已和 Taichi/Kokkos 实现对齐：完整 OSHER Riemann solver + 结构化网格。**

#### B200 (148 SMs)

| N | Cells | Sync (μs) | Async (μs) | Graph (μs) | Persistent (μs) | OH% |
|---|---|---|---|---|---|---|
| 32 | 1024 | 19.4 | 14.3 | 12.3 | **11.1 (1.7x)** | 37% |
| 64 | 4096 | 19.7 | 14.3 | 12.3 | **11.4 (1.7x)** | 38% |
| 128 | 16384 | 20.1 | 14.3 | **12.3 (1.6x)** | 12.3 (1.6x) | 39% |

B200 太强——16K cells 的 compute 几乎为零，全是 overhead。

#### 3060 (30 SMs)

| N | Cells | Sync (μs) | Async (μs) | Graph (μs) | Persistent (μs) | OH% |
|---|---|---|---|---|---|---|
| 32 | 1024 | 109.4 | 49.5 | 40.7 | **38.4 (2.85x)** | 63% |
| 64 | 4096 | 157.9 | 68.8 | 100.7 | **69.2 (2.28x)** | 56% |
| 128 | 16384 | 257.5 | 155.1 | **147.5 (1.75x)** | 148.9 | 43% |

**即使用完整 OSHER solver (106 registers/thread, fp64), overhead 仍占 43-63%。**

### 3.2 轻量 Kernel (Heat2D, 3060)

| Grid Size | Sync (μs) | Graph (μs) | Persistent (μs) | OH% |
|---|---|---|---|---|
| 128² | 73.7 | 4.2 | **3.5 (20.8x)** | 94% |
| 256² | 72.5 | **6.7 (10.8x)** | N/A (超限) | 91% |
| 512² | 80.6 | **18.9 (4.3x)** | N/A | 77% |
| 1024² | 150.7 | **83.2 (1.8x)** | N/A | 45% |
| 2048² | 417.7 | **335.2 (1.2x)** | N/A | 20% |

### 3.3 Overhead 占比 vs 问题规模 (两 GPU 综合)

```
轻量 kernel (Heat2D):
  ≤512²:   OH = 64-94%  → Graph/Persistent 给 4-20x
  1024²:   OH = 37-45%  → Graph 给 1.8x
  ≥2048²:  OH = 15-20%  → 优化收益递减

重量 kernel (OSHER, 106 regs, fp64):
  ≤64²:    OH = 56-63%  → Persistent 给 2.3-2.9x
  128²:    OH = 39-43%  → Graph 给 1.6-1.8x
```

**结论：不是"只有轻量 kernel 受 overhead 影响"。即使 106 寄存器的重量 OSHER solver，在 1K-16K cells 的网格上 overhead 仍占 40-60%。**

### 3.4 Compute-Communication Overlap (3060)

| 方案 | μs/step | 说明 |
|---|---|---|
| Persistent (不保存) | 5.85 | 纯计算 |
| **Persistent + async DMA** | **6.58 (+12.5%)** | 每 100 步保存，几乎零开销 |
| Sync loop + sync save | 76.69 | 传统方式 |

验证了 Compute Engine 和 DMA Copy Engine 在 persistent kernel 中可以并行。PERKS 和 AsyncTaichi 都没有做过这个。

### 3.5 Register Tuning on 3060 (OSHER, N=64)

| maxrregcount | Graph (μs) | 变化 |
|---|---|---|
| default (106) | 51.6 | baseline |
| 64 | 57.2 | **-11% (更慢)** |
| 48 | 60.8 | **-18% (更慢)** |
| 32 | 51.0 | ≈持平 |

**3060 上 register tuning 对 OSHER 无效**（可能因为 spill to local memory）。和 B200 上 1.4x 的结果不同。说明 register tuning 的效果**高度依赖 GPU 架构**。

### 3.6 NCU Profiling Summary (3060, 5 kernel types)

| Kernel | Regs/Thread | SM Through% (128²) | Occupancy% (128²) | 计算密度 |
|---|---|---|---|---|
| jacobi2d | 16 | 7.7 | 33.3 | 低 |
| heat2d | 18 | 7.7 | 33.1 | 低 |
| hotspot | 20 | 7.9 | 33.2 | 低 |
| srad | 26 | 14.5 | 32.0 | 中 |
| **osher** | **106** | **60.6** | **31.3** | **高** |

OSHER 用了 106 registers/thread（6× heat2d），SM throughput 高 8× → 这是 compute-intensive kernel 的典型特征。

### 3.7 Cost Model (初步)

| 指标 | 值 |
|---|---|
| MAPE (时间预测误差) | 59.4% |
| Strategy accuracy (选对策略) | 64% (9/14) |

模型能正确判断 regime（OH-dominated vs compute-dominated），但时间预测精度需要继续改进。

## 4. 关键发现

### 发现 1：Overhead 在两种 GPU 上都是问题
- 3060: overhead = 56-94% (OSHER-Heat2D, ≤512²)
- B200: overhead = 37-78% (OSHER-Heat2D, ≤512²)
- 3060 的 driver/sync 开销更大 (~77μs vs ~10μs)，但 compute 也更慢

### 发现 2：即使重 kernel 也有显著 overhead
- OSHER (106 regs, fp64, 完整 Riemann solver) 在 3060 上仍有 43-63% overhead
- 这不是只有 trivial stencil 才有的问题

### 发现 3：Strategy selection 被验证
- 3060 Heat2D 128² → persistent 最优 (20.8x)
- 3060 Heat2D 256² → persistent 不可用 (超 cooperative limit), fallback to Graph (10.8x)
- OSHER 大网格 → Graph 最优
- OSHER 小网格 → Persistent 最优

### 发现 4：Compute-Communication overlap 跨 GPU 通用
- B200: +5.3% overhead
- 3060: +12.5% overhead
- 前人 (PERKS, AsyncTaichi) 都没做过

### 发现 5：Register tuning 效果因 GPU 而异
- B200: -maxrregcount=64 给 OSHER 1.4x
- 3060: 无效甚至更慢（spill penalty）
- 说明优化策略不能假设跨 GPU 通用

### 发现 6：两种 regime 需要不同优化
| Regime | 代表 | 瓶颈 | 优化方案 |
|---|---|---|---|
| OH-dominated (≤512²) | Heat2D, Jacobi, 小网格 OSHER | launch/sync overhead | Graph / Persistent |
| Compute-dominated (≥1024²) | 大网格 OSHER, N-body | register pressure / occupancy | Register tuning (仅限特定 GPU) |

## 5. 与前人工作的关系

| 前人 | 做了什么 | 我们的差异 |
|---|---|---|
| **AsyncTaichi** (2020, 废弃) | 通用 SFG + megakernel, 1.87x | 分析了其废弃原因；我们聚焦 time-stepping |
| **PERKS** (ICS 2023) | 单 kernel persistent, 2.12x stencil | 我们: multi-kernel (OSHER=flux+update) + DMA overlap + strategy selection |
| **PyGraph** (2025 预印本) | Auto CUDA Graph for PyTorch | 我们加入 persistent 选项 + simulation + cost model |

## 6. 下一步讨论

### 选项 A：Cost-Benefit Model Paper
- 建模 "fusion 什么时候值得" (overhead vs occupancy penalty)
- 模型驱动策略选择 (Graph/Persistent/Async)
- 投 ICS / PPoPP / CGO

### 选项 B：Characterization Paper
- 36 kernel × 2 GPU 的全面测量
- 代际/跨档次对比 + AsyncTaichi 废弃分析
- 投 ISPASS / IISWC / SC analysis track

### 选项 C：Persistent + DMA Overlap 技术点
- 聚焦 compute-communication overlap 这一新发现
- 系统性验证
- 投 workshop / short paper
