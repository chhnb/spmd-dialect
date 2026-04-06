# 研究进展汇报：GPU Simulation DSL Launch Overhead 研究

## 1. 研究问题

Python GPU simulation DSLs（Taichi, Warp）每个 timestep 都要经过 Python → CUDA driver → GPU launch → sync 的完整路径，产生固定 overhead。当 GPU 计算时间很短（小/中网格）时，overhead 成为性能瓶颈。

**核心问题**：这个 overhead 有多大？什么时候它是瓶颈？能不能自动消除？

## 2. 实验设计

### 2.1 Benchmark Suite
- **36 种 kernel types**（15 种独立计算模式）
- 覆盖 9 个领域：Stencil, CFD, Particle, EM, FEM, Transport, PDE, Classic benchmarks
- 包含经典 benchmark：Jacobi3D (Parboil/PERKS), HotSpot/SRAD (Rodinia), SpMV, CG, LULESH (ECP)
- 每种 kernel 2-6 个 size，**共 130 个配置**

### 2.2 测试平台
| GPU | 定位 | SMs | 架构 |
|---|---|---|---|
| RTX 3060 Laptop | 消费级 | 30 | Ampere (sm_86) |
| B200 | 数据中心 | 148 | Blackwell (sm_90) |

### 2.3 对比的方案
| 方案 | 原理 | 来源 |
|---|---|---|
| Sync loop | 每步 sync（Taichi/Warp 默认行为）| baseline |
| Async loop | 去掉逐步 sync（Kokkos 模式）| 已知技术 |
| CUDA Graph | 录制 launch 序列，GPU 回放 | NVIDIA API (2019) |
| Persistent Kernel | 单次 launch，kernel 内部循环 + grid sync | PERKS (ICS'23) 的扩展 |
| Persistent + DMA | persistent kernel 计算同时 DMA 引擎传数据 | **我们的新发现** |

## 3. 实验结果

### 3.1 核心数据：Hydro-cal 真实工程案例 (6675 cells, 非结构化网格)

| 方案 | 3060 (μs/step) | 3060 加速 | B200 (μs/step) | B200 加速 |
|---|---|---|---|---|
| Sync loop (Taichi默认) | 84.3 | 1.0x | 15.2 | 1.0x |
| Async loop | 31.8 | 2.7x | 8.2 | 1.9x |
| CUDA Graph | 7.6 | **11.1x** | 5.3 | **2.9x** |
| Persistent Kernel | 6.1 | **13.9x** | 5.7 | **2.7x** |

Overhead 分解：

| | 3060 | B200 |
|---|---|---|
| GPU compute（Graph 下界）| 7.6 μs | 5.3 μs |
| Launch overhead | 24.2 μs | 2.9 μs |
| Sync overhead | 52.6 μs | 7.0 μs |
| **Total overhead** | **76.7 μs (91%)** | **9.9 μs (65%)** |

### 3.2 4 策略对比（结构化网格 Heat2D, 3060）

| Grid Size | Sync | Async | Graph | Persistent | OH% |
|---|---|---|---|---|---|
| 128² | 73.7 | 32.7 | 4.2 | **3.5 (20.8x)** | 94% |
| 256² | 72.5 | 30.9 | **6.7 (10.8x)** | N/A (超限) | 91% |
| 512² | 80.6 | 33.1 | **18.9 (4.3x)** | N/A | 77% |
| 1024² | 150.7 | 87.2 | **83.2 (1.8x)** | N/A | 45% |
| 2048² | 417.7 | 338.2 | **335.2 (1.2x)** | N/A | 20% |

**发现**：
- 3060 上 persistent 只能跑 128²（64 blocks < cooperative limit 120）
- 256² 以上必须用 Graph
- **Grid size 越小，overhead 占比越高，优化收益越大**

### 3.3 Compute-Communication Overlap 验证 (3060)

| 方案 | μs/step | 说明 |
|---|---|---|
| Persistent (不保存) | 5.85 | 纯计算 |
| **Persistent + async DMA** | **6.58 (+12.5%)** | 每 100 步保存一次，几乎零开销 |
| Sync loop + sync save | 76.69 | 传统方式 |

**验证了 GPU Compute Engine 和 DMA Copy Engine 可以并行**。Persistent kernel 在计算的同时，DMA 引擎可以把数据拷回 host，不需要中断 kernel。

### 3.4 36 Kernel Characterization（3060, Taichi per-step 时间）

| 分类 | B200 | 3060 | 说明 |
|---|---|---|---|
| 单 kernel stencil ≤512² | ~15 μs | ~80-90 μs | 3060 compute 更慢 |
| 单 kernel stencil 1024² | ~17 μs | ~90-130 μs | |
| 单 kernel stencil 2048² | ~21 μs | ~260 μs | compute 开始主导 |
| CG Solver (5 kern/step) | ~264 μs | ~2000 μs | 多 kernel overhead 累积 |
| StableFluids (22 kern/step) | ~328 μs | ~2000 μs | 22 次 launch 开销巨大 |
| LULESH-like (3 kern/step) | ~41 μs | ~210 μs | |

**注意**：Taichi 的 characterization 数据包含了 compute + overhead 的总和。真正的 overhead 分离需要用 CUDA Graph baseline（实验 3.2）来做。

### 3.5 Overhead 占比 vs Grid Size（两个 GPU 对比）

```
Overhead fraction (Heat2D, from CUDA experiments):

Grid    3060      B200
128²    94%       78%
256²    91%       75%
512²    77%       64%
1024²   45%       37%
2048²   20%       15%

→ 小/中网格 (≤512²): overhead = 64-94%
→ 大网格 (≥2048²): overhead = 15-20%
→ 3060 的 overhead 更高（driver/sync 开销更大）
```

## 4. 关键发现

### 发现 1：Overhead 在所有 GPU 上都很严重，不只是数据中心 GPU
- 3060 (消费级): overhead = 77-94% (≤512²)
- B200 (数据中心): overhead = 64-78% (≤512²)
- 3060 的 driver/sync 开销 (~77μs) 比 B200 (~10μs) 大 7 倍
- **但两者的纯 GPU compute 接近**（3060: 7.6μs vs B200: 5.3μs for hydro-cal）

### 发现 2：Persistent Kernel 在消费级 GPU 上收益更大
- 3060: persistent = **13.9x** (因为 overhead 更大，消除后收益更大)
- B200: persistent = 2.7x
- 但 3060 的 cooperative launch limit 更小 (120 blocks vs 444 blocks)

### 发现 3：Strategy Selection 被数据验证
- 3060 上 Heat2D 128² → persistent 可用且最优 (20.8x)
- 3060 上 Heat2D 256² → persistent 不可用，自动 fallback to Graph (10.8x)
- 选择策略取决于 grid size vs cooperative limit

### 发现 4：Compute-Communication Overlap 跨 GPU 通用
- B200: persistent + DMA overhead = 5.3%
- 3060: persistent + DMA overhead = 12.5%
- 都远好于 sync save（阻塞式传输）
- 前人工作（PERKS, AsyncTaichi）都没有做过这个

## 5. 与前人工作的对比

| 前人工作 | 做了什么 | 我们的差异 |
|---|---|---|
| **AsyncTaichi** (2020, 废弃) | 通用计算图 + megakernel fusion, 1.87x | 分析了其废弃原因（稀疏 SNode 追踪占 70% 复杂度）；我们聚焦 time-stepping loop |
| **PERKS** (ICS 2023) | 单 kernel persistent, 2.12x stencil | 我们做 **multi-kernel fusion**（CG 5个/LULESH 3个 kernel 融合为1个）+ DMA overlap |
| **PyGraph** (2025, 预印本) | 自动 CUDA Graph for PyTorch | 我们加入 persistent 选项 + 针对 simulation |

## 6. 研究方向（讨论）

### 方向 A：Characterization + Analysis Paper
- 36 kernel × 2 GPU 的全面测量
- 代际/跨档次对比
- AsyncTaichi 废弃原因分析
- 方案对比 + 设计指南
- 投 ISPASS / IISWC / SC analysis track

### 方向 B：Cost-Benefit Model + Strategy Selection
- 建 fusion cost-benefit model（什么时候 fusion 值得）
- 模型驱动的自动策略选择（Graph vs Persistent vs Async）
- 60+ kernel 验证模型精度
- 投 ICS / PPoPP / CGO

### 方向 C：聚焦 Persistent + DMA Overlap（最小可行贡献）
- 只聚焦 compute-communication overlap 这一个新技术点
- PERKS/AsyncTaichi 没做的
- 系统性实验验证
- 投 workshop / short paper

## 7. 待讨论问题

1. 三个方向哪个最合适？
2. 是否需要更多 GPU 型号的数据（A100/H100）来加强代际对比？
3. 是否需要实现自动化工具，还是只做实验分析？
