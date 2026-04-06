# 研究进展汇报：GPU Simulation DSL Launch Overhead

**测试平台**: RTX 3060 Laptop GPU (30 SMs, Ampere sm_86)

---

## 1. 研究问题

Python GPU simulation DSLs（Taichi, Warp）每个 timestep 都要经过 Python → CUDA driver → GPU launch → sync 的路径。这个固定 overhead 在 kernel 计算量小时成为主要瓶颈。

---

## 2. Benchmark 覆盖

### 2.1 Kernel 类型
- **36 种 kernel types**（15 种独立计算模式）
- 覆盖 9 个领域：Stencil, CFD, Particle, EM, FEM, Transport, PDE, Classic
- 经典 benchmark：Jacobi3D (PERKS), HotSpot/SRAD (Rodinia), SpMV, CG, LULESH (ECP)
- **~130 个测试配置** (不同 grid size)

### 2.2 真实工程案例
- **Hydro-cal F1**: 完整 OSHER Riemann solver, 结构化网格, fp64
- **Hydro-cal F2**: 完整 OSHER, 真实非结构化网格 (24020 cells), fp32
- 跨框架：Taichi / Warp / Kokkos / Triton / CUDA

---

## 3. 核心实验数据 (RTX 3060)

### 3.1 Overhead 分解：Hydro-cal (6675 cells, 2 kernels/step)

| 方案 | μs/step | 加速比 | 说明 |
|---|---|---|---|
| Sync loop (Taichi 默认模式) | 84.3 | 1.0x | 每步 cudaDeviceSynchronize |
| Async loop (去掉逐步 sync) | 31.8 | 2.7x | Kokkos 风格 |
| CUDA Graph (录制 900 步回放) | 7.6 | **11.1x** | GPU 自行回放 |
| Persistent Kernel (cooperative grid sync) | 6.1 | **13.9x** | 单次 launch，内部循环 |

```
Overhead 分解:
  GPU 纯计算:    7.6 μs  (9%)
  Launch 开销:  24.2 μs  (29%)
  Sync 开销:   52.6 μs  (62%)
  ─────────────────────
  Total OH:    76.8 μs  (91%)
```

**91% 的时间不在计算，在等。** Graph 和 Persistent 可以消除绝大部分 overhead。

### 3.2 Overhead 随 Grid Size 变化 (Heat2D stencil, 3060)

| Grid Size | Sync (μs) | Graph (μs) | Persistent (μs) | OH% | Graph 加速 |
|---|---|---|---|---|---|
| 128² (16K cells) | 73.7 | 4.2 | **3.5** | 94% | **17.6x** |
| 256² (65K cells) | 72.5 | 6.7 | N/A (超限) | 91% | **10.8x** |
| 512² (262K cells) | 80.6 | 18.9 | N/A | 77% | **4.3x** |
| 1024² (1M cells) | 150.7 | 83.2 | N/A | 45% | **1.8x** |
| 2048² (4M cells) | 417.7 | 335.2 | N/A | 20% | **1.2x** |

**趋势**: Grid 越小 → overhead 占比越高 → 优化收益越大。
- ≤512²: OH = 77-94%, Graph 给 4-18x
- ≥1024²: OH = 20-45%, 收益递减

**Persistent kernel 的限制**: 3060 cooperative limit = 120 blocks → 只有 128² (64 blocks) 能跑。256² 以上必须 fallback to Graph。这证明了 **strategy selection 的必要性**。

### 3.3 Compute-Communication Overlap (Persistent + DMA)

| 方案 | μs/step | 说明 |
|---|---|---|
| Persistent (不保存数据) | 5.85 | 纯计算 baseline |
| **Persistent + async DMA** | **6.58 (+12.5%)** | 每 100 步保存一次，DMA 引擎并行传输 |
| Sync loop + sync save | 76.69 | 传统阻塞式保存 |
| Graph (break to save) | 9.34 | Graph 必须中断回放才能保存 |

**GPU 的 Compute Engine 和 DMA Copy Engine 是独立硬件**，persistent kernel 在计算的同时可以用 DMA 传数据回 host，几乎零开销。这是 PERKS 和 AsyncTaichi 都没有做的。

### 3.4 跨框架对比：Hydro-cal F1 (完整 OSHER, 3060)

统一口径：**μs/step, fp64, 500 steps**。Taichi/Warp 数据来自修正后的 `run_all_frameworks.py`（已排除 JIT 编译时间），Kokkos 由框架脚本输出的总时间除以 500 steps，CUDA 4 策略来自 `hydro_cuda_osher` (adaptive threads v2)。

| 框架 | 32² | 64² | 128² |
|---|---|---|---|
| **CUDA (Persistent, adaptive)** | **38.5** | **56.1** | 153.4 |
| **CUDA (Graph, adaptive)** | 42.2 | 69.4 | **144.4** |
| **Kokkos (CUDA, fp64)** | 52.5 | 57.0 | 154.7 |
| CUDA (Async) | 84.9 | 75.0 | 153.4 |
| Warp (CUDA, fp64) | 131.4 | 128.1 | 129.3 |
| CUDA (Sync) | 137.4 | 149.2 | 212.4 |
| Taichi (CUDA, fp64) | 189.7 | 138.1 | 160.9 |

> **注**: 早期 Taichi 数据 (~5800 μs) 包含了 `ti.init()` + JIT 编译时间，是 benchmark 代码 bug。修正后 Taichi 实际性能 ~140-190 μs/step，与 CUDA Sync 在同一量级。

结论：
- **CUDA Persistent (adaptive threads) 在 32² 上最快** (38.5 μs)，比 Kokkos (52.5) 快 27%
- **Kokkos** 在 64² 上与 Persistent 接近 (57 vs 56)，因为 C++ 无 Python overhead + nvcc 编译质量好
- **Warp** 比 Taichi 快 ~30%，但两者都远慢于 Kokkos/Graph/Persistent
- **128² 时所有方案收敛** (129-155 μs)，compute 开始主导
- **Taichi ≈ CUDA Sync**：说明 Taichi 确实每步做了 sync，其 runtime overhead 本身不大

### 3.5 跨框架对比：其他 kernel (3060)

**Jacobi 2D** (经典 stencil):

| 框架 | 64² | 256² | 1024² | 4096² |
|---|---|---|---|---|
| Taichi | — | 0.77 ms | 0.90 ms | 5.09 ms |
| Warp | — | 0.71 ms | 0.82 ms | 3.83 ms |

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

---

## 4. 关键发现

### 发现 1: Overhead 在 RTX 3060 上非常严重
- 小/中 grid (≤512²): **overhead 占 77-94%**，GPU 大部分时间在空等
- 即使是 hydro-cal 这种有 OSHER solver 的重量 kernel: overhead 仍占 91% (简化版) 到 ~40% (完整版)
- Persistent kernel 给出 **最高 20.8x** 加速 (Heat2D 128²)

### 发现 2: Strategy Selection 是必要的
- 128² (64 blocks): persistent 可用且最优 (20.8x)
- 256² (256 blocks): persistent **不可用** (超出 cooperative limit 120)，必须用 Graph (10.8x)
- ≥1024²: overhead 占比降至 <50%，收益递减
- **需要根据 grid size 和硬件参数自动选择策略**

### 发现 3: Compute-Communication Overlap 可行
- Persistent kernel + DMA 并行: 保存数据仅 +12.5% 开销
- vs Graph (需要中断): +60% 开销
- vs Sync (阻塞): +1211% 开销
- **PERKS 和 AsyncTaichi 都没做过这个**

### 发现 4: Taichi/Warp 有固定 overhead floor
- 3060 上 Taichi 单 kernel stencil: ~80-90 μs (不管 grid 多小)
- 而 CUDA Graph 在相同 kernel 上只需 4-7 μs
- **DSL 的 Python runtime + CUDA driver overhead 是固定的，与计算量无关**

### 发现 5: 小网格下 block size 比寄存器限制更关键

**Register 限制 sweep (Persistent, 3060)**:

| maxrregcount | 32² (μs) | 64² (μs) | 128² (μs) |
|---|---|---|---|
| default (~106 regs) | 71.8 | 51.0 | 149.0 |
| 64 | 72.1 | 51.2 | 151.0 |
| 48 | 61.6 | 53.5 | 151.1 |
| 32 | 60.1 | 51.7 | 155.8 |

Register 限制最多带来 ~16% 改善 (32² 从 71.8→60.1)，且不稳定（128² 反而变慢，spill 到 local memory）。

**Adaptive block size (32² 用 128 threads，其余用 256)**:

| 配置 | 32² (μs) | 64² (μs) | 128² (μs) |
|---|---|---|---|
| 固定 256 threads | 71.8 | 51.0 | 149.0 |
| **Adaptive threads** | **38.5** | **56.1** | 153.4 |
| 改善 | **-46%** | +10% | +3% |

32² 改善原因：1024 cells / 256 threads = **4 blocks**，3060 有 30 SMs → 大量 SM 空闲。改为 128 threads → 8 blocks → SM 利用率翻倍。

对比 Kokkos：
- `ncu` 显示 Kokkos 64² 用 `128 threads x 32 blocks`，寄存器 112 regs/thread（vs CUDA 106）
- Kokkos 性能好不是因为寄存器少，而是因为 **launch geometry 更合理**
- 自适应后 CUDA Persistent 反超 Kokkos: `38.5 μs vs 52.5 μs`

**结论**: 在小 grid / 轻 workload 上，**block size 是必须建模的策略输入**。单纯限制 `maxrregcount` 不能稳定带来收益，但正确的 launch geometry 可以给 ~2x。这是 compiler / cost model 需要自动决策的。

---

## 5. 与前人工作的定位

| 前人 | 做了什么 | 我们的差异 |
|---|---|---|
| **AsyncTaichi** (2020, 废弃) | SFG 计算图 + megakernel fusion, 1.87x | 废弃了。分析废弃原因：70% 复杂度来自稀疏 SNode 追踪 |
| **PERKS** (ICS 2023) | 单 kernel persistent, 2.12x stencil | 我们做 **multi-kernel fusion** + **DMA overlap** + **strategy selection** |
| **PyGraph** (2025 预印本) | Auto CUDA Graph for PyTorch ML | 我们加入 persistent 选项，针对 simulation |

---

## 6. 研究方向（请老师指导）

### 方向 A: Cost-Benefit Model + Strategy Selection
- 建 fusion cost-benefit model: 什么时候 persistent vs Graph vs async 最优
- 输入: kernel 寄存器数、**block size**、grid size、SM 数、cooperative limit → 预测最优策略
- 先解决小 grid launch geometry，再决定要不要 graph / persistent
- 用 36 种 kernel × 多 size 验证模型精度
- 投 ICS / PPoPP / CGO

### 方向 B: Characterization + Analysis Paper
- 36 kernel × 跨框架的全面测量
- AsyncTaichi 废弃原因分析
- 设计指南: 什么情况用什么策略
- 投 ISPASS / IISWC / SC analysis

### 方向 C: 聚焦 Persistent + DMA Overlap
- Compute-Communication overlap 的系统研究
- 这是 PERKS/AsyncTaichi 都没做的新技术点
- 投 workshop / short paper

---

## 7. 汇报重点（3 张 slides）

### Slide 1: 数据
- Hydro-cal F2: `84.3 μs/step` 中有 `91%` 是 launch + sync overhead，不在计算
- Heat2D 曲线：overhead 从 `94% (128²)` 降到 `20% (2048²)`
- 结论：**overhead 是否主导，强烈依赖 kernel size / phase**

### Slide 2: 方案
- CUDA Graph: hydro 上 `11.1x` (F2), `3.25x` (F1 32²)
- Persistent kernel: hydro 上 `13.9x` (F2), `3.57x` (F1 32²)
- Persistent + adaptive block size: F1 32² 从 71.8→38.5 μs，**比 Kokkos (52.5) 还快 27%**
- Register 限制不稳定（最多 16%），但 block size 给 46% — **launch geometry 才是关键旋钮**
- Persistent + async DMA: 只比 no-save 多 `12.5%`
- 结论：**Graph / Persistent / DMA overlap 都有效，但适用区间不同；block size 是被忽略的大旋钮**

### Slide 3: 问题
- **方向 A**: 做 cost model / strategy selection（把 register、block size、grid size 都纳入）
- **方向 B**: 做大规模 characterization / analysis
- **方向 C**: 聚焦 persistent + DMA overlap 这个技术点
- 希望老师帮助判断：先走哪条路线最合适

---

## 8. 待完成

- [x] 在 3060 上跑 F1 CUDA OSHER (hydro_cuda_osher.cu) → 与 Taichi/Warp 直接对比
- [ ] 补 MacCormack 3D (AsyncTaichi benchmark)
- [ ] 补 MPM 的跨框架对比
- [ ] 如有机会在 A100/H100 上跑 → 代际对比数据
