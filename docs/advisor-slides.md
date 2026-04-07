# 汇报：GPU Simulation DSL Launch Overhead 研究

**RTX 3060 Laptop GPU (30 SMs) 全部数据**

---

## 1. 起点：跨框架性能对比发现异常

我们用 hydro-cal (浅水方程, OSHER Riemann solver) 做跨框架 benchmark：

| 框架 | 64² (4096 cells) | 128² (16384 cells) | 256² (65536 cells) |
|---|---|---|---|
| Taichi | 1.53 ms | 1.85 ms | 5.57 ms |
| Warp | 1.30 ms | 1.42 ms | 2.86 ms |
| Triton | 3.35 ms | 8.96 ms | 28.2 ms |

**观察**：Taichi 在 64² 到 128² 之间，网格大了 4 倍，但时间只从 1.53 → 1.85 ms，增长 21%。

**异常**：如果 GPU 在做有用计算，时间应该和网格大小成正比。但 64² 和 128² 几乎一样快——说明大部分时间不是在算，而是在做别的事。

---

## 2. 怎么发现是 overhead：用 CUDA 实验分离 compute 和 overhead

写了纯 CUDA 版本，4 种执行策略对比（完整 OSHER solver, fp64, 和 Taichi 对齐）：

| N | Cells | Sync (Taichi模式) | Async (去掉sync) | CUDA Graph | Persistent | **OH占比** |
|---|---|---|---|---|---|---|
| 32 | 1024 | 109.4 μs | 49.5 μs | 40.7 μs | **38.4 μs** | **63%** |
| 64 | 4096 | 157.9 μs | 68.8 μs | — | **69.2 μs** | **56%** |
| 128 | 16384 | 257.5 μs | 155.1 μs | **147.5 μs** | 148.9 μs | **43%** |

**怎么算 overhead**：Graph ≈ 纯 GPU compute（没有 launch/sync 开销）。所以：
```
overhead = Sync - Graph = 109.4 - 40.7 = 68.7 μs
OH 占比 = 68.7 / 109.4 = 63%
→ 即使完整 OSHER solver (106 registers, fp64), overhead 仍占一半以上
```

轻量 kernel (Heat2D) 更极端：

| Grid | Sync | Graph | Persistent | OH占比 |
|---|---|---|---|---|
| 128² | 73.7 μs | 4.2 μs | **3.5 μs (20.8x)** | **94%** |
| 256² | 72.5 μs | **6.7 μs (10.8x)** | N/A (超限) | **91%** |
| 512² | 80.6 μs | **18.9 μs (4.3x)** | N/A | **77%** |
| 1024² | 150.7 μs | **83.2 μs (1.8x)** | N/A | **45%** |

---

## 3. Overhead 是什么：6 层分解

```
每个 timestep，Taichi/Warp 走这条路径：

Python loop  →  DSL runtime  →  CUDA driver  →  GPU launch  →  compute  →  sync  →  Python
   ~3 μs         ~5 μs          ~15 μs                         可变        ~45 μs

3060 上实测 overhead 分解 (hydro-cal):
  Sync overhead (cudaDeviceSynchronize):  52.6 μs   ← 最大的开销!
  Launch overhead (cudaLaunchKernel):     24.2 μs
  GPU compute:                             7.6 μs   ← 实际有用计算
  总计:                                   84.3 μs
  → overhead = 76.7 μs = 91% (简化 solver)
  → overhead = 68.7 μs = 63% (完整 OSHER)
```

---

## 4. 36 种 Kernel 的系统验证

不只是 hydro-cal，我们在 36 种 kernel × 130 个配置上验证了这个问题：

| 领域 | 代表 kernel | per-step (3060 Taichi) |
|---|---|---|
| Stencil | Heat2D 256² | ~84 μs |
| CFD | LBM 256² | ~153 μs |
| Particle | NBody 1024 | ~276 μs |
| EM | FDTD 256² | ~95 μs |
| FEM | ExplicitFEM 8192 elem | ~112 μs |
| Classic | CG Solver 128² | **~2000 μs** (5 kernels/step!) |
| Classic | LULESH 64² | ~202 μs |

**CG Solver 特别严重**：5 个 kernel/step + 2 次 host readback（读 dot product 回 CPU 算 alpha/beta），每步 2000μs 里绝大部分是 overhead。

---

## 5. 发现不同 kernel 需要不同策略

### 关键问题：有些 kernel CUDA Graph 做不了

| Case | 问题 | Graph 能用? | Persistent 能用? |
|---|---|---|---|
| Heat2D | 简单 stencil，无 host 依赖 | ✅ | ✅ (小 grid) |
| LULESH | 4 kernels，无 host 依赖 | ✅ | ✅ (小 grid) |
| **CG Solver** | **5 kernels，每步要读 alpha/beta 回 CPU** | **❌ 不能用** | **✅ 唯一选择** |

CG 为什么 Graph 做不了：
```
CG 每步:
  kernel1: Ap = A*p
  kernel2: rr = dot(r,r), pAp = dot(p,Ap)    ← GPU 算出标量
  → alpha = rr / pAp                          ← 要读回 CPU 算！Graph 不能中断！
  kernel3: x += alpha*p, r -= alpha*Ap         ← 需要 alpha
  kernel4: rr_new = dot(r,r)                   ← 又要读回
  → beta = rr_new / rr                         ← 又要回 CPU！
  kernel5: p = r + beta*p
```

**Persistent fusion 怎么解决**：把 alpha/beta 的计算搬到 GPU 上做（device-side reduction + scalar compute），5 个 kernel 融合成 1 个 cooperative kernel，全程不回 CPU。

### 3060 实测结果

**CG Solver (Graph 不可用，只有 persistent 能做)**：

| N | Sync | Persistent | Speedup |
|---|---|---|---|
| 64² | 502.2 μs | **12.3 μs** | **40.8x** |
| 128² | 486.0 μs | **14.1 μs** | **34.5x** |
| 256² | 482.3 μs | **18.8 μs** | **25.7x** |

**LULESH (Graph 和 persistent 都可以)**：

| N | Sync | Graph | Persistent | Best |
|---|---|---|---|---|
| 32² | 97.2 μs | 7.5 μs | **6.2 μs (15.6x)** | Persistent |
| 64² | 173.8 μs | 7.7 μs | **6.4 μs (27.0x)** | Persistent |
| 128² | 96.0 μs | **9.5 μs (10.1x)** | N/A (超限) | Graph |

LULESH 128² persistent 超出 cooperative limit (66 blocks > 60 max)，自动 fallback 到 Graph。

---

## 6. 新发现：Compute Engine + Copy Engine 可以并行

Persistent kernel 跑的时候，DMA Copy Engine 可以同时把数据拷回 host：

| 方案 | μs/step |
|---|---|
| Persistent (不保存) | 5.85 |
| **Persistent + async DMA 保存** | **6.58 (+12.5%)** |
| Sync loop + 阻塞保存 | 76.69 |

→ 每 100 步保存一次，几乎零开销。PERKS 和 AsyncTaichi 都没做过这个。

---

## 7. 我们提出的方法论

### 自动 Strategy Selection

```
输入: simulation kernel 的特征
  - 每步几个 kernel?
  - 有没有 host readback (标量读回 CPU)?
  - grid size 多大?
  - 需要周期性保存数据吗?

Decision:
  有 host readback?
    YES → grid ≤ cooperative limit?
      YES → Persistent fusion (CG: 40x)
      NO  → Async sync elimination (2x)
    NO  → grid ≤ cooperative limit?
      YES → 比较 Graph vs Persistent (取决于 GPU)
      NO  → Graph (LULESH 128²: 10x)

输出: 推荐策略 + 预测时间 + 预期加速比
```

### Cost Model (MAPE 16%)

用实测数据拟合的解析模型：
```
T_compute = T_fixed + T_per_wave × ceil(blocks / SMs)
T_sync    = T_compute + K × (OH_launch + OH_sync)
T_graph   = T_compute + OH_graph_replay
T_persist = T_compute + K × OH_grid_sync
```

能预测每种策略的时间，自动选最优。

### 已实现的工具

- `strategy_selector.py`: 输入 kernel 参数 → 输出策略推荐
- `sim_optimizer.py`: trace Taichi 循环 → 自动检测 host readback → 推荐策略

---

## 8. 与前人工作的关系

| 前人 | 做了什么 | 我们的差异 |
|---|---|---|
| **AsyncTaichi** (2020, 废弃) | 通用计算图 + megakernel, 1.87x | 分析了废弃原因 (70% 复杂度来自稀疏追踪)；我们只聚焦 time-stepping loop |
| **PERKS** (ICS 2023) | 单 kernel persistent, 2.12x | 我们做 **multi-kernel fusion** (CG 5→1, LULESH 4→1)；PERKS 做不了 CG |
| **PyGraph** (2025) | 自动 CUDA Graph for PyTorch | 我们加 persistent 选项 (CG case)；针对 simulation 不是 ML |

**核心差异**：CG Solver 是前人都做不了的 case——Graph 不能中断，PERKS 不能 fuse 多 kernel。只有我们的 persistent multi-kernel fusion + device-side scalar computation 能做。

---

## 9. 待讨论

1. CG 的 40x 够不够作为核心贡献？
2. 方向：characterization paper 还是 systems paper？
3. 需不需要 hook 进 Taichi runtime 做完整自动化？
