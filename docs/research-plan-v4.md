# Research Plan v4: To Fuse or Not to Fuse

**Version:** 4.0
**Date:** 2026-04-05
**Status:** Draft for advisor discussion
**Supersedes:** research-plan-v1 (SPMD IR), v2/v3 (Overhead Wall)

---

## 1. One-Sentence Summary

> We build a performance model that predicts when fusing GPU simulation kernels into persistent mega-kernels helps or hurts, enabling automatic strategy selection (persistent fusion / CUDA Graph / async launch) that recovers up to 2.9x GPU utilization for Python simulation DSLs — the first work to systematically model this cost-benefit tradeoff.

---

## 2. Problem

### 2.1 The Overhead Wall (Motivation)

Python GPU simulation DSLs (Taichi, Warp) pay ~10-15 μs fixed overhead per timestep. For typical engineering meshes (1K-100K cells), GPU compute is only 2-10 μs. **Overhead exceeds compute.** This worsens each GPU generation:

| GPU | Compute (μs) | Overhead (μs) | GPU Utilization |
|---|---|---|---|
| V100 (2017) | ~26 | ~10 | 73% |
| B200 (2024) | ~5 | ~10 | **35% (measured)** |
| 2028 (proj.) | ~1.3 | ~10 | 12% |

90% of 60+ benchmark kernel configs are overhead-dominated at typical mesh sizes.

### 2.2 Three Existing Solutions, No Guidance on When to Use Which

| Solution | Speedup | Limitation |
|---|---|---|
| **CUDA Graph** | 2.9x | Static only (no convergence check, no adaptive dt) |
| **Persistent kernel** (PERKS) | 2.7x | Grid ≤ SM × occupancy; register pressure increases |
| **Async (sync elimination)** | 1.9x | Still has per-kernel launch cost |

**The open question**: given a specific simulation loop, which strategy is best? **Nobody has a model for this.**

### 2.3 The Fusion Dilemma

Fusing multiple kernels into one persistent kernel creates a three-way tradeoff:

```
         ┌─ Benefit: 消除 launch overhead (save ~10 μs/step)
         │
Fusion ──┼─ Cost 1: R_fused = max(R_1..R_K) → occupancy ↓ → compute slower
         │
         ├─ Cost 2: grid ≤ SM × occ_fused → 大网格不可用
         │
         └─ Cost 3: grid = max(G_1..G_K) → 小 phase 浪费线程
```

**When does the benefit outweigh the costs?** This depends on:
- Each kernel's register count, grid size, execution time
- GPU architecture (SM count, register file size)
- Number of kernels per step
- Whether the loop has dynamic control flow

**No prior work has modeled this tradeoff.**

---

## 3. Prior Work & Gaps

### 3.1 Directly Related Work

| Paper | Venue | What They Did | What's Missing |
|---|---|---|---|
| **AsyncTaichi** (Hu et al. 2020) | arXiv | SFG + megakernel fusion for Taichi; 1.87x CUDA | **Abandoned** (too complex for sparse SNode tracking); no cost-benefit model; no strategy selection |
| **PERKS** (Zhang et al. 2023) | ICS 2023 | Persistent kernel for single-kernel stencils; 2.12x | Single kernel only (no multi-phase fusion); no register pressure modeling; no data save |
| **PyGraph** (Ghosh et al. 2025) | arXiv | Auto CUDA Graph for PyTorch; cost-benefit for Graph-vs-stream | ML only; no persistent kernel option; no simulation workloads |
| **KLAP** (El Hajj et al. 2016) | MICRO 2016 | Kernel launch aggregation compiler; 6.58x | Dynamic parallelism only; not time-stepping loops |
| **Mirage** (Cheng et al. 2025) | OSDI 2025 | Persistent megakernel for ML inference | ML inference; no simulation; no cost model |
| **FLUDA** (NASA 2023) | AIAA SciTech | Fused flux+divergence+source in CFD; 4x | Single application; manual fusion; no general model |
| **HFuse** (2022) | CGO 2022 | Horizontal fusion; register pressure analysis | ML ops (DAG); not time-stepping loop |
| **Kernel Batching** (2025) | arXiv | Optimal CUDA Graph batch size model | Graph only; no persistent kernel alternative |
| **FreeStencil** (2024) | ICPP 2024 | JIT fusion for PDE stencils; 3.29x | Structured stencils only; no unstructured/particle |
| **Leonid** (2025) | ICS 2025 | Auto fusion in portable framework; 1.26x GPU | Kokkos-level; no persistent kernel; no DSL overhead focus |

### 3.2 Our Position (Gaps We Fill)

| Gap | Who left it | We fill it with |
|---|---|---|
| Multi-phase fusion 的 cost-benefit model | 所有前人都没有 | **解析模型 (C1)** |
| Graph vs Persistent vs Async 的自动选择 | PyGraph 只做 Graph vs Stream; 没人加 Persistent | **模型驱动的策略选择 (C2)** |
| Persistent kernel + DMA overlap 做零开销 save | PERKS/AsyncTaichi 都不处理 data save | **Compute-Communication overlap (C3)** |
| 60+ simulation kernel 的 overhead characterization + 代际趋势 | 零星测量，无系统性 | **全面 characterization (C4)** |

---

## 4. Research Contributions

### C1: Fusion Cost-Benefit Model (核心贡献)

给定一个 time-stepping loop with K kernels, 预测三种策略的性能:

**输入参数** (可从 profiling/编译信息获取):
```
Per-kernel:   R_i (registers), G_i (grid blocks), T_i (exec time μs)
Hardware:     S (SM count), RF (regs/SM), OH (launch overhead μs)
Loop:         N (steps), has_dynamic_flow (bool), save_every (int)
```

**模型公式**:

**(a) Separate launch (baseline):**
```
T_separate = N × (Σ T_i + K × OH)
```

**(b) CUDA Graph:**
```
if has_dynamic_flow: 不可用
T_graph = N × Σ T_i + T_capture    // launch overhead ≈ 0
// 但每 save_every 步需要 break: T_graph += (N/save_every) × T_break
```

**(c) Persistent fusion:**
```
R_fused = max(R_1, ..., R_K)
OCC_fused = floor(RF / R_fused) / max_warps_per_SM    // 查 occupancy 表
G_limit = OCC_fused × S × blocks_per_SM
G_fused = max(G_1, ..., G_K)

if G_fused > G_limit: 不可用

// Occupancy 损失修正: 每个 phase 因 occupancy 下降而变慢
T_phase_i = T_i × (OCC_original_i / OCC_fused)

// Grid size 浪费: 小 phase 的线程在大 grid 中空转
waste_i = (G_fused - G_i) / G_fused

T_persistent = N × (Σ T_phase_i × (1 + waste_i) + K × T_grid_sync)
// T_grid_sync ≈ 2-3 μs (已测)
```

**(d) Strategy selection:**
```
best = argmin(T_graph, T_persistent, T_separate)
with constraints:
  - Graph 不可用 if has_dynamic_flow
  - Persistent 不可用 if G_fused > G_limit
  - 如有 periodic save: Persistent + async DMA; Graph 分段
```

**验证**: 用 60+ kernel benchmark 的实测数据拟合模型参数, 验证预测 vs 实测的误差。

**学术价值**: 这是第一个 fusion cost-benefit model for GPU simulation kernels。前人要么不做建模 (PERKS, AsyncTaichi)，要么只建模 Graph batch size (Kernel Batching 2025)，没有人建模 persistent fusion 的 occupancy/register tradeoff。

### C2: Model-Driven Strategy Selection (自动选择)

基于 C1 的模型，自动选择最优执行策略:

```
Compiler:
  1. Profile 每个 kernel → 获取 R_i, G_i, T_i
  2. 代入模型 → 计算三种策略的预测时间
  3. 选最优 → 生成对应代码

这是第一个做 Graph vs Persistent vs Async 统一选择的工作。
PyGraph (2025) 只做 Graph vs Stream，不考虑 Persistent。
```

### C3: Compute-Communication Overlap (新技术)

**Persistent kernel + DMA Copy Engine 并行做零开销数据保存。**

已验证: 4.99 vs 4.74 μs/step (仅 5.3% overhead)。

前人空白:
- PERKS: 不处理 data save
- AsyncTaichi: 不处理 data save
- CUDA Graph: 需要 break out to save

这是 persistent kernel 相比 CUDA Graph 的一个**独有优势**: Graph 保存数据必须打断回放, persistent kernel 可以在计算的同时通过 DMA 引擎传输。

### C4: Comprehensive Characterization (数据贡献)

60+ simulation kernel × 多框架 (Taichi, Warp, CUDA, Kokkos) × 多 GPU (3060, B200):
- 6-layer overhead decomposition
- Overhead fraction vs problem size curves
- **代际趋势**: overhead wall 每代恶化 (首次系统论证)
- Kernel classification: overhead-dominated (54/60) vs compute-dominated (2/60)
- AsyncTaichi 废弃原因分析 (SFG complexity breakdown)

---

## 5. Evaluation Plan

### 5.1 Hardware

| GPU | SMs | Role |
|---|---|---|
| RTX 3060 | 28 | Primary (represents research users) |
| B200 | 148 | Datacenter (current access) |

### 5.2 Benchmark Suite (60+ kernels)

| Category | Examples | Count |
|---|---|---|
| Structured stencil | Heat, Wave, Jacobi, Burgers, Conv-Diff | 15+ |
| Phase field | Allen-Cahn, Cahn-Hilliard, Gray-Scott | 6+ |
| CFD | SWE, LBM, Stable Fluids, hydro-cal | 8+ |
| Particle | N-body, SPH, DEM | 6+ |
| FEM/Structure | Explicit FEM, Cloth, Mass-Spring | 6+ |
| EM / Other | FDTD Maxwell, Monte Carlo, PIC | 19+ |

### 5.3 Metrics

| Metric | What It Measures |
|---|---|
| **Model accuracy** | 预测 speedup vs 实测 speedup 的误差 (MAPE) |
| **Strategy selection accuracy** | 模型选的策略 vs oracle 最优的匹配率 |
| **GPU utilization** | compute_time / total_time (before/after) |
| **Speedup** | over Taichi/Warp default |

### 5.4 Baselines

| Baseline | Source |
|---|---|
| Taichi default (sync per step) | Our benchmark |
| Warp default | Our benchmark |
| CUDA Graph (manual) | Our implementation |
| PERKS-style persistent (single kernel) | Our implementation |
| C++ async loop (Kokkos) | Our benchmark |
| **Our model-driven selection** | Our framework |

### 5.5 Key Experiments

1. **Model validation**: 60+ kernels, 模型预测 vs 实测, 报告 MAPE
2. **Strategy selection**: 模型选的 vs oracle 最优, 报告 accuracy
3. **Ablation**: Graph-only vs Persistent-only vs Our-selection
4. **Sensitivity**: 模型对 register count 变化的敏感度
5. **Compute-comm overlap**: with/without async DMA, 零开销 save 验证
6. **Cross-generation**: 3060 vs B200, 模型在不同 GPU 上的泛化能力
7. **Real case**: hydro-cal end-to-end speedup

---

## 6. Implementation Plan

### Phase 1: Characterization + Model Data Collection (3-4 weeks)

- [ ] 在 3060 上重跑 60+ kernel benchmark
- [ ] 收集每个 kernel 的: register count, grid size, execution time, occupancy
- [ ] 测量 cooperative launch 的 grid_sync overhead (vs SM count)
- [ ] 测量 CUDA Graph capture/replay overhead
- [ ] 代际对比: 3060 vs B200

产出: 完整的测量数据集 + characterization 分析

### Phase 2: Performance Model (4-6 weeks)

- [ ] 建立解析模型 (公式 a/b/c/d)
- [ ] 用 30 个 kernel 的数据 **拟合** 模型参数 (grid_sync cost, occupancy penalty factor 等)
- [ ] 用剩余 30 个 kernel **验证** 模型预测精度
- [ ] 模型 sensitivity analysis

产出: 模型公式 + 拟合参数 + 验证结果

### Phase 3: Automatic Strategy Selection + Async Overlap (4-6 weeks)

- [ ] 实现 strategy selector (基于模型)
- [ ] 实现 persistent kernel fusion (手动 template, 10+ kernel types)
- [ ] 实现 CUDA Graph auto-capture
- [ ] 实现 compute-communication overlap (double buffer + DMA)
- [ ] End-to-end 评估

产出: 工具 + 完整评估数据

### Phase 4: Paper (4 weeks)

- [ ] 论文撰写
- [ ] 开源 benchmark suite + model + tool

---

## 7. Paper Outline

**Target venue:** ICS / PPoPP / CGO / SC

### Title Candidates

1. "To Fuse or Not to Fuse: Performance Modeling for Persistent Kernel Fusion in GPU Simulation DSLs"
2. "The Overhead Wall: When to Fuse, Graph, or Stream GPU Simulation Kernels"
3. "Model-Driven Kernel Execution Strategy Selection for Python GPU Simulation Frameworks"

### Structure

**1. Introduction** (1.5p)
- Overhead wall problem + 代际恶化
- 三种方案都有人做，但没人知道什么时候用哪个
- AsyncTaichi 的教训: 通用方法太复杂
- 我们: performance model + automatic selection

**2. Background** (1p)
- Taichi/Warp execution model
- CUDA Graph / Cooperative groups / Persistent kernels
- Register-occupancy tradeoff

**3. Characterization** (2p) — C4
- 60+ kernel overhead measurement
- 6-layer decomposition
- 代际趋势
- AsyncTaichi 为什么失败 (complexity of SFG for sparse data)

**4. The Fusion Cost-Benefit Model** (3p) — C1 (核心)
- 4.1 模型输入: kernel 参数 + hardware 参数
- 4.2 Separate launch model (baseline)
- 4.3 CUDA Graph model (优势 + 限制)
- 4.4 Persistent fusion model (occupancy penalty + grid constraint + waste)
- 4.5 Strategy selection algorithm
- 4.6 Compute-communication overlap model (persistent + DMA) — C3

**5. Evaluation** (3p) — C2
- 5.1 Model accuracy (predicted vs measured, MAPE)
- 5.2 Strategy selection accuracy (auto vs oracle)
- 5.3 End-to-end speedup (before vs after optimization)
- 5.4 Case study: hydro-cal
- 5.5 Cross-generation: 3060 vs B200

**6. Discussion** (0.5p)
- Limitations: model assumes uniform phase execution, doesn't handle irregular workloads
- What DSL frameworks should adopt

**7. Related Work** (1p)
- AsyncTaichi, PERKS, PyGraph, KLAP, HFuse, FLUDA, FreeStencil, Kernel Batching

---

## 8. Related Work Positioning

| Work | vs Ours |
|---|---|
| **AsyncTaichi** (2020) | 通用 SFG → 太复杂, 废弃。我们: 聚焦 time-stepping, 有 cost model, 更简单 |
| **PERKS** (ICS 2023) | 单 kernel persistent + register cache。我们: **multi-kernel fusion + cost model + async DMA** |
| **PyGraph** (2025) | Auto Graph for ML, cost-benefit for graph-vs-stream。我们: **加入 persistent 选项, 针对 simulation** |
| **HFuse** (CGO 2022) | Horizontal fusion + register analysis for ML。我们: **time-stepping fusion, 不同 tradeoff** |
| **FLUDA** (NASA 2023) | 手动 CFD fusion, register balance。我们: **通用模型, 自动选择, 60+ kernels 验证** |
| **Kernel Batching** (2025) | Graph batch size model。我们: **模型覆盖 Graph + Persistent + Async** |
| **FreeStencil** (ICPP 2024) | Stencil JIT fusion。我们: **通用 simulation (particle, FEM, CFD, stencil)** |

**核心差异化**: 前人要么只做一种策略 (PERKS=persistent, PyGraph=graph, FreeStencil=fusion), 要么不做建模 (AsyncTaichi, FLUDA)。**我们是第一个建立 cost-benefit model 来统一比较和自动选择的。**

---

## 9. Risk Assessment

| Risk | Mitigation |
|---|---|
| 模型精度不够 | 用 60+ kernels split train/test; 加经验修正项 |
| 3060 上 persistent 不可用 (grid 超限) | 正好展示模型的 fallback 能力 (自动选 Graph) |
| Reviewer: "model too simple" | 强调: 简单但有效, 优于 no-model baseline |
| Reviewer: "just use CUDA Graph" | (1) Graph 不支持 dynamic flow, (2) persistent + DMA overlap 比 Graph 更优于有 save 的场景 |
| Reviewer: "incremental over PERKS" | PERKS 是单 kernel; 我们是 multi-kernel fusion + model + strategy selection + async DMA |

---

## 10. Timeline

```
2026-04~05    Phase 1: Characterization + data collection (3060 + B200)
2026-05~07    Phase 2: Performance model (formulation + fitting + validation)
2026-07~09    Phase 3: Strategy selection tool + async overlap + evaluation
2026-09~10    Phase 4: Paper writing
2026-10       Submit (ICS / PPoPP / CGO 2027)
```

---

## Appendix: Validated Results (B200, 148 SMs)

### A. Strategy Comparison

| | Sync loop | Async | Graph | Persistent | Persistent+DMA |
|---|---|---|---|---|---|
| Heat2D 256² | 12.9 μs | 8.1 (1.6x) | **3.3 (3.9x)** | 4.8 (2.7x) | — |
| GrayScott 256² | 17.5 μs | 12.3 (1.4x) | **5.7 (3.1x)** | 6.5 (2.7x) | — |
| Hydro-cal 6675 | 15.2 μs | 8.2 (1.9x) | **5.3 (2.9x)** | 5.7 (2.7x) | — |
| Persistent+save | — | — | need break | — | **4.99 (≈0 save OH)** |

### B. Compute-Communication Overlap Validation

| Method | μs/step | Overhead of saving |
|---|---|---|
| Persistent (no save) | 4.74 | — |
| Persistent + async DMA copy | 4.99 | **5.3% (near zero)** |
| Sync loop + sync save | 13.63 | 189% overhead |

### C. Strategy Selection Matrix (模型需要预测这个)

| Scenario | Best Strategy | Why |
|---|---|---|
| Static loop, small grid, no save | Persistent or Graph | Both work, Graph slightly faster |
| Static loop, large grid, no save | **Graph** | Persistent grid limit exceeded |
| Static loop, periodic save | **Persistent + DMA** | Graph needs break; persistent saves for free |
| Dynamic loop (convergence), small grid | **Persistent** | Graph can't handle dynamic flow |
| Dynamic loop, large grid | **Async** | Only option |
| Per-step host dependency | **Async** | Must sync every step |
