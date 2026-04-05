# Research Plan v2: Computation Graph Optimization for GPU Simulation DSLs

**Version:** 3.0
**Date:** 2026-04-05
**Status:** Draft for advisor discussion
**Supersedes:** research-plan-v1.md (SPMD IR), research-plan-v2 (2.0/2.1)

---

## 1. One-Sentence Summary

> Python GPU simulation DSLs execute kernels one-at-a-time without global visibility; we extract the CPU-GPU kernel computation graph from simulation time-stepping loops, and apply graph-level optimizations (kernel fusion, compute-communication overlap, strategy selection) to recover up to 2.9x GPU utilization — automatically, with zero user code changes.

---

## 2. Problem Statement

### 2.1 The Execution Model Gap

Python GPU DSLs (Taichi, Warp) compile individual kernels efficiently, but execute them **one at a time, with no global view**:

```
Python (CPU):                      GPU:
  for step in range(N):
    flux_kernel(H, U, V)    →     launch → compute → exit
    sync()                   ←     idle...
    update_kernel(H, U, V)  →     launch → compute → exit
    sync()                   ←     idle...
    if step % 100 == 0:
      save(H.to_numpy())    →     memcpy D2H (blocking)
```

**Nobody sees the global picture.** Python sees one line of code at a time. GPU sees one kernel at a time. The CPU-GPU interaction pattern — which kernels run in what order, what data flows between them, where host synchronization actually is needed — is lost.

### 2.2 The Overhead Wall

This one-at-a-time execution pays fixed overhead per step:

```
Python → DSL runtime → CUDA driver → launch → compute → sync → Python
 ~3μs      ~2-3μs        ~3-5μs                          ~5-7μs
└──────────────── ~10-15 μs FIXED overhead ──────────────────────┘
```

For typical simulation meshes (1K–100K cells), GPU compute per step is only 2–10 μs. **Overhead exceeds compute.**

### 2.3 Generational Scaling: Getting Worse

GPU compute doubles every ~2 years. Launch overhead stays fixed.

| GPU Generation | Compute (μs) | Overhead (μs) | GPU Utilization |
|---|---|---|---|
| V100 (2017) | ~26 | ~10 | 73% |
| A100 (2020) | ~18 | ~10 | 64% |
| H100 (2022) | ~9 | ~10 | 47% |
| **B200 (2024)** | **~5** | **~10** | **35% (measured)** |
| Next (2026) | ~2.5 | ~10 | 21% (projected) |

90% of our 60+ benchmark kernel configurations are overhead-dominated at typical mesh sizes.

### 2.4 Root Cause: No Computation Graph

ML frameworks (PyTorch 2.0, XLA, TorchInductor) solved the analogous problem by **extracting a computation graph** from Python code, then optimizing it globally. Simulation DSLs haven't done this.

| | ML Frameworks | Simulation DSLs (current) |
|---|---|---|
| Graph extraction | torch.compile / JAX tracing | **None** |
| Graph structure | DAG of tensor ops | Time-stepping loop + host interaction |
| Optimization | Op fusion, memory planning | **None (per-kernel only)** |
| Result | Near-hardware utilization | 35% utilization on B200 |

---

## 3. Our Approach: Simulation Computation Graph

### 3.1 Core Idea

Extract the **CPU-GPU kernel computation graph** from Python simulation code, then optimize the graph to minimize overhead.

```
User code (unchanged):                Extracted computation graph:
                                      ┌──────────────────────────────────────┐
for step in range(9000):              │ LOOP (0..9000)                       │
    flux_kernel(H, U, V)             │   ┌─────┐ data  ┌────────┐          │
    update_kernel(H, U, V)    →      │   │flux │──dep──│update  │          │
    if step % 100 == 0:              │   │R:HUV│       │R:FLUX  │          │
        save(H)                       │   │W:FLX│       │W:HUV   │          │
                                      │   └─────┘       └───┬────┘          │
                                      │                      │               │
                                      │   ┌──────────────────▼────────┐     │
                                      │   │ SAVE H→host (every 100)   │     │
                                      │   └───────────────────────────┘     │
                                      └──────────────────────────────────────┘
Graph nodes: kernel launch, host op (save, sync, branch)
Graph edges: data dependency (R/W sets), control flow (loop, condition)
Annotations: grid size, register count, R/W sets per kernel
```

### 3.2 Graph-Level Optimizations

On this graph, the compiler applies a sequence of optimization passes:

```
Pass 1: SyncElimination                    — 删除不必要的 per-kernel sync
Pass 2: KernelFusion                       — 合并多 kernel 为 persistent kernel
Pass 3: ComputeCommunicationOverlap        — 异步传输与计算并行 (double buffer + DMA)
Pass 4: StrategySelection                  — 选择最优执行策略 (Graph/Persistent/Async)
Pass 5: RegisterTuning                     — fused kernel 寄存器调优
```

### 3.3 Optimization Pipeline (Full)

```
原始执行 (Taichi/Warp default):
  LOOP(N=9000) {
    launch(flux);   sync();           ← 多余的 sync
    launch(update); sync();           ← 多余的 sync
    IF(step%100==0): sync(); memcpy(H, D2H);  ← 阻塞式传输
  }
        │
        ▼  Pass 1: SyncElimination
  LOOP(N=9000) {
    launch(flux);
    launch(update);                   ← flux→update 在同一 stream 自动串行
    IF(step%100==0): sync(); memcpy(H, D2H);
  }                                   ✓ 1.9x (已验证)
        │
        ▼  Pass 2: KernelFusion
  persistent_launch(fused) {
    for step in 0..9000:
      flux_body();
      grid_sync();                    ← 替代 kernel exit + relaunch
      update_body();
      grid_sync();
  }
  + SAVE(H, every 100)               ✓ 2.7x (已验证)
        │
        ▼  Pass 3: ComputeCommunicationOverlap
  ┌─ Compute Engine (all SMs) ────────────────┐
  │ persistent_launch(fused) {                 │
  │   for step:                                │
  │     flux_body(); grid_sync();              │
  │     update_body(); grid_sync();            │
  │     if step%100==0:                        │
  │       copy u → save_buf[step%2]            │ double buffer
  │       __threadfence_system()               │
  │       *flag = step                         │ mapped memory signal
  │ }                                          │
  └────────────────────────────────────────────┘
                     ↕ mapped memory
  ┌─ Copy Engine (DMA, 独立硬件) ─────────────┐
  │ host_thread:                               │
  │   poll flag → cudaMemcpyAsync(D2H)         │ ← 与计算并行！
  └────────────────────────────────────────────┘
                                      ✓ 2.7x + 零开销数据保存 (已验证)
        │
        ▼  Pass 4: StrategySelection
  根据循环特征 + 硬件参数选择最优路径:
    static loop + grid fits    → Persistent (2.7x)
    static loop + grid too big → CUDA Graph auto-capture (2.9x)
    dynamic loop + grid fits   → Persistent (2.7x, 支持 break/convergence)
    dynamic loop + grid big    → Async sync-elimination (1.9x)
        │
        ▼  Pass 5: RegisterTuning
  分析 fused kernel 寄存器用量
  设置 -maxrregcount 优化 occupancy (额外 1.2-1.4x on compute-heavy kernels)
```

---

## 4. Research Contributions

### C1: Characterization — "The Overhead Wall" (测量)

首次系统度量 Python GPU DSL 的 launch overhead:
- 60+ kernel benchmark suite (Taichi, Warp, CUDA, Kokkos)
- 6-layer overhead decomposition
- 代际趋势分析: V100 → A100 → H100 → B200
- Classification: overhead-dominated (54/60) vs compute-dominated (2/60)

### C2: Simulation Computation Graph Extraction (计算图抽取)

从 Python simulation 代码中提取 CPU-GPU kernel 计算图:
- Nodes: kernel launches + host operations (save, sync, condition)
- Edges: data dependencies (R/W set analysis) + control flow
- Annotations: grid size, register count, iteration count
- 与 ML 框架的计算图对比: DAG vs time-stepping loop with host interactions

### C3: Graph-Level Optimization Passes (图优化)

五个编译器 pass，逐层优化:

**Pass 1: SyncElimination** — 分析 kernel 间数据依赖，删除不必要的 host synchronization。只在真正需要 host 数据的地方（save, branch on device data）保留 sync。Speedup: 1.9x.

**Pass 2: KernelFusion** — 将 time-stepping loop 中的多个 kernel 融合为单个 persistent kernel，用 cooperative grid sync 替代 kernel exit + relaunch。需要: grid size unification, bounds check insertion。Speedup: 2.7x.

**Pass 3: ComputeCommunicationOverlap** — 利用 GPU 的 Compute Engine 和 Copy Engine 是独立硬件的事实，将数据保存（D2H copy）与计算并行执行。自动插入: double buffer allocation, mapped memory flag, host-side async copy polling thread, `__threadfence_system`。验证: persistent kernel + async copy 几乎零额外开销 (4.99 vs 4.74 μs/step)。

**Pass 4: StrategySelection** — 根据计算图特征 + 硬件参数，选择最优执行策略:

| 计算图特征 | Grid Size | 策略 |
|---|---|---|
| 固定步数, 无 host 交互 | Any | CUDA Graph auto-capture (2.9x) |
| 固定步数, 有周期性 save | ≤ coop limit | Persistent + async copy (2.7x) |
| 固定步数, 有周期性 save | > coop limit | Graph 分段 + sync save |
| 有 convergence / adaptive dt | ≤ coop limit | Persistent (2.7x) |
| 有 convergence / adaptive dt | > coop limit | Async sync-elimination (1.9x) |
| 每步 host 数据依赖 | Any | Async sync-elimination (1.9x) |

**Pass 5: RegisterTuning** — 分析 fused kernel 的寄存器用量, 如果超过 occupancy sweet spot，设置 `-maxrregcount`。与 overhead 消除正交可叠加 (额外 1.2-1.4x on compute-heavy kernels)。

### C4: Evaluation (评估)

- GPU utilization before/after across 60+ kernels
- Strategy selection accuracy: auto vs oracle
- Cross-generation scaling: 3060 vs B200
- Real-world case: hydro-cal end-to-end
- Comparison with manual CUDA Graph and Kokkos

---

## 5. Computation Graph: IR Design

### 5.1 Graph Representation (MLIR Dialect)

```mlir
// High-level: captured from Python tracing
sim.timestep_loop iter(%step) = 0 to 9000 {
  sim.kernel @flux  reads(%H, %U, %V)  writes(%FLUX)  grid(105)
  sim.kernel @update reads(%FLUX)       writes(%H, %U, %V) grid(27)
  sim.save(%H) every(100)              // host data transfer
}

// After SyncElimination + KernelFusion + AsyncCopyInsertion:
sim.persistent_kernel @fused  grid(105) cooperative {
  ^body(%step: index):
    sim.phase @flux  reads(%H, %U, %V) writes(%FLUX) {
      // flux kernel body (inlined)
    }
    sim.grid_sync
    sim.phase @update reads(%FLUX) writes(%H, %U, %V) {
      // update kernel body (inlined)
    }
    sim.grid_sync
    sim.async_save(%H) every(100)
      double_buffer(%save0, %save1)
      flag(%mapped_flag)
}
sim.host_copy_loop poll(%mapped_flag) copy(%save0, %save1 -> %host_buf)

// After lowering to GPU dialect:
gpu.launch_cooperative @fused blocks(%grid) threads(%block) {
  scf.for %step = 0 to 9000 {
    // flux body ...
    gpu.barrier {scope = "grid"}
    // update body ...
    gpu.barrier {scope = "grid"}
    scf.if %step mod 100 == 0 {
      memref.copy %H -> %save_buf[%step mod 2]
      gpu.threadfence_system
      memref.store %step, %mapped_flag[]
    }
  }
}
// + host-side async copy code generation
```

### 5.2 Graph Extraction Approaches

| Approach | How | Pros | Cons |
|---|---|---|---|
| **Runtime Tracing** | 类似 `torch.compile`: 第一次执行时记录 kernel launch 序列 | 通用, 不改框架内部 | 需要 warmup run |
| **DSL Compiler Plugin** | 在 Taichi/Warp 编译栈内部抓取 | 精确的 R/W 信息 | 绑定特定框架 |
| **Python AST Analysis** | 静态分析 Python 源码的 for-loop + kernel call pattern | 最轻量 | 无法处理动态模式 |

**推荐: Runtime Tracing** — 最通用，与 `torch.compile` 同类技术，不绑定特定 DSL。

Tracing 流程:
```python
@sim_compile              # ← 我们的 decorator
def simulate(H, U, V):
    for step in range(9000):
        flux_kernel(H, U, V)
        update_kernel(H, U, V)
        if step % 100 == 0:
            save(H)

# 第一次调用:
#   1. Trace: 记录 kernel launch 序列 + R/W sets + control flow
#   2. Build graph: 构建计算图
#   3. Optimize: apply Pass 1-5
#   4. Codegen: 生成 persistent kernel + host orchestration
# 后续调用:
#   直接执行优化后的版本
```

---

## 6. Comparison with ML Framework Approaches

| | PyTorch 2.0 | XLA (JAX) | **Ours** |
|---|---|---|---|
| 抓取 | `torch.compile` dynamo trace | JAX tracing | **Runtime tracing of kernel launches** |
| 图结构 | DAG (tensor ops) | DAG (HLO ops) | **Loop + kernel + host ops** |
| 图节点 | matmul, relu, etc. | add, dot, etc. | **GPU kernel launch, sync, memcpy, branch** |
| 关键区别 | 无循环 | 无循环 | **Time-stepping loop with periodic host interaction** |
| 优化 | Op fusion, memory planning | Op fusion, layout opt | **Kernel fusion, compute-comm overlap, strategy selection** |
| Compute-Comm overlap | NCCL + compute overlap | XLA async collectives | **DMA Copy Engine + Compute Engine overlap** |
| 目标 | 最大化 tensor op 吞吐 | 同左 | **消除 CPU-GPU 交互 overhead** |

**关键差异**: ML 图是 DAG (无循环)，simulation 图是**带循环的、有周期性 host 交互的图**。这需要不同的优化策略 (persistent kernel, cooperative grid sync, double-buffer async save)，是 ML 框架的 fusion 方法不能直接搬过来的。

---

## 7. Implementation Plan

### Phase 1: Characterization (2-3 weeks)

完成测量部分的数据:
- [ ] 在 RTX 3060 上重跑 60+ kernel benchmark
- [ ] 代际对比: 3060 vs B200
- [ ] 完善 6-layer overhead decomposition (nsys 验证)
- [ ] 绘图脚本 + figure-ready 数据

### Phase 2: Graph Extraction + Manual Optimization (4-6 weeks)

手动验证计算图优化在 10+ 个代表性 kernel 上的效果:
- [ ] 定义计算图表示格式 (MLIR `sim` dialect 或 JSON intermediate)
- [ ] 手动构建 10+ 个 kernel 的计算图
- [ ] 手写优化后代码 (persistent + async copy) 验证正确性
  - Stencil: Heat2D ✅, Wave2D, Jacobi
  - Multi-field: Gray-Scott ✅, Allen-Cahn
  - CFD: SWE/hydro-cal ✅, LBM (stream+collide)
  - Particle: N-body
  - Iterative: Jacobi-convergence (dynamic loop, Graph 做不了)
- [ ] 验证 compute-communication overlap (async copy) ✅ 已验证
- [ ] 策略选择规则: 手动标注 oracle 最优 vs 规则选择

### Phase 3: Automatic Framework (6-8 weeks)

构建自动化工具:

```
模块 1: Graph Extractor (计算图抽取)
  - Runtime tracing: 拦截 kernel launch, 记录 R/W sets
  - Loop detection: 识别 time-stepping pattern
  - Save detection: 识别 host data transfer points
  输出: 计算图 (MLIR sim dialect 或 internal IR)

模块 2: Graph Optimizer (图优化器)
  - SyncElimination pass
  - KernelFusion pass (persistent kernel generation)
  - AsyncCopyInsertion pass (double buffer + DMA overlap)
  - StrategySelection pass (Graph/Persistent/Async)
  - RegisterTuning pass

模块 3: Code Generator (代码生成)
  - Persistent 路径: emit cooperative kernel + host polling code
  - Graph 路径: emit capture/replay boilerplate
  - Async 路径: emit sync-eliminated launch sequence
```

**实现路径** (选一):

**路径 A: MLIR Pass (推荐)**
- 定义 `sim` dialect: `sim.timestep_loop`, `sim.kernel`, `sim.save`, `sim.persistent_kernel`, `sim.grid_sync`, `sim.async_save`
- 优化 passes 实现为 MLIR passes
- Lowering 到 `gpu` dialect + `func` dialect (host code)
- 优点: 学术贡献清晰, 复用 MLIR 基础设施, 可独立于 Taichi/Warp
- 缺点: 需要前端桥接 (tracing → MLIR)

**路径 B: Python-level Compiler (Fallback)**
- Python 实现 graph extraction (runtime tracing)
- Python 实现 graph optimization (直接生成 CUDA 代码)
- 优点: 快速原型, 直接演示 end-to-end
- 缺点: 学术贡献较弱

**推荐**: 路径 A + 路径 B 的 tracing 前端。用 Python runtime tracing 抓图, 转成 MLIR, 在 MLIR 上做优化。

### Phase 4: Evaluation + Paper (4-6 weeks)

- [ ] 60+ kernels × {Taichi, Warp, CUDA, Kokkos, Our framework}
- [ ] GPU utilization before/after (primary metric)
- [ ] Strategy selection: auto vs oracle
- [ ] Cross-generation: 3060 vs B200
- [ ] Paper writing

---

## 8. Hardware Strategy

### RTX 3060 (Primary)

| Factor | B200 | RTX 3060 | Impact |
|---|---|---|---|
| SMs | 148 | 28 | Cooperative limit 更小 |
| Coop limit (est.) | ~444 blocks | ~56-84 blocks | hydro-cal 105 blocks 可能超限 |
| Compute | ~5 μs | ~20-30 μs (est.) | OH fraction 更低 (~30-40%) |
| 意义 | 数据中心 | **科研用户主力** | 两者都需要 |

3060 上 hydro-cal 超 cooperative limit → **自动 fallback to Graph + 分段 save**。这恰好展示了 StrategySelection pass 的价值。

叙事: "消费级 GPU 上浪费 30-40%, 数据中心 GPU 上浪费 65-85%, 且每代恶化。"

---

## 9. Paper Outline

**Target venue:** ASPLOS / CGO / PPoPP / SC

### Title Candidates

1. "Simulation Computation Graphs: Automatic Overhead Elimination for Python GPU DSLs"
2. "The Launch Overhead Wall: Computation Graph Optimization for GPU Simulation Frameworks"
3. "From Per-Kernel to Per-Graph: Closing the GPU Utilization Gap in Python Simulation DSLs"

### Structure

1. **Introduction** (1.5p)
   - Python DSLs 的承诺 vs 现实 (35% GPU utilization)
   - 根因: 无全局计算图, 逐条执行
   - ML frameworks 通过计算图解决了同类问题
   - 我们: 提取 simulation 计算图 + 图级优化

2. **Background & Motivation** (1.5p)
   - Taichi/Warp 执行模型
   - GPU 架构: Compute Engine vs Copy Engine (独立硬件)
   - CUDA Graph, cooperative groups, persistent kernels
   - 与 ML 计算图的对比 (DAG vs time-stepping loop)

3. **Characterization** (2.5p) — C1
   - 60+ kernel benchmark
   - 6-layer overhead decomposition
   - 代际趋势: overhead wall 在恶化
   - 90% kernel 在 overhead-dominated regime

4. **Simulation Computation Graph** (3p) — C2 + C3
   - 4.1 Graph extraction (runtime tracing)
   - 4.2 Graph representation (IR design)
   - 4.3 Pass 1: SyncElimination
   - 4.4 Pass 2: KernelFusion (persistent kernel)
   - 4.5 Pass 3: ComputeCommunicationOverlap (Compute Engine ∥ Copy Engine)
   - 4.6 Pass 4: StrategySelection (Graph/Persistent/Async)

5. **Evaluation** (3p) — C4
   - GPU utilization: 35% → ~95%
   - 60+ kernels × 5 frameworks
   - Strategy selection accuracy
   - Real case: hydro-cal (2.9x speedup, 零开销 save)
   - Cross-generation: 3060 vs B200

6. **Discussion** (0.5p)
   - Limitations: cooperative grid size, register pressure of fused kernel
   - What DSL frameworks should adopt
   - Complementarity with single-kernel optimization

7. **Related Work** (1p)

---

## 10. Related Work Positioning

| Work | What | How We Differ |
|---|---|---|
| CUDA Graph | Runtime API, manual capture | 我们自动抓图 + 自动选择 Graph/Persistent |
| torch.compile / XLA | ML 计算图优化 | 我们处理 **time-stepping loop**（非 DAG）+ host interaction |
| Taichi Async Engine | 尝试消除 sync (已废弃) | 我们解释失败原因 + 提供 graph-level 解决方案 |
| Persistent threads [Gupta 2012] | GPU scheduling | 我们用于 overhead 消除, 且自动化 |
| Triton / Halide | Single-kernel optimization | 互补: 我们做 graph-level, 它们做 kernel-level |
| TorchInductor fusion | DAG-based op fusion | 不同图结构 (DAG vs loop), 不同优化目标 |

**Unique position:**
- 首个系统量化 simulation DSL 的 overhead wall (60+ kernels, 代际趋势)
- 首个将 "computation graph optimization" 概念应用于 simulation time-stepping loops
- 首个验证 Compute Engine + Copy Engine 并行在 persistent kernel 中实现零开销数据保存
- 统一框架覆盖 static/dynamic loop × 大/小 grid × 有/无 host save

---

## 11. Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| 3060 cooperative limit 不够 | High | 自动 fallback to Graph (展示 strategy selection) |
| Runtime tracing 漏捕复杂模式 | Medium | 先支持最常见 pattern (fixed-step loop), 渐进扩展 |
| Fused kernel register pressure | Medium | RegisterTuning pass + strategy 回退 |
| 审稿人: "just use CUDA Graph" | Medium | (1) 我们自动化 Graph, (2) Graph 不支持 dynamic flow + save overlap, (3) 计算图是更通用的抽象 |
| Taichi/Warp API 变动 | High | Runtime tracing 不依赖框架内部 API |
| 只有 3060 数据 | Medium | B200 数据已有; 争取借用 A100/H100 |

---

## 12. Timeline

```
2026-04       Phase 1: Characterization on 3060 + finalize benchmark
2026-05-06    Phase 2: Graph extraction + manual optimization on 10+ kernels
2026-06-08    Phase 3: MLIR sim dialect + optimization passes
2026-08-09    Phase 4: Full evaluation + paper
2026-09       Submit (ASPLOS / CGO / SC)
```

---

## Appendix: Validated Results (B200)

### A. Overhead Elimination Speedups

| Method | Heat2D 256² | GrayScott 256² | Hydro-cal 6675 |
|---|---|---|---|
| Taichi (default) | 14.9 μs | 15.0 μs | ~15 μs |
| Sync loop (CUDA baseline) | 12.9 μs | 17.5 μs | 15.2 μs |
| **Pass 1: SyncElimination** | **8.1 μs (1.6x)** | **12.3 μs (1.4x)** | **8.2 μs (1.9x)** |
| **Pass 2: KernelFusion** | **4.8 μs (2.7x)** | **6.5 μs (2.7x)** | **5.7 μs (2.7x)** |
| **Pass 4: CUDA Graph** | **3.3 μs (3.9x)** | **5.7 μs (3.1x)** | **5.3 μs (2.9x)** |

### B. Compute-Communication Overlap (Pass 3)

| Method | μs/step | Save every 100 steps |
|---|---|---|
| Persistent (no save) | 4.74 | N/A |
| **Persistent + async copy** | **4.99** | **20 saves, ~zero overhead** |
| Sync loop + sync save | 13.63 | blocking |

Async copy 仅增加 0.25 μs/step (5.3% overhead) — Compute Engine 和 Copy Engine 确实并行执行。

### C. Strategy Selection Matrix

| Graph Feature | Grid Size | Best Strategy | Speedup |
|---|---|---|---|
| Static, no save | Any | CUDA Graph | 2.9x |
| Static, periodic save | ≤ coop limit | Persistent + async copy | 2.7x |
| Static, periodic save | > coop limit | Graph + segmented save | 2.7x |
| Dynamic (convergence) | ≤ coop limit | Persistent | 2.7x |
| Dynamic (convergence) | > coop limit | Async sync-elim | 1.9x |
| Per-step host dep | Any | Async sync-elim | 1.9x |

### D. 60-Kernel Classification

- Overhead-dominated (>60%): 54/60 at typical mesh sizes
- Compute-dominated (<20%): 2/60
- Transitional: 4/60
