# Research Plan v2: The Launch Overhead Wall

**Version:** 2.0
**Date:** 2026-04-05
**Status:** Draft for advisor discussion
**Supersedes:** research-plan-v1.md (SPMD IR)

---

## 1. One-Sentence Summary

> Python GPU simulation DSLs (Taichi, Warp) waste 60–85% of GPU cycles on launch overhead for typical engineering meshes; we characterize this "Launch Overhead Wall" across 60+ kernel types, show it worsens with each GPU generation, and build a compiler transformation (persistent kernel fusion) that recovers 2.7x automatically.

---

## 2. Problem Statement

### 2.1 Background

Python GPU DSLs (Taichi, Warp, Numba-CUDA) have become the dominant way researchers write simulation kernels — they provide Python-level productivity with GPU-level performance. The implicit contract is: "write Python, get bare-metal speed."

**This contract is broken for the majority of real simulation workloads.**

### 2.2 The Overhead Wall

Every timestep in a Python DSL simulation follows this path:

```
Python loop → DSL runtime → CUDA driver → GPU launch → GPU compute → Sync → Python
  ~3 μs        ~2-3 μs       ~3-5 μs                    variable     ~5-7 μs
└──────────────── ~10-15 μs FIXED overhead ────────────────────────────────────┘
```

For typical engineering simulation meshes (1K–100K cells), GPU compute per step is only 2–10 μs. The 10-15 μs fixed overhead **exceeds the compute itself**.

### 2.3 Why This Matters Now (Generational Scaling)

GPU compute capability doubles every ~2 years. Launch overhead is fixed by software architecture.

| GPU Generation | Compute (μs) | Overhead (μs) | GPU Utilization |
|---|---|---|---|
| V100 (2017) | ~26 | ~10 | 73% |
| A100 (2020) | ~18 | ~10 | 64% |
| H100 (2022) | ~9 | ~10 | 47% |
| **B200 (2024)** | **~5** | **~10** | **35% (measured)** |
| Next (2026) | ~2.5 | ~10 | 21% (projected) |
| Next+ (2028) | ~1.3 | ~10 | 12% (projected) |

**每一代 GPU 升级，Python DSL 的 GPU 利用率反而更低。**
到 2028 年，同样的模拟代码在更强的 GPU 上利用率仅 12%。

### 2.4 Who Is Affected

**90% of simulation kernel configurations are overhead-dominated.** Our benchmark of 60+ kernel types × multiple sizes shows:

- 54/60 kernels at typical mesh sizes (N ≤ 512²): overhead > 60%
- Domains: CFD, FEM, particle methods, stencil solvers, phase field, EM, etc.
- Frameworks: Taichi (~15 μs floor), Warp (~15-18 μs floor)

This is not a niche problem — it affects the core use case of these frameworks.

---

## 3. Pivoting from SPMD IR (research-plan-v1)

### 3.1 What We Learned

The original plan (v1) targeted single-kernel IR optimizations:
- Gather optimization (shared memory promotion) → **Ineffective** on modern GPUs (60MB L2 cache)
- Register pressure tuning → **1.4x**, but a known technique
- Prefetch → Conflicts with register tuning budget, crossover only at >10 ops/gather

**The single-kernel optimization ceiling is ~1.4x.** Meanwhile, eliminating launch overhead gives **2.7–2.9x**. The leverage is in the cross-kernel / runtime layer, not the single-kernel IR.

### 3.2 What We Keep

- MLIR infrastructure skills → reuse for implementing the compiler transformation
- 60+ kernel benchmark suite → evaluation infrastructure
- Deep performance analysis methodology → guides the characterization

### 3.3 What Changes

| Aspect | v1 (SPMD IR) | v2 (Overhead Wall) |
|---|---|---|
| Focus | Single-kernel IR optimization | Cross-kernel overhead elimination |
| Target | Memory access patterns | Launch / sync overhead |
| Approach | Design new IR dialect | Compiler transformation + characterization |
| Metric | Kernel execution time | GPU utilization, sims/GPU-hour |
| Ceiling | ~1.4x (register tuning) | **2.7–2.9x** (persistent kernel) |

---

## 4. Research Contributions

### Contribution 1: Characterization (测量)

**首次系统度量 Python GPU DSL 的 launch overhead，覆盖 60+ 种模拟 kernel。**

Deliverables:
- 60+ kernel benchmark suite (Taichi, Warp, CUDA, Kokkos)
- 6-layer overhead decomposition: Python → Runtime → Driver → Launch → Compute → Sync
- Overhead fraction vs problem size curves for each kernel category
- 代际趋势分析 (V100 → A100 → H100 → B200)
- Classification: overhead-dominated vs compute-dominated vs transitional

### Contribution 2: Root Cause Analysis (诊断)

**精确诊断每一层开销的来源，解释为什么 Taichi/Warp 做不到 CUDA Graph 的性能。**

Deliverables:
- Per-layer timing breakdown (using CUPTI, nsys, custom instrumentation)
- Comparison: Taichi runtime vs Warp runtime vs raw CUDA
- Analysis of why Taichi deprecated its async engine
- Codegen quality gap: Taichi LLVM NVPTX vs Warp LLVM vs nvcc

### Contribution 3: Persistent Kernel Fusion (编译器变换)

**核心技术贡献：自动将 time-stepping loop + 多 kernel 融合为单个 persistent kernel。**

The transformation:
```
INPUT (what the user writes):
  for step in range(N):
      kernel_1(args_1)     # e.g., flux calculation
      kernel_2(args_2)     # e.g., cell update

OUTPUT (what the compiler generates):
  @cooperative_launch(grid=max(grid_1, grid_2))
  def fused_persistent(args_1, args_2, steps=N):
      for step in range(steps):
          # Phase 1: kernel_1 logic
          if tid < problem_size_1:
              kernel_1_body(args_1)
          grid_sync()
          # Phase 2: kernel_2 logic
          if tid < problem_size_2:
              kernel_2_body(args_2)
          grid_sync()
```

Key challenges:
1. **Grid size unification**: different kernels may need different grid sizes → use max, guard with bounds check
2. **Applicability analysis**: detect which loops are fusible (no host-side data dependency between steps)
3. **Cooperative launch limit**: grid blocks ≤ SM_count × max_blocks_per_SM → may need tiling
4. **Escape hatches**: periodic output (every N steps), convergence checks → insert host sync points
5. **Register pressure**: fused kernel has union of all kernels' register usage → may need tuning

### Contribution 4: Evaluation Framework

- Before/after GPU utilization on 60+ kernels
- Comparison with CUDA Graph (our approach works with dynamic control flow; Graph doesn't)
- End-to-end simulation speedup on real-world cases (hydro-cal, LBM, wave equation)
- Scalability across GPU generations (3060, A100, B200)

---

## 5. Evaluation Plan

### 5.1 Hardware

| GPU | Architecture | SMs | Role |
|---|---|---|---|
| **RTX 3060** | Ampere (sm_86) | 28 | Primary development & evaluation |
| B200 | Blackwell (sm_90) | 148 | Datacenter validation (current access) |
| A100/H100 | Ampere/Hopper | 108/132 | Cross-generation comparison (if available) |

**RTX 3060 considerations:**
- 28 SMs → cooperative launch limit ~28 × occupancy blocks
- hydro-cal (105 blocks) 可能超出 3060 的 cooperative launch limit → 需要 tiling
- 但 3060 代表了大量科研用户的实际硬件
- Overhead fraction 在 3060 上可能较低（compute更慢，overhead占比更小），需要实测

### 5.2 Benchmark Suite (60+ kernels, 已基本完成)

| Category | Kernel Types | Count |
|---|---|---|
| Structured stencil | Heat, Wave, Jacobi, Advection, Burgers, Conv-Diff | 15+ |
| Phase field | Allen-Cahn, Cahn-Hilliard, Gray-Scott | 6+ |
| CFD | SWE, LBM, Stable Fluids, Euler | 8+ |
| Particle | N-body, SPH, DEM | 6+ |
| FEM | Explicit FEM, Mass-Spring, Cloth | 6+ |
| Electromagnetics | FDTD Maxwell, Helmholtz | 4+ |
| Other | Reaction-Diffusion, Monte Carlo, PIC | 15+ |

### 5.3 Metrics

Primary:
- **GPU utilization** = compute_time / total_time (not μs/step)
- **Speedup** over baseline Python DSL
- **Sims/GPU-hour** — throughput on real workloads

Secondary:
- Register usage of fused vs separate kernels
- Occupancy impact of fusion
- Compilation time overhead

### 5.4 Baselines

| Baseline | What It Measures |
|---|---|
| Taichi (default) | Current Python DSL performance |
| Warp (default) | Alternative DSL comparison |
| CUDA Graph | Best-case runtime solution (manual) |
| C++ async loop (Kokkos) | Best-case without Python overhead |
| Raw CUDA (sync per step) | Overhead without Python layer |
| **Our persistent kernel** | **Compiler-automatic solution** |

---

## 6. Implementation Plan

### Phase 1: 完善 Characterization（2–3 周）

**目标**: 完成测量论文的数据部分

- [ ] 在 RTX 3060 上重跑全部 60+ kernel benchmark
- [ ] 代际对比：3060 vs B200（如果有 A100/H100 更好）
- [ ] 完善 overhead 6层分解（用 nsys timeline 验证）
- [ ] 整理数据为论文 figure-ready 格式
- [ ] 补充 Warp 全套 overhead 数据

产出：`benchmark/CHARACTERIZATION_RESULTS.md` + 绘图脚本

### Phase 2: Persistent Kernel 变换原型（4–6 周）

**目标**: 手动验证变换在 10+ 个代表性 kernel 上的正确性和性能

- [ ] 手写 persistent kernel 版本覆盖 5 大类 kernel
  - Stencil (Heat2D) ✅ 已验证
  - Multi-field (Gray-Scott) ✅ 已验证
  - Unstructured mesh (hydro-cal) ✅ 已验证
  - Particle (N-body)
  - Multi-kernel step (LBM: stream + collide)
- [ ] 处理边界情况
  - 周期性输出（每 N 步 dump 数据）
  - 条件退出（convergence check）
  - Grid size 超出 cooperative limit → tiling / fallback to Graph
- [ ] 正确性验证：fused vs original bit-exact 对比
- [ ] 性能对比表：fused vs Taichi vs Graph vs Kokkos

产出：`benchmark/persistent_kernels/` 目录 + 10+ 手写验证用例

### Phase 3: 自动变换工具（6–8 周）

**目标**: 构建能自动做 persistent kernel fusion 的工具

两种路径（选其一或并行）：

**路径 A: Taichi Compiler Plugin**
- 在 Taichi IR (CHI IR) 层面识别 time-stepping pattern
- 插入 grid sync，融合多 kernel 为单 kernel
- 优点：直接集成到 Taichi 编译流程
- 缺点：Taichi 内部 API 不稳定

**路径 B: MLIR Pass (复用 spmd-dialect 基础设施)**
- 定义一个 `sim.timestep_loop` op
- 编写 fusion pass：`sim.timestep_loop { kernel1; kernel2 }` → `sim.persistent_kernel`
- lowering 到 `gpu.launch_cooperative`
- 优点：与现有 MLIR 工作衔接，可独立于 Taichi/Warp
- 缺点：需要前端桥接

**路径 C: Source-to-Source (最轻量)**
- Python AST 分析：识别 `for ... in range(): kernel()` pattern
- 生成等价 CUDA persistent kernel 代码
- 优点：快速原型，不依赖框架内部
- 缺点：不能直接集成

**推荐**: 路径 B（MLIR Pass），因为复用已有基础设施，且学术贡献更清晰。

### Phase 4: 评估 & 论文（4–6 周）

- [ ] 在 3060 + B200 上完整评估
- [ ] 与 CUDA Graph 对比（我们支持 dynamic control flow，Graph 不支持）
- [ ] 论文撰写
- [ ] 开源 benchmark suite + transformation tool

---

## 7. RTX 3060 Strategy

3060 与 B200 的关键差异及应对：

| 因素 | B200 | RTX 3060 | 影响 |
|---|---|---|---|
| SM 数量 | 148 | 28 | Cooperative launch limit 更小 |
| Coop limit (est.) | ~444 blocks | ~56-84 blocks | hydro-cal 105 blocks 可能超限 |
| Compute speed | ~5 μs (heat 256²) | ~20-30 μs (估计) | OH fraction 更低 |
| Memory BW | 8 TB/s | 360 GB/s | Memory-bound kernel 更慢 |
| 代表性 | 数据中心 | 科研用户主力 | 两者都需要展示 |

**3060 上的预期结果：**
- Overhead fraction: ~30-40%（不如 B200 的 65%，但仍然显著）
- Persistent kernel speedup: ~1.5-2x（不如 B200 的 2.7x）
- CUDA Graph speedup: ~1.5-2x
- **叙事角度**: "即使在消费级 GPU 上也有 30-40% 的浪费；在数据中心 GPU 上则是 65-85%"

**如果 105 blocks 超出 3060 cooperative limit：**
- 方案 1: Tiling — 将大 grid 切成多轮，每轮在 cooperative limit 内
- 方案 2: 用更小的 mesh 演示（2000 cells → 32 blocks）
- 方案 3: Fallback to CUDA Graph（自动检测，超限时回退）

---

## 8. 论文框架 (Outline)

**Target venue:** ASPLOS / CGO / PPoPP / SC (systems + compiler)

### Title (候选)

1. "The Launch Overhead Wall: Why Faster GPUs Make Python Simulation DSLs Relatively Slower"
2. "Closing the DSL-Metal Gap: Automatic Persistent Kernel Fusion for Python GPU Simulations"
3. "When Launch Overhead Exceeds Compute: Characterization and Compiler Solutions for GPU Simulation DSLs"

### Structure

1. **Introduction** (1.5 pages)
   - Python DSLs 是模拟领域的主流
   - 它们隐含的承诺: "write Python, get GPU speed"
   - 这个承诺在大多数真实工作负载上被打破了
   - 代际趋势: 问题在恶化

2. **Background** (1 page)
   - Taichi/Warp 编译 & 执行模型
   - CUDA kernel launch anatomy
   - Cooperative groups / persistent kernels

3. **Characterization** (3 pages) — Contribution 1 & 2
   - 60+ kernel benchmark suite
   - 6-layer overhead decomposition
   - Overhead fraction vs problem size
   - 代际趋势数据 (3060 vs B200)
   - Classification: overhead-dominated / compute-dominated

4. **Persistent Kernel Fusion** (3 pages) — Contribution 3
   - Transformation algorithm
   - Applicability analysis (what patterns are fusible)
   - Handling edge cases (periodic output, convergence, grid limit)
   - Implementation (MLIR pass)

5. **Evaluation** (3 pages) — Contribution 4
   - Setup: 60+ kernels × {Taichi, Warp, CUDA, Kokkos, our tool}
   - GPU utilization before / after
   - vs CUDA Graph: we handle dynamic control flow
   - Real-world case study: hydro-cal
   - Cross-generation: 3060 vs B200

6. **Discussion** (1 page)
   - Framework implications (what Taichi/Warp should change)
   - Limitations (grid size limit, register pressure)
   - Future: combined with single-kernel optimizations

7. **Related Work** (1 page)
   - CUDA Graph, Triton, Halide, persistent threads literature
   - Taichi async engine (deprecated)
   - GPU kernel fusion in ML compilers (XLA, TorchInductor)

---

## 9. Related Work Positioning

| Work | What They Do | How We Differ |
|---|---|---|
| CUDA Graph | Runtime API for launch batching | We do this as compiler transformation, no API change |
| Taichi Async Engine | Tried to eliminate sync (deprecated) | We explain why it failed + provide working alternative |
| Triton | Single-kernel optimization | We target cross-kernel overhead, complementary |
| XLA/TorchInductor fusion | Fuse ML ops | We target simulation kernels (time-stepping, not DAG) |
| Persistent threads [Gupta 2012] | GPU scheduling technique | We apply it to DSL overhead elimination, with compiler automation |
| Halide | Schedule-based optimization | Different domain (image processing), no launch overhead focus |

**Our unique position:** 我们是第一个 (a) 系统量化 Python GPU DSL 的 launch overhead，(b) 证明它在代际趋势上恶化，(c) 提出编译器自动的 persistent kernel fusion 作为解决方案。

---

## 10. Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| 3060 cooperative limit 不够 | High | Tiling + fallback to Graph |
| Persistent kernel 性能不如 Graph | Medium | 两者互补：persistent 支持 dynamic flow |
| Fused kernel register pressure 爆炸 | Medium | Register tuning pass (已验证 1.4x) |
| Taichi/Warp 内部 API 变动 | High | 用 MLIR pass 独立实现，不绑定框架 |
| 只有 3060 数据不够有说服力 | Medium | B200 数据已有；争取借用 A100/H100 |
| 审稿人认为 "just use CUDA Graph" | High | 强调: (1) Graph 不支持 dynamic flow, (2) 我们是 compiler-automatic |

---

## 11. Timeline (Tentative)

```
2026-04     Phase 1: Characterization on 3060 + finalize benchmark suite
2026-05     Phase 2: Manual persistent kernel validation (10+ kernels)
2026-06-07  Phase 3: MLIR pass implementation
2026-08     Phase 4: Full evaluation + paper writing
2026-09     Submit to ASPLOS / CGO / SC
```

---

## Appendix: Measured Results (B200)

### A. Overhead Solutions Comparison

| Method | Heat2D 256² | GrayScott 256² | Hydro-cal 6675 | Speedup Range |
|---|---|---|---|---|
| Taichi (default) | 14.9 μs | 15.0 μs | ~15 μs (est.) | baseline |
| CUDA Sync loop | 12.9 μs | 17.5 μs | 15.2 μs | 1.0x |
| C++ Async loop | 8.1 μs | 12.3 μs | 8.2 μs | 1.4–1.9x |
| CUDA Graph | 3.3 μs | 5.7 μs | 5.3 μs | 2.7–3.9x |
| **Persistent kernel** | **4.8 μs** | **6.5 μs** | **5.7 μs** | **2.4–2.7x** |

### B. 60-Kernel Classification

- Overhead-dominated (>60%): 54/60 at typical mesh sizes
- Compute-dominated (<20%): 2/60
- Transitional: 4/60
