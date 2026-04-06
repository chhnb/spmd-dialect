# Literature Survey: GPU Simulation Time-Stepping Optimization

**Date:** 2026-04-06
**Purpose:** 支撑 research-plan-v6 的完整文献列表

---

## A. GPU Performance Modeling & Analysis

1. **Williams, Waterman, Patterson**, "Roofline: An Insightful Visual Performance Model for Multicore Architectures", *CACM 2009*
   - Compute vs memory bandwidth 二维性能分析。GPU 性能建模的基石。
   - 我们 L3+L5 层的理论基础；但不建模 launch overhead 和 SM utilization。

2. **Hong, Kim**, "An Analytical Model for a GPU Architecture with Memory-level and Thread-level Parallelism Awareness", *ISCA 2009*
   - 首个 GPU 解析模型，分解为 compute cycles + memory cycles (MWP/CWP)。
   - 我们 L3 (occupancy/latency hiding) 的理论基础。

3. **Volkov**, "Better Performance at Lower Occupancy", *GTC 2010*; PhD thesis "Understanding Latency Hiding on GPUs", *UC Berkeley 2016*
   - 证明低 occupancy + 高 ILP 可达到 peak 性能。基于 Little's Law 的 GPU 执行模型。
   - 直接解释我们 register sweep 的数据：限制寄存器提高 occupancy 不一定更快。

4. **Sim, Dasgupta, Kim, Vuduc**, "A Performance Analysis Framework for Identifying Potential Benefits in GPGPU Applications", *PPoPP 2012*
   - 系统分解 GPU kernel 性能为 ILP, MLP, synchronization 等因素。
   - "Potential benefit analysis" 方法论参考。

5. **Yasin**, "A Top-Down Method for Performance Analysis and Counters Architecture", *ISPASS 2014* (500+ citations)
   - CPU pipeline slot 四层分解 (Front-end / Back-end / Bad Speculation / Retiring) → 集成进 Intel VTune。
   - **我们的直接类比和灵感来源**: GPU simulation 五层分解。

6. **Konstantinidis, Cotronis**, "Flexible Performance Modeling of GPU Architectures", *JPDC 2017*
   - Microbenchmark-driven GPU 解析模型，分解为 throughput, bandwidth, ILP, latency hiding。
   - 跨 GPU 架构的性能预测方法参考。

7. **Yang, Kurth, Williams**, "An Extended Roofline Model with Communication-Awareness for Distributed-Memory HPC Systems", *ISC HPC 2018*
   - 扩展 roofline 到分布式 GPU 系统，分解为 compute + memory + network。
   - 模型扩展方法论参考。

8. **Arafa, Badawy, Chennupati, Santhi, Eidenbenz**, "PPT-GPU: Scalable GPU Performance Modeling", *IEEE CAL 2019; ISPASS 2021*
   - 无需 cycle-level 仿真的 GPU 解析模型，考虑 warp scheduling + memory hierarchy，15% 误差。
   - 证明解析模型也能足够准确（不需要 ML）。

9. **Yang, Wang, Williams**, "An Instruction Roofline Model for GPUs", *ISPASS 2020*
   - 用 instruction count 扩展 roofline，适用于混合精度和 integer-heavy workload。
   - L5 (compute efficiency) 的测量方法参考。

---

## B. Kernel Fusion & Launch Overhead Reduction

10. **Wahib, Maruyama**, "Scalable Kernel Fusion for Memory-Bound GPU Applications", *SC 2014*
    - Scalable search 方法找最优 fusion 组合，减少 off-chip traffic。1.35x on FD applications。
    - 早期 HPC fusion 工作；无 persistent/graph 选项。

11. **Aliaga, Perez, Quintana-Orti**, "Systematic Fusion of CUDA Kernels for Iterative Sparse Linear System Solvers", *Euro-Par 2015*
    - 系统化分析 CG/BiCG/BiCGStab 的 kernel fusion，用 dependency analysis 决定 fusion。
    - 最早的 iterative solver kernel fusion 方法论。

12. **El Hajj, Gomez-Luna, Li, Chang, Milojicic, Hwu**, "KLAP: Kernel Launch Aggregation and Promotion for Optimizing Dynamic Parallelism", *MICRO 2016*
    - Source-to-source compiler 聚合 dynamic parallelism 的 kernel launch，6.58x。
    - 编译器层面 launch 聚合参考；但针对 dynamic parallelism 而非 time-stepping。

13. **Hu, Xu, Kuang, Durand**, "AsyncTaichi: Whole-Program Optimizations for Megakernel Sparse Computation and Differentiable Programming", *arXiv 2020*
    - SFG 计算图 + megakernel fusion for Taichi, 3-4x fewer launches, 1.87x speedup → **废弃**。
    - 前车之鉴: 通用方法太复杂 (70% 复杂度来自 sparse SNode 追踪)；无 cost model。

14. **Li, Zheng, Pekhimenko, Long**, "HFuse: Automatic Horizontal Fusion for GPU Kernels", *CGO 2022*
    - Horizontal fusion (两个独立 kernel 在同一 SM 上并行) + register pressure 分析, 2.5-60.8%。
    - Register pressure 分析方法参考；但做 ML DAG 而非 time-stepping。

15. **Ghosh et al.**, "PyGraph: Robust Compiler Support for CUDA Graphs in PyTorch", *arXiv 2025*
    - Auto CUDA Graph for ML; 发现 **25% 的 CUDA Graph 反而降速 (最高 397% slowdown)**。
    - **直接验证 cost model 的必要性**；但只做 Graph vs Stream，不考虑 Persistent。

16. **EuroHPC Plasma-PEPSC team**, "Boosting Performance of Iterative Applications on GPUs: Kernel Batching with CUDA Graphs", *arXiv 2025*
    - CUDA Graph batch size 解析模型 for iterative simulation (HotSpot, FDTD)。发现 50-100 nodes 最优，至少 3 次 launch 才能 amortize 创建开销。
    - **直接可比**: 我们的 Graph model 是他们的超集 (加了 Persistent + Async)。

17. **(Electronics journal)**, "Analyzing the Impact of Kernel Fusion on GPU Tensor Operation Performance", *Electronics (MDPI) 2025*
    - 实验研究 fusion 何时有效/无效，在 3 个 GPU 上测 4 种 tensor op。发现 map-reduce + atomics 时 fusion 反而降速。
    - 经验验证 fusion 不总是有效 → 支持我们的 cost model 方向。

---

## C. Persistent Kernel & Megakernel

18. **Zhang, Wahib, Chen, Meng, Wang, Endo, Matsuoka**, "PERKS: A Locality-Optimized Execution Model for Iterative Memory-bound GPU Applications", *ICS 2023*
    - 单 kernel persistent + register/shared memory cache across timesteps, 2.12x stencil, 4.86x small SpMV/CG。
    - **直接 baseline**。我们加: multi-kernel fusion + strategy selection + DMA overlap + cost model。

19. **Matsumura, Zhang, Wahib, Chen, Meng, Wang, Endo, Matsuoka**, "EBISU: Revisiting Temporal Blocking Stencil Optimizations", *ICS 2023*
    - **故意用低 occupancy (12.5%)** 换取更深的 temporal blocking on large tiles, 2.53x。
    - 关键 insight: 低 occupancy 可能更好 → 支撑我们的 L3 层非单调分析。

20. **Jia et al.** (CMU), "Mirage: A Compiler and Runtime for Mega-Kernelizing Tensor Programs", *OSDI 2025*
    - 首个自动将 multi-GPU LLM inference 编译为单个 megakernel, 1.2-6.7x latency 减少。
    - Persistent megakernel 最大规模应用；但做 ML inference 无 cost model。

21. **Hazy Research** (Stanford), "Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B", *Tech report 2025*
    - 手动将整个 Llama-1B forward pass 合并为单个 megakernel, 78% memory BW utilization on H100。
    - 展示 kernel boundary 的 "memory pipeline bubbles" 是真实瓶颈 → 类比我们的 L1 temporal。

22. **Hwang et al.** (Microsoft Research), "ARK: GPU-driven Code Execution for Distributed Deep Learning", *NSDI 2023*
    - GPU 自主调度 compute + communication，无 CPU 干预的 persistent "loop kernel"。
    - 和我们 persistent kernel 思路一致；但做 ML distributed training。

---

## D. Simulation-Specific Optimization

23. **Korch, Werner**, "Exploiting Limited Access Distance for Kernel Fusion Across Stages of Explicit One-Step Methods on GPUs", *PPAM 2019; Concurrency & Computation 2021*
    - 利用 RHS 函数的 limited access distance 跨 Runge-Kutta stages 和 time steps 做 hexagonal/trapezoidal tiling fusion。
    - 直接相关: 跨 timestep fusion for ODE solvers；但需要 limited access distance 假设。

24. **FLUDA team** (NASA Langley), "A Multi-Architecture Approach for Implicit CFD on Unstructured Grids", *AIAA SciTech 2023*
    - GPU-portable FUN3D implicit CFD，手动 fused flux+divergence+source kernels, 4x。
    - 真实 CFD 上的 fusion 效果验证；但手动、单应用、无通用模型。

25. **Yamazaki**, "Accelerating an Overhead-Sensitive Atmospheric Model on GPUs Using Asynchronous Execution and Kernel Fusion", *SC'24 Workshop (ScalAH 2024)*
    - Async execution 减 37% + kernel fusion 再减 10% for 大气模型 NICAM on A100。
    - **直接验证 overhead wall thesis** 在 production code 上；但单应用无通用模型。

26. **Al-Awar et al.** (UT Austin), "PyFuser: Dynamically Fusing Python HPC Kernels", *ISSTA 2025*
    - **首个 Python HPC kernel 动态 fusion 框架** (via PyKokkos)，lazily trace + 自动 fusion, 3.8x avg。
    - **直接竞争对手**。但无 persistent/graph，无 hardware-aware strategy selection。

27. **Zhu et al.** (Peking University), "FreeStencil: A Fine-Grained Solver Compiler with Graph and Kernel Optimizations on Structured Meshes", *ICPP 2024*
    - Stencil solver compiler，12 kernel invocations for 2 MINRES iterations, DRAM 减 34%, 3.29x。
    - 只做结构化网格线性 solver；我们覆盖更广。

28. **Leonid team**, "Exploring Automated Kernel Fusion in Performance-Portable Frameworks", *ICS 2025*
    - Kokkos-based auto fusion, 1.58x max GPU on LBM/Cavity/MD。
    - 无 persistent；无 cost model；我们可以在他们的 benchmark 上 PK。

29. **(Libraries Openly Fused project)**, "The Fused Kernel Library (FKL)", *arXiv 2025*
    - C++ metaprogramming compile-time fusion, 2x-1000x+ for library operations。
    - 工程实现参考；但需 C++ template，不适用 Python DSL。

---

## E. Auto-Tuning Frameworks

30. **Ragan-Kelley, Barnes, Adams, Paris, Durand, Amarasinghe**, "Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines", *PLDI 2013*
    - Algorithm-schedule 分离的思想先驱。
    - 我们"分离 simulation loop 的算法和执行策略"的思想来源。

31. **Chen, Moreau, Jiang, Zheng, Yan, Cowan, Shen, Wang, Hu, Ceze, Guestrin, Krishnamurthy**, "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning", *OSDI 2018*
    - ML tensor op 的 schedule auto-tune, ~10⁹ 搜索空间 + ML cost model (XGBoost)。
    - Intra-kernel 部分类似；但 TVM 不管 inter-kernel / time-stepping。

32. **Ansel, Kamil, Veeramachaneni, Ragan-Kelley, Bosboom, O'Reilly, Amarasinghe**, "OpenTuner: An Extensible Framework for Program Autotuning", *PACT 2014*
    - 通用 auto-tuning 框架，支持多种搜索策略组合。
    - 搜索策略参考。

33. **van Werkhoven**, "Kernel Tuner: A search-optimizing GPU code auto-tuner", *Future Generation Computer Systems 2019*
    - Python GPU kernel auto-tuning for HPC，支持 random/SA/Bayesian/genetic search。
    - **HPC GPU auto-tuning 标杆**；但只做 intra-kernel。

34. **Rasch, Schulze, Steuwer, Gorlatch**, "ATF: A Generic Auto-Tuning Framework", *HPDC 2018; IEEE TPDS 2021*
    - 基于约束的搜索空间描述 + 高效剪枝。Multi-objective tuning。
    - 约束处理方法参考（类似我们的 cooperative limit 约束）。

35. **Matsumura, Zohouri, Wahib, Endo, Matsuoka**, "AN5D: Automated Stencil Framework for High-Degree Temporal Blocking on GPUs", *CGO 2020*
    - 自动化将 stencil C 代码变换为 temporal blocking CUDA, up to degree 10。用 performance model 指导参数选择。
    - Model-guided auto-tuning 方法论参考；只做 structured stencil。

36. **Petrovič, Střelák, Hozzová, Filipovič**, "KTT: A CUDA/OpenCL Auto-Tuning Framework", *SPE 2022*
    - C++ runtime auto-tuning，支持 **kernel composition** (多 kernel 组合调优)。
    - Kernel composition 概念最接近我们的 inter-kernel 调优。

---

## F. Memory Traffic & Temporal Blocking (Stencil-Specific)

37. **Zhao, Basu, Williams, Hall, Johansen**, "Exploiting Reuse and Vectorization in Blocked Stencil Computations on CPUs and GPUs (BrickLib)", *SC 2019*
    - Brick 数据布局，cache miss 减 19x, TLB miss 减 49x for 3D stencils。
    - L4 (data layout) 参考。

38. **Chen, Li et al.** (Microsoft Research), "ConvStencil: Transform Stencil Computation to Matrix Multiplication on Tensor Cores", *PPoPP 2024*
    - Stencil→GEMM on Tensor Cores + stencil2row layout + dual tessellation。
    - 正交方向: TC-based stencil 加速。

39. **Zhang, Li, Yuan, Cheng, Zhang, Cao, Yang**, "LoRAStencil: Low-Rank Adaptation of Stencil Computation on Tensor Cores", *SC 2024*
    - 分解 stencil weight matrix 为 rank-1 matrices, 2.16x over prior TC approaches。
    - Memory redundancy 消除方法参考。

40. **Han, Li et al.**, "FlashFFTStencil: Bridging Fast Fourier Transforms to Memory-Efficient Stencil Computations on Tensor Core Units", *PPoPP 2025*
    - FFT-based stencil on FP64 TC + kernel tailoring 减少 memory traffic + register reuse, 2.57x。
    - 激进 temporal fusion 方向。

---

## G. Compute-Communication Overlap

41. **Agrawal, Aga, Pati, Islam**, "ConCCL: Optimizing ML Concurrent Computation and Communication with GPU DMA Engines", *ISPASS 2025*
    - Naive compute-comm overlap 只有 21% ideal; 用 GPU DMA engines 达 72% (1.67x)。
    - **最接近我们的 DMA 工作**: 验证 DMA engine > SM-based overlap。

42. **Punniyamurthy, Beckmann, Hamidouche** (AMD), "GPU-initiated Fine-grained Overlap of Collective Communication with Computation", *SC 2024*
    - GPU workgroups 完成计算后立即通信到远程 GPU, 22% speedup for GEMV。
    - Fine-grained overlap 参考。

---

## Summary Statistics

| 类别 | 数量 | 代表作 |
|---|---|---|
| A. GPU 性能建模 | 9 | Yasin 2014, Volkov 2010, Roofline 2009 |
| B. Kernel Fusion & Launch Overhead | 8 | PERKS 2023, PyGraph 2025, AsyncTaichi 2020 |
| C. Persistent Kernel & Megakernel | 5 | PERKS 2023, Mirage 2025, ARK 2023 |
| D. Simulation-Specific | 7 | PyFuser 2025, FreeStencil 2024, Yamazaki 2024 |
| E. Auto-Tuning | 7 | TVM 2018, Kernel Tuner 2019, AN5D 2020 |
| F. Memory / Temporal Blocking | 4 | BrickLib 2019, FlashFFTStencil 2025 |
| G. Compute-Comm Overlap | 2 | ConCCL 2025 |
| **Total** | **42** | |

---

## 与我们工作的关系分类

### 直接竞争 (需要 PK)
- **PERKS** (ICS 2023) — 单 kernel persistent + register cache
- **PyFuser** (ISSTA 2025) — Python HPC dynamic fusion
- **Kernel Batching** (arXiv 2025) — Graph batch model for simulation
- **FreeStencil** (ICPP 2024) — stencil solver fusion
- **Leonid** (ICS 2025) — Kokkos auto fusion

### 直接支撑我们的论点
- **PyGraph** (2025) — 25% Graph 降速 → cost model 必要
- **Yamazaki** (SC'24) — production code 有 overhead wall
- **ConCCL** (ISPASS 2025) — DMA > SM-based overlap
- **Volkov** (2010/2016) — occupancy-performance 非单调
- **EBISU** (ICS 2023) — 低 occupancy 可能更好
- **AsyncTaichi** (2020) — 通用 fusion 太复杂 → 废弃

### 技术借鉴
- **PERKS** — register caching 实现
- **EBISU** — circular multi-queue for deep temporal blocking
- **Kernel Batching** — Graph cost model 公式
- **AN5D** — model-guided parameter selection
- **Kernel Tuner** — auto-tuning API 设计
- **ARK** — GPU-driven persistent execution 工程实现
