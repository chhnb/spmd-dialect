Read and execute below with ultrathink

## Goal Tracker Setup (REQUIRED FIRST STEP)

Before starting implementation, you MUST initialize the Goal Tracker:

1. Read @/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/.humanize/rlcr/2026-03-27_08-24-48/goal-tracker.md
2. If the "Ultimate Goal" section says "[To be extracted...]", extract a clear goal statement from the plan
3. If the "Acceptance Criteria" section says "[To be defined...]", define 3-7 specific, testable criteria
4. Populate the "Active Tasks" table with tasks from the plan, mapping each to an AC
5. Write the updated goal-tracker.md

**IMPORTANT**: The IMMUTABLE SECTION can only be modified in Round 0. After this round, it becomes read-only.

---

## Implementation Plan

For all tasks that need to be completed, please use the Task system (TaskCreate, TaskUpdate, TaskList) to track each item in order of importance.
You are strictly prohibited from only addressing the most important issues - you MUST create Tasks for ALL discovered issues and attempt to resolve each one.

# SPMD Dialect MVP Implementation Plan

## Goal Description

Implement the `spmd` MLIR dialect as specified in `design-v1.md`: a structured, backend-agnostic intermediate representation for regular SPMD kernels. The MVP delivers a compilable and testable dialect with 5 ops, 5 attr classes, a build system with `spmd-opt` driver, a CPU execution path via SCF/OpenMP, and a group-memory promotion demonstration on a 2D stencil kernel.

The work proceeds through four sequential milestones:
- **Build system + tool driver** (prerequisite for all tests)
- **Dialect IR** (5 ops, 5 attrs, verifiers, legality pass, lit tests)
- **CPU closed-loop pipeline** (normalize → materialize → lower → OpenMP → LLVM)
- **Group memory promotion demo** (read-only stencil pattern, S0 → S2 → CPU)

GPU backend (Milestone 5) is defined but outside MVP scope.

---

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- **AC-1**: The project builds cleanly from source using CMake with MLIR as external dependency, and produces a working `spmd-opt` binary.
  - Positive Tests (expected to PASS):
    - `cmake --build build` completes with exit code 0 after configuring with `-DMLIR_DIR=<llvm-build>/lib/cmake/mlir`
    - `build/bin/spmd-opt --help` exits 0 and lists registered passes
    - `build/bin/spmd-opt /dev/null` exits 0 (empty input accepted)
  - Negative Tests (expected to FAIL):
    - Building without MLIR_DIR set produces a descriptive CMake error, not a silent build failure

- **AC-2**: All 5 custom attributes (`level`, `scope`, `reduction_kind`, `addr_space`, `memory_policy`) are defined in ODS, registered in the dialect, and round-trip through `spmd-opt` without error.
  - Positive Tests (expected to PASS):
    - `spmd-opt` parses `#spmd.level<group>`, `#spmd.scope<group>`, `#spmd.reduction_kind<add>`, `#spmd.addr_space<global>`, `#spmd.memory_policy<prefer_group>` in attribute position
    - `spmd-opt --mlir-print-op-generic` round-trips all attr variants without loss
  - Negative Tests (expected to FAIL):
    - `#spmd.level<invalid_level>` produces a parse error
    - `#spmd.reduction_kind<subtract>` produces a parse error

- **AC-3**: `spmd.kernel` is defined as a `UnitAttr` in ODS and accepted on `func.func` ops; `VerifySPMDKernelSubset` pass rejects S0/S1 kernels containing disallowed ops.
  - Positive Tests (expected to PASS):
    - A `func.func` with `attributes {spmd.kernel}` passes `--verify-spmd-kernel-subset`
    - A kernel containing only `spmd.*`, `arith.*`, `memref.load/store`, `affine.apply`, `func.return` passes the legality check
  - Negative Tests (expected to FAIL):
    - A kernel containing `gpu.thread_id` fails `--verify-spmd-kernel-subset` with a diagnostic
    - A kernel containing `spmd.barrier` in S0 context fails with a diagnostic

- **AC-4**: `spmd.forall` parses, prints, and verifies correctly using ODS generic assembly format (no custom format in MVP).
  - AC-4.1: Structural verifier catches malformed ops.
    - Positive Tests (expected to PASS):
      - A 2D `spmd.forall` with matching lbs/ubs/steps and correct block args parses and verifies
      - A `spmd.forall` with `spmd.tile_sizes = array<i64: 32, 8>` and rank-2 domain passes
      - A `spmd.forall` with `spmd.order = array<i64: 1, 0>` (valid permutation) passes
    - Negative Tests (expected to FAIL):
      - `spmd.forall` where lbs/ubs/steps have different lengths emits `lowerBounds, upperBounds, and steps must have equal length`
      - `spmd.forall` with `spmd.tile_sizes = array<i64: 32, 0>` emits `spmd.tile_sizes values must be positive`
      - `spmd.forall` with `spmd.order = array<i64: 0, 0>` emits permutation error

- **AC-5**: `spmd.if`, `spmd.reduce`, `spmd.barrier`, `spmd.yield` parse, print, and verify correctly.
  - AC-5.1: `spmd.if`
    - Positive Tests (expected to PASS):
      - `spmd.if %cond { spmd.yield }` (no result) parses and verifies
      - `%r = spmd.if %cond -> (f32) { spmd.yield %x : f32 } else { spmd.yield %y : f32 }` verifies
    - Negative Tests (expected to FAIL):
      - `spmd.if %idx { spmd.yield }` where `%idx : index` emits `condition must be i1`
      - `%r = spmd.if %cond -> (f32) { spmd.yield %x : f32 }` (no else) emits `else region required when op has results`
  - AC-5.2: `spmd.reduce`
    - Positive Tests (expected to PASS):
      - A 1D reduce with `spmd.kind = #spmd.reduction_kind<add>` and matching init/result type verifies
    - Negative Tests (expected to FAIL):
      - `spmd.reduce` without `spmd.kind` emits `spmd.kind attribute is required`
      - `spmd.reduce` where init type is `i32` but yield produces `f32` emits type mismatch error
  - AC-5.3: `spmd.barrier`
    - Positive Tests (expected to PASS):
      - `spmd.barrier {spmd.scope = #spmd.scope<group>}` nested inside a `spmd.forall` with `spmd.mapping = #spmd.level<group>` verifies
    - Negative Tests (expected to FAIL):
      - `spmd.barrier {spmd.scope = #spmd.scope<group>}` at function top level emits `must be nested inside a spmd.forall with spmd.mapping = #spmd.level<group>`
      - `spmd.barrier {spmd.scope = #spmd.scope<group>}` nested inside a lane-level forall (without an enclosing group forall) emits the same error

- **AC-6**: The lit test suite at `test/SPMD/` runs via `cmake --build build --target check-spmd` with all tests passing.
  - Positive Tests (expected to PASS):
    - `test/SPMD/ops.mlir` — all CHECK patterns match
    - `test/SPMD/invalid.mlir` — all `expected-error` diagnostics are triggered
  - Negative Tests (expected to FAIL):
    - Intentionally broken IR (rank mismatch in forall) causes `spmd-opt --verify-diagnostics` to report a failure if the expected-error annotation is removed

- **AC-7**: The CPU closed-loop pipeline lowers S0 IR to executable LLVM IR for three kernel patterns: elementwise, reduction, stencil (no promotion).
  - Positive Tests (expected to PASS):
    - `spmd-opt ewise.mlir --normalize-spmd --materialize-spmd-tiling --convert-spmd-to-scf --convert-scf-to-openmp | mlir-translate --mlir-to-llvmir` produces valid LLVM IR
    - Same pipeline applied to `sum.mlir` (reduction) and `stencil_nopromote.mlir` produces valid LLVM IR
  - Negative Tests (expected to FAIL):
    - Running the pipeline on S0 IR that contains `gpu.thread_id` aborts at `VerifySPMDKernelSubset` with a diagnostic

- **AC-8**: `PromoteGroupMemory` pass transforms a 2D stencil S0 kernel into S2 IR with group alloc, cooperative copy, barrier, and rewritten loads; the result compiles to CPU via the full pipeline.
  - AC-8.1: IR transformation correctness
    - Positive Tests (expected to PASS):
      - After `--promote-group-memory`, the output IR contains `memref.alloc() : memref<..., #spmd.addr_space<group>>`
      - The output IR contains a `spmd.barrier {spmd.scope = #spmd.scope<group>}` inserted between the copy and compute foralls
      - The compute forall reads from the tile buffer (group addr space memref), not the original global memref
    - Negative Tests (expected to FAIL):
      - A kernel marked `spmd.memory_policy = no_promotion` is not transformed by `--promote-group-memory`
      - A kernel whose footprint exceeds `target.maxGroupMemBytes` is not promoted (pass emits a diagnostic remark and skips)
  - AC-8.2: End-to-end executability
    - Positive Tests (expected to PASS):
      - The full pipeline `S0 → normalize → plan → materialize → promote → lower-to-scf → lower-to-openmp → mlir-translate → llc` produces an object file without errors

---

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

The implementation includes: the complete build system with `spmd-opt` driver; all 5 ops and 5 attrs with ODS generic assembly format and full verifiers; `spmd.kernel` as `UnitAttr`; `VerifySPMDKernelSubset`, `NormalizeSPMD`, `PlanSPMDSchedule`, `MaterializeTilingAndMapping`, and `PromoteGroupMemory` passes with full legality and profitability checks for read-only stencil patterns; `SPMDToSCF` and `SPMDToOpenMP` conversion passes; a complete lit test suite covering ops, invalid IR, normalize, promotion, and lowering-to-openmp; and the `TargetDescriptor` struct wired into `PlanSPMDSchedule` and `PromoteGroupMemory`.

### Lower Bound (Minimum Acceptable Scope)

The implementation includes: a working CMake build producing `spmd-opt`; all 5 ops and 5 attrs with ODS generic format and the verifier rules specified in design-v1.md §5; `spmd.kernel` as `UnitAttr`; `VerifySPMDKernelSubset` checking op whitelist and barrier context; `MaterializeTilingAndMapping` (tile hint expansion to nested forall); `PromoteGroupMemory` for read-only stencil with hardcoded tile/halo; `SPMDToSCF` sequential lowering; and passing lit tests for `ops.mlir`, `invalid.mlir`, and `promotion.mlir`.

### Allowed Choices

**Fixed by design-v1.md (no discretion):**
- Dialect name: `spmd`
- Op set: exactly `spmd.forall`, `spmd.if`, `spmd.reduce`, `spmd.barrier`, `spmd.yield`
- Attr set: exactly `level`, `scope`, `reduction_kind`, `addr_space`, `memory_policy`
- Abstract execution levels: `seq/grid/group/lane/vector`
- Abstract memory spaces: `global/group/private`
- Three IR phases: S0 / S1 / S2
- `spmd.reduce` single-dimension only in MVP
- `PromoteGroupMemory` read-only promotion only in MVP

**User-decided (from issue resolution):**
- Assembly format: ODS generic format (no `hasCustomAssemblyFormat`) for MVP; pretty syntax deferred
- Build system entry point: `tools/spmd-opt/` as prerequisite milestone (before IR work)
- `spmd.kernel`: formal `UnitAttr` definition in ODS, done together with BarrierOp group-level check

**Open choices (implementer decides):**
- Can use either `DenseI64ArrayAttr` or `ArrayAttr<I64Attr>` for `tile_sizes`/`order` (design uses `DenseI64ArrayAttr`)
- Can use `mlir::func` or `mlir::arith` includes directly or via umbrella headers
- CPU `TargetDescriptor` defaults (e.g., `maxGroupMemBytes = 48*1024`) are heuristic

**Cannot use:**
- `hasCustomAssemblyFormat = 1` on any op until pretty-syntax milestone
- `gpu.*` / `omp.*` / `cf.*` ops inside S0/S1 kernels
- Write-back (store) promotion in `PromoteGroupMemory` MVP
- Multi-dimensional `spmd.reduce`

---

## Feasibility Hints and Suggestions

> This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

**Build system (prerequisite milestone):**
Mirror `llvm-project/mlir/examples/standalone/`:
```
tools/spmd-opt/
  CMakeLists.txt     # add_llvm_executable(spmd-opt ...) + target_link_libraries
  spmd-opt.cpp       # MlirOptMain with dialect + pass registration
```
Root `CMakeLists.txt` gains `add_subdirectory(tools)`.

**ODS attribute definition pattern (avoiding custom C++ enum plumbing):**
Use `MLIR_DEFINE_ENUM_ATTR` or the `EnumAttrCase`/`EnumAttr` ODS pattern if the MLIR version supports it; otherwise define the enum in C++ `extraClassDeclaration` as currently done. The key requirement is that parsing `#spmd.level<invalid>` produces a diagnostic.

**`spmd.kernel` UnitAttr:**
```tablegen
def SPMD_KernelAttr : SPMD_Attr<"Kernel", "kernel"> {
  let summary = "Marks a func.func as an SPMD kernel entry point";
  // UnitAttr semantics: presence = true, absence = false
}
```

**BarrierOp group-level check:**
Walk ancestor ops; cast each `ForallOp` ancestor's `spmd.mapping` attr to `LevelAttr`; check `kind == LevelKind::Group`. Fail if none found.

**PromoteGroupMemory (read-only stencil, simplest viable algorithm):**
1. Walk group-level `ForallOp` bodies
2. For each `memref.load` from a `global` memref: collect all index expressions
3. Compute bounding box of all accesses within the tile (static or affine-symbolic)
4. If bounding box fits in `target.maxGroupMemBytes` and reuse count > 1: emit promotion
5. Emit: `memref.alloc` (group), lane-level copy `ForallOp`, `BarrierOp`, rewrite loads

### Relevant References

- `llvm-project/mlir/examples/standalone/` — canonical out-of-tree dialect structure (CMake, tool driver, test setup)
- `llvm-project/mlir/lib/Dialect/SCF/IR/SCFOps.cpp` — reference for `ForallOp`-style custom assembly format (for future pretty-syntax milestone)
- `llvm-project/mlir/lib/Dialect/MemRef/IR/MemRefOps.cpp` — reference for address space handling in memref
- `llvm-project/mlir/lib/Dialect/GPU/IR/GPUDialect.cpp` — reference for barrier op and memory space attrs
- `spmd-dialect/include/spmd/IR/SPMDOps.td` — current op ODS (remove `hasCustomAssemblyFormat`)
- `spmd-dialect/include/spmd/IR/SPMDAttrs.td` — current attr ODS (add `SPMD_KernelAttr`)
- `spmd-dialect/lib/IR/SPMDOps.cpp` — current verifier stubs to complete
- `spmd-dialect/test/SPMD/ops.mlir` — existing positive test cases
- `spmd-dialect/test/SPMD/invalid.mlir` — existing negative test cases

---

## Dependencies and Sequence

### Milestones

1. **Build System & Tool Driver**: Establish compilable project skeleton with `spmd-opt` binary and working lit infrastructure.
   - Setup A: Add `tools/spmd-opt/spmd-opt.cpp` and `tools/spmd-opt/CMakeLists.txt` following standalone example
   - Setup B: Add `add_subdirectory(tools)` to root CMakeLists.txt; verify `cmake --build` and `spmd-opt --help` work
   - Setup C: Configure `test/lit.cfg.py` and `test/lit.site.cfg.py.in`; verify `check-spmd` target runs (may have 0 passing tests initially)

2. **Dialect IR**: Implement all 5 attrs and 5 ops with ODS generic format, complete verifiers, `spmd.kernel` UnitAttr, BarrierOp group-level check, and `VerifySPMDKernelSubset` pass. All lit tests in `ops.mlir` and `invalid.mlir` must pass.
   - IR A: Fix `SPMDAttrs.td` — add `SPMD_KernelAttr` (UnitAttr), rename `ReductionKindEnum` parameter to `ReductionKind` for consistency
   - IR B: Fix `SPMDOps.td` — remove `hasCustomAssemblyFormat = 1` from `ForallOp`, `IfOp`, `ReduceOp`; add `assemblyFormat` strings or rely on ODS default
   - IR C: Complete `SPMDOps.cpp` verifiers — `BarrierOp` group-level `LevelAttr` type check; `ReduceOp` `ReductionKindAttr` type check; `YieldOp` context-aware zero-operand check for `ForallOp` parent
   - IR D: Register `SPMD_KernelAttr` in `SPMDDialect.cpp`; update `VerifySPMDKernelSubset` to use typed `LevelAttr`
   - IR E: Run `check-spmd`; all `ops.mlir` and `invalid.mlir` tests pass

3. **CPU Closed-Loop Pipeline**: Implement normalize → materialize → lower-to-SCF → lower-to-OpenMP passes and verify three kernel patterns compile to LLVM IR.
   - CPU A: Implement `NormalizeSPMD` — 0-based lb normalization, unit-step normalization, single-element dim folding
   - CPU B: Implement `MaterializeTilingAndMapping` — expand tile hints into nested `ForallOp`; insert bounds guard `IfOp` for non-divisible sizes
   - CPU C: Implement `SPMDToSCF` — `ForallOp → scf.for`, `IfOp → scf.if`, `ReduceOp → scf.for with iter_args`, `BarrierOp → erase`
   - CPU D: Implement `SPMDToOpenMP` — group-level `scf.for` → `omp.parallel + omp.wsloop`
   - CPU E: Add `lower-to-openmp.mlir` lit test; verify elementwise / reduction / stencil pipelines pass

4. **Group Memory Promotion Demo**: Implement `PromoteGroupMemory` for read-only stencil pattern; demonstrate full S0→S2→CPU pipeline on 2D stencil.
   - Promo A: Implement `AccessSummaryAnalysis` — affine footprint computation for accesses inside group forall
   - Promo B: Implement `PromotionPlanAnalysis` — legality (bounded footprint, reuse>1, fits in group mem) + profitability (copy cost vs savings)
   - Promo C: Implement `PromoteGroupMemory` pass — alloc group memref, insert cooperative copy forall, insert `BarrierOp`, rewrite loads
   - Promo D: Wire `TargetDescriptor` defaults (CPU: `maxGroupMemBytes = 48*1024`, `cacheLineBytes = 64`)
   - Promo E: Add `promotion.mlir` lit test; verify stencil S0 → S2 → CPU pipeline end-to-end

### Dependency Graph

```
Milestone 1 (Build System)
    └─► Milestone 2 (Dialect IR)
            └─► Milestone 3 (CPU Pipeline)
                    └─► Milestone 4 (Promotion Demo)
```

Milestone 3 depends on Milestone 2 being complete (ops must parse and verify before lowering).
Milestone 4 depends on Milestone 3 (promotion output must feed into the lowering pipeline).

---

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers from this plan document.
- These terms belong in this plan file only, not in the resulting codebase.
- Use descriptive, domain-appropriate naming in code: e.g., `normalizeForallBounds`, `emitGroupMemPromotion`, `checkBarrierScope`.

### Key Design Decisions Carried Forward from design-v1.md

- `spmd.reduce` is single-dimension only in MVP; multi-dimensional reduction is expressed via nested `spmd.reduce` ops.
- `PromoteGroupMemory` is read-only (no write-back) in MVP.
- S0/S1 kernels must not contain `spmd.barrier` or group/private addr space memrefs; `VerifySPMDKernelSubset` enforces this.
- `steps` values are verified statically when constant, and assumed > 0 otherwise (no runtime assertion inserted).
- The GPU backend (Milestone 5: `SPMDToGPU`) is defined in design-v1.md but is outside MVP scope.

---

## Original Draft Reference

See `design-v1.md` in this directory for the full design specification, IR examples, and contribution framing.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach
<Text description, pseudocode, or diagrams showing ONE possible implementation path>

### Relevant References
<Code paths and concepts that might be useful>
- <path/to/relevant/component> - <brief description>

## Dependencies and Sequence

### Milestones
1. <Milestone 1>: <Description>
   - Phase A: <...>
   - Phase B: <...>
2. <Milestone 2>: <Description>
   - Step 1: <...>
   - Step 2: <...>

<Describe relative dependencies between components, not time estimates>

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

--- Original Design Draft Start ---

# SPMD Dialect Design v1

**Version:** 0.1
**Status:** Pre-implementation spec
**Scope:** MVP — `spmd` dialect + CPU 最小闭环 + group memory promotion demo

---

## 1. 核心主张

> 设计一个面向规则 SPMD kernel 的结构化中层 IR，使前端 DSL 的并行语义可以统一表示，mapping 与 memory hierarchy 优化在该 IR 层完成，而不是分散在各 DSL/backend 中。

三个核心贡献：

1. **Iteration-space as semantic center** — `spmd.forall` 是统一的逻辑并行域表示，不绑定 thread/block id。
2. **Delayed mapping via progressive refinement** — S0（语义态）→ S1（调度态）→ S2（物化态），并行语义先于执行语义。
3. **IR-level memory hierarchy optimization** — group memory promotion 作为中层 pass，不依赖 backend。

---

## 2. 程序子集（边界）

**面向的子集：**

- elementwise / map
- stencil
- structured reduction
- tiled BLAS-style kernels
- affine / quasi-affine 访问为主
- bounded structured control flow

**明确排除（第一版）：**

- pointer chasing / alias-heavy 程序
- 非结构化 CFG
- 动态并行
- 广义异步 runtime 编排

**与 Polyhedral IR 的关系：**

本 IR 借鉴 polyhedral 的迭代域表示，但不要求完整仿射分析，并增加 SPMD 执行语义（mapping/memory-space/barrier）作为一等概念；目标不是做 polyhedral scheduler，而是为 SPMD kernel 提供结构化的中层 IR。

---

## 3. 抽象执行模型

### 3.1 执行层级

| Level    | 语义                              | GPU 映射                    | CPU 映射               |
|----------|-----------------------------------|-----------------------------|------------------------|
| `grid`   | 全局问题空间                      | grid                        | 外层 chunk             |
| `group`  | 可协作执行单元，共享 group memory | block / workgroup           | parallel worker tile   |
| `lane`   | group 内单个 SPMD instance        | thread / invocation         | worker 内部迭代        |
| `vector` | lane 内 lockstep vector 执行      | per-thread vector fragment  | SIMD lane              |

### 3.2 内存空间

| Space     | 语义              | GPU 映射                     | CPU 映射             |
|-----------|-------------------|------------------------------|----------------------|
| `global`  | kernel 全局可见   | global memory                | main memory          |
| `group`   | 同一 group 内共享 | shared / workgroup memory    | tile scratchpad      |
| `private` | 单 lane 私有      | registers / local mem        | register / stack-local |

### 3.3 同步

MVP 只定义一个同步层级：

> `spmd.barrier {spmd.scope = #spmd.scope<group>}` — 同步当前 group 所有活跃 lane，建立 group memory 可见性边界。

---

## 4. 三阶段 IR 模型

### S0：Semantic SPMD IR

前端 lowering 的输出，纯语义态。

- 有 `spmd.forall / if / reduce`
- 所有 memref 在 `global` 地址空间
- **无** barrier，**无** group/private alloc
- mapping / tile attrs 为空

### S1：Scheduled SPMD IR

加入 schedule hints 后。

- `spmd.forall` 上携带 `spmd.mapping`, `spmd.tile_sizes`, `spmd.order`, `spmd.memory_policy`
- 未物化 tile/barrier/promoted buffer
- 仍然 backend-agnostic

### S2：Materialized SPMD IR

实现导向但仍 backend-agnostic 的物化态。

- tiling 已展开为 nested `spmd.forall`，各层有显式 `spmd.mapping`
- group/private memref 已物化（带 addr space attr）
- `spmd.barrier` 已插入
- 无 `gpu.thread_id` / `omp.parallel`（那是 backend lowering 的事）

---

## 5. Dialect 定义

**Dialect 名：** `spmd`

**依赖（复用，不重造）：** `func`, `arith`, `math`, `memref`, `affine`, `vector`

**自定义内容：** 5 个 op + 5 个 attr class

---

### 5.1 Attributes

#### `#spmd.level<...>`

```
LevelAttr ::= 'seq' | 'grid' | 'group' | 'lane' | 'vector'
```

用于标记 `spmd.forall` 的 mapping 层级。

#### `#spmd.scope<...>`

```
ScopeAttr ::= 'group'
```

MVP 只支持 `group`。

#### `#spmd.reduction_kind<...>`

```
ReductionKindAttr ::= 'add' | 'mul' | 'max' | 'min' | 'and' | 'or' | 'xor'
```

#### `#spmd.addr_space<...>`

```
AddressSpaceAttr ::= 'global' | 'group' | 'private'
```

用作 `memref` 的 memory space parameter：

```mlir
memref<32x32xf32, #spmd.addr_space<group>>
```

#### `#spmd.memory_policy<...>`

```
MemoryPolicyAttr ::= 'none' | 'prefer_group' | 'prefer_private' | 'no_promotion'
```

---

### 5.2 `spmd.forall`

**作用：** 矩形 iteration domain 上的逻辑并行执行集合。

**语法：**

```mlir
// 完整形式
spmd.forall (%i, %j) = (%lb0, %lb1) to (%ub0, %ub1) step (%s0, %s1)
    [attributes {...}] {
  ...
  spmd.yield
}

// 简写（lb=0, step=1）
spmd.forall (%i, %j) in (%N, %M) [attributes {...}] {
  ...
  spmd.yield
}
```

**Operands：** variadic `(lbs, ubs, steps)`, 类型均为 `index`

**Results：** 无

**Region：** 单 region，单 block，block args = induction vars（`index`），terminator = `spmd.yield`

**可选 Attrs：**

```
spmd.mapping       : #spmd.level<...>
spmd.tile_sizes    : DenseI64ArrayAttr   // 长度 = rank，值全正
spmd.order         : DenseI64ArrayAttr   // [0..rank-1] 的排列
spmd.memory_policy : #spmd.memory_policy<...>
```

**Verifier：**

1. `lbs`, `ubs`, `steps` 长度一致，rank ≥ 1
2. block args 个数 = rank，类型全为 `index`
3. 若 `spmd.order` 存在：必须是合法排列
4. 若 `spmd.tile_sizes` 存在：长度 = rank，值全正
5. 若 `spmd.mapping` 存在：必须是合法 `LevelAttr`
6. steps 在运行时必须 > 0（静态常量时 verifier 检查；动态值由调用方保证）

**语义：**

所有 `(i0, i1, ...)` 点构成的并行实例集合，不承诺执行顺序。
跨 iteration 的非结构化冲突访问属于非法 IR（未定义行为）。

**Canonicalization：**

- 规范到 0-based + unit-step 形式
- 单元素维度折叠
- 常量 trip count 折叠
- 非矩形域 → 矩形 + `spmd.if`

---

### 5.3 `spmd.if`

**作用：** per-instance 条件分支（不同 lane 走不同分支合法）。

**语法：**

```mlir
// 无结果
spmd.if %cond {
  spmd.yield
}

// 有结果
%r = spmd.if %cond -> (f32) {
  spmd.yield %x : f32
} else {
  spmd.yield %y : f32
}
```

**Verifier：**

1. `%cond` 类型为 `i1`
2. 有结果 → then/else 都必须存在，yield 类型和数量匹配 op 结果
3. 无结果 → else 可省略

---

### 5.4 `spmd.reduce`

**作用：** 显式结构化 reduction。MVP **只支持单维 reduction**（多维用 nested `spmd.reduce` 表达）。

**语法：**

```mlir
%result = spmd.reduce (%k) = (%lb) to (%ub) step (%step)
              init(%init_val)
              attributes {spmd.kind = #spmd.reduction_kind<add>} {
  %contrib = ...
  spmd.yield %contrib : f32
}
```

**Operands：** `lb, ub, step`（`index`）+ `init`（reduction type）

**Results：** 1 个，类型与 `init` 相同

**Region：** 单 block，block arg = `%k : index`，terminator = `spmd.yield`（恰好 1 个值）

**Verifier：**

1. `spmd.kind` 必须存在且合法
2. `init` 类型 = result 类型
3. body yield 恰好 1 个值，类型匹配 result
4. MVP：body 只允许纯算术 + non-volatile load，不允许未知 side effects

**语义：**

对 `[lb, ub)` 内所有 `k` 的 contribution 做 `kind` 组合，顺序不固定。

**浮点注意：** `add/mul` 的 reassociation 仅在 fast-math 语义下合法；否则 backend 应保守保序。

---

### 5.5 `spmd.barrier`

**作用：** group-scope 同步点。

**语法：**

```mlir
spmd.barrier {spmd.scope = #spmd.scope<group>}
```

**无 operands / results / regions。**

**Verifier：**

1. `spmd.scope` 必须存在，MVP 只允许 `group`
2. 其祖先中必须存在 `spmd.mapping = #spmd.level<group>` 的 `spmd.forall`

**使用阶段：** 不由前端生成；由 `PromoteGroupMemory` pass 插入，仅出现在 S2。

---

### 5.6 `spmd.yield`

**作用：** region terminator。

**Verifier（上下文相关）：**

- 在 `spmd.forall` 中：必须为空 yield
- 在 `spmd.if` 中：类型匹配 op results
- 在`spmd.reduce` 中：恰好 1 个值，类型匹配 result

---

### 5.7 Kernel 入口约定

复用 `func.func`，添加 `spmd.kernel` attr：

```mlir
func.func @kernel(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %N: index, %M: index)
    attributes {spmd.kernel} {
  ...
  func.return
}
```

**MVP kernel 约束：**

- 无返回值，输出通过 memref 写回
- 不允许递归
- 不允许未知 side-effecting `func.call`
- 参数类型：`index / integer / float` + `memref`

---

## 6. Pass Pipeline

```
Frontend lowering  (或 hand-written IR)
       │
       ▼
 [S0] NormalizeSPMD
       │  规范 bounds/steps；非矩形域 → 矩形 + mask
       ▼
 [S0] VerifySPMDKernelSubset
       │  op 白名单；barrier 作用域；S0 纯洁性检查
       ▼
 [S0→S1] PlanSPMDSchedule
       │  写入 tile_sizes / mapping / memory_policy attrs
       ▼
 [S1→S2] MaterializeTilingAndMapping
       │  展开 nested forall；标记各层 mapping
       ▼
 [S2] PromoteGroupMemory              ← 核心创新 pass
       │  分析 tile footprint；生成 group memref；插 barrier
       ▼
 [S2] VectorizeOrPrivatizeSPMD        (MVP 轻实现)
       │
       ▼
 LowerSPMDToBackend
       ├─ CPU: → scf + OpenMP + LLVM
       └─ GPU: → gpu dialect + NVVM/ROCDL
```

---

## 7. `PromoteGroupMemory` 算法框架

```
输入:  S2 中带 spmd.mapping=group 的 spmd.forall F
输出:  F.body 内插入 group memref alloc + cooperative copy
       + barrier + rewrite

步骤:
1. 收集 F.body 内所有 memref.load/store
2. 对每个候选 memref M:
   a. 计算 tile footprint（affine 访问区域）
   b. 若含 stencil halo，扩展 footprint
   c. Legality 检查:
      - footprint 可界定
      - 多个 lane 复用同一区域（reuse count > 1）
      - 无跨 group 写后读/写后写冲突
      - sizeof(footprint) ≤ target.maxGroupMemBytes
      - address 可重写为 tile-local index
   d. Profitability 检查:
      - copy-in amortized cost < 节省的 global 访问
      - 不显著恶化 occupancy
3. 对通过检查的 M，执行 promotion:
   a. memref.alloc → group addr space
   b. 在 compute forall 前插入 cooperative copy loop (lane-level forall)
   c. 插入 spmd.barrier（copy 完成后）
   d. 重写原 load → 访问 tile-local buffer
   e. MVP: 只做 read-only promotion，不做 write-back

注意: legality / profitability 判断由分析 pass 完成；
      不可分析的访问 ≠ 非法，只是不做激进优化。
```

---

## 8. Target Descriptor

```cpp
struct TargetDescriptor {
  enum BackendKind { CPU, CUDA, ROCM, SPIRV };
  BackendKind backend;
  int simdWidth;           // CPU SIMD / GPU vector width
  int subgroupWidth;       // GPU warp/wavefront size
  int maxGroupSize;        // max threads per block / work-items per workgroup
  int maxGroupMemBytes;    // shared/workgroup memory limit (bytes)
  int cacheLineBytes;
  int l1Bytes;
  int registerBudget;      // per-lane register budget (words)
  bool supportsGroupBarrier;
};
```

`PlanSPMDSchedule` 和 `PromoteGroupMemory` 以此为输入驱动决策。

---

## 9. 完整 IR 示例

### S0：elementwise

```mlir
func.func @ewise_square(%A: memref<?x?xf32>, %B: memref<?x?xf32>,
                         %N: index, %M: index)
    attributes {spmd.kernel} {
  spmd.forall (%i, %j) in (%N, %M) {
    %x = memref.load %A[%i, %j] : memref<?x?xf32>
    %y = arith.mulf %x, %x : f32
    memref.store %y, %B[%i, %j] : memref<?x?xf32>
    spmd.yield
  }
  func.return
}
```

### S0：reduction

```mlir
func.func @sum(%A: memref<?xf32>, %out: memref<1xf32>, %N: index)
    attributes {spmd.kernel} {
  %c0   = arith.constant 0 : index
  %c1   = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %sum = spmd.reduce (%k) = (%c0) to (%N) step (%c1)
             init(%zero)
             attributes {spmd.kind = #spmd.reduction_kind<add>} {
    %x = memref.load %A[%k] : memref<?xf32>
    spmd.yield %x : f32
  }
  memref.store %sum, %out[%c0] : memref<1xf32>
  func.return
}
```

### S2：stencil with group memory promotion

```mlir
func.func @stencil2d(%A: memref<?x?xf32>, %B: memref<?x?xf32>,
                      %N: index, %M: index)
    attributes {spmd.kernel} {
  %c0  = arith.constant 0  : index
  %c1  = arith.constant 1  : index
  %c8  = arith.constant 8  : index
  %c32 = arith.constant 32 : index
  %c34 = arith.constant 34 : index
  %c10 = arith.constant 10 : index

  spmd.forall (%ii, %jj) = (%c1, %c1) to (%N, %M) step (%c32, %c8)
      attributes {spmd.mapping = #spmd.level<group>} {

    %tile = memref.alloc() : memref<34x10xf32, #spmd.addr_space<group>>

    // cooperative copy: all lanes load halo tile from global
    spmd.forall (%li, %lj) = (%c0, %c0) to (%c34, %c10) step (%c1, %c1)
        attributes {spmd.mapping = #spmd.level<lane>} {
      %gi0 = arith.subi %ii, %c1 : index
      %gj0 = arith.subi %jj, %c1 : index
      %gi  = arith.addi %gi0, %li : index
      %gj  = arith.addi %gj0, %lj : index
      %vi  = arith.cmpi ult, %gi, %N : index
      %vj  = arith.cmpi ult, %gj, %M : index
      %ok  = arith.andi %vi, %vj : i1
      spmd.if %ok {
        %v = memref.load %A[%gi, %gj] : memref<?x?xf32>
        memref.store %v, %tile[%li, %lj]
            : memref<34x10xf32, #spmd.addr_space<group>>
        spmd.yield
      } else { spmd.yield }
      spmd.yield
    }

    spmd.barrier {spmd.scope = #spmd.scope<group>}

    // compute: read from tile (group memory)
    spmd.forall (%ti, %tj) = (%c0, %c0) to (%c32, %c8) step (%c1, %c1)
        attributes {spmd.mapping = #spmd.level<lane>} {
      %i   = arith.addi %ii, %ti : index
      %j   = arith.addi %jj, %tj : index
      %ti1 = arith.addi %ti, %c1 : index
      %tj1 = arith.addi %tj, %c1 : index
      %center = memref.load %tile[%ti,  %tj ]
          : memref<34x10xf32, #spmd.addr_space<group>>
      %right  = memref.load %tile[%ti,  %tj1]
          : memref<34x10xf32, #spmd.addr_space<group>>
      %down   = memref.load %tile[%ti1, %tj ]
          : memref<34x10xf32, #spmd.addr_space<group>>
      %t0 = arith.addf %center, %right : f32
      %t1 = arith.addf %t0, %down : f32
      memref.store %t1, %B[%i, %j] : memref<?x?xf32>
      spmd.yield
    }

    spmd.yield
  }
  func.return
}
```

---

## 10. Kernel Legality Pass：允许的 op 白名单

**允许：**

- `spmd.{forall, if, reduce, barrier, yield}`
- `arith.*`, `math.*`
- `memref.{load, store, subview, cast}`
- `affine.apply`
- `vector.*`（可选）
- `func.return`

**不允许（S0/S1）：**

- `cf.*`, `scf.while`
- `gpu.*`, `omp.*`
- 未知 side-effecting `func.call`
- `group` / `private` addr space memref（S0 中不允许）
- `spmd.barrier`（S0/S1 中不允许）

---

## 11. MVP 验收标准

### Phase 1：IR 立起来

- [ ] 5 个 attr class：ODS 定义 + C++ 实现
- [ ] 5 个 op：ODS 定义 + verifier + printer/parser
- [ ] `spmd.kernel` legality pass
- [ ] lit tests：`test/SPMD/ops.mlir` + `test/SPMD/invalid.mlir`

### Phase 2：CPU 最小闭环

- [ ] `NormalizeSPMD`
- [ ] `MaterializeTilingAndMapping`
- [ ] `LowerSPMDToSCF`
- [ ] `SCF → OpenMP → LLVM`
- [ ] 跑通：elementwise / reduction / stencil（无 promotion）

### Phase 3：Group memory promotion demo

- [ ] `PromoteGroupMemory`（只做 read-only stencil pattern）
- [ ] 2D stencil 全流程：S0 → S2 → CPU

### Phase 4：GPU mapping

- [ ] `group → block`，`lane → thread`
- [ ] group addr space → shared memory
- [ ] barrier → workgroup barrier
- [ ] 跑通 stencil on CUDA

---

## 12. 文件组织

```
spmd-dialect/
├── docs/
│   └── design-v1.md                  ← 本文件
├── include/spmd/IR/
│   ├── SPMDDialect.h
│   ├── SPMDOps.h
│   ├── SPMDOps.td
│   ├── SPMDAttrs.h
│   └── SPMDAttrs.td
├── lib/
│   ├── IR/
│   │   ├── SPMDDialect.cpp
│   │   ├── SPMDOps.cpp
│   │   └── SPMDAttrs.cpp
│   ├── Analysis/
│   │   ├── AccessSummaryAnalysis.cpp
│   │   └── PromotionPlanAnalysis.cpp
│   ├── Transforms/
│   │   ├── VerifySPMDKernelSubset.cpp
│   │   ├── NormalizeSPMD.cpp
│   │   ├── PlanSPMDSchedule.cpp
│   │   ├── MaterializeTilingAndMapping.cpp
│   │   └── PromoteGroupMemory.cpp
│   └── Conversion/
│       ├── SPMDToSCF/SPMDToSCF.cpp
│       ├── SPMDToOpenMP/SPMDToOpenMP.cpp
│       └── SPMDToGPU/SPMDToGPU.cpp
└── test/SPMD/
    ├── ops.mlir
    ├── invalid.mlir
    ├── normalize.mlir
    ├── promotion.mlir
    ├── lower-to-openmp.mlir
    └── lower-to-gpu.mlir
```

---

## 13. 论文 Contribution 表述

```
C1: A structured SPMD IR centered on spmd.forall, decoupling
    iteration-space semantics from backend execution mapping.

C2: A three-phase progressive refinement framework (Semantic →
    Scheduled → Materialized) that preserves analyzability while
    enabling target-specific optimization.

C3: An IR-level group-memory promotion framework that automatically
    identifies tile-reusable footprints and materializes cooperative
    loads, barriers, and address remapping across CPU and GPU targets.
```

---

## 附：与原始方案的主要修订

1. **`spmd.reduce` MVP 限制为单维**——避免多维 reduction 语义歧义；多维用 nested `spmd.reduce`。
2. **stencil 示例修正**——index 算术须先 `arith.addi` 再作 subscript，MLIR 不支持内联表达式。
3. **steps 约束放宽**——从"必须是编译期常量"改为"运行时必须 > 0"，允许动态 tile size。
4. **`PromoteGroupMemory` 补充算法草稿**——明确 legality/profitability 的判断流程。
5. **补充与 Polyhedral IR 的关系说明**——应对 reviewer 对比质疑。

--- Original Design Draft End ---

---

## Goal Tracker Rules

Throughout your work, you MUST maintain the Goal Tracker:

1. **Before starting a task**: Mark it as "in_progress" in Active Tasks
2. **After completing a task**: Move it to "Completed and Verified" with evidence (but mark as "pending verification")
3. **If you discover the plan has errors**:
   - Do NOT silently change direction
   - Add entry to "Plan Evolution Log" with justification
   - Explain how the change still serves the Ultimate Goal
4. **If you need to defer a task**:
   - Move it to "Explicitly Deferred" section
   - Provide strong justification
   - Explain impact on Acceptance Criteria
5. **If you discover new issues**: Add to "Open Issues" table

---

Note: You MUST NOT try to exit `start-rlcr-loop` loop by lying or edit loop state file or try to execute `cancel-rlcr-loop`

After completing the work, please:
0. If you have access to the `code-simplifier` agent, use it to review and optimize the code you just wrote
1. Finalize @/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/.humanize/rlcr/2026-03-27_08-24-48/goal-tracker.md (this is Round 0, so you are initializing it - see "Goal Tracker Setup" above)
2. Commit your changes with a descriptive commit message
3. Write your work summary into @/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/.humanize/rlcr/2026-03-27_08-24-48/round-0-summary.md
