# Research Plan: A Structured SPMD Middle-End IR for Simulation Kernels

**Version:** 1.0
**Date:** 2026-03-31
**Status:** For advisor discussion

---

## 1. One-Sentence Summary

> We present a structured iteration-space SPMD middle-end for simulation-oriented kernels, where logical parallelism, reductions, synchronization, and memory hierarchy remain explicit in IR, enabling compiler-synthesized group-memory promotion and hierarchical GPU reduction, while supporting unified CPU/OpenMP/GPU lowering and practical frontend integration from Taichi-style structured kernels.

---

## 2. Research Positioning

This work occupies the layer between Python-first GPU DSLs (Taichi, Warp, Numba) and target-specific backends (CUDA, OpenMP, LLVM).

```
Taichi kernel  ──┐
Warp kernel    ──┤──→  [SPMD IR middle-end]  ──→  CPU / OpenMP / GPU
Numba kernel   ──┘
```

**What we are not doing:**
- Competing with Triton on peak AI kernel throughput
- Replacing Halide/TVM as a schedule search system
- Reimplementing the MLIR `gpu` or `linalg` dialect
- Building a new Python DSL

**What we are doing:**
- Providing the structured middle-end that these Python-first simulation DSLs currently lack
- Making group-memory promotion and hierarchical reduction into compiler-automatic passes
- Unifying CPU/OpenMP/GPU lowering from a single structured IR

---

## 3. Research Insights

### Insight 1: Parallel semantics must precede execution semantics

A kernel's structure — which iterations are logically parallel, where reductions occur, where group synchronization is needed — must be expressed before deciding threadIdx/blockIdx bindings or OpenMP pragmas.

**Implication:** The IR's primary abstraction is `spmd.forall` (logical iteration-space), not `gpu.thread_id` or `omp.parallel`. Backend mapping is a lowering decision, not a semantic one.

### Insight 2: Structured reduction must be explicit

If reduction is not explicitly represented, the compiler only sees atomic writes to a shared address. This makes hierarchical reduction impossible.

**Implication:** `spmd.reduce` is not syntactic sugar — it is a prerequisite for optimization. Without it, the backend degrades to naive atomic, which is pathological on GPU at scale.

**Evidence already in the system:** `ReduceToHierarchicalGPU` is only triggered by explicit `spmd.reduce`; the atomic baseline demonstrates the performance gap.

### Insight 3: Memory hierarchy should be synthesized by the compiler

Shared/workgroup memory orchestration (tile buffer, cooperative copy, barrier, rewritten loads) should be derived automatically from:
- Tile shape
- Access footprint
- Reuse pattern

**Implication:** `PromoteGroupMemory` is an IR-level pass, not a user annotation or a backend heuristic. The programmer writes `spmd.forall` with a tile schedule; the compiler synthesizes the rest.

### Insight 4: The right thing to unify is middle-end semantics, not source syntax

Taichi, Warp, and Numba are different frontends with different syntax. But all three produce a common kernel subset:
- Regular iteration-space
- Structured control flow
- Reduction / accumulation
- Explicit buffer access

**Implication:** Building a unified middle-end is more productive than building a new source language. Frontends lower into SPMD IR; optimizations apply once, for all frontends.

### Insight 5: AD and sparse are important but must not destabilize v1

Differentiable simulation and sparse computation are real needs in the target domain. However, adding them to the v1 core would expand scope beyond what can be solidly demonstrated.

**Position:** The IR is designed to be **AD-compatible** and **sparse-extensible**, but is not AD-complete or sparse-complete in v1. This is explicitly stated in the spec and treated as future work.

---

## 4. Related Work

### 4.1 GPU kernel DSLs: Triton and similar systems

Triton is positioned as a high-throughput GPU kernel DSL for custom DNN compute kernels, using a blocked-program SPMD style.

**Our difference:** Triton is GPU-first and targets peak AI workloads. We target simulation-oriented kernels, emphasize structured middle-end semantics, and explicitly support CPU/OpenMP lowering. We do not compete on DNN kernel throughput.

**How to phrase it in the paper:**
> Triton provides a high-throughput DSL for GPU-first DNN kernels; our work instead provides a structured middle-end where parallelism, reduction, and memory hierarchy are explicit semantic objects enabling compiler synthesis, with a unified CPU/GPU lowering path.

### 4.2 Schedule-centric systems: Halide / TensorIR / TVM

Halide centers on algorithm/schedule separation and schedule-driven optimization. TVM and TensorIR are similar.

**Our difference:** We do not primarily compete on schedule search. Our contribution is semantic IR design: structured reduction and group-memory promotion that are legality-checked compiler transformations, not schedule choices.

**How to phrase it in the paper:**
> Compared with schedule-centric systems, we emphasize a semantic middle-end where group-memory promotion and reduction-aware GPU lowering are preserved as explicit IR structure and materialized by legality-checked passes, rather than primarily driven by schedule exploration.

### 4.3 MLIR: Linalg and GPU dialect

`linalg` provides structured op abstraction over loop nests. The `gpu` dialect provides GPU execution model primitives (thread_id, workgroup memory attribution, launch).

**Our difference:** `gpu` is already post-mapping (thread-id explicit); `linalg` does not natively model SPMD execution levels, group synchronization, or backend-agnostic memory-space semantics. We occupy the middle layer between them.

**How to phrase it in the paper:**
> MLIR's `linalg` and `gpu` dialects serve important roles, but leave a gap for a structured middle-end that explicitly models logical SPMD parallelism, reduction, and group-scope synchronization before target-specific mapping — the layer our dialect addresses.

### 4.4 Python-first simulation DSLs: Taichi / Warp / Numba

These are the frontends we integrate with, not competitors.

- **Taichi:** simulation-focused Python DSL; has its own CUDA backend but no IR-level group-memory promotion for arbitrary stencils.
- **Warp:** NVIDIA simulation/graphics DSL; restricted Python kernel syntax; no automatic shared-memory synthesis.
- **Numba:** general Python JIT; CUDA path exists; no structured SPMD middle-end.

**Our position:** We are not replacing these. We provide the structured middle-end they lack. The frontend evidence (Taichi → SPMD IR → GPU) demonstrates the system handles real DSL kernels, not just hand-authored MLIR.

**Key benchmark claim:** Taichi compiles stencil kernels to CUDA without automatic shared-memory promotion. Our pipeline (Taichi → SPMD IR → `PromoteGroupMemory` → GPU) adds this promotion automatically and demonstrates speedup over Taichi's native output.

---

## 5. System Architecture

```
┌────────────────────────────────────────────────┐
│   Frontend layer                               │
│   Taichi (CHI/TIR)  Warp (AST)  Numba (TypedIR)│
└───────────────────────┬────────────────────────┘
                        │  frontend subset lowering
                        ▼
┌────────────────────────────────────────────────┐
│   S0: Semantic SPMD IR                         │
│   spmd.forall / spmd.reduce / spmd.if          │
│   spmd.barrier / spmd.yield                    │
│   global / group / private addr spaces         │
└───────────────────────┬────────────────────────┘
                        │  Normalize / PlanSchedule / Materialize
                        ▼
┌────────────────────────────────────────────────┐
│   S1 / S2: Scheduled + Materialized SPMD IR    │
│   + tile_sizes / mapping / order / policy attrs│
│   + nested foralls / group buffers / barriers  │
└───────────────────────┬────────────────────────┘
                        │  IR-level optimization
                        ▼
┌────────────────────────────────────────────────┐
│   Optimization passes                          │
│   PromoteGroupMemory                           │
│   ReduceToHierarchicalGPU                      │
└───────────────────────┬────────────────────────┘
                        │  backend lowering
                        ▼
┌────────────────────────────────────────────────┐
│   Backend lowering                             │
│   SPMDToSCF / SPMDToOpenMP / SPMDToGPU         │
└───────────────────────┬────────────────────────┘
                        │
                        ▼
              CPU serial / OpenMP / NVPTX
```

---

## 6. IR Design

### 6.1 Core Operations

| Op | Semantics |
|----|-----------|
| `spmd.forall` | Logical parallel iteration-space; parallel legality guaranteed by frontend |
| `spmd.reduce` | Explicit structured reduction; prerequisite for hierarchical lowering |
| `spmd.if` | Per-instance conditional control flow |
| `spmd.barrier` | Group-scope synchronization; group memory visibility boundary |
| `spmd.yield` | Region terminator; semantics determined by enclosing op |

### 6.2 Core Attributes

| Attribute | Role |
|-----------|------|
| `level` | Abstract execution level: seq / grid / group / lane / vector |
| `scope` | Synchronization scope |
| `reduction_kind` | Reduction combiner: add / mul / min / max / and / or / xor |
| `addr_space` | Abstract memory space: global / group / private |
| `memory_policy` | Promotion hint: prefer_group / no_promotion |
| `tile_sizes` | Schedule attribute; does not change kernel semantics |
| `mapping` | Execution level assignment; schedule attribute |
| `order` | Loop order; schedule attribute |

### 6.3 Three-Phase IR Model

**S0 — Semantic:**
Pure logical form. Expresses parallelism and reduction without any execution binding.
No group buffers. No barriers. No target-specific attributes.

**S1 — Scheduled:**
Adds tile/mapping/order/policy planning information.
Does not change S0 mathematical semantics.

**S2 — Materialized:**
Tiling and mapping made explicit via nested `spmd.forall`.
Group buffers, cooperative copy patterns, and barriers inserted.
Remains backend-agnostic.

**Semantics-preserving refinement:** S1/S2 are valid refinements of S0 provided each pass's legality conditions (specified in `pass-contracts.md`) are satisfied.

### 6.4 Program Subset

**In scope:**
- Elementwise / map
- Stencil (affine or quasi-affine access)
- Structured reduction (single-kernel)
- Tiled BLAS-style kernels
- Bounded structured control flow

**Out of scope (v1):**
- Pointer-chasing / alias-heavy programs
- Unstructured CFG
- Dynamic parallelism
- General async runtime

---

## 7. Frontend Integration

### 7.1 Taichi (Phase 2, primary)

**Interception point:** Taichi Core IR (CHI/TIR)

**Supported subset:**

| Taichi IR node | SPMD IR target |
|----------------|----------------|
| `RangeFor` | `spmd.forall` |
| `GlobalLoad` | `memref.load` on `#spmd.addr_space<global>` |
| `GlobalStore` | `memref.store` on `#spmd.addr_space<global>` |
| `BinaryOp / UnaryOp` | `arith.*` |
| `IfStmt` (non-divergent guard) | `spmd.if` |
| `AtomicOpStmt` (reduction-to-scalar) | `spmd.reduce` |
| `AtomicOpStmt` (other) | atomic fallback |

**Unsupported (v1):** StructFor, sparse SNode iteration, autodiff tape, complex runtime constructs.

**Target kernels for benchmark:** elementwise, 2D stencil, 1D/2D reduction.

**Key claim:** Taichi's native CUDA output for stencil kernels does not perform automatic shared-memory promotion. Our pipeline adds `PromoteGroupMemory` at the IR level and demonstrates speedup over Taichi's native output on the same kernel.

### 7.2 Warp (Phase 3)

**Interception point:** Python AST (`@wp.kernel` body)

Warp kernels use restricted Python: `for i in range(N)`, explicit `wp.array` indexing, explicit `wp.atomic_add`. This maps cleanly to SPMD IR.

**Target:** canonical map / stencil / reduction kernels only.

### 7.3 Numba (Phase 3)

**Interception point:** Numba Typed IR (post type-inference, pre-LLVM)

At the Typed IR level, all array accesses have explicit types, loop structures are still visible, and reduction patterns can be matched.

**Target:** canonical `@cuda.jit` map / stencil / reduction kernels only. If logical iteration-space cannot be recovered, fall back to generic lowering.

---

## 8. Optimization Passes

### 8.1 PromoteGroupMemory (Contribution 2)

**Input:** S1 SPMD IR kernel with tile schedule and `prefer_group` memory policy.

**Analysis:**
- `AccessSummaryAnalysis`: decomposes access indices as `outer_iv + inner_iv * step + const_offset`, records (minOffset, maxOffset) per dimension.
- `PromotionPlanAnalysis`: 4 legality checks: bounded access, read-only tile body, footprint ≤ 48 KB, has reuse.

**Transformation:**
1. Allocate group-space tile buffer of size `(tileSize-1)*step + maxOffset - minOffset + 1` per dimension.
2. Insert outer tiling loop + cooperative copy forall.
3. Insert `spmd.barrier` between copy and compute.
4. Rewrite loads in compute body to use tile buffer.

**Known gap:** No bank-conflict padding currently. Must benchmark before paper — if speedup is reduced, add `+1` padding per dimension.

**Claim:** Automatically synthesizes shared-memory style execution from structured IR; programmer writes only the logical kernel.

### 8.2 ReduceToHierarchicalGPU (Contribution 3)

**Input:** `spmd.reduce` in a GPU-mapped kernel.

**Output:** 4-phase hierarchical reduction:
1. Thread-strided local accumulation (each thread reduces a private partial sum).
2. Cooperative write to workgroup scratch memory.
3. Tree reduction within the workgroup.
4. Thread 0 flushes final result via atomic to global accumulator.

**v1 scope:** single f32 result, `add` combiner, reduction-to-global-scalar idiom.

**Claim:** Explicit `spmd.reduce` is a necessary precondition for this optimization. Without it, the backend only sees unstructured atomics.

---

## 9. Benchmark Plan

### 9.1 Internal ablation (correctness + effectiveness)

| Experiment | Comparison | Expected result |
|-----------|------------|-----------------|
| Stencil: promoted vs non-promoted | Same SPMD kernel, with/without `PromoteGroupMemory` | Promoted faster (2-5× target) |
| Reduction: hierarchical vs atomic | Same `spmd.reduce`, two lowering paths | Hierarchical faster at scale |
| Backend portability | Same S0 IR → CPU serial / OpenMP / GPU | All three produce correct output |

### 9.2 External baseline (frontend evidence)

| Experiment | Comparison | Key claim |
|-----------|------------|-----------|
| Taichi stencil | Taichi native CUDA vs our SPMD pipeline | Taichi does not auto-promote; we do |
| Taichi reduction | Taichi native CUDA vs our hierarchical path | `spmd.reduce` enables better reduction |

**How to validate the Taichi baseline claim:** Run the same stencil kernel through Taichi, inspect the PTX for `.shared` — if absent, the claim holds. If Taichi already promotes, narrow the claim to our automatic IR-level approach.

### 9.3 What we do not compare against

- Triton peak DNN kernel throughput (different target domain)
- cuBLAS / cuDNN (different abstraction level)
- Hand-tuned CUDA for production workloads (out of scope)

---

## 10. Research Contributions

### Contribution 1: Structured SPMD middle-end IR

A kernel-centric IR that explicitly models logical iteration-space parallelism, structured reduction, group-scope synchronization, and backend-agnostic memory spaces — positioned between Python-first simulation DSLs and target backends.

### Contribution 2: IR-level group-memory promotion

A compiler pass that automatically synthesizes shared-memory style execution (tile buffer + cooperative copy + barrier + rewritten loads) from structured access analysis over `spmd.forall` bodies, without programmer annotation.

### Contribution 3: Reduction-aware hierarchical GPU lowering

A lowering path from explicit `spmd.reduce` to hierarchical GPU reduction (thread-local accumulation → workgroup tree → atomic flush), demonstrating that structured reduction in IR is a prerequisite for optimized GPU reduction.

### Contribution 4: Frontend integration evidence

A practical demonstration that the IR can serve as a shared middle-end target for Taichi-style structured simulation kernels, with Warp and Numba subset lowering as further evidence of generality.

---

## 11. Phased Implementation Plan

### Phase 1 — Benchmark data (immediate)

- Run `PromoteGroupMemory` promoted vs non-promoted stencil benchmark on GPU node
- Check for bank conflict in current implementation (inspect PTX, measure)
- If bank conflict present: add `+1` padding in `PromotionPlanAnalysis.cpp`, re-run
- Run hierarchical vs atomic reduction benchmark
- Output: speedup tables for Contributions 2 and 3

### Phase 2 — Taichi frontend (after Phase 1)

- Pin Taichi version
- Implement CHI/TIR → SPMD IR translator (Python, no Taichi source modification needed)
- Target kernels: elementwise, 2D stencil, 1D reduction
- Run Taichi native vs SPMD pipeline benchmark
- Output: external baseline data for Contribution 4

### Phase 3 — Spec and pass contract consolidation

- `docs/semantic-spec-v1.md`: already done (v1.0)
- `docs/pass-contracts.md`: complete legality conditions for all 7 passes
- Output: unified reference for paper writing

### Phase 4 — Warp / Numba subset (after Phase 2)

- Warp: Python AST → SPMD IR prototype
- Numba: Typed IR → SPMD IR prototype
- Canonical idioms only; fallback for anything outside subset
- Output: Contribution 4 extended evidence

### Phase 5 — Paper writing

- Introduction: Insights 1–4 as motivation
- Design: Section 6 of this plan
- Evaluation: Phase 1 + Phase 2 data
- Related work: Section 4 of this plan
- Future work: AD-compatible extension, sparse extension, multi-kernel scheduling

---

## 12. Future Work (explicitly deferred)

| Topic | Why deferred |
|-------|-------------|
| Full AD (backward IR, gradient ops, tape) | Requires significant IR extension; current IR is AD-compatible but not AD-complete |
| Full sparse (sparse scheduling, load balancing) | Requires structured sparse iteration model; current IR is sparse-extensible |
| Multi-kernel scheduling / fusion | Graph-level concern; current unit is single kernel |
| Bank-conflict-aware padding (autotuned) | Current: add fixed +1 padding if needed; general solution is future work |
| Autotuning / schedule search | Heuristic tile/block decisions sufficient for v1 |

---

## 13. Current Implementation Status

| Component | Status |
|-----------|--------|
| Dialect IR (5 ops, 5 attrs, verifiers) | Done |
| S0 → S1 → S2 pass pipeline | Done |
| SPMDToSCF / SPMDToOpenMP / SPMDToGPU | Done |
| PromoteGroupMemory | Done (bank conflict padding TBD) |
| ReduceToHierarchicalGPU | Done (f32 add, single result) |
| Lit test suite (34 tests) | Done |
| Differential / robustness tests | Done |
| GPU benchmark data | **Pending (Phase 1)** |
| Taichi frontend | **Pending (Phase 2)** |
| Warp / Numba frontend | **Pending (Phase 4)** |
| semantic-spec-v1.md | Done |
| pass-contracts.md (complete) | Partial |
