# SPMD IR Semantic Specification v1

**Version:** 1.0
**Status:** Frozen for paper / proposal / frontend development
**Scope:** Single-kernel structured SPMD programs

---

## 1. Scope

### 1.1 Current Scope

This specification describes a **kernel-centric** structured SPMD middle-end IR targeting:

- Regular parallel computation within a single kernel
- Structured iteration-space
- Structured reduction
- Group-scope synchronization
- Dense / affine-or-quasi-affine memory access
- CPU / OpenMP / GPU lowering

### 1.2 Explicitly Out of Scope (v1)

- Multi-kernel scheduling / fusion / graph-level scheduling
- Full autodiff semantics
- Full sparse semantics
- Irregular pointer-heavy programs
- Unstructured CFG
- Dynamic parallelism / async runtime semantics

### 1.3 Design Goals

This IR is not intended to cover arbitrary programs. Its goal is to provide, for **regular SPMD kernels**, a middle layer that:

- Explicitly represents parallel semantics
- Explicitly represents reduction semantics
- Explicitly represents group memory and synchronization
- Supports IR-level optimization and CPU / OpenMP / GPU lowering

---

## 2. Semantic Positioning

### 2.1 IR Layer Position

This IR sits between:

- Frontend DSL / frontend subset lowering (Taichi, Warp, Numba, hand-authored MLIR)

and

- Backend execution model / target-specific lowering (NVPTX, OpenMP, CPU serial)

It is not:
- A source language
- A backend-specific GPU IR
- A whole-program runtime IR
- A new Python DSL

### 2.2 Target Frontend Ecosystem

The IR is designed to serve as a shared middle-end target for Python-first simulation DSLs:

| Frontend | Interception point | Kernel subset |
|----------|--------------------|---------------|
| Taichi | CHI / TIR | RangeFor, GlobalLoad/Store, BinaryOp, IfStmt, AtomicOpStmt |
| Warp | Python AST | for/range loops, wp.array access, wp.atomic_add |
| Numba | Typed IR (post type-inference) | @cuda.jit map / stencil / reduction idioms |

None of these frontends currently performs IR-level group-memory promotion. This IR provides that capability as a shared middle-end pass, applicable to all frontends.

**What is unified is middle-end semantics, not source syntax.** Each frontend lowers its own structured kernel subset into SPMD IR; optimizations are applied once at the middle-end level.

### 2.2 Kernel-Centric Semantic Unit

The primary semantic unit of this IR is the **single kernel**.

A module may contain multiple kernels, but the main target of optimization and lowering is:

- A `spmd.kernel`-attributed function
- Or a structured SPMD kernel body

### 2.3 `spmd.kernel` Function Attribute

A `func.func` marked with the `spmd.kernel` attribute denotes a kernel-level compilation unit in the SPMD pipeline.

The S0/S1/S2 classification applies primarily to such kernel functions, describing progressively refined forms of the same kernel body.

Ordinary `func.func` operations without `spmd.kernel` are not required to satisfy the structured SPMD kernel subset and are outside the main optimization scope.

This also answers two common questions:
- **Why `func.func` rather than a custom function op?** Because `func.func` + attribute is idiomatic MLIR and avoids duplicating the function infrastructure.
- **Why is S0/S1/S2 a function-level classification?** Because the refinement describes the overall state of a kernel body, not individual op tags.

---

## 3. Core Semantic Objects

### 3.1 Iteration-Space

Logical parallel domain, expressed by `spmd.forall`.

### 3.2 Structured Reduction

Explicit structured reduction, expressed by `spmd.reduce`.

### 3.3 Per-Instance Control Flow

Independent control flow for each logical instance, expressed by `spmd.if`.

### 3.4 Group-Scope Synchronization

Intra-group synchronization, expressed by `spmd.barrier`.

### 3.5 Abstract Memory Spaces

Backend-agnostic abstract address spaces:

- `global`
- `group`
- `private`

---

## 4. Abstract Execution Model

### 4.1 Logical SPMD Instances

`spmd.forall` defines a set of logical SPMD instances.
Each iteration point corresponds to one logical instance.

Example:

```mlir
spmd.forall (%i, %j) in (%N, %M) { ... }
```

This defines a 2D logical parallel domain where each `(i, j)` is one logical instance.

### 4.2 Abstract Execution Levels

Abstract execution hierarchy expressed via `#spmd.level<...>`:

- `seq`
- `grid`
- `group`
- `lane`
- `vector`

These levels describe **abstract execution structure**, not direct bindings to hardware instructions.

### 4.3 Backend Mapping

These levels are mapped to target execution models during lowering:

- GPU:
  - `group` → thread block / workgroup
  - `lane` → thread / invocation
- CPU:
  - `group` → parallel worker / team
  - `lane` → loop iteration
  - `vector` → SIMD lane

---

## 5. Memory Model

### 5.1 Address Spaces

The IR uses three abstract address spaces:

- `global`: globally visible buffer across the kernel
- `group`: locally shared buffer within one group
- `private`: per-lane private storage

### 5.2 Memory-Space Semantics

Address spaces are **semantic properties**, not merely implementation hints.

- `global` denotes external input/output or wide-scope state
- `group` denotes group-local cooperative storage
- `private` denotes thread/lane-local scratch or scalar state

### 5.3 Address-Space Lowering

Concrete lowering to:

- GPU workgroup / shared memory
- CPU local scratch / stack / register
- Target-specific address spaces

is a backend concern and does not change middle-layer semantics.

---

## 6. Core Operations and Their Semantics

### 6.1 `spmd.forall`

#### Semantics

`spmd.forall` represents a **logically parallel iteration-space**.

Its core semantics are not "a loop that might be parallelized." Instead:

**Semantic assumption.**
A `spmd.forall` denotes an iteration space whose instances are already known to be logically parallel, as guaranteed by frontend semantics or by conservative compiler analysis. The op itself does not infer or prove parallel legality; instead, any non-independent cross-iteration behavior must have been made explicit through structured reductions, atomics/fallback, or other legal mechanisms before reaching `spmd.forall`.

#### Execution Order

`spmd.forall` makes no commitment to iteration execution order.
Different iterations may run in parallel, be reordered, blocked, or distributed, provided program semantics are preserved.

#### Well-Formedness

If unmodeled cross-iteration conflicting accesses exist (RAW, WAR, WAW), the IR is semantically ill-formed. See Section 7.

#### Note

`spmd.forall` does not represent speculative parallelization. It represents an **already-confirmed logical parallel domain**.

---

### 6.2 `spmd.if`

#### Semantics

`spmd.if` represents per-instance conditional control flow.

Different lanes / iterations may follow different branches based on their own local data.

#### Key Points

- This is per-instance control flow, not a global condition.
- Backends may implement it as branch, predication, or mask execution.

#### Constraint

If a branch contains operations that must execute in a convergent manner (such as a barrier), additional convergence constraints apply.

---

### 6.3 `spmd.reduce`

#### Semantics

`spmd.reduce` represents a structured reduction over a logical domain.

It explicitly models:
- Reduction domain
- Init value
- Combiner
- Reduction body

#### Key Point

`spmd.reduce` is the formal semantic representation of a reduction. It is not inferred from atomic patterns or write conflicts.

#### Combiner Requirements

The combiner must satisfy the conditions for legal parallel reduction. Typically required:
- Associativity
- Commutativity

#### Numeric Semantics

- Integer reduction: exact semantics required.
- Floating-point reduction: reassociation is permitted under valid configurations; results are subject to numerical consistency tolerance, not bitwise identity.

#### Purity

In v1, the reduction body should maintain pure-computation semantics and avoid unknown side effects.

#### Two-Level Legality Distinction

`spmd.reduce` has two distinct legality levels:

1. **IR semantic well-formedness**: The reduction is a valid, well-formed IR operation (combiner is declared, types match, body has no disallowed side effects). An IR can be semantically well-formed even if it does not qualify for aggressive optimization.

2. **Pass-specific optimization legality**: Whether a specific optimization pass (e.g., `ReduceToHierarchicalGPU`) can apply. If optimization legality conditions are not met, the pass falls back to a conservative lowering. This does not make the IR ill-formed.

These two levels must not be conflated.

#### Ill-Formed Cases

If the reduction body contains disallowed side effects, unanalyzable external calls, or a combiner that does not meet associativity/commutativity requirements, the reduction must not enter the structured hierarchical optimization path. It may fall back, but must not silently miscompile.

---

### 6.4 `spmd.barrier`

#### Semantics

`spmd.barrier` represents group-scope synchronization.

Currently only supported:

```mlir
spmd.barrier {spmd.scope = #spmd.scope<group>}
```

#### Semantic Guarantees

This barrier guarantees:

1. All active lanes in the same group must reach this point before any proceed.
2. Writes to `group` memory before the barrier are visible to all same-group lanes after it.

#### Convergence Constraint

`spmd.barrier` must be placed at a convergent point reachable by all relevant lanes.
Placing a barrier inside a divergent per-instance conditional branch is illegal.

#### Backend Mapping

- GPU: workgroup / block barrier
- OpenMP / CPU: equivalent group-scope synchronization

---

### 6.5 `spmd.yield`

#### Semantics

Terminator for structured regions. Its semantics are determined by the enclosing op:

- In `spmd.forall`: terminates the iteration body.
- In `spmd.if`: returns the branch result.
- In `spmd.reduce`: returns the current reduction contribution.

---

## 7. Race Semantics and Undefined Behavior

### 7.1 General Rule

Any unmodeled conflicting cross-iteration behavior violates the semantic well-formedness conditions of `spmd.forall`.

- If such a violation is detected by the verifier or legality checks, the IR **must be rejected**.
- If it is not detected due to limited analysis capability, and the program is nevertheless lowered as structured SPMD, the resulting behavior is **undefined**.

### 7.2 Why This Matters

This rule ensures:

- `spmd.forall` maintains clean logical parallel semantics.
- The compiler does not need to "guess" user intent from write conflicts.
- Reduction / shared update / atomic update are clearly distinguished.

---

## 8. Semantic Classification of Attributes

### 8.1 Semantic Attributes

Attributes that directly affect program semantics or execution abstraction:

- `level`
- `scope`
- `reduction_kind`
- `addr_space`

### 8.2 Schedule / Implementation Attributes

Attributes that primarily affect optimization and implementation, not the mathematical kernel semantics:

- `tile_sizes`
- `order`
- `mapping`
- `memory_policy`

These may affect how computation is blocked, how it maps to group/lane levels, and whether group memory promotion is preferred. Under valid transformation conditions, they must not change program results.

---

## 9. S0 / S1 / S2 Refinement Semantics

### 9.1 S0 — Semantic State

Pure semantic form:
- Expresses logical parallel domain
- Expresses structured reduction
- Contains no target-specific materialization

### 9.2 S1 — Scheduled State

Planning / scheduling form:
- Introduces tile / mapping / order / policy planning information
- Does not change the mathematical semantics of S0

### 9.3 S2 — Materialized State

Implementation-directed, but still backend-agnostic:
- Tiling and mapping decisions are made explicit
- May contain nested `spmd.forall`
- May contain group buffers, cooperative copy patterns, and barriers
- Is a semantics-preserving refinement of S0

### 9.4 Semantics-Preserving Refinement

S1 and S2 are semantics-preserving refinements of S0, provided the legality conditions of each pass are satisfied.

The precise legality conditions of each refinement step are pass-specific and are listed in `pass-contracts.md`.

No pass may change the mathematical kernel semantics unless explicitly documented as a non-semantic transformation outside the current scope.

---

## 10. Legality and Fallback Philosophy

### 10.1 Structured Path

For regular, analyzable, well-formed cases:
- Enter the structured SPMD path
- Allow aggressive optimization

### 10.2 Fallback Path

For irregular, unanalyzable, or legality-failing cases:
- Do not enter unsafe optimization
- Use conservative lowering or a generic path

### 10.3 Principle

> **If uncertain, do not optimize.**

This is one of the core principles of this IR and its compilation strategy.

---

## 11. Frontend Obligations

When lowering a program to `spmd` IR, the frontend must satisfy:

1. Lower only iteration-spaces already known to be logically parallel into `spmd.forall`.
2. Explicitly lower reduction patterns into `spmd.reduce`.
3. Preserve shared conflicting updates that cannot be structured as reductions as explicit atomic/fallback operations.
4. Not disguise programs with evident loop-carried dependences as `spmd.forall`.

---

## 12. Backend Obligations

Backend lowering must:

1. Not break the logical parallel semantics of `spmd.forall`.
2. Preserve the legal numeric semantics of `spmd.reduce`.
3. Correctly implement the group-scope synchronization and memory visibility of `spmd.barrier`.
4. Correctly map `global / group / private` memory spaces.
5. Apply target-specific optimization only when legality conditions are satisfied.

---

## 13. AD Compatibility

### 13.1 Current Position

This IR is **AD-compatible**, but not **AD-complete**.

### 13.2 Why AD-Compatible

The IR preserves structural semantics required by future AD:
- Structured iteration domains
- Explicit reduction
- Structured control flow
- Explicit synchronization
- Explicit memory-space distinction

### 13.3 Current Non-Goals

- Backward IR
- Gradient ops
- Tape / checkpointing semantics
- Full reduction-gradient semantics

---

## 14. Sparse Extensibility

### 14.1 Current Position

This IR is **sparse-extensible**, but not **sparse-complete**.

### 14.2 What Is Allowed

The IR can represent indirect / sparse-like access.

### 14.3 What Is Optimized

The current strong optimization framework targets:
- Regular
- Structured
- Analyzable
- Dense / quasi-dense SPMD kernels

### 14.4 Current Non-Goals

- Sparse scheduling
- Sparse load balancing
- Sparse iteration partitioning
- Full sparse-specific legality and optimization framework

---

## 15. In-Scope / Out-of-Scope Summary

### In Scope

- Kernel-centric structured SPMD IR
- Logical iteration-space parallelism
- Structured reduction
- Group memory promotion
- Hierarchical GPU reduction
- CPU / OpenMP / GPU lowering

### Out of Scope

- Whole-program multi-kernel scheduling
- Full AD
- Full sparse optimization
- General async runtime semantics
- Unrestricted irregular programs

---

## Summary

> **SPMD IR v1 is semantically complete for regular single-kernel structured SPMD programs, with explicit iteration-space, reduction, synchronization, and memory-space semantics. It is AD-compatible and sparse-extensible, but not yet a full semantic system for differentiation or sparse optimization.**
