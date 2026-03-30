# Hierarchical GPU Reduction Lowering

## Goal Description

Fix the GPU reduction performance regression (currently 0.11× vs CPU serial) by
implementing a two-level hierarchical reduction: intra-block shared-memory tree
reduction followed by a single global atomic per block.

The work has two parts:
1. **Source restructuring**: rewrite the reduction benchmark kernel to use
   `spmd.reduce` (the proper semantic IR) instead of the current manual
   `spmd.forall` + `atomic_rmw` idiom.
2. **Compiler pattern**: add `ReduceToHierarchicalGPU` in `SPMDToGPU.cpp` that
   matches the `spmd.reduce`→`atomic_rmw` idiom and emits thread-strided local
   accumulation + statically-unrolled shared-memory tree reduction + single
   global atomic per block.

V1 scope is f32 Add reduction-to-scalar-accumulator only. The existing
`ReduceToSCFForGPU` pattern becomes the verified fallback for all out-of-scope
cases.

---

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for
deterministic verification.

- **AC-1**: `ReduceToHierarchicalGPU` fires for an f32 Add `spmd.reduce` whose
  result feeds a scalar `atomic_rmw addf` inside a `gpu.launch` with
  compile-time constant blockDim.
  - Positive Tests (expected to PASS):
    - `reduction-hierarchical-gpu.mlir`: IR contains a workgroup attribution
      buffer of type `memref<BLOCK_DIM x f32, workgroup>`.
    - `reduction-hierarchical-gpu.mlir`: IR contains exactly `log2(blockDim)`
      `gpu.barrier` ops in the tree section.
    - `reduction-hierarchical-gpu.mlir`: IR contains a `scf.if` guarding `tx ==
      0` that holds the single `memref.atomic_rmw addf`.
    - `reduction-hierarchical-ptx.mlir`: PTX output contains `.shared`, at least
      one `bar.sync`, and exactly one `atom.add.f32` (not N).
  - Negative Tests (expected to FAIL):
    - `reduction-hierarchical-gpu.mlir`: `spmd.reduce` op no longer present after
      lowering (replaced by the tree sequence).
    - `reduction-hierarchical-gpu.mlir`: `memref.atomic_rmw` that consumed
      `%sum` is eliminated (a new one inside the tx==0 guard replaces it).

- **AC-2**: Legality guard — pattern correctly falls back to `ReduceToSCFForGPU`
  for out-of-scope cases.
  - AC-2.1: Non-pure body (contains `func.call`) → fallback fires, debug remark
    emitted.
    - Positive: `reduction-hierarchical-fallback.mlir` (with
      `--mlir-print-diagnostics`): remark mentions "non-pure reduce body".
    - Positive: fallback produces a `scf.for` loop, not a workgroup buffer.
    - Negative: workgroup attribution absent in fallback output.
  - AC-2.2: Non-Add kind (e.g., Min or Max) → fallback fires.
    - Positive: `reduction-hierarchical-f32-kinds.mlir`: Add kernel → workgroup
      buffer present; Min/Max kernel → workgroup buffer absent.
    - Negative: Min/Max kernel must not silently produce wrong results — fallback
      `scf.for` must appear.

- **AC-3**: Atomic-only baseline (`spmd.forall` + `atomic_rmw`) continues to
  compile and produce correct PTX via `ReduceToSCFForGPU`, unchanged.
  - Positive: existing `lower-to-gpu-nvptx-reduction.mlir` RUN line passes
    unchanged.
  - Positive: PTX from this file still contains `atom.add.f32`.
  - Negative: no workgroup attribution in PTX from the atomic-only kernel.

- **AC-4**: `sum.mlir` (existing test that uses `spmd.reduce`) continues to
  pass after pattern priority change.
  - Positive: `check-quick` includes `sum.mlir` with exit 0.
  - Negative: no regression in `sum.mlir` output IR.

- **AC-5**: Source kernel restructured — the new hierarchical reduction test
  kernel uses `spmd.reduce` as its input IR.
  - Positive: `test/SPMD/lower-to-gpu-nvptx-hierarchical-reduction.mlir`
    contains `spmd.reduce` in the source (FileCheck verifies the op is present
    before lowering).
  - Negative: `spmd.forall` + `atomic_rmw` pattern does NOT appear as the sole
    reduction mechanism in the hierarchical test kernel.

- **AC-6**: Correctness — hierarchical kernel produces numerically correct
  results for all required input shapes.
  - Positive: `run_reduction.py --hierarchical` passes for sizes
    1, 32, 33, 255, 256, 257, 1024, 65536, 1M, 16M; non-multiples N=1000,
    65537; all-zeros; all-ones; 3× multi-launch with re-zeroed accumulator.
  - Positive: relative error `< 1e-3` vs numpy reference for all above cases.
  - Negative: `run_reduction.py --hierarchical` with an intentionally wrong
    result (e.g., uninitialized accumulator) must report failure.

- **AC-7** *(hard requirement)*: GPU speedup > 1× vs CPU serial for
  N ≥ 64K using the hierarchical kernel (baseline was 0.11×).
  - Positive: robustness CSV column `reduction_hierarchical` shows speedup > 1.0
    for N = 65536, 1M, 16M.
  - Negative: atomic-only column remains ≤ 1× (regression check, not a fix).

- **AC-8**: All 29 existing lit tests pass unchanged (`check-quick`).
  - Positive: `ninja check-quick` exits 0.
  - Negative: any previously-passing test that now FAILs is a blocker.

- **AC-9**: Full pipeline (`check-full`) passes end-to-end.
  - Positive: `ninja check-full` exits 0, including `run-differential.sh`
    `reduction_hierarchical` row (cpu_ok / omp_ok / gpu_ok = PASS).
  - Negative: a `check-full` failure in any kernel not modified by this work is
    a blocker.

---

## Path Boundaries

This design is highly deterministic. Upper and lower bounds converge closely.

### Upper Bound (Maximum Acceptable Scope)

The implementation includes:
- `ReduceToHierarchicalGPU` pattern with legality helper and statically-unrolled
  tree reduction for f32 Add.
- `addWorkgroupAttribution` called from within the rewrite pattern to inject the
  scratch buffer into the existing `gpu.launch`.
- 4 new lit tests covering: hierarchical IR, PTX output, fallback, and kind
  enforcement.
- Updated `run_reduction.py` with `--hierarchical` flag.
- Updated `run-robustness-validation.sh` with `reduction_hierarchical` column.
- `run-differential.sh` with `reduction_hierarchical` row.
- `sum.mlir` regression confirmed passing.

### Lower Bound (Minimum Acceptable Scope)

The implementation includes:
- `ReduceToHierarchicalGPU` pattern that satisfies AC-1 through AC-9.
- At minimum 2 lit tests: hierarchical IR check and fallback check.
- `run_reduction.py --hierarchical` correctness coverage for the required sizes.
- Robustness CSV column showing > 1× speedup for N ≥ 64K.

### Allowed Choices

- **Can use**: `launchOp.addWorkgroupAttribution(type, loc)` to inject the
  scratch buffer into the already-built `gpu.launch` from within `matchAndRewrite`.
- **Can use**: C++ `for` loop at codegen time to emit the statically-unrolled
  tree (stride from `blockDim/2` down to `1`, halving each iteration).
- **Can use**: `gpu::BarrierOp` inserted directly by the rewrite pattern (no
  `spmd.barrier` required inside a pattern running post-`gpu.launch` construction).
- **Can use**: `PatternBenefit(2)` for `ReduceToHierarchicalGPU` vs default
  benefit for `ReduceToSCFForGPU` to express priority.
- **Cannot use**: `scf.for` for the tree reduction (halving stride cannot be
  expressed as fixed step — static unrolling is required per design decision).
- **Cannot use**: warp-level `shfl.down` shuffles (deferred to V2).
- **Cannot use**: `memref.alloc` with group address space (that path is only
  valid before `ConvertSPMDToGPU` runs; the pattern executes after).
- **Cannot use**: any change to `spmd.reduce` op definition.

---

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are
> conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

**Workgroup buffer injection** (critical non-obvious step):

`ReduceToHierarchicalGPU` runs *after* `gpu.launch` is already constructed by
`ConvertSPMDToGPU`. The scratch buffer cannot go through the `memref.alloc`
collection path used by `PromoteGroupMemory`. Instead, from within
`matchAndRewrite`:

```cpp
// Find parent gpu.launch
auto launchOp = op->getParentOfType<gpu::LaunchOp>();

// Inject a new workgroup attribution block argument
auto smemType = MemRefType::get(
    {blockDim}, f32Ty,
    /*layout=*/{},
    gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup));
Value smem = launchOp.addWorkgroupAttribution(smemType, loc);
```

**Pattern registration with explicit benefit**:

```cpp
patterns.add<IfToSCFIfGPU>(ctx);
patterns.add<ReduceToHierarchicalGPU>(ctx, PatternBenefit(2));  // higher priority
patterns.add<ReduceToSCFForGPU>(ctx);                          // default benefit=1
```

**Static tree unrolling in C++**:

```cpp
// Emit log2(blockDim) scf.if + gpu.barrier pairs
for (int64_t stride = blockDim / 2; stride >= 1; stride /= 2) {
  Value strideVal = b.create<arith::ConstantIndexOp>(loc, stride);
  Value cond = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, tx, strideVal);
  b.create<scf::IfOp>(loc, cond, /*withElseRegion=*/false, [&](OpBuilder &ib, Location il) {
    Value a  = ib.create<memref::LoadOp>(il, smem, ValueRange{tx});
    Value ix = ib.create<arith::AddIOp>(il, tx, strideVal);
    Value bv = ib.create<memref::LoadOp>(il, smem, ValueRange{ix});
    Value s  = ib.create<arith::AddFOp>(il, a, bv);
    ib.create<memref::StoreOp>(il, s, smem, ValueRange{tx});
    ib.create<scf::YieldOp>(il);
  });
  b.create<gpu::BarrierOp>(loc);
}
```

**blockDim extraction** (for L5 legality check):

Walk from `spmd.reduce` up to the `gpu.launch` and inspect its `blockSizeX`
operand. If it traces back to an `arith.constant`, extract the integer value.

### Relevant References

- `lib/Conversion/SPMDToGPU/SPMDToGPU.cpp` — existing `ReduceToSCFForGPU` (lines ~211–266) and registration (lines ~593–597); pattern to add directly above.
- `lib/Transforms/PromoteGroupMemory.cpp` — shows how `memref.alloc` with group address space is created pre-lowering (contrast: hierarchical reduction uses `addWorkgroupAttribution` instead).
- `test/SPMD/lower-to-gpu-nvptx-reduction.mlir` — current atomic-only kernel; becomes the fallback baseline test.
- `test/SPMD/sum.mlir` — existing test using `spmd.reduce`; must continue to pass after priority change.
- `harness/run_reduction.py` — current harness (atomic path, 10-param ABI `atomic_sum_kernel`); `--hierarchical` flag adds a parallel path.
- `include/spmd/IR/SPMDOps.td` — `spmd.reduce` op definition (lb/ub/step/init operands, single-region body, `spmd.kind` attr).

---

## Dependencies and Sequence

### Milestones

1. **Source restructuring**: Create the hierarchical source kernel using `spmd.reduce`.
   - Review `test/SPMD/sum.mlir` to understand existing `spmd.reduce` usage and
     check it is unaffected.
   - Create `test/SPMD/lower-to-gpu-nvptx-hierarchical-reduction.mlir` with the
     new `spmd.reduce`-based kernel (AC-5). Keep the existing atomic-only file
     unchanged (AC-3).
   - Verify `check-quick` still passes before adding any new compiler code.

2. **Compiler pattern**: Add `ReduceToHierarchicalGPU` to `SPMDToGPU.cpp`.
   - Implement legality helper (`isHierarchicalReduceCandidate`) checking L1–L6.
   - Implement `matchAndRewrite`: inject workgroup attribution via
     `addWorkgroupAttribution`, emit strided scf.for, store to smem, statically
     unroll tree, emit tx==0 guard with atomic, erase original atomic_rmw.
   - Register with `PatternBenefit(2)` above `ReduceToSCFForGPU`.
   - Build and run `check-quick` (AC-8).

3. **Lit tests**: Add the 4 new lit test files (AC-1, AC-2).
   - `reduction-hierarchical-gpu.mlir` — IR-level FileCheck.
   - `reduction-hierarchical-ptx.mlir` — PTX-level FileCheck.
   - `reduction-hierarchical-fallback.mlir` — legality/fallback remark check.
   - `reduction-hierarchical-f32-kinds.mlir` — kind enforcement.
   - Milestone 3 depends on Milestone 2 (pattern must exist before PTX tests run).

4. **Harness and robustness**: Extend the Python harness and shell scripts.
   - Add `--hierarchical` flag to `harness/run_reduction.py` covering all
     required correctness cases (AC-6).
   - Add `reduction_hierarchical` column to `scripts/run-robustness-validation.sh`
     and row to `scripts/run-differential.sh` (AC-7, AC-9).
   - Milestone 4 depends on Milestone 2 (PTX must be generated first).

5. **Full validation**: Run `ninja check-full` and confirm AC-9.
   - Depends on all prior milestones complete.

---

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology
  such as "AC-", "Milestone", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code (e.g., `stride`,
  `blockDim`, `partialSum`, `scratchBuf`).
- The debug-mode fallback remark should be a plain diagnostic string describing
  the legality condition that failed, not a reference to plan item numbers.

--- Original Design Draft Start ---

# Hierarchical Reduction Lowering — Design Draft (v3)

## Problem Statement

The GPU reduction benchmark on B200 (sm_100) shows a 0.11× slowdown vs CPU
serial:

| N         | CPU serial | GPU atomic | GPU speedup |
|-----------|-----------|------------|-------------|
| 1,048,576 | 0.18 ms   | 1.58 ms    | **0.11×**   |
| 16,777,216| 2.8 ms    | 25.0 ms    | **0.11×**   |

Root cause: every thread atomically adds its partial result directly to a global
scalar accumulator (O(N) serialized atomic contention on one address).

### Current IR state (before this work)

The existing benchmark kernel in `test/SPMD/lower-to-gpu-nvptx-reduction.mlir`
does **not** use `spmd.reduce`. It uses `spmd.forall` directly, with the
accumulation loop body ending in an `atomic_rmw`:

```mlir
// Current source kernel (simplified)
spmd.forall (%tid) in (%cN) {
  %v   = memref.load %in[%tid] : memref<?xf32>
  memref.atomic_rmw addf %v, %out[] : (f32, memref<f32>) -> f32
}
```

`ReduceToSCFForGPU` is never triggered by this kernel. The bottleneck is
architectural: the forall body places one atomic per thread.

---

## Approach (Path A — chosen)

**Step 0 (source restructuring)**: Rewrite the reduction kernel source to use
`spmd.reduce` as the proper IR-level abstraction for a parallel reduction:

```mlir
// New source kernel
%sum = spmd.reduce (0) to (%cN) step (%c1) init (%zero : f32)
           { spmd.kind = #spmd.reduction_kind<add> } : f32 {
  ^bb0(%i: index):
    %v = memref.load %in[%i] : memref<?xf32>
    spmd.yield %v : f32
}
memref.atomic_rmw addf %sum, %out[] : (f32, memref<f32>) -> f32
```

`ReduceToSCFForGPU` (the existing fallback) then lowers this to the current
O(N) atomic pattern — which becomes the **atomic-only baseline** test.

`ReduceToHierarchicalGPU` (new, higher-priority) lowers the same op to the
efficient hierarchical path.

The original `spmd.forall`-based kernel is kept as the **atomic-only regression
test** (`lower-to-gpu-nvptx-reduction.mlir`), unchanged, confirming the
fallback path is not broken.

**Why Path A over keeping the forall kernel unchanged:**
- `spmd.reduce` is the right semantic IR for a parallel reduction. The
  forall+atomic idiom is a manual expansion that bypasses the abstraction.
- `ReduceToHierarchicalGPU` must match on `spmd.reduce`; it has no way to
  pattern-match inside an arbitrary forall body.
- A single source kernel tests both paths: hierarchical pattern (new) and
  fallback via `ReduceToSCFForGPU`.

---

## V1 Scope

**V1 target**: the specific idiom where `spmd.reduce` feeds directly into a
global scalar accumulator:

```mlir
%sum = spmd.reduce (0) to (%cT) step (%c1) init (%zero : f32)
           { spmd.kind = #spmd.reduction_kind<add> } : f32 { ... }
memref.atomic_rmw addf %sum, %out[] : (f32, memref<f32>) -> f32
```

**Deliberately out of scope for V1**:
- General `spmd.reduce` whose result is used for anything other than a global
  scalar accumulation
- `spmd.kind` other than `Add` (Min/Max/And/Or/Xor/Mul deferred to V2)
- `i32`/`i64` element types (deferred; `f32` first)
- Warp-level `shfl.down` shuffles
- Multi-output / non-1D reduce
- Auto-selection of block size

---

## Design Goals

1. **Fix the bottleneck**: N global atomics → `ceil(N / blockDim)` global
   atomics by doing intra-block tree reduction in shared memory first.
2. **Atomic fallback preserved**: `ReduceToSCFForGPU` stays unchanged and
   becomes the fallback for any case that does not match V1 legality.
3. **No dialect surface changes**: `spmd.reduce` op definition is unchanged.
4. **Unified workgroup memory representation**: use the same `gpu.launch
   workgroup(...)` attribution style already established by `PromoteGroupMemory`,
   not a separate allocation style.
5. **Correctness**: tolerance-based (`rel_err < 1e-3`) for floating-point due
   to reassociation.

---

## Legality Conditions (hierarchical pattern fires iff ALL hold)

| # | Condition | Rationale |
|---|-----------|-----------|
| L1 | `spmd.reduce` has exactly one result of type `f32` | V1 only; `i32`/`i64` added in V2 |
| L2 | `spmd.kind = Add` | Only associative+commutative combiner in V1 |
| L3 | reduce body is pure: only `arith.*`, `math.*`, `memref.load` | side-effecting ops unsafe to reorder |
| L4 | `spmd.reduce` is in a `gpu.launch` body (after group forall lowering) | workgroup memory only valid in kernel scope |
| L5 | `blockDim` is a compile-time constant integer | required for static scratch buffer and static tree unrolling |
| L6 | `spmd.reduce` result is used only by a single `memref.atomic_rmw addf` to a rank-0 memref | matches the reduction-to-global-accumulator idiom |

If any condition fails → fall back to `ReduceToSCFForGPU`. The fallback emits
a remark only when the pass is invoked with `--mlir-print-diagnostics` (debug
mode), to avoid noise in normal compilation.

---

## Lowering: Before and After

### Before (inside `gpu.launch` body, after group forall lowering)

```
%sum = spmd.reduce(0) to (%cT) step(%c1) init(%zero : f32) {add} : f32
memref.atomic_rmw addf %sum, %out[] : (f32, memref<f32>) -> f32
```

### After (hierarchical path, blockDim = 256 as example)

```
// Step 1: thread-strided local accumulation
//   Thread tx owns elements: tx, tx + blockDim, tx + 2·blockDim, …
//   Threads with no elements contribute %zero (identity).
%partial = scf.for %i = %tx to %tripCount step %blockDim
           iter_args(%acc = %zero) : f32 {
  %v  = <execute reduce body at index %i>
  %a2 = arith.addf %acc, %v : f32
  scf.yield %a2
}

// Step 2: scatter partial into workgroup scratch
//   Scratch buffer declared as gpu.launch workgroup attribution,
//   same style as PromoteGroupMemory's tile buffer.
%smem = <workgroup scratch memref<256 x f32, #gpu.address_space<workgroup>>>
memref.store %partial, %smem[%tx]
gpu.barrier

// Step 3: in-block tree reduction — STATICALLY UNROLLED
//   blockDim is a compile-time constant, so log2(blockDim) steps are
//   unrolled at codegen time. scf.for cannot express a halving stride,
//   so each step is emitted as a separate scf.if + gpu.barrier pair.
//
//   (unrolled for blockDim=256: strides 128, 64, 32, 16, 8, 4, 2, 1)

scf.if (%tx < 128) {
  %a = memref.load %smem[%tx]       : memref<256 x f32, workgroup>
  %b = memref.load %smem[%tx + 128] : memref<256 x f32, workgroup>
  memref.store arith.addf(%a, %b), %smem[%tx]
}
gpu.barrier

scf.if (%tx < 64) {
  %a = memref.load %smem[%tx]      : memref<256 x f32, workgroup>
  %b = memref.load %smem[%tx + 64] : memref<256 x f32, workgroup>
  memref.store arith.addf(%a, %b), %smem[%tx]
}
gpu.barrier

// … (strides 32, 16, 8, 4, 2 follow the same pattern) …

scf.if (%tx < 1) {
  %a = memref.load %smem[0] : memref<256 x f32, workgroup>
  %b = memref.load %smem[1] : memref<256 x f32, workgroup>
  memref.store arith.addf(%a, %b), %smem[0]
}
gpu.barrier

// Step 4: thread 0 flushes block result to global accumulator
scf.if (%tx == 0) {
  %block_sum = memref.load %smem[0] : memref<256 x f32, workgroup>
  memref.atomic_rmw addf %block_sum, %out[] : (f32, memref<f32>) -> f32
}
// The original atomic_rmw that used %sum is eliminated.
```

**Net reduction in global atomics**: 1,048,576 → 4,096 for N=1M, blockDim=256.

**Why static unrolling, not `scf.for`**: the tree reduction uses a halving
stride (128 → 64 → 32 → …), which cannot be expressed as a `scf.for` with a
fixed step. A loop over log2 would require dynamic stride computation and an
inner `arith.shli` — adding complexity without benefit when blockDim is a
compile-time constant. Static unrolling produces log2(blockDim) scf.if +
gpu.barrier pairs (8 pairs for blockDim=256), which is a fixed, bounded code
size and maps directly to what a hand-written CUDA kernel would emit.

---

## Thread Assignment

Step 1 uses a **thread-strided slice**:

```
thread tx owns elements: tx, tx + blockDim, tx + 2·blockDim, …
```

This is the standard strided-slice pattern: ensures even distribution, works
for any N (threads with no elements contribute the identity value `%zero`), and
maximizes memory coalescing across the warp.

The trip count for the local `scf.for` is `ceil(tripCount / blockDim)` and is
computed at codegen time from known constants.

---

## Workgroup Memory Representation

The scratch buffer follows the same convention as `PromoteGroupMemory`:

```mlir
gpu.launch ... workgroup(%red_smem : memref<BLOCK_DIM x f32,
                                              #gpu.address_space<workgroup>>) {
  ...
}
```

No new allocation style is introduced. The outlining and NVVM lowering paths
already handle workgroup attributions correctly.

---

## Implementation

### Step 0: source kernel restructuring

Change `test/SPMD/lower-to-gpu-nvptx-reduction.mlir` (the main reduction test)
to use `spmd.reduce` as the input IR. The original `spmd.forall`-based kernel
in the same file becomes (or is split into) the **atomic-only fallback
regression test**, so both lowering paths remain tested.

### Step 1: compiler pattern

Add `struct ReduceToHierarchicalGPU` in
`lib/Conversion/SPMDToGPU/SPMDToGPU.cpp`, immediately above `ReduceToSCFForGPU`.

### Pattern priority

```cpp
patterns.add<IfToSCFIfGPU>(ctx);
patterns.add<ReduceToHierarchicalGPU>(ctx);  // benefit=2  NEW
patterns.add<ReduceToSCFForGPU>(ctx);        // benefit=1  existing fallback
```

`ReduceToHierarchicalGPU::matchAndRewrite` returns `failure()` if any legality
condition (L1–L6) is not met; MLIR greedy rewriting then falls through to
`ReduceToSCFForGPU`.

### Approximate code size

| Component | Lines |
|-----------|-------|
| `ReduceToHierarchicalGPU` pattern | ~150 |
| Legality helper (`isHierarchicalReduceCandidate`) | ~35 |
| Static tree unrolling loop (C++ for loop at codegen time) | ~20 |
| Fallback remark (debug-mode only) | ~5 |
| **Total compiler changes** | **~210** |

---

## Tests

### Source kernel change

`test/SPMD/lower-to-gpu-nvptx-reduction.mlir` is updated to use `spmd.reduce`
as input. A separate file (or RUN line) retains the forall+atomic form as the
**atomic-only baseline** that exercises `ReduceToSCFForGPU`.

### New lit tests

| File | What it checks |
|------|---------------|
| `reduction-hierarchical-gpu.mlir` | workgroup alloc present, log2(blockDim) barriers, tx==0 guard, exactly 1 atomic |
| `reduction-hierarchical-ptx.mlir` | PTX: `.shared`, `bar.sync`, `atom.add.f32` (1 per block, not N) |
| `reduction-hierarchical-fallback.mlir` | body with `func.call` → remark + scf.for fallback |
| `reduction-hierarchical-f32-kinds.mlir` | Add fires; Min/Max → fallback (V1 scope enforcement) |

### Correctness harness additions (`harness/run_reduction.py`)

Add a `--hierarchical` flag that loads the hierarchical PTX. Cover:

| Case | Sizes |
|------|-------|
| Basic correctness | 1, 32, 33, 255, 256, 257, 1024, 65536, 1M, 16M |
| Non-multiple-of-blockDim | N=1000, N=65537 |
| All-zeros input | result == 0.0 |
| All-ones input  | result ≈ N (fp tolerance) |
| Multi-launch + re-zero | 3 launches, accumulator zeroed between each |

Correctness criterion: `rel_err < 1e-3` vs numpy reference (floating-point
reassociation is expected and acceptable).

### Robustness sweep (`run-robustness-validation.sh`)

Add `reduction_hierarchical` as a fourth kernel column alongside the existing
three. Expected:

| N        | atomic GPU | hierarchical GPU |
|----------|-----------|-----------------|
| 1M       | 1.58 ms   | ≪ 0.5 ms (est.) |
| 16M      | 25.0 ms   | ≪ 5.0 ms (est.) |

### Differential (`run-differential.sh`)

Add a `reduction_hierarchical` row: cpu_ok / omp_ok / gpu_ok all PASS.

---

## Acceptance Criteria

| # | Criterion | Verified by |
|---|-----------|-------------|
| H-1 | Hierarchical pattern fires for f32 Add reduction-to-scalar | lit: workgroup alloc present |
| H-2 | Atomic-only fallback still works (forall+atomic baseline unchanged) | existing lit test |
| H-3 | Legality: non-pure body → fallback + remark | new lit test |
| H-4 | Legality: non-Add kind → fallback | new lit test |
| H-5 | Correctness: all sizes, incl. non-multiples, multi-launch | harness |
| H-6 | GPU speedup > 1× vs CPU serial for N ≥ 64K (was 0.11×) | robustness CSV |
| H-7 | 29 existing lit tests unchanged | check-quick |
| H-8 | check-full passes end-to-end | check-full |

--- Original Design Draft End ---
