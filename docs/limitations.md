# SPMD Dialect Known Limitations

This document enumerates cases that are unsupported or conservatively skipped
by the current SPMD compiler. Each entry maps to a behavior category:

- **verifier error**: The IR is structurally invalid; a verifier pass emits
  `emitError` and the pipeline fails.
- **pass fail**: A pass precondition is violated; the pass emits `emitError` +
  `signalPassFailure`.
- **remark + skip**: The optimization is not applicable, but the program is
  legal; the pass emits `emitRemark` and leaves the IR unchanged.

---

## 1. Non-Structural (Unstructured) Control Flow

**Description:** The SPMD dialect requires all control flow inside `spmd.kernel`
functions to be expressed via `spmd.forall`, `spmd.if`, `spmd.reduce`, and
`spmd.barrier`. Arbitrary `cf.br`, `cf.cond_br`, or `scf.while` ops inside
kernels are not supported.

**Behavior:** verifier error — `VerifySPMDKernelSubset` rejects kernels
containing disallowed dialect ops (including `cf.*` and `scf.while`).

**Workaround:** Express all loops as `spmd.forall` or `spmd.reduce`; express
conditionals as `spmd.if`.

---

## 2. Pointer Chasing / Indirect Memory Access

**Description:** Access patterns of the form `A[B[i]]` (indirect indexing
through a second memref) cannot be analyzed by `PromotionPlanAnalysis`.
The index `B[i]` is a loaded value, not a compile-time affine expression.

**Behavior:** remark + skip — `PromoteGroupMemory` conservatively skips
promotion for any memref whose access index cannot be decomposed as
`outer_iv + inner_iv * step + const`. No error; the original global-memory
accesses remain intact.

**Workaround:** Use direct (affine) indexing. Indirect accesses can still
be executed correctly; they will not benefit from group memory promotion.

---

## 3. Write-Back Promotion (Read-Write Memrefs)

**Description:** `PromoteGroupMemory` only promotes memrefs that are read-only
inside the lane-level `spmd.forall`. If the same memref is both loaded from
and stored to (an in-place update pattern), promotion is skipped.

**Behavior:** remark + skip — The analysis layer detects the store and excludes
the memref from the promotion plan. No error; the pass proceeds without
transforming that memref.

**Workaround:** Use separate input (read-only) and output memrefs. Write-back
promotion (read → modify → write back to group memory, then flush to global)
is explicitly out of scope for this version.

---

## 4. Barrier in Divergent Control Flow (Divergent Path)

**Description:** A `spmd.barrier` must be reached by ALL threads in the group
(convergent execution). Placing a barrier inside a `spmd.if` or any conditional
branch creates divergent paths where some threads may skip the barrier.

**Behavior:** pass fail — `SPMDToGPU` checks that every `spmd.barrier` is at
the top level of the group forall body (i.e., its parent op is `gpu.launch`
after body movement). A barrier inside a `spmd.if` triggers:
`emitError("gpu.barrier must be at gpu.launch body level; found nested inside spmd.if")` +
`signalPassFailure`.

**Workaround:** Place barriers only at the group body level, between forall
operations. The `PromoteGroupMemory` pass always inserts barriers at the
correct level (after the cooperative copy forall, before the compute forall).

---

## 5. BlockDim Hardware Limit Overflow

**Description:** CUDA hardware requires the total number of threads per block
to be ≤ 1024. If the product of lane trip counts across all dimensions exceeds
this limit, the GPU kernel cannot be launched.

**Behavior:** pass fail — `SPMDToGPU` computes `blockDimTotal = product of lane
trip counts` and checks against 1024. If exceeded:
`emitError("computed blockDim N exceeds CUDA maximum of 1024")` + `signalPassFailure`.

**Workaround:** Reduce tile sizes so that the product fits within 1024. Common
safe configurations: 256×1, 128×1, 32×8, 16×16.

---

## 6. Non-Affine / High-Dynamic Indexing in Promotion

**Description:** `PromoteGroupMemory` requires that every access index into the
candidate memref be decomposable as `outer_iv + inner_iv * step + const`.
Quadratic indices (`i*i`), runtime-multiplied indices (`i * N` where N is a
function argument), or indices produced by arbitrary arithmetic chains are
treated as unbounded.

**Behavior:** remark + skip — The access summary analysis marks the dimension
as unbounded (INT64_MAX offset). The footprint becomes infinite (overflow
check), and the promotion plan entry is rejected. No error.

**Workaround:** Rewrite the loop to use a canonical affine index pattern.
Alternatively, compile without `--promote-group-memory` (use the no-promotion
path).

---

## 7. Group Memory Footprint Overflow

**Description:** The tile buffer footprint is `product(tileDims) * elemBytes`.
If this exceeds `maxGroupMemBytes` (default: 48 KB = 49,152 bytes), promoting
the memref would exceed the per-block shared memory budget.

**Behavior:** remark + skip — `PromotionPlanAnalysis` computes the footprint
and emits:
`emitRemark("promote-group-memory: skipping <memref> — tile footprint N B exceeds maxGroupMemBytes (49152 B)")`.
The memref is not promoted; global-memory accesses remain intact.

**Workaround:** Reduce tile sizes or use smaller element types. For 2D tiles,
keeping both dimensions ≤ 110 elements for f32 stays within the 48 KB budget.

---

## 8. Reduction Body with Unknown Side Effects

**Description:** `spmd.reduce` in the current pipeline lowers to a sequential
`scf.for` on CPU (SPMDToSCF/OpenMP) or to a sequential `scf.for` per thread
on GPU (SPMDToGPU). This means each thread runs a full sequential reduction,
not a parallel tree reduction. Additionally, the reduction body must not contain
calls to external functions with unknown side effects (e.g., `printf`, I/O).

**Behavior:** pass fail — If the reduction body contains ops from disallowed
dialects (e.g., `llvm.call` with unknown callees), `VerifySPMDKernelSubset`
rejects the kernel before any lowering pass runs.

**Workaround:** Keep reduction bodies pure (only `arith.*`, `math.*`). For
efficient GPU reductions, use `spmd.reduce` with `spmd.kind = add` and an
`f32` element type: `ReduceToHierarchicalGPU` will automatically lower this
to a two-level shared-memory tree reduction followed by a single per-block
global atomic, achieving >1× speedup over CPU serial for N ≥ 64K. The
`memref.atomic_rmw`-in-forall idiom remains valid as an explicit atomic-only
baseline, but is no longer the recommended approach for GPU reductions.

---

## Summary Table

| # | Limitation | Behavior |
|---|-----------|---------|
| 1 | Unstructured CFG (`cf.br`, `scf.while`) | verifier error |
| 2 | Pointer chasing / indirect index | remark + skip |
| 3 | Write-back promotion (read-write memref) | remark + skip |
| 4 | Barrier in divergent path (`spmd.if` body) | pass fail |
| 5 | BlockDim > 1024 | pass fail |
| 6 | Non-affine / high-dynamic index | remark + skip |
| 7 | Group memory footprint overflow | remark + skip |
| 8 | Reduction body with unknown side effects | verifier error |
