# SPMD Dialect Pass Contracts

This document records the input requirements (preconditions), output guarantees
(postconditions), and failure behavior for each key pass in the SPMD lowering
pipeline.

---

## NormalizeSPMD

**Pass flag:** `--normalize-spmd`

### Input Requirements
- Module contains `func.func` ops marked with `{spmd.kernel}`.
- Functions contain `spmd.forall` ops in S0 form: no `spmd.mapping` attribute,
  no `spmd.tile_sizes` attribute, no `spmd.memory_policy` attribute.
- Forall induction-variable ranges may be dynamic or constant.

### Output Guarantees
- Each `spmd.forall` body is a straight-line region with no nested foralls
  (the pass flattens or canonicalizes nested S0 foralls).
- The IR is valid for `PlanSPMDSchedule` to consume.

### Failure Behavior
- Non-recognizable IR structure within a `spmd.kernel` function:
  `emitError` + `signalPassFailure`.
- All other inputs: pass completes silently (no remark, no skip).

---

## PlanSPMDSchedule

**Pass flag:** `--plan-spmd-schedule`

### Input Requirements
- Module contains `func.func` ops marked with `{spmd.kernel}`.
- Each kernel's outermost `spmd.forall` has no `spmd.mapping` attribute
  (S0 form, unnormalized or post-normalize).
- No prior tile_sizes or memory_policy attributes.

### Output Guarantees
- Each group-level `spmd.forall` receives `spmd.mapping = #spmd.level<group>`,
  `spmd.tile_sizes`, and `spmd.memory_policy` attributes.
- Each lane-level `spmd.forall` receives `spmd.mapping = #spmd.level<lane>`.
- Tile sizes are chosen by a heuristic (MVP: fixed 32×8 for 2D, 256 for 1D).
- The IR is valid S1 and ready for `MaterializeTilingAndMapping`.

### Failure Behavior
- Unknown forall structure (e.g., forall with no recognized iteration pattern):
  `emitError` + `signalPassFailure`.
- Forall with rank > 3: `emitError` + `signalPassFailure` (GPU grid is 3D max).

---

## MaterializeTilingAndMapping

**Pass flag:** `--materialize-spmd-tiling`

### Input Requirements
- Module contains `func.func` ops marked with `{spmd.kernel}`.
- Each outermost `spmd.forall` has `spmd.mapping`, `spmd.tile_sizes`, and
  `spmd.memory_policy` attributes (S1 form, post-`PlanSPMDSchedule`).

### Output Guarantees
- Each group-level forall is split into an outer group-tile forall and an inner
  lane forall.
- The outer forall steps by `tile_size[d]` per dimension.
- The inner forall iterates `[0, tile_size[d])` per dimension with step 1.
- The resulting IR is valid S2 and ready for `PromoteGroupMemory` and conversion
  passes.

### Failure Behavior
- Forall with `spmd.mapping` attribute but no `spmd.tile_sizes`:
  `emitError` + `signalPassFailure`.
- Non-constant tile sizes (tile sizes must be compile-time constants):
  `emitError` + `signalPassFailure`.

---

## PromoteGroupMemory

**Pass flag:** `--promote-group-memory`

### Input Requirements
- Module contains `func.func` ops marked with `{spmd.kernel}` in S2 form
  (post-`MaterializeTilingAndMapping`).
- Group-level `spmd.forall` ops have `spmd.mapping = #spmd.level<group>`,
  `spmd.tile_sizes`, and `spmd.memory_policy` attributes.
- Inner lane-level `spmd.forall` ops are directly nested inside group foralls.
- All memref accesses are `memref.load` or `memref.store` (no pointer arithmetic).

### Output Guarantees
- For each group forall with `memory_policy = prefer_group` and a promotable
  memref (read-only, bounded access pattern, footprint ≤ maxGroupMemBytes):
  - A cooperative-copy lane forall (load global → store tile) is inserted before
    the compute forall.
  - A `spmd.barrier {spmd.scope = #spmd.scope<group>}` is inserted after the
    copy forall.
  - The compute lane forall loads from the group-addr-space tile buffer instead
    of the original global memref.
  - A `memref.alloc() : memref<...xf32, #spmd.addr_space<group>>` is inserted
    for the tile buffer.
- Kernels with `memory_policy = no_promotion` are left completely unchanged;
  a diagnostic remark is emitted explaining the skip.
- Kernels where no promotable memref is found (no reuse, non-affine access,
  write conflict, or footprint overflow) are left unchanged.

### Failure Behavior
- `memory_policy = no_promotion`: `emitRemark` + skip (no transformation, no error).
- No inner lane forall found: `emitRemark("promote-group-memory: no lane-level forall found; skipping")` + skip.
- Footprint exceeds `maxGroupMemBytes` (default 48 KB): `emitRemark` with
  footprint size + skip.
- Non-affine access pattern (index not decomposable as outer_iv + inner_iv*step + const):
  plan entry skipped silently (promotion not attempted).
- Write conflict (store to candidate memref inside lane forall):
  plan entry skipped silently (promotion not attempted).
- All other inputs: pass completes successfully.

---

## SPMDToSCF

**Pass flag:** `--convert-spmd-to-scf`

### Input Requirements
- Module contains `func.func` ops in S2 form (post-materialization).
- `spmd.forall` ops have `spmd.mapping` attributes.
- No group-address-space memref allocs (or they are erased before this pass).
- `spmd.if`, `spmd.reduce`, `spmd.barrier`, `spmd.yield` ops are present.

### Output Guarantees
- Each group-level `spmd.forall` → `scf.for` loop (or scf.parallel).
- Each lane-level `spmd.forall` → inner `scf.for` loop.
- `spmd.if` → `scf.if`.
- `spmd.reduce` → `scf.for` with accumulator.
- `spmd.barrier` → removed (no-op on CPU).
- `spmd.yield` → `scf.yield`.
- No `spmd.*` ops remain in the output.

### Failure Behavior
- Unrecognized `spmd.*` op that cannot be lowered: `emitError` + `signalPassFailure`.
- Missing `spmd.mapping` attribute on a forall: `emitError` + `signalPassFailure`.

---

## SPMDToOpenMP

**Pass flag:** `--convert-spmd-to-openmp`

### Input Requirements
- Same as SPMDToSCF: S2 IR with `spmd.mapping` attributes.
- `libomp` must be available at link time for the generated code to run.

### Output Guarantees
- Group-level `spmd.forall` → `omp.parallel` + `omp.wsloop` (OpenMP parallel loop).
- Lane-level `spmd.forall` → inner `scf.for` loop (single-threaded within each
  OpenMP thread's tile).
- `spmd.barrier` → `omp.barrier`.
- `spmd.if`, `spmd.reduce`, `spmd.yield` → equivalent SCF ops.
- No `spmd.*` ops remain in the output.

### Failure Behavior
- Same as SPMDToSCF: unrecognized or malformed `spmd.*` ops trigger
  `emitError` + `signalPassFailure`.

---

## SPMDToGPU

**Pass flag:** `--convert-spmd-to-gpu`

### Input Requirements
- Module contains `func.func` ops in S2 form.
- Group-level `spmd.forall` ops have `spmd.mapping = #spmd.level<group>`,
  `spmd.tile_sizes` attributes.
- Lane-level `spmd.forall` ops have constant bounds (or rank == 1 with dynamic
  bounds for the 1D case).
- Group-address-space `memref.alloc` ops may be present (inserted by
  `PromoteGroupMemory`); they will be converted to `gpu.workgroup` attributions.
- `spmd.barrier` ops must be at the group forall body level (not inside
  `spmd.if` or lane foralls).

### Output Guarantees
- Group-level `spmd.forall` → `gpu.launch` with `gridDim` computed as
  `ceildivui((ub[d] - lb[d]), step[d])` per dimension.
- `blockDim.x` = product of lane trip counts (linearized).
- Group-address-space `memref.alloc` → `gpu.workgroup` attribution in the
  `gpu.launch`; original alloc op is erased.
- Lane-level `spmd.forall` → `scf.if(threadIdx.x < tripProduct)` guard with
  row-major delinearization of `threadIdx.x` for 2D+ cases.
- `spmd.barrier` → `gpu.barrier` (with workgroup memfence if workgroup buffers exist).
- `spmd.if` → `scf.if` (via greedy patterns).
- `spmd.reduce` → `scf.for` (via greedy patterns).
- The enclosing module receives `gpu.container_module` attribute.
- A diagnostic remark is emitted for each lowered group forall with computed
  `blockDim`, workgroup buffer count, and whether 2D lane flattening was applied.

### Failure Behavior
- Group forall rank > 3: `emitError("group forall rank > 3 is not supported for GPU lowering")` + `signalPassFailure`.
- Nested group-level `spmd.forall` (group inside group): `emitError("nested group-level spmd.forall is not supported")` + `signalPassFailure`.
- Computed blockDim > 1024: `emitError("computed blockDim N exceeds CUDA maximum of 1024")` + `signalPassFailure`.
- `spmd.barrier` not at `gpu.launch` body level (e.g., inside `spmd.if` or lane forall): `emitError("gpu.barrier must be at gpu.launch body level; found nested inside ...")` + `signalPassFailure`.
- Dynamic multi-dim lane forall (rank > 1 with non-constant bounds): `emitError("dynamic multi-dim lane forall is not supported for GPU lowering")` + `signalPassFailure`.
