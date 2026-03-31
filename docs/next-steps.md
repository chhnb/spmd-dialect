# Next Steps: Optimization Roadmap

This document tracks planned GPU optimizations beyond the current v1 implementation.
Each item includes what it does, why it matters, and what IR-level changes are needed.

---

## Already Implemented

| Pass | Status |
|------|--------|
| `PromoteGroupMemory` (C2) | Done — bank conflict padding TBD |
| `ReduceToHierarchicalGPU` (C3) | Done — f32 add, single result |
| `RecognizeStructuredReductions` | **Not yet implemented** — needed for Taichi frontend |

---

## Immediate Priority

### RecognizeStructuredReductions

**What:** Promote `memref.atomic_rmw` inside `spmd.forall` to `spmd.reduce` when the destination is loop-invariant.

**Why:** Taichi (and Warp/Numba) emit `AtomicOpStmt` for all reductions. Without this pass, `ReduceToHierarchicalGPU` never fires on frontend-generated kernels.

**Conditions for promotion:**
1. `atomic_rmw` is inside a `spmd.forall` body
2. Destination memref is loop-invariant w.r.t. all forall block arguments
3. Op is a known associative + commutative combiner: `addf`, `addi`, `minf`, `maxf`, `andi`, `ori`
4. No other writes to the same destination in the forall body
5. Destination is not read inside the forall body (only after)

**IR change:** `atomic_rmw` → `spmd.reduce` + one `atomic_rmw` (per-block flush)

**Where to add:** New pass `RecognizeStructuredReductions`, run before `PlanSPMDSchedule`.

---

## Phase 2 Optimizations

### 1. Warp-Level Reduction (C3 refinement)

**What:** After the shared-memory tree reduction reaches ≤ 32 threads (one warp), replace remaining tree steps with `gpu.shuffle` (warp shuffle instructions `__shfl_down_sync`).

**Why:** Each `gpu.barrier` + shared memory step has overhead. Warp-level operations are implicit synchronization — no barrier needed within a warp.

```
Current:  blockDim=256 → 8 barrier steps (shared memory tree all the way)
Improved: 3 barrier steps (tree down to 32) + 5 warp shuffle steps (no barrier)
```

**IR change:** In `ReduceToHierarchicalGPU`, after the tree reduction loop reaches stride=32, emit `gpu.shuffle` ops instead of `memref.store/load` + `gpu.barrier`.

**Prerequisite:** `gpu.shuffle` support in the current MLIR GPU dialect.

---

### 2. PromoteGroupMemory Bank Conflict Padding

**What:** Add `+1` padding per dimension in the tile buffer allocated by `PromoteGroupMemory`.

**Why:** Without padding, threads in the same warp may access the same shared memory bank (e.g., for a 2D stencil with 32-wide tiles, column-access stride = 32 elements = 32 banks → all threads hit bank 0). This serializes shared memory accesses.

**Fix:** Change tile buffer size from:
```
dim_size = (tileSize - 1) * step + maxOffset - minOffset + 1
```
to:
```
dim_size = (tileSize - 1) * step + maxOffset - minOffset + 1 + padding
```
where `padding = 1` for the innermost dimension.

**Where:** `PromotionPlanAnalysis.cpp`, `tileDim` computation.

**Impact:** Must benchmark before and after — bank conflicts may be why stencil speedup is lower than expected.

---

### 3. Vectorized Loads (float4)

**What:** Replace scalar `memref.load %x[%i]` with `vector.load %x[%i] : vector<4xf32>` where access is coalesced and alignment allows.

**Why:** GPU memory controllers are optimized for 128-bit (16-byte) transactions. Scalar f32 loads use only 32 bits per transaction. `float4` loads achieve 4x better bandwidth utilization.

**Conditions:**
- Access stride = 1 (consecutive elements per thread)
- Tile start is 16-byte aligned (or padding ensures alignment)
- Trip count is a multiple of 4

**IR change:** New pass `VectorizeCoalescedAccess`, runs after `MaterializeTilingAndMapping`. Rewrites `memref.load/store` on unit-stride accesses to `vector.load/store`.

---

### 4. Double Buffering (Stencil Latency Hiding)

**What:** Allocate two group-memory tile buffers (A and B). While computing with buffer A, asynchronously prefetch the next tile into buffer B.

**Why:** Hides global memory latency behind compute. On A100/H100 with `cp.async`, this can fully hide memory latency for stencil kernels.

```
buf_A ← sync load tile_0
for tile k in [1, last]:
    async prefetch tile_{k} → buf_B    # overlaps with compute below
    compute tile_{k-1} using buf_A
    await buf_B
    swap(buf_A, buf_B)
compute tile_last using buf_A
```

**Cost:** Shared memory usage doubles.

**IR change:** `PromoteGroupMemory` variant with `double_buffer` mode. Adds `gpu.wait` / `gpu.async.token` ops for the async copy path.

**Prerequisite:** Target must support async copies (`cp.async` on sm_80+).

---

## Phase 3 Optimizations (Lower Priority)

### 5. Redundant Barrier Elimination

**What:** Remove `spmd.barrier` ops that `PromoteGroupMemory` inserted conservatively, when analysis proves no WAR / RAW hazard exists across the barrier.

**Why:** Each `gpu.barrier` has non-trivial overhead (thread convergence + memory fence). Removing even one barrier per tile iteration compounds over the full kernel.

**Condition for removal:** The writes before the barrier and reads after the barrier provably touch disjoint address ranges in group memory.

**Risk:** Correctness analysis is subtle. Only attempt after the rest of the pipeline is stable.

---

### 6. Occupancy-Aware Tile Sizing

**What:** Replace static `tile_sizes` defaults with a target-aware pass that selects tile sizes to maximize occupancy.

**Inputs:**
- GPU target descriptor: `maxSharedMemoryPerBlock`, `maxThreadsPerBlock`, `regsPerThread`
- Access footprint estimate from `PromotionPlanAnalysis`
- Estimated register pressure

**Output:** Updated `spmd.tile_sizes` attrs on `spmd.forall` ops.

**Why:** A 64×64 tile may give better occupancy than 32×32 even if both fit, because the larger tile amortizes kernel launch overhead. Conversely, a tile that's too large reduces occupancy by consuming too much shared memory per block.

---

## Summary Table

| Optimization | Phase | Impact | Complexity |
|-------------|-------|--------|------------|
| `RecognizeStructuredReductions` | **Now** | Enables C3 for all frontends | Low |
| Bank conflict padding | **Now** | Fix stencil speedup regression | Low |
| Warp-level reduction | 2 | -30-50% barrier overhead for reduction | Medium |
| Vectorized loads | 2 | 2-4x memory bandwidth | Medium |
| Double buffering | 2 | Hide memory latency for stencil | High |
| Barrier elimination | 3 | Incremental | High (correctness risk) |
| Occupancy-aware tiling | 3 | Incremental | Medium |
