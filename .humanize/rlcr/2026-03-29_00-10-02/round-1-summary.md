# Round 1 Summary — convert-spmd-to-gpu Pass: Correctness Fixes + Test Expansion

## Work Done

Addressed all four Codex HIGH findings from Round 0 review and expanded test coverage per AC-3, AC-4, AC-5, AC-6 requirements.

### Files Changed

| File | Change |
|------|--------|
| `lib/Conversion/SPMDToGPU/SPMDToGPU.cpp` | Four correctness fixes (see below) |
| `test/lit.cfg.py` | Added NVPTX feature detection via `llc --version` |
| `test/SPMD/lower-to-gpu.mlir` | Added RUN 3 (spmd.if + spmd.reduce — AC-3) |
| `test/SPMD/lower-to-gpu-negative.mlir` | New file: negative tests via `--verify-diagnostics` |
| `test/SPMD/lower-to-gpu-nvptx.mlir` | New file: RUN 3/4 NVPTX codegen (gated by `REQUIRES: nvptx-registered-target`) |

### Correctness Fixes in SPMDToGPU.cpp

**Fix 1 — Group IV: always add lb (was: skip if constant 0)**

Previously the pass omitted `addi blockIdx[d], lb[d]` when lb was a constant 0.
This is wrong for dynamic lb or when MLIR constant-folds constants away. Now always emits:
```
gi = blockIdx[d] * step[d] + lb[d]   // unconditional AddI
```

**Fix 2 — Rank > 3 guard**

Added an explicit check before any other processing:
```cpp
if (groupForall.getRank() > 3) {
  groupForall->emitError("group forall rank > 3 is not supported for GPU lowering");
  hadError = true; continue;
}
```

**Fix 3 — `computeMaxLinearBlockDim` uses trip counts not raw UBs**

Old code: `prod *= ub_value` (raw upper bound).
New code: `prod *= ceildiv(ub - lb, step)` per dim (trip count).
This matters whenever lb ≠ 0 or step ≠ 1 (e.g., a lane forall `[1, 33, 1]` has 32 trips, not 33).

**Fix 4 — Lane IV reconstruction: `lb + idx * step` + trip-count guard**

Old code:
- Delinearized using raw UBs as strides (wrong when lb ≠ 0 or step ≠ 1)
- Used raw UB product as guard (wrong for non-trivial bounds)
- For dynamic UBs with rank > 1: silently mapped all IVs to `tx` (generates wrong IR)
- IV not adjusted for lb/step

New code:
- `delinearizeTx` uses trip counts `ceildiv(ub-lb, step)` as row-major strides
- Returns 0-based indices; caller reconstructs `iv[d] = lb[d] + idx[d] * step[d]`
- Guard = trip count product (constant) or runtime `ceildivui(ub-lb, step)` (1D dynamic)
- For multi-dim dynamic bounds: emits error instead of generating wrong IR

**Fix 5 — Top-level group forall collection**

Previously `module.walk` collected ALL group foralls (including nested ones).
In post-order walk, the inner forall was added before the outer, so it got processed
independently first — the outer's nested-group validation never fired.
Fix: skip group foralls whose parent is another ForallOp:
```cpp
if (op->getParentOfType<ForallOp>()) return; // skip nested ones
```

### Test Expansion

**RUN 3 in `lower-to-gpu.mlir`** (AC-3):
Tests `spmd.if` → `scf.if` and `spmd.reduce` → `scf.for` lowering via the greedy
pattern rewrite inside `--convert-spmd-to-gpu` (no pre-pass needed).
Checks: `scf.if` present, `scf.for` present, `spmd.if`/`spmd.reduce` absent.

**`lower-to-gpu-negative.mlir`** (negative tests):
- Test 1: nested group forall → `expected-error` "nested group-level spmd.forall is not supported"
- Test 2: rank-4 group forall → `expected-error` "group forall rank > 3 is not supported"
Uses `-verify-diagnostics -split-input-file --convert-spmd-to-gpu`.

**`lower-to-gpu-nvptx.mlir`** (AC-5, AC-6):
- RUN 3: full pipeline (normalize→materialize→convert→outline→nvvm→translate→llc obj) exits 0
- RUN 4: PTX text for promoted stencil contains `.shared` and `.visible .entry`
Both gated with `REQUIRES: nvptx-registered-target` (NVPTX not in local llc build).
NVPTX feature detection added to `test/lit.cfg.py` via `llc --version | grep NVPTX`.

### Test Results

```
17 tests: 16 PASS, 1 UNSUPPORTED (lower-to-gpu-nvptx.mlir — no NVPTX backend)
```

All pre-existing 15 tests continue to pass.

## Goal Tracker Update Request

### Mark Completed Tasks

| AC | Task | Completed Round |
|----|------|-----------------|
| AC-1 | Add declarations to SPMDPasses.h | 0 |
| AC-1 | Implement SPMDToGPU.cpp | 0 |
| AC-1 | Add to SPMDPassRegistration.cpp | 0 |
| AC-1 | Update CMakeLists (pass + tool) | 0 |
| AC-2,3 | Group forall → gpu.launch + lane → thread args + if/reduce lowering | 0 |
| AC-4 | Workgroup buffer rebinding | 0 |
| AC-7 | All pre-existing lit tests pass | 0 |
| AC-2,4 | Write test/SPMD/lower-to-gpu.mlir (RUN 1 + RUN 2) | 0 |
| AC-2 | Fix group IV lb always added (correctness) | 1 |
| AC-2 | Fix rank > 3 guard (correctness) | 1 |
| AC-2 | Fix trip-count computation in blockDim (correctness) | 1 |
| AC-2 | Fix lane IV reconstruction lb+idx*step (correctness) | 1 |
| AC-3 | Test spmd.if + spmd.reduce via GPU pass (RUN 3 in lower-to-gpu.mlir) | 1 |
| AC-3 | Negative tests (lower-to-gpu-negative.mlir) | 1 |
| AC-5,6 | Write lower-to-gpu-nvptx.mlir (RUN 3 + RUN 4 gated by nvptx feature) | 1 |
| AC-5,6 | Add NVPTX feature detection to test/lit.cfg.py | 1 |

### Remaining / Deferred

- AC-5, AC-6 (RUN 3, RUN 4 in lower-to-gpu-nvptx.mlir): Tests written and gated by
  `REQUIRES: nvptx-registered-target`. Will execute automatically when NVPTX backend
  is present. Currently UNSUPPORTED (no NVPTX in this llc build).
