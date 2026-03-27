# Goal Tracker

<!--
This file tracks the ultimate goal, acceptance criteria, and plan evolution.
It prevents goal drift by maintaining a persistent anchor across all rounds.

RULES:
- IMMUTABLE SECTION: Do not modify after initialization
- MUTABLE SECTION: Update each round, but document all changes
- Every task must be in one of: Active, Completed, or Deferred
- Deferred items require explicit justification
-->

## IMMUTABLE SECTION
<!-- Do not modify after initialization -->

### Ultimate Goal

Implement the `spmd` MLIR dialect as specified in `design-v1.md`: a structured, backend-agnostic intermediate representation for regular SPMD kernels. The MVP delivers a compilable and testable dialect with 5 ops, 5 attr classes, a build system with `spmd-opt` driver, a CPU execution path via SCF/OpenMP, and a group-memory promotion demonstration on a 2D stencil kernel.

The work proceeds through four sequential milestones:
- **Build system + tool driver** (prerequisite for all tests)
- **Dialect IR** (5 ops, 5 attrs, verifiers, legality pass, lit tests)
- **CPU closed-loop pipeline** (normalize → materialize → lower → OpenMP → LLVM)
- **Group memory promotion demo** (read-only stencil pattern, S0 → S2 → CPU)

GPU backend (Milestone 5) is defined but outside MVP scope.

### Acceptance Criteria
<!-- Each criterion must be independently verifiable -->
<!-- Claude must extract or define these in Round 0 -->


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
      - `spmd.barrier` nested inside a lane-level forall (no enclosing group forall) emits the same error

- **AC-6**: The lit test suite at `test/SPMD/` runs via `cmake --build build --target check-spmd` with all tests passing.
  - Positive Tests (expected to PASS):
    - `test/SPMD/ops.mlir` — all CHECK patterns match
    - `test/SPMD/invalid.mlir` — all `expected-error` diagnostics are triggered
  - Negative Tests (expected to FAIL):
    - Intentionally broken IR (rank mismatch in forall) causes `spmd-opt --verify-diagnostics` to report failure when expected-error annotation is removed

- **AC-7**: The CPU closed-loop pipeline lowers S0 IR to executable LLVM IR for three kernel patterns: elementwise, reduction, stencil (no promotion).
  - Positive Tests (expected to PASS):
    - `spmd-opt ewise.mlir --normalize-spmd --materialize-spmd-tiling --convert-spmd-to-scf --convert-scf-to-openmp | mlir-translate --mlir-to-llvmir` produces valid LLVM IR
    - Same pipeline for `sum.mlir` (reduction) and `stencil_nopromote.mlir` produces valid LLVM IR
  - Negative Tests (expected to FAIL):
    - Pipeline on S0 IR containing `gpu.thread_id` aborts at `VerifySPMDKernelSubset` with a diagnostic

- **AC-8**: `PromoteGroupMemory` pass transforms a 2D stencil S0 kernel into S2 IR with group alloc, cooperative copy, barrier, and rewritten loads; result compiles to CPU via full pipeline.
  - AC-8.1: IR transformation correctness
    - Positive Tests (expected to PASS):
      - After `--promote-group-memory`, output contains `memref.alloc() : memref<..., #spmd.addr_space<group>>`
      - Output contains `spmd.barrier {spmd.scope = #spmd.scope<group>}` between copy and compute foralls
      - Compute forall reads from tile buffer (group addr space), not original global memref
    - Negative Tests (expected to FAIL):
      - Kernel with `spmd.memory_policy = no_promotion` is not transformed
      - Kernel whose footprint exceeds `target.maxGroupMemBytes` is skipped with a remark
  - AC-8.2: End-to-end executability
    - Positive Tests (expected to PASS):
      - Full pipeline `S0 → normalize → plan → materialize → promote → lower-to-scf → lower-to-openmp → mlir-translate → llc` produces an object file without errors

---

## MUTABLE SECTION
<!-- Update each round with justification for changes -->

### Plan Version: 1 (Updated: Round 0)

#### Plan Evolution Log
<!-- Document any changes to the plan with justification -->
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initial plan | - | - |

#### Active Tasks
<!-- Map each task to its target Acceptance Criterion -->
| Task | Target AC | Status | Notes |
|------|-----------|--------|-------|
| Add `tools/spmd-opt/spmd-opt.cpp` and `tools/spmd-opt/CMakeLists.txt` | AC-1 | pending | Mirror standalone example |
| Add `add_subdirectory(tools)` to root CMakeLists.txt | AC-1 | pending | |
| Configure `test/lit.cfg.py` and `test/lit.site.cfg.py.in`; verify `check-spmd` target | AC-1, AC-6 | pending | lit.cfg.py already exists |
| Add `SPMD_KernelAttr` (UnitAttr) to SPMDAttrs.td; register in dialect | AC-3 | pending | |
| Remove `hasCustomAssemblyFormat = 1` from ForallOp, IfOp, ReduceOp in SPMDOps.td | AC-4, AC-5 | pending | Use ODS generic format |
| Complete BarrierOp verifier: typed LevelAttr group-level check | AC-5.3 | pending | Currently TODO in SPMDOps.cpp |
| Complete ReduceOp verifier: typed ReductionKindAttr check | AC-5.2 | pending | |
| Complete YieldOp verifier: zero-operand check for ForallOp parent | AC-4 | pending | |
| Update VerifySPMDKernelSubset to use typed LevelAttr | AC-3 | pending | |
| Lit tests pass: ops.mlir + invalid.mlir | AC-6 | pending | Depends on build system + IR |
| Implement NormalizeSPMD pass | AC-7 | pending | |
| Implement MaterializeTilingAndMapping pass | AC-7 | pending | |
| Implement SPMDToSCF conversion (ForallOp→scf.for, IfOp→scf.if, ReduceOp→scf.for+iter_args) | AC-7 | pending | |
| Implement SPMDToOpenMP conversion (group-level scf.for → omp.parallel+wsloop) | AC-7 | pending | |
| Add lower-to-openmp.mlir lit test; verify ewise/reduction/stencil pipelines | AC-7 | pending | |
| Implement AccessSummaryAnalysis (affine footprint computation) | AC-8 | pending | |
| Implement PromotionPlanAnalysis (legality + profitability) | AC-8 | pending | |
| Implement PromoteGroupMemory pass (alloc, copy forall, barrier, rewrite loads) | AC-8 | pending | Read-only stencil only |
| Wire TargetDescriptor defaults (CPU: maxGroupMemBytes=48*1024) | AC-8 | pending | |
| Lit test promotion.mlir; verify stencil S0→S2→CPU end-to-end | AC-8 | pending | |

### Completed and Verified
<!-- Only move tasks here after Codex verification -->
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|

### Explicitly Deferred
<!-- Items here require strong justification -->
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|

### Open Issues
<!-- Issues discovered during implementation -->
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|
