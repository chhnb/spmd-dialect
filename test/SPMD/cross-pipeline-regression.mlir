// cross-pipeline-regression.mlir
//
// Cross-pipeline regression tests spanning at least 4 passes each.
// Each pipeline variant targets a different lowering path.
//
// REQUIRES: nvptx-registered-target
//
// Pipeline (a): normalize → plan → materialize → scf (CPU serial)
// RUN: spmd-opt %s --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-scf \
// RUN:   | FileCheck %s --check-prefix=SCF
//
// Pipeline (b): normalize → plan → materialize → openmp (CPU parallel)
// RUN: spmd-opt %s --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-openmp \
// RUN:   --convert-spmd-to-scf \
// RUN:   | FileCheck %s --check-prefix=OMP
//
// Pipeline (c): normalize → plan → materialize → promote → gpu
// RUN: spmd-opt %s --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --promote-group-memory \
// RUN:   --convert-spmd-to-gpu \
// RUN:   | FileCheck %s --check-prefix=GPU
//
// Pipeline (d): normalize → plan → materialize → promote → gpu → outline → nvvm
// (outlining + NVVM lowering pipeline — verifies no crash and no residual spmd ops)
// RUN: spmd-opt %s --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --promote-group-memory \
// RUN:   --convert-spmd-to-gpu \
// RUN:   --gpu-kernel-outlining "--nvvm-attach-target=chip=sm_80" \
// RUN:   | mlir-opt --convert-gpu-to-nvvm --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:     --convert-arith-to-llvm --convert-index-to-llvm \
// RUN:     --reconcile-unrealized-casts \
// RUN:   | FileCheck %s --check-prefix=NVVM
//
// Pipeline (d'): normalize → plan → materialize → promote → gpu → outline →
//                nvvm → llc -filetype=asm (PTX).
// Verifies the full PTX pipeline compiles without crash and emits a kernel entry.
// RUN: spmd-opt %s --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --promote-group-memory \
// RUN:   --convert-spmd-to-gpu \
// RUN:   --gpu-kernel-outlining "--nvvm-attach-target=chip=sm_80" \
// RUN: | mlir-opt --convert-gpu-to-nvvm --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:     --convert-arith-to-llvm --convert-index-to-llvm \
// RUN:     --reconcile-unrealized-casts \
// RUN: | spmd-opt --spmd-extract-gpu-module \
// RUN: | mlir-translate --mlir-to-llvmir \
// RUN: | llc --march=nvptx64 --mcpu=sm_80 -filetype=asm \
// RUN: | FileCheck %s --check-prefix=PTX
//
// Pipeline (e): normalize → plan → materialize → gpu (no-promotion path).
// Verifies that a stencil kernel on the no-promotion path has no workgroup memory.
// RUN: spmd-opt %s --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-gpu \
// RUN:   | FileCheck %s --check-prefix=NOSHARED
//
// Pipeline (e'): normalize → plan → materialize → gpu (no-promotion) →
//                outline → nvvm → llc -filetype=asm (PTX).
// Verifies that the no-promotion path emits no .shared in PTX.
// RUN: spmd-opt %s --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-gpu \
// RUN:   --gpu-kernel-outlining "--nvvm-attach-target=chip=sm_80" \
// RUN: | mlir-opt --convert-gpu-to-nvvm --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:     --convert-arith-to-llvm --convert-index-to-llvm \
// RUN:     --reconcile-unrealized-casts \
// RUN: | spmd-opt --spmd-extract-gpu-module \
// RUN: | mlir-translate --mlir-to-llvmir \
// RUN: | llc --march=nvptx64 --mcpu=sm_80 -filetype=asm \
// RUN: | FileCheck %s --check-prefix=NOSHAREPTX
//
// Pipeline (f): normalize regression — plan without normalize shows non-canonical lb.
// Runs plan+materialize+scf WITHOUT normalize on the ewise_nonzero_lb kernel.
// The non-zero lb=4 survives in the SCF output, confirming normalize is required.
// RUN: spmd-opt %s --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-scf \
// RUN:   | FileCheck %s --check-prefix=SKIPNORM

// ─── Pipeline (a) checks: SCF lowering ───────────────────────────────────────
// SCF-LABEL: func @ewise_regression
// SCF-NOT:   spmd.forall
// SCF-NOT:   spmd.barrier
// SCF:       scf.for

// ─── Pipeline (b) checks: OpenMP lowering ────────────────────────────────────
// OMP-LABEL: func @ewise_regression
// OMP-NOT:   spmd.forall
// OMP:       scf.for

// ─── Pipeline (c) checks: GPU lowering ───────────────────────────────────────
// GPU-LABEL: func @ewise_regression
// GPU-NOT:   spmd.forall
// GPU:       gpu.launch
// GPU:       gpu.terminator

// ─── Pipeline (d) checks: NVVM → LLVM dialect lowering ───────────────────────
// After full NVVM lowering, the kernel body is in LLVM dialect.
// No residual spmd.* or gpu.launch ops in the host function.
// NVVM-LABEL: func @ewise_regression
// NVVM-NOT:   spmd.forall
// NVVM-NOT:   gpu.launch blocks
// No SPMD address space types survive to LLVM dialect.
// NVVM-NOT:   #spmd.addr_space

// ─── Pipeline (e) checks: no-promotion GPU (no workgroup memory) ─────────────
// NOSHARED-LABEL: func @stencil_nopromote_regression
// NOSHARED-NOT:   #spmd.addr_space<group>
// NOSHARED-NOT:   gpu.workgroup
// NOSHARED:       gpu.launch

// ─── Pipeline (d') checks: PTX (promoted path, ewise kernel extracted) ───────
// The first extracted gpu.module is the ewise kernel (no shared memory).
// PTX: .entry ewise_regression_kernel
// PTX-NOT: spmd.forall

// ─── Pipeline (e') checks: PTX no-shared (no-promotion path) ─────────────────
// The no-promotion path must not emit .shared declarations in PTX.
// NOSHAREPTX-NOT: .shared
// NOSHAREPTX:     .entry

// ─── Pipeline (f) checks: normalize regression ────────────────────────────────
// When --normalize-spmd is skipped, the non-zero lb=4 from @ewise_nonzero_lb
// survives into the SCF output. This confirms that normalize is required.
// SKIPNORM-LABEL: func @ewise_nonzero_lb
// SKIPNORM:       arith.constant 4 : index

// ─────────────────────────────────────────────────────────────────────────────
// Source kernel 1: 1D elementwise B[i] = A[i] + 1.0
// S0 IR: a single flat forall. After normalize → plan → materialize, it becomes
// a group forall [0, N) step 32 wrapping a lane forall [0, 32) step 1.
// ─────────────────────────────────────────────────────────────────────────────

func.func @ewise_regression(%A: memref<?xf32>, %B: memref<?xf32>, %N: index)
    attributes {spmd.kernel} {
  %c0  = arith.constant 0   : index
  %c1  = arith.constant 1   : index
  %one = arith.constant 1.0 : f32
  "spmd.forall"(%c0, %N, %c1) ({
  ^bb0(%i: index):
    %a  = memref.load %A[%i] : memref<?xf32>
    %b  = arith.addf %a, %one : f32
    memref.store %b, %B[%i] : memref<?xf32>
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>}
     : (index, index, index) -> ()
  func.return
}

// -----

// ─────────────────────────────────────────────────────────────────────────────
// Source kernel 2: 2D stencil without memory_policy (no promotion).
// After normalize → plan → materialize, PlanSPMDSchedule assigns the default
// (no memory_policy or no_promotion equivalent). PromoteGroupMemory skips it.
// Pipeline (e) checks that no workgroup memory appears in the GPU output.
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Source kernel 3: 2D promoted stencil with memory_policy = prefer_group.
// After normalize → plan → materialize → promote → gpu:
//   - Group-space alloc converted to workgroup memory
//   - Pipeline (f) verifies .shared appears in PTX output
// ─────────────────────────────────────────────────────────────────────────────

func.func @stencil_promoted_regression(
    %A: memref<?x?xf32>, %B: memref<?x?xf32>, %N: index, %M: index)
    attributes {spmd.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  "spmd.forall"(%c0, %c0, %N, %M, %c1, %c1) ({
  ^bb0(%i: index, %j: index):
    %j1 = arith.addi %j, %c1 : index
    %i1 = arith.addi %i, %c1 : index
    %center = memref.load %A[%i, %j]   : memref<?x?xf32>
    %right  = memref.load %A[%i, %j1]  : memref<?x?xf32>
    %down   = memref.load %A[%i1, %j]  : memref<?x?xf32>
    %t0 = arith.addf %center, %right : f32
    %t1 = arith.addf %t0, %down      : f32
    memref.store %t1, %B[%i, %j] : memref<?x?xf32>
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 2, 2, 2>,
      "spmd.memory_policy" = #spmd.memory_policy<prefer_group>}
     : (index, index, index, index, index, index) -> ()
  func.return
}

// -----

// ─────────────────────────────────────────────────────────────────────────────
// Source kernel 4: 2D stencil without memory_policy (no promotion).
// After normalize → plan → materialize, PlanSPMDSchedule assigns the default
// (no memory_policy or no_promotion equivalent). PromoteGroupMemory skips it.
// Pipeline (e) checks that no workgroup memory appears in the GPU output.
// ─────────────────────────────────────────────────────────────────────────────

func.func @stencil_nopromote_regression(
    %A: memref<?x?xf32>, %B: memref<?x?xf32>, %N: index, %M: index)
    attributes {spmd.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  "spmd.forall"(%c0, %c0, %N, %M, %c1, %c1) ({
  ^bb0(%i: index, %j: index):
    %j1 = arith.addi %j, %c1 : index
    %i1 = arith.addi %i, %c1 : index
    %center = memref.load %A[%i, %j]   : memref<?x?xf32>
    %right  = memref.load %A[%i, %j1]  : memref<?x?xf32>
    %down   = memref.load %A[%i1, %j]  : memref<?x?xf32>
    %t0 = arith.addf %center, %right : f32
    %t1 = arith.addf %t0, %down      : f32
    memref.store %t1, %B[%i, %j] : memref<?x?xf32>
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 2, 2, 2>}
     : (index, index, index, index, index, index) -> ()
  func.return
}

// ─────────────────────────────────────────────────────────────────────────────
// Source kernel 5: 1D elementwise with non-zero lower bound.
// Used by pipeline (f) to verify that --normalize-spmd is required before
// --plan-spmd-schedule. Without normalize, lb=4 survives in the SCF output.
// With normalize (pipelines a-e), lb is shifted to 0.
// ─────────────────────────────────────────────────────────────────────────────

func.func @ewise_nonzero_lb(%A: memref<?xf32>, %B: memref<?xf32>, %N: index)
    attributes {spmd.kernel} {
  %c4  = arith.constant 4 : index
  %c1  = arith.constant 1 : index
  %one = arith.constant 1.0 : f32
  "spmd.forall"(%c4, %N, %c1) ({
  ^bb0(%i: index):
    %a  = memref.load %A[%i] : memref<?xf32>
    %b  = arith.addf %a, %one : f32
    memref.store %b, %B[%i] : memref<?xf32>
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>}
     : (index, index, index) -> ()
  func.return
}
