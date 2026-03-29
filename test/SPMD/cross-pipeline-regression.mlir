// cross-pipeline-regression.mlir
//
// Cross-pipeline regression tests spanning at least 4 passes each.
// Each pipeline variant targets a different lowering path.
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
// Pipeline (e): normalize → plan → materialize → gpu (no-promotion path).
// Verifies that a stencil kernel on the no-promotion path has no workgroup memory.
// RUN: spmd-opt %s --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-gpu \
// RUN:   | FileCheck %s --check-prefix=NOSHARED
//
// Pipeline (f): full PTX pipeline is tested in lower-to-gpu-nvptx-promoted.mlir.
// (spmd-extract-gpu-module extracts only the first gpu.module; running it on
// this multi-kernel file would extract the ewise kernel which has no .shared.
// The promoted stencil PTX coverage is provided by lower-to-gpu-nvptx-promoted.mlir.)

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

// (Pipeline (f) PTX checks removed; see lower-to-gpu-nvptx-promoted.mlir)

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
