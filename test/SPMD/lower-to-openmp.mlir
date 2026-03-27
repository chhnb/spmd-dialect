// Lower SPMD kernels through the full CPU pipeline.
//
// Pipeline A (SCF baseline):
//   normalize → plan → materialize-tiling → convert-to-scf
//   All spmd ops become scf; no OpenMP.
//
// Pipeline B (OpenMP parallel):
//   normalize → plan → materialize-tiling → convert-to-openmp → convert-to-scf
//   Group-level foralls become omp.parallel+wsloop+loop_nest;
//   lane-level foralls become scf.for.

// ── Pipeline A: SCF baseline ──────────────────────────────────────────────
// RUN: spmd-opt %s --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-scf \
// RUN:   | FileCheck %s --check-prefix=SCF

// SCF-LABEL: func @ewise_kernel
// SCF-NOT: spmd.
// SCF: scf.for

// ── Pipeline B: OpenMP ────────────────────────────────────────────────────
// RUN: spmd-opt %s --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-openmp \
// RUN:   --convert-spmd-to-scf \
// RUN:   | FileCheck %s --check-prefix=OMP

// OMP-LABEL: func @ewise_kernel
// OMP-NOT: spmd.forall
// OMP: omp.parallel
// OMP: omp.loop_nest
// OMP: scf.for

func.func @ewise_kernel(%A: memref<?xf32>, %B: memref<?xf32>,
                         %C: memref<?xf32>, %N: index)
    attributes {spmd.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  "spmd.forall"(%c0, %N, %c1) ({
  ^bb0(%i: index):
    %a = memref.load %A[%i] : memref<?xf32>
    %b = memref.load %B[%i] : memref<?xf32>
    %c = arith.addf %a, %b : f32
    memref.store %c, %C[%i] : memref<?xf32>
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>} : (index, index, index) -> ()
  func.return
}

// -----

// RUN: spmd-opt %s --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-scf \
// RUN:   | FileCheck %s --check-prefix=REDUCE

// REDUCE-LABEL: func @reduce_sum
// REDUCE-NOT: spmd.reduce
// REDUCE: scf.for
// REDUCE: iter_args

func.func @reduce_sum(%A: memref<?xf32>, %N: index) -> f32
    attributes {spmd.kernel} {
  %c0    = arith.constant 0 : index
  %c1    = arith.constant 1 : index
  %czero = arith.constant 0.0 : f32
  %sum = "spmd.reduce"(%c0, %N, %c1, %czero) ({
  ^bb0(%k: index):
    %v = memref.load %A[%k] : memref<?xf32>
    "spmd.yield"(%v) : (f32) -> ()
  }) {"spmd.kind" = #spmd.reduction_kind<add>} : (index, index, index, f32) -> f32
  func.return %sum : f32
}

// -----

// ── Pipeline C: stencil → OpenMP MLIR ─────────────────────────────────────
// RUN: spmd-opt %s --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-openmp \
// RUN:   --convert-spmd-to-scf \
// RUN:   | FileCheck %s --check-prefix=STENCIL_OMP

// STENCIL_OMP-LABEL: func @stencil_kernel
// STENCIL_OMP-NOT: spmd.forall
// STENCIL_OMP: omp.parallel
// STENCIL_OMP: omp.loop_nest

// ── Pipeline D: stencil → OpenMP → LLVM IR ────────────────────────────────
// RUN: spmd-opt %s --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-openmp \
// RUN:   --convert-spmd-to-scf --convert-scf-to-cf --convert-arith-to-llvm \
// RUN:   --finalize-memref-to-llvm --convert-func-to-llvm --convert-cf-to-llvm \
// RUN:   --reconcile-unrealized-casts \
// RUN:   | mlir-translate --mlir-to-llvmir \
// RUN:   | FileCheck %s --check-prefix=STENCIL_LLVM

// STENCIL_LLVM: define

func.func @stencil_kernel(%A: memref<?xf32>, %B: memref<?xf32>, %N: index)
    attributes {spmd.kernel} {
  %c1   = arith.constant 1 : index
  %Nm1  = arith.subi %N, %c1 : index
  "spmd.forall"(%c1, %Nm1, %c1) ({
  ^bb0(%i: index):
    %i1     = arith.addi %i, %c1 : index
    %center = memref.load %A[%i ] : memref<?xf32>
    %right  = memref.load %A[%i1] : memref<?xf32>
    %out    = arith.addf %center, %right : f32
    memref.store %out, %B[%i] : memref<?xf32>
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>} : (index, index, index) -> ()
  func.return
}
