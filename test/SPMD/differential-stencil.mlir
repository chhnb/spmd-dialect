// differential-stencil.mlir — Stencil kernel for CPU/OpenMP differential testing.
//
// Smoke test: verify the SCF and OpenMP pipelines produce valid LLVM IR.
// RUN: spmd-opt %s --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-scf \
// RUN: | mlir-opt \
// RUN:   --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm \
// RUN:   --convert-index-to-llvm --convert-func-to-llvm \
// RUN:   --finalize-memref-to-llvm --reconcile-unrealized-casts \
// RUN: | mlir-translate --mlir-to-llvmir \
// RUN: | FileCheck %s --check-prefix=SCF
//
// RUN: spmd-opt %s --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-openmp --convert-spmd-to-scf \
// RUN: | mlir-opt \
// RUN:   --convert-openmp-to-llvm --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:   --convert-arith-to-llvm --convert-index-to-llvm --convert-func-to-llvm \
// RUN:   --finalize-memref-to-llvm --reconcile-unrealized-casts \
// RUN: | mlir-translate --mlir-to-llvmir \
// RUN: | FileCheck %s --check-prefix=OMP
//
// SCF: define void @stencil_cpu(
// OMP: define void @stencil_cpu(
//
// Semantics: B[i,j] = A[i,j] + A[i,j+1] + A[i+1,j]
//   A has shape (N+1, M+1) to provide one extra row/col for boundary access.
//   B has shape (N, M); the forall runs over [0,N) × [0,M).
//
// This is the same stencil semantics as lower-to-gpu-nvptx-promoted.mlir but
// expressed without memory_policy so it compiles cleanly via SCF and OpenMP
// paths without triggering group memory promotion.
//
// Used by scripts/run-differential.sh to compile SCF and OpenMP shared libraries.

func.func @stencil_cpu(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %N: index, %M: index)
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
      "spmd.memory_policy" = #spmd.memory_policy<no_promotion>}
     : (index, index, index, index, index, index) -> ()
  func.return
}
