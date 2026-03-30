// stencil-nopromote-only.mlir
//
// AC-9: Promotion ablation negative input.
// A 2D stencil kernel with NO memory_policy attribute — promote-group-memory
// must leave it unchanged (no group-space alloc inserted).
//
// Used by check-full.sh ablation 4b to verify that the no-promotion path
// produces no #spmd.addr_space<group> output after the full tiling pipeline.
//
// RUN: spmd-opt %s --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --promote-group-memory \
// RUN:   | FileCheck %s
//
// CHECK-LABEL: func @stencil_nopromote
// No group-space allocation should appear — promotion was skipped.
// CHECK-NOT: #spmd.addr_space<group>

func.func @stencil_nopromote(
    %A: memref<?x?xf32>, %B: memref<?x?xf32>, %N: index, %M: index)
    attributes {spmd.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  "spmd.forall"(%c0, %c0, %N, %M, %c1, %c1) ({
  ^bb0(%i: index, %j: index):
    %j1 = arith.addi %j, %c1 : index
    %i1 = arith.addi %i, %c1 : index
    %center = memref.load %A[%i, %j]  : memref<?x?xf32>
    %right  = memref.load %A[%i, %j1] : memref<?x?xf32>
    %down   = memref.load %A[%i1, %j] : memref<?x?xf32>
    %t0 = arith.addf %center, %right : f32
    %t1 = arith.addf %t0, %down      : f32
    memref.store %t1, %B[%i, %j] : memref<?x?xf32>
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 2, 2, 2>,
      "spmd.memory_policy" = #spmd.memory_policy<no_promotion>}
     : (index, index, index, index, index, index) -> ()
  func.return
}
