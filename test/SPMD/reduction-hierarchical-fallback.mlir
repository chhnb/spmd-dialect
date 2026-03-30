// reduction-hierarchical-fallback.mlir — AC-2.1
//
// ReduceToHierarchicalGPU must fall back to ReduceToSCFForGPU when the
// reduce body is non-pure (contains a func.call).
//
// Expected: remark "non-pure reduce body", no workgroup buffer, scf.for.
//
// RUN: spmd-opt %s --normalize-spmd --convert-spmd-to-gpu 2>&1 \
// RUN:   | FileCheck %s --check-prefix=FALLBACK

// FALLBACK:     remark: hierarchical reduction lowering skipped: non-pure reduce body
// FALLBACK-NOT: workgroup(
// FALLBACK:     scf.for

// Declare an external function that makes the reduce body non-pure.
func.func private @compute_partial(%idx: index) -> f32

func.func @reduce_non_pure(%out: memref<f32>, %N: index)
    attributes {spmd.kernel} {
  %c0   = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  "spmd.forall"(%c0, %N, %c256) ({
  ^bb0(%tile_start: index):
    %c0_i   = arith.constant 0 : index
    %c1     = arith.constant 1 : index
    %c256_i = arith.constant 256 : index
    %zero   = arith.constant 0.0 : f32
    %sum = "spmd.reduce"(%c0_i, %c256_i, %c1, %zero) ({
    ^bb1(%local_i: index):
      // func.call makes the body non-pure → L3 check fails → fallback
      %gidx = arith.addi %tile_start, %local_i : index
      %v = func.call @compute_partial(%gidx) : (index) -> f32
      "spmd.yield"(%v) : (f32) -> ()
    }) {"spmd.kind" = #spmd.reduction_kind<add>} : (index, index, index, f32) -> f32
    memref.atomic_rmw addf %sum, %out[] : (f32, memref<f32>) -> f32
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.mapping" = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 256>}
     : (index, index, index) -> ()
  func.return
}
