// RUN: spmd-opt %s --verify-spmd-kernel-subset | FileCheck %s

// Positive: a legal spmd.kernel function using only the S0/S1 allowed
// op subset (spmd.*, arith.*, memref.load/store, func.return) must pass
// --verify-spmd-kernel-subset without any diagnostic.

// CHECK-LABEL: func @legal_kernel
func.func @legal_kernel(%A: memref<?xf32>, %B: memref<?xf32>, %N: index)
    attributes {spmd.kernel} {
  %c0   = arith.constant 0 : index
  %c1   = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32

  // CHECK: spmd.forall
  "spmd.forall"(%c0, %N, %c1) ({
  ^bb0(%i: index):
    %v = memref.load %A[%i] : memref<?xf32>
    %w = arith.addf %v, %zero : f32
    memref.store %w, %B[%i] : memref<?xf32>
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>} : (index, index, index) -> ()

  func.return
}
