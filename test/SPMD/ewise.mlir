// RUN: spmd-opt %s --normalize-spmd --materialize-spmd-tiling \
// RUN:   --convert-spmd-to-scf | FileCheck %s

// Elementwise addition: C[i] = A[i] + B[i]
// After the CPU pipeline there must be no spmd.* ops remaining, and
// the iteration must be expressed as scf.for.

// CHECK-LABEL: func @ewise
// CHECK-NOT: spmd.
// CHECK: scf.for
// CHECK: memref.load
// CHECK: arith.addf
// CHECK: memref.store

func.func @ewise(%A: memref<?xf32>, %B: memref<?xf32>, %C: memref<?xf32>,
                  %N: index)
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

// 2D elementwise: C[i,j] = A[i,j] * B[i,j]

// CHECK-LABEL: func @ewise2d
// CHECK-NOT: spmd.
// CHECK: scf.for
// CHECK: scf.for

func.func @ewise2d(%A: memref<?x?xf32>, %B: memref<?x?xf32>,
                    %C: memref<?x?xf32>, %N: index, %M: index)
    attributes {spmd.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  "spmd.forall"(%c0, %c0, %N, %M, %c1, %c1) ({
  ^bb0(%i: index, %j: index):
    %a = memref.load %A[%i, %j] : memref<?x?xf32>
    %b = memref.load %B[%i, %j] : memref<?x?xf32>
    %c = arith.mulf %a, %b : f32
    memref.store %c, %C[%i, %j] : memref<?x?xf32>
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 2, 2, 2>} : (index, index, index, index, index, index) -> ()
  func.return
}
