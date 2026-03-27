// Lower SPMD kernels through the full CPU pipeline.
//
// Pipeline: normalize → materialize-tiling → convert-to-scf → (scf remains)
// The final --convert-scf-to-openmp (builtin) or --convert-spmd-to-openmp
// step is invoked to produce OpenMP IR.

// RUN: spmd-opt %s --normalize-spmd --materialize-spmd-tiling \
// RUN:   --convert-spmd-to-scf | FileCheck %s

// After --convert-spmd-to-scf there must be no remaining spmd.* ops.
// CHECK-LABEL: func @ewise_kernel
// CHECK-NOT: spmd.
// CHECK: scf.for

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

// RUN: spmd-opt %s --normalize-spmd --materialize-spmd-tiling \
// RUN:   --convert-spmd-to-scf | FileCheck %s --check-prefix=REDUCE

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
