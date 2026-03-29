// RUN: spmd-opt %s --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-scf | FileCheck %s

// Pipeline: full lowering to LLVM IR via mlir-translate.
// RUN: spmd-opt %s --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling \
// RUN:   --convert-spmd-to-scf --convert-scf-to-cf --convert-arith-to-llvm \
// RUN:   --finalize-memref-to-llvm --convert-func-to-llvm --convert-cf-to-llvm \
// RUN:   --reconcile-unrealized-casts \
// RUN:   | mlir-translate --mlir-to-llvmir \
// RUN:   | FileCheck %s --check-prefix=LLVM

// LLVM: define

// Parallel reduction: sum all elements of A.
// After lowering, spmd.reduce becomes scf.for with iter_args for the
// accumulator, and the combiner (arith.addf) is applied each iteration.

// CHECK-LABEL: func @sum
// CHECK-NOT: spmd.reduce
// CHECK: scf.for
// CHECK: iter_args
// CHECK: arith.addf
// CHECK: scf.yield

func.func @sum(%A: memref<?xf32>, %N: index) -> f32
    attributes {spmd.kernel} {
  %c0    = arith.constant 0 : index
  %c1    = arith.constant 1 : index
  %czero = arith.constant 0.0 : f32
  %result = "spmd.reduce"(%c0, %N, %c1, %czero) ({
  ^bb0(%k: index):
    %v = memref.load %A[%k] : memref<?xf32>
    "spmd.yield"(%v) : (f32) -> ()
  }) {"spmd.kind" = #spmd.reduction_kind<add>} : (index, index, index, f32) -> f32
  func.return %result : f32
}

// -----

// spmd.if: conditional select pattern.
// CHECK-LABEL: func @cond_select
// CHECK-NOT: spmd.if
// CHECK: scf.if

func.func @cond_select(%cond: i1, %x: f32, %y: f32) -> f32
    attributes {spmd.kernel} {
  %r = "spmd.if"(%cond) ({
    "spmd.yield"(%x) : (f32) -> ()
  }, {
    "spmd.yield"(%y) : (f32) -> ()
  }) : (i1) -> f32
  func.return %r : f32
}
