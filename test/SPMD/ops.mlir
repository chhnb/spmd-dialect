// RUN: spmd-opt %s | FileCheck %s

// ---- S0: elementwise ----

// CHECK-LABEL: func @ewise_square
func.func @ewise_square(%A: memref<?x?xf32>, %B: memref<?x?xf32>,
                         %N: index, %M: index)
    attributes {spmd.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: spmd.forall
  "spmd.forall"(%c0, %c0, %N, %M, %c1, %c1) ({
  ^bb0(%i: index, %j: index):
    %x = memref.load %A[%i, %j] : memref<?x?xf32>
    %y = arith.mulf %x, %x : f32
    memref.store %y, %B[%i, %j] : memref<?x?xf32>
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 2, 2, 2>} : (index, index, index, index, index, index) -> ()
  func.return
}

// ---- S0: reduction ----

// CHECK-LABEL: func @sum
func.func @sum(%A: memref<?xf32>, %out: memref<1xf32>, %N: index)
    attributes {spmd.kernel} {
  %c0   = arith.constant 0 : index
  %c1   = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  // CHECK: spmd.reduce
  %sum = "spmd.reduce"(%c0, %N, %c1, %zero) ({
  ^bb0(%k: index):
    %x = memref.load %A[%k] : memref<?xf32>
    "spmd.yield"(%x) : (f32) -> ()
  }) {"spmd.kind" = #spmd.reduction_kind<add>} : (index, index, index, f32) -> f32
  memref.store %sum, %out[%c0] : memref<1xf32>
  func.return
}

// ---- Attributes round-trip ----

// CHECK-LABEL: func @attr_check
func.func @attr_check(%N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: #spmd.level<group>
  "spmd.forall"(%c0, %N, %c1) ({
  ^bb0(%i: index):
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.mapping" = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 32>,
      "spmd.memory_policy" = #spmd.memory_policy<prefer_group>} :
      (index, index, index) -> ()
  func.return
}

// ---- spmd.if: no result ----

// CHECK-LABEL: func @guarded_store
func.func @guarded_store(%A: memref<?xf32>, %N: index, %i: index) {
  %cond = arith.cmpi ult, %i, %N : index
  // CHECK: spmd.if
  "spmd.if"(%cond) ({
    %v = memref.load %A[%i] : memref<?xf32>
    memref.store %v, %A[%i] : memref<?xf32>
    "spmd.yield"() : () -> ()
  }, {
    "spmd.yield"() : () -> ()
  }) : (i1) -> ()
  func.return
}

// ---- spmd.if: with result ----

// CHECK-LABEL: func @select_val
func.func @select_val(%cond: i1, %x: f32, %y: f32) -> f32 {
  // CHECK: spmd.if
  %r = "spmd.if"(%cond) ({
    "spmd.yield"(%x) : (f32) -> ()
  }, {
    "spmd.yield"(%y) : (f32) -> ()
  }) : (i1) -> f32
  func.return %r : f32
}
