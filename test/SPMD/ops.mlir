// RUN: spmd-opt %s | FileCheck %s

// ---- S0: elementwise ----

// CHECK-LABEL: func @ewise_square
func.func @ewise_square(%A: memref<?x?xf32>, %B: memref<?x?xf32>,
                         %N: index, %M: index)
    attributes {spmd.kernel} {
  // CHECK: spmd.forall
  spmd.forall (%i, %j) in (%N, %M) {
    %x = memref.load %A[%i, %j] : memref<?x?xf32>
    %y = arith.mulf %x, %x : f32
    memref.store %y, %B[%i, %j] : memref<?x?xf32>
    spmd.yield
  }
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
  %sum = spmd.reduce (%k) = (%c0) to (%N) step (%c1)
             init(%zero)
             attributes {spmd.kind = #spmd.reduction_kind<add>} {
    %x = memref.load %A[%k] : memref<?xf32>
    spmd.yield %x : f32
  }
  memref.store %sum, %out[%c0] : memref<1xf32>
  func.return
}

// ---- S1: with schedule hints ----

// CHECK-LABEL: func @ewise_sched
func.func @ewise_sched(%A: memref<?x?xf32>, %B: memref<?x?xf32>,
                        %N: index, %M: index)
    attributes {spmd.kernel} {
  // CHECK: spmd.mapping = #spmd.level<group>
  // CHECK: spmd.tile_sizes = array<i64: 32, 8>
  spmd.forall (%i, %j) in (%N, %M)
      attributes {
        spmd.mapping = #spmd.level<group>,
        spmd.tile_sizes = array<i64: 32, 8>
      } {
    %x = memref.load %A[%i, %j] : memref<?x?xf32>
    %y = arith.mulf %x, %x : f32
    memref.store %y, %B[%i, %j] : memref<?x?xf32>
    spmd.yield
  }
  func.return
}

// ---- spmd.if: no result ----

// CHECK-LABEL: func @guarded_store
func.func @guarded_store(%A: memref<?xf32>, %N: index, %i: index) {
  %cond = arith.cmpi ult, %i, %N : index
  // CHECK: spmd.if
  spmd.if %cond {
    %v = memref.load %A[%i] : memref<?xf32>
    memref.store %v, %A[%i] : memref<?xf32>
    spmd.yield
  }
  func.return
}

// ---- spmd.if: with result ----

// CHECK-LABEL: func @select_val
func.func @select_val(%cond: i1, %x: f32, %y: f32) -> f32 {
  // CHECK: spmd.if
  %r = spmd.if %cond -> (f32) {
    spmd.yield %x : f32
  } else {
    spmd.yield %y : f32
  }
  func.return %r : f32
}
