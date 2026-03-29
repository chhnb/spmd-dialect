// RUN: spmd-opt %s | FileCheck %s
// RUN: spmd-opt %s --mlir-print-op-generic | FileCheck %s --check-prefix=GENERIC

// All 5 custom attrs must survive generic round-trip (ops in generic
// form, attrs unchanged).  These global GENERIC checks confirm presence
// in output order: @sum (reduction_kind), then @attr_check body ops
// (addr_space in memref.load type, scope in spmd.barrier attrs), then the
// forall closing "})" which carries level and memory_policy.
// GENERIC: #spmd.reduction_kind<add>
// GENERIC: #spmd.addr_space<global>
// GENERIC: #spmd.scope<group>
// GENERIC: #spmd.level<group>
// GENERIC: #spmd.memory_policy<prefer_group>

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
  // CHECK: #spmd.reduction_kind<add>
  %sum = "spmd.reduce"(%c0, %N, %c1, %zero) ({
  ^bb0(%k: index):
    %x = memref.load %A[%k] : memref<?xf32>
    "spmd.yield"(%x) : (f32) -> ()
  }) {"spmd.kind" = #spmd.reduction_kind<add>} : (index, index, index, f32) -> f32
  memref.store %sum, %out[%c0] : memref<1xf32>
  func.return
}

// ---- Attributes round-trip ----
// All 5 custom attrs are exercised here and in @sum above.

// CHECK-LABEL: func @attr_check
// Attrs appear in output order: forall attrs first (level, memory_policy),
// then memref type (addr_space), then barrier (scope).
// CHECK: #spmd.level<group>
// CHECK: #spmd.memory_policy<prefer_group>
// CHECK: #spmd.addr_space<global>
// CHECK: #spmd.scope<group>
func.func @attr_check(%A: memref<?xf32, #spmd.addr_space<global>>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  "spmd.forall"(%c0, %N, %c1) ({
  ^bb0(%i: index):
    %v = memref.load %A[%i] : memref<?xf32, #spmd.addr_space<global>>
    "spmd.barrier"() {"spmd.scope" = #spmd.scope<group>} : () -> ()
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
