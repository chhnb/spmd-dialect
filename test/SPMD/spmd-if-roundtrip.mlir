// RUN: spmd-opt %s | FileCheck %s
// RUN: spmd-opt %s | spmd-opt | FileCheck %s
// RUN: spmd-opt %s --mlir-print-op-generic | spmd-opt | FileCheck %s
//
// Regression test for spmd.if custom assembly format roundtrip.
// The declarative format emitted `: ()` for zero-result ops which the MLIR
// parser rejected as "expected non-function type".  The custom printer/parser
// must roundtrip both the zero-result and result-bearing forms.

// ---- Zero-result form: no "-> ()" should appear in output ----

// CHECK-LABEL: func @if_no_result
func.func @if_no_result(%A: memref<?xf32>, %N: index, %i: index) {
  %cond = arith.cmpi ult, %i, %N : index
  // CHECK: spmd.if %{{.*}} : i1
  // CHECK-NOT: -> ()
  // CHECK: spmd.yield
  spmd.if %cond : i1 {
    %v = memref.load %A[%i] : memref<?xf32>
    memref.store %v, %A[%i] : memref<?xf32>
    spmd.yield
  }
  func.return
}

// ---- Zero-result form with else: no "-> ()" should appear ----

// CHECK-LABEL: func @if_no_result_else
func.func @if_no_result_else(%A: memref<?xf32>, %N: index, %i: index) {
  %cond = arith.cmpi ult, %i, %N : index
  // CHECK: spmd.if %{{.*}} : i1
  // CHECK-NOT: -> ()
  // CHECK: else
  spmd.if %cond : i1 {
    %v = memref.load %A[%i] : memref<?xf32>
    memref.store %v, %A[%i] : memref<?xf32>
    spmd.yield
  } else {
    spmd.yield
  }
  func.return
}

// ---- Result-bearing form: "-> (f32)" must appear ----

// CHECK-LABEL: func @if_with_result
func.func @if_with_result(%cond: i1, %x: f32, %y: f32) -> f32 {
  // CHECK: spmd.if %{{.*}} : i1 -> (f32)
  %r = spmd.if %cond : i1 -> (f32) {
    spmd.yield %x : f32
  } else {
    spmd.yield %y : f32
  }
  func.return %r : f32
}
