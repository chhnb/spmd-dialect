// verify-spmd-gpu-ready.mlir
//
// Lit tests for --verify-spmd-gpu-ready.
//
// RUN: (spmd-opt %s --split-input-file --verify-spmd-gpu-ready; true) \
// RUN:   | FileCheck %s --check-prefix=GPUREADY
// RUN: spmd-opt %s --split-input-file --verify-spmd-gpu-ready \
// RUN:   -verify-diagnostics
//
// See verify-spmd-invariants.mlir for --verify-spmd-promotion-invariant tests.

// Test 1: positive -- clean S2 IR (group + lane foralls) passes.
// GPUREADY-LABEL: func @clean_s2
func.func @clean_s2(%A: memref<?xf32>, %B: memref<?xf32>, %N: index)
    attributes {spmd.kernel} {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c32 = arith.constant 32 : index

  "spmd.forall"(%c0, %N, %c32) ({
  ^bb0(%ii: index):
    "spmd.forall"(%c0, %c32, %c1) ({
    ^bb0(%ti: index):
      %i = arith.addi %ii, %ti : index
      %v = memref.load %A[%i] : memref<?xf32>
      memref.store %v, %B[%i] : memref<?xf32>
      "spmd.yield"() : () -> ()
    }) {operandSegmentSizes = array<i32: 1, 1, 1>,
        "spmd.mapping" = #spmd.level<lane>} : (index, index, index) -> ()
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.mapping" = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 32>}
     : (index, index, index) -> ()
  func.return
}

// -----

// Test 2: negative -- spmd.barrier inside scf.if inside a group forall.
// A barrier in a conditional block is a divergent barrier (undefined GPU behavior).
func.func @barrier_in_scf_if(%A: memref<?xf32>, %N: index, %cond: i1)
    attributes {spmd.kernel} {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c32 = arith.constant 32 : index

  "spmd.forall"(%c0, %N, %c32) ({
  ^bb0(%ii: index):
    scf.if %cond {
      // expected-error@+1 {{spmd.barrier must not be nested inside scf.if}}
      "spmd.barrier"() {spmd.scope = #spmd.scope<group>} : () -> ()
    }
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.mapping" = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 32>}
     : (index, index, index) -> ()
  func.return
}

// -----

// Test 2b: negative -- spmd.forall inside a gpu.launch body (partial lowering).
// The outer group forall was converted to gpu.launch but the inner lane forall
// was accidentally left behind — an illegal mixed-dialect state.
func.func @spmd_forall_in_launch(%A: memref<?xf32>, %N: index) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %gx  = arith.ceildivui %N, %c32 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gbx = %gx, %gby = %c1, %gbz = %c1)
             threads(%tx, %ty, %tz) in (%tbx = %c32, %tby = %c1, %tbz = %c1) {
    %ii = arith.muli %bx, %c32 : index
    // expected-error@+1 {{spmd.forall found inside gpu.launch body}}
    "spmd.forall"(%c0, %c32, %c1) ({
    ^bb0(%ti: index):
      %i = arith.addi %ii, %ti : index
      %v = memref.load %A[%i] : memref<?xf32>
      memref.store %v, %A[%i] : memref<?xf32>
      "spmd.yield"() : () -> ()
    }) {operandSegmentSizes = array<i32: 1, 1, 1>,
        "spmd.mapping" = #spmd.level<lane>} : (index, index, index) -> ()
    gpu.terminator
  }
  func.return
}

// -----

// Test 2c: negative -- group-space alloc inside a gpu.launch body (stale state).
// A group-space alloc should be allocated OUTSIDE the launch, not inside it.
func.func @group_alloc_in_launch(%N: index) {
  %c1   = arith.constant 1 : index
  %c32  = arith.constant 32 : index
  %zero = arith.constant 0.0 : f32
  %gx   = arith.ceildivui %N, %c32 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gbx = %gx, %gby = %c1, %gbz = %c1)
             threads(%tx, %ty, %tz) in (%tbx = %c32, %tby = %c1, %tbz = %c1) {
    // expected-error@+1 {{group-address-space memref.alloc found inside gpu.launch body}}
    %tile = memref.alloc() : memref<32xf32, #spmd.addr_space<group>>
    memref.store %zero, %tile[%tx] : memref<32xf32, #spmd.addr_space<group>>
    gpu.terminator
  }
  func.return
}

// -----

// Test 3: negative -- spmd.reduce inside a gpu.launch body.
// This indicates the reduction was not lowered before GPU outlining.
func.func @reduce_in_launch(%A: memref<?xf32>, %N: index) {
  %c0   = arith.constant 0   : index
  %c1   = arith.constant 1   : index
  %c32  = arith.constant 32  : index
  %zero = arith.constant 0.0 : f32
  %gx   = arith.ceildivui %N, %c32 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gbx = %gx, %gby = %c1, %gbz = %c1)
             threads(%tx, %ty, %tz) in (%tbx = %c32, %tby = %c1, %tbz = %c1) {
    // expected-error@+1 {{spmd.reduce found inside gpu.launch body}}
    %sum = "spmd.reduce"(%c0, %N, %c1, %zero) ({
    ^bb0(%i: index):
      %v = memref.load %A[%i] : memref<?xf32>
      "spmd.yield"(%v) : (f32) -> ()
    }) {"spmd.kind" = #spmd.reduction_kind<add>} : (index, index, index, f32) -> f32
    gpu.terminator
  }
  func.return
}
