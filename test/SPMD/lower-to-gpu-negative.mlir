// lower-to-gpu-negative.mlir — negative tests for --convert-spmd-to-gpu.
// Verifies that malformed inputs produce the expected error diagnostics.
//
// RUN: spmd-opt %s -verify-diagnostics -split-input-file --convert-spmd-to-gpu

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: nested group-level forall — must emit error on the inner forall.
// ─────────────────────────────────────────────────────────────────────────────

func.func @nested_group(%A: memref<?xf32>, %N: index) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c16 = arith.constant 16 : index

  "spmd.forall"(%c0, %N, %c16) ({
  ^bb0(%ii: index):
    // expected-error@+1 {{nested group-level spmd.forall is not supported}}
    "spmd.forall"(%c0, %c16, %c1) ({
    ^bb0(%ti: index):
      "spmd.yield"() : () -> ()
    }) {operandSegmentSizes = array<i32: 1, 1, 1>,
        "spmd.mapping" = #spmd.level<group>,
        "spmd.tile_sizes" = array<i64: 16>} : (index, index, index) -> ()
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.mapping" = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 16>}
     : (index, index, index) -> ()
  func.return
}

// -----

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: group forall with rank > 3 — GPU has at most a 3D grid.
// ─────────────────────────────────────────────────────────────────────────────

func.func @rank4_group(%A: memref<f32>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index

  // expected-error@+1 {{group forall rank > 3 is not supported for GPU lowering}}
  "spmd.forall"(%c0, %c0, %c0, %c0, %c4, %c4, %c4, %c4, %c1, %c1, %c1, %c1) ({
  ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 4, 4, 4>,
      "spmd.mapping" = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 4, 4, 4, 4>}
     : (index, index, index, index, index, index, index, index,
        index, index, index, index) -> ()
  func.return
}

// -----

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: computed blockDim > 1024 — CUDA maximum is 1024 threads per block.
// Lane forall has 33×32 = 1056 trip count → exceeds limit.
// ─────────────────────────────────────────────────────────────────────────────

func.func @blockdim_overflow(%A: memref<?xf32>) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c33 = arith.constant 33 : index
  %c32 = arith.constant 32 : index

  // expected-error@+1 {{computed blockDim 1056 exceeds CUDA maximum of 1024}}
  "spmd.forall"(%c0, %c1, %c1) ({
  ^bb0(%ii: index):
    "spmd.forall"(%c0, %c0, %c33, %c32, %c1, %c1) ({
    ^bb0(%ti: index, %tj: index):
      "spmd.yield"() : () -> ()
    }) {operandSegmentSizes = array<i32: 2, 2, 2>,
        "spmd.mapping" = #spmd.level<lane>} : (index, index, index, index, index, index) -> ()
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.mapping" = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 1>}
     : (index, index, index) -> ()
  func.return
}
