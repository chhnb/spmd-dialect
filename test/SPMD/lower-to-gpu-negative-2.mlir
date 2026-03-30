// lower-to-gpu-negative-2.mlir — five new negative/boundary tests for --convert-spmd-to-gpu.
//
// RUN: spmd-opt %s -verify-diagnostics -split-input-file --convert-spmd-to-gpu
// RUN: (spmd-opt %s -split-input-file --convert-spmd-to-gpu; true) 2>&1 | FileCheck %s --check-prefix=NOERR

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: spmd.barrier nested inside spmd.if (divergent control flow).
// After moving group body to gpu.launch, the barrier's parent is spmd.if,
// not gpu.launch → error.
// ─────────────────────────────────────────────────────────────────────────────

func.func @barrier_in_if(%A: memref<?xf32>, %N: index, %cond: i1) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c32 = arith.constant 32 : index

  // expected-remark@+1 {{convert-spmd-to-gpu: gridDim=}}
  "spmd.forall"(%c0, %N, %c32) ({
  ^bb0(%ii: index):
    "spmd.if"(%cond) ({
      // expected-error@+1 {{gpu.barrier must be at gpu.launch body level}}
      "spmd.barrier"() {spmd.scope = #spmd.scope<group>} : () -> ()
      "spmd.yield"() : () -> ()
    }, {
      "spmd.yield"() : () -> ()
    }) : (i1) -> ()
    "spmd.forall"(%c0, %c32, %c1) ({
    ^bb0(%ti: index):
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

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: spmd.barrier nested inside a lane-level forall.
// After moving group body to gpu.launch, the barrier is inside the lane forall
// (a ForallOp), not directly inside the gpu.launch body → error.
// ─────────────────────────────────────────────────────────────────────────────

func.func @barrier_in_lane_forall(%A: memref<?xf32>, %N: index) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c32 = arith.constant 32 : index

  // expected-remark@+1 {{convert-spmd-to-gpu: gridDim=}}
  "spmd.forall"(%c0, %N, %c32) ({
  ^bb0(%ii: index):
    "spmd.forall"(%c0, %c32, %c1) ({
    ^bb0(%ti: index):
      // expected-error@+1 {{gpu.barrier must be at gpu.launch body level}}
      "spmd.barrier"() {spmd.scope = #spmd.scope<group>} : () -> ()
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

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: Single-dim lane forall with 1025 threads — just over the CUDA limit.
// ─────────────────────────────────────────────────────────────────────────────

func.func @blockdim_1025(%A: memref<?xf32>) {
  %c0    = arith.constant 0    : index
  %c1    = arith.constant 1    : index
  %c1025 = arith.constant 1025 : index

  // expected-error@+1 {{computed blockDim 1025 exceeds CUDA maximum of 1024}}
  "spmd.forall"(%c0, %c1, %c1) ({
  ^bb0(%ii: index):
    "spmd.forall"(%c0, %c1025, %c1) ({
    ^bb0(%ti: index):
      "spmd.yield"() : () -> ()
    }) {operandSegmentSizes = array<i32: 1, 1, 1>,
        "spmd.mapping" = #spmd.level<lane>} : (index, index, index) -> ()
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.mapping" = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 1>}
     : (index, index, index) -> ()
  func.return
}

// -----

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Dynamic multi-dim lane forall (rank=2, non-constant bounds).
// When ub/lb/step are not constant and rank > 1, delinearization is impossible.
// ─────────────────────────────────────────────────────────────────────────────

func.func @dynamic_multidim_lane(%A: memref<?x?xf32>, %N: index, %M: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // expected-remark@+1 {{convert-spmd-to-gpu: gridDim=}}
  "spmd.forall"(%c0, %c1, %c1) ({
  ^bb0(%ii: index):
    // expected-error@+1 {{dynamic multi-dim lane forall is not supported for GPU lowering}}
    "spmd.forall"(%c0, %c0, %N, %M, %c1, %c1) ({
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

// -----

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: 2D lane forall with all-constant bounds — verify row-major
// delinearization order. tx is split as: dim0 = tx / 8, dim1 = tx % 8.
// This is a positive correctness check (no error expected).
// The delinearized indices must appear as divui/remui in the output.
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Test 5b: 2D lane forall with 256×256 threads = 65,536 total — well above the
// CUDA maximum of 1024 threads/block. This verifies the blockDim overflow
// check for multi-dimensional lane foralls (flattened total exceeds limit).
// ─────────────────────────────────────────────────────────────────────────────

func.func @blockdim_2d_overflow(%A: memref<?x?xf32>, %N: index, %M: index) {
  %c0   = arith.constant 0   : index
  %c1   = arith.constant 1   : index
  %c256 = arith.constant 256 : index

  // 256 × 256 = 65,536 threads → exceeds CUDA max of 1024.
  // expected-error@+1 {{computed blockDim 65536 exceeds CUDA maximum of 1024}}
  "spmd.forall"(%c0, %c1, %c1) ({
  ^bb0(%ii: index):
    "spmd.forall"(%c0, %c0, %c256, %c256, %c1, %c1) ({
    ^bb0(%ti: index, %tj: index):
      %v = memref.load %A[%ti, %tj] : memref<?x?xf32>
      memref.store %v, %A[%ti, %tj] : memref<?x?xf32>
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

// -----

// NOERR-LABEL: func @lane_2d_delinearize
// NOERR: arith.divui
// NOERR: arith.remui
func.func @lane_2d_delinearize(%A: memref<?x?xf32>, %B: memref<?x?xf32>,
                                %N: index, %M: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // expected-remark@+1 {{convert-spmd-to-gpu: gridDim=}}
  "spmd.forall"(%c0, %c0, %N, %M, %c4, %c8) ({
  ^bb0(%ii: index, %jj: index):
    // 2D lane forall [0,4) x [0,8) — constant bounds, all-const.
    // Row-major flatten: linearId = ti * 8 + tj.
    // Delinearize: ti = tx / 8, tj = tx % 8.
    "spmd.forall"(%c0, %c0, %c4, %c8, %c1, %c1) ({
    ^bb0(%ti: index, %tj: index):
      %i = arith.addi %ii, %ti : index
      %j = arith.addi %jj, %tj : index
      %v = memref.load %A[%i, %j] : memref<?x?xf32>
      memref.store %v, %B[%i, %j] : memref<?x?xf32>
      "spmd.yield"() : () -> ()
    }) {operandSegmentSizes = array<i32: 2, 2, 2>,
        "spmd.mapping" = #spmd.level<lane>} : (index, index, index, index, index, index) -> ()
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 2, 2, 2>,
      "spmd.mapping" = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 4, 8>}
     : (index, index, index, index, index, index) -> ()
  func.return
}
