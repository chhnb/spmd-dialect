// cross-pipeline-fail.mlir
//
// Fail-side regression tests: verify that structurally invalid inputs are
// correctly rejected or flagged by the pipeline passes. Each test uses a
// kernel that violates a specific precondition and checks the expected error.
//
// RUN: spmd-opt %s -split-input-file -verify-diagnostics \
// RUN:   --verify-spmd-gpu-ready

// ─────────────────────────────────────────────────────────────────────────────
// Fail-side regression 1: spmd.barrier in a divergent path (scf.if).
// --verify-spmd-gpu-ready must reject this kernel before GPU lowering.
// This prevents silent undefined behavior on GPU (divergent barrier).
// ─────────────────────────────────────────────────────────────────────────────

func.func @fail_barrier_divergent(%A: memref<?xf32>, %N: index, %cond: i1)
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
    "spmd.forall"(%c0, %c32, %c1) ({
    ^bb0(%ti: index):
      %i = arith.addi %ii, %ti : index
      %v = memref.load %A[%i] : memref<?xf32>
      memref.store %v, %A[%i] : memref<?xf32>
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
// Fail-side regression 2: spmd.forall nested inside gpu.launch body.
// This is a partial-lowering artifact — the outer group forall was converted
// to gpu.launch, but the inner lane forall was accidentally left behind.
// --verify-spmd-gpu-ready must flag this as a structural error.
// ─────────────────────────────────────────────────────────────────────────────

func.func @fail_forall_in_launch(%A: memref<?xf32>, %N: index) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %gx  = arith.ceildivui %N, %c32 : index
  // The outer group forall was already lowered to gpu.launch (e.g. by a
  // prior partial pass run), but the inner lane forall was not converted.
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
