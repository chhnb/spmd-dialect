// lower-to-gpu.mlir — lit tests for --convert-spmd-to-gpu (IR checks only)
//
// RUN 1: elementwise kernel GPU IR check (non-promoted path).
// RUN: spmd-opt %s --split-input-file --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-gpu \
// RUN:   | FileCheck %s --check-prefix=GPU
//
// RUN 2: promoted stencil workgroup memory + barrier check.
// RUN: spmd-opt %s --split-input-file --promote-group-memory \
// RUN:   --convert-spmd-to-gpu \
// RUN:   | FileCheck %s --check-prefix=WG

// ─── GPU checks (RUN 1): basic IR structure ──────────────────────────────────
// GPU-LABEL: func @ewise
// GPU:       gpu.launch
// GPU:       gpu.terminator
// GPU-NOT:   spmd.forall

// ─── WG checks (RUN 2): workgroup memory and barrier ─────────────────────────
// WG-LABEL: func @promoted_stencil
// WG:       gpu.launch
// WG-SAME:  workgroup({{.*}}#gpu.address_space<workgroup>
// WG:       gpu.barrier{{.*}}#gpu.address_space<workgroup>
// WG-NOT:   memref.alloc
// WG:       gpu.terminator

// ─────────────────────────────────────────────────────────────────────────────
// Test function 1: elementwise add (1D, non-promoted).
// After --normalize-spmd --plan-spmd-schedule --materialize-spmd-tiling this
// becomes a group forall [0, N) step 32 wrapping a lane forall [0, 32) step 1.
// ─────────────────────────────────────────────────────────────────────────────

func.func @ewise(%A: memref<?xf32>, %B: memref<?xf32>, %C: memref<?xf32>,
                  %N: index)
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
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.mapping" = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 32>}
     : (index, index, index) -> ()
  func.return
}

// -----

// ─────────────────────────────────────────────────────────────────────────────
// Test function 2: 2D stencil (promoted path — has group-addr-space alloc +
// spmd.barrier after --promote-group-memory).
// Used by RUN 2 (WG checks).
// ─────────────────────────────────────────────────────────────────────────────

func.func @promoted_stencil(%A: memref<?x?xf32>, %B: memref<?x?xf32>,
                              %N: index, %M: index)
    attributes {spmd.kernel} {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c8  = arith.constant 8 : index
  %c32 = arith.constant 32 : index

  "spmd.forall"(%c0, %c0, %N, %M, %c32, %c8) ({
  ^bb0(%ii: index, %jj: index):

    "spmd.forall"(%c0, %c0, %c32, %c8, %c1, %c1) ({
    ^bb0(%ti: index, %tj: index):
      %i  = arith.addi %ii, %ti : index
      %j  = arith.addi %jj, %tj : index
      %i1 = arith.addi %i,  %c1 : index
      %j1 = arith.addi %j,  %c1 : index
      %center = memref.load %A[%i,  %j ] : memref<?x?xf32>
      %right  = memref.load %A[%i,  %j1] : memref<?x?xf32>
      %down   = memref.load %A[%i1, %j ] : memref<?x?xf32>
      %t0 = arith.addf %center, %right : f32
      %t1 = arith.addf %t0, %down      : f32
      memref.store %t1, %B[%i, %j] : memref<?x?xf32>
      "spmd.yield"() : () -> ()
    }) {operandSegmentSizes = array<i32: 2, 2, 2>,
        "spmd.mapping" = #spmd.level<lane>} : (index, index, index, index, index, index) -> ()

    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 2, 2, 2>,
      "spmd.mapping" = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 32, 8>,
      "spmd.memory_policy" = #spmd.memory_policy<prefer_group>}
     : (index, index, index, index, index, index) -> ()

  func.return
}
