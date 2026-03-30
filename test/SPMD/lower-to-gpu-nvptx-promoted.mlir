// lower-to-gpu-nvptx-promoted.mlir — NVPTX codegen smoke test (promoted path).
//
// REQUIRES: nvptx-registered-target
//
// PTX content check for the promoted stencil: the device PTX must contain
// .shared (workgroup memory demoted to PTX shared space) and .visible .entry
// (exported kernel entry point).
// This test input is pre-planned: the forall ops already carry spmd.mapping,
// spmd.tile_sizes, and spmd.memory_policy attributes, so --normalize-spmd,
// --plan-spmd-schedule, and --materialize-spmd-tiling are omitted here.
// Note: scripts/dump-pipeline.sh runs the full 6-stage pipeline (including
// materialize) on this same input file, as required by AC-9.2.
// RUN: spmd-opt %s --promote-group-memory \
// RUN:   --convert-spmd-to-gpu \
// RUN:   --gpu-kernel-outlining "--nvvm-attach-target=chip=sm_80" \
// RUN:   | mlir-opt --convert-gpu-to-nvvm \
// RUN:   --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm \
// RUN:   --convert-index-to-llvm --reconcile-unrealized-casts \
// RUN:   | spmd-opt --spmd-extract-gpu-module \
// RUN:   | mlir-translate --mlir-to-llvmir \
// RUN:   | llc --march=nvptx64 --mcpu=sm_80 -filetype=asm \
// RUN:   | FileCheck %s --check-prefix=PTX

// ─── PTX checks ───────────────────────────────────────────────────────────────
// PTX: .visible .entry
// PTX: .shared

// ─────────────────────────────────────────────────────────────────────────────
// Test function: 2D stencil (promoted path).
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
