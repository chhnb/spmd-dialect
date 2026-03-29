// lower-to-gpu-nvptx.mlir — NVPTX codegen smoke tests for --convert-spmd-to-gpu.
//
// REQUIRES: nvptx-registered-target
//
// RUN 3 (AC-5): non-promoted elementwise — full pipeline to NVPTX obj exits 0.
// RUN: spmd-opt %s --split-input-file --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-gpu \
// RUN:   --gpu-kernel-outlining \
// RUN:   --nvvm-attach-target="module=ewise_kernel chip=sm_80 O=3" \
// RUN:   --gpu.module(convert-gpu-to-nvvm,convert-nvvm-to-llvm) \
// RUN:   --gpu-to-llvm \
// RUN:   | mlir-translate --mlir-to-llvmir \
// RUN:   | llc --march=nvptx64 --mcpu=sm_80 -filetype=obj -o /dev/null
//
// RUN 4 (AC-6): promoted stencil — PTX text contains .shared and .visible .entry.
// RUN: spmd-opt %s --split-input-file --promote-group-memory \
// RUN:   --convert-spmd-to-gpu \
// RUN:   --gpu-kernel-outlining \
// RUN:   --nvvm-attach-target="module=promoted_stencil_kernel chip=sm_80 O=3" \
// RUN:   --gpu.module(convert-gpu-to-nvvm,convert-nvvm-to-llvm) \
// RUN:   --gpu-to-llvm \
// RUN:   | mlir-translate --mlir-to-llvmir \
// RUN:   | llc --march=nvptx64 --mcpu=sm_80 -filetype=asm \
// RUN:   | FileCheck %s --check-prefix=PTX

// ─── PTX checks (RUN 4) ───────────────────────────────────────────────────────
// PTX: .shared
// PTX: .visible .entry

// ─────────────────────────────────────────────────────────────────────────────
// Test function 1: elementwise add (1D, non-promoted).
// Processed by RUN 3.
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
// Test function 2: 2D stencil (promoted path).
// Processed by RUN 4 — PTX must contain .shared (workgroup memory) and
// .visible .entry (exported kernel entry).
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
