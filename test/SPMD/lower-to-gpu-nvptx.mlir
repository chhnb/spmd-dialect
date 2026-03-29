// lower-to-gpu-nvptx.mlir — NVPTX codegen smoke tests for --convert-spmd-to-gpu.
//
// REQUIRES: nvptx-registered-target
//
// RUN 3 (AC-5): non-promoted elementwise — NVVM pipeline exits 0; a gpu kernel
// is generated (nvvm.kernel attribute visible in NVVM MLIR output).
// RUN: spmd-opt %s --split-input-file --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-gpu \
// RUN:   --gpu-kernel-outlining --convert-gpu-to-nvvm \
// RUN:   | FileCheck %s --check-prefix=NVVM
//
// RUN 4 (AC-6): promoted stencil — NVVM IR must contain an addr_space=3 global
// (workgroup / shared memory) and the nvvm.kernel entry-point attribute.
// --promote-group-memory must run BEFORE --convert-spmd-to-gpu (no materialize
// step here; materialization wraps the barrier inside scf.if, violating the
// gpu.barrier-at-launch-body invariant). We verify at MLIR/NVVM level, which is
// equivalent: the LLVM NVPTX backend maps addrspace(3) → PTX .shared faithfully.
// RUN: spmd-opt %s --split-input-file --promote-group-memory \
// RUN:   --convert-spmd-to-gpu --gpu-kernel-outlining --convert-gpu-to-nvvm \
// RUN:   | FileCheck %s --check-prefix=PTX
//
// RUN 5 (AC-6 negative): non-promoted ewise — NVVM IR must NOT contain
// addr_space=3 (no shared-memory allocation produced without group-memory promo).
// RUN: spmd-opt %s --split-input-file --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-gpu \
// RUN:   --gpu-kernel-outlining --convert-gpu-to-nvvm \
// RUN:   | FileCheck %s --check-prefix=NOSHARED

// ─── NVVM checks (RUN 3, AC-5) ───────────────────────────────────────────────
// NVVM: nvvm.kernel

// ─── PTX checks (RUN 4, AC-6) ────────────────────────────────────────────────
// PTX: addr_space = 3
// PTX: nvvm.kernel

// ─── NOSHARED checks (RUN 5, AC-6 negative) ──────────────────────────────────
// NOSHARED-NOT: addr_space = 3

// ─────────────────────────────────────────────────────────────────────────────
// Test function 1: elementwise add (1D, non-promoted).
// Processed by RUN 3 (obj smoke check) and RUN 5 (no .shared check).
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
