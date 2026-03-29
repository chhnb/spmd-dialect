// lower-to-gpu-nvptx-reduction.mlir — NVPTX codegen test: atomic reduction.
//
// REQUIRES: nvptx-registered-target
//
// Each thread atomically adds its element to a scalar accumulator.
// Uses the non-promoted pipeline (same as ewise).
//
// PTX content check: the reduction kernel entry must be present and must
// contain an atomic float add instruction.
// RUN: spmd-opt %s --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-gpu \
// RUN:   --gpu-kernel-outlining "--nvvm-attach-target=chip=sm_80" \
// RUN:   | mlir-opt --convert-gpu-to-nvvm \
// RUN:   --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm \
// RUN:   --convert-index-to-llvm --reconcile-unrealized-casts \
// RUN:   | spmd-opt --spmd-extract-gpu-module \
// RUN:   | mlir-translate --mlir-to-llvmir \
// RUN:   | llc --march=nvptx64 --mcpu=sm_80 -filetype=asm \
// RUN:   | FileCheck %s --check-prefix=PTX

// PTX: .visible .entry
// PTX: atom{{.*}}add.f32

// ─────────────────────────────────────────────────────────────────────────────
// Parallel reduction: each thread atomically adds A[i] to a scalar output.
// The scalar output must be zero-initialized by the caller before launch.
// N must be a multiple of TILE_SIZE (256) to avoid bounds issues.
// ─────────────────────────────────────────────────────────────────────────────

func.func @atomic_sum(%A: memref<?xf32>, %out: memref<f32>, %N: index)
    attributes {spmd.kernel} {
  %c0   = arith.constant 0 : index
  %c1   = arith.constant 1 : index
  "spmd.forall"(%c0, %N, %c1) ({
  ^bb0(%i: index):
    %v = memref.load %A[%i] : memref<?xf32>
    memref.atomic_rmw addf %v, %out[] : (f32, memref<f32>) -> f32
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.mapping" = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 256>}
     : (index, index, index) -> ()
  func.return
}
