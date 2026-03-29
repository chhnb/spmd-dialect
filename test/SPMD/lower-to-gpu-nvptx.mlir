// lower-to-gpu-nvptx.mlir — NVPTX codegen smoke tests (non-promoted path).
//
// REQUIRES: nvptx-registered-target
//
// RUN 3 (AC-5): non-promoted elementwise — pipeline exits 0; PTX kernel entry
// is generated (assembly attribute with .visible .entry present).
// RUN: spmd-opt %s --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-gpu \
// RUN:   --gpu-kernel-outlining "--nvvm-attach-target=chip=sm_80" \
// RUN:   | mlir-opt --convert-gpu-to-nvvm \
// RUN:   --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm \
// RUN:   --convert-index-to-llvm --reconcile-unrealized-casts \
// RUN:   "--gpu-module-to-binary=format=isa" \
// RUN:   | FileCheck %s --check-prefix=SMOKE
//
// RUN 5 (AC-6 negative): non-promoted ewise must NOT emit .shared PTX.
// RUN: spmd-opt %s --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-gpu \
// RUN:   --gpu-kernel-outlining "--nvvm-attach-target=chip=sm_80" \
// RUN:   | mlir-opt --convert-gpu-to-nvvm \
// RUN:   --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm \
// RUN:   --convert-index-to-llvm --reconcile-unrealized-casts \
// RUN:   "--gpu-module-to-binary=format=isa" \
// RUN:   | FileCheck %s --check-prefix=NOSHARED

// ─── SMOKE checks (RUN 3, AC-5) ──────────────────────────────────────────────
// SMOKE: assembly =
// SMOKE: .visible .entry

// ─── NOSHARED checks (RUN 5, AC-6 negative) ──────────────────────────────────
// NOSHARED-NOT: .shared

// ─────────────────────────────────────────────────────────────────────────────
// Test function: elementwise add (1D, non-promoted).
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
