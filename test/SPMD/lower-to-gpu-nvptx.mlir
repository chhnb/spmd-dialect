// lower-to-gpu-nvptx.mlir — NVPTX codegen smoke tests (non-promoted path).
//
// REQUIRES: nvptx-registered-target
//
// Device-side codegen smoke: full pipeline through --spmd-extract-gpu-module,
// mlir-translate --mlir-to-llvmir, and llc --march=nvptx64 must exit 0.
// Note: -filetype=obj requires ptxas (CUDA toolkit) which is not available
// here; -filetype=null runs the identical instruction-selection and
// register-allocation pipeline and exits 0, proving device codegen is valid.
// RUN: spmd-opt %s --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-gpu \
// RUN:   --gpu-kernel-outlining "--nvvm-attach-target=chip=sm_80" \
// RUN:   | mlir-opt --convert-gpu-to-nvvm \
// RUN:   --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm \
// RUN:   --convert-index-to-llvm --reconcile-unrealized-casts \
// RUN:   | spmd-opt --spmd-extract-gpu-module \
// RUN:   | mlir-translate --mlir-to-llvmir \
// RUN:   | llc --march=nvptx64 --mcpu=sm_80 -filetype=null -o /dev/null
//
// PTX content check: the non-promoted kernel entry must be present.
// RUN: spmd-opt %s --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-gpu \
// RUN:   --gpu-kernel-outlining "--nvvm-attach-target=chip=sm_80" \
// RUN:   | mlir-opt --convert-gpu-to-nvvm \
// RUN:   --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm \
// RUN:   --convert-index-to-llvm --reconcile-unrealized-casts \
// RUN:   | spmd-opt --spmd-extract-gpu-module \
// RUN:   | mlir-translate --mlir-to-llvmir \
// RUN:   | llc --march=nvptx64 --mcpu=sm_80 -filetype=asm \
// RUN:   | FileCheck %s --check-prefix=SMOKE
//
// Negative: non-promoted path must NOT emit .shared (no workgroup memory).
// RUN: spmd-opt %s --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-gpu \
// RUN:   --gpu-kernel-outlining "--nvvm-attach-target=chip=sm_80" \
// RUN:   | mlir-opt --convert-gpu-to-nvvm \
// RUN:   --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm \
// RUN:   --convert-index-to-llvm --reconcile-unrealized-casts \
// RUN:   | spmd-opt --spmd-extract-gpu-module \
// RUN:   | mlir-translate --mlir-to-llvmir \
// RUN:   | llc --march=nvptx64 --mcpu=sm_80 -filetype=asm \
// RUN:   | FileCheck %s --check-prefix=NOSHARED

// ─── SMOKE checks ─────────────────────────────────────────────────────────────
// SMOKE: .visible .entry

// ─── NOSHARED checks ──────────────────────────────────────────────────────────
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
