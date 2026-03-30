// lower-to-gpu-nvptx-hierarchical-reduction.mlir
//
// REQUIRES: nvptx-registered-target
//
// Source kernel uses spmd.reduce (the proper IR-level abstraction) feeding a
// global scalar accumulator via atomic_rmw. ReduceToHierarchicalGPU lowers
// this to a two-level hierarchical reduction:
//   - thread-strided local accumulation (scf.for per thread)
//   - shared-memory tree reduction (statically unrolled, log2(blockDim) steps)
//   - single global atomic per block (inside tx==0 guard)
//
// Pipeline: normalize → convert-spmd-to-gpu (no materialize, so the
// spmd.reduce stays at the group-forall body level and becomes a top-level
// op inside gpu.launch — required so gpu.barrier can be emitted at launch
// body level rather than inside a conditional).
//
// SRC check: verify spmd.reduce is present in the source before lowering.
// RUN: FileCheck --check-prefix=SRC %s < %s
// SRC: spmd.reduce
//
// GPU-level IR check: verify hierarchical lowering fired.
// RUN: spmd-opt %s --normalize-spmd --convert-spmd-to-gpu \
// RUN:   | FileCheck %s --check-prefix=GPU
//
// GPU-LABEL: gpu.launch
// GPU-SAME:  workgroup(
// GPU-NOT:   spmd.reduce
// GPU:       scf.for
// GPU:       gpu.barrier
// GPU:       arith.cmpi eq
//
// PTX-level check: shared memory, sync barriers, one atomic add.
// RUN: spmd-opt %s --normalize-spmd --convert-spmd-to-gpu \
// RUN:   --gpu-kernel-outlining "--nvvm-attach-target=chip=sm_80" \
// RUN:   | mlir-opt --convert-gpu-to-nvvm \
// RUN:   --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm \
// RUN:   --convert-index-to-llvm --reconcile-unrealized-casts \
// RUN:   | spmd-opt --spmd-extract-gpu-module \
// RUN:   | mlir-translate --mlir-to-llvmir \
// RUN:   | llc --march=nvptx64 --mcpu=sm_80 -filetype=asm \
// RUN:   | FileCheck %s --check-prefix=PTX
//
// PTX: .visible .entry
// PTX: .shared
// PTX: bar.sync
// PTX: atom{{.*}}add.f32

// ─────────────────────────────────────────────────────────────────────────────
// Source kernel: uses spmd.reduce to express the parallel reduction.
// The group forall (lb=0, ub=N, step=256) creates ceil(N/256) GPU blocks.
// Each block holds spmd.reduce over 256 local elements (indices 0..255).
// ReduceToHierarchicalGPU lowers the reduce to a shared-memory tree.
//
// Bounds safety: the reduce body clamps the global index to [0, N) so that
// out-of-range threads in the last block contribute zero (identity value).
// ─────────────────────────────────────────────────────────────────────────────

func.func @hierarchical_sum(%A: memref<?xf32>, %out: memref<f32>, %N: index)
    attributes {spmd.kernel} {
  %c0   = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  // One group forall iteration per 256-element tile → one GPU block per tile.
  "spmd.forall"(%c0, %N, %c256) ({
  ^bb0(%tile_start: index):
    %c0_i   = arith.constant 0 : index
    %c1     = arith.constant 1 : index
    %c256_i = arith.constant 256 : index
    %zero   = arith.constant 0.0 : f32
    // Reduce 256 local elements. Thread tx processes A[tile_start + tx].
    // Out-of-range accesses (last tile) are masked to contribute 0.
    %sum = "spmd.reduce"(%c0_i, %c256_i, %c1, %zero) ({
    ^bb1(%local_i: index):
      %gidx = arith.addi %tile_start, %local_i : index
      %in_bounds = arith.cmpi ult, %gidx, %N : index
      %safe_idx = arith.select %in_bounds, %gidx, %c0_i : index
      %loaded = memref.load %A[%safe_idx] : memref<?xf32>
      %v = arith.select %in_bounds, %loaded, %zero : f32
      "spmd.yield"(%v) : (f32) -> ()
    }) {"spmd.kind" = #spmd.reduction_kind<add>} : (index, index, index, f32) -> f32
    // One atomic per block (not one per thread as in the atomic-only baseline).
    memref.atomic_rmw addf %sum, %out[] : (f32, memref<f32>) -> f32
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.mapping" = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 256>}
     : (index, index, index) -> ()
  func.return
}
