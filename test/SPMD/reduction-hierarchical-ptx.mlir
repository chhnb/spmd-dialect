// reduction-hierarchical-ptx.mlir — AC-1 PTX check
//
// REQUIRES: nvptx-registered-target
//
// Verifies that the hierarchical reduction emits correct NVPTX:
//   - .shared section (workgroup buffer)
//   - bar.sync (gpu.barrier)
//   - exactly the expected atomic add (atom...add.f32)
//
// RUN: spmd-opt %s --normalize-spmd --convert-spmd-to-gpu \
// RUN:   --gpu-kernel-outlining "--nvvm-attach-target=chip=sm_80" \
// RUN:   | mlir-opt --convert-gpu-to-nvvm \
// RUN:   --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm \
// RUN:   --convert-index-to-llvm --reconcile-unrealized-casts \
// RUN:   | spmd-opt --spmd-extract-gpu-module \
// RUN:   | mlir-translate --mlir-to-llvmir \
// RUN:   | llc --march=nvptx64 --mcpu=sm_80 -filetype=asm \
// RUN:   | FileCheck %s --check-prefix=PTX

// PTX: .visible .entry
// PTX: .shared
// PTX: bar.sync
// PTX: atom{{.*}}add.f32

// Kernel: 16 threads per block, one tile per block.
// blockDim=16 → 4-step tree reduction (strides 8, 4, 2, 1).

func.func @hierarchical_sum_ptx(%A: memref<?xf32>, %out: memref<f32>, %N: index)
    attributes {spmd.kernel} {
  %c0  = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  "spmd.forall"(%c0, %N, %c16) ({
  ^bb0(%tile_start: index):
    %c0_i  = arith.constant 0 : index
    %c1    = arith.constant 1 : index
    %c16_i = arith.constant 16 : index
    %zero  = arith.constant 0.0 : f32
    %sum = "spmd.reduce"(%c0_i, %c16_i, %c1, %zero) ({
    ^bb1(%local_i: index):
      %gidx = arith.addi %tile_start, %local_i : index
      %in_bounds = arith.cmpi ult, %gidx, %N : index
      %safe_idx  = arith.select %in_bounds, %gidx, %c0_i : index
      %loaded    = memref.load %A[%safe_idx] : memref<?xf32>
      %v         = arith.select %in_bounds, %loaded, %zero : f32
      "spmd.yield"(%v) : (f32) -> ()
    }) {"spmd.kind" = #spmd.reduction_kind<add>} : (index, index, index, f32) -> f32
    memref.atomic_rmw addf %sum, %out[] : (f32, memref<f32>) -> f32
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.mapping" = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 16>}
     : (index, index, index) -> ()
  func.return
}
