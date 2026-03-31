// reduction-hierarchical-gpu.mlir — AC-1 GPU IR check
//
// Verifies that ReduceToHierarchicalGPU fires and produces the expected
// GPU IR structure: workgroup buffer, scf.for, barrier(s), tx==0 guard,
// and a single atomic add.  No PTX backend required.
//
// Also verifies the general-bounds fix: the scf.for upper bound must be the
// reduce op's ub (not blockDim) so that reductions where ub != blockDim
// execute the correct number of iterations.
//
// RUN: spmd-opt %s --normalize-spmd --convert-spmd-to-gpu \
// RUN:   | FileCheck %s

// CHECK-LABEL: func @hierarchical_sum_small
// CHECK:       gpu.launch
// CHECK-SAME:  workgroup({{.*}} : memref<4xf32, #gpu.address_space<workgroup>>)
// CHECK-NOT:   spmd.reduce
// CHECK:       scf.for
// CHECK:       memref.store {{.*}} memref<4xf32, #gpu.address_space<workgroup>>
// blockDim=4 → 3 barriers: 1 after scatter, 2 tree-reduction steps (strides 2, 1).
// CHECK:       gpu.barrier
// CHECK:       scf.if
// CHECK:       gpu.barrier
// CHECK:       scf.if
// CHECK:       gpu.barrier
// CHECK-NOT:   gpu.barrier
// tx==0 condition and guard:
// CHECK:       arith.cmpi eq
// CHECK:       scf.if
// Exactly one atomic per block (inside the tx==0 guard):
// CHECK-COUNT-1: memref.atomic_rmw addf
// CHECK-NOT:     memref.atomic_rmw addf

// Kernel: sum 4 elements per block with 4 threads.
// blockDim=4 produces a 2-step tree reduction (strides 2, 1).

func.func @hierarchical_sum_small(%A: memref<?xf32>, %out: memref<f32>, %N: index)
    attributes {spmd.kernel} {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  "spmd.forall"(%c0, %N, %c4) ({
  ^bb0(%tile_start: index):
    %c0_i  = arith.constant 0 : index
    %c1    = arith.constant 1 : index
    %c4_i  = arith.constant 4 : index
    %zero  = arith.constant 0.0 : f32
    %sum = "spmd.reduce"(%c0_i, %c4_i, %c1, %zero) ({
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
      "spmd.tile_sizes" = array<i64: 4>}
     : (index, index, index) -> ()
  func.return
}

// ── General-bounds regression ─────────────────────────────────────────────────
//
// blockDim=4 but reduce ub=8: each thread processes 2 elements (indices
// tx, tx+4).  The smem scratch buffer must still be blockDim=4 slots, but the
// scf.for upper bound must be the reduce's ub (8), not blockDim (4).
//
// CHECK-LABEL: func @hierarchical_sum_general_bounds
// CHECK:       gpu.launch
// CHECK-SAME:  workgroup({{.*}} : memref<4xf32, #gpu.address_space<workgroup>>)
// CHECK-NOT:   spmd.reduce
// Regression: scf.for upper bound is the reduce ub (8), not blockDim (4).
// CHECK:       arith.constant 8 : index
// CHECK:       scf.for
// CHECK:       memref.store {{.*}} memref<4xf32, #gpu.address_space<workgroup>>
// blockDim=4 → 2 tree steps (strides 2, 1) plus 1 scatter barrier = 3 barriers.
// CHECK:       gpu.barrier
// CHECK:       scf.if
// CHECK:       gpu.barrier
// CHECK:       scf.if
// CHECK:       gpu.barrier
// CHECK-NOT:   gpu.barrier
// CHECK:       arith.cmpi eq
// CHECK:       scf.if
// CHECK-COUNT-1: memref.atomic_rmw addf
// CHECK-NOT:     memref.atomic_rmw addf

func.func @hierarchical_sum_general_bounds(%A: memref<?xf32>, %out: memref<f32>,
                                            %N: index)
    attributes {spmd.kernel} {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  // blockDim = 4 (spmd.forall step), reduce ub = 8 (2 elements per thread).
  "spmd.forall"(%c0, %N, %c4) ({
  ^bb0(%tile_start: index):
    %c0_i  = arith.constant 0 : index
    %c1    = arith.constant 1 : index
    %c8_i  = arith.constant 8 : index
    %zero  = arith.constant 0.0 : f32
    %sum = "spmd.reduce"(%c0_i, %c8_i, %c1, %zero) ({
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
      "spmd.tile_sizes" = array<i64: 4>}
     : (index, index, index) -> ()
  func.return
}
