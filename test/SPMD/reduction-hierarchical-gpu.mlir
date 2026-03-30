// reduction-hierarchical-gpu.mlir — AC-1 GPU IR check
//
// Verifies that ReduceToHierarchicalGPU fires and produces the expected
// GPU IR structure: workgroup buffer, scf.for, barrier(s), tx==0 guard,
// and a single atomic add.  No PTX backend required.
//
// RUN: spmd-opt %s --normalize-spmd --convert-spmd-to-gpu \
// RUN:   | FileCheck %s

// CHECK-LABEL: func @hierarchical_sum_small
// CHECK:       gpu.launch
// CHECK-SAME:  workgroup({{.*}} : memref<4xf32, #gpu.address_space<workgroup>>)
// CHECK-NOT:   spmd.reduce
// CHECK:       scf.for
// CHECK:       memref.store {{.*}} memref<4xf32, #gpu.address_space<workgroup>>
// CHECK:       gpu.barrier
// CHECK:       scf.if
// CHECK:       gpu.barrier
// CHECK:       arith.cmpi eq
// CHECK:       scf.if
// CHECK:       memref.atomic_rmw addf

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
