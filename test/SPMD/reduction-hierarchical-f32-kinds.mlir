// reduction-hierarchical-f32-kinds.mlir — AC-2.2
//
// ReduceToHierarchicalGPU only fires for f32 Add reductions.
// For any other kind (mul, max, min, etc.) it must fall back to
// ReduceToSCFForGPU: no workgroup buffer, plain scf.for.
//
// RUN: spmd-opt %s --normalize-spmd --convert-spmd-to-gpu \
// RUN:   | FileCheck %s

// Verify fallback for mul — no shared memory, falls back to scf.for.
// CHECK-LABEL: func @reduce_mul
// CHECK:       gpu.launch
// CHECK-NOT:   workgroup(
// CHECK:       scf.for
// CHECK-NOT:   workgroup(

// Verify fallback for max — same expectation.
// CHECK-LABEL: func @reduce_max
// CHECK:       gpu.launch
// CHECK-NOT:   workgroup(
// CHECK:       scf.for

func.func @reduce_mul(%A: memref<?xf32>, %out: memref<f32>, %N: index)
    attributes {spmd.kernel} {
  %c0   = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  "spmd.forall"(%c0, %N, %c256) ({
  ^bb0(%tile_start: index):
    %c0_i   = arith.constant 0 : index
    %c1     = arith.constant 1 : index
    %c256_i = arith.constant 256 : index
    %one    = arith.constant 1.0 : f32
    // spmd.kind = mul → L2 fails → ReduceToHierarchicalGPU skips → fallback
    %prod = "spmd.reduce"(%c0_i, %c256_i, %c1, %one) ({
    ^bb1(%local_i: index):
      %gidx = arith.addi %tile_start, %local_i : index
      %v = memref.load %A[%gidx] : memref<?xf32>
      "spmd.yield"(%v) : (f32) -> ()
    }) {"spmd.kind" = #spmd.reduction_kind<mul>} : (index, index, index, f32) -> f32
    memref.atomic_rmw addf %prod, %out[] : (f32, memref<f32>) -> f32
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.mapping" = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 256>}
     : (index, index, index) -> ()
  func.return
}

func.func @reduce_max(%A: memref<?xf32>, %out: memref<f32>, %N: index)
    attributes {spmd.kernel} {
  %c0   = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  "spmd.forall"(%c0, %N, %c256) ({
  ^bb0(%tile_start: index):
    %c0_i    = arith.constant 0 : index
    %c1      = arith.constant 1 : index
    %c256_i  = arith.constant 256 : index
    %neg_inf = arith.constant 0xFF800000 : f32
    // spmd.kind = max → L2 fails → fallback
    %mx = "spmd.reduce"(%c0_i, %c256_i, %c1, %neg_inf) ({
    ^bb1(%local_i: index):
      %gidx = arith.addi %tile_start, %local_i : index
      %v = memref.load %A[%gidx] : memref<?xf32>
      "spmd.yield"(%v) : (f32) -> ()
    }) {"spmd.kind" = #spmd.reduction_kind<max>} : (index, index, index, f32) -> f32
    memref.atomic_rmw addf %mx, %out[] : (f32, memref<f32>) -> f32
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.mapping" = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 256>}
     : (index, index, index) -> ()
  func.return
}
