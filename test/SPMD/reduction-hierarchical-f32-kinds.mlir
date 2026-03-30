// reduction-hierarchical-f32-kinds.mlir — AC-2.2
//
// ReduceToHierarchicalGPU only fires for f32 Add reductions.
// - Positive (Add): workgroup buffer present, pattern fires.
// - Negative (mul, max, min): no workgroup buffer, fallback scf.for.
//
// RUN: spmd-opt %s --normalize-spmd --convert-spmd-to-gpu \
// RUN:   | FileCheck %s

// Positive case: Add → hierarchical pattern fires, workgroup buffer present.
// CHECK-LABEL: func @reduce_add
// CHECK:       gpu.launch
// CHECK-SAME:  workgroup(
// CHECK-NOT:   spmd.reduce

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

// Verify fallback for min — same expectation.
// CHECK-LABEL: func @reduce_min
// CHECK:       gpu.launch
// CHECK-NOT:   workgroup(
// CHECK:       scf.for

func.func @reduce_add(%A: memref<?xf32>, %out: memref<f32>, %N: index)
    attributes {spmd.kernel} {
  %c0   = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  "spmd.forall"(%c0, %N, %c256) ({
  ^bb0(%tile_start: index):
    %c0_i   = arith.constant 0 : index
    %c1     = arith.constant 1 : index
    %c256_i = arith.constant 256 : index
    %zero   = arith.constant 0.0 : f32
    %sum = "spmd.reduce"(%c0_i, %c256_i, %c1, %zero) ({
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
      "spmd.tile_sizes" = array<i64: 256>}
     : (index, index, index) -> ()
  func.return
}

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

func.func @reduce_min(%A: memref<?xf32>, %out: memref<f32>, %N: index)
    attributes {spmd.kernel} {
  %c0   = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  "spmd.forall"(%c0, %N, %c256) ({
  ^bb0(%tile_start: index):
    %c0_i   = arith.constant 0 : index
    %c1     = arith.constant 1 : index
    %c256_i = arith.constant 256 : index
    %inf    = arith.constant 0x7F800000 : f32
    %mn = "spmd.reduce"(%c0_i, %c256_i, %c1, %inf) ({
    ^bb1(%local_i: index):
      %gidx = arith.addi %tile_start, %local_i : index
      %v = memref.load %A[%gidx] : memref<?xf32>
      "spmd.yield"(%v) : (f32) -> ()
    }) {"spmd.kind" = #spmd.reduction_kind<min>} : (index, index, index, f32) -> f32
    memref.atomic_rmw addf %mn, %out[] : (f32, memref<f32>) -> f32
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.mapping" = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 256>}
     : (index, index, index) -> ()
  func.return
}
