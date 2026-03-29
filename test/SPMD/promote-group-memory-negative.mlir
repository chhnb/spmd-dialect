// promote-group-memory-negative.mlir
//
// Five new negative and boundary tests for --promote-group-memory.
//
// RUN: spmd-opt %s -split-input-file --promote-group-memory | FileCheck %s
// RUN: spmd-opt %s -split-input-file --promote-group-memory -verify-diagnostics

// ─────────────────────────────────────────────────────────────────────────────
// Test 1 (a): Non-affine index — inner IV multiplied by a dynamic arg.
// The analysis cannot decompose `%ti * %N` as outer_iv + inner_iv*step + const.
// Promotion must be skipped; no group alloc or barrier inserted.
// ─────────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func @nonaffine_index
// CHECK-NOT: memref.alloc() : memref<{{.*}}#spmd.addr_space<group>>
// CHECK-NOT: spmd.barrier
func.func @nonaffine_index(%A: memref<?xf32>, %B: memref<?xf32>, %N: index)
    attributes {spmd.kernel} {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c32 = arith.constant 32 : index

  // expected-remark@+1 {{promote-group-memory: skipping}}
  "spmd.forall"(%c0, %N, %c32) ({
  ^bb0(%ii: index):
    "spmd.forall"(%c0, %c32, %c1) ({
    ^bb0(%ti: index):
      // Non-affine: multiply inner IV by a dynamic argument.
      // The access summary analysis cannot decompose this.
      %idx = arith.muli %ti, %N : index
      %v   = memref.load %A[%idx] : memref<?xf32>
      memref.store %v, %B[%ti] : memref<?xf32>
      "spmd.yield"() : () -> ()
    }) {operandSegmentSizes = array<i32: 1, 1, 1>,
        "spmd.mapping" = #spmd.level<lane>} : (index, index, index) -> ()
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.mapping" = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 32>,
      "spmd.memory_policy" = #spmd.memory_policy<prefer_group>}
     : (index, index, index) -> ()
  func.return
}

// -----

// ─────────────────────────────────────────────────────────────────────────────
// Test 2 (b): Write conflict — lane body writes to the candidate memref.
// isReadOnly(A, laneForall) returns false; A is excluded from the plan.
// Promotion must be skipped; no group alloc or barrier inserted.
// ─────────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func @write_conflict
// CHECK-NOT: memref.alloc() : memref<{{.*}}#spmd.addr_space<group>>
// CHECK-NOT: spmd.barrier
func.func @write_conflict(%A: memref<?x?xf32>, %N: index, %M: index)
    attributes {spmd.kernel} {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c8  = arith.constant 8 : index
  %c32 = arith.constant 32 : index

  // expected-remark@+1 {{promote-group-memory: skipping}}
  "spmd.forall"(%c0, %c0, %N, %M, %c32, %c8) ({
  ^bb0(%ii: index, %jj: index):
    "spmd.forall"(%c0, %c0, %c32, %c8, %c1, %c1) ({
    ^bb0(%ti: index, %tj: index):
      %i  = arith.addi %ii, %ti : index
      %j  = arith.addi %jj, %tj : index
      %j1 = arith.addi %j,  %c1 : index
      // Load from A with stencil pattern (would normally be promotable).
      %v  = memref.load %A[%i, %j1] : memref<?x?xf32>
      // Write back to A — disqualifies A from promotion.
      memref.store %v, %A[%i, %j] : memref<?x?xf32>
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

// -----

// ─────────────────────────────────────────────────────────────────────────────
// Test 3 (c): memory_policy = no_promotion → skip with remark.
// The pass emits a remark and leaves the kernel completely unchanged.
// ─────────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func @explicit_no_promotion
// CHECK-NOT: memref.alloc() : memref<{{.*}}#spmd.addr_space<group>>
// CHECK-NOT: spmd.barrier
func.func @explicit_no_promotion(%A: memref<?x?xf32>, %B: memref<?x?xf32>,
                                   %N: index, %M: index)
    attributes {spmd.kernel} {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c8  = arith.constant 8 : index
  %c32 = arith.constant 32 : index

  // expected-remark@+1 {{promote-group-memory: skipping, memory_policy=no_promotion}}
  "spmd.forall"(%c0, %c0, %N, %M, %c32, %c8) ({
  ^bb0(%ii: index, %jj: index):
    "spmd.forall"(%c0, %c0, %c32, %c8, %c1, %c1) ({
    ^bb0(%ti: index, %tj: index):
      %i  = arith.addi %ii, %ti : index
      %j  = arith.addi %jj, %tj : index
      %j1 = arith.addi %j, %c1  : index
      %i1 = arith.addi %i, %c1  : index
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
      "spmd.memory_policy" = #spmd.memory_policy<no_promotion>}
     : (index, index, index, index, index, index) -> ()
  func.return
}

// -----

// ─────────────────────────────────────────────────────────────────────────────
// Test 4 (d): Footprint overflow — different configuration from the existing
// promotion_oversized_remark.mlir test.
// 1D kernel with tile_size=64: stencil reads A[i] and A[i+32].
// Extent = (64-1)*1 + 32 - 0 + 1 = 96 elements × 4 bytes = 384 B < limit.
// BUT: tile_size=4096 with stencil offset +1 → extent=4097 × 4 = 16,388 B OK.
// Use tile_size=4096, offset range [0, 4096]:
//   extent = (4096-1)*1 + 4096 - 0 + 1 = 8192 elements × 4 bytes = 32,768 B < 49,152.
// Use tile_size=4096, 2D stencil 4096×4 = 16,384 × 4 = 65,536 > 49,152.
// Actually use tile_size=[128, 64] with maxOffset=(1,1):
//   extent = (128-1+2) × (64-1+2) = 129 × 65 = 8385 × 4 = 33,540 B < 49,152.
// Let's use tile_size=[128, 128]:
//   extent = (128-1+2) × (128-1+2) = 129 × 129 = 16,641 × 4 = 66,564 > 49,152.
// → footprint overflow, remark emitted.
// ─────────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func @footprint_overflow_1d
// CHECK-NOT: memref.alloc() : memref<{{.*}}#spmd.addr_space<group>>
// CHECK-NOT: spmd.barrier
func.func @footprint_overflow_1d(%A: memref<?xf32>, %B: memref<?xf32>,
                                  %N: index) attributes {spmd.kernel} {
  %c0    = arith.constant 0    : index
  %c1    = arith.constant 1    : index
  %c4096 = arith.constant 4096 : index

  // tile_size=4096, stencil reads A[i], A[i+1], ..., A[i+4096].
  // extent = (4096-1)*1 + 4096 - 0 + 1 = 8192 elements × 4 B = 32,768 B < 49,152 B.
  // Not actually oversized for 1D — but if we access A[i+12288]:
  // extent = (4096-1)*1 + 12288 - 0 + 1 = 16,384 × 4 = 65,536 > 49,152.
  // expected-remark@+1 {{promote-group-memory: skipping}}
  "spmd.forall"(%c0, %N, %c4096) ({
  ^bb0(%ii: index):
    "spmd.forall"(%c0, %c4096, %c1) ({
    ^bb0(%ti: index):
      %i   = arith.addi %ii, %ti : index
      // Stencil with large offset: reads A[i] and A[i + 12288].
      %off = arith.constant 12288 : index
      %i2  = arith.addi %i, %off  : index
      %a0  = memref.load %A[%i]  : memref<?xf32>
      %a1  = memref.load %A[%i2] : memref<?xf32>
      %sum = arith.addf %a0, %a1  : f32
      memref.store %sum, %B[%i]  : memref<?xf32>
      "spmd.yield"() : () -> ()
    }) {operandSegmentSizes = array<i32: 1, 1, 1>,
        "spmd.mapping" = #spmd.level<lane>} : (index, index, index) -> ()
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.mapping" = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 4096>,
      "spmd.memory_policy" = #spmd.memory_policy<prefer_group>}
     : (index, index, index) -> ()
  func.return
}

// -----

// ─────────────────────────────────────────────────────────────────────────────
// Test 5 (e): Post-promotion invariant — after promote-group-memory, the
// group-address-space tile buffer is present. After convert-spmd-to-gpu
// converts it to a gpu.workgroup attribution, no group-space alloc should
// remain. This test checks the intermediate state: after promotion only, the
// tile buffer IS present (and the original global loads are gone).
// AC-2.1 positive test.
// ─────────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func @stencil_post_promotion_invariant
// After promotion: tile buffer alloc exists in group addr space.
// CHECK: memref.alloc() : memref<33x9xf32, #spmd.addr_space<group>>
// Cooperative copy forall present.
// CHECK: spmd.forall{{.*}}spmd.mapping = #spmd.level<lane>
// Barrier present (group scope).
// CHECK: spmd.barrier
// Compute forall reads from tile buffer, NOT from original A.
// CHECK-NOT: memref.load %arg0
// CHECK: memref.load{{.*}}memref<33x9xf32, #spmd.addr_space<group>>
func.func @stencil_post_promotion_invariant(
    %A: memref<?x?xf32>, %B: memref<?x?xf32>, %N: index, %M: index)
    attributes {spmd.kernel} {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c8  = arith.constant 8 : index
  %c32 = arith.constant 32 : index

  // expected-remark@+1 {{promote-group-memory: promoting}}
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
