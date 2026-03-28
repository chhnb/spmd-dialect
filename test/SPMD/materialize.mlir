// RUN: spmd-opt %s --materialize-spmd-tiling | FileCheck %s

// Tests for MaterializeTilingAndMapping pass.
//
// Key correctness property:
//   orig_iv = outer_iv + inner_iv * original_step
//
//   - step == 1 (common post-NormalizeSPMD): no MulIOp for IV computation
//   - step == 2: inner IV is scaled by MulIOp before AddIOp

// ── 1. Unit step: orig_iv = outer_iv + inner_iv (no IV MulIOp) ───────────────
//
// CHECK-LABEL: func @unit_step
// Outer group forall: outer_step = tile_size(4) * orig_step(1) = 4.
// CHECK:       spmd.forall{{.*}}spmd.mapping = #spmd.level<group>
// Inner lane forall: [0, 4) step 1.
// CHECK:       spmd.forall{{.*}}spmd.mapping = #spmd.level<lane>
// With step=1 the IV compute is a plain addi; no muli emitted for the IV.
// CHECK:       arith.addi
// CHECK:       arith.cmpi ult
func.func @unit_step(%A: memref<?xf32>, %B: memref<?xf32>, %N: index)
    attributes {spmd.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  "spmd.forall"(%c0, %N, %c1) ({
  ^bb0(%i: index):
    %v = memref.load %A[%i] : memref<?xf32>
    memref.store %v, %B[%i] : memref<?xf32>
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.mapping" = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 4>} : (index, index, index) -> ()
  func.return
}

// -----

// ── 2. Non-unit step (s=2): orig_iv = outer_iv + inner_iv * 2 ─────────────────
//
// Input forall: [0, N) step 2, tile_size=4
//   outer_step = 4 * 2 = 8  (constant-folded by rewriter)
//   inner forall: [0, 4) step 1
//   tile element k → global offset k*2 from tile origin
//
// CHECK-LABEL: func @nonunit_step
// Outer forall step is the constant 8 (4*2, folded).
// CHECK:       arith.constant 8
// CHECK:       spmd.forall{{.*}}spmd.mapping = #spmd.level<group>
// Inner lane forall: [0, 4) step 1.
// CHECK:       spmd.forall{{.*}}spmd.mapping = #spmd.level<lane>
// Scaled inner IV: arith.muli %inner_iv, %c2 (step=2).
// CHECK:       arith.muli
// orig_iv = outer_iv + scaled_inner.
// CHECK:       arith.addi
// Boundary guard.
// CHECK:       arith.cmpi ult
func.func @nonunit_step(%A: memref<?xf32>, %B: memref<?xf32>, %N: index)
    attributes {spmd.kernel} {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  "spmd.forall"(%c0, %N, %c2) ({
  ^bb0(%i: index):
    %v = memref.load %A[%i] : memref<?xf32>
    memref.store %v, %B[%i] : memref<?xf32>
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.mapping" = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 4>} : (index, index, index) -> ()
  func.return
}
