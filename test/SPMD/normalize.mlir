// RUN: spmd-opt %s --normalize-spmd | FileCheck %s

// Tests for NormalizeSPMD pass:
//   - NormalizeForallBounds: rewrite lb/step to 0/1 canonical form
//   - FoldSingleIterationDims (single-trip): fold exact-one-iteration dims
//   - FoldSingleIterationDims (zero-trip): erase zero-iteration foralls

// ── 1. Already canonical — no change ────────────────────────────────────────

// CHECK-LABEL: func @already_canonical
// CHECK:       spmd.forall
// CHECK-NOT:   arith.muli
func.func @already_canonical(%N: index) attributes {spmd.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  "spmd.forall"(%c0, %N, %c1) ({
  ^bb0(%i: index):
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>} : (index, index, index) -> ()
  func.return
}

// -----

// ── 2. Non-zero lb, non-unit step: normalize to [0, ceildiv(ub-lb,step)) s=1 ─

// CHECK-LABEL: func @normalize_lb_step
// New upper bound computed as ceildiv(ub-lb, step) — appears before the forall.
// CHECK:       arith.subi
// CHECK:       arith.ceildivsi
// The new forall starts at 0, step 1.
// CHECK:       spmd.forall
// (IV remapping ops are DCE'd when the test body has no uses of the IV)
func.func @normalize_lb_step(%N: index) attributes {spmd.kernel} {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  "spmd.forall"(%c2, %N, %c3) ({
  ^bb0(%i: index):
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>} : (index, index, index) -> ()
  func.return
}

// -----

// ── 3. Single-trip forall: ub=1, lb=0, step=1 → inline body once ────────────

// The forall is eliminated; body is inlined (dead constant IV DCE'd by rewriter).
// CHECK-LABEL: func @single_trip
// CHECK-NOT:   spmd.forall
// CHECK-NOT:   spmd.yield
func.func @single_trip() attributes {spmd.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  "spmd.forall"(%c0, %c1, %c1) ({
  ^bb0(%i: index):
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>} : (index, index, index) -> ()
  func.return
}

// -----

// ── 4. Zero-trip forall: ub <= lb → erase entirely, body not executed ────────

// The forall and its body are erased.
// CHECK-LABEL: func @zero_trip
// CHECK-NOT:   spmd.forall
// CHECK-NOT:   spmd.yield
func.func @zero_trip() attributes {spmd.kernel} {
  %c5 = arith.constant 5 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  // ub(3) <= lb(5): zero-trip
  "spmd.forall"(%c5, %c3, %c1) ({
  ^bb0(%i: index):
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>} : (index, index, index) -> ()
  func.return
}

// -----

// ── 5. Zero-trip: ub == lb exactly ───────────────────────────────────────────

// CHECK-LABEL: func @zero_trip_equal
// CHECK-NOT:   spmd.forall
func.func @zero_trip_equal() attributes {spmd.kernel} {
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  // ub(4) == lb(4): zero-trip
  "spmd.forall"(%c4, %c4, %c1) ({
  ^bb0(%i: index):
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>} : (index, index, index) -> ()
  func.return
}

// -----

// ── 6. 2-D: one dim single-trip, one dim multi-trip → lower-rank forall ──────

// CHECK-LABEL: func @partial_single_trip
// The single-trip dim (dim0: [0,1) s=1) is folded; a 1-D forall over dim1 remains.
// Pretty-printer emits single-bound form (no operandSegmentSizes in output).
// CHECK:       spmd.forall(%c0) to(%arg0) step(%c1)
// The 2-D form should no longer appear.
// CHECK-NOT:   spmd.forall(%c0, %c0)
func.func @partial_single_trip(%N: index) attributes {spmd.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  "spmd.forall"(%c0, %c0, %c1, %N, %c1, %c1) ({
  ^bb0(%i: index, %j: index):
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 2, 2, 2>} : (index, index, index, index, index, index) -> ()
  func.return
}
