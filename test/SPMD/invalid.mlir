// RUN: spmd-opt %s -verify-diagnostics -split-input-file

// ---- rank mismatch: lbs/ubs/steps lengths differ ----

func.func @bad_rank(%N: index, %M: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{lowerBounds, upperBounds, and steps must have equal length}}
  "spmd.forall"(%c0, %N, %M, %c1) ({
  ^bb0(%i: index):
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 2, 1>} : (index, index, index, index) -> ()
  func.return
}

// -----

// ---- tile_sizes length != rank ----

func.func @bad_tile_sizes(%N: index, %M: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{spmd.tile_sizes length must equal rank}}
  "spmd.forall"(%c0, %c0, %N, %M, %c1, %c1) ({
  ^bb0(%i: index, %j: index):
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 2, 2, 2>,
      "spmd.tile_sizes" = array<i64: 32>} : (index, index, index, index, index, index) -> ()
  func.return
}

// -----

// ---- tile_sizes with non-positive value ----

func.func @bad_tile_sizes_zero(%N: index, %M: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{spmd.tile_sizes values must be positive}}
  "spmd.forall"(%c0, %c0, %N, %M, %c1, %c1) ({
  ^bb0(%i: index, %j: index):
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 2, 2, 2>,
      "spmd.tile_sizes" = array<i64: 32, 0>} : (index, index, index, index, index, index) -> ()
  func.return
}

// -----

// ---- spmd.order is not a permutation ----

func.func @bad_order(%N: index, %M: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{spmd.order must be a permutation}}
  "spmd.forall"(%c0, %c0, %N, %M, %c1, %c1) ({
  ^bb0(%i: index, %j: index):
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 2, 2, 2>,
      "spmd.order" = array<i64: 0, 0>} : (index, index, index, index, index, index) -> ()
  func.return
}

// -----

// ---- reduce: spmd.kind missing ----

func.func @reduce_no_kind(%N: index) {
  %c0   = arith.constant 0 : index
  %c1   = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  // expected-error@+1 {{spmd.kind attribute is required}}
  %r = "spmd.reduce"(%c0, %N, %c1, %zero) ({
  ^bb0(%k: index):
    "spmd.yield"(%zero) : (f32) -> ()
  }) : (index, index, index, f32) -> f32
  func.return
}

// -----

// ---- reduce: init type mismatch ----

func.func @reduce_type_mismatch(%N: index) {
  %c0   = arith.constant 0 : index
  %c1   = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %izero = arith.constant 0 : i32
  // expected-error@+1 {{init type must match result type}}
  %r = "spmd.reduce"(%c0, %N, %c1, %zero) ({
  ^bb0(%k: index):
    "spmd.yield"(%izero) : (i32) -> ()
  }) {"spmd.kind" = #spmd.reduction_kind<add>} : (index, index, index, f32) -> i32
  func.return
}

// -----

// ---- if: condition is not i1 ----
// IfOp::verify() is the sole enforcer (ODS uses AnyType for $condition so
// verifyInvariantsImpl() does not check the type).  Use generic format so the
// parser accepts the index-typed operand; verification then fires the custom
// verifier which emits "condition must be i1".

func.func @if_bad_cond(%idx: index) {
  // expected-error@+1 {{condition must be i1}}
  "spmd.if"(%idx) ({
    "spmd.yield"() : () -> ()
  }, {
    "spmd.yield"() : () -> ()
  }) : (index) -> ()
  func.return
}

// -----

// ---- if: results but no else ----
// Provide an empty else region ({}) so the generic-format parser accepts two
// regions.  The verifier then checks the else-region invariant and emits
// "else region required when op has results".

func.func @if_no_else(%c: i1, %x: f32) {
  // expected-error@+1 {{else region required when op has results}}
  %r = "spmd.if"(%c) ({
    "spmd.yield"(%x) : (f32) -> ()
  }, {
  }) : (i1) -> f32
  func.return
}

// -----

// ---- reduce: yield type does not match result type ----
// init type matches result type (both i32), but the body yields f32 — wrong.

func.func @reduce_yield_type_mismatch(%N: index) {
  %c0    = arith.constant 0 : index
  %c1    = arith.constant 1 : index
  %izero = arith.constant 0 : i32
  %fzero = arith.constant 0.0 : f32
  // expected-error@+1 {{yielded type must match result type}}
  %r = "spmd.reduce"(%c0, %N, %c1, %izero) ({
  ^bb0(%k: index):
    "spmd.yield"(%fzero) : (f32) -> ()
  }) {"spmd.kind" = #spmd.reduction_kind<add>} : (index, index, index, i32) -> i32
  func.return
}

// -----

// ---- barrier: outside group forall ----

func.func @barrier_no_group() {
  // expected-error@+1 {{spmd.barrier must be nested inside a spmd.forall with spmd.mapping = #spmd.level<group>}}
  "spmd.barrier"() {"spmd.scope" = #spmd.scope<group>} : () -> ()
  func.return
}

// -----

// ---- barrier: nested inside lane-level forall only (no enclosing group) ----
// BarrierOp::verify() walks up the parent chain; a lane-level forall without
// an outer group forall must fail with the same diagnostic.

func.func @barrier_lane_only(%N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  "spmd.forall"(%c0, %N, %c1) ({
  ^bb0(%i: index):
    // expected-error@+1 {{spmd.barrier must be nested inside a spmd.forall with spmd.mapping = #spmd.level<group>}}
    "spmd.barrier"() {"spmd.scope" = #spmd.scope<group>} : () -> ()
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.mapping" = #spmd.level<lane>} : (index, index, index) -> ()
  func.return
}

// Note: the VerifyKernelSubset negative tests live in invalid_subset.mlir,
// which uses --verify-spmd-kernel-subset in its RUN line.
