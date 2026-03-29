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
// Note: ODS type constraint on $condition fires at parse time with
// "must be 1-bit signless integer", before the IfOp verifier check
// ("condition must be i1") can run.  Both enforce the same invariant;
// this test exercises the user-visible diagnostic path.

func.func @if_bad_cond(%idx: index) {
  // expected-error@+1 {{must be 1-bit signless integer}}
  "spmd.if"(%idx) ({
    "spmd.yield"() : () -> ()
  }, {
    "spmd.yield"() : () -> ()
  }) : (index) -> ()
  func.return
}

// -----

// ---- if: results but no else ----
// Note: the MLIR parser requires the op to declare exactly the number of
// regions its ODS regionList specifies (2 for spmd.if).  "expected 2 regions"
// fires at parse time, before the IfOp verifier ("else region required when
// op has results") can run.  Both enforce the same invariant.

func.func @if_no_else(%c: i1, %x: f32) {
  // expected-error@+1 {{expected 2 regions}}
  %r = "spmd.if"(%c) ({
    "spmd.yield"(%x) : (f32) -> ()
  }) : (i1) -> f32
  func.return
}

// -----

// ---- barrier: outside group forall ----

func.func @barrier_no_group() {
  // expected-error@+1 {{spmd.barrier must be nested inside a spmd.forall with spmd.mapping = #spmd.level<group>}}
  "spmd.barrier"() {"spmd.scope" = #spmd.scope<group>} : () -> ()
  func.return
}

// Note: the VerifyKernelSubset negative tests live in invalid_subset.mlir,
// which uses --verify-spmd-kernel-subset in its RUN line.
