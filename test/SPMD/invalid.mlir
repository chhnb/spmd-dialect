// RUN: spmd-opt %s -verify-diagnostics -split-input-file

// ---- rank mismatch: lbs/ubs/steps lengths differ ----

func.func @bad_rank(%N: index, %M: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{lowerBounds, upperBounds, and steps must have equal length}}
  spmd.forall (%i) = (%c0) to (%N, %M) step (%c1) {
    spmd.yield
  }
  func.return
}

// -----

// ---- tile_sizes length != rank ----

func.func @bad_tile_sizes(%N: index, %M: index) {
  // expected-error@+1 {{spmd.tile_sizes length must equal rank}}
  spmd.forall (%i, %j) in (%N, %M)
      attributes {spmd.tile_sizes = array<i64: 32>} {
    spmd.yield
  }
  func.return
}

// -----

// ---- tile_sizes with non-positive value ----

func.func @bad_tile_sizes_zero(%N: index, %M: index) {
  // expected-error@+1 {{spmd.tile_sizes values must be positive}}
  spmd.forall (%i, %j) in (%N, %M)
      attributes {spmd.tile_sizes = array<i64: 32, 0>} {
    spmd.yield
  }
  func.return
}

// -----

// ---- spmd.order is not a permutation ----

func.func @bad_order(%N: index, %M: index) {
  // expected-error@+1 {{spmd.order must be a permutation}}
  spmd.forall (%i, %j) in (%N, %M)
      attributes {spmd.order = array<i64: 0, 0>} {
    spmd.yield
  }
  func.return
}

// -----

// ---- reduce: spmd.kind missing ----

func.func @reduce_no_kind(%N: index) {
  %c0   = arith.constant 0 : index
  %c1   = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  // expected-error@+1 {{spmd.kind attribute is required}}
  %r = spmd.reduce (%k) = (%c0) to (%N) step (%c1) init(%zero) {
    spmd.yield %zero : f32
  }
  func.return
}

// -----

// ---- reduce: init type mismatch ----

func.func @reduce_type_mismatch(%N: index) {
  %c0   = arith.constant 0 : index
  %c1   = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  // expected-error@+1 {{init type must match result type}}
  %r = spmd.reduce (%k) = (%c0) to (%N) step (%c1)
           init(%zero)
           attributes {spmd.kind = #spmd.reduction_kind<add>} {
    %v = arith.constant 0 : i32   // wrong type
    spmd.yield %v : i32
  }
  func.return
}

// -----

// ---- if: condition is not i1 ----

func.func @if_bad_cond(%c: index) {
  // expected-error@+1 {{condition must be i1}}
  spmd.if %c {
    spmd.yield
  }
  func.return
}

// -----

// ---- if: results but no else ----

func.func @if_no_else(%c: i1, %x: f32) {
  // expected-error@+1 {{else region required when op has results}}
  %r = spmd.if %c -> (f32) {
    spmd.yield %x : f32
  }
  func.return
}

// -----

// ---- barrier: outside group forall ----

func.func @barrier_no_group() {
  // expected-error@+1 {{spmd.barrier must be nested inside a spmd.forall with spmd.mapping = #spmd.level<group>}}
  spmd.barrier {spmd.scope = #spmd.scope<group>}
  func.return
}

// -----

// ---- VerifyKernelSubset: barrier in S0 kernel ----

func.func @s0_has_barrier(%N: index) attributes {spmd.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  spmd.forall (%i) = (%c0) to (%N) step (%c1)
      attributes {spmd.mapping = #spmd.level<group>} {
    // expected-error@+1 {{spmd.barrier must not appear in S0/S1 kernel}}
    spmd.barrier {spmd.scope = #spmd.scope<group>}
    spmd.yield
  }
  func.return
}
