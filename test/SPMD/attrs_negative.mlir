// RUN: spmd-opt %s -verify-diagnostics -split-input-file

// ---- Negative: #spmd.level<invalid_level> ----
// MLIR's enum parser emits two diagnostics when given an unknown enum case:
//   (1) "expected one of [...]" at the bad token
//   (2) "failed to parse ... parameter 'value'" at the closing >

// expected-error@+2 {{expected one of [seq, grid, group, lane, vector]}}
// expected-error@+1 {{failed to parse SPMD_LevelAttr parameter 'value'}}
func.func @bad_level(%N: index) attributes {"spmd.mapping" = #spmd.level<invalid_level>} {
  func.return
}

// -----

// ---- Negative: #spmd.reduction_kind<subtract> ----

// expected-error@+2 {{expected one of [add, mul, max, min, and, or, xor]}}
// expected-error@+1 {{failed to parse SPMD_ReductionKindAttr parameter 'value'}}
func.func @bad_reduction_kind(%N: index) attributes {"spmd.kind" = #spmd.reduction_kind<subtract>} {
  func.return
}
