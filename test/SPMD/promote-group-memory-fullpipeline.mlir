// promote-group-memory-fullpipeline.mlir
//
// AC-2.1: Full pipeline invariant test.
// After --promote-group-memory --convert-spmd-to-gpu:
//   (a) no group-address-space memref.alloc remains (converted and erased)
//   (b) gpu.workgroup attribution present (lowered from group-space alloc)
//   (c) gpu.launch present (lowered from group forall)
//
// RUN: spmd-opt %s --promote-group-memory --convert-spmd-to-gpu \
// RUN:   | FileCheck %s
// RUN: spmd-opt %s --promote-group-memory --convert-spmd-to-gpu \
// RUN:   -verify-diagnostics

// ─────────────────────────────────────────────────────────────────────────────
// Positive test: 2D stencil with prefer_group policy.
// After the full promote → gpu pipeline:
//   - Group-space alloc must be gone (converted to workgroup memory)
//   - gpu.workgroup attribution must appear
//   - gpu.launch must be present
// ─────────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func @stencil_full_pipeline
// After gpu lowering: no group-space alloc survives.
// CHECK-NOT: memref.alloc() : memref<{{.*}}#spmd.addr_space<group>>
// No SPMD group address space types remain.
// CHECK-NOT: #spmd.addr_space<group>
// The kernel is inside a gpu.launch block with a workgroup memory parameter.
// "gpu.launch ... workgroup(..." is the expected output form.
// CHECK: gpu.launch
// CHECK-SAME: workgroup(
// CHECK: gpu.terminator
func.func @stencil_full_pipeline(
    %A: memref<?x?xf32>, %B: memref<?x?xf32>, %N: index, %M: index)
    attributes {spmd.kernel} {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c8  = arith.constant 8 : index
  %c32 = arith.constant 32 : index

  // expected-remark@+2 {{promote-group-memory: promoting 1 memref(s), reuseCount=}}
  // expected-remark@+1 {{convert-spmd-to-gpu: gridDim=}}
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
