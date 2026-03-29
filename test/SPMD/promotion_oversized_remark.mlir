// RUN: spmd-opt %s --promote-group-memory -verify-diagnostics

// Negative: oversized tile footprint — promote-group-memory must emit a remark
// and skip promotion.
//
// 2-D stencil: reads A[i,j], A[i,j+1], A[i+1,j]  →  maxOffset=(1,1)
// tile_size = [128, 128],  step = 1,  extent = 129 × 129
// footprintBytes = 129 × 129 × 4 = 66,564 > maxGroupMemBytes (49,152)
//
// The reuse gate PASSES (maxOffset > minOffset in both dims).
// The footprint gate fires: pass emits a remark and skips promotion.

func.func @stencil2d_oversized(%A: memref<?x?xf32>, %B: memref<?x?xf32>,
                                %N: index, %M: index)
    attributes {spmd.kernel} {
  %c0   = arith.constant 0   : index
  %c1   = arith.constant 1   : index
  %c128 = arith.constant 128 : index

  // expected-remark@+1 {{promote-group-memory: skipping}}
  "spmd.forall"(%c0, %c0, %N, %M, %c128, %c128) ({
  ^bb0(%ii: index, %jj: index):
    "spmd.forall"(%c0, %c0, %c128, %c128, %c1, %c1) ({
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
      "spmd.tile_sizes" = array<i64: 128, 128>,
      "spmd.memory_policy" = #spmd.memory_policy<prefer_group>}
     : (index, index, index, index, index, index) -> ()

  func.return
}
