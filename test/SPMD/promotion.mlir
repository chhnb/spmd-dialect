// RUN: spmd-opt %s --promote-group-memory | FileCheck %s

// S2 stencil: after PromoteGroupMemory pass the global loads inside the inner
// lane forall are replaced by a cooperative-copy forall + barrier + compute
// forall that reads from the group-memory tile buffer.
//
// The outer forall has tile_sizes=[32,8] and memory_policy=prefer_group.
// Stencil pattern: loads A[i,j], A[i,j+1], A[i+1,j] → halo = (1,1)
//   tile buffer size = (32+1+1) × (8+1+1) = 34 × 10

// CHECK-LABEL: func @stencil2d
func.func @stencil2d(%A: memref<?x?xf32>, %B: memref<?x?xf32>,
                      %N: index, %M: index)
    attributes {spmd.kernel} {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c8  = arith.constant 8 : index
  %c32 = arith.constant 32 : index

  // Outer group forall: iterates over tiles.
  // Uses generic format so this test is robust to assemblyFormat changes.
  "spmd.forall"(%c0, %c0, %N, %M, %c32, %c8) ({
  ^bb0(%ii: index, %jj: index):

    // Inner lane forall: each lane handles one element.
    "spmd.forall"(%c0, %c0, %c32, %c8, %c1, %c1) ({
    ^bb0(%ti: index, %tj: index):
      %i  = arith.addi %ii, %ti : index
      %j  = arith.addi %jj, %tj : index
      %i1 = arith.addi %i,  %c1 : index
      %j1 = arith.addi %j,  %c1 : index
      // These three loads are candidates for promotion.
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

// After --promote-group-memory:
//   1. A tile buffer in group addr space is allocated.
//   2. A cooperative copy lane forall (load A → store tile) is inserted.
//   3. spmd.barrier with group scope is inserted.
//   4. The compute lane forall loads from the tile buffer instead of A.

// CHECK: memref.alloc() : memref<34x10xf32, #spmd.addr_space<group>>
// CHECK: "spmd.forall"{{.*}}"spmd.mapping" = #spmd.level<lane>
// CHECK: memref.load %A
// CHECK: memref.store{{.*}}memref<34x10xf32, #spmd.addr_space<group>>
// CHECK: "spmd.barrier"() {{"spmd.scope" = #spmd.scope<group>}}
// CHECK: "spmd.forall"{{.*}}"spmd.mapping" = #spmd.level<lane>
// CHECK-NOT: memref.load %A
// CHECK: memref.load{{.*}}memref<34x10xf32, #spmd.addr_space<group>>

// -----

// Negative: no_promotion policy → pass must not transform this kernel.

// CHECK-LABEL: func @no_promote
func.func @no_promote(%A: memref<?x?xf32>, %B: memref<?x?xf32>,
                       %N: index, %M: index)
    attributes {spmd.kernel} {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c8  = arith.constant 8 : index
  %c32 = arith.constant 32 : index

  "spmd.forall"(%c0, %c0, %N, %M, %c32, %c8) ({
  ^bb0(%ii: index, %jj: index):
    "spmd.forall"(%c0, %c0, %c32, %c8, %c1, %c1) ({
    ^bb0(%ti: index, %tj: index):
      %i = arith.addi %ii, %ti : index
      %j = arith.addi %jj, %tj : index
      %v = memref.load %A[%i, %j] : memref<?x?xf32>
      memref.store %v, %B[%i, %j] : memref<?x?xf32>
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

// With no_promotion, no group alloc should be inserted.
// CHECK-NOT: memref.alloc() : memref<{{.*}}#spmd.addr_space<group>>
// CHECK-NOT: "spmd.barrier"

// -----

// AC-8.2: End-to-end pipeline: group memory promotion → LLVM IR → object code.
//
// RUN: spmd-opt %s --promote-group-memory --convert-spmd-to-scf \
// RUN:   --convert-scf-to-cf --convert-arith-to-llvm \
// RUN:   --finalize-memref-to-llvm --convert-func-to-llvm --convert-cf-to-llvm \
// RUN:   --reconcile-unrealized-casts \
// RUN:   | mlir-translate --mlir-to-llvmir \
// RUN:   | llc -o /dev/null

// (No FileCheck prefix needed: the pipeline succeeds iff all tools exit 0.)

func.func @stencil_e2e(%A: memref<?xf32>, %B: memref<?xf32>, %N: index)
    attributes {spmd.kernel} {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c32 = arith.constant 32 : index

  // Outer group forall over tiles of size 32.
  "spmd.forall"(%c0, %N, %c32) ({
  ^bb0(%ii: index):

    // Inner lane forall: each lane handles one element within the tile.
    "spmd.forall"(%c0, %c32, %c1) ({
    ^bb0(%ti: index):
      %i  = arith.addi %ii, %ti : index
      %i1 = arith.addi %i,  %c1 : index
      %center = memref.load %A[%i ] : memref<?xf32>
      %right  = memref.load %A[%i1] : memref<?xf32>
      %out    = arith.addf %center, %right : f32
      memref.store %out, %B[%i] : memref<?xf32>
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
