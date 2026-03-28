// RUN: spmd-opt %s --promote-group-memory | FileCheck %s

// S2 stencil: after PromoteGroupMemory pass the global loads inside the inner
// lane forall are replaced by a cooperative-copy forall + barrier + compute
// forall that reads from the group-memory tile buffer.
//
// The outer forall has tile_sizes=[32,8] and memory_policy=prefer_group.
// Stencil pattern: loads A[i,j], A[i,j+1], A[i+1,j] → maxOffset = (1,1)
//   tile buffer size = (32+1) × (8+1) = 33 × 9

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

// CHECK: memref.alloc() : memref<33x9xf32, #spmd.addr_space<group>>
// CHECK: spmd.forall{{.*}}spmd.mapping = #spmd.level<lane>
// CHECK: memref.load %arg0
// CHECK: memref.store{{.*}}memref<33x9xf32, #spmd.addr_space<group>>
// CHECK: spmd.barrier {spmd.scope = #spmd.scope<group>}
// CHECK: spmd.forall{{.*}}spmd.mapping = #spmd.level<lane>
// CHECK-NOT: memref.load %arg0
// CHECK: memref.load{{.*}}memref<33x9xf32, #spmd.addr_space<group>>

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
// CHECK-NOT: spmd.barrier

// -----

// Negative: prefer_group policy but no reuse (simple copy, offset always 0).
// The reuse/profitability gate must reject promotion because maxOffset ==
// minOffset == 0 in every dimension — there is no stencil-like element sharing.

// CHECK-LABEL: func @copy_no_reuse
func.func @copy_no_reuse(%A: memref<?x?xf32>, %B: memref<?x?xf32>,
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
      // Simple copy: each lane reads A[i,j] (offset 0 in both dims) — no reuse.
      %v = memref.load %A[%i, %j] : memref<?x?xf32>
      memref.store %v, %B[%i, %j] : memref<?x?xf32>
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

// No-reuse: PromoteGroupMemory must not insert a group alloc or barrier.
// CHECK-NOT: memref.alloc() : memref<{{.*}}#spmd.addr_space<group>>
// CHECK-NOT: spmd.barrier

// -----

// Positive: prefer_group kernel with step=2 (stepped stencil regression).
//
// outer forall: step=8 = tile_size(4) * origStep(2)
// inner forall: [0, 4) step 1 — lane i processes global index outer + i*2
// Stencil accesses: A[outer+i*2] and A[outer+i*2+1]
//   minOffset=0, maxOffset=1, step=2, tile_size=4
//   extent = (tile_size-1)*step + maxOffset - minOffset + 1
//          = (4-1)*2 + 1 - 0 + 1 = 8
//
// After --promote-group-memory the tile buffer must be memref<8xf32, group>
// (NOT memref<5xf32, group> which the old formula would produce).

// STEP-LABEL: func @stencil1d_stepped
// CHECK-LABEL: func @stencil1d_stepped
// CHECK: memref.alloc() : memref<8xf32, #spmd.addr_space<group>>
// Copy loop covers the full 8-element dense range.
// CHECK: spmd.forall{{.*}}to(%c8{{[^)]*}})
// CHECK: spmd.barrier {spmd.scope = #spmd.scope<group>}
// Compute forall reads from group tile, not from original A.
// CHECK: spmd.forall{{.*}}spmd.mapping = #spmd.level<lane>
// CHECK-NOT: memref.load %arg0
// CHECK: memref.load{{.*}}memref<8xf32, #spmd.addr_space<group>>
func.func @stencil1d_stepped(%A: memref<?xf32>, %B: memref<?xf32>, %N: index)
    attributes {spmd.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // Outer group forall: step=8 = tile_size(4) * origStep(2).
  "spmd.forall"(%c0, %N, %c8) ({
  ^bb0(%ii: index):
    // Inner lane forall: [0, 4) step 1; tile element k → global ii + k*2.
    "spmd.forall"(%c0, %c4, %c1) ({
    ^bb0(%ti: index):
      %scaled = arith.muli %ti, %c2 : index
      %orig   = arith.addi %ii, %scaled : index
      %orig1  = arith.addi %orig, %c1 : index
      // Stencil: two loads with offsets 0 and +1 from orig.
      %a0 = memref.load %A[%orig]  : memref<?xf32>
      %a1 = memref.load %A[%orig1] : memref<?xf32>
      %sum = arith.addf %a0, %a1 : f32
      memref.store %sum, %B[%orig] : memref<?xf32>
      "spmd.yield"() : () -> ()
    }) {operandSegmentSizes = array<i32: 1, 1, 1>,
        "spmd.mapping" = #spmd.level<lane>} : (index, index, index) -> ()
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.mapping" = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 4>,
      "spmd.memory_policy" = #spmd.memory_policy<prefer_group>}
     : (index, index, index) -> ()
  func.return
}

// -----

// AC-8.2: End-to-end pipeline for a 2-D S0 stencil kernel.
//
// Starts from unscheduled S0 IR (no spmd.mapping / tile_sizes / memory_policy)
// and runs the complete CPU pipeline:
//   normalize → plan → materialize → promote → convert-to-openmp →
//   convert-to-scf → [LLVM lowering] → mlir-translate → llc
//
// Pipeline succeeds iff all tools exit 0 (no FileCheck prefix needed).
//
// RUN: spmd-opt %s --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --promote-group-memory \
// RUN:   --convert-spmd-to-openmp --convert-spmd-to-scf \
// RUN:   --spmd-erase-memory-spaces \
// RUN:   --convert-scf-to-cf --convert-arith-to-llvm \
// RUN:   --finalize-memref-to-llvm --convert-func-to-llvm --convert-cf-to-llvm \
// RUN:   --convert-openmp-to-llvm --reconcile-unrealized-casts \
// RUN:   | mlir-translate --mlir-to-llvmir \
// RUN:   | llc -o /dev/null

// 2-D S0 stencil: B[i,j] = A[i,j] + A[i,j+1] + A[i+1,j]
// Loops over interior: i in [0, N-1), j in [0, M-1).
// PlanSPMDSchedule assigns the outer forall mapping=group and tile_sizes,
// MaterializeTilingAndMapping splits it, PromoteGroupMemory promotes A reads.
func.func @stencil2d_e2e(%A: memref<?x?xf32>, %B: memref<?x?xf32>,
                          %N: index, %M: index)
    attributes {spmd.kernel} {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %Nm1 = arith.subi %N, %c1 : index
  %Mm1 = arith.subi %M, %c1 : index

  // Pure S0: a single flat forall over all interior points.
  "spmd.forall"(%c0, %c0, %Nm1, %Mm1, %c1, %c1) ({
  ^bb0(%i: index, %j: index):
    %j1 = arith.addi %j, %c1 : index
    %i1 = arith.addi %i, %c1 : index
    %center = memref.load %A[%i,  %j ] : memref<?x?xf32>
    %right  = memref.load %A[%i,  %j1] : memref<?x?xf32>
    %down   = memref.load %A[%i1, %j ] : memref<?x?xf32>
    %t0 = arith.addf %center, %right : f32
    %t1 = arith.addf %t0,     %down  : f32
    memref.store %t1, %B[%i, %j] : memref<?x?xf32>
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 2, 2, 2>}
     : (index, index, index, index, index, index) -> ()

  func.return
}
