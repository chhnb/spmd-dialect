// AC-8.1 + AC-8.2: promoted stencil IR correctness and end-to-end executability.
//
// RUN line 1 (AC-8.1): promote and lower to SCF; FileCheck verifies the IR
// transformations required by the plan:
//   - group-memory alloc is inserted
//   - spmd.barrier is inserted between copy and compute foralls
//   - the compute forall loads from the tile buffer, not the original global memref
//
// RUN: spmd-opt %s --promote-group-memory \
// RUN:   --convert-spmd-to-openmp --convert-spmd-to-scf \
// RUN:   | FileCheck %s
//
// RUN line 2 (AC-8.2): full pipeline to object file via llc -filetype=obj.
// Succeeds iff all tools exit 0 (promoted kernel compiles to native code).
//
// RUN: spmd-opt %s --promote-group-memory \
// RUN:   --convert-spmd-to-openmp --convert-spmd-to-scf \
// RUN:   --spmd-erase-memory-spaces \
// RUN:   --convert-scf-to-cf --convert-arith-to-llvm \
// RUN:   --finalize-memref-to-llvm --convert-func-to-llvm --convert-cf-to-llvm \
// RUN:   --convert-openmp-to-llvm --reconcile-unrealized-casts \
// RUN:   | mlir-translate --mlir-to-llvmir \
// RUN:   | llc -filetype=obj -o /dev/null

// 2-D stencil in S2 form:
//   - outer group forall: tile_sizes=[32, 8], memory_policy=prefer_group
//   - inner lane forall: [0,32)×[0,8) step 1
//   - reads A[i,j], A[i,j+1], A[i+1,j]  →  minOffset=(0,0), maxOffset=(1,1)
//   - tile buffer: (32+1)×(8+1) = 33×9 elements (footprint = 33×9×4 = 1188 B < 49152 B)
//
// After --promote-group-memory the output must contain:
//   1. memref.alloc() : memref<33x9xf32, #spmd.addr_space<group>>
//   2. spmd.barrier {spmd.scope = #spmd.scope<group>} between copy and compute
//   3. compute forall loads from the tile buffer (not from %arg0)

// CHECK-LABEL: func @promoted_stencil
// 1. Tile buffer allocated in group addr space.
// CHECK: memref.alloc() : memref<33x9xf32, #spmd.addr_space<group>>
// 2. Cooperative copy loop: loads from original A, stores to tile buffer.
//    Lane foralls are lowered to scf.for by --convert-spmd-to-scf.
// CHECK: scf.for
// CHECK: memref.load %arg0
// CHECK: memref.store{{.*}}memref<33x9xf32
// 3. Group-scope barrier (promoted to omp.barrier by --convert-spmd-to-openmp).
// CHECK: omp.barrier
// 4. Compute loop: reads only from tile buffer (no load from original %arg0).
// CHECK: scf.for
// CHECK-NOT: memref.load %arg0
// CHECK: memref.load{{.*}}memref<33x9xf32

func.func @promoted_stencil(%A: memref<?x?xf32>, %B: memref<?x?xf32>,
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
