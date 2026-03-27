// RUN: spmd-opt %s -promote-group-memory | FileCheck %s

// S2 stencil: after PromoteGroupMemory pass, the global load should be
// replaced by a cooperative copy into group memory followed by a barrier,
// and the compute should read from the tile buffer.

// CHECK-LABEL: func @stencil2d
func.func @stencil2d(%A: memref<?x?xf32>, %B: memref<?x?xf32>,
                      %N: index, %M: index)
    attributes {spmd.kernel} {
  %c0  = arith.constant 0  : index
  %c1  = arith.constant 1  : index
  %c8  = arith.constant 8  : index
  %c32 = arith.constant 32 : index
  %c34 = arith.constant 34 : index
  %c10 = arith.constant 10 : index

  spmd.forall (%ii, %jj) = (%c1, %c1) to (%N, %M) step (%c32, %c8)
      attributes {
        spmd.mapping = #spmd.level<group>,
        spmd.memory_policy = #spmd.memory_policy<prefer_group>
      } {
    spmd.forall (%ti, %tj) = (%c0, %c0) to (%c32, %c8) step (%c1, %c1)
        attributes {spmd.mapping = #spmd.level<lane>} {
      // These loads are candidates for promotion to group memory.
      // After pass: should become reads from tile buffer.
      %i  = arith.addi %ii, %ti : index
      %j  = arith.addi %jj, %tj : index
      %i1 = arith.addi %i, %c1 : index
      %j1 = arith.addi %j, %c1 : index
      // CHECK-NOT: memref.load %A
      %center = memref.load %A[%i,  %j ] : memref<?x?xf32>
      %right  = memref.load %A[%i,  %j1] : memref<?x?xf32>
      %down   = memref.load %A[%i1, %j ] : memref<?x?xf32>
      %t0 = arith.addf %center, %right : f32
      %t1 = arith.addf %t0, %down : f32
      memref.store %t1, %B[%i, %j] : memref<?x?xf32>
      spmd.yield
    }
    spmd.yield
  }
  func.return
}

// After pass, we expect:
// CHECK: memref.alloc() : memref<34x10xf32, #spmd.addr_space<group>>
// CHECK: spmd.forall{{.*}}spmd.mapping = #spmd.level<lane>
// CHECK: memref.load %A
// CHECK: memref.store{{.*}}#spmd.addr_space<group>
// CHECK: spmd.barrier {spmd.scope = #spmd.scope<group>}
// CHECK: spmd.forall{{.*}}spmd.mapping = #spmd.level<lane>
// CHECK: memref.load{{.*}}#spmd.addr_space<group>
