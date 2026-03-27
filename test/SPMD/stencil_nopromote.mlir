// RUN: spmd-opt %s --normalize-spmd --materialize-spmd-tiling \
// RUN:   --promote-group-memory --convert-spmd-to-scf | FileCheck %s

// Pipeline LLVM: full lowering to LLVM IR.
// RUN: spmd-opt %s --normalize-spmd --materialize-spmd-tiling \
// RUN:   --promote-group-memory --convert-spmd-to-scf \
// RUN:   --convert-scf-to-cf --convert-arith-to-llvm \
// RUN:   --finalize-memref-to-llvm --convert-func-to-llvm --convert-cf-to-llvm \
// RUN:   --reconcile-unrealized-casts \
// RUN:   | mlir-translate --mlir-to-llvmir \
// RUN:   | FileCheck %s --check-prefix=LLVM

// LLVM: define

// 1-D stencil with no_promotion policy.
// PromoteGroupMemory must NOT transform this kernel (no group alloc / barrier).
// After the full CPU pipeline, only scf ops remain.

// CHECK-LABEL: func @stencil1d_nopromote
// CHECK-NOT: spmd.
// CHECK-NOT: memref.alloc(){{.*}}#spmd.addr_space<group>
// CHECK-NOT: "spmd.barrier"
// CHECK: scf.for
// CHECK: memref.load
// CHECK: arith.addf

func.func @stencil1d_nopromote(%A: memref<?xf32>, %B: memref<?xf32>,
                                 %N: index)
    attributes {spmd.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Loop over interior points: i in [1, N-1).
  %Nm1 = arith.subi %N, %c1 : index
  "spmd.forall"(%c1, %Nm1, %c1) ({
  ^bb0(%i: index):
    %i1 = arith.addi %i, %c1 : index
    %center = memref.load %A[%i ] : memref<?xf32>
    %right  = memref.load %A[%i1] : memref<?xf32>
    %out    = arith.addf %center, %right : f32
    memref.store %out, %B[%i] : memref<?xf32>
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.memory_policy" = #spmd.memory_policy<no_promotion>}
     : (index, index, index) -> ()

  func.return
}
