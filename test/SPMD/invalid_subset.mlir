// RUN: spmd-opt %s --verify-spmd-kernel-subset -verify-diagnostics -split-input-file

// ---- VerifyKernelSubset: barrier in S0/S1 kernel ----
// spmd.barrier is allowed by BarrierOp::verify() when inside a group forall,
// but must be rejected by --verify-spmd-kernel-subset (it is an S2-only op).

func.func @s0_has_barrier(%N: index) attributes {spmd.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  "spmd.forall"(%c0, %N, %c1) ({
  ^bb0(%i: index):
    // expected-error@+1 {{spmd.barrier must not appear in S0/S1 kernel}}
    "spmd.barrier"() {"spmd.scope" = #spmd.scope<group>} : () -> ()
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.mapping" = #spmd.level<group>} : (index, index, index) -> ()
  func.return
}

// -----

// ---- VerifyKernelSubset: group-addr-space operand in S0 ----

// expected-error@+1 {{spmd.kernel function signature has group/private memref argument}}
func.func @s0_group_memref_operand(
    %A: memref<?xf32, #spmd.addr_space<group>>,
    %N: index) attributes {spmd.kernel} {
  func.return
}

// -----

// ---- VerifyKernelSubset: disallowed dialect (gpu) ----

func.func @s0_gpu_op(%N: index) attributes {spmd.kernel} {
  // expected-error@+1 {{dialect 'gpu' is not allowed in spmd.kernel}}
  "gpu.barrier"() : () -> ()
  func.return
}
