// lower-to-gpu.mlir — lit tests for --convert-spmd-to-gpu (IR checks only)
//
// Elementwise kernel: verify launch geometry, block-arg thread mapping,
// in-bounds guards, and no gpu.thread_id (non-promoted path).
// RUN: spmd-opt %s --split-input-file --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-gpu \
// RUN:   | FileCheck %s --check-prefix=GPU
//
// Promoted stencil: verify workgroup memory attribution, barrier placement
// at gpu.launch body level, and no residual memref.alloc.
// RUN: spmd-opt %s --split-input-file --promote-group-memory \
// RUN:   --convert-spmd-to-gpu \
// RUN:   | FileCheck %s --check-prefix=WG
//
// spmd.if + spmd.reduce: verify both are lowered to scf.if / scf.for
// inside the gpu.launch region with no residual spmd ops.
// RUN: spmd-opt %s --split-input-file --convert-spmd-to-gpu \
// RUN:   | FileCheck %s --check-prefix=IFRED
//
// NVVM IR check: verify gpu-kernel-outlining produces nvvm.kernel attribute
// and uses PTX sreg intrinsics for block/thread ids (no NVPTX target needed).
// RUN: spmd-opt %s --split-input-file --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-gpu \
// RUN:   --gpu-kernel-outlining "--nvvm-attach-target=chip=sm_80" \
// RUN:   --convert-gpu-to-nvvm \
// RUN:   | FileCheck %s --check-prefix=NVVM
//
// Negative: convert-spmd-to-scf before convert-spmd-to-gpu must not crash;
// all spmd ops are gone before the GPU pass runs.
// RUN: spmd-opt %s --split-input-file --convert-spmd-to-scf \
// RUN:   --convert-spmd-to-gpu \
// RUN:   | FileCheck %s --check-prefix=SCFGPU
//
// Negative: mlir-translate on a module with gpu.launch (before outlining)
// must fail, confirming the device pipeline requires outlining first.
// RUN: spmd-opt %s --split-input-file --normalize-spmd --plan-spmd-schedule \
// RUN:   --materialize-spmd-tiling --convert-spmd-to-gpu \
// RUN:   | not mlir-translate --mlir-to-llvmir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PREOUTLINE
//
// Negative: gpu-kernel-outlining without --convert-spmd-to-gpu leaves
// spmd.forall intact and produces no gpu.launch.
// RUN: spmd-opt %s --split-input-file --gpu-kernel-outlining \
// RUN:   | FileCheck %s --check-prefix=NOCONV

// ─── GPU checks: launch geometry + block-arg thread mapping ──────────────────
// GPU:       gpu.container_module
// GPU-LABEL: func @ewise
// GPU:       arith.ceildivui
// GPU:       gpu.launch
// GPU-SAME:  threads({{.*}}) in (%{{.*}} = %c32
// GPU-NOT:   gpu.thread_id
// GPU:       scf.if
// GPU:       scf.if
// GPU:       memref.load
// GPU:       gpu.terminator
// GPU-NOT:   spmd.forall

// ─── WG checks: workgroup memory attribution, barrier ordering, use-rebinding ─
// WG-LABEL: func @promoted_stencil
// WG:       gpu.launch
// WG-SAME:  workgroup({{.*}}#gpu.address_space<workgroup>
// WG:       scf.if
// WG:       memref.store {{.*}}#gpu.address_space<workgroup>
// Barrier must be at gpu.launch body level (NOT nested in scf.if):
// after the inner memref.store the next 2 lines close the two scf.ifs, then
// gpu.barrier appears unconditionally at the top-level launch body.
// WG-NEXT:  }
// WG-NEXT:  }
// WG-NEXT:  gpu.barrier memfence [#gpu.address_space<workgroup>]
// WG:       scf.if
// WG-NOT:   memref.alloc
// WG:       gpu.terminator

// ─── IFRED checks: spmd.if → scf.if, spmd.reduce → scf.for ──────────────────
// IFRED-LABEL: func @if_reduce_kernel
// IFRED:        gpu.launch
// IFRED-NOT:    spmd.if
// IFRED-NOT:    spmd.reduce
// IFRED:        scf.if
// IFRED:        scf.for
// IFRED:        gpu.terminator

// ─── NVVM checks: outline → nvvm IR structure ────────────────────────────────
// NVVM:         gpu.container_module
// NVVM-LABEL:   func @ewise
// NVVM:         gpu.launch_func
// NVVM:         gpu.module {{.*}} [#nvvm.target
// NVVM:         llvm.func
// NVVM-SAME:    nvvm.kernel
// NVVM:         nvvm.read.ptx.sreg.ctaid.x
// NVVM:         nvvm.read.ptx.sreg.tid.x

// ─── SCFGPU checks: no spmd ops remain after both passes, no crash ───────────
// SCFGPU-LABEL: func @ewise
// SCFGPU-NOT:   spmd.forall
// SCFGPU-NOT:   spmd.if
// SCFGPU-NOT:   spmd.reduce

// ─── PREOUTLINE checks: pre-outline translation fails as expected ─────────────
// Module with gpu.launch (before outlining+gpu-to-llvm) fails mlir-translate
// because the host func.func / scf / memref dialect ops are not LLVM-translatable.
// PREOUTLINE: error:

// ─── NOCONV checks: spmd.forall persists when conversion pass is absent ───────
// Without --convert-spmd-to-gpu, gpu-kernel-outlining is a no-op: spmd.forall
// remains and no gpu.launch is generated.
// NOCONV-LABEL: func @ewise
// NOCONV:       spmd.forall
// NOCONV-NOT:   gpu.launch

// ─────────────────────────────────────────────────────────────────────────────
// Test function 1: elementwise add (1D, non-promoted).
// After --normalize-spmd --plan-spmd-schedule --materialize-spmd-tiling this
// becomes a group forall [0, N) step 32 wrapping a lane forall [0, 32) step 1.
// ─────────────────────────────────────────────────────────────────────────────

func.func @ewise(%A: memref<?xf32>, %B: memref<?xf32>, %C: memref<?xf32>,
                  %N: index)
    attributes {spmd.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  "spmd.forall"(%c0, %N, %c1) ({
  ^bb0(%i: index):
    %a = memref.load %A[%i] : memref<?xf32>
    %b = memref.load %B[%i] : memref<?xf32>
    %c = arith.addf %a, %b : f32
    memref.store %c, %C[%i] : memref<?xf32>
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.mapping" = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 32>}
     : (index, index, index) -> ()
  func.return
}

// -----

// ─────────────────────────────────────────────────────────────────────────────
// Test function 2: 2D stencil (promoted path — has group-addr-space alloc +
// spmd.barrier after --promote-group-memory). Used by WG checks.
// ─────────────────────────────────────────────────────────────────────────────

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

// -----

// ─────────────────────────────────────────────────────────────────────────────
// Test function 3: spmd.if + spmd.reduce inside a GPU kernel.
// --convert-spmd-to-gpu must lower these via the greedy pattern rewrite
// (IfToSCFIfGPU, ReduceToSCFForGPU) without needing any pre-pass.
// Used by IFRED checks.
// ─────────────────────────────────────────────────────────────────────────────

func.func @if_reduce_kernel(%A: memref<?xf32>, %out: memref<f32>,
                             %N: index, %threshold: f32)
    attributes {spmd.kernel} {
  %c0    = arith.constant 0 : index
  %c1    = arith.constant 1 : index
  %c32   = arith.constant 32 : index
  %zero  = arith.constant 0.0 : f32

  "spmd.forall"(%c0, %N, %c32) ({
  ^bb0(%ii: index):

    "spmd.forall"(%c0, %c32, %c1) ({
    ^bb0(%ti: index):
      %i = arith.addi %ii, %ti : index
      %v = memref.load %A[%i] : memref<?xf32>
      %cond = arith.cmpf ogt, %v, %threshold : f32
      "spmd.if"(%cond) ({
        memref.store %threshold, %A[%i] : memref<?xf32>
        "spmd.yield"() : () -> ()
      }, {
        "spmd.yield"() : () -> ()
      }) : (i1) -> ()
      "spmd.yield"() : () -> ()
    }) {operandSegmentSizes = array<i32: 1, 1, 1>,
        "spmd.mapping" = #spmd.level<lane>} : (index, index, index) -> ()

    // spmd.reduce: compute per-group sum over 32 elements.
    %sum = "spmd.reduce"(%c0, %c32, %c1, %zero) ({
    ^bb0(%k: index):
      %v = memref.load %A[%k] : memref<?xf32>
      "spmd.yield"(%v) : (f32) -> ()
    }) {"spmd.kind" = #spmd.reduction_kind<add>} : (index, index, index, f32) -> f32
    memref.store %sum, %out[] : memref<f32>

    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.mapping" = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 32>}
     : (index, index, index) -> ()

  func.return
}
