// verify-spmd-invariants.mlir
//
// Lit tests for --verify-spmd-promotion-invariant.
//
// RUN: spmd-opt %s --split-input-file --verify-spmd-promotion-invariant \
// RUN:   | FileCheck %s --check-prefix=PROM
// RUN: spmd-opt %s --split-input-file --verify-spmd-promotion-invariant \
// RUN:   -verify-diagnostics
//
// See verify-spmd-gpu-ready.mlir for --verify-spmd-gpu-ready tests.

// Test 1: positive -- no group-space alloc in a post-GPU-lowering kernel.
// PROM-LABEL: func @clean_kernel
// PROM-NOT:   memref.alloc
func.func @clean_kernel(%A: memref<?xf32>, %B: memref<?xf32>, %N: index) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %gx  = arith.ceildivui %N, %c32 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gbx = %gx, %gby = %c1, %gbz = %c1)
             threads(%tx, %ty, %tz) in (%tbx = %c32, %tby = %c1, %tbz = %c1) {
    %ii = arith.muli %bx, %c32 : index
    %i  = arith.addi %ii, %tx  : index
    %inb = arith.cmpi ult, %i, %N : index
    scf.if %inb {
      %v = memref.load %A[%i] : memref<?xf32>
      memref.store %v, %B[%i] : memref<?xf32>
    }
    gpu.terminator
  }
  func.return
}

// -----

// Test 2: negative -- surviving group-space alloc triggers error.
func.func @stale_group_alloc(%A: memref<?xf32>, %N: index) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %gx  = arith.ceildivui %N, %c32 : index
  // expected-error@+1 {{group-address-space memref.alloc should not exist after convert-spmd-to-gpu}}
  %tile = memref.alloc() : memref<32xf32, #spmd.addr_space<group>>
  gpu.launch blocks(%bx, %by, %bz) in (%gbx = %gx, %gby = %c1, %gbz = %c1)
             threads(%tx, %ty, %tz) in (%tbx = %c32, %tby = %c1, %tbz = %c1) {
    %v = memref.load %tile[%tx] : memref<32xf32, #spmd.addr_space<group>>
    memref.store %v, %tile[%tx] : memref<32xf32, #spmd.addr_space<group>>
    gpu.terminator
  }
  func.return
}

// -----

// Test 3: negative -- two stale group allocs in one function; both reported.
func.func @two_stale_allocs(%N: index) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %gx  = arith.ceildivui %N, %c32 : index
  // expected-error@+1 {{group-address-space memref.alloc should not exist after convert-spmd-to-gpu}}
  %t0 = memref.alloc() : memref<32xf32,   #spmd.addr_space<group>>
  // expected-error@+1 {{group-address-space memref.alloc should not exist after convert-spmd-to-gpu}}
  %t1 = memref.alloc() : memref<32x8xf32, #spmd.addr_space<group>>
  gpu.launch blocks(%bx, %by, %bz) in (%gbx = %gx, %gby = %c1, %gbz = %c1)
             threads(%tx, %ty, %tz) in (%tbx = %c32, %tby = %c1, %tbz = %c1) {
    gpu.terminator
  }
  func.return
}
