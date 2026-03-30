// verify-spmd-invariants.mlir
//
// Lit tests for --verify-spmd-promotion-invariant.
//
// RUN: (spmd-opt %s --split-input-file --verify-spmd-promotion-invariant; true) \
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
    // expected-error@+1 {{memref.load uses a group-address-space memref after convert-spmd-to-gpu}}
    %v = memref.load %tile[%tx] : memref<32xf32, #spmd.addr_space<group>>
    // expected-error@+1 {{memref.store uses a group-address-space memref after convert-spmd-to-gpu}}
    memref.store %v, %tile[%tx] : memref<32xf32, #spmd.addr_space<group>>
    gpu.terminator
  }
  func.return
}

// -----

// Test 2b: negative -- dangling group-space load (alloc was partially erased
// but the load still references the group-space type).
func.func @dangling_group_load(%A: memref<?xf32>, %N: index) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %gx  = arith.ceildivui %N, %c32 : index
  // A group-space alloc that was NOT erased (partial conversion).
  %tile = memref.alloc() : memref<32xf32, #spmd.addr_space<group>>
  // expected-error@-1 {{group-address-space memref.alloc should not exist after convert-spmd-to-gpu}}
  gpu.launch blocks(%bx, %by, %bz) in (%gbx = %gx, %gby = %c1, %gbz = %c1)
             threads(%tx, %ty, %tz) in (%tbx = %c32, %tby = %c1, %tbz = %c1) {
    // Load from the group-space tile — dangling use.
    // expected-error@+1 {{memref.load uses a group-address-space memref after convert-spmd-to-gpu}}
    %v = memref.load %tile[%tx] : memref<32xf32, #spmd.addr_space<group>>
    gpu.terminator
  }
  func.return
}

// -----

// Test 2c: negative -- dangling group-space store.
func.func @dangling_group_store(%N: index) {
  %c0   = arith.constant 0 : index
  %c1   = arith.constant 1 : index
  %c32  = arith.constant 32 : index
  %zero = arith.constant 0.0 : f32
  %gx   = arith.ceildivui %N, %c32 : index
  // expected-error@+1 {{group-address-space memref.alloc should not exist after convert-spmd-to-gpu}}
  %tile = memref.alloc() : memref<32xf32, #spmd.addr_space<group>>
  gpu.launch blocks(%bx, %by, %bz) in (%gbx = %gx, %gby = %c1, %gbz = %c1)
             threads(%tx, %ty, %tz) in (%tbx = %c32, %tby = %c1, %tbz = %c1) {
    // expected-error@+1 {{memref.store uses a group-address-space memref after convert-spmd-to-gpu}}
    memref.store %zero, %tile[%tx] : memref<32xf32, #spmd.addr_space<group>>
    gpu.terminator
  }
  func.return
}

// -----

// Test 2d: positive -- gpu.launch with workgroup buffer AND gpu.barrier passes.
// PROM-LABEL: func @launch_with_barrier
func.func @launch_with_barrier(%A: memref<?xf32>,
                                %wg: memref<32xf32, #gpu.address_space<workgroup>>,
                                %N: index) {
  %c0   = arith.constant 0 : index
  %c1   = arith.constant 1 : index
  %c32  = arith.constant 32 : index
  %zero = arith.constant 0.0 : f32
  %gx   = arith.ceildivui %N, %c32 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gbx = %gx, %gby = %c1, %gbz = %c1)
             threads(%tx, %ty, %tz) in (%tbx = %c32, %tby = %c1, %tbz = %c1) {
    memref.store %zero, %wg[%tx] : memref<32xf32, #gpu.address_space<workgroup>>
    gpu.barrier
    %v = memref.load %wg[%tx] : memref<32xf32, #gpu.address_space<workgroup>>
    memref.store %v, %A[%tx] : memref<?xf32>
    gpu.terminator
  }
  func.return
}

// -----

// Test 4: negative -- function has both gpu.workgroup memref arg AND group alloc.
// This is the partial-conversion coexistence state: Check 4 fires (on the func)
// and Check 1 also fires (on the alloc).
// expected-error@+1 {{function has coexisting #gpu.address_space<workgroup> memrefs and #spmd.addr_space<group> allocs}}
func.func @partial_coexistence(%wg: memref<32xf32, #gpu.address_space<workgroup>>,
                                %N: index) {
  // expected-error@+1 {{group-address-space memref.alloc should not exist after convert-spmd-to-gpu}}
  %stale = memref.alloc() : memref<32xf32, #spmd.addr_space<group>>
  func.return
}

// -----

// Test 5: negative -- gpu.launch uses workgroup memref but has no gpu.barrier.
// Check 5 fires: cooperative-barrier post-condition violated.
func.func @launch_no_barrier(%A: memref<?xf32>,
                              %wg: memref<32xf32, #gpu.address_space<workgroup>>,
                              %N: index) {
  %c1  = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %gx  = arith.ceildivui %N, %c32 : index
  // expected-error@+1 {{gpu.launch uses #gpu.address_space<workgroup> memrefs but contains no gpu.barrier}}
  gpu.launch blocks(%bx, %by, %bz) in (%gbx = %gx, %gby = %c1, %gbz = %c1)
             threads(%tx, %ty, %tz) in (%tbx = %c32, %tby = %c1, %tbz = %c1) {
    // No gpu.barrier — cooperative-barrier post-condition violated.
    %v = memref.load %wg[%tx] : memref<32xf32, #gpu.address_space<workgroup>>
    memref.store %v, %A[%tx] : memref<?xf32>
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
