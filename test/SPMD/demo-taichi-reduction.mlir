// demo-taichi-reduction.mlir
//
// End-to-end demo: Taichi Python → Taichi CHI IR → S0 SPMD IR (atomic_rmw)
// → RecognizeStructuredReductions → spmd.reduce
// → ReduceToHierarchicalGPU + SPMDToGPU → GPU IR.
//
// ─── Step 1: Taichi Python source ────────────────────────────────────────────
//
//   @ti.kernel
//   def array_sum(x: ti.template(), result: ti.template()):
//       for i in x:
//           result[None] += x[i]
//
// ─── Step 2: Taichi CHI IR (internal; our translator reads this) ─────────────
//
//   RangeFor $i in [0, N):
//     $ptr_x      = GlobalPtrStmt(x, [$i])          # depends on loop iv
//     $val        = LoadStmt($ptr_x)
//     $ptr_result = GlobalPtrStmt(result, [])        # loop-invariant (no iv)
//     $atomic     = AtomicOpStmt(add, $ptr_result, $val)
//
//   Key observation at this layer:
//   - $ptr_result has NO dependency on $i  → loop-invariant destination
//   - op = add (associative + commutative)
//   - $val is purely local
//   → Our translator can directly emit spmd.reduce here, OR emit atomic_rmw
//     and let RecognizeStructuredReductions handle it.
//
// ─── Step 3: S0 SPMD IR (what the Taichi translator emits) ───────────────────
//
//   spmd.forall (%i) in (%N) {
//     %v = memref.load %x[%i]
//     memref.atomic_rmw addf %v, %result[] ← loop-invariant dest, addf
//   }
//
// RecognizeStructuredReductions sees:
//   • atomic_rmw inside spmd.forall
//   • dest %result[] is loop-invariant (zero-dimensional memref, no iv dependency)
//   • op = addf  (associative + commutative)
//   • no other writes to %result[] in the loop body
// → safe to promote to spmd.reduce
//
// ─── Stage 1: after RecognizeStructuredReductions ────────────────────────────
//
//   spmd.forall (%tile_start) in (%N) step 4 [group] {
//     %sum = spmd.reduce (%i) = (0) to (4) step (1) init(0.0) [add] {
//       %gidx = tile_start + i
//       %v    = load %x[%gidx]
//       spmd.yield %v
//     }
//     atomic_rmw addf %sum, %result[]   ← one atomic per block (final flush)
//   }
//
// ─── Stage 2: after ReduceToHierarchicalGPU + SPMDToGPU ──────────────────────
//
//   gpu.launch blocks(...) threads(4, 1, 1)
//     workgroup(%scratch : memref<4xf32, workgroup>) {
//       // Phase 1: each thread accumulates its elements locally
//       scf.for %i = tx to 4 step 4 { partial += x[tile+i] }
//       scratch[tx] = partial
//       gpu.barrier
//       // Phase 2: tree reduction in workgroup memory
//       scf.if tx < 2 { scratch[tx] += scratch[tx+2] }
//       gpu.barrier
//       scf.if tx < 1 { scratch[tx] += scratch[tx+1] }
//       gpu.barrier
//       // Phase 3: thread 0 flushes to global
//       scf.if tx == 0 { atomic_rmw addf scratch[0], result[] }
//     }
//
// Key insight: without spmd.reduce, the compiler only sees N atomic writes
// to the same address and cannot safely apply the hierarchical pattern.
// With spmd.reduce, the combiner (add) and accumulator are explicit → legal.
//
// ─────────────────────────────────────────────────────────────────────────────
//
// STAGE 1 TEST: verify RecognizeStructuredReductions fires.
// (pass not yet implemented — this test documents the expected transformation)
//
// RUN-TODO: spmd-opt %s --recognize-structured-reductions \
// RUN-TODO:   | FileCheck %s --check-prefix=RECOG
//
// RECOG-LABEL: func @taichi_array_sum_stage1
// RECOG-NOT:   memref.atomic_rmw addf {{.*}} memref<f32>
// RECOG:       spmd.reduce
// RECOG:       spmd.kind = #spmd.reduction_kind<add>
// RECOG:       memref.atomic_rmw addf {{.*}} memref<f32>
//
// STAGE 2 TEST: verify hierarchical GPU lowering fires on the promoted form.
//
// RUN: spmd-opt %s --normalize-spmd --convert-spmd-to-gpu \
// RUN:   | FileCheck %s --check-prefix=GPU
//
// GPU-LABEL: func @taichi_array_sum_stage2
// GPU:        gpu.launch
// GPU-SAME:   workgroup({{.*}} : memref<4xf32, #gpu.address_space<workgroup>>)
// GPU-NOT:    spmd.reduce
// GPU:        scf.for
// GPU:        gpu.barrier
// GPU:        arith.cmpi eq
// GPU:        scf.if
// GPU-COUNT-1: memref.atomic_rmw addf
// GPU-NOT:     memref.atomic_rmw addf

// ─────────────────────────────────────────────────────────────────────────────
// Stage 0 IR: raw Taichi translator output (atomic_rmw, no spmd.reduce)
// Corresponds to Step 3: direct lowering of CHI AtomicOpStmt without analysis.
// This is what RecognizeStructuredReductions receives as input.
// ─────────────────────────────────────────────────────────────────────────────

func.func @taichi_array_sum_stage0(%x: memref<?xf32>, %result: memref<f32>, %N: index)
    attributes {spmd.kernel} {
  %c0   = arith.constant 0 : index
  %c1   = arith.constant 1 : index

  // RangeFor → spmd.forall (tile_sizes not yet assigned; comes from scheduler)
  "spmd.forall"(%c0, %N, %c1) ({
  ^bb0(%i: index):
    // GlobalLoad → memref.load
    %v = memref.load %x[%i] : memref<?xf32>
    // AtomicOpStmt(add, result[None], x[i])
    // dest = %result[] — zero-dimensional, no loop iv dependency → loop-invariant
    // op  = addf        — associative + commutative
    // RecognizeStructuredReductions fires here.
    memref.atomic_rmw addf %v, %result[] : (f32, memref<f32>) -> f32
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>}
     : (index, index, index) -> ()
  func.return
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage 1: after RecognizeStructuredReductions
// atomic_rmw inside forall → spmd.reduce; one atomic_rmw remains for the
// final per-block flush to the global accumulator.
// ─────────────────────────────────────────────────────────────────────────────

func.func @taichi_array_sum_stage1(%x: memref<?xf32>, %result: memref<f32>, %N: index)
    attributes {spmd.kernel} {
  %c0   = arith.constant 0 : index
  %c1   = arith.constant 1 : index
  %c4   = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32

  "spmd.forall"(%c0, %N, %c4) ({
  ^bb0(%tile_start: index):
    // Promoted: the loop body is now a structured spmd.reduce.
    // The reduction domain [0, 4) matches the tile size (blockDim=4).
    %sum = "spmd.reduce"(%c0, %c4, %c1, %zero) ({
    ^bb1(%local_i: index):
      %gidx = arith.addi %tile_start, %local_i : index
      %in_bounds = arith.cmpi ult, %gidx, %N : index
      %safe_idx  = arith.select %in_bounds, %gidx, %c0 : index
      %loaded    = memref.load %x[%safe_idx] : memref<?xf32>
      %v         = arith.select %in_bounds, %loaded, %zero : f32
      "spmd.yield"(%v) : (f32) -> ()
    }) {"spmd.kind" = #spmd.reduction_kind<add>} : (index, index, index, f32) -> f32
    // One atomic per block: flush the block-level partial sum to global.
    memref.atomic_rmw addf %sum, %result[] : (f32, memref<f32>) -> f32
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.mapping"    = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 4>}
     : (index, index, index) -> ()
  func.return
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage 2: already-promoted form, fed directly to ReduceToHierarchicalGPU.
// Identical to stage1; exists as a separate entry point for the GPU test.
// ─────────────────────────────────────────────────────────────────────────────

func.func @taichi_array_sum_stage2(%x: memref<?xf32>, %result: memref<f32>, %N: index)
    attributes {spmd.kernel} {
  %c0   = arith.constant 0 : index
  %c1   = arith.constant 1 : index
  %c4   = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32

  "spmd.forall"(%c0, %N, %c4) ({
  ^bb0(%tile_start: index):
    %sum = "spmd.reduce"(%c0, %c4, %c1, %zero) ({
    ^bb1(%local_i: index):
      %gidx = arith.addi %tile_start, %local_i : index
      %in_bounds = arith.cmpi ult, %gidx, %N : index
      %safe_idx  = arith.select %in_bounds, %gidx, %c0 : index
      %loaded    = memref.load %x[%safe_idx] : memref<?xf32>
      %v         = arith.select %in_bounds, %loaded, %zero : f32
      "spmd.yield"(%v) : (f32) -> ()
    }) {"spmd.kind" = #spmd.reduction_kind<add>} : (index, index, index, f32) -> f32
    memref.atomic_rmw addf %sum, %result[] : (f32, memref<f32>) -> f32
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.mapping"    = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 4>}
     : (index, index, index) -> ()
  func.return
}
