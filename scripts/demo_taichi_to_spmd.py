#!/usr/bin/env python3
"""
demo_taichi_to_spmd.py
======================
End-to-end demo: Taichi Python → Taichi CHI IR → S0 SPMD IR → GPU IR

Run on a GPU compute node:
    python demo_taichi_to_spmd.py

Requirements (already in spmd-venv):
    taichi==1.7.4

spmd-opt must be on PATH or set SPMD_OPT env var:
    export SPMD_OPT=/path/to/spmd-dialect/build/bin/spmd-opt
"""

import os
import sys
import subprocess
import textwrap

# Taichi and spmd-opt both require GLIBC 2.32+.
# On the login node (GLIBC 2.28) they will be skipped automatically.
# On a GPU compute node, both run and produce real output.

try:
    import taichi as ti
    HAS_TAICHI = True
except Exception:
    HAS_TAICHI = False

SPMD_OPT = os.environ.get("SPMD_OPT",
    os.path.join(os.path.dirname(__file__),
                 "spmd-dialect/build/bin/spmd-opt"))

SEP  = "─" * 72
SEP2 = "═" * 72

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Taichi Python source
# ─────────────────────────────────────────────────────────────────────────────

print(SEP2)
print("STEP 1 — Taichi Python source")
print(SEP2)

TAICHI_SOURCE = textwrap.dedent("""\
    @ti.kernel
    def array_sum(x: ti.template(), result: ti.template()):
        for i in x:              # parallel RangeFor over all elements
            result[None] += x[i] # AtomicOpStmt(add, result[None], x[i])
""")
print(TAICHI_SOURCE)

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Capture Taichi CHI IR
# ─────────────────────────────────────────────────────────────────────────────

print(SEP2)
print("STEP 2 — Taichi CHI IR  (ti.init print_ir=True)")
print(SEP2)

if not HAS_TAICHI:
    print("  [login node] Taichi unavailable (GLIBC 2.32 required).")
    print("  On compute node this prints the actual CHI IR.")
    print("  Example CHI IR for array_sum:")
    print(textwrap.dedent("""\
      [taichi] version 1.7.4
      ==> stmt list:
        $0  : AllocaStmt<float32>                 // result partial
        $1  : RangeFor (0 to N step 1)
          $2  : GlobalPtrStmt(x, [$1.loop_var])   // ptr to x[i]
          $3  : GlobalLoadStmt($2)                // load x[i]
          $4  : GlobalPtrStmt(result, [])         // ptr to result[None]  ← no loop var
          $5  : AtomicOpStmt(add, $4, $3)         // result[None] += x[i]
    """))
else:
    import io, contextlib, numpy as np
    ti.init(arch=ti.cpu, print_ir=True)
    N = 16
    x      = ti.field(dtype=ti.f32, shape=N)
    result = ti.field(dtype=ti.f32, shape=())
    x.from_numpy(np.ones(N, dtype="float32"))
    result[None] = 0.0

    chi_buf = io.StringIO()
    with contextlib.redirect_stdout(chi_buf):
        @ti.kernel
        def array_sum(x: ti.template(), result: ti.template()):
            for i in x:
                result[None] += x[i]
        array_sum(x, result)

    chi_ir = chi_buf.getvalue()
    print(chi_ir if chi_ir.strip() else
          "(CHI IR printed to stderr — see terminal output above)")
    print(f"  result = {result[None]}  (expected {float(N)})")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Annotated mapping  CHI IR → SPMD IR
# ─────────────────────────────────────────────────────────────────────────────

print(SEP2)
print("STEP 3 — CHI IR node mapping to SPMD IR")
print(SEP2)

MAPPING = textwrap.dedent("""\
  Taichi CHI IR node                    →  SPMD IR
  ─────────────────────────────────────────────────────────────────────
  RangeFor $i in [0, N)                 →  spmd.forall (%i) in (%N)
  GlobalPtrStmt(x, [$i])                →  (SSA val, used as memref idx)
  LoadStmt($ptr_x)                      →  memref.load %x[%i]
  GlobalPtrStmt(result, [])  ← no $i   →  %result (loop-invariant ptr)
  AtomicOpStmt(add, $ptr_result, $val)  →  memref.atomic_rmw addf %v, %result[]

  Key observation:
    $ptr_result has NO dependency on loop iv $i
    → RecognizeStructuredReductions promotes atomic_rmw to spmd.reduce
""")
print(MAPPING)

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: S0 SPMD IR  (what the Taichi translator emits — stage 0)
# ─────────────────────────────────────────────────────────────────────────────

print(SEP2)
print("STEP 4 — S0 SPMD IR  (raw translator output, atomic_rmw)")
print(SEP2)

STAGE0_IR = textwrap.dedent("""\
    func.func @array_sum(%x: memref<?xf32>, %result: memref<f32>, %N: index)
        attributes {spmd.kernel} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      "spmd.forall"(%c0, %N, %c1) ({
      ^bb0(%i: index):
        // LoadStmt
        %v = memref.load %x[%i] : memref<?xf32>
        // AtomicOpStmt — dest %result[] is loop-invariant
        memref.atomic_rmw addf %v, %result[] : (f32, memref<f32>) -> f32
        "spmd.yield"() : () -> ()
      }) {operandSegmentSizes = array<i32: 1, 1, 1>}
         : (index, index, index) -> ()
      func.return
    }
""")
print(STAGE0_IR)

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: S0 SPMD IR  (after RecognizeStructuredReductions — stage 1)
# ─────────────────────────────────────────────────────────────────────────────

print(SEP2)
print("STEP 5 — S0 SPMD IR after RecognizeStructuredReductions")
print("         atomic_rmw (loop-invariant dest) → spmd.reduce")
print(SEP2)

STAGE1_IR = textwrap.dedent("""\
    func.func @array_sum(%x: memref<?xf32>, %result: memref<f32>, %N: index)
        attributes {spmd.kernel} {
      %c0   = arith.constant 0 : index
      %c1   = arith.constant 1 : index
      %c4   = arith.constant 4 : index   // blockDim (from scheduler)
      %zero = arith.constant 0.0 : f32

      "spmd.forall"(%c0, %N, %c4) ({
      ^bb0(%tile_start: index):
        // *** PROMOTED: atomic_rmw → spmd.reduce ***
        %sum = "spmd.reduce"(%c0, %c4, %c1, %zero) ({
        ^bb1(%local_i: index):
          %gidx      = arith.addi %tile_start, %local_i : index
          %in_bounds = arith.cmpi ult, %gidx, %N : index
          %safe_idx  = arith.select %in_bounds, %gidx, %c0 : index
          %loaded    = memref.load %x[%safe_idx] : memref<?xf32>
          %v         = arith.select %in_bounds, %loaded, %zero : f32
          "spmd.yield"(%v) : (f32) -> ()
        }) {"spmd.kind" = #spmd.reduction_kind<add>}
           : (index, index, index, f32) -> f32
        // one atomic per block: flush partial sum to global accumulator
        memref.atomic_rmw addf %sum, %result[] : (f32, memref<f32>) -> f32
        "spmd.yield"() : () -> ()
      }) {operandSegmentSizes = array<i32: 1, 1, 1>,
          "spmd.mapping"    = #spmd.level<group>,
          "spmd.tile_sizes" = array<i64: 4>}
         : (index, index, index) -> ()
      func.return
    }
""")
print(STAGE1_IR)

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: run spmd-opt → GPU IR  (ReduceToHierarchicalGPU + SPMDToGPU)
# ─────────────────────────────────────────────────────────────────────────────

print(SEP2)
print("STEP 6 — GPU IR via spmd-opt  (ReduceToHierarchicalGPU + SPMDToGPU)")
print(SEP2)

STAGE2_MLIR = """\
func.func @array_sum(%x: memref<?xf32>, %result: memref<f32>, %N: index)
    attributes {spmd.kernel} {
  %c0   = arith.constant 0 : index
  %c1   = arith.constant 1 : index
  %c4   = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  "spmd.forall"(%c0, %N, %c4) ({
  ^bb0(%tile_start: index):
    %sum = "spmd.reduce"(%c0, %c4, %c1, %zero) ({
    ^bb1(%local_i: index):
      %gidx      = arith.addi %tile_start, %local_i : index
      %in_bounds = arith.cmpi ult, %gidx, %N : index
      %safe_idx  = arith.select %in_bounds, %gidx, %c0 : index
      %loaded    = memref.load %x[%safe_idx] : memref<?xf32>
      %v         = arith.select %in_bounds, %loaded, %zero : f32
      "spmd.yield"(%v) : (f32) -> ()
    }) {"spmd.kind" = #spmd.reduction_kind<add>}
       : (index, index, index, f32) -> f32
    memref.atomic_rmw addf %sum, %result[] : (f32, memref<f32>) -> f32
    "spmd.yield"() : () -> ()
  }) {operandSegmentSizes = array<i32: 1, 1, 1>,
      "spmd.mapping"    = #spmd.level<group>,
      "spmd.tile_sizes" = array<i64: 4>}
     : (index, index, index) -> ()
  func.return
}
"""

if not os.path.exists(SPMD_OPT):
    print(f"  [skip] spmd-opt not found at {SPMD_OPT}")
    print("  Set SPMD_OPT env var or run on compute node with build/ present.")
else:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w",
                                    delete=False) as f:
        f.write(STAGE2_MLIR)
        tmp = f.name
    result_proc = subprocess.run(
        [SPMD_OPT, tmp,
         "--normalize-spmd",
         "--convert-spmd-to-gpu"],
        capture_output=True, text=True)
    os.unlink(tmp)
    if result_proc.returncode == 0:
        print(result_proc.stdout)
    else:
        print("  [error]", result_proc.stderr[:800])

print(SEP2)
print("STEP 6 — expected GPU IR structure (hierarchical reduction pattern):")
print(SEP2)

EXPECTED_GPU = textwrap.dedent("""\
    gpu.launch blocks(...) threads(4, 1, 1)
        workgroup(%scratch : memref<4xf32, #gpu.address_space<workgroup>>) {

      // Phase 1: each thread accumulates its tile elements locally
      scf.for %i = %tx to 4 step 4 {
        %v = memref.load %x[%tile_start + %i]
        %acc = arith.addf %acc, %v
      }
      memref.store %acc, %scratch[%tx]    // scatter to workgroup memory
      gpu.barrier

      // Phase 2: tree reduction in workgroup memory  (log2(4) = 2 steps)
      scf.if %tx < 2 { scratch[tx] += scratch[tx+2] }
      gpu.barrier
      scf.if %tx < 1 { scratch[tx] += scratch[tx+1] }
      gpu.barrier

      // Phase 3: thread 0 flushes one atomic to global accumulator
      scf.if %tx == 0 {
        memref.atomic_rmw addf %scratch[0], %result[]
      }
    }
    //
    // Without spmd.reduce: N threads each do atomic_rmw → serialized at memory
    // With spmd.reduce:    log2(blockDim) barrier steps + 1 atomic per block
""")
print(EXPECTED_GPU)

print(SEP2)
print("DEMO COMPLETE")
print(SEP2)
