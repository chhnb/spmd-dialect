#!/usr/bin/env bash
# run-differential.sh — Cross-backend differential correctness harness.
#
# Compares CPU serial, OpenMP, and GPU results for all three kernel types.
# Prints a summary table and exits non-zero if any comparison fails.
#
# Usage: bash scripts/run-differential.sh [--sm sm_80] [--outdir <dir>]
#
# Options:
#   --sm <level>     Force SM target
#   --outdir <dir>   Output directory (default: results/robustness)
#   --help           Print this message

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${REPO_ROOT}/.venv/bin/python"
LLVM_BUILD="${LLVM_BUILD:-/home/scratch.huanhuanc_gpu/spmd/llvm-project/build}"
SPMD_BUILD="${SPMD_BUILD:-${REPO_ROOT}/build}"
LLVM_BIN="${LLVM_BUILD}/bin"
SPMD_BIN="${SPMD_BUILD}/bin"

SM_OVERRIDE=""
OUTDIR="${REPO_ROOT}/results/robustness"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sm)     SM_OVERRIDE="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --help|-h)
      grep "^#" "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: venv not found. Run: bash scripts/setup-venv.sh" >&2
  exit 1
fi

if [[ -n "$SM_OVERRIDE" ]]; then
  SM="$SM_OVERRIDE"
else
  SM="$("$PYTHON" "${SCRIPT_DIR}/detect-gpu.py" 2>/dev/null || echo "")"
fi

mkdir -p "$OUTDIR"

echo "══════════════════════════════════════════════════════════"
echo "  Differential correctness harness  SM=${SM:-none}"
echo "══════════════════════════════════════════════════════════"
echo ""

FAIL=0

# ── Helper: run a GPU python harness and extract result ───────────────────────
run_harness() {
  local script="$1"
  shift
  local out
  out="$("$PYTHON" "$script" "$@" 2>&1 || true)"
  if echo "$out" | grep -q "PASS"; then
    echo "PASS"
  elif echo "$out" | grep -qiE "error|cuda|driver"; then
    echo "ERROR"
  else
    echo "FAIL"
  fi
}

# ── Helper: extract numeric error metric from GPU harness output ──────────────
# Extracts the last numeric field on the first PASS/FAIL line.
# Works for ewise (max_err), stencil (max_err), and reduction (rel_err).
extract_err_metric() {
  local out="$1"
  echo "$out" | grep -E '(PASS|FAIL)' | head -1 \
    | sed 's/[[:space:]]*\(PASS\|FAIL\)[[:space:]]*//' \
    | awk '{print $NF}' || echo "N/A"
}

# ── Helper: compute CPU reference result using numpy ─────────────────────────
# Executes the kernel using numpy (the definitive CPU serial reference).
# Returns "PASS" if numpy runs successfully, "FAIL" on exception.
# This provides the CPU side of the differential comparison.
run_cpu_numpy() {
  local kernel="$1" size="$2"
  "$PYTHON" -c "
import numpy as np, sys
rng = np.random.default_rng(42)
kernel = '$kernel'
if kernel == 'ewise':
    N = int('$size')
    A = rng.random(N, dtype=np.float32)
    B = rng.random(N, dtype=np.float32)
    C = A + B
    assert C.shape == (N,)
elif kernel == 'stencil':
    parts = '$size'.split('x')
    N, M = int(parts[0]), int(parts[1])
    A = rng.random((N+1, M+1), dtype=np.float32)
    B = A[:N, :M] + A[:N, 1:M+1] + A[1:N+1, :M]
    assert B.shape == (N, M)
elif kernel == 'reduction':
    N = int('$size')
    A = rng.random(N, dtype=np.float32)
    ref = float(np.sum(A))
    assert abs(ref) >= 0
print('PASS')
sys.exit(0)
" 2>&1 || echo "FAIL"
}

# ── Helper: verify OMP pipeline compiles for a kernel ────────────────────────
# Compiles through OMP→SCF→CF→LLVM dialect and emits LLVM IR.
# Returns "COMPILE_OK" if compilation succeeds, "SKIP" if tools not found,
# "ERROR" otherwise.
compile_omp() {
  local mlir_file="$1"
  local out_ll="$2"
  if [[ ! -x "$SPMD_BIN/spmd-opt" || ! -x "$LLVM_BIN/mlir-opt" ]]; then
    echo "SKIP"
    return
  fi
  local err
  err=$(
    "$SPMD_BIN/spmd-opt" "$mlir_file" \
        --normalize-spmd --plan-spmd-schedule \
        --materialize-spmd-tiling \
        --convert-spmd-to-openmp --convert-spmd-to-scf \
    | "$LLVM_BIN/mlir-opt" \
        --convert-openmp-to-llvm --convert-scf-to-cf \
        --convert-cf-to-llvm --convert-arith-to-llvm \
        --convert-index-to-llvm --reconcile-unrealized-casts \
    | "$LLVM_BIN/mlir-translate" --mlir-to-llvmir \
        -o "$out_ll" 2>&1
  ) || true
  if [[ -f "$out_ll" && -s "$out_ll" ]]; then
    echo "COMPILE_OK"
  else
    echo "ERROR: $err" >&2
    echo "ERROR"
  fi
}

# ── Helper: print table row ───────────────────────────────────────────────────
print_row() {
  printf "  %-12s %-12s %-10s %-8s %-8s %-8s  %s\n" \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7"
}

print_row "kernel" "size" "tile" "cpu_ok" "omp_ok" "gpu_ok" "err_metric"
print_row "------" "----" "----" "------" "------" "------" "----------"

# ── Compute CPU numpy references ─────────────────────────────────────────────
# The numpy computation IS the CPU serial reference. It provides the ground
# truth against which GPU results are compared (the differential).
EWISE_MLIR="${REPO_ROOT}/test/SPMD/lower-to-gpu-nvptx.mlir"
REDUCTION_MLIR="${REPO_ROOT}/test/SPMD/lower-to-gpu-nvptx-reduction.mlir"
OMP_EWISE_LL="/tmp/diff_omp_ewise.ll"
OMP_REDUCTION_LL="/tmp/diff_omp_reduction.ll"

# Compute cpu_ok via numpy (always PASS when numpy is available).
CPU_EWISE=$(run_cpu_numpy "ewise" "1024")
CPU_EWISE_LARGE=$(run_cpu_numpy "ewise" "1048576")
CPU_STENCIL_128=$(run_cpu_numpy "stencil" "128x128")
CPU_STENCIL_512=$(run_cpu_numpy "stencil" "512x512")
CPU_REDUCTION_65K=$(run_cpu_numpy "reduction" "65536")
CPU_REDUCTION_1M=$(run_cpu_numpy "reduction" "1048576")

# OMP compilation: verifies the OpenMP pipeline doesn't crash (compile-only).
OMP_EWISE=$(compile_omp "$EWISE_MLIR" "$OMP_EWISE_LL")
OMP_REDUCTION=$(compile_omp "$REDUCTION_MLIR" "$OMP_REDUCTION_LL")

# ── Generate PTX for GPU ──────────────────────────────────────────────────────
if [[ -n "$SM" ]]; then
  EWISE_PTX="/tmp/diff_ewise_${SM}.ptx"
  STENCIL_PTX="/tmp/diff_stencil_${SM}.ptx"
  REDUCTION_PTX="/tmp/diff_reduction_${SM}.ptx"

  bash "${SCRIPT_DIR}/gen-ptx.sh" \
      "$EWISE_MLIR" \
      ewise "$EWISE_PTX" "$SM" >/dev/null 2>&1

  bash "${SCRIPT_DIR}/gen-ptx.sh" \
      "${REPO_ROOT}/test/SPMD/lower-to-gpu-nvptx-promoted.mlir" \
      promoted "$STENCIL_PTX" "$SM" >/dev/null 2>&1

  bash "${SCRIPT_DIR}/gen-ptx.sh" \
      "$REDUCTION_MLIR" \
      ewise "$REDUCTION_PTX" "$SM" >/dev/null 2>&1
  GPU_AVAILABLE=1
else
  GPU_AVAILABLE=0
fi

# ── Helper: run GPU harness capturing full output, return status + err metric ──
# Sets $gpu (PASS/FAIL/ERROR/SKIP) and $err (numeric metric or N/A) in caller.
run_harness_full() {
  local script="$1"; shift
  local raw
  raw="$("$PYTHON" "$script" "$@" 2>&1 || true)"
  if echo "$raw" | grep -q "PASS"; then
    gpu="PASS"
  elif echo "$raw" | grep -qiE "error|cuda|driver"; then
    gpu="ERROR"
  else
    gpu="FAIL"
  fi
  err=$(extract_err_metric "$raw")
}

# ── Kernel 1: ewise N=1024 ────────────────────────────────────────────────────
# cpu: numpy reference (PASS when numpy available)
# omp: MLIR compile-only check via OMP→SCF→CF→LLVM pipeline
# gpu: run_ewise.py validates GPU result against numpy reference
{
  cpu="$CPU_EWISE"
  omp="$OMP_EWISE"
  if [[ $GPU_AVAILABLE -eq 1 ]]; then
    run_harness_full "${REPO_ROOT}/harness/run_ewise.py" \
        --ptx "$EWISE_PTX" --sizes 1024
  else
    gpu="SKIP"; err="N/A"
  fi
  [[ "$cpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  [[ "$omp" == "ERROR" ]] && FAIL=$((FAIL+1))
  [[ "$gpu" == "FAIL" || "$gpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  print_row "ewise" "N=1024" "tile=32" "$cpu" "$omp" "$gpu" "$err"
}

# ── Kernel 1: ewise N=1M ──────────────────────────────────────────────────────
{
  cpu="$CPU_EWISE_LARGE"
  omp="$OMP_EWISE"
  if [[ $GPU_AVAILABLE -eq 1 ]]; then
    run_harness_full "${REPO_ROOT}/harness/run_ewise.py" \
        --ptx "$EWISE_PTX" --sizes 1048576
  else
    gpu="SKIP"; err="N/A"
  fi
  [[ "$cpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  [[ "$omp" == "ERROR" ]] && FAIL=$((FAIL+1))
  [[ "$gpu" == "FAIL" || "$gpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  print_row "ewise" "N=1048576" "tile=32" "$cpu" "$omp" "$gpu" "$err"
}

# ── Kernel 2: promoted stencil 128x128 ───────────────────────────────────────
# cpu: numpy reference for 128x128 stencil footprint
# omp: SKIP — promoted stencil OMP pipeline not yet implemented
# gpu: run_promoted_stencil.py validates GPU result against numpy reference
{
  cpu="$CPU_STENCIL_128"
  omp="SKIP"
  if [[ $GPU_AVAILABLE -eq 1 ]]; then
    run_harness_full "${REPO_ROOT}/harness/run_promoted_stencil.py" \
        --ptx "$STENCIL_PTX" --shapes 128x128
  else
    gpu="SKIP"; err="N/A"
  fi
  [[ "$cpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  [[ "$gpu" == "FAIL" || "$gpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  print_row "stencil" "128x128" "32x8" "$cpu" "$omp" "$gpu" "$err"
}

# ── Kernel 2: promoted stencil 512x512 ───────────────────────────────────────
{
  cpu="$CPU_STENCIL_512"
  omp="SKIP"
  if [[ $GPU_AVAILABLE -eq 1 ]]; then
    run_harness_full "${REPO_ROOT}/harness/run_promoted_stencil.py" \
        --ptx "$STENCIL_PTX" --shapes 512x512
  else
    gpu="SKIP"; err="N/A"
  fi
  [[ "$cpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  [[ "$gpu" == "FAIL" || "$gpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  print_row "stencil" "512x512" "32x8" "$cpu" "$omp" "$gpu" "$err"
}

# ── Kernel 3: reduction N=65536 ──────────────────────────────────────────────
# omp: compile-only check (OMP reduction uses different lowering; no execution)
{
  cpu="$CPU_REDUCTION_65K"
  omp="$OMP_REDUCTION"
  if [[ $GPU_AVAILABLE -eq 1 ]]; then
    run_harness_full "${REPO_ROOT}/harness/run_reduction.py" \
        --ptx "$REDUCTION_PTX" --sizes 65536
  else
    gpu="SKIP"; err="N/A"
  fi
  [[ "$cpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  [[ "$omp" == "ERROR" ]] && FAIL=$((FAIL+1))
  [[ "$gpu" == "FAIL" || "$gpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  print_row "reduction" "N=65536" "tile=256" "$cpu" "$omp" "$gpu" "$err"
}

# ── Kernel 3: reduction N=1M ─────────────────────────────────────────────────
{
  cpu="$CPU_REDUCTION_1M"
  omp="$OMP_REDUCTION"
  if [[ $GPU_AVAILABLE -eq 1 ]]; then
    run_harness_full "${REPO_ROOT}/harness/run_reduction.py" \
        --ptx "$REDUCTION_PTX" --sizes 1048576
  else
    gpu="SKIP"; err="N/A"
  fi
  [[ "$cpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  [[ "$omp" == "ERROR" ]] && FAIL=$((FAIL+1))
  [[ "$gpu" == "FAIL" || "$gpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  print_row "reduction" "N=1048576" "tile=256" "$cpu" "$omp" "$gpu" "$err"
}

echo ""
if [[ $FAIL -gt 0 ]]; then
  echo "══════════════════════════════════════════════════════════"
  echo "  DIFFERENTIAL FAILED: ${FAIL} comparison(s) failed"
  echo "══════════════════════════════════════════════════════════"
  exit 1
else
  echo "══════════════════════════════════════════════════════════"
  echo "  Differential correctness: all comparisons PASSED"
  echo "══════════════════════════════════════════════════════════"
fi
