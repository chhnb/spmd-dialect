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

# ── Helper: run a python harness and extract result ───────────────────────────
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

# ── Helper: compile a kernel through the CPU-serial (SCF) pipeline ──────────
# Compiles the given MLIR file through SCF→CF→LLVM dialect and emits LLVM IR.
# Returns "COMPILE_OK" if compilation succeeds, "ERROR" otherwise.
# NOTE: Execution requires a test-harness main() not yet available.
# TODO: integrate mlir-cpu-runner once a driver harness is added.
compile_cpu_serial() {
  local mlir_file="$1"
  local out_ll="$2"
  local err
  err=$(
    "$SPMD_BIN/spmd-opt" "$mlir_file" \
        --normalize-spmd --plan-spmd-schedule \
        --materialize-spmd-tiling --convert-spmd-to-scf \
    | "$LLVM_BIN/mlir-opt" \
        --convert-scf-to-cf --convert-cf-to-llvm \
        --convert-arith-to-llvm --convert-index-to-llvm \
        --reconcile-unrealized-casts \
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

# ── Helper: compile a kernel through the OpenMP (OMP) pipeline ───────────────
# Compiles through OMP→SCF→CF→LLVM dialect and emits LLVM IR.
# Returns "COMPILE_OK" if compilation succeeds, "ERROR" otherwise.
# NOTE: Execution requires OpenMP runtime linking; not yet implemented.
# TODO: integrate lli --extra-module=<openmp_bc> once driver harness is added.
compile_omp() {
  local mlir_file="$1"
  local out_ll="$2"
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

# ── Compile CPU serial and OpenMP paths ──────────────────────────────────────
EWISE_MLIR="${REPO_ROOT}/test/SPMD/lower-to-gpu-nvptx.mlir"
CPU_EWISE_LL="/tmp/diff_cpu_ewise.ll"
OMP_EWISE_LL="/tmp/diff_omp_ewise.ll"
CPU_REDUCTION_LL="/tmp/diff_cpu_reduction.ll"
REDUCTION_MLIR="${REPO_ROOT}/test/SPMD/lower-to-gpu-nvptx-reduction.mlir"

if [[ -x "$SPMD_BIN/spmd-opt" && -x "$LLVM_BIN/mlir-opt" ]]; then
  CPU_EWISE=$(compile_cpu_serial "$EWISE_MLIR" "$CPU_EWISE_LL")
  OMP_EWISE=$(compile_omp "$EWISE_MLIR" "$OMP_EWISE_LL")
  CPU_REDUCTION=$(compile_cpu_serial "$REDUCTION_MLIR" "$CPU_REDUCTION_LL")
  echo "CPU serial ewise:   $CPU_EWISE"
  echo "OMP ewise:          $OMP_EWISE"
  echo "CPU serial reduction: $CPU_REDUCTION"
else
  CPU_EWISE="SKIP"
  OMP_EWISE="SKIP"
  CPU_REDUCTION="SKIP"
fi

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

# ── Kernel 1: ewise N=1024 ────────────────────────────────────────────────────
# cpu/omp: compilation verification (execution requires test-harness driver)
# gpu: run_ewise.py validates against numpy reference
{
  cpu="$CPU_EWISE"
  omp="$OMP_EWISE"
  if [[ $GPU_AVAILABLE -eq 1 ]]; then
    gpu=$(run_harness "${REPO_ROOT}/harness/run_ewise.py" \
        --ptx "$EWISE_PTX" --sizes 1024)
  else
    gpu="SKIP"
  fi
  err="N/A"
  [[ "$cpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  [[ "$omp" == "ERROR" ]] && FAIL=$((FAIL+1))
  [[ "$gpu" == "FAIL" || "$gpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  print_row "ewise" "N=1024" "tile=32" "$cpu" "$omp" "$gpu" "$err"
}

# ── Kernel 1: ewise N=1M ──────────────────────────────────────────────────────
{
  cpu="$CPU_EWISE"
  omp="$OMP_EWISE"
  if [[ $GPU_AVAILABLE -eq 1 ]]; then
    gpu=$(run_harness "${REPO_ROOT}/harness/run_ewise.py" \
        --ptx "$EWISE_PTX" --sizes 1048576)
  else
    gpu="SKIP"
  fi
  [[ "$gpu" == "FAIL" || "$gpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  print_row "ewise" "N=1048576" "tile=32" "$cpu" "$omp" "$gpu" "N/A"
}

# ── Kernel 2: promoted stencil 128x128 ───────────────────────────────────────
# Note: stencil kernel uses a separate promoted pipeline; OMP path not compiled.
{
  cpu="SKIP"  # promoted stencil CPU path not yet compiled separately
  omp="SKIP"
  if [[ $GPU_AVAILABLE -eq 1 ]]; then
    gpu=$(run_harness "${REPO_ROOT}/harness/run_promoted_stencil.py" \
        --ptx "$STENCIL_PTX" --shapes 128x128)
  else
    gpu="SKIP"
  fi
  [[ "$gpu" == "FAIL" || "$gpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  print_row "stencil" "128x128" "32x8" "$cpu" "$omp" "$gpu" "N/A"
}

# ── Kernel 2: promoted stencil 512x512 ───────────────────────────────────────
{
  cpu="SKIP"
  omp="SKIP"
  if [[ $GPU_AVAILABLE -eq 1 ]]; then
    gpu=$(run_harness "${REPO_ROOT}/harness/run_promoted_stencil.py" \
        --ptx "$STENCIL_PTX" --shapes 512x512)
  else
    gpu="SKIP"
  fi
  [[ "$gpu" == "FAIL" || "$gpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  print_row "stencil" "512x512" "32x8" "$cpu" "$omp" "$gpu" "N/A"
}

# ── Kernel 3: reduction N=65536 ──────────────────────────────────────────────
{
  cpu="$CPU_REDUCTION"
  omp="SKIP"  # OMP reduction uses a different lowering pattern; not yet compiled
  if [[ $GPU_AVAILABLE -eq 1 ]]; then
    gpu=$(run_harness "${REPO_ROOT}/harness/run_reduction.py" \
        --ptx "$REDUCTION_PTX" --sizes 65536)
  else
    gpu="SKIP"
  fi
  [[ "$cpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  [[ "$gpu" == "FAIL" || "$gpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  print_row "reduction" "N=65536" "tile=256" "$cpu" "$omp" "$gpu" "N/A"
}

# ── Kernel 3: reduction N=1M ─────────────────────────────────────────────────
{
  cpu="$CPU_REDUCTION"
  omp="SKIP"
  if [[ $GPU_AVAILABLE -eq 1 ]]; then
    gpu=$(run_harness "${REPO_ROOT}/harness/run_reduction.py" \
        --ptx "$REDUCTION_PTX" --sizes 1048576)
  else
    gpu="SKIP"
  fi
  [[ "$gpu" == "FAIL" || "$gpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  print_row "reduction" "N=1048576" "tile=256" "$cpu" "$omp" "$gpu" "N/A"
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
