#!/usr/bin/env bash
# run-differential.sh — Cross-backend differential correctness harness.
#
# Compares CPU serial (SCF), OpenMP parallel, and GPU results for all three
# kernel types. Prints a summary table and exits non-zero if any comparison
# fails.
#
# Usage: bash scripts/run-differential.sh [--sm sm_80] [--outdir <dir>]
#
# Options:
#   --sm <level>     Force SM target
#   --outdir <dir>   Output directory (default: results/robustness)
#   --help           Print this message
#
# ── Backend implementation ────────────────────────────────────────────────────
#
# cpu_ok  (CPU serial — SPMD→SCF→LLVM→native)
#   Compiles the SPMD kernel through SCF→CF→LLVM→native object, links as a
#   shared library, and invokes harness/run_host.py via ctypes. Compares the
#   kernel output against the numpy reference; reports PASS/FAIL/ERROR.
#
# omp_ok  (OpenMP parallel — SPMD→OMP→LLVM→native)
#   Same pipeline but adds --convert-spmd-to-openmp before SCF lowering.
#   Links with -fopenmp (libomp via clang). Invokes run_host.py and compares
#   against numpy. Reports PASS/FAIL/ERROR.
#
# gpu_ok  (GPU execution + numeric comparison vs numpy)
#   Runs the SPMD-compiled GPU kernel via the CUDA Python harness and compares
#   against numpy numerically. Requires CUDA GPU. Reports SKIP when no GPU.
#
# ── Python detection ──────────────────────────────────────────────────────────
#
# For host execution (cpu_ok, omp_ok), run_host.py requires Python with numpy
# and ctypes (stdlib). We try the venv first, then the system python3.
# For GPU harnesses, the venv Python is required (cuda_driver is there).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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

# Python for GPU harnesses (needs cuda_driver in venv)
PYTHON_GPU="${REPO_ROOT}/.venv/bin/python"

# Python for host runners (needs numpy + ctypes; finds system python3 if venv broken)
_find_host_python() {
  for py in "${REPO_ROOT}/.venv/bin/python" python3 python; do
    if command -v "$py" >/dev/null 2>&1 && "$py" -c "import numpy, ctypes" 2>/dev/null; then
      echo "$py"
      return 0
    fi
  done
  return 1
}
PYTHON_HOST="$(_find_host_python || echo "")"

# C linker for shared libs: prefer clang (links libomp) then gcc
_find_c_linker() {
  for cc in clang gcc cc; do
    if command -v "$cc" >/dev/null 2>&1; then echo "$cc"; return 0; fi
  done
  return 1
}
CC_LINKER="$(_find_c_linker || echo "gcc")"


if [[ -n "$SM_OVERRIDE" ]]; then
  SM="$SM_OVERRIDE"
else
  SM=""
  if [[ -x "$PYTHON_GPU" ]]; then
    SM="$("$PYTHON_GPU" "${SCRIPT_DIR}/detect-gpu.py" 2>/dev/null || echo "")"
  fi
fi

mkdir -p "$OUTDIR"

echo "══════════════════════════════════════════════════════════"
echo "  Differential correctness harness  SM=${SM:-none}"
echo "══════════════════════════════════════════════════════════"
echo ""

FAIL=0

# ── Helper: compile MLIR source to a position-independent shared library ──────
# Usage: _compile_so <src_mlir> <pipeline_type> <out_so>
# pipeline_type: "scf" or "omp"
# Returns: 0 on success, 1 on failure (prints reason to stderr)
_compile_so() {
  local src="$1" pipeline="$2" out_so="$3"
  local obj
  obj="$(mktemp /tmp/diff_XXXXXX.o)"

  # Build the MLIR lowering pipeline
  local omp_flags=""
  if [[ "$pipeline" == "omp" ]]; then
    omp_flags="--convert-spmd-to-openmp"
  fi

  "$SPMD_BIN/spmd-opt" "$src" \
      --normalize-spmd --plan-spmd-schedule \
      --materialize-spmd-tiling $omp_flags --convert-spmd-to-scf \
  | "$LLVM_BIN/mlir-opt" \
      $( [[ "$pipeline" == "omp" ]] && echo "--convert-openmp-to-llvm" ) \
      --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm \
      --convert-index-to-llvm --convert-func-to-llvm --finalize-memref-to-llvm \
      --reconcile-unrealized-casts \
  | "$LLVM_BIN/mlir-translate" --mlir-to-llvmir \
  | "$LLVM_BIN/llc" -filetype=obj -relocation-model=pic -o "$obj" 2>/dev/null || {
    rm -f "$obj"
    echo "ERROR: MLIR→native compilation failed for $src ($pipeline)" >&2
    return 1
  }

  local -a link_flags=(-shared -fPIC)
  if [[ "$pipeline" == "omp" ]]; then
    # MLIR emits __kmpc_* (LLVM/Intel OMP ABI); prefer libiomp5 over libgomp.
    # --no-as-needed forces NEEDED entry even when ld sees only unresolved refs
    # (in -shared mode ld allows undefined symbols, so --as-needed would skip it).
    local _omp_dir
    for _omp_dir in /home/scratch.huanhuanc_gpu/local/lib "$HOME/.local/lib" /usr/lib/x86_64-linux-gnu /usr/lib; do
      if [[ -f "$_omp_dir/libiomp5.so" ]]; then
        link_flags+=(-L"$_omp_dir" -Wl,--no-as-needed -liomp5 -Wl,--as-needed -Wl,-rpath,"$_omp_dir")
        break
      fi
    done
    if [[ "${link_flags[*]}" != *iomp5* ]]; then
      link_flags+=(-fopenmp)
    fi
  fi
  $CC_LINKER "${link_flags[@]}" "$obj" -o "$out_so" 2>/dev/null || {
    rm -f "$obj"
    echo "ERROR: shared-lib link failed for $src ($pipeline)" >&2
    return 1
  }
  rm -f "$obj"
  return 0
}

# ── Helper: compute dominant err_metric across backends ───────────────────────
# Usage: _dominant_err <cpu_err> <omp_err> <gpu_err>
# Returns: gpu_err if GPU ran (not N/A), else max(cpu_err, omp_err) by float.
_dominant_err() {
  local ce="$1" oe="$2" ge="$3"
  if [[ "$ge" != "N/A" ]]; then echo "$ge"; return; fi
  if [[ "$ce" == "N/A" && "$oe" == "N/A" ]]; then echo "N/A"; return; fi
  if [[ "$ce" == "N/A" ]]; then echo "$oe"; return; fi
  if [[ "$oe" == "N/A" ]]; then echo "$ce"; return; fi
  echo "$ce $oe" | awk '{printf "%s", ($1+0 > $2+0) ? $1 : $2}'
}

# ── Helper: run host kernel via run_host.py ───────────────────────────────────
# Sets $host (PASS/FAIL/ERROR/SKIP) and $err (numeric metric) in caller scope.
_run_host() {
  local so="$1" kernel="$2" arg_name="$3" arg_val="$4"
  if [[ -z "$PYTHON_HOST" ]]; then
    host="SKIP"; err="N/A"; return
  fi
  if [[ ! -f "$so" ]]; then
    host="ERROR"; err="N/A"; return
  fi
  local out
  out="$("$PYTHON_HOST" "${REPO_ROOT}/harness/run_host.py" \
      --lib "$so" --kernel "$kernel" "--${arg_name}" "$arg_val" 2>&1 || true)"
  if echo "$out" | grep -q "PASS"; then
    host="PASS"
  elif echo "$out" | grep -qiE "error|Error|Exception"; then
    host="ERROR"
  else
    host="FAIL"
  fi
  # Extract last numeric field from the first PASS/FAIL line
  err=$(echo "$out" | grep -E '(PASS|FAIL)' | head -1 \
        | sed 's/[[:space:]]*\(PASS\|FAIL\)[[:space:]]*//' \
        | awk '{print $NF}' || echo "N/A")
}

# ── Helper: run GPU harness capturing full output, return status + err ─────────
# Sets $gpu (PASS/FAIL/ERROR/SKIP) and $err (numeric metric) in caller scope.
_run_gpu() {
  local script="$1"; shift
  if [[ -z "$SM" || ! -x "$PYTHON_GPU" ]]; then
    gpu="SKIP"; err="N/A"; return
  fi
  local raw
  raw="$("$PYTHON_GPU" "$script" "$@" 2>&1 || true)"
  if echo "$raw" | grep -q "PASS"; then
    gpu="PASS"
  elif echo "$raw" | grep -qiE "error|cuda|driver"; then
    gpu="ERROR"
  else
    gpu="FAIL"
  fi
  err=$(echo "$raw" | grep -E '(PASS|FAIL)' | head -1 \
        | sed 's/[[:space:]]*\(PASS\|FAIL\)[[:space:]]*//' \
        | awk '{print $NF}' || echo "N/A")
}

# ── Helper: print table row ───────────────────────────────────────────────────
print_row() {
  printf "  %-12s %-12s %-10s %-8s %-8s %-8s  %s\n" \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7"
}

print_row "kernel" "size" "tile" "cpu_ok" "omp_ok" "gpu_ok" "err_metric"
print_row "------" "----" "----" "------" "------" "------" "----------"

# ── Compile SCF and OMP shared libraries ─────────────────────────────────────

EWISE_SRC="${REPO_ROOT}/test/SPMD/lower-to-gpu-nvptx.mlir"
STENCIL_SRC="${REPO_ROOT}/test/SPMD/differential-stencil.mlir"
REDUCTION_SRC="${REPO_ROOT}/test/SPMD/lower-to-gpu-nvptx-reduction.mlir"
HIERARCHICAL_SRC="${REPO_ROOT}/test/SPMD/lower-to-gpu-nvptx-hierarchical-reduction.mlir"

EWISE_SCF_SO="/tmp/diff_ewise_scf_$$.so"
EWISE_OMP_SO="/tmp/diff_ewise_omp_$$.so"
STENCIL_SCF_SO="/tmp/diff_stencil_scf_$$.so"
STENCIL_OMP_SO="/tmp/diff_stencil_omp_$$.so"
REDUCTION_SCF_SO="/tmp/diff_reduction_scf_$$.so"
REDUCTION_OMP_SO="/tmp/diff_reduction_omp_$$.so"

echo "Compiling CPU/OMP kernels..."
_compile_so "$EWISE_SRC"     scf "$EWISE_SCF_SO"     || true
_compile_so "$EWISE_SRC"     omp "$EWISE_OMP_SO"     || true
_compile_so "$STENCIL_SRC"   scf "$STENCIL_SCF_SO"   || true
_compile_so "$STENCIL_SRC"   omp "$STENCIL_OMP_SO"   || true
_compile_so "$REDUCTION_SRC" scf "$REDUCTION_SCF_SO" || true
_compile_so "$REDUCTION_SRC" omp "$REDUCTION_OMP_SO" || true
echo ""

# ── Generate PTX for GPU kernels ──────────────────────────────────────────────
if [[ -n "$SM" ]]; then
  EWISE_PTX="/tmp/diff_ewise_${SM}_$$.ptx"
  STENCIL_PTX="/tmp/diff_stencil_${SM}_$$.ptx"
  REDUCTION_PTX="/tmp/diff_reduction_${SM}_$$.ptx"
  HIERARCHICAL_PTX="/tmp/diff_hierarchical_${SM}_$$.ptx"

  bash "${SCRIPT_DIR}/gen-ptx.sh" \
      "$EWISE_SRC" ewise "$EWISE_PTX" "$SM" >/dev/null 2>&1 || true

  bash "${SCRIPT_DIR}/gen-ptx.sh" \
      "${REPO_ROOT}/test/SPMD/lower-to-gpu-nvptx-promoted.mlir" \
      promoted "$STENCIL_PTX" "$SM" >/dev/null 2>&1 || true

  bash "${SCRIPT_DIR}/gen-ptx.sh" \
      "$REDUCTION_SRC" ewise "$REDUCTION_PTX" "$SM" >/dev/null 2>&1 || true

  bash "${SCRIPT_DIR}/gen-ptx.sh" \
      "$HIERARCHICAL_SRC" hierarchical "$HIERARCHICAL_PTX" "$SM" >/dev/null 2>&1 || true
fi

# ── Kernel 1: ewise N=1024 ────────────────────────────────────────────────────
{
  _run_host "$EWISE_SCF_SO" ewise sizes 1024; cpu="$host"; cpu_err="$err"
  _run_host "$EWISE_OMP_SO" ewise sizes 1024; omp="$host"; omp_err="$err"
  if [[ -n "$SM" ]]; then
    _run_gpu "${REPO_ROOT}/harness/run_ewise.py" \
        --ptx "$EWISE_PTX" --sizes 1024
    gpu="$gpu"; gpu_err="$err"
  else
    gpu="SKIP"; gpu_err="N/A"
  fi
  err_metric="$(_dominant_err "$cpu_err" "$omp_err" "$gpu_err")"
  [[ "$cpu" == "ERROR" || "$cpu" == "FAIL" ]] && FAIL=$((FAIL+1))
  [[ "$omp" == "ERROR" || "$omp" == "FAIL" ]] && FAIL=$((FAIL+1))
  [[ "$gpu" == "FAIL"  || "$gpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  print_row "ewise" "N=1024" "tile=32" "$cpu" "$omp" "$gpu" "$err_metric"
}

# ── Kernel 1: ewise N=1M ──────────────────────────────────────────────────────
{
  _run_host "$EWISE_SCF_SO" ewise sizes 1048576; cpu="$host"; cpu_err="$err"
  _run_host "$EWISE_OMP_SO" ewise sizes 1048576; omp="$host"; omp_err="$err"
  if [[ -n "$SM" ]]; then
    _run_gpu "${REPO_ROOT}/harness/run_ewise.py" \
        --ptx "$EWISE_PTX" --sizes 1048576
    gpu="$gpu"; gpu_err="$err"
  else
    gpu="SKIP"; gpu_err="N/A"
  fi
  err_metric="$(_dominant_err "$cpu_err" "$omp_err" "$gpu_err")"
  [[ "$cpu" == "ERROR" || "$cpu" == "FAIL" ]] && FAIL=$((FAIL+1))
  [[ "$omp" == "ERROR" || "$omp" == "FAIL" ]] && FAIL=$((FAIL+1))
  [[ "$gpu" == "FAIL"  || "$gpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  print_row "ewise" "N=1048576" "tile=32" "$cpu" "$omp" "$gpu" "$err_metric"
}

# ── Kernel 2: stencil 128x128 ─────────────────────────────────────────────────
{
  _run_host "$STENCIL_SCF_SO" stencil shapes 128x128; cpu="$host"; cpu_err="$err"
  _run_host "$STENCIL_OMP_SO" stencil shapes 128x128; omp="$host"; omp_err="$err"
  if [[ -n "$SM" ]]; then
    _run_gpu "${REPO_ROOT}/harness/run_promoted_stencil.py" \
        --ptx "$STENCIL_PTX" --shapes 128x128 --tile-row 32 --tile-col 8
    gpu="$gpu"; gpu_err="$err"
  else
    gpu="SKIP"; gpu_err="N/A"
  fi
  err_metric="$(_dominant_err "$cpu_err" "$omp_err" "$gpu_err")"
  [[ "$cpu" == "ERROR" || "$cpu" == "FAIL" ]] && FAIL=$((FAIL+1))
  [[ "$omp" == "ERROR" || "$omp" == "FAIL" ]] && FAIL=$((FAIL+1))
  [[ "$gpu" == "FAIL"  || "$gpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  print_row "stencil" "128x128" "32x8" "$cpu" "$omp" "$gpu" "$err_metric"
}

# ── Kernel 2: stencil 512x512 ─────────────────────────────────────────────────
{
  _run_host "$STENCIL_SCF_SO" stencil shapes 512x512; cpu="$host"; cpu_err="$err"
  _run_host "$STENCIL_OMP_SO" stencil shapes 512x512; omp="$host"; omp_err="$err"
  if [[ -n "$SM" ]]; then
    _run_gpu "${REPO_ROOT}/harness/run_promoted_stencil.py" \
        --ptx "$STENCIL_PTX" --shapes 512x512 --tile-row 32 --tile-col 8
    gpu="$gpu"; gpu_err="$err"
  else
    gpu="SKIP"; gpu_err="N/A"
  fi
  err_metric="$(_dominant_err "$cpu_err" "$omp_err" "$gpu_err")"
  [[ "$cpu" == "ERROR" || "$cpu" == "FAIL" ]] && FAIL=$((FAIL+1))
  [[ "$omp" == "ERROR" || "$omp" == "FAIL" ]] && FAIL=$((FAIL+1))
  [[ "$gpu" == "FAIL"  || "$gpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  print_row "stencil" "512x512" "32x8" "$cpu" "$omp" "$gpu" "$err_metric"
}

# ── Kernel 3: reduction N=65536 ──────────────────────────────────────────────
{
  _run_host "$REDUCTION_SCF_SO" reduction sizes 65536; cpu="$host"; cpu_err="$err"
  _run_host "$REDUCTION_OMP_SO" reduction sizes 65536; omp="$host"; omp_err="$err"
  if [[ -n "$SM" ]]; then
    _run_gpu "${REPO_ROOT}/harness/run_reduction.py" \
        --ptx "$REDUCTION_PTX" --sizes 65536
    gpu="$gpu"; gpu_err="$err"
  else
    gpu="SKIP"; gpu_err="N/A"
  fi
  err_metric="$(_dominant_err "$cpu_err" "$omp_err" "$gpu_err")"
  [[ "$cpu" == "ERROR" || "$cpu" == "FAIL" ]] && FAIL=$((FAIL+1))
  [[ "$omp" == "ERROR" || "$omp" == "FAIL" ]] && FAIL=$((FAIL+1))
  [[ "$gpu" == "FAIL"  || "$gpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  print_row "reduction" "N=65536" "tile=256" "$cpu" "$omp" "$gpu" "$err_metric"
}

# ── Kernel 3: reduction N=1M ─────────────────────────────────────────────────
{
  _run_host "$REDUCTION_SCF_SO" reduction sizes 1048576; cpu="$host"; cpu_err="$err"
  _run_host "$REDUCTION_OMP_SO" reduction sizes 1048576; omp="$host"; omp_err="$err"
  if [[ -n "$SM" ]]; then
    _run_gpu "${REPO_ROOT}/harness/run_reduction.py" \
        --ptx "$REDUCTION_PTX" --sizes 1048576
    gpu="$gpu"; gpu_err="$err"
  else
    gpu="SKIP"; gpu_err="N/A"
  fi
  err_metric="$(_dominant_err "$cpu_err" "$omp_err" "$gpu_err")"
  [[ "$cpu" == "ERROR" || "$cpu" == "FAIL" ]] && FAIL=$((FAIL+1))
  [[ "$omp" == "ERROR" || "$omp" == "FAIL" ]] && FAIL=$((FAIL+1))
  [[ "$gpu" == "FAIL"  || "$gpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  print_row "reduction" "N=1048576" "tile=256" "$cpu" "$omp" "$gpu" "$err_metric"
}

# ── Kernel 4: reduction_hierarchical ─────────────────────────────────────────
# cpu_ok / omp_ok are SKIP: the hierarchical kernel is GPU-specific (shared-memory
# tree reduction + single global atomic). The SCF/OMP backends use the baseline
# atomic path, not the hierarchical path.  GPU correctness vs numpy is the
# meaningful comparison here.
{
  cpu="SKIP"; cpu_err="N/A"
  omp="SKIP"; omp_err="N/A"
  if [[ -n "$SM" ]]; then
    _run_gpu "${REPO_ROOT}/harness/run_reduction.py" \
        --hierarchical --ptx "$HIERARCHICAL_PTX" --sizes 65536
    gpu="$gpu"; gpu_err="$err"
  else
    gpu="SKIP"; gpu_err="N/A"
  fi
  err_metric="$(_dominant_err "$cpu_err" "$omp_err" "$gpu_err")"
  [[ "$gpu" == "FAIL" || "$gpu" == "ERROR" ]] && FAIL=$((FAIL+1))
  print_row "hier-reduct" "N=65536" "tile=256" "$cpu" "$omp" "$gpu" "$err_metric"
}

# ── Cleanup temp shared libs ──────────────────────────────────────────────────
rm -f "$EWISE_SCF_SO" "$EWISE_OMP_SO" \
      "$STENCIL_SCF_SO" "$STENCIL_OMP_SO" \
      "$REDUCTION_SCF_SO" "$REDUCTION_OMP_SO"
[[ -n "$SM" ]] && rm -f "$EWISE_PTX" "$STENCIL_PTX" "$REDUCTION_PTX" \
                         "${HIERARCHICAL_PTX:-}" 2>/dev/null || true

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
