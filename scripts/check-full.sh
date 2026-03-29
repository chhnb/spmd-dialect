#!/usr/bin/env bash
# check-full.sh — Full regression suite.
#
# Runs all lit tests, full GPU sweep (3 kernels × 8 sizes × 3 tile configs),
# promotion ablation, and differential correctness comparison.
# Requires a CUDA GPU.
#
# Usage: bash scripts/check-full.sh [options]
#
# Options:
#   --lit-path <path>  Path to llvm-lit
#   --build <dir>      SPMD build directory (default: ./build)
#   --sm <level>       Force SM level (e.g. sm_80)
#   --no-gpu           Skip GPU parts (run lit only)
#   --verbose          Verbose lit output
#   --help             Print this message

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLVM_BUILD="${LLVM_BUILD:-/home/scratch.huanhuanc_gpu/spmd/llvm-project/build}"
SPMD_BUILD="${SPMD_BUILD:-${REPO_ROOT}/build}"
LIT="${LIT:-${LLVM_BUILD}/bin/llvm-lit}"
SM_OVERRIDE=""
SKIP_GPU=0
VERBOSE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lit-path)  LIT="$2"; shift 2 ;;
    --build)     SPMD_BUILD="$2"; shift 2 ;;
    --sm)        SM_OVERRIDE="$2"; shift 2 ;;
    --no-gpu)    SKIP_GPU=1; shift ;;
    --verbose)   VERBOSE=1; shift ;;
    --help|-h)
      grep "^#" "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -x "$LIT" ]]; then
  echo "ERROR: llvm-lit not found at $LIT" >&2
  exit 1
fi

LIT_ARGS=(-j8)
if [[ $VERBOSE -eq 1 ]]; then
  LIT_ARGS+=(-v)
fi

echo "══════════════════════════════════════════════════════"
echo "  check-full: all lit + GPU sweep + differential"
echo "══════════════════════════════════════════════════════"

# Step 1: full lit suite
echo ""
echo "── Step 1: full lit suite ───────────────────────────"
"$LIT" "${LIT_ARGS[@]}" "${SPMD_BUILD}/test/SPMD"

# Step 2: robustness sweep + differential
if [[ $SKIP_GPU -eq 0 ]]; then
  SM_FLAG=""
  if [[ -n "$SM_OVERRIDE" ]]; then
    SM_FLAG="--sm ${SM_OVERRIDE}"
  fi

  echo ""
  echo "── Step 2: robustness sweep ─────────────────────────"
  bash "${SCRIPT_DIR}/run-robustness-validation.sh" ${SM_FLAG}

  echo ""
  echo "── Step 3: differential correctness ────────────────"
  bash "${SCRIPT_DIR}/run-differential.sh" ${SM_FLAG}
else
  echo ""
  echo "── Steps 2-3: GPU parts skipped (--no-gpu) ─────────"
fi

echo ""
echo "══════════════════════════════════════════════════════"
echo "  check-full: PASSED"
echo "══════════════════════════════════════════════════════"
