#!/usr/bin/env bash
# check-medium.sh — Medium regression suite.
#
# Runs all lit tests plus a small correctness sweep (ewise + stencil, ≤3 sizes).
# Requires CUDA GPU for the sweep portion; skips GPU sweep gracefully if no GPU.
#
# Usage: bash scripts/check-medium.sh [options]
#
# Options:
#   --lit-path <path>  Path to llvm-lit
#   --build <dir>      SPMD build directory (default: ./build)
#   --sm <level>       Force SM level (e.g. sm_80)
#   --no-gpu           Skip GPU sweep even if GPU is available
#   --verbose          Verbose lit output
#   --help             Print this message

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLVM_BUILD="${LLVM_BUILD:-/home/scratch.huanhuanc_gpu/spmd/llvm-project/build}"
SPMD_BUILD="${SPMD_BUILD:-${REPO_ROOT}/build}"
LIT="${LIT:-${LLVM_BUILD}/bin/llvm-lit}"
PYTHON="${REPO_ROOT}/.venv/bin/python"
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

LIT_ARGS=(-j4)
if [[ $VERBOSE -eq 1 ]]; then
  LIT_ARGS+=(-v)
fi

echo "══════════════════════════════════════════════════════"
echo "  check-medium: full lit suite + small GPU sweep"
echo "══════════════════════════════════════════════════════"

# Step 1: full lit suite
echo ""
echo "── Step 1: full lit test suite ─────────────────────"
"$LIT" "${LIT_ARGS[@]}" "${SPMD_BUILD}/test/SPMD"

# Step 2: GPU correctness sweep (small subset)
if [[ $SKIP_GPU -eq 0 ]] && [[ -x "$PYTHON" ]]; then
  if [[ -n "$SM_OVERRIDE" ]]; then
    SM="$SM_OVERRIDE"
  else
    SM="$("$PYTHON" "${SCRIPT_DIR}/detect-gpu.py" 2>/dev/null || echo "")"
  fi

  if [[ -n "$SM" ]]; then
    echo ""
    echo "── Step 2: small GPU correctness sweep (SM=${SM}) ──"

    bash "${SCRIPT_DIR}/gen-ptx.sh" \
        "${REPO_ROOT}/test/SPMD/lower-to-gpu-nvptx.mlir" \
        ewise /tmp/ewise_${SM}.ptx "${SM}"

    "$PYTHON" "${REPO_ROOT}/harness/run_ewise.py" \
        --ptx "/tmp/ewise_${SM}.ptx" \
        --sizes "32,1024,1000000"

    bash "${SCRIPT_DIR}/gen-ptx.sh" \
        "${REPO_ROOT}/test/SPMD/lower-to-gpu-nvptx-promoted.mlir" \
        promoted /tmp/stencil_${SM}.ptx "${SM}"

    "$PYTHON" "${REPO_ROOT}/harness/run_promoted_stencil.py" \
        --ptx "/tmp/stencil_${SM}.ptx" \
        --shapes "64x64,512x512"
  else
    echo ""
    echo "── Step 2: GPU sweep skipped (no GPU detected) ──"
  fi
else
  echo ""
  echo "── Step 2: GPU sweep skipped ───────────────────────"
fi

echo ""
echo "══════════════════════════════════════════════════════"
echo "  check-medium: PASSED"
echo "══════════════════════════════════════════════════════"
