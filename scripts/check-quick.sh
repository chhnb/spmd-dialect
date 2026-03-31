#!/usr/bin/env bash
# check-quick.sh — Quick regression suite.
#
# Runs verifier + legality lit tests and CPU pipeline smoke tests.
# Designed to complete in under 30 seconds on a development machine.
#
# Usage: bash scripts/check-quick.sh [--lit-path <path>] [--build <dir>]
#
# Options:
#   --lit-path <path>  Path to llvm-lit (default: auto-detect from LLVM_BUILD)
#   --build <dir>      SPMD build directory (default: ./build)
#   --verbose          Show lit output even on success
#   --help             Print this message

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLVM_BUILD="${LLVM_BUILD:-/home/scratch.huanhuanc_gpu/spmd/llvm-project/build}"
SPMD_BUILD="${SPMD_BUILD:-${REPO_ROOT}/build}"
LIT="${LIT:-${LLVM_BUILD}/bin/llvm-lit}"
VERBOSE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lit-path)  LIT="$2"; shift 2 ;;
    --build)     SPMD_BUILD="$2"; shift 2 ;;
    --verbose)   VERBOSE=1; shift ;;
    --help|-h)
      grep "^#" "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -x "$LIT" ]]; then
  echo "ERROR: llvm-lit not found at $LIT" >&2
  echo "       Set LLVM_BUILD or pass --lit-path" >&2
  exit 1
fi

LIT_ARGS=(-j4)
if [[ $VERBOSE -eq 1 ]]; then
  LIT_ARGS+=(-v)
fi

echo "═══════════════════════════════════════════"
echo "  check-quick: verifier + legality + smoke"
echo "═══════════════════════════════════════════"

# Run lit on the subset of tests relevant for quick checks:
#   - invalid/negative/verifier tests
#   - attrs tests
#   - ewise/normalize/materialize smoke
#   - sum (spmd.reduce regression: exercises pattern priority after ReduceToHierarchicalGPU)
"$LIT" "${LIT_ARGS[@]}" \
  --filter="invalid|negative|verifier|attrs|verify-spmd|subset|ewise|normalize|materialize|sum|reduction-hierarchical-gpu" \
  "${SPMD_BUILD}/test/SPMD"

echo ""
echo "═══════════════════════════════════════════"
echo "  check-quick: PASSED"
echo "═══════════════════════════════════════════"
