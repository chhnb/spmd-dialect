#!/usr/bin/env bash
# run-validation.sh — One-click GPU correctness validation.
#
# Detects the GPU, generates architecture-specific PTX, then runs
# the Python harness (using CUDA Driver API — no ptxas, no pycuda needed).
#
# Usage:
#   bash scripts/run-validation.sh [--sm sm_80] [--perf]
#
# Options:
#   --sm <level>   Force a specific SM target (e.g. sm_80, sm_90, sm_100).
#                  Default: auto-detect from the installed GPU.
#   --perf         Also run performance benchmarks after correctness tests.
#   --help         Print this message.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HARNESS="${REPO_ROOT}/harness"

# Use the repo-local venv (created by scripts/setup-venv.sh).
PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: venv not found. Run: bash scripts/setup-venv.sh" >&2
  exit 1
fi

SM_OVERRIDE=""
PERF_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sm)     SM_OVERRIDE="$2"; shift 2 ;;
    --perf)   PERF_FLAG="--perf"; shift ;;
    --help|-h)
      grep "^#" "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ── Step 0: detect GPU ────────────────────────────────────────────────────────
if [[ -n "$SM_OVERRIDE" ]]; then
  SM="$SM_OVERRIDE"
else
  SM="$("$PYTHON" "${SCRIPT_DIR}/detect-gpu.py")"
fi
echo "══════════════════════════════════════════════════════"
echo "  Target GPU SM level : ${SM}"
echo "══════════════════════════════════════════════════════"
echo ""

# ── Step 1: generate PTX for detected architecture ────────────────────────────
echo "── Step 1: PTX generation ───────────────────────────"
bash "${SCRIPT_DIR}/gen-ptx.sh" \
    "${REPO_ROOT}/test/SPMD/lower-to-gpu-nvptx.mlir" \
    ewise /tmp/ewise_${SM}.ptx "${SM}"

bash "${SCRIPT_DIR}/gen-ptx.sh" \
    "${REPO_ROOT}/test/SPMD/lower-to-gpu-nvptx-promoted.mlir" \
    promoted /tmp/stencil_${SM}.ptx "${SM}"

bash "${SCRIPT_DIR}/gen-ptx.sh" \
    "${REPO_ROOT}/test/SPMD/lower-to-gpu-nvptx-reduction.mlir" \
    ewise /tmp/reduction_${SM}.ptx "${SM}"

echo ""

# ── Step 2: correctness harness ──────────────────────────────────────────────
echo "── Step 2: ewise_kernel correctness ────────────────"
"$PYTHON" "${HARNESS}/run_ewise.py" \
    --ptx "/tmp/ewise_${SM}.ptx" \
    --sizes "32,100,1024,10000,1000000" \
    ${PERF_FLAG:+--perf} \
    ${PERF_FLAG:+--perf-sizes} ${PERF_FLAG:+100000,1000000,10000000}

echo ""
echo "── Step 3: promoted_stencil_kernel correctness ─────"
"$PYTHON" "${HARNESS}/run_promoted_stencil.py" \
    --ptx "/tmp/stencil_${SM}.ptx" \
    --shapes "64x64,128x128,512x512,1024x1024" \
    ${PERF_FLAG:+--perf}

echo ""
echo "── Step 4: atomic_sum_kernel correctness ────────────"
"$PYTHON" "${HARNESS}/run_reduction.py" \
    --ptx "/tmp/reduction_${SM}.ptx" \
    --sizes "1024,65536,1048576" \
    ${PERF_FLAG:+--perf}

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Validation complete."
echo "══════════════════════════════════════════════════════"
