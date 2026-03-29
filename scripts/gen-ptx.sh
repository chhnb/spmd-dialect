#!/usr/bin/env bash
# gen-ptx.sh — Generate PTX from an SPMD kernel .mlir file.
#
# Usage:
#   gen-ptx.sh <mlir_file> <pipeline> <output.ptx> [sm_level]
#
# pipeline:
#   ewise     — non-promoted path: normalize → plan → materialize → convert-to-gpu
#   promoted  — promoted path:     promote-group-memory → convert-to-gpu
#
# sm_level (optional):
#   e.g. sm_100, sm_90, sm_80.  Defaults to auto-detecting the installed GPU.
#   Pass "sm_80" to force Ampere-compatible output, etc.
#
# Examples:
#   gen-ptx.sh test/SPMD/lower-to-gpu-nvptx.mlir ewise /tmp/ewise.ptx
#   gen-ptx.sh test/SPMD/lower-to-gpu-nvptx.mlir ewise /tmp/ewise_sm80.ptx sm_80
#   gen-ptx.sh test/SPMD/lower-to-gpu-nvptx-promoted.mlir promoted /tmp/stencil.ptx

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Tool paths ────────────────────────────────────────────────────────────────
LLVM_BUILD="${LLVM_BUILD:-/home/scratch.huanhuanc_gpu/spmd/llvm-project/build}"
SPMD_BUILD="${SPMD_BUILD:-${REPO_ROOT}/build}"
LLVM_BIN="${LLVM_BUILD}/bin"
SPMD_BIN="${SPMD_BUILD}/bin"

# ── Arguments ─────────────────────────────────────────────────────────────────
MLIR_FILE="${1:?Usage: gen-ptx.sh <mlir_file> <pipeline> <output.ptx> [sm_level]}"
PIPELINE="${2:?pipeline required: ewise | promoted}"
OUTPUT="${3:?output .ptx path required}"
SM_ARG="${4:-}"  # optional override

# ── SM level detection ────────────────────────────────────────────────────────
if [[ -n "$SM_ARG" ]]; then
    SM="$SM_ARG"
else
    PYTHON="${SCRIPT_DIR}/../.venv/bin/python"
    SM="$("${PYTHON}" "${SCRIPT_DIR}/detect-gpu.py" 2>/dev/null || echo "sm_80")"
fi
echo "[gen-ptx] target: ${SM}  pipeline: ${PIPELINE}  input: $(basename "$MLIR_FILE")"

# ── Common NVVM → LLVM IR → PTX lowering ─────────────────────────────────────
lower_to_ptx() {
    # $1 = MLIR stream (from spmd-opt), already past gpu-kernel-outlining
    "$LLVM_BIN/mlir-opt" \
        --convert-gpu-to-nvvm \
        --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm \
        --convert-index-to-llvm --reconcile-unrealized-casts \
    | "$SPMD_BIN/spmd-opt" --spmd-extract-gpu-module \
    | "$LLVM_BIN/mlir-translate" --mlir-to-llvmir \
    | "$LLVM_BIN/llc" --march=nvptx64 --mcpu="$SM" -filetype=asm -o "$OUTPUT"
}

# ── Pipeline selection ─────────────────────────────────────────────────────────
case "$PIPELINE" in
  ewise)
    "$SPMD_BIN/spmd-opt" "$MLIR_FILE" \
        --normalize-spmd --plan-spmd-schedule \
        --materialize-spmd-tiling --convert-spmd-to-gpu \
        --gpu-kernel-outlining "--nvvm-attach-target=chip=${SM}" \
    | lower_to_ptx
    ;;

  promoted)
    "$SPMD_BIN/spmd-opt" "$MLIR_FILE" \
        --promote-group-memory --convert-spmd-to-gpu \
        --gpu-kernel-outlining "--nvvm-attach-target=chip=${SM}" \
    | lower_to_ptx
    ;;

  *)
    echo "ERROR: unknown pipeline '${PIPELINE}'. Use: ewise | promoted" >&2
    exit 1
    ;;
esac

echo "[gen-ptx] wrote ${OUTPUT}"
