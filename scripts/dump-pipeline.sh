#!/usr/bin/env bash
# dump-pipeline.sh — Dump MLIR IR at each stage of the SPMD lowering pipeline.
#
# Produces one .mlir file per stage, useful for debugging pass behavior.
#
# Usage:
#   bash scripts/dump-pipeline.sh <input.mlir> [--outdir <dir>] [--stage <name>]
#
# Options:
#   --outdir <dir>   Output directory (default: /tmp/spmd-pipeline-dump)
#   --stage <name>   Dump only a specific stage and exit.
#                    Valid: normalize, materialize, promote, gpu, outline, nvvm
#   --sm <level>     SM level for NVVM stage (default: sm_80)
#   --help           Print this message
#
# Output files:
#   <outdir>/01-after-normalize.mlir
#   <outdir>/02-after-materialize.mlir
#   <outdir>/03-after-promote.mlir
#   <outdir>/04-after-gpu.mlir
#   <outdir>/05-after-outline.mlir
#   <outdir>/06-after-nvvm.mlir

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLVM_BUILD="${LLVM_BUILD:-/home/scratch.huanhuanc_gpu/spmd/llvm-project/build}"
SPMD_BUILD="${SPMD_BUILD:-${REPO_ROOT}/build}"
LLVM_BIN="${LLVM_BUILD}/bin"
SPMD_BIN="${SPMD_BUILD}/bin"

OUTDIR="/tmp/spmd-pipeline-dump"
STAGE_FILTER=""
SM="sm_80"

INPUT="${1:-}"
if [[ -z "$INPUT" ]]; then
  echo "Usage: dump-pipeline.sh <input.mlir> [options]" >&2
  exit 1
fi
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outdir)  OUTDIR="$2"; shift 2 ;;
    --stage)   STAGE_FILTER="$2"; shift 2 ;;
    --sm)      SM="$2"; shift 2 ;;
    --help|-h)
      grep "^#" "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# Validate stage filter.
if [[ -n "$STAGE_FILTER" ]]; then
  case "$STAGE_FILTER" in
    normalize|materialize|promote|gpu|outline|nvvm) ;;
    *)
      echo "ERROR: unknown stage '${STAGE_FILTER}'" >&2
      echo "       Valid stages: normalize, materialize, promote, gpu, outline, nvvm" >&2
      exit 1 ;;
  esac
fi

mkdir -p "$OUTDIR"
INPUT_ABS="$(cd "$(dirname "$INPUT")" && pwd)/$(basename "$INPUT")"

echo "Input:   $INPUT_ABS"
echo "Outdir:  $OUTDIR"
echo ""

dump_stage() {
  local stage_name="$1"
  local file_num="$2"
  local outfile="${OUTDIR}/${file_num}-after-${stage_name}.mlir"
  shift 2

  # $@ = spmd-opt options to reach this stage
  echo "── Stage: ${stage_name} ──────────────────────────────"
  "$SPMD_BIN/spmd-opt" "$INPUT_ABS" "$@" -o "$outfile"
  echo "   Written: $outfile"

  # Validate the dump is re-parseable.
  if "$SPMD_BIN/spmd-opt" "$outfile" -o /dev/null 2>/dev/null; then
    echo "   Verified: re-parseable ✓"
  else
    echo "   WARNING: output is not re-parseable by spmd-opt (may need mlir-opt)"
  fi
  echo ""

  if [[ -n "$STAGE_FILTER" ]] && [[ "$STAGE_FILTER" == "$stage_name" ]]; then
    echo "Requested stage '${STAGE_FILTER}' complete."
    exit 0
  fi
}

# ── Stage 1: after normalize ──────────────────────────────────────────────────
dump_stage "normalize" "01" \
  --normalize-spmd --plan-spmd-schedule

# ── Stage 2: after materialize ────────────────────────────────────────────────
dump_stage "materialize" "02" \
  --normalize-spmd --plan-spmd-schedule \
  --materialize-spmd-tiling

# ── Stage 3: after promote ────────────────────────────────────────────────────
dump_stage "promote" "03" \
  --normalize-spmd --plan-spmd-schedule \
  --materialize-spmd-tiling --promote-group-memory

# ── Stage 4: after GPU lowering ───────────────────────────────────────────────
dump_stage "gpu" "04" \
  --normalize-spmd --plan-spmd-schedule \
  --materialize-spmd-tiling --promote-group-memory \
  --convert-spmd-to-gpu

# ── Stage 5: after kernel outlining ──────────────────────────────────────────
dump_stage "outline" "05" \
  --normalize-spmd --plan-spmd-schedule \
  --materialize-spmd-tiling --promote-group-memory \
  --convert-spmd-to-gpu \
  "--gpu-kernel-outlining" "--nvvm-attach-target=chip=${SM}"

# ── Stage 6: after NVVM lowering ─────────────────────────────────────────────
STAGE6="${OUTDIR}/06-after-nvvm.mlir"
echo "── Stage: nvvm ──────────────────────────────────────"
"$SPMD_BIN/spmd-opt" "$INPUT_ABS" \
  --normalize-spmd --plan-spmd-schedule \
  --materialize-spmd-tiling --promote-group-memory \
  --convert-spmd-to-gpu \
  "--gpu-kernel-outlining" "--nvvm-attach-target=chip=${SM}" \
  | "$LLVM_BIN/mlir-opt" \
    --convert-gpu-to-nvvm \
    --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm \
    --convert-index-to-llvm --reconcile-unrealized-casts \
  -o "$STAGE6"
echo "   Written: $STAGE6"
echo ""

echo "══════════════════════════════════════════════════════"
echo "  Pipeline dump complete. Files in: $OUTDIR"
echo "══════════════════════════════════════════════════════"
