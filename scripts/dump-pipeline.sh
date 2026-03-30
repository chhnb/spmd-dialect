#!/usr/bin/env bash
# dump-pipeline.sh — Dump MLIR IR at each stage of the SPMD lowering pipeline.
#
# Produces one .mlir file per stage, useful for debugging pass behavior.
# Stages 1-3 (SPMD dialect present) are emitted in generic form so that
# stock mlir-opt can validate them with --allow-unregistered-dialect.
# Stages 4-6 (SPMD dialect absent) are validated by plain mlir-opt.
# Any validation failure causes the script to exit non-zero.
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
#   <outdir>/01-after-normalize.mlir    (--normalize-spmd; generic form)
#   <outdir>/02-after-materialize.mlir  (+ --plan-spmd-schedule --materialize-spmd-tiling; generic form)
#   <outdir>/03-after-promote.mlir      (+ --promote-group-memory; generic form)
#   <outdir>/04-after-gpu.mlir          (+ --convert-spmd-to-gpu)
#   <outdir>/05-after-outline.mlir      (+ --gpu-kernel-outlining)
#   <outdir>/06-after-nvvm.mlir         (mlir-opt NVVM lowering)

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

# dump_spmd_stage <stage_name> <file_num> <spmd-opt args...>
#
# Emits stage in generic form (--mlir-print-op-generic) so stock mlir-opt
# can validate it via --allow-unregistered-dialect.  Validation failure is
# fatal (exits non-zero).
dump_spmd_stage() {
  local stage_name="$1"
  local file_num="$2"
  local outfile="${OUTDIR}/${file_num}-after-${stage_name}.mlir"
  shift 2

  echo "── Stage: ${stage_name} ──────────────────────────────"
  "$SPMD_BIN/spmd-opt" "$INPUT_ABS" "$@" --mlir-print-op-generic -o "$outfile"
  echo "   Written: $outfile"

  # Validate using stock mlir-opt with --allow-unregistered-dialect so it
  # accepts SPMD dialect ops in generic form without needing the SPMD plugin.
  if "$LLVM_BIN/mlir-opt" --allow-unregistered-dialect "$outfile" -o /dev/null 2>/dev/null; then
    echo "   Verified: re-parseable ✓ (mlir-opt --allow-unregistered-dialect)"
  else
    echo "   ERROR: output is not re-parseable by mlir-opt --allow-unregistered-dialect" >&2
    exit 1
  fi
  echo ""

  if [[ -n "$STAGE_FILTER" ]] && [[ "$STAGE_FILTER" == "$stage_name" ]]; then
    echo "Requested stage '${STAGE_FILTER}' complete."
    exit 0
  fi
}

# dump_lowered_stage <stage_name> <file_num> <spmd-opt args...>
#
# Stages 4-5: SPMD dialect fully lowered to GPU/SCF/LLVM.  Validate with
# plain mlir-opt (no --allow-unregistered-dialect needed).
dump_lowered_stage() {
  local stage_name="$1"
  local file_num="$2"
  local outfile="${OUTDIR}/${file_num}-after-${stage_name}.mlir"
  shift 2

  echo "── Stage: ${stage_name} ──────────────────────────────"
  "$SPMD_BIN/spmd-opt" "$INPUT_ABS" "$@" -o "$outfile"
  echo "   Written: $outfile"

  if "$LLVM_BIN/mlir-opt" "$outfile" -o /dev/null 2>/dev/null; then
    echo "   Verified: re-parseable ✓ (mlir-opt)"
  else
    echo "   ERROR: output is not re-parseable by mlir-opt" >&2
    exit 1
  fi
  echo ""

  if [[ -n "$STAGE_FILTER" ]] && [[ "$STAGE_FILTER" == "$stage_name" ]]; then
    echo "Requested stage '${STAGE_FILTER}' complete."
    exit 0
  fi
}

# ── Stage 1: after normalize ────────────────────────────────────────────────
# NormalizeSPMD transforms flat foralls to canonical (group, lane) nesting.
# Emitted in generic form; validated by mlir-opt --allow-unregistered-dialect.
dump_spmd_stage "normalize" "01" \
  --normalize-spmd

# ── Stage 2: after materialize ──────────────────────────────────────────────
# PlanSPMDSchedule assigns tile_sizes and memory_policy; MaterializeSPMDTiling
# expands the tiled iteration domain.
# Emitted in generic form; validated by mlir-opt --allow-unregistered-dialect.
dump_spmd_stage "materialize" "02" \
  --normalize-spmd --plan-spmd-schedule --materialize-spmd-tiling

# ── Stage 3: after promote ──────────────────────────────────────────────────
# PromoteGroupMemory inserts shared-memory allocations and barriers.
# Emitted in generic form; validated by mlir-opt --allow-unregistered-dialect.
dump_spmd_stage "promote" "03" \
  --normalize-spmd --plan-spmd-schedule --materialize-spmd-tiling \
  --promote-group-memory

# ── Stage 4: after GPU lowering ─────────────────────────────────────────────
# ConvertSPMDToGPU replaces SPMD dialect with gpu/scf/arith dialect ops.
dump_lowered_stage "gpu" "04" \
  --normalize-spmd --plan-spmd-schedule --materialize-spmd-tiling \
  --promote-group-memory \
  --convert-spmd-to-gpu

# ── Stage 5: after kernel outlining ─────────────────────────────────────────
dump_lowered_stage "outline" "05" \
  --normalize-spmd --plan-spmd-schedule --materialize-spmd-tiling \
  --promote-group-memory \
  --convert-spmd-to-gpu \
  "--gpu-kernel-outlining" "--nvvm-attach-target=chip=${SM}"

# ── Stage 6: after NVVM lowering ────────────────────────────────────────────
STAGE6="${OUTDIR}/06-after-nvvm.mlir"
echo "── Stage: nvvm ──────────────────────────────────────"
"$SPMD_BIN/spmd-opt" "$INPUT_ABS" \
  --normalize-spmd --plan-spmd-schedule --materialize-spmd-tiling \
  --promote-group-memory \
  --convert-spmd-to-gpu \
  "--gpu-kernel-outlining" "--nvvm-attach-target=chip=${SM}" \
  | "$LLVM_BIN/mlir-opt" \
    --convert-gpu-to-nvvm \
    --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm \
    --convert-index-to-llvm --reconcile-unrealized-casts \
  -o "$STAGE6"
echo "   Written: $STAGE6"
if "$LLVM_BIN/mlir-opt" "$STAGE6" -o /dev/null 2>/dev/null; then
  echo "   Verified: re-parseable ✓ (mlir-opt)"
else
  echo "   ERROR: nvvm stage output is not re-parseable by mlir-opt" >&2
  exit 1
fi
echo ""

if [[ -n "$STAGE_FILTER" ]] && [[ "$STAGE_FILTER" == "nvvm" ]]; then
  echo "Requested stage 'nvvm' complete."
  exit 0
fi

echo "══════════════════════════════════════════════════════"
echo "  Pipeline dump complete. Files in: $OUTDIR"
echo "══════════════════════════════════════════════════════"
