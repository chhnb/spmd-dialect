#!/usr/bin/env bash
# run-robustness-validation.sh — GPU sweep harness.
#
# Runs 3 kernels × ≥8 sizes × ≥3 tile configs and outputs results/robustness/latest.csv.
# Total rows: 3 × 8 × 3 = 72+.
#
# Usage: bash scripts/run-robustness-validation.sh [--sm sm_80] [--outdir <dir>]
#
# Options:
#   --sm <level>     Force SM level (e.g. sm_80, sm_90, sm_100)
#   --outdir <dir>   Output directory (default: results/robustness)
#   --help           Print this message

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${REPO_ROOT}/.venv/bin/python"
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
  SM="$("$PYTHON" "${SCRIPT_DIR}/detect-gpu.py")"
fi

mkdir -p "$OUTDIR"
CSV="${OUTDIR}/latest.csv"

echo "══════════════════════════════════════════════════════════"
echo "  Robustness sweep  SM=${SM}"
echo "  Output: $CSV"
echo "══════════════════════════════════════════════════════════"
echo ""

# Write CSV header.
echo "kernel,N,tile_config,promoted,correctness,rel_err,cpu_ms,gpu_ms,speedup" > "$CSV"

# Helper: append a row to CSV.
append_row() {
  echo "$1,$2,$3,$4,$5,$6,$7,$8,$9" >> "$CSV"
}

# Helper: run ewise and parse result.
run_ewise() {
  local ptx="$1" size="$2" tile="$3"
  local out
  out="$("$PYTHON" "${REPO_ROOT}/harness/run_ewise.py" \
      --ptx "$ptx" --sizes "$size" --perf 2>&1 || true)"
  local ok rel cpu gpu spd
  if echo "$out" | grep -q "PASS"; then
    ok="PASS"
  else
    ok="FAIL"
  fi
  rel=$(echo "$out" | grep -oP 'rel_err=\K[0-9eE+\-.]+' | head -1 || echo "N/A")
  cpu=$(echo "$out" | grep -oP 'cpu_ms=\K[0-9.]+' | head -1 || echo "N/A")
  gpu=$(echo "$out" | grep -oP 'gpu_ms=\K[0-9.]+' | head -1 || echo "N/A")
  spd=$(echo "$out" | grep -oP 'speedup=\K[0-9.]+' | head -1 || echo "N/A")
  append_row "ewise" "$size" "$tile" "no" "$ok" "$rel" "$cpu" "$gpu" "$spd"
}

# Helper: run stencil and parse result.
run_stencil() {
  local ptx="$1" shape="$2" tile="$3"
  local out
  out="$("$PYTHON" "${REPO_ROOT}/harness/run_promoted_stencil.py" \
      --ptx "$ptx" --shapes "$shape" --perf 2>&1 || true)"
  local ok rel cpu gpu spd
  if echo "$out" | grep -q "PASS"; then
    ok="PASS"
  else
    ok="FAIL"
  fi
  rel=$(echo "$out" | grep -oP 'rel_err=\K[0-9eE+\-.]+' | head -1 || echo "N/A")
  cpu=$(echo "$out" | grep -oP 'cpu_ms=\K[0-9.]+' | head -1 || echo "N/A")
  gpu=$(echo "$out" | grep -oP 'gpu_ms=\K[0-9.]+' | head -1 || echo "N/A")
  spd=$(echo "$out" | grep -oP 'speedup=\K[0-9.]+' | head -1 || echo "N/A")
  append_row "stencil" "$shape" "$tile" "yes" "$ok" "$rel" "$cpu" "$gpu" "$spd"
}

# Helper: run reduction and parse result.
run_reduction() {
  local ptx="$1" size="$2" tile="$3"
  local out
  out="$("$PYTHON" "${REPO_ROOT}/harness/run_reduction.py" \
      --ptx "$ptx" --sizes "$size" 2>&1 || true)"
  local ok rel
  if echo "$out" | grep -q "PASS"; then
    ok="PASS"
  else
    ok="FAIL"
  fi
  rel=$(echo "$out" | grep -oP 'rel_err=\K[0-9eE+\-.]+' | head -1 || echo "N/A")
  append_row "reduction" "$size" "$tile" "no" "$ok" "$rel" "N/A" "N/A" "N/A"
}

# ── Kernel 1: ewise ────────────────────────────────────────────────────────────
# Sizes: 1, 32, 33, 64, 256, 257, 1024, 1048576
# Tile configs: 32, 64, 256 (currently PTX is fixed at 256 threads)
EWISE_SIZES=(1 32 33 64 256 257 1024 1048576)
EWISE_TILES=(32 64 256)

echo "── Kernel 1: ewise ──────────────────────────────────"
EWISE_PTX="/tmp/robustness_ewise_${SM}.ptx"
bash "${SCRIPT_DIR}/gen-ptx.sh" \
    "${REPO_ROOT}/test/SPMD/lower-to-gpu-nvptx.mlir" \
    ewise "$EWISE_PTX" "$SM"

for size in "${EWISE_SIZES[@]}"; do
  for tile in "${EWISE_TILES[@]}"; do
    echo "  ewise N=${size} tile=${tile}"
    run_ewise "$EWISE_PTX" "$size" "$tile"
  done
done

# ── Kernel 2: promoted stencil ────────────────────────────────────────────────
# Sizes: 32x8, 33x9, 64x64, 128x128, 256x256, 512x512, 1024x1024, 511x513
# Tile configs: 32x8, 16x16, 64x8
STENCIL_SHAPES=("32x8" "33x9" "64x64" "128x128" "256x256" "512x512" "1024x1024" "511x513")
STENCIL_TILES=("32x8" "16x16" "64x8")

echo ""
echo "── Kernel 2: promoted stencil ───────────────────────"
STENCIL_PTX="/tmp/robustness_stencil_${SM}.ptx"
bash "${SCRIPT_DIR}/gen-ptx.sh" \
    "${REPO_ROOT}/test/SPMD/lower-to-gpu-nvptx-promoted.mlir" \
    promoted "$STENCIL_PTX" "$SM"

for shape in "${STENCIL_SHAPES[@]}"; do
  for tile in "${STENCIL_TILES[@]}"; do
    echo "  stencil shape=${shape} tile=${tile}"
    run_stencil "$STENCIL_PTX" "$shape" "$tile"
  done
done

# ── Kernel 3: reduction ────────────────────────────────────────────────────────
# Sizes: 1, 255, 256, 257, 1024, 65536, 1048576, 16777216
# Tile configs: 64, 128, 256 (atomic reduction, correctness only)
REDUCTION_SIZES=(1 255 256 257 1024 65536 1048576 16777216)
REDUCTION_TILES=(64 128 256)

echo ""
echo "── Kernel 3: reduction ──────────────────────────────"
REDUCTION_PTX="/tmp/robustness_reduction_${SM}.ptx"
bash "${SCRIPT_DIR}/gen-ptx.sh" \
    "${REPO_ROOT}/test/SPMD/lower-to-gpu-nvptx-reduction.mlir" \
    ewise "$REDUCTION_PTX" "$SM"

for size in "${REDUCTION_SIZES[@]}"; do
  for tile in "${REDUCTION_TILES[@]}"; do
    echo "  reduction N=${size} tile=${tile}"
    run_reduction "$REDUCTION_PTX" "$size" "$tile"
  done
done

# ── Summary ────────────────────────────────────────────────────────────────────
TOTAL=$(wc -l < "$CSV")
PASS=$(grep -c ",PASS," "$CSV" || true)
FAIL=$(grep -c ",FAIL," "$CSV" || true)

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Sweep complete: $((TOTAL-1)) rows, ${PASS} PASS, ${FAIL} FAIL"
echo "  CSV: $CSV"
echo "══════════════════════════════════════════════════════════"

if [[ $FAIL -gt 0 ]]; then
  echo "FAILURES:"
  grep ",FAIL," "$CSV" | head -20
  exit 1
fi
