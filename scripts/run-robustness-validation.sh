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

# ── Harness output format note ─────────────────────────────────────────────────
# The harnesses output tabular data with three different row formats:
#   ewise:     "    N   max_err   PASS|FAIL"
#   stencil:   "( N, M)   max_err   PASS|FAIL"  (shape column is NxM text)
#   reduction: "    N   gpu_sum   ref_sum   rel_err   PASS|FAIL"
# Performance rows: "    N   cpu_ms   gpu_ms   speedup" (ewise/reduction only)
#
# Parsers use a generic "last field before PASS/FAIL" approach to handle all formats.

# Extract the numeric error field from the first correctness data row.
# Works for all three harness formats by extracting the last numeric field
# on the PASS/FAIL line (ewise: col 2, stencil: col 2, reduction: col 4).
_parse_rel_err() {
  echo "$1" | grep -E '(PASS|FAIL)' | head -1 \
    | sed 's/[[:space:]]*\(PASS\|FAIL\)[[:space:]]*//' \
    | awk '{print $NF}' || echo "N/A"
}

# Extract t_cpu (column 2 of the first perf data row: "N cpu_ms gpu_ms speedup").
_parse_cpu_ms() {
  echo "$1" | grep -E '^\s+[0-9]+\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+' \
    | head -1 | awk '{print $2}' || echo "N/A"
}

# Extract t_gpu (column 3 of the first perf data row).
_parse_gpu_ms() {
  echo "$1" | grep -E '^\s+[0-9]+\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+' \
    | head -1 | awk '{print $3}' || echo "N/A"
}

# Extract speedup (column 4, strip trailing 'x').
_parse_speedup() {
  echo "$1" | grep -E '^\s+[0-9]+\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+' \
    | head -1 | awk '{gsub(/x$/,"",$4); print $4}' || echo "N/A"
}

# Helper: classify harness output as PASS, SKIP, or FAIL.
# SKIP is returned when the harness rejects the input before GPU launch
# (e.g., N not a multiple of tile size → AssertionError) or when no GPU
# driver is present. FAIL is returned for a wrong numeric result.
_classify_result() {
  local out="$1"
  if echo "$out" | grep -q "PASS"; then
    echo "PASS"
  elif echo "$out" | grep -qiE "assert|AssertionError|SKIP|no cuda|no gpu|driver"; then
    echo "SKIP"
  else
    echo "FAIL"
  fi
}

# Helper: patch spmd.tile_sizes in an MLIR file and compile a fresh PTX.
# Usage: _compile_ptx_for_tile <src_mlir> <kernel_type> <tile> <old_tile> <sm> <out_ptx>
# Replaces `array<i64: <old_tile>>` with `array<i64: <tile>>` in a temp copy,
# then compiles it with gen-ptx.sh. Each tile size requires a separate PTX because
# blockDim.x (= TILE) is baked in as a constant by the MLIR→PTX pipeline.
_compile_ptx_for_tile() {
  local src="$1" ktype="$2" tile="$3" old_tile="$4" sm="$5" out_ptx="$6"
  local tmp
  tmp="$(mktemp /tmp/spmd_tile_XXXXXX.mlir)"
  sed "s/array<i64: ${old_tile}>/array<i64: ${tile}>/g" "$src" > "$tmp"
  bash "${SCRIPT_DIR}/gen-ptx.sh" "$tmp" "$ktype" "$out_ptx" "$sm" >/dev/null 2>&1 || true
  rm -f "$tmp"
}

# Helper: run ewise and parse result.
# Each tile config uses its own PTX (compiled with that tile's blockDim.x baked in).
run_ewise() {
  local ptx="$1" size="$2" tile="$3"
  local out
  out="$("$PYTHON" "${REPO_ROOT}/harness/run_ewise.py" \
      --ptx "$ptx" --sizes "$size" --perf --tile-size "$tile" 2>&1 || true)"
  local ok rel cpu gpu spd
  ok=$(_classify_result "$out")
  rel=$(_parse_rel_err "$out")
  cpu=$(_parse_cpu_ms  "$out")
  gpu=$(_parse_gpu_ms  "$out")
  spd=$(_parse_speedup "$out")
  append_row "ewise" "$size" "$tile" "no" "$ok" "$rel" "$cpu" "$gpu" "$spd"
}

# Helper: run stencil and parse result.
# Non-multiple-of-tile shapes (e.g., 33x9) are rejected by the harness with
# an AssertionError and are classified as SKIP, not FAIL.
run_stencil() {
  local ptx="$1" shape="$2" tile="$3"
  local out
  out="$("$PYTHON" "${REPO_ROOT}/harness/run_promoted_stencil.py" \
      --ptx "$ptx" --shapes "$shape" --perf 2>&1 || true)"
  local ok rel cpu gpu spd
  ok=$(_classify_result "$out")
  rel=$(_parse_rel_err "$out")
  cpu=$(_parse_cpu_ms  "$out")
  gpu=$(_parse_gpu_ms  "$out")
  spd=$(_parse_speedup "$out")
  append_row "stencil" "$shape" "$tile" "yes" "$ok" "$rel" "$cpu" "$gpu" "$spd"
}

# Helper: run reduction and parse result.
# Each tile config uses its own PTX (compiled with that tile's blockDim.x baked in).
run_reduction() {
  local ptx="$1" size="$2" tile="$3"
  local out
  out="$("$PYTHON" "${REPO_ROOT}/harness/run_reduction.py" \
      --ptx "$ptx" --sizes "$size" --tile-size "$tile" 2>&1 || true)"
  local ok rel
  ok=$(_classify_result "$out")
  rel=$(_parse_rel_err "$out")
  append_row "reduction" "$size" "$tile" "no" "$ok" "$rel" "N/A" "N/A" "N/A"
}

# ── Kernel 1: ewise ────────────────────────────────────────────────────────────
# Sizes: 1, 32, 33, 64, 256, 257, 1024, 1048576
# Tile configs: 32, 64, 256 — each gets its own compiled PTX because blockDim.x
# is baked in as a constant by the MLIR→PTX pipeline.
EWISE_SIZES=(1 32 33 64 256 257 1024 1048576)
EWISE_TILES=(32 64 256)
EWISE_SRC="${REPO_ROOT}/test/SPMD/lower-to-gpu-nvptx.mlir"
EWISE_BASE_TILE=32   # spmd.tile_sizes in the source MLIR

echo "── Kernel 1: ewise ──────────────────────────────────"

for tile in "${EWISE_TILES[@]}"; do
  EWISE_PTX="/tmp/robustness_ewise_${SM}_t${tile}.ptx"
  if [[ "$tile" -eq "$EWISE_BASE_TILE" ]]; then
    bash "${SCRIPT_DIR}/gen-ptx.sh" "$EWISE_SRC" ewise "$EWISE_PTX" "$SM"
  else
    _compile_ptx_for_tile "$EWISE_SRC" ewise "$tile" "$EWISE_BASE_TILE" "$SM" "$EWISE_PTX"
  fi
  for size in "${EWISE_SIZES[@]}"; do
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
# Tile configs: 64, 128, 256 — each gets its own PTX (atomic reduction, correctness only)
REDUCTION_SIZES=(1 255 256 257 1024 65536 1048576 16777216)
REDUCTION_TILES=(64 128 256)
REDUCTION_SRC="${REPO_ROOT}/test/SPMD/lower-to-gpu-nvptx-reduction.mlir"
REDUCTION_BASE_TILE=256   # spmd.tile_sizes in the source MLIR

echo ""
echo "── Kernel 3: reduction ──────────────────────────────"

for tile in "${REDUCTION_TILES[@]}"; do
  REDUCTION_PTX="/tmp/robustness_reduction_${SM}_t${tile}.ptx"
  if [[ "$tile" -eq "$REDUCTION_BASE_TILE" ]]; then
    bash "${SCRIPT_DIR}/gen-ptx.sh" "$REDUCTION_SRC" ewise "$REDUCTION_PTX" "$SM"
  else
    _compile_ptx_for_tile "$REDUCTION_SRC" ewise "$tile" "$REDUCTION_BASE_TILE" "$SM" "$REDUCTION_PTX"
  fi
  for size in "${REDUCTION_SIZES[@]}"; do
    echo "  reduction N=${size} tile=${tile}"
    run_reduction "$REDUCTION_PTX" "$size" "$tile"
  done
done

# ── Summary ────────────────────────────────────────────────────────────────────
TOTAL=$(wc -l < "$CSV")
PASS=$(grep -c ",PASS," "$CSV" || true)
SKIP=$(grep -c ",SKIP," "$CSV" || true)
FAIL=$(grep -c ",FAIL," "$CSV" || true)

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Sweep complete: $((TOTAL-1)) rows, ${PASS} PASS, ${SKIP} SKIP, ${FAIL} FAIL"
echo "  CSV: $CSV"
echo "══════════════════════════════════════════════════════════"

if [[ $FAIL -gt 0 ]]; then
  echo "FAILURES:"
  grep ",FAIL," "$CSV" | head -20
  exit 1
fi
