#!/usr/bin/env bash
# demo_taichi_to_spmd.sh — Taichi → SPMD IR → GPU IR end-to-end demo.
#
# Usage:
#   bash scripts/demo_taichi_to_spmd.sh
#
# Requires GPU compute node (GLIBC >= 2.32).
# On a login node, Taichi and spmd-opt steps are skipped gracefully;
# all IR transformation stages are still printed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${REPO_ROOT}/.venv/bin/python"
export SPMD_OPT="${REPO_ROOT}/build/bin/spmd-opt"

if [[ ! -f "$PYTHON" ]]; then
  echo "ERROR: .venv not found. Run: bash scripts/setup-venv.sh" >&2
  exit 1
fi

exec "$PYTHON" "${SCRIPT_DIR}/demo_taichi_to_spmd.py" "$@"
