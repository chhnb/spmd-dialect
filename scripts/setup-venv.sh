#!/usr/bin/env bash
# setup-venv.sh — Create the project virtualenv using uv.
#
# The venv is placed at <repo>/.venv.
# uv cache is redirected to /home/scratch.huanhuanc_gpu/.uv-cache to
# avoid filling the home directory (5 GB quota).
#
# Usage:
#   bash scripts/setup-venv.sh
#
# After running:
#   source .venv/bin/activate      (optional, for interactive use)
#   bash scripts/run-validation.sh (uses .venv/bin/python directly)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Require uv ────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  echo "ERROR: 'uv' not found. Install with: curl -Ls https://astral.sh/uv/install.sh | sh"
  exit 1
fi

# ── Cache on scratch (home disk is small) ─────────────────────────────────────
export UV_CACHE_DIR="${UV_CACHE_DIR:-/home/scratch.huanhuanc_gpu/.uv-cache}"

# ── Create venv ───────────────────────────────────────────────────────────────
echo "Creating .venv in ${REPO_ROOT} ..."
cd "$REPO_ROOT"
uv venv .venv --python 3.12

# ── Install dependencies ──────────────────────────────────────────────────────
echo "Installing dependencies ..."
uv pip install --python .venv/bin/python numpy

echo ""
echo "Done.  Python: $(.venv/bin/python --version)"
echo "Run:   bash scripts/run-validation.sh"
