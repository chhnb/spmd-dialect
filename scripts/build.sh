#!/usr/bin/env bash
# build.sh — One-click build for SPMD Dialect on Ubuntu.
#
# Builds everything from source:
#   1. Install system dependencies (cmake, ninja, clang, python3, uv)
#   2. Clone + build LLVM/MLIR   (~1-2 h, skipped if already built)
#   3. Build spmd-dialect         (~5 min)
#   4. Set up Python venv (uv)
#
# Usage:
#   bash scripts/build.sh                   # full build
#   bash scripts/build.sh --skip-llvm       # skip LLVM build (already done)
#   bash scripts/build.sh --venv-only       # only (re)create Python venv
#   JOBS=8 bash scripts/build.sh            # limit parallel jobs
#   LLVM_BUILD=/path bash scripts/build.sh  # custom LLVM build dir
#   LLVM_SRC=/path   bash scripts/build.sh  # custom LLVM source dir
#
# Environment variables:
#   LLVM_SRC    LLVM source tree  (default: <repo>/../llvm-project)
#   LLVM_BUILD  LLVM build dir    (default: <repo>/../llvm-project/build)
#   JOBS        Parallel jobs     (default: nproc)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLVM_SRC="${LLVM_SRC:-${REPO_ROOT}/../llvm-project}"
LLVM_BUILD="${LLVM_BUILD:-${LLVM_SRC}/build}"
SPMD_BUILD="${REPO_ROOT}/build"
JOBS="${JOBS:-$(nproc)}"

SKIP_LLVM=false
VENV_ONLY=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-llvm)  SKIP_LLVM=true;  shift ;;
    --venv-only)  VENV_ONLY=true;  shift ;;
    --help|-h)
      grep "^#" "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ── Banner ────────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════╗"
echo "║          SPMD Dialect — One-click Build              ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Repo : ${REPO_ROOT}"
echo "║  LLVM : ${LLVM_BUILD}"
echo "║  Jobs : ${JOBS}"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
if [[ "$VENV_ONLY" == false ]]; then

# ── Step 1: System dependencies ───────────────────────────────────────────────
echo "── Step 1: System dependencies ──────────────────────────"
if command -v apt-get &>/dev/null; then
  MISSING=()
  for pkg in cmake ninja-build clang lld python3 python3-pip git curl; do
    dpkg -s "$pkg" &>/dev/null || MISSING+=("$pkg")
  done
  if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "  Installing: ${MISSING[*]}"
    sudo apt-get update -qq
    sudo apt-get install -y "${MISSING[@]}"
  else
    echo "  All system deps present."
  fi
else
  echo "  Not Ubuntu/Debian — skipping apt-get. Ensure cmake, ninja, clang are in PATH."
fi
echo ""

# ── Step 2: uv ────────────────────────────────────────────────────────────────
echo "── Step 2: uv (Python package manager) ──────────────────"
if ! command -v uv &>/dev/null; then
  echo "  Installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$PATH"
fi
echo "  uv: $(uv --version)"
echo ""

# ── Step 3: LLVM/MLIR ─────────────────────────────────────────────────────────
echo "── Step 3: LLVM/MLIR ────────────────────────────────────"
if [[ "$SKIP_LLVM" == true ]]; then
  echo "  --skip-llvm: using existing build at ${LLVM_BUILD}"
elif [[ -f "${LLVM_BUILD}/lib/cmake/mlir/MLIRConfig.cmake" ]]; then
  echo "  Already built at ${LLVM_BUILD} — skipping."
  echo "  (pass --skip-llvm=false or delete ${LLVM_BUILD} to rebuild)"
else
  # Clone if needed
  if [[ ! -d "${LLVM_SRC}/.git" ]]; then
    LLVM_VERSION_FILE="${REPO_ROOT}/LLVM_VERSION"
    if [[ -f "$LLVM_VERSION_FILE" ]]; then
      LLVM_TAG="$(cat "$LLVM_VERSION_FILE" | tr -d '[:space:]')"
    else
      LLVM_TAG="llvmorg-20.1.7"
      echo "  WARNING: LLVM_VERSION not found; defaulting to ${LLVM_TAG}."
    fi
    echo "  Cloning llvm-project @ ${LLVM_TAG} (shallow, ~800 MB) ..."
    git clone --depth 1 --branch "$LLVM_TAG" \
        https://github.com/llvm/llvm-project.git "$LLVM_SRC"
  else
    echo "  Source already present at ${LLVM_SRC}"
  fi

  echo "  Configuring LLVM/MLIR ..."
  mkdir -p "$LLVM_BUILD"
  cmake -S "${LLVM_SRC}/llvm" -B "$LLVM_BUILD" \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DLLVM_ENABLE_PROJECTS="mlir" \
      -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
      2>&1 | tail -5

  echo "  Building LLVM/MLIR with ${JOBS} jobs (this takes 1-2 hours) ..."
  echo "  Progress is logged to ${LLVM_BUILD}/build.log"
  cmake --build "$LLVM_BUILD" -- -j"${JOBS}" \
      2>&1 | tee "${LLVM_BUILD}/build.log" | grep -E "^\[|error:|warning: " | tail -20 || true
fi

# Verify
if [[ ! -f "${LLVM_BUILD}/lib/cmake/mlir/MLIRConfig.cmake" ]]; then
  echo "ERROR: MLIR build not found at ${LLVM_BUILD}" >&2
  echo "       Try: LLVM_BUILD=/path/to/your/llvm/build bash scripts/build.sh --skip-llvm" >&2
  exit 1
fi
echo "  MLIR OK: ${LLVM_BUILD}"
echo ""

# ── Step 4: Build spmd-dialect ────────────────────────────────────────────────
echo "── Step 4: Build spmd-dialect ───────────────────────────"
mkdir -p "$SPMD_BUILD"
cmake -S "$REPO_ROOT" -B "$SPMD_BUILD" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DMLIR_DIR="${LLVM_BUILD}/lib/cmake/mlir" \
    -DLLVM_DIR="${LLVM_BUILD}/lib/cmake/llvm" \
    -DMLIR_INCLUDE_TESTS=ON \
    -DLLVM_EXTERNAL_LIT="${LLVM_BUILD}/bin/llvm-lit" \
    2>&1 | tail -5

cmake --build "$SPMD_BUILD" -- -j"${JOBS}"
echo "  Built: ${SPMD_BUILD}/bin/spmd-opt"
echo ""

fi  # end VENV_ONLY guard

# ── Step 5: Python venv ────────────────────────────────────────────────────────
echo "── Step 5: Python venv (uv) ─────────────────────────────"
if ! command -v uv &>/dev/null; then
  export PATH="$HOME/.cargo/bin:$PATH"
fi
bash "${SCRIPT_DIR}/setup-venv.sh"
echo ""

# ── Done ──────────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Build complete.                                     ║"
echo "║                                                      ║"
echo "║  Run demos :  bash scripts/demo.sh --sm sm_90        ║"
echo "║  Run tests :  bash scripts/check-quick.sh            ║"
echo "╚══════════════════════════════════════════════════════╝"
