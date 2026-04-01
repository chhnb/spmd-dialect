#!/bin/bash
# Setup script for simulation benchmark suite.
# Run on a compute/GPU node where GLIBC >= 2.32 and nvcc are available.

set -e

VENV="/home/scratch.huanhuanc_gpu/spmd/spmd-venv"
UV="uv"
PIP_PYTHON="--python $VENV/bin/python"

echo "=== Simulation Benchmark Setup ==="
echo "Python: $($VENV/bin/python --version)"
echo "GLIBC: $(ldd --version 2>&1 | head -1)"
echo ""

# ---- Python packages ----

echo "[1/5] Installing PyTorch + Triton..."
$UV pip install torch --index-url https://download.pytorch.org/whl/cu124 $PIP_PYTHON 2>&1 | tail -3
# Triton comes bundled with PyTorch for CUDA

echo "[2/5] Installing NVIDIA Warp..."
$UV pip install warp-lang $PIP_PYTHON 2>&1 | tail -3

echo "[3/5] Checking Taichi (already installed)..."
$VENV/bin/python -c "import taichi; print('  taichi OK:', taichi.__version__)" 2>&1 || echo "  taichi FAILED (GLIBC issue?)"

echo "[4/5] Installing Numba..."
$UV pip install numba $PIP_PYTHON 2>&1 | tail -3

echo "[5/5] Installing JAX (optional)..."
$UV pip install jax[cuda12] $PIP_PYTHON 2>&1 | tail -3 || echo "  JAX install failed (non-critical)"

echo ""
echo "=== Verification ==="
$VENV/bin/python -c "
packages = ['numpy', 'taichi', 'torch', 'triton', 'warp', 'numba']
for p in packages:
    try:
        m = __import__(p)
        v = getattr(m, '__version__', 'unknown')
        print(f'  {p:12s} {v}')
    except Exception as e:
        print(f'  {p:12s} FAILED: {e}')
"

echo ""
echo "=== CUDA check ==="
$VENV/bin/python -c "
import torch
print(f'  torch.cuda.available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "Python packages ready. For Kokkos/Halide C++ benchmarks, see benchmark/cpp/README.md"
