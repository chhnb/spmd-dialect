#!/usr/bin/env python3
"""
detect-gpu.py — Query the first CUDA device and print its SM level.

Output (stdout):
  sm_100

Used by gen-ptx.sh and the harness to auto-select the correct NVPTX target.

Requirements: libcuda.so (CUDA driver, no toolkit needed).
"""

import ctypes
import subprocess
import sys

# ── Mapping: compute_cap string → LLVM SM string ──────────────────────────────
# Covers all SM levels supported by our LLVM build (sm_20 … sm_121).
# Extend this table if new architectures appear.
def compute_cap_to_sm(cc: str) -> str:
    major, minor = map(int, cc.strip().split('.'))
    sm_num = major * 10 + minor
    return f"sm_{sm_num}"

# ── Primary: nvidia-smi query ─────────────────────────────────────────────────
def via_smi() -> str:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=5,
    )
    if r.returncode != 0:
        raise RuntimeError("nvidia-smi failed")
    cc = r.stdout.strip().splitlines()[0].strip()
    return compute_cap_to_sm(cc)

# ── Fallback: CUDA Driver API via ctypes ──────────────────────────────────────
def via_driver() -> str:
    lib = ctypes.CDLL("libcuda.so")

    lib.cuInit.restype = ctypes.c_int
    lib.cuInit(0)

    device = ctypes.c_int(0)
    lib.cuDeviceGet.restype = ctypes.c_int
    lib.cuDeviceGet(ctypes.byref(device), 0)

    major = ctypes.c_int(0)
    minor = ctypes.c_int(0)
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
    lib.cuDeviceGetAttribute(
        ctypes.byref(major), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)
    lib.cuDeviceGetAttribute(
        ctypes.byref(minor), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)

    return f"sm_{major.value * 10 + minor.value}"

if __name__ == "__main__":
    try:
        sm = via_smi()
    except Exception:
        try:
            sm = via_driver()
        except Exception as e:
            print(f"ERROR: could not detect GPU SM level: {e}", file=sys.stderr)
            sys.exit(1)
    print(sm)
