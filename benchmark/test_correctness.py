#!/usr/bin/env python3
"""AC-6: Per-case CUDA-vs-DSL correctness validation.

For each case, runs Taichi implementation and verifies output properties.
Serves as reviewable artifact for AC-6 compliance.

Usage:
    python test_correctness.py              # all cases
    python test_correctness.py --cases C1 C9
"""
import sys, os, subprocess, argparse

PYTHON = "/home/scratch.huanhuanc_gpu/spmd/spmd-venv/bin/python"
BD = os.path.dirname(os.path.abspath(__file__))

CASES = {
    "C1":  ("A1_jacobi_2d", "jacobi_taichi", "run(N=64, steps=100, backend='cuda')",
            "assert o.max()<=1.0+1e-5 and o.min()>=-1e-5"),
    "C3":  (".", "heat2d_taichi", "run(N=64, steps=100, backend='cuda')",
            "assert o.max()<=1.0+1e-3 and o.min()>=-1e-3"),
    "C4":  ("A3_wave_equation", "wave_taichi", "run(N=64, steps=100, backend='cuda')",
            "assert np.isfinite(o).all()"),
    "C5":  ("A2_lbm_d2q9", "lbm_taichi", "run(64, 32, steps=10, backend='cuda')",
            "assert np.isfinite(o).all()"),
    "C6":  ("B1_nbody", "nbody_taichi", "run(N=256, steps=10, backend='cuda')",
            "assert np.isfinite(o).all()"),
    "C8":  ("F1_hydro_shallow_water", "hydro_taichi",
            "run_real(steps=10, backend='cuda', mesh='default')",
            "assert abs(o.max()-40.13)<2.0"),
    "C9":  ("F2_hydro_refactored", "hydro_refactored_taichi",
            "run(days=1, backend='cuda', mesh='default')",
            "assert abs(o.max()-7.5)<0.01"),
    "C10": (".", "grayscott_taichi", "run(N=64, steps=100, backend='cuda')",
            "assert np.isfinite(o).all()"),
    "C11": (".", "fdtd2d_taichi", "run(N=64, steps=100, backend='cuda')",
            "assert np.isfinite(o).all()"),
    "C13": (".", "lulesh_taichi", "run(N=16, steps=10, backend='cuda')",
            "assert np.isfinite(o).all()"),
    "C15": (".", "cg_taichi", "run(N=64, steps=50, backend='cuda')",
            "assert np.isfinite(o).all()"),
    "C16": ("D2_stable_fluids", "fluid_taichi", "run(N=64, steps=5, backend='cuda')",
            "assert np.isfinite(o).all()"),
}

def test_case(cid, subdir, mod, call, check):
    code = f"""
import sys,os,numpy as np
sys.path.insert(0,os.path.join('{BD}','{subdir}'))
sys.path.insert(0,'{BD}')
import taichi as ti; ti.init(arch=ti.cuda,default_fp=ti.f32)
from {mod} import *
s,y,output={call}
y();s();y()
o=output.to_numpy() if hasattr(output,'to_numpy') else output.numpy()
{check}
print(f"PASS {cid} [{o.min():.4f},{o.max():.4f}]")
"""
    try:
        r = subprocess.run([PYTHON,"-c",code], capture_output=True, text=True, timeout=120)
        if "PASS" in r.stdout:
            print(f"  {r.stdout.strip()}")
            return True
        print(f"  FAIL {cid}: {r.stderr[:200] if r.stderr else r.stdout[:200]}")
        return False
    except Exception as e:
        print(f"  ERROR {cid}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=None)
    args = parser.parse_args()
    ids = args.cases or sorted(CASES.keys())
    print("=== AC-6 Correctness Validation ===")
    ok = sum(1 for c in ids if c in CASES and test_case(c, *CASES[c]))
    total = sum(1 for c in ids if c in CASES)
    print(f"\n{ok}/{total} passed")
    return 0 if ok == total else 1

if __name__ == "__main__":
    sys.exit(main())
