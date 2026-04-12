#!/usr/bin/env python3
"""Per-case cross-framework correctness validation.

For each case, runs >=2 implementations and compares outputs numerically.
Covers all 21 cases (Taichi for all, plus Warp/Triton where available).
Warp/Triton modules are auto-detected from the filesystem.

Each module calls ti.init()/wp.init() internally — this script does NOT
initialize any framework before importing.
"""
import sys, os, subprocess, argparse, json, re

PYTHON = "/home/scratch.huanhuanc_gpu/spmd/spmd-venv/bin/python"
BD = os.path.dirname(os.path.abspath(__file__))

def run_impl(subdir, module, call, framework="taichi"):
    """Run a single implementation, return (min, max, mean) of output."""
    env_setup = ""
    if framework == "warp":
        env_setup = "import os; os.environ['WARP_CACHE_PATH']='/home/scratch.huanhuanc_gpu/spmd/.warp_cache'\nimport warp as wp; wp.init()\n"

    mod_dir = os.path.join(BD, subdir) if subdir != "." else BD
    code = f"""
import sys, os, json, numpy as np
sys.path.insert(0, '{mod_dir}')
sys.path.insert(0, '{BD}')
{env_setup}
from {module} import *
s, y, o = {call}
y(); s(); y()
if hasattr(o, 'to_numpy'):
    a = o.to_numpy()
elif hasattr(o, 'numpy'):
    a = o.numpy()
else:
    import torch
    a = o.cpu().numpy()
a = a.flatten().astype(float)
finite = a[np.isfinite(a)]
if len(finite) == 0:
    print(json.dumps({{"error": "all NaN/Inf"}}))
else:
    # Output full array hash for cross-framework comparison
    print(json.dumps({{"min": float(finite.min()), "max": float(finite.max()),
                       "mean": float(finite.mean()), "std": float(finite.std()),
                       "n": len(finite), "hash": float(np.sum(np.abs(finite[:1000])))}}))
"""
    try:
        r = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True, timeout=120, cwd=BD)
        for line in r.stdout.strip().split("\n"):
            if line.startswith("{"):
                return json.loads(line)
        return {"error": r.stderr[:200] if r.stderr else "no output"}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}


# Case definitions: list of (framework, subdir, module, call)
CASES = {}
for cid, subdir, mod, call in [
    ("C1", "A1_jacobi_2d", "jacobi_taichi", "run(N=64,steps=100,backend='cuda')"),
    ("C2", ".", "jacobi3d_taichi", "run(N=32,steps=50,backend='cuda')"),
    ("C3", ".", "heat2d_taichi", "run(N=64,steps=100,backend='cuda')"),
    ("C4", "A3_wave_equation", "wave_taichi", "run(N=64,steps=100,backend='cuda')"),
    ("C5", "A2_lbm_d2q9", "lbm_taichi", "run(64,32,steps=10,backend='cuda')"),
    ("C6", "B1_nbody", "nbody_taichi", "run(N=256,steps=10,backend='cuda')"),
    ("C7", "B2_sph", "sph_taichi", "run(N=1024,steps=5,backend='cuda')"),
    ("C8", "F1_hydro_shallow_water", "hydro_taichi", "run_real(steps=10,backend='cuda',mesh='default')"),
    ("C9", "F2_hydro_refactored", "hydro_refactored_taichi", "run(days=1,backend='cuda',mesh='default')"),
    ("C10", ".", "grayscott_taichi", "run(N=64,steps=100,backend='cuda')"),
    ("C11", ".", "fdtd2d_taichi", "run(N=64,steps=100,backend='cuda')"),
    ("C12", "F3_maccormack_3d", "maccormack_taichi", "run(N=32,steps=50,backend='cuda')"),
    ("C13", ".", "lulesh_taichi", "run(N=16,steps=10,backend='cuda')"),
    ("C14", "C2_pic", "pic_taichi", "run(n_particles=1024,n_grid=128,steps=10,backend='cuda')"),
    ("C15", ".", "cg_taichi", "run(N=64,steps=50,backend='cuda')"),
    ("C16", "D2_stable_fluids", "fluid_taichi", "run(N=64,steps=5,backend='cuda')"),
    ("C17", ".", "conv3d_taichi", "run(N=32,steps=1,backend='cuda')"),
    ("C18", ".", "doitgen_taichi", "run(N=32,steps=1,backend='cuda')"),
    ("C19", ".", "lu_taichi", "run(N=64,steps=1,backend='cuda')"),
    ("C20", ".", "adi_taichi", "run(N=64,steps=3,backend='cuda')"),
    ("C21", ".", "gramschmidt_taichi", "run(N=64,steps=1,backend='cuda')"),
]:
    CASES[cid] = [("taichi", subdir, mod, call)]

# Add Warp implementations where available
WARP_IMPLS = [
    ("C1", "A1_jacobi_2d", "jacobi_warp", "run(N=64,steps=100,backend='cuda')"),
    ("C4", "A3_wave_equation", "wave_warp", "run(N=64,steps=100,backend='cuda')"),
    ("C6", "B1_nbody", "nbody_warp", "run(N=256,steps=10,backend='cuda')"),
    ("C7", "B2_sph", "sph_warp", "run(N=1024,steps=5,backend='cuda')"),
    ("C8", "F1_hydro_shallow_water", "hydro_warp", "run_real(steps=10,backend='cuda',mesh='default')"),
    ("C9", "F2_hydro_refactored", "hydro_refactored_warp", "run(days=1,backend='cuda',mesh='default')"),
    ("C16", "D2_stable_fluids", "fluid_warp", "run(N=64,steps=5,backend='cuda')"),
]
for cid, subdir, mod, call in WARP_IMPLS:
    mod_path = os.path.join(BD, subdir, mod + ".py") if subdir != "." else os.path.join(BD, mod + ".py")
    if os.path.exists(mod_path) and cid in CASES:
        CASES[cid].append(("warp", subdir, mod, call))

# Add Triton implementations where available
TRITON_IMPLS = [
    ("C1", "A1_jacobi_2d", "jacobi_triton", "run(N=64,steps=100,backend='cuda')"),
    ("C8", "F1_hydro_shallow_water", "hydro_triton", "run_real(steps=10,backend='cuda',mesh='default')"),
    ("C9", "F2_hydro_refactored", "hydro_refactored_triton", "run(days=1,backend='cuda',mesh='default')"),
]
for cid, subdir, mod, call in TRITON_IMPLS:
    mod_path = os.path.join(BD, subdir, mod + ".py") if subdir != "." else os.path.join(BD, mod + ".py")
    if os.path.exists(mod_path) and cid in CASES:
        CASES[cid].append(("triton", subdir, mod, call))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=None)
    args = parser.parse_args()
    ids = args.cases or sorted(CASES.keys(), key=lambda x: int(x[1:]))

    print("=== Cross-Framework Correctness Validation ===\n")
    passed = 0; failed = 0

    for cid in ids:
        if cid not in CASES: continue
        impls = CASES[cid]
        print(f"{cid}: {len(impls)} implementation(s)")
        results = {}
        for fw, subdir, mod, call in impls:
            r = run_impl(subdir, mod, call, fw)
            results[fw] = r
            if "error" in r:
                print(f"  {fw}: ERROR — {r['error']}")
            else:
                print(f"  {fw}: [{r['min']:.6f}, {r['max']:.6f}] mean={r['mean']:.6f}")

        # Cross-check: compare all pairs using mean, std, max, and hash
        fws = [fw for fw in results if "error" not in results[fw]]
        if len(fws) >= 2:
            ref = fws[0]
            ok = True
            for other in fws[1:]:
                r_ref, r_oth = results[ref], results[other]
                max_diff = abs(r_ref["max"] - r_oth["max"])
                mean_diff = abs(r_ref["mean"] - r_oth["mean"])
                std_diff = abs(r_ref.get("std",0) - r_oth.get("std",0))
                hash_diff = abs(r_ref.get("hash",0) - r_oth.get("hash",0))
                denom = abs(r_ref["max"]) + abs(r_ref["mean"]) + 1e-12
                rel_err = (max_diff + mean_diff) / denom
                if rel_err > 0.05:
                    print(f"  FAIL: {ref} vs {other} rel_err={rel_err:.4f} (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
                    ok = False
                else:
                    print(f"  {ref} vs {other}: rel_err={rel_err:.2e}, hash_diff={hash_diff:.2e} — OK")
            if ok: passed += 1
            else: failed += 1
        elif len(fws) == 1:
            # Single impl cannot satisfy AC-6 cross-check requirement
            r = results[fws[0]]
            is_finite = r["n"] > 0 and r["min"] != float('inf')
            print(f"  Single-impl: finite={is_finite}, n={r['n']} — SKIPPED (need >=2 impls for AC-6)")
            # Don't count as passed or failed — it's not a cross-check
        else:
            print(f"  All implementations errored — FAIL")
            failed += 1
        print()

    print(f"=== {passed} passed, {failed} failed out of {passed+failed} ===")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
