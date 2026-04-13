#!/usr/bin/env python3
"""First-divergence diagnostic for AC-6 failing cases.

Runs each case at dense step checkpoints comparing GPU vs CPU (and NumPy
where available) to identify the step range where divergence exceeds 5%.
Checkpoints are dense around known transition regions (NaN onset, divergence).

Usage:
    python divergence_diagnostic.py [--cases C11 C14 C18 C21] [--sparse]
"""
import subprocess, tempfile, numpy as np, argparse, os

PYTHON = "/home/scratch.huanhuanc_gpu/spmd/spmd-venv/bin/python"
BD = os.path.dirname(os.path.abspath(__file__))

CASES = {
    "C11": {
        "mod": "fdtd2d_taichi",
        "call": "N={N},steps={st},backend='{backend}'",
        "N": 64,
        # Dense around NaN transition (step 50-100)
        "checkpoints": list(range(1, 11)) + list(range(15, 55, 5)) + list(range(55, 101)),
    },
    "C14": {
        "mod": "pic_taichi",
        "subdir": "C2_pic",
        "call": "n_particles={N},n_grid=128,steps={st},backend='{backend}'",
        "N": 1024,
        # Dense around divergence (step 50-100)
        "checkpoints": list(range(1, 11)) + list(range(15, 55, 5)) + list(range(55, 101)),
    },
    "C18": {
        "mod": "doitgen_taichi",
        "call": "N={N},steps={st},backend='{backend}'",
        "N": 32,
        # Dense around NaN transition (step 20-50)
        "checkpoints": list(range(1, 11)) + list(range(15, 55, 5)) + list(range(45, 55)),
    },
    "C21": {
        "mod": "gramschmidt_taichi",
        "call": "N={N},steps={st},backend='{backend}'",
        "N": 32,
        # Dense at start (diverges at step 1)
        "checkpoints": list(range(1, 11)) + [20, 50, 100],
        "numpy_ref": {
            "mod": "numpy_refs",
            "call": "run_gramschmidt(N={N},steps={st})",
        },
    },
}


def run_case(mod, call_tpl, N, backend, st, subdir=None):
    """Run a single case at given step count and return (array, nan_count) or None."""
    npy = tempfile.mktemp(suffix=".npy")
    call = call_tpl.format(N=N, st=st, backend=backend)
    path_inserts = f'sys.path.insert(0, "{BD}")'
    if subdir:
        path_inserts = f'sys.path.insert(0, "{BD}/{subdir}"); {path_inserts}'
    code = f"""
import sys, numpy as np
{path_inserts}
from {mod} import run
s, y, o = run({call})
s(); y()
a = o.to_numpy().flatten().astype(np.float32)
np.save("{npy}", a)
nan_ct = int(np.isnan(a).sum())
print(f"n={{len(a)}} nan={{nan_ct}}")
"""
    try:
        r = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True,
                           timeout=300, cwd=BD)
        for line in r.stdout.split("\n"):
            if line.startswith("n="):
                nan_ct = int(line.split("nan=")[1])
                if nan_ct == 0 and os.path.exists(npy):
                    return np.load(npy), 0
                return None, nan_ct
    except Exception as e:
        return None, -1
    return None, -1


def run_numpy_ref(mod, call_tpl, N, st):
    """Run NumPy reference implementation."""
    npy = tempfile.mktemp(suffix=".npy")
    call = call_tpl.format(N=N, st=st)
    code = f"""
import sys, numpy as np
sys.path.insert(0, "{BD}")
from {mod} import *
_, _, o = {call}
a = np.array(o).flatten().astype(np.float32) if not isinstance(o, np.ndarray) else o.flatten().astype(np.float32)
np.save("{npy}", a)
nan_ct = int(np.isnan(a).sum())
print(f"n={{len(a)}} nan={{nan_ct}}")
"""
    try:
        r = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True,
                           timeout=300, cwd=BD)
        for line in r.stdout.split("\n"):
            if line.startswith("n="):
                nan_ct = int(line.split("nan=")[1])
                if nan_ct == 0 and os.path.exists(npy):
                    return np.load(npy), 0
                return None, nan_ct
    except Exception:
        pass
    return None, -1


def compare(a, b):
    """Compute max relative error between two arrays."""
    if a is None or b is None or a.shape != b.shape:
        return float("inf")
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b)) + 1e-12
    return float((diff / denom).max())


def diagnose(cid, cfg):
    """Run stepwise diagnostic for a case, every step from 1 to 100."""
    print(f"\n=== {cid}: First-divergence diagnostic (steps 1-100) ===")
    mod = cfg["mod"]
    call = cfg["call"]
    N = cfg["N"]
    subdir = cfg.get("subdir")
    numpy_cfg = cfg.get("numpy_ref")
    first_diverge = None
    first_nan = None

    for st in cfg["checkpoints"]:
        a_gpu, nan_gpu = run_case(mod, call, N, "cuda", st, subdir)
        a_cpu, nan_cpu = run_case(mod, call, N, "cpu", st, subdir)

        # Optional NumPy reference
        a_np = None
        if numpy_cfg:
            a_np, nan_np = run_numpy_ref(numpy_cfg["mod"], numpy_cfg["call"], N, st)

        if a_gpu is None or a_cpu is None:
            if first_nan is None:
                first_nan = st
            line = f"  steps={st:>4d}: GPU_NaN={nan_gpu>0!s:>5s}  CPU_NaN={nan_cpu>0!s:>5s}"
            if a_np is not None:
                line += f"  NumPy_available=True"
            elif numpy_cfg:
                line += f"  NumPy_NaN=True"
            print(line)
            if nan_gpu > 0 and nan_cpu > 0:
                print(f"  → Both produce NaN at step {st}.")
            continue

        max_rel = compare(a_gpu, a_cpu)
        status = "OK" if max_rel <= 0.05 else "DIVERGED"
        if status == "DIVERGED" and first_diverge is None:
            first_diverge = st

        line = f"  steps={st:>4d}: GPU_vs_CPU={max_rel:.6f} [{status}]"
        if a_np is not None:
            rel_gpu_np = compare(a_gpu, a_np)
            rel_cpu_np = compare(a_cpu, a_np)
            line += f"  GPU_vs_NumPy={rel_gpu_np:.6f}  CPU_vs_NumPy={rel_cpu_np:.6f}"
        print(line)

    # Summary
    if first_diverge:
        print(f"  SUMMARY: First divergence >5% at step {first_diverge}")
    if first_nan:
        print(f"  SUMMARY: First NaN at step {first_nan}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=list(CASES.keys()))
    parser.add_argument("--sparse", action="store_true",
                        help="Use sparse checkpoints (1,2,3,5,10,20,50,100) instead of every step")
    args = parser.parse_args()

    for cid in args.cases:
        if cid in CASES:
            cfg = dict(CASES[cid])
            if args.sparse:
                cfg["checkpoints"] = [1, 2, 3, 5, 10, 20, 50, 100]
            diagnose(cid, cfg)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
