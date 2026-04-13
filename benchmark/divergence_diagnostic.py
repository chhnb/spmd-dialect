#!/usr/bin/env python3
"""First-divergence diagnostic for AC-6 failing cases.

Runs each case step-by-step comparing GPU vs CPU after each checkpoint
to identify the first step where divergence exceeds 5%.

Usage:
    python divergence_diagnostic.py [--cases C11 C14 C18 C21]
"""
import subprocess, tempfile, numpy as np, argparse, os

PYTHON = "/home/scratch.huanhuanc_gpu/spmd/spmd-venv/bin/python"
BD = os.path.dirname(os.path.abspath(__file__))

CASES = {
    "C11": {
        "mod": "fdtd2d_taichi",
        "call": "N={N},steps={st},backend='{backend}'",
        "N": 64,
        "checkpoints": [1, 2, 5, 10, 20, 30, 50, 100],
    },
    "C14": {
        "mod": "pic_taichi",
        "subdir": "C2_pic",
        "call": "n_particles={N},n_grid=128,steps={st},backend='{backend}'",
        "N": 1024,
        "checkpoints": [1, 2, 5, 10, 20, 50, 100],
    },
    "C18": {
        "mod": "doitgen_taichi",
        "call": "N={N},steps={st},backend='{backend}'",
        "N": 32,
        "checkpoints": [1, 2, 5, 10, 20, 50, 100],
    },
    "C21": {
        "mod": "gramschmidt_taichi",
        "call": "N={N},steps={st},backend='{backend}'",
        "N": 32,
        "checkpoints": [1, 5, 10, 50, 100],
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


def diagnose(cid, cfg):
    """Run stepwise diagnostic for a case."""
    print(f"\n=== {cid}: First-divergence diagnostic ===")
    mod = cfg["mod"]
    call = cfg["call"]
    N = cfg["N"]
    subdir = cfg.get("subdir")

    for st in cfg["checkpoints"]:
        a_gpu, nan_gpu = run_case(mod, call, N, "cuda", st, subdir)
        a_cpu, nan_cpu = run_case(mod, call, N, "cpu", st, subdir)

        if a_gpu is None or a_cpu is None:
            print(f"  steps={st:>4d}: GPU_NaN={nan_gpu>0!s:>5s}  CPU_NaN={nan_cpu>0!s:>5s}")
            if nan_gpu > 0 and nan_cpu > 0:
                print(f"  → Both produce NaN at step {st}.")
            continue

        if a_gpu.shape != a_cpu.shape:
            print(f"  steps={st:>4d}: shape mismatch {a_gpu.shape} vs {a_cpu.shape}")
            continue

        diff = np.abs(a_gpu - a_cpu)
        denom = np.maximum(np.abs(a_gpu), np.abs(a_cpu)) + 1e-12
        max_rel = float((diff / denom).max())
        status = "OK" if max_rel <= 0.05 else "DIVERGED"
        print(f"  steps={st:>4d}: max_rel_error={max_rel:.6f}  [{status}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=list(CASES.keys()))
    args = parser.parse_args()

    for cid in args.cases:
        if cid in CASES:
            diagnose(cid, CASES[cid])

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
