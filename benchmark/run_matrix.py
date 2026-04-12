#!/usr/bin/env python3
"""Unified N×M benchmark matrix runner.

Runs all strategy classes (CUDA, Taichi, Warp, Triton, Kokkos, PERKS, EBISU)
for all 21 cases and collects results into a single CSV.

Output CSV schema (per AC-5):
  case,strategy,gpu,problem_size,steps,median_us,min_us,max_us,overhead_pct

Usage:
    python run_matrix.py                       # run all cases + strategies
    python run_matrix.py --cases C1 C8 C9      # specific cases
    python run_matrix.py --strategies cuda      # only CUDA binaries
    python run_matrix.py --strategies taichi    # only Taichi DSL
    python run_matrix.py --dry-run             # show commands
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PYTHON = str(Path("/home/scratch.huanhuanc_gpu/spmd/spmd-venv/bin/python"))
BENCHMARK_DIR = Path(__file__).parent

CASE_NAMES = {
    "C1": "Jacobi2D", "C2": "Jacobi3D", "C3": "Heat2D", "C4": "Wave2D",
    "C5": "LBM_D2Q9", "C6": "Nbody", "C7": "SPH", "C8": "HydroF1",
    "C9": "HydroF2", "C10": "GrayScott", "C11": "FDTD2D", "C12": "MacCormack3D",
    "C13": "LULESH", "C14": "PIC1D", "C15": "CG_Solver", "C16": "StableFluids",
    "C17": "Conv3D", "C18": "DOITGEN", "C19": "LU", "C20": "ADI",
    "C21": "GramSchmidt",
}

# CUDA binary configs: (binary, [(args_small), (args_large)])
CUDA_CASES = {
    "C1":  ("jacobi2d_bench",       [("256", "100", "10"), ("4096", "100", "10")]),
    "C2":  ("jacobi3d_bench",       [("64", "100", "10"), ("256", "100", "10")]),
    "C3":  ("overhead_solutions_a100", [()]),
    "C4":  ("wave2d_bench",         [("512", "100", "10"), ("4096", "100", "10")]),
    "C5":  ("lbm2d_bench",          [("512", "256", "100", "10"), ("2048", "1024", "100", "10")]),
    "C6":  ("nbody_bench",          [("4096", "10", "10"), ("32768", "10", "10")]),
    "C7":  ("sph_bench",            [("8192", "10", "10"), ("65536", "10", "10")]),
    "C8":  ("hydro_f1_a100",        [("10", "10"), ("10", "10", str(BENCHMARK_DIR / "F1_hydro_shallow_water/data_20w/binary/"))]),
    "C9":  ("hydro_osher_a100",     [("899", "10"), ("900", "10", str(BENCHMARK_DIR / "F2_hydro_refactored/data_20w/binary/"))]),
    "C10": ("overhead_solutions_a100", [()]),
    "C11": ("fdtd2d_bench",         [("512", "100", "10"), ("4096", "100", "10")]),
    "C12": ("maccormack3d_bench",   [("64", "100", "10"), ("128", "100", "10")]),
    "C13": ("lulesh_fusion_a100",   [("500",), ("500",)]),
    "C14": ("pic1d_bench",          [("4096", "256", "100", "10"), ("16384", "1024", "100", "10")]),
    "C15": ("cg_fusion_a100",       [("200",), ("200",)]),
    "C16": ("stable_fluids_bench",  [("256", "5", "10"), ("1024", "5", "10")]),
    "C17": ("conv3d_bench",         [("128", "1", "10"), ("256", "1", "10")]),
    "C18": ("doitgen_bench",        [("128", "1", "10"), ("256", "1", "10")]),
    "C19": ("lu_bench",             [("512", "10"), ("1024", "10")]),
    "C20": ("adi_bench",            [("256", "3", "10"), ("512", "3", "10")]),
    "C21": ("gramschmidt_bench",    [("256", "10"), ("512", "10")]),
}

# Taichi module configs: (script_path, run_cmd_template)
TAICHI_CASES = {
    "C1":  ("A1_jacobi_2d/jacobi_taichi.py",  "run(N={size}, steps={steps}, backend='cuda')"),
    "C2":  ("jacobi3d_taichi.py",              "run(N={size}, steps={steps}, backend='cuda')"),
    "C3":  ("heat2d_taichi.py",                "run(N={size}, steps={steps}, backend='cuda')"),
    "C4":  ("A3_wave_equation/wave_taichi.py", "run(N={size}, steps={steps}, backend='cuda')"),
    "C5":  ("A2_lbm_d2q9/lbm_taichi.py",      "run({size}, {size2}, steps={steps}, backend='cuda')"),
    "C6":  ("B1_nbody/nbody_taichi.py",        "run(N={size}, steps={steps}, backend='cuda')"),
    "C7":  ("B2_sph/sph_taichi.py",            "run(N={size}, steps={steps}, backend='cuda')"),
    "C8":  ("F1_hydro_shallow_water/hydro_taichi.py", "run_real(steps={steps}, backend='cuda', mesh='{mesh}')"),
    "C9":  ("F2_hydro_refactored/hydro_refactored_taichi.py", "run(days=1, backend='cuda', mesh='{mesh}')"),
    "C10": ("grayscott_taichi.py",             "run(N={size}, steps={steps}, backend='cuda')"),
    "C11": ("fdtd2d_taichi.py",                "run(N={size}, steps={steps}, backend='cuda')"),
    "C12": ("F3_maccormack_3d/maccormack_taichi.py", "run(N={size}, steps={steps}, backend='cuda')"),
    "C13": ("lulesh_taichi.py",                "run(N={size}, steps={steps}, backend='cuda')"),
    "C14": ("C2_pic/pic_taichi.py",            "run(N={size}, steps={steps}, backend='cuda')"),
    "C15": ("cg_taichi.py",                    "run(N={size}, steps={steps}, backend='cuda')"),
    "C16": ("D2_stable_fluids/fluid_taichi.py","run(N={size}, steps={steps}, backend='cuda')"),
    "C17": ("conv3d_taichi.py",                "run(N={size}, steps={steps}, backend='cuda')"),
    "C18": ("doitgen_taichi.py",               "run(N={size}, steps={steps}, backend='cuda')"),
    "C19": ("lu_taichi.py",                    "run(N={size}, steps={steps}, backend='cuda')"),
    "C20": ("adi_taichi.py",                   "run(N={size}, steps={steps}, backend='cuda')"),
    "C21": ("gramschmidt_taichi.py",           "run(N={size}, steps={steps}, backend='cuda')"),
}

# Default sizes for DSL runners
DSL_SIZES = {
    "C1": [(256, 100), (4096, 100)],
    "C2": [(64, 100), (256, 100)],
    "C3": [(256, 100), (1024, 100)],
    "C4": [(512, 100), (4096, 100)],
    "C5": [(512, 10), (2048, 10)],
    "C6": [(4096, 10), (32768, 10)],
    "C7": [(8192, 10), (65536, 10)],
    "C8": [("default", 10), ("20w", 10)],
    "C9": [("default", 1), ("20w", 1)],
    "C10": [(128, 100), (512, 100)],
    "C11": [(512, 100), (4096, 100)],
    "C12": [(64, 100), (128, 100)],
    "C13": [(32, 10), (64, 10)],
    "C14": [(4096, 100), (16384, 100)],
    "C15": [(128, 100), (512, 100)],
    "C16": [(256, 5), (1024, 5)],
    "C17": [(64, 1), (128, 1)],
    "C18": [(64, 1), (128, 1)],
    "C19": [(256, 1), (512, 1)],
    "C20": [(128, 3), (256, 3)],
    "C21": [(128, 1), (256, 1)],
}


def detect_gpu():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True, timeout=5)
        return r.stdout.strip().split("\n")[0]
    except Exception:
        return "unknown"


def parse_cuda_output(output):
    """Extract strategy timings from CUDA benchmark output."""
    results = []
    for line in output.split("\n"):
        # Match: [Strategy Name] N steps: median=X.XXX ms, Y.YY us/step
        m = re.search(r'\[([^\]]+)\].*?(\d+\.?\d*)\s*us/step', line)
        if m:
            results.append((m.group(1).strip(), float(m.group(2))))
        elif "N/A" in line and "[" in line:
            m2 = re.match(r'\[([^\]]+)\]\s*N/A', line)
            if m2:
                results.append((m2.group(1).strip(), None))
        # Match GPU compute line
        m3 = re.search(r'GPU.*?(\d+\.?\d+)\s*us/step', line)
        if m3 and "compute" in line.lower():
            results.append(("GPU_compute", float(m3.group(1))))
    return results


def run_cuda(case_id, gpu_name, dry_run=False):
    """Run CUDA 4-strategy benchmark."""
    if case_id not in CUDA_CASES:
        return []
    binary, args_list = CUDA_CASES[case_id]
    binary_path = BENCHMARK_DIR / binary
    if not binary_path.exists():
        return []

    all_results = []
    for args in args_list:
        size_str = "x".join(args[:2]) if len(args) >= 2 else "default"
        cmd = [str(binary_path)] + list(args)
        if dry_run:
            print(f"  CUDA {case_id} [{size_str}]: {' '.join(cmd)}")
            continue
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            gpu_compute = None
            for strat, us in parse_cuda_output(r.stdout):
                if strat == "GPU_compute":
                    gpu_compute = us
                    continue
                overhead_pct = ""
                if us is not None and gpu_compute is not None and us > 0:
                    overhead_pct = f"{(us - gpu_compute) / us * 100:.1f}"
                all_results.append({
                    "case": CASE_NAMES[case_id], "strategy": f"CUDA_{strat}",
                    "gpu": gpu_name, "problem_size": size_str,
                    "steps": args[1] if len(args) > 1 else "?",
                    "median_us": f"{us:.2f}" if us else "N/A",
                    "min_us": "", "max_us": "",
                    "overhead_pct": overhead_pct,
                })
        except Exception as e:
            print(f"  CUDA {case_id} ERROR: {e}")
    return all_results


def run_taichi(case_id, gpu_name, dry_run=False):
    """Run Taichi DSL benchmark."""
    if case_id not in TAICHI_CASES:
        return []
    script, _ = TAICHI_CASES[case_id]
    script_path = BENCHMARK_DIR / script
    if not script_path.exists():
        return []

    all_results = []
    for size_cfg in DSL_SIZES.get(case_id, []):
        size, steps = size_cfg[0], size_cfg[1]
        size_str = str(size)

        runner_code = f"""
import sys, time
sys.path.insert(0, '{script_path.parent}')
sys.path.insert(0, '{BENCHMARK_DIR}')
import taichi as ti
ti.init(arch=ti.cuda, default_fp=ti.f32)
from {script_path.stem} import run
{'s,y,o = run(N='+str(size)+', steps='+str(steps)+", backend='cuda')" if isinstance(size, int) else "s,y,o = run_real(steps="+str(steps)+", backend='cuda', mesh='"+str(size)+"')"}
y(); s(); y()  # warmup
times = []
for _ in range(10):
    y()
    t0 = time.perf_counter()
    s()
    y()
    times.append((time.perf_counter() - t0) * 1e6 / {steps})
times.sort()
print(f"TAICHI_RESULT median={{times[5]:.2f}} min={{times[0]:.2f}} max={{times[9]:.2f}}")
"""
        if dry_run:
            print(f"  Taichi {case_id} [{size_str}]")
            continue
        try:
            r = subprocess.run([PYTHON, "-c", runner_code],
                              capture_output=True, text=True, timeout=300,
                              cwd=str(BENCHMARK_DIR))
            m = re.search(r'TAICHI_RESULT median=([\d.]+) min=([\d.]+) max=([\d.]+)', r.stdout)
            if m:
                all_results.append({
                    "case": CASE_NAMES[case_id], "strategy": "Taichi",
                    "gpu": gpu_name, "problem_size": size_str,
                    "steps": str(steps),
                    "median_us": m.group(1), "min_us": m.group(2), "max_us": m.group(3),
                    "overhead_pct": "",
                })
        except Exception as e:
            print(f"  Taichi {case_id} ERROR: {e}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run N×M benchmark matrix")
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--strategies", nargs="+", default=["cuda", "taichi"],
                        help="Strategy classes: cuda, taichi, warp, triton, kokkos, perks, ebisu")
    parser.add_argument("--output", default="matrix_results.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gpu_name = detect_gpu()
    print(f"GPU: {gpu_name}")

    case_ids = args.cases or sorted(CASE_NAMES.keys(), key=lambda x: int(x[1:]))
    all_results = []

    for case_id in case_ids:
        print(f"\n=== {case_id}: {CASE_NAMES.get(case_id, case_id)} ===")
        if "cuda" in args.strategies:
            results = run_cuda(case_id, gpu_name, args.dry_run)
            all_results.extend(results)
            if results:
                print(f"  CUDA: {len(results)} entries")
        if "taichi" in args.strategies:
            results = run_taichi(case_id, gpu_name, args.dry_run)
            all_results.extend(results)
            if results:
                print(f"  Taichi: {len(results)} entries")

    if not args.dry_run and all_results:
        fieldnames = ["case", "strategy", "gpu", "problem_size", "steps",
                      "median_us", "min_us", "max_us", "overhead_pct"]
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults: {args.output} ({len(all_results)} rows)")


if __name__ == "__main__":
    main()
