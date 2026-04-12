#!/usr/bin/env python3
"""Unified N×M benchmark matrix runner.

Runs all strategy classes and collects results into a single CSV.
Strategies: cuda, taichi, warp, triton, kokkos, perks, ebisu

Output CSV schema (AC-5):
  case,strategy,gpu,problem_size,steps,median_us,min_us,max_us,overhead_pct
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PYTHON = "/home/scratch.huanhuanc_gpu/spmd/spmd-venv/bin/python"
BENCHMARK_DIR = Path(__file__).parent

CASE_NAMES = {
    "C1": "Jacobi2D", "C2": "Jacobi3D", "C3": "Heat2D", "C4": "Wave2D",
    "C5": "LBM_D2Q9", "C6": "Nbody", "C7": "SPH", "C8": "HydroF1",
    "C9": "HydroF2", "C10": "GrayScott", "C11": "FDTD2D", "C12": "MacCormack3D",
    "C13": "LULESH", "C14": "PIC1D", "C15": "CG_Solver", "C16": "StableFluids",
    "C17": "Conv3D", "C18": "DOITGEN", "C19": "LU", "C20": "ADI",
    "C21": "GramSchmidt",
}

# CUDA configs: binary, [(args_for_size1), (args_for_size2)]
CUDA_CONFIGS = {
    "C1":  ("jacobi2d_bench",  [("256","100","10"), ("4096","100","10")]),
    "C2":  ("jacobi3d_bench",  [("64","100","10"), ("256","100","10")]),
    "C4":  ("wave2d_bench",    [("512","100","10"), ("4096","100","10")]),
    "C5":  ("lbm2d_bench",     [("512","256","100","10"), ("2048","1024","100","10")]),
    "C6":  ("nbody_bench",     [("4096","10","10"), ("32768","10","10")]),
    "C7":  ("sph_bench",       [("8192","10","10"), ("65536","10","10")]),
    "C8":  ("hydro_f1_a100",   [("10","10"), ("10","10",str(BENCHMARK_DIR/"F1_hydro_shallow_water/data_20w/binary/"))]),
    "C9":  ("hydro_osher_a100",[("899","10"), ("900","10",str(BENCHMARK_DIR/"F2_hydro_refactored/data_20w/binary/"))]),
    "C11": ("fdtd2d_bench",    [("512","100","10"), ("4096","100","10")]),
    "C12": ("maccormack3d_bench",[("64","100","10"), ("128","100","10")]),
    "C13": ("lulesh_fusion_a100",[("500","10"), ("500","10")]),
    "C14": ("pic1d_bench",     [("4096","256","100","10"), ("16384","1024","100","10")]),
    "C15": ("cg_fusion_a100",  [("200","10"), ("200","10")]),
    "C16": ("stable_fluids_bench",[("256","5","10"), ("1024","5","10")]),
    "C17": ("conv3d_bench",    [("128","1","10"), ("256","1","10")]),
    "C18": ("doitgen_bench",   [("128","1","10"), ("256","1","10")]),
    "C19": ("lu_bench",        [("512","10"), ("1024","10")]),
    "C20": ("adi_bench",       [("256","3","10"), ("512","3","10")]),
    "C21": ("gramschmidt_bench",[("128","10"), ("256","10")]),
}

# C3/C10 use overhead_solutions which runs all sizes internally
# Parse the table output for these

KOKKOS_CONFIGS = {
    "C1":  ("cpp/kokkos/build-cuda/jacobi_2d_kokkos", [("256","100","10"), ("4096","100","10")]),
    "C8":  ("cpp/kokkos/build-cuda/hydro_swe_kokkos", [("--real","10","10"), ("--real","10","10",str(BENCHMARK_DIR/"F1_hydro_shallow_water/data_20w/binary/"))]),
    "C9":  ("cpp/kokkos/build-cuda/hydro_refactored_kokkos", [("899","10"), ("900","10",str(BENCHMARK_DIR/"F2_hydro_refactored/data_20w/binary/"))]),
}


def detect_gpu():
    try:
        r = subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],
                          capture_output=True, text=True, timeout=5)
        return r.stdout.strip().split("\n")[0]
    except Exception:
        return "unknown"


def parse_cuda_output(output):
    """Parse all CUDA benchmark output formats. Returns list of (strategy, us_per_step) + gpu_compute."""
    results = []
    gpu_compute = None

    for line in output.split("\n"):
        # Format 1: [Strategy Name] N steps: median=X.XXX ms, Y.YY us/step
        m = re.search(r'\[([^\]]+)\].*?(\d+\.?\d*)\s*us/step', line)
        if m:
            results.append((m.group(1).strip(), float(m.group(2))))
            continue

        # Format 2: [Strategy] N/A (reason)
        m2 = re.search(r'\[([^\]]+)\]\s*N/A', line)
        if m2:
            results.append((m2.group(1).strip(), None))
            continue

        # Format 3: [Strategy] median=X.XXX ms, Y.YY us/launch (N launches)
        m3 = re.search(r'\[([^\]]+)\]\s*median=[\d.]+\s*ms,\s*([\d.]+)\s*us/launch', line)
        if m3:
            results.append((m3.group(1).strip(), float(m3.group(2))))
            continue

        # Format 4: [N] Strategy description: Y.Y us/step
        m4 = re.search(r'\[\d+\]\s+(.+?):\s+([\d.]+)\s*us/step', line)
        if m4:
            results.append((m4.group(1).strip(), float(m4.group(2))))
            continue

        # Format 5: [N] Strategy: N/A (reason)
        m5 = re.search(r'\[\d+\]\s+(.+?):\s*N/A', line)
        if m5:
            results.append((m5.group(1).strip(), None))
            continue

        # GPU compute baseline
        m6 = re.search(r'GPU.*?(\d+\.?\d+)\s*us/step', line)
        if m6 and ('compute' in line.lower() or 'total' in line.lower()):
            gpu_compute = float(m6.group(1))

    return results, gpu_compute


def parse_overhead_solutions(output):
    """Parse overhead_solutions table output for C3 (Heat2D) and C10 (GrayScott)."""
    results = []
    for line in output.split("\n"):
        # Format: Heat2D 128sq    12.44    6.19    3.51    3.35  |  ...
        m = re.match(r'\s*(Heat2D|GrayScott)\s+(\d+)sq\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.N/A]+)', line)
        if m:
            kernel, size, sync, async_v, graph, persist = m.groups()
            case = "C3" if kernel == "Heat2D" else "C10"
            size_str = f"{size}x{size}"
            results.append((case, size_str, "Sync", float(sync)))
            results.append((case, size_str, "Async", float(async_v)))
            results.append((case, size_str, "Graph", float(graph)))
            if persist != "N/A":
                results.append((case, size_str, "Persistent", float(persist)))
            else:
                results.append((case, size_str, "Persistent", None))
    return results


def run_cuda_binary(case_id, gpu_name, dry_run=False):
    """Run a CUDA benchmark binary."""
    if case_id not in CUDA_CONFIGS:
        return []
    binary, args_list = CUDA_CONFIGS[case_id]
    binary_path = BENCHMARK_DIR / binary
    if not binary_path.exists():
        print(f"    SKIP: {binary_path} not found")
        return []

    all_results = []
    for args in args_list:
        size_str = args[0] if args else "default"
        cmd = [str(binary_path)] + list(args)
        if dry_run:
            print(f"    {' '.join(cmd)}")
            continue
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            timings, gpu_compute = parse_cuda_output(r.stdout)
            for strat, us in timings:
                oh = ""
                if us is not None and gpu_compute is not None and us > 0:
                    oh = f"{max(0, (us - gpu_compute) / us * 100):.1f}"
                all_results.append({
                    "case": CASE_NAMES[case_id], "strategy": f"CUDA_{strat}",
                    "gpu": gpu_name, "problem_size": size_str,
                    "steps": args[1] if len(args) > 1 else "",
                    "median_us": f"{us:.2f}" if us is not None else "N/A",
                    "min_us": "", "max_us": "", "overhead_pct": oh,
                })
        except Exception as e:
            print(f"    ERROR: {e}")
    return all_results


def run_overhead_solutions(gpu_name, dry_run=False):
    """Run overhead_solutions for C3 (Heat2D) and C10 (GrayScott)."""
    binary = BENCHMARK_DIR / "overhead_solutions_a100"
    if not binary.exists():
        return []
    if dry_run:
        print(f"    {binary}")
        return []
    try:
        r = subprocess.run([str(binary)], capture_output=True, text=True, timeout=300)
        entries = parse_overhead_solutions(r.stdout)
        results = []
        for case_id, size_str, strat, us in entries:
            results.append({
                "case": CASE_NAMES[case_id], "strategy": f"CUDA_{strat}",
                "gpu": gpu_name, "problem_size": size_str,
                "steps": "100", "median_us": f"{us:.2f}" if us is not None else "N/A",
                "min_us": "", "max_us": "", "overhead_pct": "",
            })
        return results
    except Exception as e:
        print(f"    ERROR: {e}")
        return []


def run_taichi_case(case_id, gpu_name, dry_run=False):
    """Run Taichi DSL for a case with proper per-case dispatch."""
    # Map case to module path and call signature
    TAICHI_MAP = {
        "C1":  ("A1_jacobi_2d", "jacobi_taichi", "run(N={sz}, steps={st}, backend='cuda')"),
        "C2":  (".", "jacobi3d_taichi", "run(N={sz}, steps={st}, backend='cuda')"),
        "C3":  (".", "heat2d_taichi", "run(N={sz}, steps={st}, backend='cuda')"),
        "C4":  ("A3_wave_equation", "wave_taichi", "run(N={sz}, steps={st}, backend='cuda')"),
        "C5":  ("A2_lbm_d2q9", "lbm_taichi", "run({sz}, {sz2}, steps={st}, backend='cuda')"),
        "C6":  ("B1_nbody", "nbody_taichi", "run(N={sz}, steps={st}, backend='cuda')"),
        "C7":  ("B2_sph", "sph_taichi", "run(N={sz}, steps={st}, backend='cuda')"),
        "C8":  ("F1_hydro_shallow_water", "hydro_taichi", "run_real(steps={st}, backend='cuda', mesh='{mesh}')"),
        "C9":  ("F2_hydro_refactored", "hydro_refactored_taichi", "run(days=1, backend='cuda', mesh='{mesh}')"),
        "C10": (".", "grayscott_taichi", "run(N={sz}, steps={st}, backend='cuda')"),
        "C11": (".", "fdtd2d_taichi", "run(N={sz}, steps={st}, backend='cuda')"),
        "C12": ("F3_maccormack_3d", "maccormack_taichi", "run(N={sz}, steps={st}, backend='cuda')"),
        "C13": (".", "lulesh_taichi", "run(N={sz}, steps={st}, backend='cuda')"),
        "C14": ("C2_pic", "pic_taichi", "run(N={sz}, steps={st}, backend='cuda')"),
        "C15": (".", "cg_taichi", "run(N={sz}, steps={st}, backend='cuda')"),
        "C16": ("D2_stable_fluids", "fluid_taichi", "run(N={sz}, steps={st}, backend='cuda')"),
        "C17": (".", "conv3d_taichi", "run(N={sz}, steps={st}, backend='cuda')"),
        "C18": (".", "doitgen_taichi", "run(N={sz}, steps={st}, backend='cuda')"),
        "C19": (".", "lu_taichi", "run(N={sz}, steps={st}, backend='cuda')"),
        "C20": (".", "adi_taichi", "run(N={sz}, steps={st}, backend='cuda')"),
        "C21": (".", "gramschmidt_taichi", "run(N={sz}, steps={st}, backend='cuda')"),
    }

    SIZES = {
        "C1": [(256,100),(4096,100)], "C2": [(64,100),(256,100)],
        "C3": [(256,100),(1024,100)], "C4": [(512,100),(4096,100)],
        "C5": [(512,10),(2048,10)], "C6": [(4096,10),(32768,10)],
        "C7": [(8192,10),(65536,10)], "C8": [("default",10),("20w",10)],
        "C9": [("default",1),("20w",1)], "C10": [(128,100),(512,100)],
        "C11": [(512,100),(4096,100)], "C12": [(64,100),(128,100)],
        "C13": [(32,10),(64,10)], "C14": [(4096,100),(16384,100)],
        "C15": [(128,100),(512,100)], "C16": [(256,5),(1024,5)],
        "C17": [(64,1),(128,1)], "C18": [(64,1),(128,1)],
        "C19": [(256,1),(512,1)], "C20": [(128,3),(256,3)],
        "C21": [(128,1),(256,1)],
    }

    if case_id not in TAICHI_MAP:
        return []

    subdir, module, call_template = TAICHI_MAP[case_id]
    module_dir = str(BENCHMARK_DIR / subdir) if subdir != "." else str(BENCHMARK_DIR)
    results = []

    for sz, st in SIZES.get(case_id, []):
        size_str = str(sz)
        is_mesh = isinstance(sz, str) and sz in ("default", "20w")

        # Build the call string
        call = call_template.format(sz=sz, st=st, sz2=sz//2 if isinstance(sz,int) else 256, mesh=sz)

        code = f"""
import sys, time
sys.path.insert(0, '{module_dir}')
sys.path.insert(0, '{BENCHMARK_DIR}')
import taichi as ti
ti.init(arch=ti.cuda, default_fp=ti.f32)
from {module} import *
s, y, o = {call}
y(); s(); y()
times = []
for _ in range(10):
    y(); t0 = time.perf_counter(); s(); y()
    times.append((time.perf_counter() - t0) * 1e6 / {st})
times.sort()
print(f"RESULT {{times[5]:.2f}} {{times[0]:.2f}} {{times[9]:.2f}}")
"""
        if dry_run:
            print(f"    Taichi {case_id} [{size_str}]")
            continue
        try:
            r = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True,
                             timeout=300, cwd=str(BENCHMARK_DIR))
            m = re.search(r'RESULT ([\d.]+) ([\d.]+) ([\d.]+)', r.stdout)
            if m:
                results.append({
                    "case": CASE_NAMES[case_id], "strategy": "Taichi",
                    "gpu": gpu_name, "problem_size": size_str, "steps": str(st),
                    "median_us": m.group(1), "min_us": m.group(2), "max_us": m.group(3),
                    "overhead_pct": "",
                })
            else:
                print(f"    Taichi {case_id} [{size_str}]: no output")
                if r.stderr:
                    print(f"      stderr: {r.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print(f"    Taichi {case_id} [{size_str}]: TIMEOUT")
        except Exception as e:
            print(f"    Taichi {case_id} [{size_str}]: {e}")
    return results


def run_kokkos(case_id, gpu_name, dry_run=False):
    """Run Kokkos benchmark."""
    if case_id not in KOKKOS_CONFIGS:
        return []
    binary, args_list = KOKKOS_CONFIGS[case_id]
    binary_path = BENCHMARK_DIR / binary
    if not binary_path.exists():
        return []
    results = []
    for args in args_list:
        size_str = args[0] if args else "default"
        cmd = [str(binary_path)] + list(args)
        if dry_run:
            print(f"    Kokkos: {' '.join(cmd)}")
            continue
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            m = re.search(r'median=([\d.]+)ms', r.stdout)
            if m:
                total_ms = float(m.group(1))
                steps_m = re.search(r'steps=(\d+)', r.stdout)
                steps_val = int(steps_m.group(1)) if steps_m else 100
                us = total_ms * 1000 / steps_val
                results.append({
                    "case": CASE_NAMES[case_id], "strategy": "Kokkos",
                    "gpu": gpu_name, "problem_size": size_str, "steps": str(steps_val),
                    "median_us": f"{us:.2f}", "min_us": "", "max_us": "", "overhead_pct": "",
                })
        except Exception as e:
            print(f"    Kokkos {case_id}: {e}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run N×M benchmark matrix")
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--strategies", nargs="+",
                        default=["cuda", "taichi", "kokkos"],
                        help="cuda, taichi, warp, triton, kokkos, perks, ebisu")
    parser.add_argument("--output", default="matrix_results.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gpu_name = detect_gpu()
    print(f"GPU: {gpu_name}")
    print(f"Strategies: {args.strategies}")
    print(f"Output: {args.output}")

    case_ids = args.cases or sorted(CASE_NAMES.keys(), key=lambda x: int(x[1:]))
    all_results = []

    # C3/C10 special handling (overhead_solutions)
    if "cuda" in args.strategies:
        if any(c in case_ids for c in ("C3", "C10")):
            print("\n=== C3/C10: overhead_solutions ===")
            results = run_overhead_solutions(gpu_name, args.dry_run)
            all_results.extend(results)
            if results:
                print(f"  CUDA: {len(results)} entries")

    for case_id in case_ids:
        if case_id in ("C3", "C10") and "cuda" in args.strategies:
            continue  # already handled above
        print(f"\n=== {case_id}: {CASE_NAMES.get(case_id, case_id)} ===")

        if "cuda" in args.strategies:
            results = run_cuda_binary(case_id, gpu_name, args.dry_run)
            all_results.extend(results)
            if results:
                print(f"  CUDA: {len(results)} entries")

        if "taichi" in args.strategies:
            results = run_taichi_case(case_id, gpu_name, args.dry_run)
            all_results.extend(results)
            if results:
                print(f"  Taichi: {len(results)} entries")

        if "kokkos" in args.strategies:
            results = run_kokkos(case_id, gpu_name, args.dry_run)
            all_results.extend(results)
            if results:
                print(f"  Kokkos: {len(results)} entries")

    if not args.dry_run and all_results:
        fields = ["case","strategy","gpu","problem_size","steps",
                   "median_us","min_us","max_us","overhead_pct"]
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_results)
        print(f"\nResults: {args.output} ({len(all_results)} rows)")


if __name__ == "__main__":
    main()
