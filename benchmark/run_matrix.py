#!/usr/bin/env python3
"""Unified N×M benchmark matrix runner.

Dispatches: cuda, taichi, warp, triton, tilelang, kokkos, perks, ebisu
CSV schema: case,strategy,gpu,problem_size,steps,median_us,min_us,max_us,overhead_pct
"""
import argparse, csv, os, re, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

PYTHON = "/home/scratch.huanhuanc_gpu/spmd/spmd-venv/bin/python"
BD = Path(__file__).parent

CN = {"C1":"Jacobi2D","C2":"Jacobi3D","C3":"Heat2D","C4":"Wave2D",
      "C5":"LBM_D2Q9","C6":"Nbody","C7":"SPH","C8":"HydroF1",
      "C9":"HydroF2","C10":"GrayScott","C11":"FDTD2D","C12":"MacCormack3D",
      "C13":"LULESH","C14":"PIC1D","C15":"CG_Solver","C16":"StableFluids",
      "C17":"Conv3D","C18":"DOITGEN","C19":"LU","C20":"ADI","C21":"GramSchmidt"}

# --- CUDA binary configs: binary, [(args, size_label)] ---
CUDA = {
    "C1": ("jacobi2d_bench",  [("256 100 10","256"),("4096 100 10","4096")]),
    "C2": ("jacobi3d_bench",  [("64 100 10","64"),("256 100 10","256")]),
    "C4": ("wave2d_bench",    [("512 100 10","512"),("4096 100 10","4096")]),
    "C5": ("lbm2d_bench",     [("512 256 100 10","512x256"),("2048 1024 100 10","2048x1024")]),
    "C6": ("nbody_bench",     [("4096 10 10","4096"),("32768 10 10","32768")]),
    "C7": ("sph_bench",       [("8192 10 10","8192"),("65536 10 10","65536")]),
    "C8": ("hydro_f1_a100",   [("10 10","6675"),("10 10 "+str(BD/"F1_hydro_shallow_water/data_20w/binary/"),"207234")]),
    "C9": ("hydro_osher_a100",[("899 10","24020"),("100 10 "+str(BD/"F2_hydro_refactored/data_20w/binary")+"/","207234")]),
    "C11":("fdtd2d_bench",    [("512 100 10","512"),("4096 100 10","4096")]),
    "C12":("maccormack3d_bench",[("64 100 10","64"),("128 100 10","128")]),
    "C13":("lulesh_fusion_a100",[("32 500","N=32"),("64 500","N=64")]),
    "C14":("pic1d_bench",     [("4096 256 100 10","4096p"),("16384 1024 100 10","16384p")]),
    "C15":("cg_fusion_a100",  [("64 200","N=64"),("256 200","N=256")]),
    "C16":("stable_fluids_bench",[("256 5 10","256"),("1024 5 10","1024")]),
    "C17":("conv3d_bench",    [("128 1 10","128"),("256 1 10","256")]),
    "C18":("doitgen_bench",   [("128 1 10","128"),("256 1 10","256")]),
    "C19":("lu_bench",        [("512 10","512"),("1024 10","1024")]),
    "C20":("adi_bench",       [("256 3 10","256"),("512 3 10","512")]),
    "C21":("gramschmidt_bench",[("128 10","128"),("256 10","256")]),
}

# --- Taichi configs: (subdir, module, call_template, [(size,steps,extra)]) ---
# Each module calls ti.init() internally — DO NOT call ti.init() in wrapper
TAICHI = {
    "C1": ("A1_jacobi_2d","jacobi_taichi","run(N={sz},steps={st},backend='cuda')",[(256,100),(4096,100)]),
    "C2": (".","jacobi3d_taichi","run(N={sz},steps={st},backend='cuda')",[(64,100),(256,100)]),
    "C3": (".","heat2d_taichi","run(N={sz},steps={st},backend='cuda')",[(256,100),(1024,100)]),
    "C4": ("A3_wave_equation","wave_taichi","run(N={sz},steps={st},backend='cuda')",[(512,100),(4096,100)]),
    "C5": ("A2_lbm_d2q9","lbm_taichi","run({sz},{sz2},steps={st},backend='cuda')",[(512,10),(2048,10)]),
    "C6": ("B1_nbody","nbody_taichi","run(N={sz},steps={st},backend='cuda')",[(4096,10),(32768,10)]),
    "C7": ("B2_sph","sph_taichi","run(N={sz},steps={st},backend='cuda')",[(8192,10),(65536,10)]),
    "C8": ("F1_hydro_shallow_water","hydro_taichi","run_real(steps={st},backend='cuda',mesh='{mesh}')",[("default",10),("20w",10)]),
    "C9": ("F2_hydro_refactored","hydro_refactored_taichi","run(days=1,backend='cuda',mesh='{mesh}')",[("default",900),("20w",7200)]),
    "C10":(".","grayscott_taichi","run(N={sz},steps={st},backend='cuda')",[(128,100),(512,100)]),
    "C11":(".","fdtd2d_taichi","run(N={sz},steps={st},backend='cuda')",[(512,100),(4096,100)]),
    "C12":("F3_maccormack_3d","maccormack_taichi","run(N={sz},steps={st},backend='cuda')",[(64,100),(128,100)]),
    "C13":(".","lulesh_taichi","run(N={sz},steps={st},backend='cuda')",[(32,10),(64,10)]),
    "C14":("C2_pic","pic_taichi","run(n_particles={sz},n_grid={g},steps={st},backend='cuda')",[(4096,256,100),(16384,1024,100)]),
    "C15":(".","cg_taichi","run(N={sz},steps={st},backend='cuda')",[(64,100),(256,100)]),
    "C16":("D2_stable_fluids","fluid_taichi","run(N={sz},steps={st},backend='cuda')",[(256,5),(1024,5)]),
    "C17":(".","conv3d_taichi","run(N={sz},steps={st},backend='cuda')",[(128,1),(256,1)]),
    "C18":(".","doitgen_taichi","run(N={sz},steps={st},backend='cuda')",[(128,1),(256,1)]),
    "C19":(".","lu_taichi","run(N={sz},steps={st},backend='cuda')",[(512,1),(1024,1)]),
    "C20":(".","adi_taichi","run(N={sz},steps={st},backend='cuda')",[(256,3),(512,3)]),
    "C21":(".","gramschmidt_taichi","run(N={sz},steps={st},backend='cuda')",[(128,1),(256,1)]),
}

KOKKOS = {
    "C1": ("cpp/kokkos/build-cuda/jacobi_2d_kokkos",[("256 100 10","256"),("4096 100 10","4096")]),
    "C8": ("cpp/kokkos/build-cuda/hydro_swe_kokkos",[("--real 10 10","6675"),("--real 10 10 "+str(BD/"F1_hydro_shallow_water/data_20w/binary/"),"207234")]),
    "C9": ("cpp/kokkos/build-cuda/hydro_refactored_kokkos",[("899 10","24020"),("100 10 "+str(BD/"F2_hydro_refactored/data_20w/binary")+"/","207234")]),
}

# Configured step counts per case (fallback when binary output has no step marker)
CUDA_STEPS = {
    "C1":"100","C2":"100","C4":"100","C5":"100","C6":"10","C7":"10",
    "C8":"10","C9":"900","C11":"100","C12":"100","C13":"500","C14":"100",
    "C15":"200","C16":"5","C17":"1","C18":"1","C19":"1","C20":"3","C21":"1",
}

PERKS_2D = {
    "C1": [("PERKS/stencil/2dstencil/build/init/2d5pt/2d5pt_{strat}.exe",
            "--dimx 4096 --dimy 4096 --iter 100 --fp32 --warmup","4096x4096")],
}
PERKS_3D = {
    "C2": [("PERKS/stencil/3dstencil/build/init/3d7pt/3d7pt_{strat}.exe",
            "256 256 256 100","256x256x256")],
}


def gpu_name():
    try:
        r = subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],
                          capture_output=True, text=True, timeout=5)
        return r.stdout.strip().split("\n")[0]
    except: return "unknown"


def normalize_strategy(name):
    """Map parsed CUDA strategy names to canonical form."""
    n = name.lower().strip()
    if 'sync' in n and 'async' not in n:
        return 'Sync'
    if 'async' in n:
        return 'Async'
    if 'graph' in n:
        return 'Graph'
    if 'persistent' in n or 'fused' in n:
        return 'Persistent'
    # Numbered strategies from CG/LULESH: [1]=Sync, [2]=Async, [3]=Graph, [4]=Persistent
    if n in ('1',): return 'Sync'
    if n in ('2',): return 'Async'
    if n in ('3',): return 'Graph'
    if n in ('4',): return 'Persistent'
    return name  # keep as-is if unrecognized


def parse_cuda(output):
    """Parse CUDA benchmark output → [(strategy, us_per_step)] + gpu_compute."""
    results = []; compute = None
    for line in output.split("\n"):
        # [Name] ... Y.YY us/step
        m = re.search(r'\[([^\]]+)\].*?([\d.]+)\s*us/step', line)
        if m: results.append((m.group(1).strip(), float(m.group(2)))); continue
        # [Name] ... Y.YY us/launch
        m = re.search(r'\[([^\]]+)\].*?median=[\d.]+\s*ms.*?([\d.]+)\s*us/launch', line)
        if m: results.append((m.group(1).strip(), float(m.group(2)))); continue
        # [N] Description: Y.Y us/step
        m = re.search(r'\[\d+\]\s+(.+?):\s+([\d.]+)\s*us/step', line)
        if m: results.append((m.group(1).strip(), float(m.group(2)))); continue
        # N/A or SKIPPED — capture reason from the line
        m = re.search(r'\[([^\]]+)\]\s*(N/A|SKIPPED)\s*(.*)', line)
        if m:
            reason = m.group(3).strip("() ") if m.group(3) else ""
            results.append((m.group(1).strip(), f"N/A ({reason})" if reason else "N/A (no reason given)"))
            continue
        m = re.search(r'\[\d+\]\s+(.+?):\s*(N/A|SKIPPED)\s*(.*)', line)
        if m:
            reason = m.group(3).strip("() ") if m.group(3) else ""
            results.append((m.group(1).strip(), f"N/A ({reason})" if reason else "N/A (no reason given)"))
            continue
        # GPU compute
        m = re.search(r'GPU.*?([\d.]+)\s*us/step', line)
        if m and ('compute' in line.lower() or 'total' in line.lower()):
            compute = float(m.group(1))
    # Normalize strategy names to canonical form
    results = [(normalize_strategy(name), us) for name, us in results]
    return results, compute


def parse_overhead_solutions(output):
    """Parse overhead_solutions table → [(case_id, size, strategy, us, steps)]."""
    rows = []
    # Per overhead_solutions.cu: size-specific step counts
    heat_steps  = {"128":2000,"256":1000,"512":1000,"1024":500,"2048":200}
    gs_steps    = {"128":2000,"256":1000,"512":500,"1024":200}
    for line in output.split("\n"):
        m = re.match(r'\s*(Heat2D|GrayScott)\s+(\d+)sq\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.N/A]+)', line)
        if m:
            kernel,sz,sync,asyn,graph,pers = m.groups()
            cid = "C3" if kernel=="Heat2D" else "C10"
            smap = heat_steps if kernel=="Heat2D" else gs_steps
            st = smap.get(sz, 100)
            for name,val in [("Sync",sync),("Async",asyn),("Graph",graph),("Persistent",pers)]:
                if val == "N/A":
                    us = "N/A (grid exceeds cooperative launch limit)"
                else:
                    us = float(val)
                rows.append((cid, f"{sz}x{sz}", name, us, st))
    return rows


def run_binary(binary, args_str, size_label, case_id, gpu, dry_run, strategy_prefix="CUDA", configured_steps=""):
    path = BD/binary
    if not path.exists(): return []
    cmd = [str(path)] + args_str.split()
    if dry_run: print(f"    {' '.join(cmd)}"); return []
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        timings, compute = parse_cuda(r.stdout)
        # Try to find step count from output (various formats)
        steps_val = ""
        # Note: do NOT match "launches" — that's launch count, not step count
        for pat in [r'(\d+)\s*steps', r'(\d+)\s+CG iterations',
                    r'(\d+)\s+iterations', r'steps[=:]\s*(\d+)',
                    r'tsteps=(\d+)']:
            m_steps = re.search(pat, r.stdout)
            if m_steps:
                steps_val = m_steps.group(1)
                break
        # Fallback to configured step count from args
        if not steps_val and configured_steps:
            steps_val = str(configured_steps)
        rows = []
        seen_strats = set()
        for strat, us in timings:
            is_na = isinstance(us, str) and us.startswith("N/A")
            if not is_na and us is not None and compute is not None and us > 0:
                oh = f"{max(0,(us-compute)/us*100):.1f}"
            elif not is_na and us is not None and compute is None:
                oh = "N/A"
            else:
                oh = ""
            median_val = us if is_na else (f"{us:.2f}" if us else "N/A")
            rows.append({"case":CN[case_id],"strategy":f"{strategy_prefix}_{strat}","gpu":gpu,
                        "problem_size":size_label,"steps":steps_val,
                        "median_us":median_val,
                        "min_us":"","max_us":"","overhead_pct":oh})
            seen_strats.add(strat)
        # Ensure all 4 expected CUDA strategies have a row (AC-1: no silent skips)
        if strategy_prefix == "CUDA":
            # Collect N/A reasons from binary output
            na_reasons = {}
            for line in r.stdout.split("\n"):
                if "N/A" in line or "SKIPPED" in line:
                    for s in ["Sync","Async","Graph","Persistent"]:
                        if s.lower() in line.lower():
                            reason = line.strip()[:80]
                            na_reasons[s] = reason
            for expected in ["Sync", "Async", "Graph", "Persistent"]:
                if expected not in seen_strats:
                    reason = na_reasons.get(expected, "not reported by binary")
                    rows.append({"case":CN[case_id],"strategy":f"CUDA_{expected}",
                                "gpu":gpu,"problem_size":size_label,"steps":steps_val,
                                "median_us":f"N/A ({reason})",
                                "min_us":"","max_us":"","overhead_pct":""})
        return rows
    except Exception as e:
        print(f"    ERROR: {e}"); return []


def run_dsl(case_id, framework, gpu, dry_run):
    """Run a DSL (taichi/warp/triton) for a case."""
    cfg_map = {"taichi": TAICHI}
    if framework not in cfg_map or case_id not in cfg_map[framework]:
        return []
    subdir, mod, call_tpl, sizes = cfg_map[framework][case_id]
    mod_dir = str(BD/subdir) if subdir != "." else str(BD)
    rows = []
    for size_cfg in sizes:
        sz, st = size_cfg[0], size_cfg[-1]
        is_mesh = isinstance(sz, str)
        sz2 = sz//2 if isinstance(sz, int) else 256
        g = size_cfg[1] if len(size_cfg) == 3 else sz2
        call = call_tpl.format(sz=sz, st=st, sz2=sz2, mesh=sz, g=g)
        # Normalize size labels to match CUDA (e.g., "default"→"6675", "20w"→"207234")
        SIZE_LABEL_MAP = {"default": {"C8":"6675","C9":"24020"}, "20w": {"C8":"207234","C9":"207234"}}
        size_label = SIZE_LABEL_MAP.get(str(sz), {}).get(case_id, str(sz))

        # NO ti.init here — the module's run() does it
        code = f"""
import sys,time
sys.path.insert(0,'{mod_dir}')
sys.path.insert(0,'{BD}')
from {mod} import *
s,y,o={call}
y();s();y()
ts=[]
for _ in range(10):
    y();t0=time.perf_counter();s();y()
    ts.append((time.perf_counter()-t0)*1e6/{st})
ts.sort()
print(f"R {{ts[5]:.2f}} {{ts[0]:.2f}} {{ts[9]:.2f}}")
"""
        if dry_run: print(f"    {framework} {case_id} [{size_label}]"); continue
        try:
            r = subprocess.run([PYTHON,"-c",code], capture_output=True, text=True, timeout=300, cwd=str(BD))
            m = re.search(r'R ([\d.]+) ([\d.]+) ([\d.]+)', r.stdout)
            if m:
                rows.append({"case":CN[case_id],"strategy":framework.capitalize(),
                            "gpu":gpu,"problem_size":size_label,"steps":str(st),
                            "median_us":m.group(1),"min_us":m.group(2),"max_us":m.group(3),
                            "overhead_pct":""})
            else:
                err = r.stderr[:150] if r.stderr else r.stdout[:150]
                print(f"    {framework} {case_id} [{size_label}]: no result — {err}")
        except subprocess.TimeoutExpired:
            print(f"    {framework} {case_id} [{size_label}]: TIMEOUT")
        except Exception as e:
            print(f"    {framework} {case_id} [{size_label}]: {e}")
    return rows


def run_warp(case_id, gpu, dry_run):
    """Run Warp DSL for cases with Warp implementations."""
    WARP_MAP = {
        "C1": ("A1_jacobi_2d","jacobi_warp","run(N={sz},steps={st},backend='cuda')",[(256,100),(4096,100)]),
        "C4": ("A3_wave_equation","wave_warp","run(N={sz},steps={st},backend='cuda')",[(512,100),(4096,100)]),
        "C6": ("B1_nbody","nbody_warp","run(N={sz},steps={st},backend='cuda')",[(4096,10),(32768,10)]),
        "C7": ("B2_sph","sph_warp","run(N={sz},steps={st},backend='cuda')",[(8192,10),(65536,10)]),
        "C8": ("F1_hydro_shallow_water","hydro_warp","run_real(steps={st},backend='cuda',mesh='{mesh}')",[("default",10),("20w",10)]),
        "C9": ("F2_hydro_refactored","hydro_refactored_warp","run(days=1,backend='cuda',mesh='{mesh}')",[("default",900),("20w",7200)]),
        "C16":("D2_stable_fluids","fluid_warp","run(N={sz},steps={st},backend='cuda')",[(256,5),(1024,5)]),
    }
    if case_id not in WARP_MAP: return []
    subdir, mod, call_tpl, sizes = WARP_MAP[case_id]
    mod_dir = str(BD/subdir)
    rows = []
    for sz, st in sizes:
        sz2 = sz//2 if isinstance(sz, int) else 256
        call = call_tpl.format(sz=sz, st=st, sz2=sz2, mesh=sz)
        code = f"""
import sys,time,os
os.environ['WARP_CACHE_PATH']='/home/scratch.huanhuanc_gpu/spmd/.warp_cache'
sys.path.insert(0,'{mod_dir}')
sys.path.insert(0,'{BD}')
import warp as wp; wp.init()
from {mod} import *
s,y,o={call}
y();s();y()
ts=[]
for _ in range(10):
    y();t0=time.perf_counter();s();y()
    ts.append((time.perf_counter()-t0)*1e6/{st})
ts.sort()
print(f"R {{ts[5]:.2f}} {{ts[0]:.2f}} {{ts[9]:.2f}}")
"""
        if dry_run: print(f"    warp {case_id} [{sz}]"); continue
        try:
            r = subprocess.run([PYTHON,"-c",code], capture_output=True, text=True, timeout=300, cwd=str(BD))
            m = re.search(r'R ([\d.]+) ([\d.]+) ([\d.]+)', r.stdout)
            if m:
                slm = {"default":{"C8":"6675","C9":"24020"},"20w":{"C8":"207234","C9":"207234"}}
                plabel = slm.get(str(sz),{}).get(case_id, str(sz))
                rows.append({"case":CN[case_id],"strategy":"Warp","gpu":gpu,
                            "problem_size":plabel,"steps":str(st),
                            "median_us":m.group(1),"min_us":m.group(2),"max_us":m.group(3),
                            "overhead_pct":""})
        except Exception as e:
            print(f"    warp {case_id}: {e}")
    return rows


def run_triton(case_id, gpu, dry_run):
    """Run Triton DSL for cases with Triton implementations."""
    TRITON_MAP = {
        "C1": ("A1_jacobi_2d","jacobi_triton","run(N={sz},steps={st},backend='cuda')",[(256,100),(4096,100)]),
        "C8": ("F1_hydro_shallow_water","hydro_triton","run_real(steps={st},backend='cuda',mesh='{mesh}')",[("default",10),("20w",10)]),
        "C9": ("F2_hydro_refactored","hydro_refactored_triton","run(days=1,backend='cuda',mesh='{mesh}')",[("default",900),("20w",7200)]),
    }
    if case_id not in TRITON_MAP: return []
    subdir, mod, call_tpl, sizes = TRITON_MAP[case_id]
    mod_dir = str(BD/subdir)
    rows = []
    for sz, st in sizes:
        sz2 = sz//2 if isinstance(sz, int) else 256
        call = call_tpl.format(sz=sz, st=st, sz2=sz2, mesh=sz)
        code = f"""
import sys,time
sys.path.insert(0,'{mod_dir}')
sys.path.insert(0,'{BD}')
from {mod} import *
s,y,o={call}
y();s();y()
ts=[]
for _ in range(10):
    y();t0=time.perf_counter();s();y()
    ts.append((time.perf_counter()-t0)*1e6/{st})
ts.sort()
print(f"R {{ts[5]:.2f}} {{ts[0]:.2f}} {{ts[9]:.2f}}")
"""
        if dry_run: print(f"    triton {case_id} [{sz}]"); continue
        try:
            r = subprocess.run([PYTHON,"-c",code], capture_output=True, text=True, timeout=300, cwd=str(BD))
            m = re.search(r'R ([\d.]+) ([\d.]+) ([\d.]+)', r.stdout)
            if m:
                slm = {"default":{"C8":"6675","C9":"24020"},"20w":{"C8":"207234","C9":"207234"}}
                plabel = slm.get(str(sz),{}).get(case_id, str(sz))
                rows.append({"case":CN[case_id],"strategy":"Triton","gpu":gpu,
                            "problem_size":plabel,"steps":str(st),
                            "median_us":m.group(1),"min_us":m.group(2),"max_us":m.group(3),
                            "overhead_pct":""})
        except Exception as e:
            print(f"    triton {case_id}: {e}")
    return rows


def run_tilelang(case_id, gpu, dry_run):
    """Run TileLang DSL for cases with TileLang implementations."""
    TILELANG_MAP = {
        "C1": ("A1_jacobi_2d","jacobi_tilelang","run(N={sz},steps={st},backend='cuda')",[(64,100),(256,100),(4096,100)]),
        # C8 TileLang: run_real() loads mesh correctly but JIT compilation of F1 Osher solver
        # is too slow (>10 min per kernel). Correctness validated via test_correctness.py.
        # C9 TileLang: F2 refactored 2-kernel pattern compiles and runs successfully.
        "C9": ("F2_hydro_refactored","hydro_refactored_tilelang","run(days=1,backend='cuda',mesh='{mesh}')",[("default",900)]),
    }
    if case_id not in TILELANG_MAP: return []
    subdir, mod, call_tpl, sizes = TILELANG_MAP[case_id]
    mod_dir = str(BD/subdir)
    rows = []
    for sz, st in sizes:
        call = call_tpl.format(sz=sz, st=st, mesh=sz)
        code = f"""
import sys,time,os
os.environ.setdefault('CUDA_HOME','/home/scratch.huanhuanc_gpu/spmd/cuda-toolkit')
sys.path.insert(0,'{mod_dir}')
sys.path.insert(0,'{BD}')
from {mod} import *
import torch
s,y,o={call}
y();s();y()
ts=[]
for _ in range(10):
    y();t0=time.perf_counter();s();y()
    ts.append((time.perf_counter()-t0)*1e6/{st})
ts.sort()
print(f"R {{ts[5]:.2f}} {{ts[0]:.2f}} {{ts[9]:.2f}}")
"""
        if dry_run: print(f"    tilelang {case_id} [{sz}]"); continue
        try:
            r = subprocess.run([PYTHON,"-c",code], capture_output=True, text=True, timeout=600, cwd=str(BD))
            m = re.search(r'R ([\d.]+) ([\d.]+) ([\d.]+)', r.stdout)
            if m:
                slm = {"default":{"C8":"6675","C9":"24020"},"20w":{"C8":"207234","C9":"207234"}}
                plabel = slm.get(str(sz),{}).get(case_id, str(sz))
                rows.append({"case":CN[case_id],"strategy":"TileLang","gpu":gpu,
                            "problem_size":plabel,"steps":str(st),
                            "median_us":m.group(1),"min_us":m.group(2),"max_us":m.group(3),
                            "overhead_pct":""})
        except Exception as e:
            print(f"    tilelang {case_id}: {e}")
    return rows


def run_perks(case_id, gpu, dry_run):
    """Run PERKS baseline/gen/genwr for stencil cases."""
    configs = {**PERKS_2D, **PERKS_3D}
    if case_id not in configs: return []
    rows = []
    for tpl, args, size_label in configs[case_id]:
        for strat in ["baseline","gen","genwr"]:
            binary = str(BD/tpl.format(strat=strat))
            if not os.path.exists(binary): continue
            cmd = [binary] + args.split()
            if dry_run: print(f"    PERKS {strat}: {' '.join(cmd)}"); continue
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                # Parse: field 11 is time for baseline/gen/genwr
                fields = r.stdout.strip().split("\t")
                time_ms = None
                for f in fields:
                    try:
                        v = float(f)
                        if 0.001 < v < 100 and time_ms is None:
                            pass  # skip SM size
                        elif time_ms is None and 0.001 < v < 100:
                            time_ms = v
                    except: pass
                # Simpler: just take field index 11 (0-based 10)
                if len(fields) > 11:
                    try: time_ms = float(fields[10])
                    except: pass
                if time_ms:
                    rows.append({"case":CN[case_id],"strategy":f"PERKS_{strat}","gpu":gpu,
                                "problem_size":size_label,"steps":"100",
                                "median_us":f"{time_ms*1000/100:.2f}","min_us":"","max_us":"",
                                "overhead_pct":""})
            except Exception as e:
                print(f"    PERKS {case_id} {strat}: {e}")
    return rows


def run_ebisu(case_id, gpu, dry_run):
    """Run EBISU temporal blocking for stencil cases."""
    EBISU_MAP = {
        "C1": ("EBISU/2dstencil/build/init/2d5pt/2d5pt_ebisu.exe",
               "--dimx 4096 --dimy 4096 --iter 100 --fp32 --warmup", "4096x4096"),
        "C2": ("EBISU/3dstencil/build/init/3d7pt/3d7pt_ebisu.exe",
               "256 256 256 100", "256x256x256"),
    }
    if case_id not in EBISU_MAP: return []
    binary, args, size_label = EBISU_MAP[case_id]
    path = BD/binary
    if not path.exists(): return []
    if dry_run: print(f"    EBISU: {path} {args}"); return []
    try:
        r = subprocess.run([str(path)]+args.split(), capture_output=True, text=True, timeout=60)
        fields = r.stdout.strip().split()
        # EBISU output: time is one of the later float fields
        time_ms = None
        for f in fields:
            try:
                v = float(f)
                if 0.01 < v < 10: time_ms = v
            except: pass
        if time_ms:
            return [{"case":CN[case_id],"strategy":"EBISU","gpu":gpu,
                     "problem_size":size_label,"steps":"100",
                     "median_us":f"{time_ms*1000/100:.2f}","min_us":"","max_us":"",
                     "overhead_pct":""}]
    except Exception as e:
        print(f"    EBISU {case_id}: {e}")
    return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=None)
    p.add_argument("--strategies", nargs="+", default=["cuda","taichi","warp","triton","tilelang","kokkos","perks","ebisu"])
    p.add_argument("--output", default="matrix_results.csv")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    gpu = gpu_name()
    print(f"GPU: {gpu}\nStrategies: {a.strategies}\nOutput: {a.output}")
    ids = a.cases or sorted(CN.keys(), key=lambda x: int(x[1:]))
    all_rows = []

    # C3/C10: overhead_solutions (aggregate binary)
    if "cuda" in a.strategies and any(c in ids for c in ("C3","C10")):
        print("\n=== C3/C10: overhead_solutions ===")
        binary = BD/"overhead_solutions_a100"
        if binary.exists() and not a.dry_run:
            r = subprocess.run([str(binary)], capture_output=True, text=True, timeout=300)
            parsed = parse_overhead_solutions(r.stdout)
            # Group by (case, size) to find Async as GPU-compute baseline
            async_baseline = {}
            for cid, sz, strat, us, st in parsed:
                is_na = isinstance(us, str) and str(us).startswith("N/A")
                if strat == "Async" and not is_na and us is not None:
                    async_baseline[(cid, sz)] = us
            for cid, sz, strat, us, st in parsed:
                if cid not in ids: continue
                is_na = isinstance(us, str) and str(us).startswith("N/A")
                oh = ""
                compute = async_baseline.get((cid, sz))
                if not is_na and us is not None and compute is not None and us > 0:
                    oh = f"{max(0,(us-compute)/us*100):.1f}"
                median_val = us if is_na else (f"{us:.2f}" if us else "N/A")
                all_rows.append({"case":CN[cid],"strategy":f"CUDA_{strat}","gpu":gpu,
                                "problem_size":sz,"steps":str(st),
                                "median_us":median_val,
                                "min_us":"","max_us":"","overhead_pct":oh})
            print(f"  {len([r for r in all_rows if r['case'] in ('Heat2D','GrayScott')])} entries")

    for cid in ids:
        if cid in ("C3","C10") and a.strategies == ["cuda"]:
            continue  # C3/C10 CUDA handled via overhead_solutions above
        print(f"\n=== {cid}: {CN[cid]} ===")
        if "cuda" in a.strategies and cid not in ("C3","C10"):
            rows = []
            if cid in CUDA:
                binary_name, size_configs = CUDA[cid]
                for args_str, label in size_configs:
                    rows.extend(run_binary(binary_name, args_str, label, cid, gpu, a.dry_run,
                                          configured_steps=CUDA_STEPS.get(cid, "")))
            all_rows.extend(rows)
            if rows: print(f"  CUDA: {len(rows)}")
        if "taichi" in a.strategies:
            rows = run_dsl(cid, "taichi", gpu, a.dry_run)
            all_rows.extend(rows)
            if rows: print(f"  Taichi: {len(rows)}")
        if "warp" in a.strategies:
            rows = run_warp(cid, gpu, a.dry_run)
            all_rows.extend(rows)
            if rows: print(f"  Warp: {len(rows)}")
        if "triton" in a.strategies:
            rows = run_triton(cid, gpu, a.dry_run)
            all_rows.extend(rows)
            if rows: print(f"  Triton: {len(rows)}")
        if "tilelang" in a.strategies:
            rows = run_tilelang(cid, gpu, a.dry_run)
            all_rows.extend(rows)
            if rows: print(f"  TileLang: {len(rows)}")
        if "kokkos" in a.strategies and cid in KOKKOS:
            binary_name = KOKKOS[cid][0]
            kokkos_rows = []
            for args_str, label in KOKKOS[cid][1]:
                path = BD/binary_name
                if not path.exists(): continue
                if a.dry_run: print(f"    Kokkos: {path} {args_str}"); continue
                try:
                    cmd = [str(path)] + args_str.split()
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    inv_rows = []  # per-invocation rows
                    # Parse Kokkos CSV output lines
                    for line in r.stdout.split("\n"):
                        if line.startswith("CSV:"):
                            parts = [p.strip() for p in line[4:].split(",")]
                            if len(parts) >= 7:
                                # 7-field: name,N,steps,min,median,avg,max
                                steps_k = parts[2]
                                median_ms = float(parts[4])
                                min_ms = float(parts[3])
                                max_ms = float(parts[6])
                                st_int = max(1, int(steps_k))
                                inv_rows.append({"case":CN[cid],"strategy":"Kokkos","gpu":gpu,
                                                "problem_size":label,"steps":steps_k,
                                                "median_us":f"{median_ms*1000/st_int:.2f}",
                                                "min_us":f"{min_ms*1000/st_int:.2f}",
                                                "max_us":f"{max_ms*1000/st_int:.2f}",
                                                "overhead_pct":""})
                            elif len(parts) >= 3:
                                # 3-field: name,steps,median_ms (e.g. kokkos_f2)
                                steps_k = parts[1]
                                median_ms = float(parts[2])
                                st_int = max(1, int(steps_k))
                                inv_rows.append({"case":CN[cid],"strategy":"Kokkos","gpu":gpu,
                                                "problem_size":label,"steps":steps_k,
                                                "median_us":f"{median_ms*1000/st_int:.2f}",
                                                "min_us":"","max_us":"",
                                                "overhead_pct":""})
                    # Fallback: parse "median=X.XXXms" or "N steps, median=X.XXXms"
                    if not inv_rows:
                        m = re.search(r'median=([\d.]+)\s*ms', r.stdout)
                        if m:
                            ms = float(m.group(1))
                            m2 = re.search(r'(\d+)\s*steps', r.stdout)
                            st = int(m2.group(1)) if m2 else 100
                            inv_rows.append({"case":CN[cid],"strategy":"Kokkos","gpu":gpu,
                                            "problem_size":label,"steps":str(st),
                                            "median_us":f"{ms*1000/st:.2f}","min_us":"","max_us":"",
                                            "overhead_pct":""})
                    kokkos_rows.extend(inv_rows)
                except Exception as e:
                    print(f"    Kokkos {cid}: {e}")
            all_rows.extend(kokkos_rows)
            if kokkos_rows: print(f"  Kokkos: {len(kokkos_rows)}")
        if "perks" in a.strategies:
            rows = run_perks(cid, gpu, a.dry_run)
            all_rows.extend(rows)
            if rows: print(f"  PERKS: {len(rows)}")
        if "ebisu" in a.strategies:
            rows = run_ebisu(cid, gpu, a.dry_run)
            all_rows.extend(rows)
            if rows: print(f"  EBISU: {len(rows)}")

    if not a.dry_run and all_rows:
        # Normalize problem_size labels: strip N=, x dimensions, p suffix
        def norm_size(s):
            s = s.strip()
            if s.startswith("N="): s = s[2:]
            if s.endswith("p"): s = s[:-1]
            # For multi-dim like "4096x4096" or "256x256x256", keep first number
            if "x" in s: s = s.split("x")[0]
            return s
        for row in all_rows:
            row["problem_size"] = norm_size(row["problem_size"])

        # Fill overhead_pct using CUDA compute baseline
        # Step 1: from CUDA rows with real overhead_pct (compute = median * (1 - oh/100))
        cuda_compute = {}  # (case, size) -> gpu_compute_us
        for row in all_rows:
            if row["strategy"].startswith("CUDA_") and row["overhead_pct"] not in ("", "N/A"):
                try:
                    us = float(row["median_us"])
                    oh = float(row["overhead_pct"])
                    comp = us * (1 - oh / 100)
                    key = (row["case"], row["problem_size"])
                    if key not in cuda_compute or comp < cuda_compute[key]:
                        cuda_compute[key] = comp
                except (ValueError, TypeError):
                    pass
        # Step 2: for cases without measured baseline, use fastest CUDA timing as proxy
        cuda_fastest = {}
        for row in all_rows:
            if row["strategy"].startswith("CUDA_") and row["median_us"] and not row["median_us"].startswith("N/A"):
                try:
                    us = float(row["median_us"])
                    key = (row["case"], row["problem_size"])
                    if key not in cuda_fastest or us < cuda_fastest[key]:
                        cuda_fastest[key] = us
                except (ValueError, TypeError):
                    pass
        for key in cuda_fastest:
            if key not in cuda_compute:
                cuda_compute[key] = cuda_fastest[key]  # proxy: fastest CUDA ≈ compute
        # Also fill N/A CUDA rows with proxy overhead
        for row in all_rows:
            if row["strategy"].startswith("CUDA_") and row["overhead_pct"] == "N/A":
                key = (row["case"], row["problem_size"])
                if key in cuda_compute and row["median_us"] and not row["median_us"].startswith("N/A"):
                    try:
                        us = float(row["median_us"])
                        comp = cuda_compute[key]
                        if us > 0:
                            row["overhead_pct"] = f"{max(0,(us-comp)/us*100):.1f}"
                    except (ValueError, TypeError):
                        pass
        for row in all_rows:
            if not row["strategy"].startswith("CUDA_") and row["overhead_pct"] == "":
                key = (row["case"], row["problem_size"])
                if key in cuda_compute and row["median_us"] and not row["median_us"].startswith("N/A"):
                    try:
                        us = float(row["median_us"])
                        comp = cuda_compute[key]
                        if us > 0:
                            row["overhead_pct"] = f"{max(0,(us-comp)/us*100):.1f}"
                    except (ValueError, TypeError):
                        pass

        fields = ["case","strategy","gpu","problem_size","steps","median_us","min_us","max_us","overhead_pct"]
        with open(a.output, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
            csv.DictWriter(f, fieldnames=fields).writerows(all_rows)
        print(f"\nResults: {a.output} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
