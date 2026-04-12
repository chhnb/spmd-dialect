#!/usr/bin/env python3
"""Milestone 5: Generate analysis from matrix_results.csv.

Writes 4 output files to --output-dir:
1. overhead_vs_kernels.csv — Overhead% by strategy × case (smallest size)
2. speedup_heatmap.csv — Speedup vs CUDA Sync for each strategy×case
3. dsl_decomposition.csv — GPU compute / CUDA Sync / DSL time breakdown
4. size_scaling.csv — Overhead% across sizes per case

Also prints human-readable tables to stdout.
"""
import argparse, csv, os, sys
from collections import defaultdict

KERN = {"Jacobi2D":2,"Jacobi3D":2,"Heat2D":2,"Wave2D":1,"LBM_D2Q9":1,
        "Nbody":2,"SPH":2,"HydroF1":1,"HydroF2":2,"GrayScott":2,
        "FDTD2D":3,"MacCormack3D":3,"LULESH":4,"PIC1D":4,"CG_Solver":5,
        "StableFluids":102,"Conv3D":510,"DOITGEN":512,"LU":4096,"ADI":4097,
        "GramSchmidt":6144}


def load(path):
    with open(path) as f: return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Written: {path} ({len(rows)} rows)")


def plot1(data, od):
    """Overhead% vs kernels/step, smallest size per case."""
    by = defaultdict(lambda: defaultdict(dict))
    for r in data:
        if r.get("overhead_pct") and r["overhead_pct"]:
            by[r["case"]][r["problem_size"]][r["strategy"]] = float(r["overhead_pct"])

    rows = []
    for case in sorted(by.keys(), key=lambda c: KERN.get(c,0)):
        smallest = min(by[case].keys())
        for strat, oh in by[case][smallest].items():
            rows.append({"case":case,"kernels_per_step":KERN.get(case,0),
                        "problem_size":smallest,"strategy":strat,"overhead_pct":f"{oh:.1f}"})
    write_csv(os.path.join(od,"overhead_vs_kernels.csv"), rows,
              ["case","kernels_per_step","problem_size","strategy","overhead_pct"])

    print("\n=== Overhead% vs Kernels/Step ===")
    strats = sorted(set(r["strategy"] for r in rows))
    print(f"{'Case':15s} {'K/S':>5s}" + "".join(f"{s:>15s}" for s in strats))
    for case in sorted(set(r["case"] for r in rows), key=lambda c: KERN.get(c,0)):
        line = f"{case:15s} {KERN.get(case,0):5d}"
        case_rows = {r["strategy"]: r["overhead_pct"] for r in rows if r["case"]==case}
        for s in strats:
            line += f"{case_rows.get(s,'—'):>14s}%" if s in case_rows else f"{'—':>15s}"
        print(line)


def plot2(data, od):
    """Speedup heatmap vs CUDA Sync."""
    times = defaultdict(lambda: defaultdict(dict))
    for r in data:
        if r["median_us"] and r["median_us"] != "N/A":
            times[r["case"]][r["problem_size"]][r["strategy"]] = float(r["median_us"])

    rows = []
    for case in sorted(times.keys(), key=lambda c: KERN.get(c,0)):
        smallest = min(times[case].keys())
        st = times[case][smallest]
        sync = next((st[k] for k in st if "Sync" in k and "CUDA" in k), None)
        if not sync: continue
        for strat, us in st.items():
            rows.append({"case":case,"problem_size":smallest,"strategy":strat,
                        "median_us":f"{us:.2f}","speedup_vs_sync":f"{sync/us:.2f}"})
    write_csv(os.path.join(od,"speedup_heatmap.csv"), rows,
              ["case","problem_size","strategy","median_us","speedup_vs_sync"])

    print("\n=== Speedup vs CUDA Sync ===")
    strats = sorted(set(r["strategy"] for r in rows))
    print(f"{'Case':15s}" + "".join(f"{s:>15s}" for s in strats))
    for case in sorted(set(r["case"] for r in rows), key=lambda c: KERN.get(c,0)):
        line = f"{case:15s}"
        cr = {r["strategy"]: r["speedup_vs_sync"] for r in rows if r["case"]==case}
        for s in strats:
            line += f"{cr.get(s,'—'):>14s}x" if s in cr else f"{'—':>15s}"
        print(line)


def plot3(data, od):
    """DSL overhead decomposition."""
    by = defaultdict(lambda: defaultdict(dict))
    for r in data:
        if r["median_us"] and r["median_us"] != "N/A":
            by[r["case"]][r["problem_size"]][r["strategy"]] = float(r["median_us"])

    rows = []
    print("\n=== DSL Overhead Decomposition ===")
    print(f"{'Case':15s} {'Size':>8s} {'GPU_min':>10s} {'CUDA_Sync':>10s} {'Taichi':>10s} {'Warp':>10s} {'Triton':>10s}")
    for case in sorted(by.keys(), key=lambda c: KERN.get(c,0)):
        for size in sorted(by[case].keys()):
            s = by[case][size]
            cuda_vals = [v for k,v in s.items() if "CUDA" in k and v > 0]
            if not cuda_vals: continue
            gpu_min = min(cuda_vals)
            sync = next((s[k] for k in s if "Sync" in k and "CUDA" in k), None)
            taichi = s.get("Taichi")
            warp = s.get("Warp")
            triton = s.get("Triton")
            row = {"case":case,"problem_size":size,"gpu_compute_us":f"{gpu_min:.1f}",
                   "cuda_sync_us":f"{sync:.1f}" if sync else "",
                   "taichi_us":f"{taichi:.1f}" if taichi else "",
                   "warp_us":f"{warp:.1f}" if warp else "",
                   "triton_us":f"{triton:.1f}" if triton else ""}
            rows.append(row)
            print(f"{case:15s} {size:>8s} {gpu_min:10.1f} "
                  f"{sync if sync else 0:10.1f} "
                  f"{taichi if taichi else 0:10.1f} "
                  f"{warp if warp else 0:10.1f} "
                  f"{triton if triton else 0:10.1f}")
    write_csv(os.path.join(od,"dsl_decomposition.csv"), rows,
              ["case","problem_size","gpu_compute_us","cuda_sync_us","taichi_us","warp_us","triton_us"])


def plot4(data, od):
    """Size scaling: overhead% across sizes."""
    by = defaultdict(lambda: defaultdict(dict))
    for r in data:
        if r.get("overhead_pct") and r["overhead_pct"]:
            by[r["case"]][r["problem_size"]][r["strategy"]] = float(r["overhead_pct"])

    rows = []
    print("\n=== Size Scaling ===")
    for case in sorted(by.keys(), key=lambda c: KERN.get(c,0)):
        if len(by[case]) < 2: continue
        print(f"\n  {case}:")
        strats = sorted(set(s for sz in by[case].values() for s in sz))
        print(f"    {'Size':>10s}" + "".join(f"{s:>15s}" for s in strats))
        for size in sorted(by[case].keys()):
            line = f"    {size:>10s}"
            for s in strats:
                v = by[case][size].get(s)
                line += f"{v:14.1f}%" if v else f"{'—':>15s}"
            print(line)
            for s in strats:
                v = by[case][size].get(s)
                if v:
                    rows.append({"case":case,"problem_size":size,"strategy":s,"overhead_pct":f"{v:.1f}"})
    write_csv(os.path.join(od,"size_scaling.csv"), rows,
              ["case","problem_size","strategy","overhead_pct"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv_file")
    p.add_argument("--output-dir", default="plots")
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    data = load(a.csv_file)
    print(f"Loaded {len(data)} rows from {a.csv_file}")
    plot1(data, a.output_dir)
    plot2(data, a.output_dir)
    plot3(data, a.output_dir)
    plot4(data, a.output_dir)
    print(f"\nAll outputs in {a.output_dir}/")


if __name__ == "__main__":
    main()
