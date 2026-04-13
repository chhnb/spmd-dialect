#!/usr/bin/env python3
"""Generate analysis from matrix_results.csv.

Writes 4 output files to --output-dir:
1. overhead_vs_kernels.csv — Overhead% by strategy × case (smallest size)
2. speedup_heatmap.csv — Speedup vs CUDA Sync for each strategy×case
3. dsl_decomposition.csv — GPU compute / CUDA Sync / DSL time breakdown
4. size_scaling.csv — Overhead% across sizes per case

Also prints human-readable tables to stdout.
"""
import argparse, csv, os, re, sys
from collections import defaultdict

KERN = {"Jacobi2D":2,"Jacobi3D":2,"Heat2D":2,"Wave2D":1,"LBM_D2Q9":1,
        "Nbody":2,"SPH":2,"HydroF1":1,"HydroF2":2,"GrayScott":2,
        "FDTD2D":3,"MacCormack3D":3,"LULESH":4,"PIC1D":4,"CG_Solver":5,
        "StableFluids":102,"Conv3D":510,"DOITGEN":512,"LU":4096,"ADI":4097,
        "GramSchmidt":6144}


def load(path):
    with open(path) as f: return list(csv.DictReader(f))


def size_sort_key(s):
    """Sort size strings numerically: '256' < '4096', '64x64' < '256x256', 'default' < '20w'."""
    if s in ("default",): return 0
    if s in ("20w",): return 207234
    # Extract first number
    m = re.match(r'(\d+)', s)
    return int(m.group(1)) if m else 0


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Written: {path} ({len(rows)} rows)")


def try_matplotlib():
    """Try to import matplotlib. Returns (plt, True) or (None, False)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt, True
    except ImportError:
        return None, False


def plot1(data, od):
    """Overhead% vs kernels/step, smallest size per case."""
    by = defaultdict(lambda: defaultdict(dict))
    for r in data:
        oh = r.get("overhead_pct", "")
        if oh and oh != "N/A":
            try:
                by[r["case"]][r["problem_size"]][r["strategy"]] = float(oh)
            except ValueError:
                pass

    rows = []
    for case in sorted(by.keys(), key=lambda c: KERN.get(c,0)):
        smallest = min(by[case].keys(), key=size_sort_key)
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

    # Generate PNG if matplotlib available
    plt, has_mpl = try_matplotlib()
    if has_mpl and rows:
        fig, ax = plt.subplots(figsize=(14, 6))
        cases_sorted = sorted(set(r["case"] for r in rows), key=lambda c: KERN.get(c,0))
        strats_list = sorted(set(r["strategy"] for r in rows))
        x = range(len(cases_sorted))
        width = 0.8 / max(len(strats_list), 1)
        for i, strat in enumerate(strats_list):
            vals = []
            for c in cases_sorted:
                v = next((float(r["overhead_pct"]) for r in rows if r["case"]==c and r["strategy"]==strat), 0)
                vals.append(v)
            ax.bar([xi + i*width for xi in x], vals, width, label=strat)
        ax.set_xlabel("Case (sorted by kernels/step)")
        ax.set_ylabel("Overhead %")
        ax.set_title("Launch Overhead vs Kernel Complexity")
        ax.set_xticks([xi + width*len(strats_list)/2 for xi in x])
        ax.set_xticklabels([f"{c}\n({KERN.get(c,0)})" for c in cases_sorted], rotation=45, ha='right', fontsize=6)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        png_path = os.path.join(od, "overhead_vs_kernels.png")
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"  Plot: {png_path}")


def plot2(data, od):
    """Speedup heatmap vs CUDA Sync."""
    times = defaultdict(lambda: defaultdict(dict))
    for r in data:
        if r["median_us"] and not r["median_us"].startswith("N/A"):
            times[r["case"]][r["problem_size"]][r["strategy"]] = float(r["median_us"])

    rows = []
    for case in sorted(times.keys(), key=lambda c: KERN.get(c,0)):
        # Pick smallest size that has CUDA_Sync (prefer one with DSL data too)
        sizes_sorted = sorted(times[case].keys(), key=size_sort_key)
        best_size = None
        for sz in sizes_sorted:
            has_sync = any("Sync" in k and "CUDA" in k for k in times[case][sz])
            has_dsl = any(k in ("Taichi","Warp","Triton","Kokkos") for k in times[case][sz])
            if has_sync and has_dsl:
                best_size = sz; break
        if best_size is None:
            for sz in sizes_sorted:
                if any("Sync" in k and "CUDA" in k for k in times[case][sz]):
                    best_size = sz; break
        if best_size is None: continue
        st = times[case][best_size]
        sync = next((st[k] for k in st if "Sync" in k and "CUDA" in k), None)
        if not sync: continue
        for strat, us in st.items():
            rows.append({"case":case,"problem_size":best_size,"strategy":strat,
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

    # Generate actual heatmap PNG using imshow
    plt, has_mpl = try_matplotlib()
    if has_mpl and rows:
        import numpy as np
        cases_sorted = sorted(set(r["case"] for r in rows), key=lambda c: KERN.get(c,0))
        strats_list = sorted(set(r["strategy"] for r in rows))
        matrix = np.full((len(strats_list), len(cases_sorted)), np.nan)
        for r in rows:
            si = strats_list.index(r["strategy"]) if r["strategy"] in strats_list else -1
            ci = cases_sorted.index(r["case"]) if r["case"] in cases_sorted else -1
            if si >= 0 and ci >= 0:
                matrix[si, ci] = float(r["speedup_vs_sync"])
        fig, ax = plt.subplots(figsize=(16, 8))
        masked = np.ma.masked_invalid(matrix)
        im = ax.imshow(masked, aspect='auto', cmap='RdYlGn', vmin=0, vmax=max(3, np.nanmax(matrix)))
        ax.set_xticks(range(len(cases_sorted)))
        ax.set_xticklabels(cases_sorted, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(len(strats_list)))
        ax.set_yticklabels(strats_list, fontsize=7)
        ax.set_title("Speedup vs CUDA Sync (green=faster)")
        for si in range(len(strats_list)):
            for ci in range(len(cases_sorted)):
                v = matrix[si, ci]
                if v > 0:
                    ax.text(ci, si, f"{v:.1f}x", ha='center', va='center', fontsize=5,
                           color='white' if v > 2 else 'black')
        fig.colorbar(im, ax=ax, label="Speedup", shrink=0.6)
        plt.tight_layout()
        png_path = os.path.join(od, "speedup_heatmap.png")
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"  Plot: {png_path}")


def plot3(data, od):
    """DSL overhead decomposition."""
    by = defaultdict(lambda: defaultdict(dict))
    for r in data:
        if r["median_us"] and not r["median_us"].startswith("N/A"):
            by[r["case"]][r["problem_size"]][r["strategy"]] = float(r["median_us"])

    # Extract GPU compute baselines from CUDA rows ONLY (not DSL)
    gpu_baselines = defaultdict(dict)
    for r in data:
        if not r.get("strategy", "").startswith("CUDA_"):
            continue  # Only use CUDA rows for GPU compute baseline
        oh_str = r.get("overhead_pct", "")
        if oh_str and oh_str != "N/A" and r.get("median_us") and not r["median_us"].startswith("N/A"):
            try:
                us = float(r["median_us"])
                oh = float(oh_str)
            except ValueError:
                continue
            compute = us * (1 - oh / 100)
            key = (r["case"], r["problem_size"])
            if key not in gpu_baselines or compute < gpu_baselines[key]:
                gpu_baselines[key] = compute

    rows = []
    print("\n=== DSL Overhead Decomposition ===")
    print(f"{'Case':15s} {'Size':>8s} {'GPU_base':>10s} {'CUDA_Sync':>10s} {'Taichi':>10s} {'Warp':>10s} {'Triton':>10s}")
    for case in sorted(by.keys(), key=lambda c: KERN.get(c,0)):
        for size in sorted(by[case].keys(), key=size_sort_key):
            s = by[case][size]
            # Use recorded GPU baseline if available, else fastest CUDA
            gpu_base = gpu_baselines.get((case, size))
            if gpu_base is None:
                cuda_vals = [v for k,v in s.items() if "CUDA" in k and v > 0]
                gpu_base = min(cuda_vals) if cuda_vals else None
            if gpu_base is None: continue
            sync = next((s[k] for k in s if "Sync" in k and "CUDA" in k), None)
            taichi = s.get("Taichi")
            warp = s.get("Warp")
            triton = s.get("Triton")
            row = {"case":case,"problem_size":size,"gpu_compute_us":f"{gpu_base:.1f}",
                   "cuda_sync_us":f"{sync:.1f}" if sync else "",
                   "taichi_us":f"{taichi:.1f}" if taichi else "",
                   "warp_us":f"{warp:.1f}" if warp else "",
                   "triton_us":f"{triton:.1f}" if triton else ""}
            rows.append(row)
            print(f"{case:15s} {size:>8s} {gpu_base:10.1f} "
                  f"{sync if sync else 0:10.1f} "
                  f"{taichi if taichi else 0:10.1f} "
                  f"{warp if warp else 0:10.1f} "
                  f"{triton if triton else 0:10.1f}")
    write_csv(os.path.join(od,"dsl_decomposition.csv"), rows,
              ["case","problem_size","gpu_compute_us","cuda_sync_us","taichi_us","warp_us","triton_us"])

    plt, has_mpl = try_matplotlib()
    if has_mpl and rows:
        import numpy as np
        cases = sorted(set(r["case"] for r in rows), key=lambda c: KERN.get(c,0))
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(cases))
        cols = [("gpu_compute_us","GPU compute"),("cuda_sync_us","CUDA Sync"),("taichi_us","Taichi"),("warp_us","Warp")]
        width = 0.8 / len(cols)
        for i, (col, label) in enumerate(cols):
            vals = [float(next((r[col] for r in rows if r["case"]==c and r[col]), 0)) for c in cases]
            ax.bar(x + i*width, vals, width, label=label, alpha=0.8)
        ax.set_ylabel("Time (us/step)")
        ax.set_title("DSL Overhead Decomposition")
        ax.set_xticks(x + width*len(cols)/2)
        ax.set_xticklabels(cases, rotation=45, ha='right', fontsize=7)
        ax.legend(fontsize=7)
        plt.tight_layout()
        png = os.path.join(od, "dsl_decomposition.png")
        plt.savefig(png, dpi=150); plt.close()
        print(f"  Plot: {png}")


def plot4(data, od):
    """Size scaling: overhead% across sizes."""
    by = defaultdict(lambda: defaultdict(dict))
    for r in data:
        oh = r.get("overhead_pct", "")
        if oh and oh != "N/A":
            try:
                by[r["case"]][r["problem_size"]][r["strategy"]] = float(oh)
            except ValueError:
                pass

    rows = []
    print("\n=== Size Scaling ===")
    for case in sorted(by.keys(), key=lambda c: KERN.get(c,0)):
        if len(by[case]) < 2: continue
        print(f"\n  {case}:")
        strats = sorted(set(s for sz in by[case].values() for s in sz))
        print(f"    {'Size':>10s}" + "".join(f"{s:>15s}" for s in strats))
        for size in sorted(by[case].keys(), key=size_sort_key):
            line = f"    {size:>10s}"
            for s in strats:
                v = by[case][size].get(s)
                line += f"{v:14.1f}%" if v is not None else f"{'—':>15s}"
            print(line)
            for s in strats:
                v = by[case][size].get(s)
                if v is not None:
                    rows.append({"case":case,"problem_size":size,"strategy":s,"overhead_pct":f"{v:.1f}"})
    write_csv(os.path.join(od,"size_scaling.csv"), rows,
              ["case","problem_size","strategy","overhead_pct"])

    plt, has_mpl = try_matplotlib()
    if has_mpl and rows:
        import numpy as np
        cases = sorted(set(r["case"] for r in rows), key=lambda c: KERN.get(c,0))[:12]  # top 12
        fig, ax = plt.subplots(figsize=(12, 5))
        for case in cases:
            cr = [(size_sort_key(r["problem_size"]), float(r["overhead_pct"]))
                  for r in rows if r["case"]==case and r["strategy"]=="CUDA_Sync"]
            if cr:
                cr.sort()
                ax.plot([c[0] for c in cr], [c[1] for c in cr], 'o-', label=case, markersize=4)
        ax.set_xlabel("Problem Size")
        ax.set_ylabel("Overhead % (CUDA Sync)")
        ax.set_title("Overhead vs Problem Size")
        ax.set_xscale('log')
        ax.legend(fontsize=6, ncol=3)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        png = os.path.join(od, "size_scaling.png")
        plt.savefig(png, dpi=150); plt.close()
        print(f"  Plot: {png}")


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
