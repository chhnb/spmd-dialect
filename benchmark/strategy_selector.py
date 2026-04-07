#!/usr/bin/env python3
"""
Automatic Strategy Selector Demo

Given a simulation kernel's characteristics, recommends the optimal
execution strategy (Sync / Async / Graph / Persistent) and predicts
performance.

Usage:
  python strategy_selector.py --kernel heat2d --gpu 3060 --cells 16384
  python strategy_selector.py --kernel osher --gpu b200 --cells 4096 --host-readback
  python strategy_selector.py --kernel cg --gpu 3060 --cells 4096 --host-readback
  python strategy_selector.py --all
"""
from __future__ import annotations
import argparse, math, json, sys
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

# =========================================================================
# Hardware database
# =========================================================================
@dataclass(frozen=True)
class GPU:
    name: str
    sms: int
    coop_blocks_per_sm: int
    oh_sync: float      # cudaDeviceSynchronize overhead (μs)
    oh_launch: float     # cudaLaunchKernel overhead (μs)
    oh_graph_fixed: float  # Graph replay overhead per step (μs)
    oh_grid_sync: float  # cooperative grid_sync cost (μs)

GPUS = {
    "3060": GPU("RTX 3060 Laptop", sms=30, coop_blocks_per_sm=4,
                oh_sync=45.0, oh_launch=25.0, oh_graph_fixed=2.0, oh_grid_sync=2.5),
    "b200": GPU("NVIDIA B200", sms=148, coop_blocks_per_sm=3,
                oh_sync=5.5, oh_launch=2.5, oh_graph_fixed=0.5, oh_grid_sync=2.0),
}

# =========================================================================
# Kernel database (from NCU profiling + measurements)
# =========================================================================
@dataclass(frozen=True)
class KernelInfo:
    name: str
    regs_per_thread: int
    kernels_per_step: int
    needs_host_readback: bool  # e.g., CG needs alpha/beta from host
    t_fixed: float  # μs (fixed overhead per invocation, from fitting)
    t_per_wave: float  # μs per wave (from fitting)
    description: str

KERNELS = {
    "heat2d": {
        "3060": KernelInfo("Heat2D", 18, 1, False, 0.19, 0.61, "5-point stencil, fp32"),
        "b200": KernelInfo("Heat2D", 18, 1, False, 2.25, 0.50, "5-point stencil, fp32"),
    },
    "osher": {
        "3060": KernelInfo("OSHER SWE", 106, 1, False, 32.3, 38.4, "Riemann solver, fp64, 106 regs"),
        "b200": KernelInfo("OSHER SWE", 106, 1, False, 6.15, 6.15, "Riemann solver, fp64, 106 regs"),
    },
    "jacobi2d": {
        "3060": KernelInfo("Jacobi2D", 16, 1, False, 2.22, 0.42, "5-point stencil, fp32"),
        "b200": KernelInfo("Jacobi2D", 16, 1, False, 2.0, 0.40, "5-point stencil, fp32"),
    },
    "cg": {
        "3060": KernelInfo("CG Solver", 24, 5, True, 5.0, 1.0, "5 kernels/step, needs host readback"),
        "b200": KernelInfo("CG Solver", 24, 5, True, 3.0, 0.5, "5 kernels/step, needs host readback"),
    },
    "lulesh": {
        "3060": KernelInfo("LULESH-like", 20, 4, False, 3.0, 0.8, "4 kernels/step, Lagrangian hydro"),
        "b200": KernelInfo("LULESH-like", 20, 4, False, 2.5, 0.5, "4 kernels/step, Lagrangian hydro"),
    },
    "hotspot": {
        "3060": KernelInfo("HotSpot", 20, 1, False, 3.69, 0.47, "Rodinia thermal, fp32"),
        "b200": KernelInfo("HotSpot", 20, 1, False, 2.5, 0.40, "Rodinia thermal, fp32"),
    },
    "srad": {
        "3060": KernelInfo("SRAD", 26, 1, False, 4.44, 0.59, "Rodinia anisotropic diffusion, fp32"),
        "b200": KernelInfo("SRAD", 26, 1, False, 3.0, 0.45, "Rodinia anisotropic diffusion, fp32"),
    },
}

# =========================================================================
# Strategy predictor
# =========================================================================
def predict(gpu_name: str, kernel_name: str, n_cells: int,
            block_size: int = 256, has_host_readback: Optional[bool] = None,
            has_periodic_save: bool = False) -> Dict:
    gpu = GPUS[gpu_name]
    ki = KERNELS.get(kernel_name, {}).get(gpu_name)
    if ki is None:
        return {"error": f"Unknown kernel '{kernel_name}' for GPU '{gpu_name}'"}

    if has_host_readback is None:
        has_host_readback = ki.needs_host_readback

    n_blocks = math.ceil(n_cells / block_size)
    waves = max(1, math.ceil(n_blocks / gpu.sms))
    K = ki.kernels_per_step

    # Compute time
    t_compute = ki.t_fixed + ki.t_per_wave * waves

    # Cooperative launch limit
    coop_limit = gpu.coop_blocks_per_sm * gpu.sms
    persistent_ok = n_blocks <= coop_limit and not has_host_readback

    # Predict each strategy
    strategies = {}

    # Sync: K kernels × (launch + sync) per step
    t_sync = t_compute + K * (gpu.oh_launch + gpu.oh_sync)
    strategies["sync"] = {"time": t_sync, "feasible": True,
                          "note": f"{K} launches + {K} syncs per step"}

    # Async: K launches, sync only when needed
    extra_sync = 2 * gpu.oh_sync if has_host_readback else 0  # CG needs 2 readbacks
    t_async = t_compute + K * gpu.oh_launch + extra_sync
    strategies["async"] = {"time": t_async, "feasible": True,
                          "note": f"{K} launches" + (f" + {2} host readbacks" if has_host_readback else "")}

    # Graph: only works without host readback
    if not has_host_readback:
        t_graph = t_compute + gpu.oh_graph_fixed
        strategies["graph"] = {"time": t_graph, "feasible": True,
                              "note": "Capture + replay, zero launch overhead"}
    else:
        strategies["graph"] = {"time": float("inf"), "feasible": False,
                              "note": "INFEASIBLE: needs host readback (alpha/beta in CG)"}

    # Persistent: only works if grid fits in cooperative limit
    if persistent_ok:
        t_persist = t_compute + K * gpu.oh_grid_sync
        strategies["persistent"] = {"time": t_persist, "feasible": True,
                                   "note": f"Fused, {K} grid_syncs, {n_blocks}/{coop_limit} blocks used"}
    elif has_host_readback:
        # Persistent CAN handle host readback by computing scalars on device
        t_persist = t_compute + K * gpu.oh_grid_sync + 2.0  # extra for device-side reduction
        if n_blocks <= coop_limit:
            strategies["persistent"] = {"time": t_persist, "feasible": True,
                                       "note": f"Fused + device-side scalar compute (no host readback)"}
        else:
            strategies["persistent"] = {"time": float("inf"), "feasible": False,
                                       "note": f"INFEASIBLE: grid {n_blocks} > cooperative limit {coop_limit}"}
    else:
        strategies["persistent"] = {"time": float("inf"), "feasible": False,
                                   "note": f"INFEASIBLE: grid {n_blocks} > cooperative limit {coop_limit}"}

    # Select best
    feasible = {k: v for k, v in strategies.items() if v["feasible"]}
    best = min(feasible, key=lambda k: feasible[k]["time"])

    # Classify regime
    oh_fraction = (t_sync - t_compute) / t_sync if t_sync > 0 else 0
    regime = "OH-dominated" if oh_fraction > 0.5 else "Transitional" if oh_fraction > 0.2 else "Compute-dominated"

    # Optimization recommendation
    if regime == "Compute-dominated":
        opt_note = "Consider register tuning (-maxrregcount) for occupancy improvement"
    elif best == "persistent" and has_periodic_save:
        opt_note = "Use DMA Copy Engine overlap for zero-overhead checkpointing"
    else:
        opt_note = f"Use {best} strategy for {strategies[best]['time']:.1f} μs/step ({t_sync/strategies[best]['time']:.1f}x vs sync)"

    return {
        "gpu": gpu_name,
        "kernel": kernel_name,
        "cells": n_cells,
        "blocks": n_blocks,
        "waves": waves,
        "kernels_per_step": K,
        "regs_per_thread": ki.regs_per_thread,
        "t_compute": t_compute,
        "oh_fraction": oh_fraction,
        "regime": regime,
        "strategies": strategies,
        "best_strategy": best,
        "best_time": strategies[best]["time"],
        "sync_time": t_sync,
        "speedup": t_sync / strategies[best]["time"],
        "recommendation": opt_note,
    }


def print_result(r: Dict):
    if "error" in r:
        print(f"  ERROR: {r['error']}")
        return

    print(f"  GPU: {r['gpu']}  Kernel: {r['kernel']}  Cells: {r['cells']}")
    print(f"  Blocks: {r['blocks']}  Waves: {r['waves']}  Regs/thread: {r['regs_per_thread']}")
    print(f"  Kernels/step: {r['kernels_per_step']}  Compute: {r['t_compute']:.1f} μs")
    print(f"  Regime: {r['regime']} (OH fraction: {r['oh_fraction']*100:.0f}%)")
    print(f"  ---")
    for name, s in r["strategies"].items():
        marker = " ←★ BEST" if name == r["best_strategy"] else ""
        if s["feasible"]:
            print(f"    {name:<12}: {s['time']:>7.1f} μs  ({s['note']}){marker}")
        else:
            print(f"    {name:<12}: {'N/A':>7}     ({s['note']})")
    print(f"  ---")
    print(f"  ★ Recommendation: {r['recommendation']}")
    print(f"  ★ Speedup: {r['speedup']:.1f}x over sync baseline")
    print()


def main():
    ap = argparse.ArgumentParser(description="Automatic Strategy Selector")
    ap.add_argument("--kernel", type=str, help="Kernel name (heat2d, osher, cg, lulesh, ...)")
    ap.add_argument("--gpu", type=str, default="3060", help="GPU (3060 or b200)")
    ap.add_argument("--cells", type=int, default=16384, help="Number of cells/unknowns")
    ap.add_argument("--host-readback", action="store_true", help="Kernel needs host scalar readback")
    ap.add_argument("--periodic-save", action="store_true", help="Need periodic D2H checkpoint")
    ap.add_argument("--all", action="store_true", help="Run all demo cases")
    ap.add_argument("--json", action="store_true", help="Output JSON")
    args = ap.parse_args()

    if args.all:
        print("=" * 70)
        print("  Strategy Selector Demo — All Cases")
        print("=" * 70)

        cases = [
            ("3060", "heat2d", 128*128, False, False, "Light stencil, small grid"),
            ("3060", "heat2d", 1024*1024, False, False, "Light stencil, large grid"),
            ("3060", "osher", 4096, False, False, "Heavy solver (106 regs), small grid"),
            ("3060", "osher", 16384, False, False, "Heavy solver, medium grid"),
            ("3060", "cg", 4096, True, False, "CG (5 kern/step, host readback!)"),
            ("3060", "cg", 16384, True, False, "CG large"),
            ("3060", "lulesh", 4096, False, False, "LULESH (4 kern/step)"),
            ("b200", "heat2d", 128*128, False, False, "B200: light stencil"),
            ("b200", "osher", 4096, False, False, "B200: heavy solver"),
            ("b200", "cg", 4096, True, False, "B200: CG with host readback"),
            ("b200", "lulesh", 4096, False, False, "B200: LULESH"),
            ("3060", "heat2d", 128*128, False, True, "Light stencil + periodic save → DMA overlap"),
        ]

        all_results = []
        for gpu, kernel, cells, hr, ps, desc in cases:
            print(f"\n{'─'*60}")
            print(f"  Case: {desc}")
            print(f"{'─'*60}")
            r = predict(gpu, kernel, cells, has_host_readback=hr, has_periodic_save=ps)
            print_result(r)
            all_results.append(r)

        # Summary table
        print(f"\n{'='*70}")
        print(f"  Summary")
        print(f"{'='*70}")
        print(f"{'Case':<45} {'Best':>10} {'Speedup':>8} {'Regime':<15}")
        print("-" * 80)
        for (gpu,k,c,hr,ps,desc), r in zip(cases, all_results):
            if "error" not in r:
                print(f"{desc:<45} {r['best_strategy']:>10} {r['speedup']:>7.1f}x {r['regime']:<15}")

    elif args.kernel:
        r = predict(args.gpu, args.kernel, args.cells,
                    has_host_readback=args.host_readback, has_periodic_save=args.periodic_save)
        if args.json:
            print(json.dumps(r, indent=2, default=str))
        else:
            print_result(r)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
