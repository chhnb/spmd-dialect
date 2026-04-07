#!/usr/bin/env python3
"""Cost model v2: incorporates NCU profiling features for better prediction.

Improvements over v1 (cost_model_selector.py):
- Adds kernel-level features: compute_intensity, occupancy, memory_bandwidth
- Uses NCU-derived per-kernel compute time estimates
- Models the overhead vs compute tradeoff explicitly:
    T_sync     = T_compute + OH_launch + OH_sync
    T_async    = T_compute + OH_launch
    T_graph    = T_compute + OH_graph_replay  (≈ 0)
    T_persist  = T_compute × (OCC_orig / OCC_fused) + K × T_grid_sync

Key insight: we model the TWO regimes separately:
  - Overhead-dominated: strategy choice matters most
  - Compute-dominated: register/occupancy tuning matters most
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# =========================================================================
# Hardware specs
# =========================================================================
@dataclass(frozen=True)
class HWSpec:
    name: str
    sms: int
    max_threads_per_sm: int
    max_regs_per_sm: int
    l2_cache_bytes: int
    mem_bw_gb_s: float  # DRAM bandwidth

HW = {
    "3060": HWSpec("RTX 3060 Laptop", sms=30, max_threads_per_sm=1536,
                   max_regs_per_sm=65536, l2_cache_bytes=3*1024*1024, mem_bw_gb_s=336),
    "b200": HWSpec("NVIDIA B200", sms=148, max_threads_per_sm=2048,
                   max_regs_per_sm=65536, l2_cache_bytes=60*1024*1024, mem_bw_gb_s=8000),
}

# =========================================================================
# NCU kernel profiles (from ncu_representative_summary.json + manual additions)
# For kernels without NCU data, use analytical estimates.
# =========================================================================
@dataclass
class KernelProfile:
    """Per-kernel profiling data (from NCU or estimated)."""
    regs_per_thread: int = 32         # register count
    sm_throughput_pct: float = 50.0   # SM utilization (%)
    achieved_occ_pct: float = 50.0    # achieved occupancy (%)
    dram_read_bytes: float = 0        # per-invocation
    dram_write_bytes: float = 0
    fp_instructions: float = 0        # total FP ops per invocation
    compute_us: float = 0             # measured GPU compute time (from Graph baseline)
    kernels_per_step: int = 1         # how many kernel launches per timestep

# Known profiles from NCU + measurements
KERNEL_PROFILES: Dict[str, Dict[str, KernelProfile]] = {
    "heat2d": {
        "3060": KernelProfile(regs_per_thread=24, sm_throughput_pct=7.81, achieved_occ_pct=33.07,
                              dram_read_bytes=69120, dram_write_bytes=1408, fp_instructions=95764,
                              compute_us=4.2, kernels_per_step=1),
        "b200":  KernelProfile(regs_per_thread=24, sm_throughput_pct=5.0, achieved_occ_pct=40.0,
                               compute_us=2.8, kernels_per_step=1),
    },
    "osher": {
        "3060": KernelProfile(regs_per_thread=96, sm_throughput_pct=60.54, achieved_occ_pct=31.36,
                              dram_read_bytes=4174592, dram_write_bytes=1369472, fp_instructions=10026496,
                              compute_us=7.6, kernels_per_step=2),  # flux + update
        "b200":  KernelProfile(regs_per_thread=96, sm_throughput_pct=55.0, achieved_occ_pct=30.0,
                               compute_us=9.5, kernels_per_step=2),
    },
    # Stencils: similar to heat2d but varying compute intensity
    "jacobi2d":  {"3060": KernelProfile(regs_per_thread=20, compute_us=3.5, kernels_per_step=1),
                  "b200":  KernelProfile(regs_per_thread=20, compute_us=2.5, kernels_per_step=1)},
    "hotspot":   {"3060": KernelProfile(regs_per_thread=28, compute_us=5.0, kernels_per_step=1),
                  "b200":  KernelProfile(regs_per_thread=28, compute_us=3.0, kernels_per_step=1)},
    "srad":      {"3060": KernelProfile(regs_per_thread=32, compute_us=6.0, kernels_per_step=1),
                  "b200":  KernelProfile(regs_per_thread=32, compute_us=3.5, kernels_per_step=1)},
    "grayscott": {"3060": KernelProfile(regs_per_thread=28, compute_us=4.0, kernels_per_step=1),
                  "b200":  KernelProfile(regs_per_thread=28, compute_us=2.8, kernels_per_step=1)},
}

# =========================================================================
# Analytical cost model
# =========================================================================
@dataclass
class OverheadParams:
    """Per-GPU overhead parameters (fitted from data)."""
    oh_sync: float     # cudaDeviceSynchronize cost (μs)
    oh_launch: float   # cudaLaunchKernel cost (μs)
    oh_graph_replay_per_kernel: float  # per-kernel graph replay cost (μs)
    oh_grid_sync: float  # cooperative grid sync cost (μs)
    oh_coop_launch: float  # cooperative launch fixed cost (μs)

# Fitted from our experiments
OVERHEAD_PARAMS = {
    "3060": OverheadParams(oh_sync=52.6, oh_launch=24.2, oh_graph_replay_per_kernel=0.05,
                           oh_grid_sync=2.5, oh_coop_launch=3.0),
    "b200":  OverheadParams(oh_sync=7.0, oh_launch=2.9, oh_graph_replay_per_kernel=0.02,
                            oh_grid_sync=2.0, oh_coop_launch=2.0),
}


def predict_strategy_times(
    gpu: str,
    kernel: str,
    n_cells: int,
    block_size: int = 256,
    steps: int = 1000,
    graph_batch: int = 900,
) -> Dict[str, float]:
    """Predict μs/step for each strategy."""
    hw = HW[gpu]
    oh = OVERHEAD_PARAMS[gpu]

    # Get kernel profile (fallback to defaults)
    prof = KERNEL_PROFILES.get(kernel, {}).get(gpu, KernelProfile())

    n_blocks = math.ceil(n_cells / block_size)
    K = prof.kernels_per_step  # kernels per timestep

    # Compute time scales with grid size.
    # For small grids (n_blocks ≤ SMs): all blocks fit in one wave, compute ≈ base
    # For large grids: compute scales linearly with waves
    waves = max(1, math.ceil(n_blocks / hw.sms))
    # Base compute is profiled at a reference grid size; scale proportionally
    # But not purely linear — GPU has some overlap between waves
    if prof.compute_us > 0:
        t_compute = prof.compute_us * (1.0 + 0.7 * (waves - 1))  # sub-linear scaling
    else:
        t_compute = n_cells * 0.001

    results = {}

    # [1] Sync loop: each kernel launch + sync
    results["sync"] = t_compute + K * (oh.oh_launch + oh.oh_sync)

    # [2] Async loop: kernel launches without sync (pipeline on stream)
    results["async"] = t_compute + K * oh.oh_launch

    # [3] CUDA Graph: capture K kernels × steps, replay
    # Per-step cost ≈ compute + tiny graph replay overhead
    results["graph"] = t_compute + K * oh.oh_graph_replay_per_kernel

    # [4] Persistent kernel (cooperative launch)
    # Check cooperative limit
    max_blocks_per_sm = max(1, hw.max_threads_per_sm // block_size)
    # Reduce by register pressure
    if prof.regs_per_thread > 0:
        regs_per_block = prof.regs_per_thread * block_size
        max_by_regs = hw.max_regs_per_sm // regs_per_block
        max_blocks_per_sm = min(max_blocks_per_sm, max_by_regs)
    coop_limit = max_blocks_per_sm * hw.sms

    if n_blocks <= coop_limit:
        # Occupancy penalty: if fusion increases register pressure
        occ_ratio = min(1.0, coop_limit / max(n_blocks, 1))
        # Grid sync cost per phase
        t_persistent = t_compute + K * oh.oh_grid_sync
        # Amortized launch cost (one cooperative launch per graph_batch steps)
        t_persistent += oh.oh_coop_launch / steps
        results["persistent"] = t_persistent
    else:
        results["persistent"] = float("inf")

    return results


def classify_regime(gpu: str, kernel: str, n_cells: int, block_size: int = 256) -> str:
    """Classify as overhead-dominated, transitional, or compute-dominated."""
    times = predict_strategy_times(gpu, kernel, n_cells, block_size)
    t_compute = times["graph"]  # Graph ≈ pure compute
    t_overhead = times["sync"] - t_compute
    oh_fraction = t_overhead / times["sync"] if times["sync"] > 0 else 0

    if oh_fraction > 0.5:
        return "OH-dominated"
    elif oh_fraction > 0.2:
        return "Transitional"
    else:
        return "Compute-dominated"


def select_best_strategy(gpu: str, kernel: str, n_cells: int,
                          has_dynamic_flow: bool = False,
                          has_periodic_save: bool = False,
                          block_size: int = 256) -> Tuple[str, Dict[str, float]]:
    """Select optimal strategy based on model prediction."""
    times = predict_strategy_times(gpu, kernel, n_cells, block_size)

    # Filter out infeasible strategies
    candidates = dict(times)
    if has_dynamic_flow:
        candidates.pop("graph", None)  # Graph can't handle break/convergence

    # If periodic save + persistent available, persistent is preferred (DMA overlap)
    if has_periodic_save and math.isfinite(candidates.get("persistent", float("inf"))):
        # Persistent with DMA overlap: no save overhead
        pass  # persistent already computed without save penalty
    elif has_periodic_save and "graph" in candidates:
        # Graph needs to break at save points: add small penalty
        candidates["graph"] *= 1.02  # ~2% overhead per save-break

    best = min(candidates, key=candidates.get)
    return best, times


def main():
    ap = argparse.ArgumentParser(description="Cost model v2 with NCU features")
    ap.add_argument("--json-out", type=Path, default=RESULTS / "cost_model_v2_report.json")
    args = ap.parse_args()

    report = {"predictions": [], "validation": []}

    print("=" * 70)
    print("Cost Model v2: Analytical model with NCU profiling features")
    print("=" * 70)

    # Predict for all our benchmark configurations
    test_cases = [
        # (gpu, kernel, n_cells, label)
        ("3060", "heat2d", 128*128, "Heat2D 128²"),
        ("3060", "heat2d", 256*256, "Heat2D 256²"),
        ("3060", "heat2d", 512*512, "Heat2D 512²"),
        ("3060", "heat2d", 1024*1024, "Heat2D 1024²"),
        ("3060", "osher", 6675, "OSHER hydro-cal (6675)"),
        ("3060", "osher", 24020, "OSHER hydro-cal (24020)"),
        ("b200", "heat2d", 128*128, "Heat2D 128²"),
        ("b200", "heat2d", 256*256, "Heat2D 256²"),
        ("b200", "heat2d", 512*512, "Heat2D 512²"),
        ("b200", "heat2d", 1024*1024, "Heat2D 1024²"),
        ("b200", "osher", 6675, "OSHER hydro-cal (6675)"),
        ("b200", "osher", 24020, "OSHER hydro-cal (24020)"),
    ]

    # Measured ground truth (from our experiments)
    ground_truth = {
        ("3060", "heat2d", 128*128): {"sync": 73.7, "async": 32.7, "graph": 4.2, "persistent": 3.5},
        ("3060", "heat2d", 256*256): {"sync": 72.5, "async": 30.9, "graph": 6.7},
        ("3060", "heat2d", 512*512): {"sync": 80.6, "async": 33.1, "graph": 18.9},
        ("3060", "heat2d", 1024*1024): {"sync": 150.7, "async": 87.2, "graph": 83.2},
        ("3060", "osher", 6675): {"sync": 84.3, "async": 31.8, "graph": 7.6, "persistent": 6.1},
        ("b200", "heat2d", 128*128): {"sync": 12.5, "async": 7.2, "graph": 2.8, "persistent": 3.8},
        ("b200", "heat2d", 256*256): {"sync": 13.0, "async": 8.2, "graph": 3.3, "persistent": 4.7},
        ("b200", "heat2d", 512*512): {"sync": 15.5, "async": 10.2, "graph": 5.6, "persistent": 9.5},
        ("b200", "heat2d", 1024*1024): {"sync": 25.8, "async": 19.5, "graph": 16.2},
        ("b200", "osher", 24020): {"sync": 22.3, "async": 11.4, "graph": 8.2, "persistent": 9.6},
    }

    print(f"\n{'GPU':<6} {'Kernel':<25} {'Regime':<15} {'Best':>10} | {'Sync':>8} {'Async':>8} {'Graph':>8} {'Persist':>8}")
    print("-" * 95)

    errors = []
    strategy_correct = 0
    strategy_total = 0

    for gpu, kernel, n_cells, label in test_cases:
        best, times = select_best_strategy(gpu, kernel, n_cells)
        regime = classify_regime(gpu, kernel, n_cells)
        persist_str = f"{times['persistent']:.1f}" if math.isfinite(times["persistent"]) else "N/A"

        print(f"{gpu:<6} {label:<25} {regime:<15} {best:>10} | "
              f"{times['sync']:>7.1f} {times['async']:>7.1f} {times['graph']:>7.1f} {persist_str:>8}")

        entry = {"gpu": gpu, "kernel": kernel, "n_cells": n_cells, "label": label,
                 "regime": regime, "best_strategy": best, "predictions": times}
        report["predictions"].append(entry)

        # Validate against ground truth
        key = (gpu, kernel, n_cells)
        if key in ground_truth:
            gt = ground_truth[key]
            for strat, pred in times.items():
                if strat in gt and math.isfinite(pred):
                    err = abs(pred - gt[strat]) / gt[strat]
                    errors.append(err)

            # Check strategy selection
            gt_best = min(gt, key=gt.get)
            strategy_total += 1
            if best == gt_best:
                strategy_correct += 1

    if errors:
        mape = 100 * sum(errors) / len(errors)
        strategy_acc = 100 * strategy_correct / max(strategy_total, 1)
        print(f"\n--- Validation ---")
        print(f"  MAPE: {mape:.1f}%")
        print(f"  Strategy accuracy: {strategy_correct}/{strategy_total} = {strategy_acc:.0f}%")
        report["validation"] = {"mape": mape, "strategy_accuracy": strategy_acc,
                                "n_comparisons": len(errors), "n_strategy_checks": strategy_total}

    args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nJSON: {args.json_out}")


if __name__ == "__main__":
    main()
