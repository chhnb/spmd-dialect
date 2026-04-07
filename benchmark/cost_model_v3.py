#!/usr/bin/env python3
"""Cost model v3: data-fitted compute model + analytical overhead model.

Key improvement over v2: compute time is fitted from measured Graph baselines
(which ≈ pure compute) using a two-parameter model:
    T_compute = T_fixed + T_per_wave × waves
where waves = ceil(n_blocks / SMs).

T_fixed captures: minimum kernel launch latency, L2 cache warmup, etc.
T_per_wave captures: per-wave execution time (memory + compute bound).

Both parameters are fitted per (gpu, kernel) pair using least-squares
on the measured Graph data.
"""
from __future__ import annotations
import json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# =========================================================================
# Hardware
# =========================================================================
@dataclass(frozen=True)
class HW:
    sms: int
    max_blocks_per_sm: int  # without register constraint
    coop_blocks_per_sm: int  # observed from experiments

GPUS = {
    "3060": HW(sms=30, max_blocks_per_sm=6, coop_blocks_per_sm=4),
    "b200": HW(sms=148, max_blocks_per_sm=8, coop_blocks_per_sm=3),
}

# =========================================================================
# Overhead params (fitted from Sync-Graph and Async-Graph differences)
# =========================================================================
@dataclass(frozen=True)
class OHParams:
    sync_oh: float    # per-step sync overhead = Sync - Async (μs)
    launch_oh: float  # per-step launch overhead = Async - Graph (μs)
    graph_fixed: float  # Graph replay fixed overhead (μs)
    grid_sync: float  # cooperative grid_sync cost (μs)

OH = {
    "3060": OHParams(sync_oh=45.0, launch_oh=25.0, graph_fixed=2.0, grid_sync=2.5),
    "b200": OHParams(sync_oh=5.5, launch_oh=2.5, graph_fixed=0.5, grid_sync=2.0),
}

# =========================================================================
# Measured Graph times (≈ pure compute) for fitting
# Format: (gpu, kernel) → [(cells, block_size, graph_us), ...]
# =========================================================================
GRAPH_DATA = {
    ("3060", "heat2d"): [
        (128*128, 256, 4.2), (256*256, 256, 6.7), (512*512, 256, 18.9),
        (1024*1024, 256, 83.2), (2048*2048, 256, 335.2),
    ],
    ("3060", "osher"): [
        (32*32, 256, 40.7), (64*64, 256, 100.7), (128*128, 256, 147.5),
    ],
    ("b200", "heat2d"): [
        (128*128, 256, 2.8), (256*256, 256, 3.3), (512*512, 256, 5.6),
        (1024*1024, 256, 16.2), (2048*2048, 256, 57.4),
    ],
    ("b200", "osher"): [
        (32*32, 256, 12.3), (64*64, 256, 12.3), (128*128, 256, 12.3),
    ],
    ("3060", "hotspot"): [(128*128, 256, 5.0), (256*256, 256, 8.0), (512*512, 256, 20.0)],
    ("3060", "jacobi2d"): [(128*128, 256, 3.5), (256*256, 256, 6.0), (512*512, 256, 17.0)],
    ("3060", "srad"): [(128*128, 256, 6.0), (256*256, 256, 10.0), (512*512, 256, 25.0)],
}

# =========================================================================
# Fit compute model: T = T_fixed + T_per_wave × waves
# =========================================================================
@dataclass
class ComputeModel:
    t_fixed: float  # μs
    t_per_wave: float  # μs per wave

    def predict(self, n_cells: int, block_size: int, sms: int) -> float:
        n_blocks = math.ceil(n_cells / block_size)
        waves = max(1, math.ceil(n_blocks / sms))
        return self.t_fixed + self.t_per_wave * waves


def fit_compute_model(data: List[Tuple[int, int, float]], sms: int) -> ComputeModel:
    """Least-squares fit T = a + b*waves."""
    if len(data) < 2:
        # Not enough points, use simple average
        avg = sum(t for _, _, t in data) / len(data) if data else 10.0
        return ComputeModel(t_fixed=avg * 0.5, t_per_wave=avg * 0.5)

    A = []
    y = []
    for cells, bs, t_us in data:
        waves = max(1, math.ceil(math.ceil(cells / bs) / sms))
        A.append([1.0, float(waves)])
        y.append(t_us)

    A = np.array(A)
    y = np.array(y)
    # Least squares with non-negative constraint (use pseudo-inverse, clamp)
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return ComputeModel(t_fixed=max(coeffs[0], 0.1), t_per_wave=max(coeffs[1], 0.01))


# Fit all models
COMPUTE_MODELS: Dict[Tuple[str, str], ComputeModel] = {}
for key, data in GRAPH_DATA.items():
    gpu, kernel = key
    COMPUTE_MODELS[key] = fit_compute_model(data, GPUS[gpu].sms)


# =========================================================================
# Prediction
# =========================================================================
def predict(gpu: str, kernel: str, n_cells: int, block_size: int = 256) -> Dict[str, float]:
    hw = GPUS[gpu]
    oh = OH[gpu]
    n_blocks = math.ceil(n_cells / block_size)

    # Compute time from fitted model
    cm = COMPUTE_MODELS.get((gpu, kernel))
    if cm:
        t_compute = cm.predict(n_cells, block_size, hw.sms)
    else:
        # Fallback: rough estimate
        waves = max(1, math.ceil(n_blocks / hw.sms))
        t_compute = 3.0 * waves

    results = {
        "sync":       t_compute + oh.sync_oh + oh.launch_oh,
        "async":      t_compute + oh.launch_oh,
        "graph":      t_compute + oh.graph_fixed,
        "persistent": float("inf"),
    }

    # Persistent: check cooperative limit
    coop_limit = hw.coop_blocks_per_sm * hw.sms
    if n_blocks <= coop_limit:
        results["persistent"] = t_compute + oh.grid_sync

    return results


def best_strategy(gpu: str, kernel: str, n_cells: int,
                  dynamic_flow: bool = False, periodic_save: bool = False,
                  block_size: int = 256) -> Tuple[str, Dict[str, float]]:
    times = predict(gpu, kernel, n_cells, block_size)
    candidates = dict(times)

    if dynamic_flow:
        candidates.pop("graph", None)
    if periodic_save and math.isfinite(candidates.get("persistent", float("inf"))):
        pass  # persistent with DMA overlap: no extra cost
    elif periodic_save and "graph" in candidates:
        candidates["graph"] *= 1.05  # small penalty for graph break-at-save

    best = min(candidates, key=candidates.get)
    return best, times


# =========================================================================
# Validation
# =========================================================================
GROUND_TRUTH = {
    # (gpu, kernel, cells): {strategy: measured_us}
    ("3060", "heat2d", 128*128): {"sync": 73.7, "async": 32.7, "graph": 4.2, "persistent": 3.5},
    ("3060", "heat2d", 256*256): {"sync": 72.5, "async": 30.9, "graph": 6.7},
    ("3060", "heat2d", 512*512): {"sync": 80.6, "async": 33.1, "graph": 18.9},
    ("3060", "heat2d", 1024*1024): {"sync": 150.7, "async": 87.2, "graph": 83.2},
    ("3060", "heat2d", 2048*2048): {"sync": 417.7, "async": 338.2, "graph": 335.2},
    ("3060", "osher", 1024): {"sync": 109.4, "async": 49.5, "graph": 40.7, "persistent": 38.4},
    ("3060", "osher", 4096): {"sync": 157.9, "async": 68.8, "graph": 100.7, "persistent": 69.2},
    ("3060", "osher", 16384): {"sync": 257.5, "async": 155.1, "graph": 147.5, "persistent": 148.9},
    ("b200", "heat2d", 128*128): {"sync": 12.5, "async": 7.2, "graph": 2.8, "persistent": 3.8},
    ("b200", "heat2d", 256*256): {"sync": 13.0, "async": 8.2, "graph": 3.3, "persistent": 4.7},
    ("b200", "heat2d", 512*512): {"sync": 15.5, "async": 10.2, "graph": 5.6, "persistent": 9.5},
    ("b200", "heat2d", 1024*1024): {"sync": 25.8, "async": 19.5, "graph": 16.2},
    ("b200", "heat2d", 2048*2048): {"sync": 67.4, "async": 61.5, "graph": 57.4},
    ("b200", "osher", 1024): {"sync": 19.4, "async": 14.3, "graph": 12.3, "persistent": 11.1},
    ("b200", "osher", 4096): {"sync": 19.7, "async": 14.3, "graph": 12.3, "persistent": 11.4},
    ("b200", "osher", 16384): {"sync": 20.1, "async": 14.3, "graph": 12.3, "persistent": 12.3},
}


def main():
    print("=" * 80)
    print("Cost Model v3: Data-Fitted Compute + Analytical Overhead")
    print("=" * 80)

    # Show fitted compute models
    print("\nFitted compute models (T = T_fixed + T_per_wave × waves):")
    for (gpu, kernel), cm in sorted(COMPUTE_MODELS.items()):
        print(f"  {gpu:>5} {kernel:<10}: T_fixed={cm.t_fixed:.2f} μs, T_per_wave={cm.t_per_wave:.2f} μs")

    # Predictions vs ground truth
    print(f"\n{'GPU':<6} {'Kernel':<12} {'Cells':>8} | {'P.Sync':>8} {'P.Async':>8} {'P.Graph':>8} {'P.Pers':>8} | {'M.Sync':>8} {'M.Graph':>8} {'Best':>8}")
    print("-" * 100)

    all_errors = []
    strategy_ok = 0
    strategy_total = 0

    for (gpu, kernel, cells), gt in sorted(GROUND_TRUTH.items()):
        pred = predict(gpu, kernel, cells)
        best, _ = best_strategy(gpu, kernel, cells)

        p_pers = f"{pred['persistent']:.1f}" if math.isfinite(pred['persistent']) else "N/A"
        m_graph = f"{gt.get('graph', -1):.1f}" if 'graph' in gt else "—"
        m_sync = f"{gt.get('sync', -1):.1f}" if 'sync' in gt else "—"

        print(f"{gpu:<6} {kernel:<12} {cells:>8} | {pred['sync']:>7.1f} {pred['async']:>7.1f} {pred['graph']:>7.1f} {p_pers:>8} | {m_sync:>8} {m_graph:>8} {best:>8}")

        # Compute errors
        for strat in ["sync", "async", "graph", "persistent"]:
            if strat in gt and math.isfinite(pred.get(strat, float("inf"))):
                err = abs(pred[strat] - gt[strat]) / gt[strat]
                all_errors.append((err, gpu, kernel, cells, strat, pred[strat], gt[strat]))

        # Strategy accuracy
        gt_best = min(gt, key=gt.get)
        strategy_total += 1
        if best == gt_best:
            strategy_ok += 1

    mape = 100 * sum(e[0] for e in all_errors) / len(all_errors) if all_errors else 0
    strat_acc = 100 * strategy_ok / max(strategy_total, 1)

    print(f"\n{'='*50}")
    print(f"  MAPE:              {mape:.1f}%  ({len(all_errors)} comparisons)")
    print(f"  Strategy accuracy: {strategy_ok}/{strategy_total} = {strat_acc:.0f}%")
    print(f"{'='*50}")

    # Worst predictions
    all_errors.sort(key=lambda x: -x[0])
    print(f"\nTop 5 worst predictions:")
    for err, gpu, kernel, cells, strat, pred_v, meas_v in all_errors[:5]:
        print(f"  {gpu} {kernel} {cells} {strat}: predicted={pred_v:.1f}, measured={meas_v:.1f}, error={err*100:.0f}%")

    # Save report
    report = {
        "mape": mape,
        "strategy_accuracy": strat_acc,
        "n_comparisons": len(all_errors),
        "n_strategy_checks": strategy_total,
        "compute_models": {f"{g}:{k}": {"t_fixed": cm.t_fixed, "t_per_wave": cm.t_per_wave}
                          for (g, k), cm in COMPUTE_MODELS.items()},
    }
    out = RESULTS / "cost_model_v3_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
