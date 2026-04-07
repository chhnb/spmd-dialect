#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

CASES = {
    "heat2d_128_graph": RESULTS / "ncu_heat2d_128.csv",
    "osher_64_persistent": RESULTS / "ncu_osher_64_persistent.csv",
    "osher_128_graph": RESULTS / "ncu_osher_128_graph.csv",
}

METRICS = {
    "dram__bytes_read.sum": "dram_bytes_read",
    "dram__bytes_write.sum": "dram_bytes_write",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy_pct",
    "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum": "fp32_inst",
    "smsp__sass_thread_inst_executed_op_fp64_pred_on.sum": "fp64_inst",
}


def parse_case(path: Path) -> dict:
    rows = []
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or not line.startswith('"'):
                continue
            rows.append(next(csv.reader([line])))
    header = rows[0]
    data_rows = rows[1:]
    idx = {name: i for i, name in enumerate(header)}
    out = {
        "kernel_name": None,
        "block_size": None,
        "grid_size": None,
        "metrics": {},
    }
    for row in data_rows:
        metric_name = row[idx["Metric Name"]]
        metric_val = row[idx["Metric Value"]]
        if out["kernel_name"] is None:
            out["kernel_name"] = row[idx["Kernel Name"]]
            out["block_size"] = row[idx["Block Size"]]
            out["grid_size"] = row[idx["Grid Size"]]
        key = METRICS.get(metric_name)
        if key is None:
            continue
        try:
            val = float(metric_val)
        except ValueError:
            val = metric_val
        out["metrics"][key] = val
    return out


def fmt_metric(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if v >= 100:
        return f"{v:.2f}"
    return f"{v:.2f}"


def main() -> None:
    summary = {case: parse_case(path) for case, path in CASES.items()}
    json_path = RESULTS / "ncu_representative_summary.json"
    md_path = RESULTS / "ncu_representative_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Representative NCU Summary",
        "",
        "These metrics summarize three representative kernels for the 3060 cost-model decomposition.",
        "",
        "| Case | Kernel | Block | Grid | SM Throughput % | Achieved Occupancy % | DRAM Read (B) | DRAM Write (B) | FP32/FP64 Inst |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case, data in summary.items():
        m = data["metrics"]
        fp_inst = m.get("fp32_inst", m.get("fp64_inst", 0.0))
        lines.append(
            "| {case} | `{kernel}` | `{block}` | `{grid}` | {sm:.2f} | {occ:.2f} | {rd} | {wr} | {fp} |".format(
                case=case,
                kernel=data["kernel_name"].split("(")[0],
                block=data["block_size"],
                grid=data["grid_size"],
                sm=m.get("sm_throughput_pct", 0.0),
                occ=m.get("achieved_occupancy_pct", 0.0),
                rd=fmt_metric(m.get("dram_bytes_read", 0.0)),
                wr=fmt_metric(m.get("dram_bytes_write", 0.0)),
                fp=fmt_metric(fp_inst),
            )
        )
    lines += [
        "",
        "## Notes",
        "",
        "- `heat2d_128_graph` uses the `k_heat2d` kernel from `pk_matrix_benchmark.cu` under the `graph` strategy.",
        "- `osher_64_persistent` uses the `persistent_kernel` path from `hydro_cuda_osher.cu`.",
        "- `osher_128_graph` uses the `shallow_water_step` kernel under the `graph` path.",
        "- FP instruction count is currently a proxy metric (`fp32_inst` for Heat2D, `fp64_inst` for Osher).",
    ]
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
