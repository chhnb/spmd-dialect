#!/usr/bin/env python3
"""Analytical cost model and selector for pk_matrix benchmark data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class Hardware:
    name: str
    sms: int
    reg_file: int


@dataclass(frozen=True)
class Record:
    gpu: str
    kernel: str
    n: int
    cells: int
    b: int
    strategy: str
    us_per_step: float

    @property
    def group(self) -> Tuple[str, str, int]:
        return (self.gpu, self.kernel, self.n)


HARDWARES: Dict[str, Hardware] = {
    "3060": Hardware("RTX 3060 Laptop GPU", sms=30, reg_file=65536),
    "b200": Hardware("NVIDIA B200", sms=148, reg_file=65536),
}


def stable_half_split(group: Tuple[str, str, int]) -> str:
    digest = hashlib.md5(repr(group).encode()).hexdigest()
    return "train" if int(digest, 16) % 2 == 0 else "valid"


def parse_matrix_csv(path: Path, gpu: str) -> Tuple[List[Record], List[Tuple[str, str, int, int, str]]]:
    records: List[Record] = []
    illegal: List[Tuple[str, str, int, int, str]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("kernel,"):
            continue
        parts = line.split(",")
        if len(parts) != 7:
            continue
        kernel, n, cells, b, strategy, us_per_step, _speedup = parts
        if us_per_step == "N/A":
            illegal.append((gpu, kernel, int(n), int(b), strategy))
            continue
        records.append(
            Record(
                gpu=gpu,
                kernel=kernel,
                n=int(n),
                cells=int(cells),
                b=int(b),
                strategy=strategy,
                us_per_step=float(us_per_step),
            )
        )
    return records, illegal


def build_dataset() -> Tuple[List[Record], List[Tuple[str, str, int, int, str]]]:
    all_records: List[Record] = []
    illegal: List[Tuple[str, str, int, int, str]] = []
    for gpu, fname in [("3060", "3060_pk_matrix.csv"), ("b200", "b200_pk_matrix.csv")]:
        rows, bad = parse_matrix_csv(RESULTS / fname, gpu)
        all_records.extend(rows)
        illegal.extend(bad)
    return all_records, illegal


def infer_persistent_limits(records: Iterable[Record]) -> Dict[Tuple[str, str, int], int]:
    legal_max: Dict[Tuple[str, str, int], int] = {}
    for r in records:
        if r.strategy != "persistent":
            continue
        n_blocks = math.ceil(r.cells / r.b)
        key = (r.gpu, r.kernel, r.b)
        legal_max[key] = max(legal_max.get(key, 0), n_blocks)
    return legal_max


class LinearCostModel:
    def __init__(self) -> None:
        self.front_keys: List[Tuple[str, str]] = []
        self.device_keys: List[Tuple[str, str, str]] = []
        self.coeffs: np.ndarray | None = None
        self.persistent_limits: Dict[Tuple[str, str, int], int] = {}

    def _feature_maps(self, records: Iterable[Record]) -> None:
        front = sorted({(r.gpu, r.strategy) for r in records})
        device: List[Tuple[str, str, str]] = []
        gpus = sorted({r.gpu for r in records})
        kernels = sorted({r.kernel for r in records})
        for gpu in gpus:
            for kernel in kernels:
                for term in ("waves", "waves_over_cov", "work"):
                    device.append((gpu, kernel, term))
        self.front_keys = front
        self.device_keys = device

    def legal(self, record: Record) -> bool:
        if record.strategy != "persistent":
            return True
        key = (record.gpu, record.kernel, record.b)
        if key not in self.persistent_limits:
            return False
        n_blocks = math.ceil(record.cells / record.b)
        return n_blocks <= self.persistent_limits[key]

    def _row_features(self, r: Record) -> np.ndarray:
        hw = HARDWARES[r.gpu]
        n_blocks = math.ceil(r.cells / r.b)
        coverage = min(1.0, n_blocks / hw.sms) if hw.sms else 1.0
        waves = math.ceil(n_blocks / hw.sms) if hw.sms else 1
        work = r.cells / max(1.0, r.b * hw.sms)

        feats: List[float] = []
        for key in self.front_keys:
            feats.append(1.0 if key == (r.gpu, r.strategy) else 0.0)
        for key in self.device_keys:
            gpu, kernel, term = key
            if (gpu, kernel) != (r.gpu, r.kernel):
                feats.append(0.0)
                continue
            if term == "waves":
                feats.append(float(waves))
            elif term == "waves_over_cov":
                feats.append(float(waves) / max(coverage, 1e-6))
            elif term == "work":
                feats.append(float(work))
            else:
                feats.append(0.0)
        return np.asarray(feats, dtype=np.float64)

    def fit(self, train: List[Record], legality_source: List[Record]) -> None:
        self._feature_maps(train)
        self.persistent_limits = infer_persistent_limits(legality_source)
        x = np.stack([self._row_features(r) for r in train])
        y = np.asarray([r.us_per_step for r in train], dtype=np.float64)
        ridge = 1e-6
        xtx = x.T @ x + ridge * np.eye(x.shape[1], dtype=np.float64)
        xty = x.T @ y
        self.coeffs = np.linalg.solve(xtx, xty)

    def predict(self, record: Record) -> float:
        if self.coeffs is None:
            raise RuntimeError("model not fitted")
        if not self.legal(record):
            return float("inf")
        pred = float(self._row_features(record) @ self.coeffs)
        return max(pred, 1e-3)

    def dump_coefficients(self) -> Dict[str, Dict[str, float]]:
        if self.coeffs is None:
            return {}
        out: Dict[str, Dict[str, float]] = {"frontend": {}, "device": {}}
        idx = 0
        for gpu, strategy in self.front_keys:
            out["frontend"][f"{gpu}:{strategy}"] = float(self.coeffs[idx])
            idx += 1
        for gpu, kernel, term in self.device_keys:
            out["device"][f"{gpu}:{kernel}:{term}"] = float(self.coeffs[idx])
            idx += 1
        return out


def split_records(records: List[Record]) -> Tuple[List[Record], List[Record]]:
    train: List[Record] = []
    valid: List[Record] = []
    for r in records:
        (train if stable_half_split(r.group) == "train" else valid).append(r)
    return train, valid


def mape(rows: Iterable[Tuple[float, float]]) -> float:
    vals = [abs(pred - truth) / truth for pred, truth in rows if truth > 0 and math.isfinite(pred)]
    return 100.0 * (sum(vals) / max(len(vals), 1))


def evaluate_selector(model: LinearCostModel, valid: List[Record]) -> Dict[str, float]:
    by_group: Dict[Tuple[str, str, int], List[Record]] = {}
    for r in valid:
        by_group.setdefault(r.group, []).append(r)

    pred_truth_pairs = [(model.predict(r), r.us_per_step) for r in valid]
    top1_within_5 = 0
    strategy_ok = 0
    regrets: List[float] = []

    for rows in by_group.values():
        oracle = min(rows, key=lambda r: r.us_per_step)
        predicted = min(rows, key=lambda r: model.predict(r))
        if predicted.us_per_step <= oracle.us_per_step * 1.05:
            top1_within_5 += 1
        if predicted.strategy == oracle.strategy:
            strategy_ok += 1
        regrets.append((predicted.us_per_step - oracle.us_per_step) / oracle.us_per_step)

    total_groups = len(by_group)
    return {
        "mape": mape(pred_truth_pairs),
        "top1_within_5": 100.0 * top1_within_5 / max(total_groups, 1),
        "strategy_family_accuracy": 100.0 * strategy_ok / max(total_groups, 1),
        "normalized_regret": 100.0 * (sum(regrets) / max(len(regrets), 1)),
        "n_validation_rows": float(len(valid)),
        "n_validation_groups": float(total_groups),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=Path, default=RESULTS / "cost_model_report.json")
    args = ap.parse_args()

    records, _illegal = build_dataset()
    train, valid = split_records(records)

    model = LinearCostModel()
    model.fit(train, records)
    metrics = evaluate_selector(model, valid)

    report = {
        "train_rows": len(train),
        "valid_rows": len(valid),
        "coefficients": model.dump_coefficients(),
        "metrics": metrics,
    }
    args.json_out.write_text(json.dumps(report, indent=2))

    print("Cost model report")
    print(f"  train rows: {len(train)}")
    print(f"  valid rows: {len(valid)}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  Top-1 within 5%: {metrics['top1_within_5']:.2f}%")
    print(f"  Strategy-family accuracy: {metrics['strategy_family_accuracy']:.2f}%")
    print(f"  Normalized regret: {metrics['normalized_regret']:.2f}%")
    print(f"  JSON: {args.json_out}")


if __name__ == "__main__":
    main()
