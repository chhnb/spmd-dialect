#!/usr/bin/env python3
"""Extended simulation-kernel overhead characterization on Taichi.

This complements ``run_overhead_characterization.py`` by benchmarking real
simulation kernels already present in the repository, instead of only the
compact inline kernels from the base script.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

import taichi as ti


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
PYTHON_OH_US = 15.0


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def bench(step_fn, n_steps: int) -> float:
    for _ in range(5):
        step_fn()
    ti.sync()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        step_fn()
    ti.sync()
    return (time.perf_counter() - t0) * 1e6 / n_steps


def classify(us: float):
    oh_pct = min(PYTHON_OH_US / us * 100.0, 100.0) if us > 0 else 0.0
    if oh_pct > 50.0:
        tag = "OH-DOM"
    elif oh_pct < 20.0:
        tag = "COMPUTE"
    else:
        tag = "TRANS"
    return oh_pct, tag


def print_section(title: str):
    print(f"\n{'=' * 68}\n  {title}\n{'=' * 68}")


def record(results, name: str, domain: str, size: str, n_elem: int, step_fn, n_steps: int):
    us = bench(step_fn, n_steps)
    oh_pct, tag = classify(us)
    results.append((name, domain, size, n_elem, us, oh_pct, tag))
    print(f"  {name:<30} {size:>14} {us:>8.1f} us  OH~{oh_pct:>3.0f}%  [{tag}]")


def main():
    parser = argparse.ArgumentParser(description="Extended simulation kernel characterization")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "simulation_characterization_extended.csv",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    lbm = load_module("simext_lbm_taichi", ROOT / "A2_lbm_d2q9" / "lbm_taichi.py")
    sph = load_module("simext_sph_taichi", ROOT / "B2_sph" / "sph_taichi.py")
    mpm = load_module("simext_mpm_taichi", ROOT / "C1_mpm" / "mpm_taichi.py")
    fluid = load_module("simext_fluid_taichi", ROOT / "D2_stable_fluids" / "fluid_taichi.py")
    hydro = load_module("simext_hydro_taichi", ROOT / "F1_hydro_shallow_water" / "hydro_taichi.py")
    hydro_ref = load_module(
        "simext_hydro_ref_taichi", ROOT / "F2_hydro_refactored" / "hydro_refactored_taichi.py"
    )
    euler = load_module("simext_euler_taichi", ROOT / "A4_euler_compressible" / "euler_taichi.py")
    pic = load_module("simext_pic_taichi", ROOT / "C2_pic" / "pic_taichi.py")
    cloth = load_module("simext_cloth_taichi", ROOT / "D1_cloth" / "cloth_taichi.py")
    dem = load_module("simext_dem_taichi", ROOT / "B3_dem" / "dem_taichi.py")

    print_section("16. LBM D2Q9")
    for nx, ny in [(512, 256), (1024, 512), (2048, 1024)]:
        step_fn, _, _ = lbm.run(nx, ny, steps=10, backend="cuda")
        record(results, "LBM_D2Q9", "CFD", f"{nx}x{ny}", nx * ny, step_fn, 100)
        ti.reset()

    print_section("17. SPH Neighbor Grid")
    for n in [1024, 4096, 16384]:
        step_fn, _, _ = sph.run(n, steps=5, backend="cuda")
        record(results, "SPH_Grid", "Particle", f"N={n}", n, step_fn, 100 if n <= 4096 else 40)
        ti.reset()

    print_section("18. MPM88")
    for n_grid, n_particles in [(64, 4096), (128, 8192), (256, 16384)]:
        step_fn, _, _ = mpm.run(n_grid=n_grid, n_particles=n_particles, steps=1, backend="cuda")
        record(results, "MPM88", "MultiKernel", f"grid{n_grid}_p{n_particles}", n_particles, step_fn, 100)
        ti.reset()

    print_section("19. Stable Fluids")
    for n in [128, 256, 512]:
        step_fn, _, _ = fluid.run(n, steps=1, jacobi_iters=50, backend="cuda")
        record(results, "StableFluids", "MultiKernel", f"{n}x{n}", n * n, step_fn, 30)
        ti.reset()

    print_section("20. Hydro SWE (Osher)")
    for n in [32, 64, 128, 256]:
        step_fn, _, _ = hydro.run(n, steps=10, backend="cuda")
        record(results, "HydroSWE_Osher", "CFD", f"{n}x{n}", n * n, step_fn, 50)
        ti.reset()

    print_section("21. Hydro Refactored")
    for days in [1, 5, 10]:
        step_fn, _, _ = hydro_ref.run(days=days, backend="cuda")
        record(results, "HydroRefactored", "MultiKernel", f"6675cells_{days}d", 6675, step_fn, 20)
        ti.reset()

    print_section("22. Compressible Euler")
    for n in [128, 256, 512]:
        step_fn, _, _ = euler.run(n, steps=10, backend="cuda")
        record(results, "Euler2D_Rusanov", "CFD", f"{n}x{n}", n * n, step_fn, 50)
        ti.reset()

    print_section("23. PIC Electrostatic")
    for np_, ng in [(4096, 256), (16384, 512), (65536, 1024)]:
        step_fn, _, _ = pic.run(n_particles=np_, n_grid=ng, steps=1, backend="cuda")
        record(results, "PIC1D", "ParticleGrid", f"p{np_}_g{ng}", np_, step_fn, 30)
        ti.reset()

    print_section("24. Cloth Spring-Mass")
    for n in [32, 64, 128, 256]:
        step_fn, _, _ = cloth.run(n, steps=20, backend="cuda")
        record(results, "ClothSpring", "Structure", f"{n}x{n}", n * n, step_fn, 40)
        ti.reset()

    print_section("25. DEM")
    for n in [2048, 8192, 32768]:
        step_fn, _, _ = dem.run(n, steps=5, backend="cuda")
        record(results, "DEM_Grid", "Particle", f"N={n}", n, step_fn, 40)
        ti.reset()

    print(f"\n{'=' * 88}")
    print(f"  EXTENDED SUMMARY: {len(results)} configurations across 10 additional kernel families")
    print(f"{'=' * 88}")
    print(f"{'Kernel':<24} {'Domain':<14} {'Size':>18} {'us/step':>10} {'OH%':>6} {'Class':>10}")
    print("-" * 88)
    oh_dom = trans = compute_dom = 0
    for name, domain, size, _, us, oh_pct, tag in results:
        if tag == "OH-DOM":
            oh_dom += 1
        elif tag == "TRANS":
            trans += 1
        else:
            compute_dom += 1
        print(f"{name:<24} {domain:<14} {size:>18} {us:>9.1f} {oh_pct:>5.0f}% {tag:>10}")

    print(
        f"\nClassification: OH-dominated={oh_dom}  Transitional={trans}  Compute-dominated={compute_dom}"
    )

    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel", "domain", "size", "elements", "us_per_step", "oh_pct", "class"])
        writer.writerows(results)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
