#!/usr/bin/env python3
"""Generate binary mesh data for the CUDA hydro_osher_benchmark."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from mesh_loader import load_mesh

def generate(mesh_name="default"):
    mesh = load_mesh(mesh=mesh_name)
    CELL = mesh["CELL"]

    # Output directory
    if mesh_name == "default":
        out_dir = os.path.join(os.path.dirname(__file__), "data", "binary")
    else:
        out_dir = os.path.join(os.path.dirname(__file__), f"data_{mesh_name}", "binary")
    os.makedirs(out_dir, exist_ok=True)

    # Cell arrays [CELL] — fp32
    for name in ["H", "U", "V", "Z", "W", "ZBC", "ZB1", "AREA", "FNC"]:
        mesh[name].astype(np.float32).tofile(os.path.join(out_dir, f"{name}.bin"))

    # Edge arrays [4*CELL] — fp32
    for name in ["SIDE", "COSF", "SINF", "SLCOS", "SLSIN", "KLAS"]:
        mesh[name].astype(np.float32).tofile(os.path.join(out_dir, f"{name}.bin"))

    # NAC — int32
    mesh["NAC"].astype(np.int32).tofile(os.path.join(out_dir, "NAC.bin"))

    # FLUX buffers (zeros)
    zeros_edge = np.zeros(4 * CELL, dtype=np.float32)
    for i in range(4):
        zeros_edge.tofile(os.path.join(out_dir, f"FLUX{i}.bin"))

    # params.txt
    with open(os.path.join(out_dir, "params.txt"), "w") as f:
        f.write(f"{CELL}\n")
        f.write(f"{mesh['HM1']}\n")
        f.write(f"{mesh['HM2']}\n")
        f.write(f"{mesh['DT']}\n")
        f.write(f"{mesh['steps_per_day']}\n")

    print(f"Generated binary data for mesh='{mesh_name}' in {out_dir}")
    print(f"  CELL={CELL}, DT={mesh['DT']}, steps_per_day={mesh['steps_per_day']}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "default"
    generate(name)
