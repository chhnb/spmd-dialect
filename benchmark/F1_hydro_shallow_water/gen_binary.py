#!/usr/bin/env python3
"""Generate binary mesh data for F1 CUDA/Kokkos benchmarks."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from mesh_loader import load_hydro_mesh

def generate(mesh_name="default"):
    mesh = load_hydro_mesh(mesh=mesh_name)
    CEL = mesh["CEL"]

    if mesh_name == "default":
        out_dir = os.path.join(os.path.dirname(__file__), "data", "binary")
    else:
        out_dir = os.path.join(os.path.dirname(__file__), f"data_{mesh_name}", "binary")
    os.makedirs(out_dir, exist_ok=True)

    # F1 uses 1-indexed arrays: [5][CEL+1] for edge arrays, [CEL+1] for cell arrays
    # For binary, store as flat fp64 arrays

    # Cell arrays [CEL+1] — fp64
    for name in ["H", "U", "V", "Z", "W", "ZBC", "ZB1", "FNC", "AREA"]:
        mesh[name].astype(np.float64).tofile(os.path.join(out_dir, f"{name}.bin"))

    # Edge arrays [5][CEL+1] — fp64, stored as flat [5*(CEL+1)]
    for name in ["SIDE", "COSF", "SINF", "SLCOS", "SLSIN"]:
        mesh[name].astype(np.float64).tofile(os.path.join(out_dir, f"{name}.bin"))

    # Integer arrays [5][CEL+1]
    for name in ["NAC", "KLAS", "NV"]:
        arr = mesh[name]
        if arr.ndim == 1:
            arr.astype(np.int32).tofile(os.path.join(out_dir, f"{name}.bin"))
        else:
            arr.astype(np.int32).tofile(os.path.join(out_dir, f"{name}.bin"))

    # MBQ, NNQ, MBZ, NNZ — boundary data
    for name in ["MBQ", "NNQ", "MBZ", "NNZ"]:
        mesh[name].astype(np.int32).tofile(os.path.join(out_dir, f"{name}.bin"))

    # params.txt
    with open(os.path.join(out_dir, "params.txt"), "w") as f:
        f.write(f"{CEL}\n")
        f.write(f"{mesh['NOD']}\n")
        f.write(f"{mesh['HM1']}\n")
        f.write(f"{mesh['HM2']}\n")
        f.write(f"{mesh['NZ']}\n")
        f.write(f"{mesh['NQ']}\n")

    print(f"Generated F1 binary data for mesh='{mesh_name}' in {out_dir}")
    print(f"  CEL={CEL}, NOD={mesh['NOD']}, HM1={mesh['HM1']}, HM2={mesh['HM2']}")

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "default"
    generate(name)
