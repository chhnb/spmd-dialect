"""Load the real hydro-cal mesh from data/ directory.

Reads all input files, computes geometry (SIDE, COSF, SINF, AREA),
and returns a dict of 1-indexed numpy arrays ready for the benchmark kernels.
"""
import math
import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _read_lines(filename):
    """Read non-empty, non-comment lines from a data file."""
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r") as f:
        lines = []
        for line in f:
            s = line.strip()
            if s and not s.startswith("!") and not s.startswith("#"):
                lines.append(s)
        return lines


def load_hydro_mesh():
    """Load the full hydro-cal 2D mesh and return a dict of arrays.

    Returns dict with keys:
        CEL, NOD, HM1, HM2, NZ, NQ,
        NAC, KLAS, SIDE, COSF, SINF, SLCOS, SLSIN,
        AREA, ZBC, ZB1, FNC, NV,
        H, U, V, Z, W,
        MBQ, NNQ, MBZ, NNZ,
        XC, YC
    All cell arrays are 1-indexed (index 0 unused).
    2D arrays are [5][CEL+1] (edge index 1..4).
    """

    # ---- GIRD.DAT: NOD, CEL ----
    lines = _read_lines("GIRD.DAT")
    NOD = int(lines[0].split()[0])
    CEL = int(lines[1].split()[0])

    # ---- DEPTH.DAT: HM1, HM2 ----
    lines = _read_lines("DEPTH.DAT")
    HM1 = float(lines[0].split()[0])
    HM2 = float(lines[1].split()[0])

    # ---- BOUNDARY.DAT ----
    lines = _read_lines("BOUNDARY.DAT")
    NZ = int(lines[0].split()[0])
    NQ = int(lines[1].split()[0])

    # ---- PXY.DAT: node coordinates ----
    lines = _read_lines("PXY.DAT")
    # First line is count
    XP = np.zeros(NOD + 1, dtype=np.float64)
    YP = np.zeros(NOD + 1, dtype=np.float64)
    for k in range(1, NOD + 1):
        parts = lines[k].split()
        XP[k] = float(parts[1])
        YP[k] = float(parts[2])
    # Normalize to origin
    XP[1:] -= XP[1:].min()
    YP[1:] -= YP[1:].min()

    # ---- PNAP.DAT: cell-to-node ----
    lines = _read_lines("PNAP.DAT")
    NAP = np.zeros((5, CEL + 1), dtype=np.int32)
    for k in range(1, CEL + 1):
        parts = lines[k].split()
        for j in range(1, 5):
            NAP[j][k] = int(parts[j])

    # ---- PNAC.DAT: cell neighbors ----
    lines = _read_lines("PNAC.DAT")
    NAC = np.zeros((5, CEL + 1), dtype=np.int32)
    for k in range(1, CEL + 1):
        parts = lines[k].split()
        for j in range(1, 5):
            NAC[j][k] = int(parts[j])

    # ---- PKLAS.DAT: edge types ----
    lines = _read_lines("PKLAS.DAT")
    KLAS = np.zeros((5, CEL + 1), dtype=np.int32)
    for k in range(1, CEL + 1):
        parts = lines[k].split()
        for j in range(1, 5):
            KLAS[j][k] = int(parts[j])

    # ---- PZBC.DAT: bed elevation ----
    lines = _read_lines("PZBC.DAT")
    # First line is header "PZBC" or count — skip non-numeric
    ZBC = np.zeros(CEL + 1, dtype=np.float64)
    idx = 0
    for line in lines:
        try:
            val = float(line)
            idx += 1
            if idx <= CEL:
                ZBC[idx] = val
        except ValueError:
            continue

    # ---- MBQ.DAT: Q boundaries ----
    lines = _read_lines("MBQ.DAT")
    NQ_actual = int(lines[0]) if lines else 0
    MBQ = np.zeros(NQ_actual + 1, dtype=np.int32)
    NNQ = np.zeros(NQ_actual + 1, dtype=np.int32)
    for k in range(1, NQ_actual + 1):
        parts = lines[k].split()
        MBQ[k] = int(parts[1])
        NNQ[k] = int(parts[2])

    # ---- MBZ.DAT: Z boundaries ----
    lines = _read_lines("MBZ.DAT")
    NZ_actual = int(lines[0]) if lines else 0
    MBZ = np.zeros(max(NZ_actual + 1, 1), dtype=np.int32)
    NNZ = np.zeros(max(NZ_actual + 1, 1), dtype=np.int32)
    for k in range(1, NZ_actual + 1):
        parts = lines[k].split()
        MBZ[k] = int(parts[1])
        NNZ[k] = int(parts[2])

    # ---- Initial conditions ----
    def _read_cell_values(filename):
        lines = _read_lines(filename)
        arr = np.zeros(CEL + 1, dtype=np.float64)
        idx = 0
        for line in lines:
            try:
                val = float(line)
                idx += 1
                if idx <= CEL:
                    arr[idx] = val
            except ValueError:
                continue
        return arr

    Z_init = _read_cell_values("INITIALLEVEL.DAT")
    U_init = _read_cell_values("INITIALU1.DAT")
    V_init = _read_cell_values("INITIALV1.DAT")
    CV = _read_cell_values("CV.DAT")  # Manning n

    # ---- CONLINK.TXT: 1D-2D coupling ----
    lines = _read_lines("CONLINK.TXT")
    NLINK0 = int(lines[0]) if lines else 0
    # Modify KLAS for coupling boundaries
    for k in range(1, NLINK0 + 1):
        parts = lines[k].split()
        bnd_edge = int(parts[2])   # NLINK2[1] = 2D boundary edge number
        bnd_type = int(parts[3])   # NLINK2[2] = boundary type (13 or 14)
        # Find cells with MBQ matching this edge and set KLAS
        if bnd_type == 13:
            for i in range(1, NQ_actual + 1):
                if NNQ[i] == bnd_edge:
                    cell = MBQ[i]
                    for j in range(1, 5):
                        if NAC[j][cell] == 0 and KLAS[j][cell] == 0:
                            KLAS[j][cell] = 13

    # ---- Compute geometry ----
    NV = np.zeros(CEL + 1, dtype=np.int32)
    SIDE = np.zeros((5, CEL + 1), dtype=np.float64)
    COSF = np.zeros((5, CEL + 1), dtype=np.float64)
    SINF = np.zeros((5, CEL + 1), dtype=np.float64)
    AREA = np.zeros(CEL + 1, dtype=np.float64)
    XC = np.zeros(CEL + 1, dtype=np.float64)
    YC = np.zeros(CEL + 1, dtype=np.float64)

    for i in range(1, CEL + 1):
        if NAP[1][i] == 0:
            continue

        # Determine vertex count
        na4 = NAP[4][i]
        if na4 == 0 or na4 == NAP[1][i]:
            NV[i] = 3
        else:
            NV[i] = 4

        nw = [0, NAP[1][i], NAP[2][i], NAP[3][i], NAP[4][i]]

        # Centroid
        sx = 0.0
        sy = 0.0
        for j in range(1, NV[i] + 1):
            sx += XP[nw[j]]
            sy += YP[nw[j]]
        XC[i] = sx / NV[i]
        YC[i] = sy / NV[i]

        # Area: triangle (1,2,3)
        x1, y1 = XP[nw[1]], YP[nw[1]]
        x2, y2 = XP[nw[2]], YP[nw[2]]
        x3, y3 = XP[nw[3]], YP[nw[3]]
        AREA[i] = abs((y3 - y1) * (x2 - x1) - (x3 - x1) * (y2 - y1)) / 2.0
        if NV[i] == 4:
            x4, y4 = XP[nw[4]], YP[nw[4]]
            AREA[i] += abs((y4 - y1) * (x3 - x1) - (x4 - x1) * (y3 - y1)) / 2.0

        # Edge geometry
        for j in range(1, NV[i] + 1):
            n1 = nw[j]
            n2 = nw[(j % NV[i]) + 1]
            dx = XP[n1] - XP[n2]
            dy = YP[n2] - YP[n1]
            length = math.sqrt(dx * dx + dy * dy)
            SIDE[j][i] = length
            if length > 0.0:
                SINF[j][i] = dx / length
                COSF[j][i] = dy / length

    SLCOS = SIDE * COSF
    SLSIN = SIDE * SINF

    # ---- Derived arrays ----
    ZB1 = ZBC + HM1
    FNC = 9.81 * CV * CV   # g * n^2

    # H = Z - ZBC (water depth)
    H = np.maximum(Z_init - ZBC, HM1)
    W = np.sqrt(U_init ** 2 + V_init ** 2)

    return dict(
        CEL=CEL, NOD=NOD, HM1=HM1, HM2=HM2, NZ=NZ, NQ=NQ,
        NAC=NAC, KLAS=KLAS, SIDE=SIDE, COSF=COSF, SINF=SINF,
        SLCOS=SLCOS, SLSIN=SLSIN,
        AREA=AREA, ZBC=ZBC, ZB1=ZB1, FNC=FNC, NV=NV,
        H=H, U=U_init.copy(), V=V_init.copy(), Z=Z_init.copy(), W=W,
        MBQ=MBQ, NNQ=NNQ, MBZ=MBZ, NNZ=NNZ,
        XC=XC, YC=YC,
    )


if __name__ == "__main__":
    mesh = load_hydro_mesh()
    print(f"Loaded mesh: {mesh['CEL']} cells, {mesh['NOD']} nodes")
    print(f"HM1={mesh['HM1']}, HM2={mesh['HM2']}")
    print(f"NZ={mesh['NZ']}, NQ={mesh['NQ']}")
    print(f"H range: [{mesh['H'][1:].min():.4f}, {mesh['H'][1:].max():.4f}]")
    print(f"Z range: [{mesh['Z'][1:].min():.4f}, {mesh['Z'][1:].max():.4f}]")
    print(f"ZBC range: [{mesh['ZBC'][1:].min():.4f}, {mesh['ZBC'][1:].max():.4f}]")
    print(f"AREA range: [{mesh['AREA'][1:].min():.4f}, {mesh['AREA'][1:].max():.4f}]")
    print(f"SIDE[1] range: [{mesh['SIDE'][1][1:].min():.4f}, {mesh['SIDE'][1][1:].max():.4f}]")
    print(f"FNC range: [{mesh['FNC'][1:].min():.6f}, {mesh['FNC'][1:].max():.6f}]")
    # Count KLAS types
    from collections import Counter
    klas_counts = Counter()
    for j in range(1, 5):
        for i in range(1, mesh['CEL'] + 1):
            klas_counts[mesh['KLAS'][j][i]] += 1
    print(f"KLAS types: {dict(klas_counts)}")
