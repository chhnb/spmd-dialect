"""Load the real hydro-cal mesh for the refactored kernel (F2).

Data layout: edges stored as flat [4*CELL] arrays (0-indexed).
Cells stored as [CELL] arrays (0-indexed).
NAC stores 1-indexed neighbor IDs (0 means no neighbor).

This loader reads from data/ directory and returns numpy arrays
in the EXACT layout expected by the refactored CUDA kernel.
"""
import math
import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _read_lines(filename):
    """Read non-empty, non-comment lines from a data file."""
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'r', encoding='latin-1') as f:
        lines = []
        for line in f:
            s = line.strip()
            if s and not s.startswith("!") and not s.startswith("#"):
                lines.append(s)
        return lines


def load_mesh():
    """Load the refactored hydro-cal mesh.

    Returns dict with:
        CELL, NOD, HM1, HM2, DT, MDT, NDAYS, JL,
        # Cell arrays [CELL] (0-indexed)
        H, U, V, Z, W, ZBC, ZB1, AREA, FNC,
        # Side arrays [4*CELL] (edge j of cell i at index 4*i+j)
        NAC, KLAS, SIDE, COSF, SINF, SLCOS, SLSIN,
        FLUX0, FLUX1, FLUX2, FLUX3,  (output buffers)
        # Boundary data
        QT, DQT, ZT, DZT, NQ, NZ, NHQ,
        BoundaryFeature,
        # Mesh geometry
        XC, YC, NV
    """
    # ---- GIRD.DAT: NOD, CEL ----
    lines = _read_lines("GIRD.DAT")
    NOD = int(lines[0].split()[0])
    CELL = int(lines[1].split()[0])

    # ---- DEPTH.DAT: HM1, HM2 ----
    lines = _read_lines("DEPTH.DAT")
    HM1 = float(lines[0].split()[0])
    HM2 = float(lines[1].split()[0])

    # ---- TIME.DAT: MDT, NDAYS ----
    lines = _read_lines("TIME.DAT") if os.path.exists(os.path.join(DATA_DIR, "TIME.DAT")) else ["3600", "50", "1"]
    MDT = int(lines[0].split()[0])
    NDAYS = int(lines[1].split()[0])

    # ---- CALTIME.DAT: DT ----
    lines = _read_lines("CALTIME.DAT") if os.path.exists(os.path.join(DATA_DIR, "CALTIME.DAT")) else ["0", "4"]
    DT = float(lines[1].split()[0])

    # ---- JL.DAT ----
    JL = 0.0
    if os.path.exists(os.path.join(DATA_DIR, "JL.DAT")):
        lines = _read_lines("JL.DAT")
        if lines:
            JL = float(lines[0].split()[0])

    # ---- BOUNDARY.DAT ----
    lines = _read_lines("BOUNDARY.DAT")
    NZ = int(lines[0].split()[0])
    NQ = int(lines[1].split()[0])
    NZQ = int(lines[2].split()[0]) if len(lines) > 2 else 0
    NHQ = int(lines[3].split()[0]) if len(lines) > 3 else 5

    # ---- PXY.DAT: node coordinates ----
    lines = _read_lines("PXY.DAT")
    XP = np.zeros(NOD + 1, dtype=np.float32)
    YP = np.zeros(NOD + 1, dtype=np.float32)
    for k in range(1, NOD + 1):
        parts = lines[k].split()
        XP[k] = float(parts[1])
        YP[k] = float(parts[2])
    XP[1:] -= XP[1:].min()
    YP[1:] -= YP[1:].min()

    # ---- PNAP.DAT: cell-to-node ----
    lines = _read_lines("PNAP.DAT")
    NAP = np.zeros((5, CELL + 1), dtype=np.int32)
    for k in range(1, CELL + 1):
        parts = lines[k].split()
        for j in range(1, 5):
            NAP[j][k] = int(parts[j])

    # ---- PNAC.DAT: cell neighbors (1-indexed, 0=no neighbor) ----
    lines = _read_lines("PNAC.DAT")
    # Refactored layout: NAC[4*cell + edge], stores 1-indexed (kernel does NC-1)
    NAC = np.zeros(4 * CELL, dtype=np.int32)
    for k in range(1, CELL + 1):
        parts = lines[k].split()
        for j in range(1, 5):
            NAC[4 * (k - 1) + (j - 1)] = int(parts[j])

    # ---- PKLAS.DAT: edge types ----
    lines = _read_lines("PKLAS.DAT")
    KLAS = np.zeros(4 * CELL, dtype=np.float32)  # float in refactored version
    for k in range(1, CELL + 1):
        parts = lines[k].split()
        for j in range(1, 5):
            KLAS[4 * (k - 1) + (j - 1)] = float(parts[j])

    # ---- PZBC.DAT: bed elevation ----
    lines = _read_lines("PZBC.DAT")
    ZBC = np.zeros(CELL, dtype=np.float32)
    idx = 0
    for line in lines:
        try:
            val = float(line)
            if idx < CELL:
                ZBC[idx] = val
                idx += 1
        except ValueError:
            continue

    # ---- Initial conditions ----
    def _read_cell_values(filename):
        lines = _read_lines(filename)
        arr = np.zeros(CELL, dtype=np.float32)
        idx = 0
        for line in lines:
            try:
                val = float(line)
                if idx < CELL:
                    arr[idx] = val
                    idx += 1
            except ValueError:
                continue
        return arr

    Z_init = _read_cell_values("INITIALLEVEL.DAT")
    U_init = _read_cell_values("INITIALU1.DAT")
    V_init = _read_cell_values("INITIALV1.DAT")
    CV = _read_cell_values("CV.DAT")

    # ---- Compute geometry ----
    NV = np.zeros(CELL, dtype=np.int32)
    SIDE = np.zeros(4 * CELL, dtype=np.float32)
    COSF = np.zeros(4 * CELL, dtype=np.float32)
    SINF = np.zeros(4 * CELL, dtype=np.float32)
    AREA = np.zeros(CELL, dtype=np.float32)
    XC = np.zeros(CELL, dtype=np.float32)
    YC = np.zeros(CELL, dtype=np.float32)

    for i in range(CELL):
        ci = i + 1  # 1-indexed for NAP
        if NAP[1][ci] == 0:
            continue
        na4 = NAP[4][ci]
        if na4 == 0 or na4 == NAP[1][ci]:
            NV[i] = 3
        else:
            NV[i] = 4

        nw = [0, NAP[1][ci], NAP[2][ci], NAP[3][ci], NAP[4][ci]]
        sx = sy = 0.0
        for j in range(1, NV[i] + 1):
            sx += XP[nw[j]]
            sy += YP[nw[j]]
        XC[i] = sx / NV[i]
        YC[i] = sy / NV[i]

        x1, y1 = XP[nw[1]], YP[nw[1]]
        x2, y2 = XP[nw[2]], YP[nw[2]]
        x3, y3 = XP[nw[3]], YP[nw[3]]
        AREA[i] = abs((y3 - y1) * (x2 - x1) - (x3 - x1) * (y2 - y1)) / 2.0
        if NV[i] == 4:
            x4, y4 = XP[nw[4]], YP[nw[4]]
            AREA[i] += abs((y4 - y1) * (x3 - x1) - (x4 - x1) * (y3 - y1)) / 2.0

        for j in range(1, NV[i] + 1):
            n1 = nw[j]
            n2 = nw[(j % NV[i]) + 1]
            dx = XP[n1] - XP[n2]
            dy = YP[n2] - YP[n1]
            length = math.sqrt(dx * dx + dy * dy)
            edge_idx = 4 * i + (j - 1)
            SIDE[edge_idx] = length
            if length > 0:
                SINF[edge_idx] = dx / length
                COSF[edge_idx] = dy / length

    SLCOS = SIDE * COSF
    SLSIN = SIDE * SINF
    ZB1 = ZBC + HM1
    FNC = 9.81 * CV * CV
    H = np.maximum(Z_init - ZBC, HM1).astype(np.float32)
    W = np.sqrt(U_init ** 2 + V_init ** 2).astype(np.float32)

    # FLUX buffers (output, initialized to 0)
    FLUX0 = np.zeros(4 * CELL, dtype=np.float32)
    FLUX1 = np.zeros(4 * CELL, dtype=np.float32)
    FLUX2 = np.zeros(4 * CELL, dtype=np.float32)
    FLUX3 = np.zeros(4 * CELL, dtype=np.float32)

    # Boundary data (simplified - QT, DQT for time-varying boundaries)
    # For the benchmark, we initialize these to 0 since we're testing the compute kernel
    QT = np.zeros(NDAYS * CELL, dtype=np.float32)
    DQT = np.zeros(NDAYS * CELL, dtype=np.float32)
    BoundaryFeature = np.zeros(CELL, dtype=np.float32)

    # MBQ data
    lines = _read_lines("MBQ.DAT")
    NQ_actual = int(lines[0]) if lines else 0

    steps_per_day = int(MDT / DT)

    return dict(
        CELL=CELL, NOD=NOD, HM1=np.float32(HM1), HM2=np.float32(HM2),
        DT=np.float32(DT), MDT=MDT, NDAYS=NDAYS, JL=np.float32(JL),
        steps_per_day=steps_per_day, NQ=NQ, NZ=NZ,
        H=H, U=U_init.astype(np.float32), V=V_init.astype(np.float32),
        Z=Z_init.astype(np.float32), W=W,
        ZBC=ZBC, ZB1=ZB1.astype(np.float32), AREA=AREA, FNC=FNC.astype(np.float32),
        NAC=NAC, KLAS=KLAS, SIDE=SIDE, COSF=COSF, SINF=SINF,
        SLCOS=SLCOS.astype(np.float32), SLSIN=SLSIN.astype(np.float32),
        FLUX0=FLUX0, FLUX1=FLUX1, FLUX2=FLUX2, FLUX3=FLUX3,
        QT=QT, DQT=DQT, BoundaryFeature=BoundaryFeature,
        XC=XC, YC=YC, NV=NV,
    )


if __name__ == "__main__":
    mesh = load_mesh()
    print(f"Loaded mesh: {mesh['CELL']} cells")
    print(f"HM1={mesh['HM1']}, HM2={mesh['HM2']}, DT={mesh['DT']}")
    print(f"Steps/day={mesh['steps_per_day']}, NDAYS={mesh['NDAYS']}")
    print(f"H range: [{mesh['H'].min():.4f}, {mesh['H'].max():.4f}]")
    print(f"Z range: [{mesh['Z'].min():.4f}, {mesh['Z'].max():.4f}]")
    print(f"AREA range: [{mesh['AREA'][mesh['AREA']>0].min():.1f}, {mesh['AREA'].max():.1f}]")
    print(f"SIDE range: [{mesh['SIDE'][mesh['SIDE']>0].min():.1f}, {mesh['SIDE'].max():.1f}]")

    from collections import Counter
    klas_counts = Counter()
    for v in mesh['KLAS']:
        klas_counts[int(v)] += 1
    print(f"KLAS types: {dict(klas_counts)}")
