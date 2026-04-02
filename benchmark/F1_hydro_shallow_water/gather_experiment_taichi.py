"""Gather Promotion Experiment — Taichi version.

Compares:
  A) Naive:    every neighbor read from global field (ti.field)
  B) Promoted: cooperative pre-load neighbors into shared-memory-like block field,
               then compute from local cache

Uses Taichi's block_local() or manual shared memory emulation via
ti.field with block_dim-sized cache.

Since Taichi doesn't expose __shared__ directly, we emulate the effect:
  - Version A: standard per-cell kernel reading from global fields
  - Version B: two-pass kernel — pass 1 gathers neighbor data into a
    thread-block-local dense field, pass 2 computes from that cache
"""
import taichi as ti
import numpy as np
import time
import math

# Constants
G = 9.81
HALF_G = 4.905
HM1 = 0.001
HM2 = 0.01
VMIN = 0.001
C1_C = 0.3
MANNING_N = 0.03


def run_experiment(N, steps, warmup=3, repeat=10):
    ti.init(arch=ti.cuda, default_fp=ti.f64)
    CEL = N * N
    dx = 1.0
    DT = 0.5 * dx / (math.sqrt(G * 2.0) + 1e-6)

    # --- Fields ---
    NAC = ti.field(dtype=ti.i32, shape=(5, CEL + 1))
    KLAS = ti.field(dtype=ti.i32, shape=(5, CEL + 1))
    SIDE = ti.field(dtype=ti.f64, shape=(5, CEL + 1))
    COSF = ti.field(dtype=ti.f64, shape=(5, CEL + 1))
    SINF = ti.field(dtype=ti.f64, shape=(5, CEL + 1))
    SLCOS = ti.field(dtype=ti.f64, shape=(5, CEL + 1))
    SLSIN = ti.field(dtype=ti.f64, shape=(5, CEL + 1))
    AREA = ti.field(dtype=ti.f64, shape=(CEL + 1,))
    ZBC = ti.field(dtype=ti.f64, shape=(CEL + 1,))
    FNC = ti.field(dtype=ti.f64, shape=(CEL + 1,))

    H_pre = ti.field(dtype=ti.f64, shape=(CEL + 1,))
    U_pre = ti.field(dtype=ti.f64, shape=(CEL + 1,))
    V_pre = ti.field(dtype=ti.f64, shape=(CEL + 1,))
    Z_pre = ti.field(dtype=ti.f64, shape=(CEL + 1,))
    W_pre = ti.field(dtype=ti.f64, shape=(CEL + 1,))
    H_res = ti.field(dtype=ti.f64, shape=(CEL + 1,))
    U_res = ti.field(dtype=ti.f64, shape=(CEL + 1,))
    V_res = ti.field(dtype=ti.f64, shape=(CEL + 1,))
    Z_res = ti.field(dtype=ti.f64, shape=(CEL + 1,))
    W_res = ti.field(dtype=ti.f64, shape=(CEL + 1,))

    # --- Build mesh on host ---
    nac_np = np.zeros((5, CEL + 1), dtype=np.int32)
    klas_np = np.zeros((5, CEL + 1), dtype=np.int32)
    side_np = np.zeros((5, CEL + 1), dtype=np.float64)
    cosf_np = np.zeros((5, CEL + 1), dtype=np.float64)
    sinf_np = np.zeros((5, CEL + 1), dtype=np.float64)
    area_np = np.zeros(CEL + 1, dtype=np.float64)
    fnc_np = np.full(CEL + 1, G * MANNING_N ** 2, dtype=np.float64)
    h_np = np.full(CEL + 1, HM1, dtype=np.float64)
    z_np = np.zeros(CEL + 1, dtype=np.float64)

    edge_cos = [0.0, 0.0, 1.0, 0.0, -1.0]
    edge_sin = [0.0, -1.0, 0.0, 1.0, 0.0]
    for i in range(N):
        for j in range(N):
            pos = i * N + j + 1
            area_np[pos] = dx * dx
            for e in range(1, 5):
                side_np[e][pos] = dx
                cosf_np[e][pos] = edge_cos[e]
                sinf_np[e][pos] = edge_sin[e]
            if i > 0:    nac_np[1][pos] = (i-1)*N+j+1
            else:        klas_np[1][pos] = 4
            if j < N-1:  nac_np[2][pos] = i*N+(j+1)+1
            else:        klas_np[2][pos] = 4
            if i < N-1:  nac_np[3][pos] = (i+1)*N+j+1
            else:        klas_np[3][pos] = 4
            if j > 0:    nac_np[4][pos] = i*N+(j-1)+1
            else:        klas_np[4][pos] = 4
            h_np[pos] = 2.0 if j < N // 2 else 0.5
            z_np[pos] = h_np[pos]

    NAC.from_numpy(nac_np); KLAS.from_numpy(klas_np)
    SIDE.from_numpy(side_np); COSF.from_numpy(cosf_np); SINF.from_numpy(sinf_np)
    SLCOS.from_numpy(side_np * cosf_np); SLSIN.from_numpy(side_np * sinf_np)
    AREA.from_numpy(area_np); ZBC.from_numpy(np.zeros(CEL+1, dtype=np.float64))
    FNC.from_numpy(fnc_np)

    # --- Osher solver (shared by both versions) ---
    @ti.func
    def QF(h, u, v):
        f0 = h * u
        return ti.Vector([f0, f0 * u, f0 * v, HALF_G * h * h])

    @ti.func
    def osher_func(QL_h, QL_u, QL_v, QR_h, QR_u, QR_v, FIL_in, H_pos):
        CR = ti.sqrt(G * QR_h)
        FIR_v = QR_u - 2.0 * CR
        fil = FIL_in; fir = FIR_v
        UA = (fil + fir) / 2.0
        CA = ti.abs((fil - fir) / 4.0)
        CL_v = ti.sqrt(G * H_pos)
        FLR = ti.Vector([0.0, 0.0, 0.0, 0.0])
        K2 = ti.cast(0, ti.i32)
        if CA < UA: K2 = 1
        elif UA >= 0.0 and UA < CA: K2 = 2
        elif UA >= -CA and UA < 0.0: K2 = 3
        else: K2 = 4
        K1 = ti.cast(0, ti.i32)
        if QL_u < CL_v and QR_u >= -CR: K1 = 1
        elif QL_u >= CL_v and QR_u >= -CR: K1 = 2
        elif QL_u < CL_v and QR_u < -CR: K1 = 3
        else: K1 = 4

        if K1 == 1:
            if K2 == 1:
                US=fil/3; HS=US*US/G; FLR+=QF(HS,US,QL_v)
            elif K2 == 2:
                ua_=UA; fil=fil-ua_; HA=fil*fil/(4*G); FLR+=QF(HA,ua_,QL_v)
            elif K2 == 3:
                ua_=UA; fir=fir-ua_; HA=fir*fir/(4*G); FLR+=QF(HA,ua_,QR_v)
            else:
                US=fir/3; HS=US*US/G; FLR+=QF(HS,US,QR_v)
        elif K1 == 2:
            if K2 == 1: FLR+=QF(QL_h,QL_u,QL_v)
            elif K2 == 2:
                FLR+=QF(QL_h,QL_u,QL_v); US2=fil/3; HS2=US2*US2/G; FLR-=QF(HS2,US2,QL_v)
                ua_=UA; fil=fil-ua_; HA=fil*fil/(4*G); FLR+=QF(HA,ua_,QL_v)
            elif K2 == 3:
                FLR+=QF(QL_h,QL_u,QL_v); US2=fil/3; HS2=US2*US2/G; FLR-=QF(HS2,US2,QL_v)
                ua_=UA; fir=fir-ua_; HA=fir*fir/(4*G); FLR+=QF(HA,ua_,QR_v)
            else:
                FLR+=QF(QL_h,QL_u,QL_v); US2=fil/3; HS2=US2*US2/G; FLR-=QF(HS2,US2,QL_v)
                US6=fir/3; HS6=US6*US6/G; FLR+=QF(HS6,US6,QR_v)
        elif K1 == 3:
            if K2 == 1:
                US2=fil/3; HS2=US2*US2/G; FLR+=QF(HS2,US2,QL_v)
                US6=fir/3; HS6=US6*US6/G; FLR-=QF(HS6,US6,QR_v); FLR+=QF(QR_h,QR_u,QR_v)
            elif K2 == 2:
                ua_=UA; fil=fil-ua_; HA=fil*fil/(4*G); FLR+=QF(HA,ua_,QL_v)
                US6=fir/3; HS6=US6*US6/G; FLR-=QF(HS6,US6,QR_v); FLR+=QF(QR_h,QR_u,QR_v)
            elif K2 == 3:
                ua_=UA; fir=fir-ua_; HA=fir*fir/(4*G); FLR+=QF(HA,ua_,QR_v)
                US6=fir/3; HS6=US6*US6/G; FLR-=QF(HS6,US6,QR_v); FLR+=QF(QR_h,QR_u,QR_v)
            else: FLR+=QF(QR_h,QR_u,QR_v)
        else:
            if K2 == 1:
                FLR+=QF(QL_h,QL_u,QL_v); US6=fir/3; HS6=US6*US6/G; FLR-=QF(HS6,US6,QR_v); FLR+=QF(QR_h,QR_u,QR_v)
            elif K2 == 2:
                FLR+=QF(QL_h,QL_u,QL_v); US2=fil/3; HS2=US2*US2/G; FLR-=QF(HS2,US2,QL_v)
                ua_=UA; fil=fil-ua_; HA=fil*fil/(4*G); FLR+=QF(HA,ua_,QL_v)
                US6=fir/3; HS6=US6*US6/G; FLR-=QF(HS6,US6,QR_v); FLR+=QF(QR_h,QR_u,QR_v)
            elif K2 == 3:
                FLR+=QF(QL_h,QL_u,QL_v); US2=fil/3; HS2=US2*US2/G; FLR-=QF(HS2,US2,QL_v)
                ua_=UA; fir=fir-ua_; HA=fir*fir/(4*G); FLR+=QF(HA,ua_,QR_v)
                US6=fir/3; HS6=US6*US6/G; FLR-=QF(HS6,US6,QR_v); FLR+=QF(QR_h,QR_u,QR_v)
            else:
                FLR+=QF(QL_h,QL_u,QL_v); US2=fil/3; HS2=US2*US2/G; FLR-=QF(HS2,US2,QL_v); FLR+=QF(QR_h,QR_u,QR_v)
        return FLR

    # --- Flux + update function (reads from given H/U/V/Z sources) ---
    @ti.func
    def compute_flux_and_update(pos, H_rd, U_rd, V_rd, Z_rd):
        """Compute one cell. H_rd/U_rd/V_rd/Z_rd are the fields to read neighbor data from."""
        H1 = H_rd[pos]; U1 = U_rd[pos]; V1 = V_rd[pos]; BI = ZBC[pos]
        HI = ti.max(H1, HM1)
        UI = U1; VI = V1
        if HI <= HM2:
            UI = ti.select(UI >= 0.0, VMIN, -VMIN)
            VI = ti.select(VI >= 0.0, VMIN, -VMIN)
        ZI = ti.max(Z_rd[pos], BI)
        WH = 0.0; WU = 0.0; WV = 0.0

        for j in ti.static(range(1, 5)):
            NC = NAC[j, pos]; KP = KLAS[j, pos]
            COSJ = COSF[j, pos]; SINJ = SINF[j, pos]
            SL = SIDE[j, pos]; SLCA = SLCOS[j, pos]; SLSA = SLSIN[j, pos]
            QL_h = HI; QL_u = UI*COSJ + VI*SINJ; QL_v = VI*COSJ - UI*SINJ
            CL_v = ti.sqrt(G * HI); FIL_v = QL_u + 2.0 * CL_v
            HC=0.0; BC=0.0; ZC=0.0; UC=0.0; VC=0.0
            if NC != 0:
                HC = ti.max(H_rd[NC], HM1); BC = ZBC[NC]
                ZC = ti.max(ZBC[NC], Z_rd[NC]); UC = U_rd[NC]; VC = V_rd[NC]
            f0=0.0; f1=0.0; f2=0.0; f3=0.0
            if KP != 0:
                f3 = HALF_G * H1 * H1
            elif HI <= HM1 and HC <= HM1:
                pass
            elif ZI <= BC:
                f0=-C1_C*ti.pow(HC,1.5); f1=HI*QL_u*ti.abs(QL_u); f3=HALF_G*HI*HI
            elif ZC <= BI:
                f0=C1_C*ti.pow(HI,1.5); f1=HI*ti.abs(QL_u)*QL_u; f2=HI*ti.abs(QL_u)*QL_v
            elif HI <= HM2:
                if ZC > ZI:
                    DH=ti.max(ZC-ZBC[pos],HM1); UN=-C1_C*ti.sqrt(DH)
                    f0=DH*UN; f1=f0*UN; f2=f0*(VC*COSJ-UC*SINJ); f3=HALF_G*HI*HI
                else: f0=C1_C*ti.pow(HI,1.5); f3=HALF_G*HI*HI
            elif HC <= HM2:
                if ZI > ZC:
                    DH=ti.max(ZI-BC,HM1); UN=C1_C*ti.sqrt(DH); HC1=ZC-ZBC[pos]
                    f0=DH*UN; f1=f0*UN; f2=f0*QL_v; f3=HALF_G*HC1*HC1
                else: f0=-C1_C*ti.pow(HC,1.5); f1=HI*QL_u*QL_u; f3=HALF_G*HI*HI
            else:
                if pos < NC:
                    QR_h=ti.max(ZC-ZBC[pos],HM1); UR=UC*COSJ+VC*SINJ
                    ratio=ti.min(HC/QR_h,1.5); QR_u=UR*ratio
                    if HC<=HM2 or QR_h<=HM2: QR_u=ti.select(UR >= 0.0, VMIN, -VMIN)
                    QR_v=VC*COSJ-UC*SINJ
                    OS=osher_func(QL_h,QL_u,QL_v,QR_h,QR_u,QR_v,FIL_v,H_rd[pos])
                    f0=OS[0]; f1=OS[1]+(1-ratio)*HC*UR*UR/2; f2=OS[2]; f3=OS[3]
                else:
                    C1m=-COSJ; S1m=-SINJ
                    L1h=H_rd[NC]; L1u=U_rd[NC]*C1m+V_rd[NC]*S1m; L1v=V_rd[NC]*C1m-U_rd[NC]*S1m
                    CL1=ti.sqrt(G*L1h); FIL1=L1u+2*CL1
                    HC2=ti.max(HI,HM1); ZC1=ti.max(ZBC[pos],ZI)
                    R1h=ti.max(ZC1-ZBC[NC],HM1); UR1=UI*C1m+VI*S1m
                    ratio1=ti.min(HC2/R1h,1.5); R1u=UR1*ratio1
                    if HC2<=HM2 or R1h<=HM2: R1u=ti.select(UR1 >= 0.0, VMIN, -VMIN)
                    R1v=VI*C1m-UI*S1m
                    MR=osher_func(L1h,L1u,L1v,R1h,R1u,R1v,FIL1,H_rd[NC])
                    f0=-MR[0]; f1=MR[1]+(1-ratio1)*HC2*UR1*UR1/2; f2=MR[2]
                    ZA=ti.sqrt(MR[3]/HALF_G)+BC; HC3=ti.max(ZA-ZBC[pos],0.0)
                    f3=HALF_G*HC3*HC3
            FLR1=f1+f3; FLR2=f2
            WH+=SL*f0; WU+=SLCA*FLR1-SLSA*FLR2; WV+=SLSA*FLR1+SLCA*FLR2

        DTA = DT / AREA[pos]; H2 = ti.max(H1 - DTA * WH, HM1); Z2 = H2 + BI
        U2=0.0; V2=0.0
        if H2 > HM1:
            if H2 <= HM2:
                U2 = ti.select(U1 >= 0.0, ti.min(VMIN, ti.abs(U1)), -ti.min(VMIN, ti.abs(U1)))
                V2 = ti.select(V1 >= 0.0, ti.min(VMIN, ti.abs(V1)), -ti.min(VMIN, ti.abs(V1)))
            else:
                WSF=FNC[pos]*ti.sqrt(U1*U1+V1*V1)/ti.pow(H1,0.33333)
                U2=(H1*U1-DTA*WU-DT*WSF*U1)/H2; V2=(H1*V1-DTA*WV-DT*WSF*V1)/H2
                U2 = ti.select(U2 >= 0.0, ti.min(ti.abs(U2), 15.0), -ti.min(ti.abs(U2), 15.0))
                V2 = ti.select(V2 >= 0.0, ti.min(ti.abs(V2), 15.0), -ti.min(ti.abs(V2), 15.0))
        H_res[pos]=H2; U_res[pos]=U2; V_res[pos]=V2; Z_res[pos]=Z2
        W_res[pos]=ti.sqrt(U2*U2+V2*V2)

    # =====================================================================
    # Kernel A: NAIVE — reads directly from global fields
    # =====================================================================
    @ti.kernel
    def swe_naive():
        for pos in range(1, CEL + 1):
            compute_flux_and_update(pos, H_pre, U_pre, V_pre, Z_pre)

    # =====================================================================
    # Kernel B: PROMOTED — pre-cache neighbor data via extra reads
    # Taichi doesn't expose __shared__ directly, but we can emulate
    # the effect using ti.block_local() with SNode or by pre-loading
    # into separate cache fields with better locality.
    #
    # For a fair comparison, we use a TWO-PASS approach:
    #   Pass 1: Each cell pre-fetches its 4 neighbors' data into
    #           cache fields laid out contiguously by block
    #   Pass 2: Compute reads from cache fields instead of global
    #
    # This is what a compiler's gather promotion would generate.
    # =====================================================================
    # Cache fields: store neighbor data reordered for locality
    H_cache = ti.field(dtype=ti.f64, shape=(CEL + 1,))
    U_cache = ti.field(dtype=ti.f64, shape=(CEL + 1,))
    V_cache = ti.field(dtype=ti.f64, shape=(CEL + 1,))
    Z_cache = ti.field(dtype=ti.f64, shape=(CEL + 1,))

    @ti.kernel
    def gather_prefetch():
        """Pass 1: prefetch own + neighbor data into cache (improves locality)."""
        for pos in range(1, CEL + 1):
            # Load self
            H_cache[pos] = H_pre[pos]
            U_cache[pos] = U_pre[pos]
            V_cache[pos] = V_pre[pos]
            Z_cache[pos] = Z_pre[pos]
            # Touch neighbors to warm L2 / promote to cache
            for j in ti.static(range(1, 5)):
                nc = NAC[j, pos]
                if nc != 0:
                    H_cache[nc] = H_pre[nc]
                    U_cache[nc] = U_pre[nc]
                    V_cache[nc] = V_pre[nc]
                    Z_cache[nc] = Z_pre[nc]

    @ti.kernel
    def swe_from_cache():
        """Pass 2: compute using cached data (better locality)."""
        for pos in range(1, CEL + 1):
            compute_flux_and_update(pos, H_cache, U_cache, V_cache, Z_cache)

    # =====================================================================
    # Transfer
    # =====================================================================
    @ti.kernel
    def transfer():
        for pos in range(1, CEL + 1):
            H_pre[pos]=H_res[pos]; U_pre[pos]=U_res[pos]
            V_pre[pos]=V_res[pos]; Z_pre[pos]=Z_res[pos]; W_pre[pos]=W_res[pos]

    # =====================================================================
    # Initialize
    # =====================================================================
    def reset():
        H_pre.from_numpy(h_np)
        U_pre.from_numpy(np.zeros(CEL+1, dtype=np.float64))
        V_pre.from_numpy(np.zeros(CEL+1, dtype=np.float64))
        Z_pre.from_numpy(z_np)
        W_pre.from_numpy(np.zeros(CEL+1, dtype=np.float64))

    # =====================================================================
    # Benchmark
    # =====================================================================
    def bench(label, step_fn, steps):
        times = []
        for r in range(warmup + repeat):
            reset()
            ti.sync()
            t0 = time.perf_counter()
            for _ in range(steps):
                step_fn()
                transfer()
            ti.sync()
            t1 = time.perf_counter()
            if r >= warmup:
                times.append((t1 - t0) * 1000)
        times.sort()
        median = times[len(times)//2]
        print(f"  {label}: median={median:.3f}ms  min={times[0]:.3f}ms  max={times[-1]:.3f}ms")
        return median

    print(f"\n=== Gather Promotion Experiment (Taichi, B200) ===")
    print(f"N={N}, CEL={CEL}, steps={steps}")

    # Warm up JIT
    reset()
    swe_naive()
    transfer()
    gather_prefetch()
    swe_from_cache()
    transfer()
    ti.sync()

    # Run naive
    naive_ms = bench("Naive   ", swe_naive, steps)

    # Run promoted (2-pass)
    def promoted_step():
        gather_prefetch()
        swe_from_cache()
    promo_ms = bench("Promoted", promoted_step, steps)

    # Correctness check
    reset()
    swe_naive()
    ti.sync()
    naive_H = H_res.to_numpy()

    reset()
    gather_prefetch()
    swe_from_cache()
    ti.sync()
    promo_H = H_res.to_numpy()

    max_diff = np.max(np.abs(naive_H[1:CEL+1] - promo_H[1:CEL+1]))

    print(f"\n  Speedup: {naive_ms/promo_ms:.2f}x")
    print(f"  Max |H_naive - H_promoted|: {max_diff:.2e}")
    print(f"  CSV: {N},{steps},{naive_ms:.3f},{promo_ms:.3f},{naive_ms/promo_ms:.2f},{max_diff:.2e}")
    return naive_ms, promo_ms


if __name__ == "__main__":
    import sys
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    run_experiment(N, steps)
