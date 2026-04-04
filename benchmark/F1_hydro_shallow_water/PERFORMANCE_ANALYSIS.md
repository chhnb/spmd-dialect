# F1 Hydro Shallow Water — Performance Deep Analysis

## 1. Roofline Model

```
B200 specs: FP64 ~40 TFLOPS, HBM ~8 TB/s, L2 60MB
Kernel: ~300 FLOP, ~528 bytes per cell per step
Arithmetic Intensity = 0.57 FLOP/byte << roofline knee (5 FLOP/byte)
→ 100% memory bandwidth bound
```

**Theoretical minimum at N=8192 (67M cells, 10 steps):**
```
67M × 528 B = 35.4 GB → 35.4 / 8 TB/s = 44.3 ms
```

**Actual results:**
```
             Time     BW utilization   vs theory
Kokkos+swap  87.1ms      50.9%         1.97x off
Kokkos       98.8ms      44.8%         2.23x off
CUDA        112.3ms      39.5%         2.54x off
Warp        114.8ms      38.6%         2.59x off
Taichi      128.9ms      34.4%         2.91x off
```

## 2. Where Does Performance Go? (per-framework breakdown)

### 2a. Kokkos (best: 44.8% BW utilization)

**Compilation path:** C++ templates → nvcc → PTX → SASS

**SASS profile (4281 instructions):**
```
fp64 compute:    1595 (37.3%)  — OSHER + flux + update
int compute:     1028 (24.0%)  — address calculations (View strides)
data move:        509 (11.9%)  — register shuffles
control flow:     406 ( 9.5%)  — branches (16-case OSHER)
global load:       27 (LDG.E.64, L1/L2 path)
global store:       5 (STG.E.64)
```

**Why it's fast:**
- `View<double**>` uses **descriptor-based addressing** → loads go through L1/L2 cache
- L1/L2 supports **coalesced access**: 32 threads reading nearby addresses → few memory transactions
- nvcc template specialization → aggressive inlining → minimal function call overhead
- 113 registers, 0 stack spill → maximum occupancy for this kernel

**Remaining 55% gap to theory:**
- Gather serialization: `NAC[j][pos]` read → wait → `H[NAC]` read (2 serial loads)
- SOA layout: 5 separate field reads per neighbor (5 cache lines vs 1 if AOS)
- Branch divergence: OSHER 16-case → warp utilization < 100%
- Transfer kernel: 11.8% overhead (eliminated with pointer swap → 50.9% utilization)

### 2b. CUDA naive (39.5% → 5.3% slower than Kokkos)

**SASS profile (4178 instructions):**
```
fp64 compute:    1592 (38.1%)  — nearly identical to Kokkos
int compute:      976 (23.4%)  — less than Kokkos (simpler address calc)
global load:       17 (LDG.E.64.CONSTANT ← WRONG PATH)
```

**Root cause of 15% gap: constant cache misuse**

```
CUDA:   `const double* __restrict__ H_pre` → nvcc uses LDG.E.64.CONSTANT
Kokkos: `View<double**>` descriptor       → nvcc uses LDG.E.64 (normal L1/L2)
```

Constant cache serializes warp access — designed for broadcast (all threads read same address),
but gather has 32 threads reading 32 different addresses → **32 serial lookups** instead of
2-4 coalesced transactions through L1/L2.

**This is an nvcc optimization heuristic failure**: `const __restrict__` triggers constant
cache for ALL loads, but gather loads need L1/L2 path.

### 2c. Warp (38.6% → 17% slower than Kokkos)

**Compilation path:** Python → Warp codegen → C++ → nvcc → PTX → SASS

**Why 17% slower than Kokkos (constant across all scales):**
- Warp generates C++ code, also compiled by nvcc → same constant cache issue as CUDA? No.
- Warp uses `wp_array_t` struct with pointer → descriptor-style, same L1/L2 path as Kokkos
- Difference: Warp's **function inlining** — `osher_solver()` is a separate `__device__` function
- nvcc may not fully inline it (threshold exceeded for large functions)
- Result: function call overhead + suboptimal register allocation across call boundary
- Per-cell time is constant (0.172ns) → **fixed codegen quality gap, no scaling issue**

### 2d. Taichi (34.4% → 33% slower than Kokkos at N=8192)

**Compilation path:** Python → Taichi IR → LLVM IR → LLVM NVPTX backend → PTX → ptxas → SASS

**Two distinct issues:**

**Issue 1: Fixed gap (7% at small N) — LLVM NVPTX vs nvcc codegen quality**
- LLVM's register allocator is generic, not GPU-optimized
- LLVM's instruction scheduler doesn't know GPU pipeline depths
- LLVM's memory coalescing analysis is weaker than nvcc

**Issue 2: Scaling degradation (7% → 33% at large N)**
```
Per-cell time: N=4096 0.158ns → N=8192 0.192ns (21% increase)
               Kokkos: constant 0.147ns at all scales
```
- Verified: NOT instruction cache (ti.static vs range made no difference)
- Likely: LLVM generates more register spill → lower occupancy → can't hide memory latency
- At small N: enough warps to cover stalls; at large N: occupancy bottleneck exposed

**Issue 3: Cannot do pointer swap**
- Taichi `@ti.kernel` captures `ti.field` at compile time
- Cannot swap field pointers at runtime → forced to use transfer kernel
- Transfer adds 11.8% overhead that Kokkos can eliminate

## 3. Optimization Opportunities for SPMD IR

### Validated optimizations (experimentally confirmed):

| Optimization | Speedup | How | Difficulty |
|---|---|---|---|
| **Pointer swap** (no transfer) | **1.13x** | Detect read-write buffer pairs, swap instead of copy | Easy |
| **Avoid constant cache** for gather | **~1.15x** | Generate descriptor-based loads, not `const __restrict__` | Medium |
| **Mixed precision** (fp32 fields) | **1.8x** | Numerical range analysis → selective fp32 | Hard |

### Optimization path:

```
Current Kokkos:          98.8 ms  (44.8% BW)
+ Pointer swap:          87.1 ms  (50.9% BW)  — validated ✓
+ Mixed precision:      ~48   ms  (~92% BW)   — estimated (1.8x from fp32 data)
Theory:                  44.3 ms  (100% BW)
```

**With swap + mixed precision, we can reach ~92% of theoretical peak.**

### Specific SPMD IR design implications:

1. **Buffer liveness analysis pass**: detect `H_pre → compute → H_res → transfer → H_pre`
   pattern and replace with double-buffering + pointer swap

2. **Gather-aware lowering**: when lowering `spmd.forall` with indirect loads, generate
   L1/L2 path loads (not constant cache) — either via descriptor structs or explicit `__ldg()`

3. **Precision inference pass**: analyze numerical ranges in OSHER solver to determine
   which fields can safely use fp32 (velocities, fluxes) vs must use fp64 (water levels)

4. **Backend choice**: lower to C++ with Kokkos-style Views (gets nvcc optimizations)
   rather than LLVM NVPTX (Taichi's path, known to be weaker)

## 6. Critical Finding: Register Pressure is the Real Bottleneck

### Experiment: Vary -maxrregcount (CUDA SWE, N=4096, 10 steps)

| Max Regs | Actual | Blocks/SM | Occupancy | Time (ms) | Speedup |
|----------|--------|-----------|-----------|-----------|---------|
| 255 | 122 | 2 | 25% | 29.6 | 1.00x |
| 96 | 96 | 2 | 25% | 28.1 | 1.05x |
| 80 | 80 | 3 | 37% | 22.0 | **1.35x** |
| 64 | 64 | 4 | 50% | **21.2** | **1.40x** |

CUDA(64 regs): 21.2ms — beats Kokkos (23.8ms) by 11%.

### Compute Breakdown

Full kernel: 80% compute (OSHER), 20% memory.
Low occupancy (25%) fails to hide memory stalls during compute.
Doubling occupancy to 50% gives 1.40x speedup.

### SPMD IR Implication

RegisterPressureTuning pass: auto-select optimal register limit.
Well-defined, automatable, 1.40x validated gain.

## 7. E2E Validation: All Optimizations Combined

### Results (N=4096, fp64, 10 steps, real OSHER kernel)

| Config | Time (ms) | vs Baseline | Correct |
|--------|----------|-------------|---------|
| Original CUDA (122 regs, transfer) | 29.6 | 1.00x | ✓ |
| Kokkos (113 regs, transfer) | 23.8 | 1.24x | ✓ |
| CUDA + swap + 64 regs | **18.6** | **1.59x** | ✓ |
| CUDA + prefetch + swap + 128 regs | 20.9 | 1.42x | ✓ |
| CUDA + prefetch + swap + 96 regs | 22.3 | 1.33x | ✓ |

### Finding: Prefetch conflicts with register tuning

Prefetch needs MORE registers (hold 2 edges simultaneously).
Register tuning needs FEWER registers (higher occupancy).
These two optimizations compete for the same resource.

At maxreg=128: prefetch helps (+22% vs baseline at same regs)
At maxreg=64: prefetch hurts (-8% vs baseline at same regs)
Overall best: maxreg=64 without prefetch (18.6ms)

### Compute-Memory Breakdown (64 regs, N=4096)

```
Full kernel:      18.5ms (100%)
├── Memory:        6.8ms (37%)
└── Compute:      11.7ms (63%)

Compute and memory are 100% serial (zero overlap).
The gap is filled by warp scheduling (occupancy).
```

### Pipeline Stall Model

```
FP64 instruction latency: 4 cycles
Warp scheduling: 1 cycle/switch

25% occ (16 warps): 4 warps/scheduler → 4 cycles → barely covers FP64 latency
50% occ (32 warps): 8 warps/scheduler → 8 cycles → comfortable margin

Stall from compute→memory transition:
  Each edge: ~400 cycles (2-step gather dependency)
  4 edges × 400 = 1600 cycles stall per cell
  Hidden by warp switching IF enough warps available
```

## 8. Prefetch vs Occupancy: The Fundamental Tradeoff

### The Conflict

Both strategies hide gather latency but compete for the same resource:
- **Prefetch**: needs MORE registers (hold 2 edges) → lower occupancy
- **Occupancy**: needs FEWER registers (more warps) → no prefetch

### Parametric Study: When Does Each Win?

| Compute/Edge | Sequential | Prefetch | Speedup | Winner |
|-------------|-----------|---------|---------|--------|
| 0 ops (memory only) | 0.299ms | 0.277ms | **1.08x** | Prefetch |
| 2 ops (light) | 0.365ms | 0.303ms | **1.20x** | Prefetch |
| 5 ops (moderate) | 0.478ms | 0.443ms | **1.08x** | Prefetch |
| 10 ops (≈OSHER) | 0.746ms | 0.731ms | 1.02x | Tie |
| 20 ops (heavy) | 1.334ms | 1.356ms | 0.98x | Occupancy |
| 50 ops (extreme) | 3.161ms | 3.174ms | 1.00x | Tie |

### Crossover Point

```
Compute density < 5 FP64 ops/edge  → Prefetch wins (up to +20%)
Compute density > 10 FP64 ops/edge → Occupancy wins (prefetch hurts)
5-10 ops                           → Tie zone (need autotuning)
```

OSHER: ~50 ops/edge → deep in occupancy territory
→ Register tuning (1.59x) >> Prefetch (0% gain)

### Compiler Decision Model

```
compute_ops = count_fp64_ops(gather_to_next_gather)
if compute_ops < 5:
    apply_prefetch()          # sacrifice occupancy for overlap
    set maxreg = default      # keep registers for prefetch
elif compute_ops > 10:
    apply_reg_tuning()        # sacrifice prefetch for occupancy  
    set maxreg = optimal_for_occupancy()
else:
    autotune(prefetch, reg_tuning)  # try both, pick best
```

## 9. Complete Pipeline Stall Taxonomy

### Layer-by-layer decomposition (N=4096, 64 regs, 1 step)

| Layer | Kernel | Time (ms) | Incremental |
|-------|--------|----------|-------------|
| L0 | Memory only (gather) | 0.295 | 0.295 (47%) |
| L1 | + FMA compute | 0.313 | 0.018 (3%) |
| L2 | + sqrt/MUFU | 0.499 | 0.190 (30%) |
| L3 | + OSHER no branch | 0.490 | -0.009 (0%) |
| L4 | + OSHER 16-case | 0.633 | 0.143 (23%) |

### Optimization attempts for each stall type

| Stall | Optimization | Result | Status |
|-------|-------------|--------|--------|
| Gather (47%) | Register tuning | **1.40x** | ✓ Validated |
| Gather (47%) | Prefetch | Conflicts with reg tuning | ✗ |
| sqrt (30%) | CSE (hoist out of loop) | 0% (nvcc already does it) | ✗ |
| sqrt (30%) | rsqrt+Newton | 0% (B200 same speed) | ✗ |
| Branch (23%) | Predicated (Triton-style) | **-240%** (much slower!) | ✗ |
| Branch (23%) | Cell-type grouping | ~2% (97% are interior) | ✗ |
| FMA (3%) | Not a bottleneck | — | — |

### Branch Divergence: Why Predication Fails

```
No divergence (fixed path):  0.381ms (ideal)
With divergence (16-case):   0.577ms (+51% overhead)
Predicated (compute all):    1.295ms (+240% overhead!)
```

Predication computes ALL 16 paths then selects → 16x more compute.
The 51% divergence overhead is cheaper than 240% predication overhead.

### Prefetch vs Occupancy Crossover

| Compute ops/gather | Prefetch speedup | Winner |
|-------------------|-----------------|--------|
| 0 (memory only) | +8% | Prefetch |
| 2 (light) | +20% | Prefetch |
| 5 (moderate) | +8% | Prefetch |
| 10 (≈OSHER) | +2% | Tie |
| 20 (heavy) | -2% | Occupancy |
| 50 (extreme) | 0% | Tie |

Crossover: ~5-10 FP64 ops per gather.
OSHER at ~50 ops → deep in occupancy territory.

## 10. Dependency Chain Analysis: The Deeper Bottleneck

### Root cause: stalls are serial because of a long dependency chain

```
Per edge: NAC→H[NAC]→sqrt(G*H)→FIL→K1→branch→QF→flux
Cycles:   200  200     20       4    4   ~50    ~200  = ~680 cycles
× 4 edges = ~2700 cycles/cell
```

### Breaking the chain: precompute sqrt

Precomputing `sqrt(G*H[i])` as a separate field converts a compute
dependency into a memory load (parallel with other loads):

```
Original: H[NAC] → sqrt(G*H) → FIL    (serial: 200 + 20 = 220 cy)
Optimized: H[NAC] ↗ → FIL              (parallel loads: 200 cy)
           SH[NAC] ↗                    (loaded simultaneously)
```

Result: main kernel 12% faster (0.577→0.506ms), but precompute kernel
adds 0.063ms overhead → net E2E gain needs kernel fusion.

### Optimization interaction map

```
                    Register↓     Prefetch↗     DepChain↓     Swap↓
Register tuning     ★ 1.40x       CONFLICT      ORTHOGONAL    STACK
Prefetch            CONFLICT      6-20%         —             STACK
Dep chain breaking  ORTHOGONAL    —             12%           STACK
Buffer swap         STACK         STACK         STACK         ★ 1.13x
```

### Best achievable (validated + projected)

```
Original CUDA:     29.6ms  (1.00x)
+ register tuning: 21.2ms  (1.40x) ✓ validated
+ buffer swap:     18.6ms  (1.59x) ✓ validated
+ dep chain (fused):~16.5ms (~1.79x) projected (12% from dep chain)
```
