# SPMD Dialect

A structured SPMD middle-end IR for simulation-oriented kernels, built on MLIR.

**Core idea:** Taichi / Warp / Numba kernels lower into a single structured IR where logical parallelism, reductions, and memory hierarchy remain explicit — enabling compiler-automatic group-memory promotion and hierarchical GPU reduction across all frontends.

---

## Requirements

| Requirement | Notes |
|-------------|-------|
| GLIBC ≥ 2.32 | **GPU compute node required.** Login nodes are typically too old. |
| CUDA-capable GPU | For GPU tests and demo |
| CMake ≥ 3.20 | |
| Clang ≥ 15 | Used for both LLVM and dialect build |
| Ninja | Build system |
| Python 3.12 | Via `uv` (installed automatically) |
| `uv` | Python package manager — installed by `build.sh` if missing |

---

## Quick Start (GPU compute node)

```bash
# 1. Clone
git clone <this-repo> spmd-dialect
cd spmd-dialect

# 2. Build everything (LLVM + dialect + Python venv)
#    First run: ~1-2 h (LLVM build). Subsequent runs: ~5 min.
bash scripts/build.sh

# 3. Run tests
bash scripts/check-quick.sh          # fast lit tests (~30 s)
bash scripts/check-medium.sh         # + robustness tests
bash scripts/check-full.sh           # full suite

# 4. Run GPU demos
bash scripts/demo.sh --sm sm_90      # auto-detects SM if omitted

# 5. Run Taichi → SPMD IR demo
bash scripts/demo_taichi_to_spmd.sh
```

---

## Build Details

### LLVM / MLIR

`build.sh` clones `llvmorg-20.1.7` (pinned in `LLVM_VERSION`) and builds with:

```
-DLLVM_ENABLE_PROJECTS=mlir
-DLLVM_TARGETS_TO_BUILD=X86;NVPTX
-DLLVM_ENABLE_ASSERTIONS=ON
```

To use an existing LLVM build:

```bash
LLVM_BUILD=/path/to/llvm/build bash scripts/build.sh --skip-llvm
```

### spmd-dialect

```bash
cmake -S . -B build \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DMLIR_DIR=<llvm-build>/lib/cmake/mlir \
    -DLLVM_DIR=<llvm-build>/lib/cmake/llvm \
    -DLLVM_EXTERNAL_LIT=<llvm-build>/bin/llvm-lit

cmake --build build
```

Binary: `build/bin/spmd-opt`

### Python venv

```bash
bash scripts/setup-venv.sh      # creates .venv/, installs numpy + taichi
source .venv/bin/activate       # optional for interactive use
```

The venv is at `.venv/` inside the repo.
`uv` cache is redirected to `/home/scratch.huanhuanc_gpu/.uv-cache` (scratch disk).

---

## Running Tests

```bash
# Lit tests only (no GPU required)
bash scripts/check-quick.sh

# + robustness / regression
bash scripts/check-medium.sh

# Full suite including GPU validation
bash scripts/check-full.sh

# Specific test
build/bin/spmd-opt test/SPMD/reduction-hierarchical-gpu.mlir \
    --normalize-spmd --convert-spmd-to-gpu | FileCheck test/SPMD/reduction-hierarchical-gpu.mlir
```

---

## GPU Demos

```bash
# Auto-detect SM
bash scripts/demo.sh

# Explicit SM
bash scripts/demo.sh --sm sm_90

# Individual demos
bash scripts/demo.sh --only elementwise
bash scripts/demo.sh --only stencil
bash scripts/demo.sh --only reduction
```

---

## Taichi → SPMD IR Demo

Shows the full pipeline:
**Taichi Python → Taichi CHI IR → S0 SPMD IR (atomic) → RecognizeStructuredReductions → spmd.reduce → ReduceToHierarchicalGPU → GPU IR**

```bash
bash scripts/demo_taichi_to_spmd.sh
```

On a login node (GLIBC < 2.32): prints annotated IR at each stage without executing.
On a GPU compute node: prints real Taichi CHI IR and real GPU IR from `spmd-opt`.

---

## Project Layout

```
spmd-dialect/
├── include/SPMD/          IR, attrs, passes (ODS / TableGen)
├── lib/
│   ├── IR/                SPMDDialect, SPMDOps, SPMDAttrs
│   ├── Transforms/        NormalizeSPMD, PlanSPMDSchedule,
│   │                      MaterializeTilingAndMapping,
│   │                      PromoteGroupMemory,
│   │                      ReduceToHierarchicalGPU, ...
│   ├── Conversion/        SPMDToSCF, SPMDToOpenMP, SPMDToGPU
│   └── Analysis/          AccessSummaryAnalysis, PromotionPlanAnalysis
├── test/SPMD/             34 lit tests (positive + negative + full pipeline)
├── scripts/               build.sh, demo.sh, check-*.sh, setup-venv.sh
├── docs/
│   ├── semantic-spec-v1.md   IR semantic specification
│   ├── research-plan-v1.md   Research plan (advisor discussion)
│   ├── design-v1.md          IR design document
│   ├── pass-contracts.md     Per-pass legality contracts
│   └── limitations.md        Known limitations and workarounds
├── LLVM_VERSION           Pinned LLVM tag (llvmorg-20.1.7)
└── README.md              This file
```

---

## Key Passes

| Pass | Input → Output | Purpose |
|------|---------------|---------|
| `NormalizeSPMD` | S0 → S0 | Canonicalize forall (0-based, unit-step) |
| `PlanSPMDSchedule` | S0 → S1 | Add tile/mapping schedule attrs |
| `MaterializeTilingAndMapping` | S1 → S2 | Expand tiling into nested foralls |
| `PromoteGroupMemory` | S2 → S2 | Synthesize shared memory + barrier |
| `ReduceToHierarchicalGPU` | S2 → S2 | Hierarchical GPU reduction lowering |
| `SPMDToSCF` | S2 → SCF | CPU serial lowering |
| `SPMDToOpenMP` | S2 → OMP | CPU parallel lowering |
| `SPMDToGPU` | S2 → GPU | GPU lowering (NVPTX) |

---

## Troubleshooting

**`spmd-opt: GLIBC_2.32 not found`**
You are on a login node. Switch to a GPU compute node.

**`taichi: GLIBC_2.32 not found`**
Same issue. The Taichi demo requires a compute node.

**`.venv/bin/python` broken after relogin**
The venv symlinks to a Python that may not exist on the current node.
Fix: `bash scripts/setup-venv.sh` (uses `--clear` to rebuild).

**LLVM build OOM**
Reduce jobs: `JOBS=4 bash scripts/build.sh --skip-llvm`
Or add `-DLLVM_PARALLEL_LINK_JOBS=2` to limit linker parallelism.
