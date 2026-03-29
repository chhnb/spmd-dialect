# SPMD Compiler Robustness Hardening

## Goal Description

Advance the SPMD dialect compiler from a working prototype to a robust, trustworthy, and regression-safe system. The system already correctly lowers SPMD kernels to CPU/OpenMP/GPU backends and validates real GPU execution. This plan hardens that foundation along five dimensions: (1) written pass contracts documenting each pass's pre/post-conditions and failure behavior; (2) expanded negative and boundary lit tests for `PromoteGroupMemory` and `SPMDToGPU`; (3) automated cross-pipeline regression tests; (4) new compiler-side invariant verifier passes; (5) a systematic GPU sweep harness and differential correctness test comparing all three backends, backed by a three-tier regression command suite and a pipeline dump debug tool.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Pass contracts and limitations are documented for all seven key passes
  - Positive Tests (expected to PASS):
    - `docs/pass-contracts.md` exists and contains a section for each of the seven passes (`NormalizeSPMD`, `PlanSPMDSchedule`, `MaterializeTilingAndMapping`, `PromoteGroupMemory`, `SPMDToSCF`, `SPMDToOpenMP`, `SPMDToGPU`), with subsections for input requirements, output guarantees, and failure behavior
    - `docs/limitations.md` exists and enumerates at least eight unsupported or conservatively-skipped cases (non-structural CFG, pointer chasing, write-back promotion, barrier in divergent path, blockDim overflow, non-affine indexing, footprint overflow, reduction body with unknown side effects), mapping each to one of: verifier error / pass fail / remark+skip
  - Negative Tests (expected to FAIL):
    - A pass contract entry that omits the "failure behavior" subsection fails a doc lint check (or is visibly incomplete by inspection)
    - An unsupported case that is not listed in `limitations.md` but silently miscompiles is a regression that this document should prevent

- AC-2: `PromoteGroupMemory` has ≥5 new negative lit tests covering missing edge cases
  - Positive Tests (expected to PASS):
    - `test/SPMD/` contains at least 5 new `// NEGATIVE` or `// CHECK-NOT` lit test cases targeting `PromoteGroupMemory` specifically; running `lit test/SPMD/` passes all of them
    - The 5 cases include: (a) non-affine / indirect index in tile access → promotion skipped; (b) tile body that writes to the promoted memref → promotion skipped; (c) `memory_policy = no_promotion` attribute → promotion skipped and remark emitted; (d) access footprint exceeds group memory limit → promotion skipped and remark emitted; (e) invariant check: after successful promotion, no original `memref.alloc` for the promoted buffer remains in the IR
  - Negative Tests (expected to FAIL):
    - A kernel with a tile body that writes to the candidate memref must NOT produce a promoted IR with `.shared`; if it does, the test fails
    - A kernel with `memory_policy = no_promotion` must NOT produce a barrier or group buffer; if it does, the test fails
  - AC-2.1: Promotion invariant test verifies post-conditions of successful promotion
    - Positive: after `--promote-group-memory`, a `CHECK-NOT: memref.alloc` / `CHECK: gpu.workgroup` pair holds in the IR for the standard stencil case
    - Negative: IR that has both an old `memref.alloc` and a new `gpu.workgroup` attribution side-by-side fails the invariant check

- AC-3: `SPMDToGPU` has ≥5 new negative lit tests covering missing edge cases
  - Positive Tests (expected to PASS):
    - `test/SPMD/` contains at least 5 new negative cases targeting `SPMDToGPU`; running `lit test/SPMD/` passes all of them
    - The 5 cases include: (a) `spmd.barrier` nested inside `scf.if` (divergent control flow) → `SPMDToGPU` emits error and fails; (b) a leftover `gpu.workgroup` attribution after GPU lowering where no corresponding `workgroup` alloc was consumed → invariant test detects the stale attribution; (c) lane forall with 2D index range where flattening order is verified (row-major assertion holds); (d) group forall tile_size exceeding hardware blockDim limit (>1024 total threads) → `SPMDToGPU` emits error and fails; (e) group forall nested inside another group forall → `SPMDToGPU` emits error and fails
  - Negative Tests (expected to FAIL):
    - A kernel with `spmd.barrier` inside `scf.if` must cause `convert-spmd-to-gpu` to return a non-zero exit code; if it succeeds silently, the test fails
    - A lane forall with tile range [0,256)×[0,256) (total 65536 threads) must be rejected by `SPMDToGPU`

- AC-4: ≥5 cross-pipeline regression lit tests cover the full S0-to-target lowering sequences
  - Positive Tests (expected to PASS):
    - `test/SPMD/` contains at least 5 distinct multi-pass pipeline `RUN` sequences, each spanning at least 4 passes and targeting a different pipeline variant: (a) `normalize → plan → materialize → scf`; (b) `normalize → plan → materialize → openmp`; (c) `normalize → plan → materialize → promote → gpu`; (d) `normalize → plan → materialize → promote → gpu → nvvm → llvm → ptx` (ending with `llc -filetype=asm`); (e) `normalize → plan → materialize → gpu` (no-promotion path, verifying absence of `.shared` in PTX)
    - All 5 pipelines produce the expected `FileCheck` output and exit 0
  - Negative Tests (expected to FAIL):
    - Skipping `--normalize-spmd` before `--plan-spmd-schedule` in pipeline (a) causes a pass failure or wrong output that the `FileCheck` pattern detects
    - Running the no-promotion pipeline (e) on the stencil source and checking `NOSHARED-NOT: .shared` must fail if promotion is accidentally applied

- AC-5: `VerifySPMDPromotionInvariant` and `VerifySPMDGPUReady` passes are implemented and integrated
  - AC-5.1: `VerifySPMDPromotionInvariant` pass correctly checks post-promotion IR
    - Positive Tests (expected to PASS):
      - A correctly promoted stencil IR (group buffer present, barrier present, no original alloc) passes the verifier with exit code 0
      - The pass is registered and accessible via `spmd-opt --verify-spmd-promotion-invariant`
    - Negative Tests (expected to FAIL):
      - An IR that has both a `memref.alloc` and a `gpu.workgroup` attribution for the same buffer fails the verifier with an `emitError`
      - An IR where the cooperative copy barrier is absent after a group buffer allocation fails the verifier
      - An IR with a dangling use of the original alloc (after promotion replaced the alloc) fails the verifier
  - AC-5.2: `VerifySPMDGPUReady` pass correctly checks pre-GPU-lowering IR
    - Positive Tests (expected to PASS):
      - A correctly materialized kernel (no residual `spmd.*` ops other than the expected forall/yield structure) passes the verifier
      - The pass is registered and accessible via `spmd-opt --verify-spmd-gpu-ready`
    - Negative Tests (expected to FAIL):
      - An IR containing a residual `spmd.reduce` inside a `gpu.launch` body (which would be unhandled) fails the verifier
      - An IR where a `spmd.barrier` appears inside an `scf.if` block fails the verifier

- AC-6: `PromoteGroupMemory` and `SPMDToGPU` emit actionable diagnostic remarks
  - Positive Tests (expected to PASS):
    - Running `spmd-opt --promote-group-memory --mlir-print-op-stats` (or equivalent remark flag) on the standard promoted stencil emits a remark containing: promotion decision (promote/skip), reuse count, footprint in bytes, and memory policy
    - Running `spmd-opt --convert-spmd-to-gpu` on a kernel emits a remark or debug message containing: computed gridDim, computed blockDim, number of workgroup buffers, whether 2D lane flattening was applied
    - A `no_promotion` kernel emits a remark explaining why promotion was skipped (policy decision)
  - Negative Tests (expected to FAIL):
    - Passing a kernel that is skipped for promotion without any remark is a gap; the remark must be present and contain the reason

- AC-7: GPU sweep harness covers 3 kernels × ≥8 sizes × ≥3 tile configs and outputs a CSV
  - Positive Tests (expected to PASS):
    - `bash scripts/run-robustness-validation.sh` runs on a GPU machine and produces `results/robustness/latest.csv` containing at least 72 rows (3 kernels × 8 sizes × 3 configs)
    - For ewise: sizes ∈ {1, 32, 33, 64, 256, 257, 1024, 1048576}, tile configs ∈ {32, 64, 256}; all correctness entries show PASS
    - For promoted stencil: sizes ∈ {(32,8), (33,9), (64,64), (128,128), (256,256), (512,512), (1024,1024), (511,513)}, tile configs ∈ {(32,8), (16,16), (64,8)}; all interior correctness entries show PASS
    - For reduction: sizes ∈ {1, 255, 256, 257, 1024, 65536, 1048576, 16777216}, tile configs ∈ {64, 128, 256}; all rel_err < 1e-3 entries show PASS
    - The CSV columns include: kernel, N (or NxM), tile_config, promoted (yes/no), correctness (PASS/FAIL), rel_err, cpu_ms, gpu_ms, speedup
  - Negative Tests (expected to FAIL):
    - A kernel launched with a tile config that exceeds blockDim=1024 is caught before GPU launch and reported as SKIP or ERROR in the CSV, not as a silent wrong result
    - A stencil launched with non-multiple-of-tile N (e.g., N=33 with tile_row=32) for the promoted path is rejected with an assertion error, not silently producing incorrect output

- AC-8: Differential correctness harness automatically compares CPU serial, OpenMP, and GPU results for all three kernel types
  - Positive Tests (expected to PASS):
    - `bash scripts/run-differential.sh` runs all three backends for each kernel and prints a summary table with columns: kernel, size, tile_config, cpu_ok, omp_ok, gpu_ok, err_metric
    - For ewise at N=1024: cpu_ok=PASS, omp_ok=PASS, gpu_ok=PASS, err_metric=0
    - For promoted stencil at (128,128): all three backends agree on interior values within tolerance
    - For reduction at N=65536: CPU serial sum, OpenMP parallel sum, and GPU atomic sum all agree within rel_err < 1e-3
    - The script exits 0 only if all comparisons pass; otherwise it exits non-zero and marks failing rows
  - Negative Tests (expected to FAIL):
    - Intentionally computing `C[i] = A[i] * B[i]` (multiply instead of add) in the GPU ewise kernel causes `gpu_ok=FAIL` and the script exits non-zero
    - Running the script without a GPU (no CUDA driver) causes the GPU column to show `SKIP` or `ERROR` rather than a Python traceback

- AC-9: Three-tier regression command suite and pipeline dump script are operational
  - AC-9.1: Regression commands
    - Positive Tests (expected to PASS):
      - `bash scripts/check-quick.sh` runs verifier tests, basic legality lit tests, and CPU pipeline smoke tests; exits 0 on a clean build
      - `bash scripts/check-medium.sh` runs all of quick plus cross-pipeline regression tests and a small correctness sweep (ewise + stencil at ≤3 sizes); exits 0 on a clean build
      - `bash scripts/check-full.sh` runs all of medium plus full lit suite, all GPU sweep sizes/tiles, promotion ablation, and differential correctness; exits 0 on a clean build
    - Negative Tests (expected to FAIL):
      - Introducing a broken negative test (one that should fail but now passes) causes `check-quick.sh` to return non-zero
      - Removing a required FileCheck pattern from a cross-pipeline test causes `check-medium.sh` to return non-zero
  - AC-9.2: Pipeline dump script
    - Positive Tests (expected to PASS):
      - `bash scripts/dump-pipeline.sh test/SPMD/lower-to-gpu-nvptx-promoted.mlir` produces 6 intermediate IR dump files (after each of: normalize, materialize, promote, gpu-lowering, outlining, nvvm-lowering)
      - Each dump file is valid MLIR that can be re-parsed by `mlir-opt`
    - Negative Tests (expected to FAIL):
      - Requesting a non-existent stage name (e.g., `--stage foo`) exits non-zero with a clear error message rather than silently producing an empty file

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
The implementation includes: `docs/pass-contracts.md` and `docs/limitations.md` with full contract tables for all 7 passes; ≥5 new negative lit tests each for `PromoteGroupMemory` and `SPMDToGPU`; ≥5 cross-pipeline regression lit tests; both `VerifySPMDPromotionInvariant` and `VerifySPMDGPUReady` implemented as registered MLIR passes with lit tests; actionable remarks in both `PromoteGroupMemory` and `SPMDToGPU`; `scripts/run-robustness-validation.sh` producing a 72+ row CSV; `scripts/run-differential.sh` comparing all three backends; `scripts/check-quick.sh`, `check-medium.sh`, `check-full.sh`; and `scripts/dump-pipeline.sh` producing 6 staged IR dumps.

### Lower Bound (Minimum Acceptable Scope)
The implementation includes: written contracts for all 7 passes in `docs/pass-contracts.md`; `docs/limitations.md` with ≥8 unsupported cases; exactly 5 new negative lit tests for each of `PromoteGroupMemory` and `SPMDToGPU`; 5 cross-pipeline regression tests; both verifier passes registered and functional with lit tests; basic remarks in `PromoteGroupMemory` (promote/skip decision, footprint, policy) and `SPMDToGPU` (gridDim, blockDim); sweep harness producing ≥72 CSV rows on a GPU machine; differential harness comparing all three backends at ≥1 size per kernel; three regression scripts; and `dump-pipeline.sh` producing 6 staged dumps.

### Allowed Choices
- Can use: MLIR `emitRemark` / `emitError` / `signalPassFailure` for pass diagnostics; Python for sweep and differential harnesses; bash for orchestration scripts; `FileCheck` for all lit test assertions; existing `cuda_driver.py` for GPU harness; OpenMP compilation via LLVM's `libomp` for the OpenMP baseline
- Cannot use: external testing frameworks (pytest, gtest) for lit tests — all new tests must be MLIR lit tests using `RUN:` and `FileCheck`; external libraries beyond numpy in Python harnesses; write-back promotion, autotuning, ROCm/Vulkan backends, or new kernel types — these are explicitly out of scope for this plan

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

**Pass contracts**: Each entry in `docs/pass-contracts.md` follows a three-section template: (1) *Input requirements* — the IR invariants and attribute states that must hold before the pass runs; (2) *Output guarantees* — what the pass adds, removes, or preserves; (3) *Failure behavior* — which conditions trigger `emitError + signalPassFailure` versus `emitRemark + skip`.

**Negative lit tests**: Each new test file follows the existing pattern in `test/SPMD/lower-to-gpu-negative.mlir` — a function that triggers the bad condition followed by `// expected-error` or `// CHECK-NOT` annotations. The `--verify-diagnostics` mlir-opt flag enables expected-error matching.

**Verifier passes**: A lightweight MLIR function pass that walks the IR looking for invariant violations using `op.emitError(...)` followed by `signalPassFailure()`. No new dialect ops needed — the passes use `isa<>` / `dyn_cast<>` to query existing op types (e.g., checking that no `memref.alloc` remains after promotion by scanning `gpu.func` bodies for both `memref.alloc` ops and `gpu.workgroup` attributions).

**Diagnostic remarks**: Add `--mlir-print-ir-before/after` or use `pass.getAnalysis<DataFlowSolver>()` for statistics. Simpler: emit remarks using `op.emitRemark("PromoteGroupMemory: skipped — " + reason)` at the point of decision in the existing `PromoteGroupMemory.cpp`.

**Sweep harness**: The existing `harness/run_ewise.py`, `run_promoted_stencil.py`, and `run_reduction.py` each have `--sizes` / `--shapes` flags and `--perf`. Wrap them in `scripts/run-robustness-validation.sh` with nested loops over sizes and tile configs, capturing stdout into CSV rows. Tile config variation requires regenerating PTX per config (passing tile constants as MLIR constants or as `--nvvm-attach-target` SM changes).

**Differential harness**: For CPU serial, compile and run via the existing SCF → LLVM → `lli` or `llc → clang` path. For OpenMP, use the `--convert-spmd-to-openmp` path + `libomp`. For GPU, use the existing CUDA harness. A wrapper script calls all three and diffs the outputs.

**Three-tier regression scripts**: `check-quick.sh` runs `lit test/SPMD/ -j4 --filter "invalid|legality|attrs"`; `check-medium.sh` runs `lit test/SPMD/ -j4` plus `run-validation.sh --sizes 32,1024`; `check-full.sh` runs `lit test/SPMD/ -j8` plus `run-robustness-validation.sh` plus `run-differential.sh`.

### Relevant References
- `lib/Transforms/PromoteGroupMemory.cpp` — existing promotion logic with footprint and policy checks; promotion skip paths are the insertion points for new remarks
- `lib/Conversion/SPMDToGPU/SPMDToGPU.cpp` — GPU conversion; barrier placement and workgroup attribution rebinding are the insertion points for new diagnostics
- `test/SPMD/lower-to-gpu-negative.mlir` — existing negative test pattern (nested group forall, rank > 3, blockDim overflow) to follow for new SPMDToGPU tests
- `test/SPMD/promotion_oversized_remark.mlir` — existing PromoteGroupMemory remark test to follow for non-affine and write-conflict cases
- `harness/cuda_driver.py` — CUDA Driver API wrapper; `harness/run_ewise.py`, `run_promoted_stencil.py`, `run_reduction.py` — existing harnesses to extend for sweep
- `scripts/run-validation.sh` — existing one-click validation to use as template for `run-robustness-validation.sh`
- `include/spmd/Transforms/SPMDPasses.h` — pass registration header; new verifier passes register here

## Dependencies and Sequence

### Milestones

1. Documentation foundation: Pass contracts and limitations docs
   - Write `docs/pass-contracts.md` with all 7 pass contracts
   - Write `docs/limitations.md` with ≥8 unsupported case entries
   - No code changes required; unblocks AC-1 and provides reference for all subsequent work

2. Negative test expansion: PromoteGroupMemory and SPMDToGPU
   - Add ≥5 new negative lit tests for `PromoteGroupMemory` (AC-2)
   - Add ≥5 new negative lit tests for `SPMDToGPU` (AC-3)
   - Add ≥5 cross-pipeline regression tests (AC-4)
   - Depends on Milestone 1 (contracts define what "should fail" means)

3. Compiler-side verifier passes: VerifySPMDPromotionInvariant and VerifySPMDGPUReady
   - Implement both passes in `lib/Transforms/`
   - Register in `SPMDPasses.h` and `CMakeLists.txt`
   - Add lit tests for each verifier
   - Depends on Milestone 2 (the negative tests validate the same invariants the verifiers check)

4. Diagnostic remarks: PromoteGroupMemory and SPMDToGPU
   - Add `emitRemark` calls at all promotion decision points in `PromoteGroupMemory.cpp`
   - Add `emitRemark` calls for gridDim/blockDim/workgroup buffer count in `SPMDToGPU.cpp`
   - Add lit tests verifying remark output using `--verify-diagnostics`
   - Can proceed in parallel with Milestone 3; depends on Milestone 1 for what to report

5. GPU sweep and differential harness
   - Implement `scripts/run-robustness-validation.sh` with nested size/tile/promotion loops, CSV output
   - Implement `scripts/run-differential.sh` with CPU serial + OpenMP + GPU comparison
   - Requires GPU access (B200); depends on existing PTX generation pipeline
   - Can proceed in parallel with Milestones 3–4

6. Developer tooling: regression scripts and pipeline dump
   - Implement `scripts/check-quick.sh`, `check-medium.sh`, `check-full.sh`
   - Implement `scripts/dump-pipeline.sh`
   - Depends on Milestones 2–5 (scripts invoke the tests and harnesses built in those milestones)

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead
- Pass remark messages should be human-readable strings (e.g., `"group memory promotion skipped: non-affine access at ..."`) rather than codes or IDs
- New lit test files follow the naming convention of the existing test directory (lower-kebab-case, descriptive of the pass and condition being tested)

--- Original Design Draft Start ---

# SPMD Compiler Robustness Roadmap

## 目标

把当前 prototype 提升成一个"健壮、可信、可回归、可扩展"的编译系统。
核心三件事：
1. 把每个 pass 的输入/输出语义和失败方式钉死
2. 把测试从"流程跑通"升级成"覆盖边界、负例、跨后端一致性"
3. 把 GPU 路径做成稳定的、可 sweep、可回归的验证体系

---

## 工作流 A：Pass Contract 与 IR Invariant 固化

### A1. 给每个 pass 写 contract

新增 `docs/pass-contracts.md`，每个 pass 用统一模板记录：
- 输入要求（前置条件）
- 输出保证（后置条件）
- 失败行为（error / remark / skip）

优先写这 7 个 pass：
1. `NormalizeSPMD`
2. `PlanSPMDSchedule`
3. `MaterializeTilingAndMapping`
4. `PromoteGroupMemory`
5. `SPMDToSCF`
6. `SPMDToOpenMP`
7. `SPMDToGPU`

### A2. 明确 unsupported cases

新增 `docs/limitations.md`，列出不支持或保守跳过的情况：
- 非结构化 CFG
- pointer chasing / 间接访存
- write-back promotion
- barrier 处于 divergent path
- 超过 blockDim 硬件限制的 tile 配置
- 非 affine / 高度动态 index 的 aggressive promotion
- group memory footprint 超限
- reduction body 含未知副作用

### A3. 统一失败行为规则

| 情况 | 行为 |
|------|------|
| IR 结构不合法 | verifier error |
| pass 前置条件被破坏 | emitError + signalPassFailure |
| 优化不适用，但原程序合法 | emitRemark + skip |
| target 限制不满足（如 blockDim > 1024） | emitError + signalPassFailure |
| promotion 没收益 | remark + skip |

---

## 工作流 B：测试体系加固

### B1. 测试分层

5 层结构：
- 层 1：Verifier / legality tests
- 层 2：Pass-local tests（单个 pass 正向、负向、skip）
- 层 3：Cross-pipeline tests（多 pass 串联）
- 层 4：Differential correctness tests（CPU/OpenMP/GPU 输出一致性）
- 层 5：Stress / sweep tests（尺寸、tile、promotion 边界）

### B2. 需要补充的 lit tests

#### Legality / negative tests
- `spmd.forall` 非法 rank / tile / order
- `spmd.barrier` 放在错误层级
- `spmd.barrier` 嵌在 `scf.if` 里
- `#spmd.addr_space<group>` alloc 出现在不允许位置
- `spmd.reduce` body 含非法副作用
- GPU lowering 时 blockDim 超限
- promotion 后旧 alloc 未删干净（invariant test）

#### `PromoteGroupMemory` 专项 negative tests
- footprint 超限 → 不 promote
- reuse = 1 → 不 promote
- memory policy = `no_promotion` → 不 promote
- 非 affine access → 不 promote
- 有 write 混入 → 不 promote
- promoted 后验证：出现 group buffer、出现 barrier、load 被 rewrite、无残留旧 alloc

#### `SPMDToGPU` 专项 tests
- non-promoted ewise
- promoted stencil
- reduction with atomic path
- barrier 不在 launch body 顶层时 fail
- workgroup attribution 生成成功
- lane forall 2D → 1D flatten 顺序稳定

### B3. Cross-pipeline regression tests

- `normalize -> materialize -> scf`
- `normalize -> materialize -> openmp`
- `normalize -> materialize -> promote -> gpu`
- `normalize -> materialize -> promote -> gpu -> nvvm`
- `normalize -> materialize -> no-promotion -> gpu`

---

## 工作流 C：GPU 稳定性验证体系

### C1. 统一 sweep 脚本

新增 `scripts/run-robustness-validation.sh`，支持：
- 编译 CPU / OpenMP / GPU 三后端
- 不同 problem size sweep
- 不同 tile 配置 sweep
- promoted / non-promoted 对照
- 结果收集为 CSV

### C2. Problem size sweep

1D kernel（ewise / reduction）：1, 31, 32, 33, 63, 64, 65, 255, 256, 257, 1024, 1M
2D kernel（stencil）：7×7, 31×31, 32×8, 33×9, 64×64, 511×513, 512×512, 1024×1024

### C3. Tile / block 配置 sweep

1D：tile ∈ {32, 64, 128, 256}
2D：tile ∈ {16×16, 32×8, 8×32, 33×9}

### C4. Promoted / non-promoted 对照

对 stencil 两条路径：
- 对比 correctness、runtime、PTX 是否含 `.shared`、group memory footprint、barrier 是否出现、launch 参数

### C5. Reduction 稳定性回归

- launch 前 accumulator 清零是否稳定
- 多次 launch 无残留状态
- 输入全 0 / 全 1 / 随机 / 变化尺寸
- rel_err 稳定在 < 1e-3 范围
- 不同 tile 配置下仍然正确

### C6. Differential correctness 自动化

对每个 kernel 自动比较 CPU serial / OpenMP / GPU 结果。
输出格式：

| kernel | size | config | cpu_ok | omp_ok | gpu_ok | err_metric | remark |
|--------|------|--------|--------|--------|--------|------------|--------|

---

## 工作流 D：诊断和可调试性

### D1. Pass remarks

给以下 pass 加系统化 remark：
- `PlanSPMDSchedule`：选了什么 tile、什么 mapping、原因
- `PromoteGroupMemory`：promote/skip 原因、reuse count、footprint bytes、memory policy
- `SPMDToGPU`：gridDim/blockDim、workgroup buffers 数量、是否 flatten 2D lane forall

### D2. Pipeline dump 脚本

新增 `scripts/dump-pipeline.sh`，支持 dump 各阶段 IR：
- after normalize
- after materialize
- after promote
- after gpu lowering
- after outlining
- after nvvm lowering

### D3. 可选 verifier pass

#### `VerifySPMDPromotionInvariant`
- promoted path 中旧 alloc 已删
- barrier 位置正确
- workgroup buffer use 一致
- no dangling use

#### `VerifySPMDGPUReady`
- 无残留 spmd op
- workgroup buffer 合法
- blockDim 不超限
- barrier 不在 divergent 条件内

---

## 工作流 E：CI / 回归基线

### E1. 固定 robustness baseline

不轻易改：dialect surface、pass 顺序、当前 heuristic、supported kernel subset、promotion rules。
每次修改必须附带 contract 更新 + 测试更新 + regression 结果。

### E2. 三档回归命令

```bash
bash scripts/check-quick.sh    # verifier + lit smoke + CPU pipeline
bash scripts/check-medium.sh  # cross-pipeline + small sweep
bash scripts/check-full.sh    # all lit + CPU/OpenMP/GPU differential + promotion sweep
```

### E3. 结果归档

每次 full run 输出：
- `results/robustness/latest.csv`
- `results/robustness/latest.md`

---

## 量化目标

| 类别 | 量化指标 |
|------|----------|
| Pass contracts | 7 个关键 pass 全部有 written contract |
| Negative tests | PromoteGroupMemory 和 SPMDToGPU 各补 ≥ 5 个 negative lit test |
| Cross-pipeline tests | 至少 5 条全流水线回归测试 |
| GPU sweep | 3 个 kernel × ≥ 8 个 size × ≥ 3 个 tile config |
| Differential correctness | CPU/OpenMP/GPU 三后端自动比对，all PASS |
| Regression commands | quick / medium / full 三档均可复用 |
| Diagnostics | PromoteGroupMemory 和 SPMDToGPU 有可读 remark 输出 |

---

## 明确不做（避免目标漂移）

- write-back promotion
- autotuning
- ROCm / Vulkan backend
- 更多新 kernel
- 论文排版

--- Original Design Draft End ---
