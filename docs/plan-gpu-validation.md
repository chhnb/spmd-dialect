# GPU Real-Execution Validation Harness

## Goal Description

Advance SPMD lowering from "can generate PTX" to "runs correctly on real GPU hardware and meets measurable performance targets." This requires building a complete validation harness covering three kernel types (elementwise, promoted stencil with shared memory, and reduction), verifying numerical correctness against CPU reference implementations, and demonstrating that GPU execution achieves at least 10× wall-clock speedup over NumPy for memory-bandwidth-bound kernels at large problem sizes. The environment is a B200 GPU (sm_100) but the harness must auto-detect the GPU SM level and support any sm_80+ architecture.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: All three kernels load and execute on a real CUDA GPU without runtime errors
  - Positive Tests (expected to PASS):
    - `cuda_driver.py` initializes the CUDA context via `cuInit + cuCtxCreate` and loads a PTX file via `cuModuleLoadData` without returning a non-zero CUresult
    - `run_ewise.py --ptx ewise.ptx` exits with code 0 on a machine with a supported GPU
    - `run_promoted_stencil.py --ptx stencil.ptx` exits with code 0 on a machine with a supported GPU
    - `run_reduction.py --ptx reduction.ptx` exits with code 0 on a machine with a supported GPU
  - Negative Tests (expected to FAIL):
    - Passing a nonexistent PTX file path causes `load_ptx` to raise a `RuntimeError` with the CUresult error name
    - Passing a PTX with `.target sm_999` causes `cuModuleLoadData` to return a non-zero CUresult and the harness reports the error message rather than silently producing wrong output

- AC-2: Elementwise kernel (`C[i] = A[i] + B[i]`) produces numerically correct results across a range of problem sizes
  - Positive Tests (expected to PASS):
    - For N ∈ {32, 100, 1024, 10000, 1000000}, `run_ewise.py` reports `max_err = 0.00e+00` and `PASS` for each size
    - Running with `--sm sm_80` generates PTX targeting sm_80 that still executes correctly on any sm_80+ GPU (forward compatibility)
  - Negative Tests (expected to FAIL):
    - A harness that passes wrong ABI arguments (e.g., passes tile_size=0 or swaps A and B descriptors) produces FAIL or a CUDA error, not silent wrong results
    - N not a multiple of 32 with a kernel that has no boundary guard causes at least one index to be out of range (the harness asserts before launch)

- AC-3: Promoted stencil kernel (`B[i,j] = A[i,j] + A[i,j+1] + A[i+1,j]`) produces numerically correct results on the interior of the output array
  - Positive Tests (expected to PASS):
    - For shapes ∈ {(64,64), (128,128), (512,512), (1024,1024)}, `run_promoted_stencil.py` reports `max_err = 0.00e+00` and `PASS` on the interior slice `[:N-1, :M-1]`
    - Static shared memory (1188 B declared in PTX) is used correctly: `cuLaunchKernel` with `shared_bytes=0` does not cause an error
  - Negative Tests (expected to FAIL):
    - An N not divisible by TILE_ROW=32 or M not divisible by TILE_COL=8 is rejected by the harness with an assertion error before GPU launch
    - Setting `block=(256, 1, 1)` instead of `(297, 1, 1)` causes incorrect results or a CUDA error (wrong cooperative-copy thread count)

- AC-4: Reduction kernel (`sum(A)`) is implemented as a GPU-compatible SPMD MLIR kernel and produces correct scalar output
  - AC-4.1: A new MLIR source file uses only `spmd.forall` with group-level tiling (so the `promoted` PTX pipeline can lower it to a GPU kernel)
    - Positive Tests (expected to PASS):
      - `gen-ptx.sh <reduction.mlir> promoted /tmp/reduction.ptx` succeeds without errors and writes a PTX file containing a `.entry` function
      - The PTX `.entry` function name is used correctly by `run_reduction.py` to call `get_function`
    - Negative Tests (expected to FAIL):
      - Using the existing `sum.mlir` (which uses only lane-level forall with no group dimension) with the `promoted` pipeline fails at PTX generation or produces a kernel with no block-level parallelism
  - AC-4.2: The reduction harness verifies numerical correctness against `numpy.sum`
    - Positive Tests (expected to PASS):
      - For N ∈ {1024, 65536, 1048576}, `run_reduction.py` reports `max_err < 1e-3` (f32 accumulation tolerance) and `PASS`
    - Negative Tests (expected to FAIL):
      - A GPU reduction that omits the inter-warp accumulation step reports a result that differs from `numpy.sum` by more than tolerance, causing `FAIL`

- AC-5: Elementwise kernel achieves ≥ 10× wall-clock speedup over NumPy for N ≥ 1,000,000
  - Positive Tests (expected to PASS):
    - `run_ewise.py --ptx ewise.ptx --perf` reports GPU time such that `cpu_ms / gpu_ms ≥ 10.0` for N = 1,000,000
    - The performance table includes CPU time, GPU time, and speedup ratio columns
  - Negative Tests (expected to FAIL):
    - N = 32 (trivially small) is not expected to show ≥ 10× speedup; the harness does not report this size as a performance failure

- AC-6: Promoted stencil kernel achieves ≥ 10× wall-clock speedup over NumPy for shape ≥ (512, 512)
  - Positive Tests (expected to PASS):
    - `run_promoted_stencil.py --ptx stencil.ptx --perf` reports `cpu_ms / gpu_ms ≥ 10.0` for shape (512, 512) or larger
  - Negative Tests (expected to FAIL):
    - Shape (64, 64) is not expected to show ≥ 10× speedup due to launch overhead; absence of this guarantee for small shapes is correct behavior

- AC-7: The one-click validation script auto-detects GPU SM level and runs all correctness tests end-to-end
  - Positive Tests (expected to PASS):
    - `bash scripts/run-validation.sh` on a B200 machine auto-detects sm_100, generates PTX for both pipelines, and reports all correctness tests passing without manual intervention
    - `bash scripts/run-validation.sh --sm sm_80` overrides to sm_80 and generates Ampere-compatible PTX that executes correctly on sm_80+ hardware
    - `bash scripts/run-validation.sh --perf` additionally prints performance tables for all kernels
  - Negative Tests (expected to FAIL):
    - Running on a machine with no NVIDIA GPU causes `detect-gpu.py` to fall back to sm_80 and the script to continue rather than silently producing garbage; attempting to load PTX on a machine with no GPU causes a clear CUDA error, not a Python traceback

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
The implementation includes all five scripts (`detect-gpu.py`, `gen-ptx.sh`, `setup-venv.sh`, `run-validation.sh`, `cuda_driver.py`) plus three harness files (`run_ewise.py`, `run_promoted_stencil.py`, `run_reduction.py`) and one new MLIR source for the reduction kernel. The Python harness uses `ctypes` to call the CUDA Driver API directly. All correctness and performance ACs pass. Documentation in `docs/gpu-validation.md` reflects verified actual output.

### Lower Bound (Minimum Acceptable Scope)
The implementation includes `cuda_driver.py`, harness files for all three kernels with correctness testing, a GPU-compatible reduction MLIR source, PTX generation scripts for both pipelines, and a one-click `run-validation.sh`. Performance measurement (`--perf`) is included but the ≥ 10× speedup AC must be demonstrated for at least ewise at N=1M and stencil at 512×512.

### Allowed Choices
- Can use: Python `ctypes` with `libcuda.so` (CUDA Driver API); `uv venv` with NumPy as the only Python dependency; `nvidia-smi` for SM detection with `cuDeviceGetAttribute` as fallback; `bash` for orchestration scripts; PTX forward compatibility (sm_80 PTX running on sm_100 via JIT re-compilation)
- Cannot use: `pycuda`, `cupy`, or any third-party CUDA Python binding; `ptxas` (Driver API JIT is sufficient); CUDA Toolkit (only Driver is required); hardcoded SM level in scripts without an override flag; CI/CD integration (out of scope per project constraints)

> **Note on Deterministic Designs**: The ABI mapping for each kernel (parameter order, descriptor field layout, grid/block dimensions) is fixed by the LLVM IR output of `mlir-translate`. These are not choices — they must match exactly. The "Allowed Choices" section above applies only to tooling and orchestration decisions.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

**Reduction kernel design**: The existing `sum.mlir` uses only a lane-level `spmd.forall` (no group dimension), which cannot be lowered to a block-parallel GPU kernel via the `promoted` pipeline. A new reduction MLIR file should use a two-level structure: an outer group-level `spmd.forall` over tiles and an inner lane-level `spmd.forall` over elements within the tile, plus an `spmd.reduce` to accumulate within the group. After lowering with `--promote-group-memory --convert-spmd-to-gpu`, the result should be a PTX kernel with block-level reduction using shared memory.

**ABI derivation workflow**: For any new kernel, derive the ABI from the LLVM IR output of `mlir-translate --mlir-to-llvmir`, not from the PTX text. The IR makes parameter roles unambiguous: `getelementptr [2 x i64], ptr %X, i32 0, i64 %Y` reveals that `%Y` is a dimension index into a sizes array, while `udiv i64 %tid, %Z` reveals that `%Z` is a tiling constant for linearization. Only after verifying the IR-derived ABI should the harness be written.

**Performance measurement**: Wall-clock timing in Python (via `time.perf_counter`) includes kernel launch overhead. For N=1M, GPU compute time dominates and the speedup will exceed 10×. For correctness timing, include `cd.synchronize()` after the timed GPU loop to ensure the GPU has finished.

```
Reduction MLIR sketch (pseudocode):
  func @reduction_kernel(%A: memref<?xf32>, %out: memref<f32>) {
    spmd.forall group(%bid : N / TILE) {
      // allocate shared scratch
      spmd.forall lane(%tid : TILE) {
        // load A[bid*TILE + tid] into shared
      }
      spmd.barrier
      // tree reduction in shared memory
      spmd.reduce add %local_sum into %out
    }
  }
```

### Relevant References
- `harness/cuda_driver.py` — ctypes CUDA Driver API wrapper; `launch()` signature and `DevicePtr` type
- `harness/run_ewise.py` — reference for 1D memref descriptor layout and correctness test structure
- `harness/run_promoted_stencil.py` — reference for 2D memref descriptor layout, static shared memory, cooperative-copy constants
- `test/SPMD/lower-to-gpu-nvptx.mlir` — source for ewise kernel (non-promoted pipeline)
- `test/SPMD/lower-to-gpu-nvptx-promoted.mlir` — source for stencil kernel (promoted pipeline)
- `scripts/gen-ptx.sh` — PTX generation for both pipelines with SM override support
- `scripts/detect-gpu.py` — GPU SM level detection via nvidia-smi and CUDA Driver API fallback

## Dependencies and Sequence

### Milestones
1. Infrastructure: CUDA Driver wrapper and PTX generation pipeline are working
   - Verify `cuda_driver.py` can initialize, allocate, and launch a kernel
   - Verify `gen-ptx.sh` auto-detects SM and produces valid PTX for both pipelines
   - Verify `.venv` with NumPy can be created via `setup-venv.sh`

2. Existing kernel correctness: Elementwise and promoted stencil harnesses pass all correctness tests
   - Verify `run_ewise.py` reports PASS for all sizes including N=1M
   - Verify `run_promoted_stencil.py` reports PASS for all shapes up to (1024, 1024)
   - Both depend on Milestone 1

3. Reduction kernel: New MLIR source + PTX + harness pass correctness
   - Design and write the GPU-compatible reduction MLIR source
   - Derive the new kernel's ABI from `mlir-translate --mlir-to-llvmir` output
   - Write `run_reduction.py` with correctness comparison against `numpy.sum`
   - Depends on Milestone 1; independent of Milestone 2

4. Performance validation: Correctness-passing kernels demonstrate ≥ 10× speedup
   - Run `run_ewise.py --perf` and confirm speedup ≥ 10× at N=1M
   - Run `run_promoted_stencil.py --perf` and confirm speedup ≥ 10× at (512, 512)
   - Depends on Milestones 2 (for ewise and stencil performance)

5. One-click orchestration: `run-validation.sh` runs all kernels end-to-end
   - Integrate all three kernel harnesses into `run-validation.sh`
   - Verify auto-detection, SM override, and `--perf` flag all work
   - Depends on Milestones 2, 3, 4

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

--- Original Design Draft Start ---

# GPU 真实执行验证

## 目标

把 SPMD lowering 从"能生成 PTX"推进到"能在真实 GPU 上执行，且结果正确"。
本文档覆盖：架构检测、PTX 生成、执行 harness 设计、兼容性矩阵、运行方法。

---

## 环境依赖

| 依赖 | 必须 | 说明 |
|------|------|------|
| CUDA Driver (`libcuda.so`) | ✓ | 加载 PTX 并 JIT 编译；无需 CUDA Toolkit |
| Python 3 + numpy | ✓ | harness 和 CPU reference |
| spmd-opt / mlir-opt / mlir-translate / llc | ✓ | PTX 生成（已有） |
| NVIDIA GPU (任何 sm_20+) | ✓ | 实际执行 |
| ptxas | ✗ | 不需要；Driver API 内部做 JIT |
| pycuda / cupy | ✗ | 不需要；用 ctypes 直接调 Driver API |

验证当前环境（本机 B200）：

```bash
python3 scripts/detect-gpu.py   # 预期输出: sm_100
nvidia-smi                      # 应显示 B200
python3 -c "import ctypes; ctypes.CDLL('libcuda.so'); print('OK')"
```

---

## 架构兼容性

### GPU → SM 级别映射

| GPU 系列 | compute_cap | SM 级别 | LLVM 支持 |
|---------|------------|---------|-----------|
| Blackwell B200 / B100 | 10.0 | sm_100 | ✓ (本 build) |
| Hopper H100 / H800 | 9.0 | sm_90 | ✓ |
| Ada Lovelace RTX 4090 | 8.9 | sm_89 | ✓ |
| Ampere A100 | 8.0 | sm_80 | ✓ |
| Ampere RTX 3090 | 8.6 | sm_86 | ✓ |
| Turing T4 | 7.5 | sm_75 | ✓ |
| Volta V100 | 7.0 | sm_70 | ✓ |

本 LLVM build 支持范围：sm_20 … sm_121（覆盖所有现役 GPU）。

### 自动检测

```bash
python3 scripts/detect-gpu.py   # 查询第一块 GPU 的 SM 级别
```

PTX 的**前向兼容性**：用 `sm_80` 生成的 PTX 可以在 B200 (sm_100) 上运行
（Driver 会 JIT 重编译），但不会利用 Blackwell 特有指令。
如果要针对 B200 优化，使用 `sm_100`（默认行为）。

### 多 GPU 场景

```bash
# 强制使用 sm_80（兼容所有 Ampere+ 卡）
bash scripts/run-validation.sh --sm sm_80

# 强制 sm_100（仅限 Blackwell）
bash scripts/run-validation.sh --sm sm_100

# 自动检测（推荐）
bash scripts/run-validation.sh
```

---

## PTX 生成流水线

### 非 promoted 路径（elementwise）

```
SPMD kernel (.mlir)
  ─ --normalize-spmd
  ─ --plan-spmd-schedule
  ─ --materialize-spmd-tiling
  ─ --convert-spmd-to-gpu
  ─ --gpu-kernel-outlining --nvvm-attach-target=chip=<SM>
  │
  ─ mlir-opt --convert-gpu-to-nvvm ...
  ─ spmd-opt --spmd-extract-gpu-module
  ─ mlir-translate --mlir-to-llvmir
  ─ llc --march=nvptx64 --mcpu=<SM> -filetype=asm
  │
  → ewise_kernel.ptx (.version 8.6, .target sm_100)
```

### Promoted 路径（stencil with shared memory）

```
SPMD kernel (.mlir)
  ─ --promote-group-memory           ← 插入 group alloc + barrier
  ─ --convert-spmd-to-gpu            ← group alloc → workgroup attribution
  ─ --gpu-kernel-outlining --nvvm-attach-target=chip=<SM>
  │
  ─ (同上 NVVM → PTX 链路)
  │
  → promoted_stencil_kernel.ptx (.shared 1188 bytes)
```

### 脚本用法

```bash
# 自动检测 SM，生成 PTX
bash scripts/gen-ptx.sh test/SPMD/lower-to-gpu-nvptx.mlir ewise /tmp/ewise.ptx

# 指定 SM
bash scripts/gen-ptx.sh test/SPMD/lower-to-gpu-nvptx.mlir ewise /tmp/ewise_sm80.ptx sm_80
bash scripts/gen-ptx.sh test/SPMD/lower-to-gpu-nvptx-promoted.mlir promoted /tmp/stencil.ptx sm_100
```

---

## Kernel ABI（从 LLVM IR 精确反推，已验证）

ABI 从 LLVM IR（`mlir-translate --mlir-to-llvmir` 输出）分析，比直接读 PTX 更可靠。

### `ewise_kernel`（17 个参数）

`memref<?xf32>` 展开为 5-field descriptor: (base_ptr, aligned_ptr, offset, size, stride)

```
param_0   : i64   tile_size = 32  (blockDim limit)
param_1   : i64   N               (array length)
param_2   : u64   A.base_ptr
param_3   : u64   A.aligned_ptr   ← 实际数据指针
param_4   : i64   A.offset = 0
param_5   : i64   A.size = N
param_6   : i64   A.stride = 1
param_7   : u64   B.base_ptr
param_8   : u64   B.aligned_ptr
param_9   : i64   B.offset = 0
param_10  : i64   B.size = N
param_11  : i64   B.stride = 1
param_12  : u64   C.base_ptr
param_13  : u64   C.aligned_ptr
param_14  : i64   C.offset = 0
param_15  : i64   C.size = N
param_16  : i64   C.stride = 1
```

Launch: `grid=(⌈N/32⌉, 1, 1)`, `block=(32, 1, 1)`

### `promoted_stencil_kernel`（21 个参数）

`memref<?x?xf32>` 展开为 7-field descriptor: (base, aligned, off, size0, size1, stride0, stride1)

```
param_0   : i64   TILE_ROW = 32        (blockIdx.x step)
param_1   : i64   TILE_COL = 8         (blockIdx.y step; compute 线程 linearisation 除数)
param_2   : i64   TILE_COL+1 = 9       (cooperative copy linearisation 内层列数)
param_3   : i64   (TILE_ROW+1)*(TILE_COL+1) = 297   (cooperative copy 线程上界)
param_4   : u64   A.base_ptr
param_5   : u64   A.aligned_ptr        ← 实际数据指针
param_6   : i64   A.offset = 0
param_7   : i64   A.size[0] = N
param_8   : i64   A.size[1] = M
param_9   : i64   A.stride[0] = M      (row-major)
param_10  : i64   A.stride[1] = 1
param_11  : i64   0                    (常量: dim-index for row boundary check)
param_12  : i64   1                    (常量: dim-index for col boundary check; 也是 stencil offset)
param_13  : i64   TILE_ROW*TILE_COL = 256   (compute phase 线程上界)
param_14  : u64   B.base_ptr
param_15  : u64   B.aligned_ptr
param_16  : i64   B.offset = 0
param_17  : i64   B.size[0] = N
param_18  : i64   B.size[1] = M
param_19  : i64   B.stride[0] = M
param_20  : i64   B.stride[1] = 1
```

Launch: `grid=(⌈N/32⌉, ⌈M/8⌉, 1)`, `block=(297, 1, 1)`
Shared memory: 1188 B 静态声明于 PTX，`cuLaunchKernel` 传 `shared_bytes=0`。

**重要约束**：N 必须是 32 的倍数，M 必须是 8 的倍数。
kernel 对 B 的写入无边界检查，非对齐尺寸会导致越界写。

---

## Harness 设计

### cuda_driver.py

用 Python `ctypes` 直接调用 CUDA Driver API (`libcuda.so`)，无第三方依赖。
Driver API 的 `cuModuleLoadData` 接收 PTX 字节串，内部 JIT 编译为 CUBIN。

核心 API：
```python
cd.init()                        # cuInit + cuCtxCreate
mod = cd.load_ptx("foo.ptx")    # cuModuleLoadData(PTX)
fn  = cd.get_function(mod, "ewise_kernel")
ptr = cd.alloc(N * 4)           # cuMemAlloc
cd.memcpy_h2d(ptr, numpy_arr)   # cuMemcpyHtoD
cd.launch(fn, grid, block, *args)  # cuLaunchKernel
cd.synchronize()                 # cuCtxSynchronize
cd.memcpy_d2h(numpy_arr, ptr)   # cuMemcpyDtoH
ptr.free()                       # cuMemFree
```

### run_ewise.py

```bash
.venv/bin/python harness/run_ewise.py --ptx /tmp/ewise_sm100.ptx
.venv/bin/python harness/run_ewise.py --ptx /tmp/ewise_sm100.ptx --perf
```

实际输出（B200, sm_100, 2026-03-29）：
```
=== Correctness ===
           N     max_err  result
          32    0.00e+00  PASS
         100    0.00e+00  PASS
        1024    0.00e+00  PASS
       10000    0.00e+00  PASS
     1000000    0.00e+00  PASS
```

### run_promoted_stencil.py

```bash
.venv/bin/python harness/run_promoted_stencil.py --ptx /tmp/stencil_sm100.ptx
.venv/bin/python harness/run_promoted_stencil.py --ptx /tmp/stencil_sm100.ptx --perf
```

实际输出（B200, sm_100, 2026-03-29）：
```
=== Correctness ===
         shape     max_err  result
(  64,  64)       0.00e+00  PASS
( 128, 128)       0.00e+00  PASS
( 512, 512)       0.00e+00  PASS
(1024,1024)       0.00e+00  PASS
```

---

## 一键运行

```bash
cd /home/scratch.huanhuanc_gpu/spmd/spmd-dialect

# correctness only（auto-detect GPU = sm_100 on B200）
bash scripts/run-validation.sh

# correctness + performance
bash scripts/run-validation.sh --perf

# 强制 sm_80（兼容测试）
bash scripts/run-validation.sh --sm sm_80
```

---

## 调试指南

### ABI 不匹配时

PTX 每个实际读取的参数都有 `ld.param` 指令：

```bash
grep "ld.param" /tmp/ewise_sm100.ptx
```

对照 `harness/run_ewise.py` 中的 `memref1d()` 调用顺序，逐参数验证。

### 结果错误时

1. 缩小问题：用 `--sizes 32` 跑最小 case
2. 检查 grid/block 计算是否与 PTX 中的 `%tid.x`, `%ctaid.x` 逻辑一致
3. 检查 stride：如果数组不是 stride=1，需要相应修改 descriptor

### JIT 编译失败时

Driver 会通过 `cuModuleLoadData` 返回非零 CUresult，`cuda_driver.py` 会打印
完整的错误名和描述（`cuGetErrorName` + `cuGetErrorString`）。

常见原因：PTX 的 `.target sm_XX` 高于驱动实际支持的范围（通常不会出现，
因为驱动 JIT 兼容旧 PTX）；或 PTX 语法有误（运行 `llc -filetype=asm` 时会先报错）。

---

## 文件清单

```
spmd-dialect/
  scripts/
    detect-gpu.py          ← GPU SM 级别检测
    gen-ptx.sh             ← PTX 生成（支持 SM 参数）
    run-validation.sh      ← 一键验证脚本
  harness/
    cuda_driver.py         ← ctypes CUDA Driver API 封装
    run_ewise.py           ← ewise correctness + perf
    run_promoted_stencil.py← stencil correctness + perf
  docs/
    gpu-validation.md      ← 本文档
```

---

## 下一步

1. 运行 `bash scripts/run-validation.sh` 确认 correctness PASS
2. 运行 `bash scripts/run-validation.sh --perf` 收集 CPU vs GPU 数据
3. 把 correctness 结果截图/日志作为论文 "Evaluation" 章节的证据
4. 如需更多 benchmark（reduction、2D elementwise），在 `test/SPMD/` 加新 .mlir
   文件，然后在 `harness/` 加对应 harness，流程完全一致

--- Original Design Draft End ---
