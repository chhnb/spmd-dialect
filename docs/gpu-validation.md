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

=== Performance (wall-clock, no data transfer) ===

           N    cpu_ms    gpu_ms   speedup
      100000     0.024     0.019       1.3x
     1000000     0.453     0.020      22.1x
    10000000    10.881     0.206      52.8x
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

=== Performance ===

         shape    cpu_ms    gpu_ms   speedup
( 512, 512)        0.330     0.024      13.8x
(1024,1024)        2.817     0.044      63.5x
(2048,2048)       11.317     0.112     101.0x
(4096,4096)       46.883     0.285     164.7x
```

### run_reduction.py

```bash
.venv/bin/python harness/run_reduction.py --ptx /tmp/reduction_sm100.ptx
.venv/bin/python harness/run_reduction.py --ptx /tmp/reduction_sm100.ptx --perf
```

实际输出（B200, sm_100, 2026-03-29）：
```
=== Correctness ===
           N         gpu_sum         ref_sum     rel_err  result
        1024      527.198730      527.198608    2.32e-07  PASS
       65536    32708.308594    32708.339844    9.55e-07  PASS
     1048576   524550.687500   524541.875000    1.68e-05  PASS
```

注：`atomic_sum_kernel` 使用全局原子加法，性能不及 CPU numpy（后者使用 SIMD 向量化）。
此 kernel 的目标是**正确性验证**，而非性能竞争。

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
  test/SPMD/
    lower-to-gpu-nvptx-reduction.mlir ← atomic sum kernel source
  harness/
    cuda_driver.py         ← ctypes CUDA Driver API 封装（含 memset）
    run_ewise.py           ← ewise correctness + perf
    run_promoted_stencil.py← stencil correctness + perf
    run_reduction.py       ← atomic sum correctness + perf
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
