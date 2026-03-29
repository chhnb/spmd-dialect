# SPMDToGPU 实现方案 v4（最终执行版）

## 目标

将 `spmd` 方言的 S1/S2 IR 降级到 MLIR GPU 方言，最终通过官方 NVVM pipeline 生成 NVPTX backend output。

研究目标：**同一个 spmd.forall S0 kernel → CPU obj ✓（已完成）→ NVPTX backend output ✓（本方案目标）**，完成研究闭环。

---

## 核心映射表

### 逻辑映射（`convert-spmd-to-gpu` 输出）

| SPMD 概念 | GPU 映射 |
|-----------|----------|
| `spmd.forall` (mapping=group) | `gpu.launch blocks=gridDim threads=blockDim` |
| `spmd.forall` (mapping=lane) | launch region 自带 block args `%tx/%ty/%tz`（不另造 `gpu.thread_id`）|
| `spmd.barrier {scope=group}` | `gpu.barrier memfence [#gpu.address_space<workgroup>]`（promoted path；非 promoted path 用 plain `gpu.barrier`）|
| `#spmd.addr_space<group>` buffer | `gpu.launch workgroup(...)` attribution → `memref<..., #gpu.address_space<workgroup>>`；原 `memref.alloc` 删除，所有 uses 重写为 attribution block arg |
| `#spmd.addr_space<private>` | `#gpu.address_space<private>` |
| `#spmd.addr_space<global>` | 保持默认（无 memory space annotation）|
| `spmd.if` | **由前置 `SPMDToSCF` 消除** → `scf.if` |
| `spmd.reduce` | **由前置 `SPMDToSCF` 消除** → `scf.for + iter_args` |
| `spmd.yield` | 终止符，不产生指令 |

### NVVM 目标备注

`#gpu.address_space<workgroup/private>` 由后续 `convert-gpu-to-nvvm` 映射到 target-specific numeric address space。`SPMDToGPU` pass **不直接写裸整数**。

---

## IR 变换示例

### Phase 1：elementwise kernel（无 promotion）

**输入 S1 IR**：

```mlir
func.func @ewise(%A: memref<?xf32>, %B: memref<?xf32>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  spmd.forall (%c0) to (%N) step (%c256) {
  ^bb0(%gi: index):
    spmd.forall (%c0) to (%c256) step (%c1) {
    ^bb0(%li: index):
      %idx = arith.addi %gi, %li : index
      %v = memref.load %A[%idx] : memref<?xf32>
      memref.store %v, %B[%idx] : memref<?xf32>
      spmd.yield
    } {spmd.mapping = #spmd.level<lane>}
    spmd.yield
  } {spmd.mapping = #spmd.level<group>}
  return
}
```

**输出（`convert-spmd-to-gpu` 之后）**：

```mlir
func.func @ewise(%A: memref<?xf32>, %B: memref<?xf32>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  %gridX = arith.ceildivui %N, %c256 : index
  // launch region block args: %bx,%by,%bz,%tx,%ty,%tz + dim args
  gpu.launch blocks(%bx, %by, %bz) in (%gx=%gridX, %gy=%c1, %gz=%c1)
             threads(%tx, %ty, %tz) in (%sx=%c256, %sy=%c1, %sz=%c1) {
    %gi = arith.muli %bx, %c256 : index
    %idx = arith.addi %gi, %tx : index
    %inBounds = arith.cmpi ult, %idx, %N : index
    scf.if %inBounds {
      %v = memref.load %A[%idx] : memref<?xf32>
      memref.store %v, %B[%idx] : memref<?xf32>
    }
    gpu.terminator
  }
  return
}
```

注意：直接使用 `gpu.launch` region 自带的 `%bx`/`%tx` block args，无需额外插入 `gpu.thread_id`。

### Phase 2：promoted stencil（含 workgroup memory attribution）

**输入 S2 IR（PromoteGroupMemory 输出）**：

```mlir
spmd.forall (%ii, %jj) to (%N, %M) step (%c32, %c8) {
  %T = memref.alloc() : memref<33x9xf32, #spmd.addr_space<group>>
  // cooperative copy lane forall（从全局内存 load 到 %T）
  spmd.barrier {spmd.scope = #spmd.scope<group>}
  // compute lane forall（从 %T load）
} {spmd.mapping = #spmd.level<group>}
```

**输出（`convert-spmd-to-gpu` 之后）**：

```mlir
gpu.launch blocks(%bx, %by, %bz) in (%gx, %gy, %c1)
           threads(%tx, %ty, %tz) in (%sx, %sy, %c1)
           workgroup(%T : memref<33x9xf32, #gpu.address_space<workgroup>>) {
  // %T 现在是 workgroup attribution block arg；原 memref.alloc 已删除
  // cooperative copy（%tx 线性展开覆盖 33×9=297 个元素）
  %flat = %tx
  %copy_ok = arith.cmpi ult, %flat, %c297 : index
  scf.if %copy_ok {
    // boundary guard
    scf.if %inBounds {
      %v = memref.load %A[...]
      memref.store %v, %T[...]   // uses 已重写为 attribution block arg
    }
  }
  // barrier 在收敛点（copy scf.if 外侧），声明 workgroup memory 可见性
  gpu.barrier memfence [#gpu.address_space<workgroup>]
  // compute（仅前 256 threads）
  scf.if %compute_ok {
    %c = memref.load %T[...]
    ...
  }
  gpu.terminator
}
```

**重绑定流程（collectWorkgroupBuffers 的完整职责）**：

1. 识别 group forall body 内所有 `memref.alloc(..., #spmd.addr_space<group>)` op
2. 将其加入 `gpu.launch` 的 `workgroup(...)` attribution，类型改为 `#gpu.address_space<workgroup>`
3. 用 `replaceAllUsesWith` 把原 `%T` 的所有 uses 重写为 launch body 里对应的 attribution block arg
4. 删除原 `memref.alloc` op

步骤 3+4 是必须做的收口：若跳过，IR 会同时保留旧 alloc 和新 attribution，导致 `gpu-kernel-outlining` / NVVM lowering 出现重复定义。

---

## `gpu.barrier` 放置约束

`gpu.barrier`（含 `memfence` 变体）同步整个 workgroup，要求所有 work items 均在收敛路径上到达该点。

**正确**：barrier 放在 cooperative copy `scf.if %copy_ok` 的**外侧**（所有 thread 均到达）。

**错误**：barrier 放在 `scf.if %inBounds` 的**内侧**（部分 thread 可能跳过 → workgroup deadlock）。

`SPMDToGPU` pass 在插入 `gpu.barrier` 时应做 structural check：barrier 的直接父 region 必须是 `gpu.launch` body，不能嵌套在 `scf.if` 内。

---

## blockDim 确定策略（correctness-first heuristic）

| 情况 | blockDim（MVP heuristic）|
|------|--------------------------|
| 无 promotion，lane forall 1D | tile_size[0] |
| 无 promotion，lane forall 2D | tile_size[0] × tile_size[1]（展开为 1D）|
| 有 promotion | max(∏extent_copy, ∏tile_compute)，硬限制 ≤ 1024 |

**注**：此 heuristic 以正确性为优先，不保证性能最优。warp-size 对齐、2D/3D thread layout、occupancy 调优留待后续阶段。

---

## Pipeline 设计

### Pass 职责划分

`SPMDToGPU` 的职责收缩到只处理并行结构映射：

- `spmd.forall(group)` → `gpu.launch`
- `spmd.forall(lane)` → thread/block id（launch block args）
- `spmd.barrier` → `gpu.barrier [memfence workgroup]`
- group buffer → `workgroup(...)` attribution + 重绑定 + 删除原 alloc
- `#spmd.addr_space<*>` → `#gpu.address_space<*>`

`spmd.if` / `spmd.reduce` 由**前置 `SPMDToSCF`** 消除，`SPMDToGPU` 不重新实现。

### 完整 GPU Pipeline

```
S0 kernel
  │
  ▼ --normalize-spmd
  ▼ --plan-spmd-schedule
  ▼ --materialize-spmd-tiling
  ▼ --promote-group-memory              (可选；生成 group addr space alloc + barrier)
  │
  ▼ --convert-spmd-structured-to-scf   // 先消 spmd.if / spmd.reduce → scf
  │   (复用已有 SPMDToSCF，只跑 if/reduce pattern)
  │
  ▼ --convert-spmd-to-gpu              ← 本方案新增 pass
  │   group forall → gpu.launch (gridDim/blockDim)
  │   lane forall → launch block args
  │   group buffer → workgroup attribution（含重绑定 + 删除原 alloc）
  │   spmd.barrier → gpu.barrier memfence [workgroup]
  │   #spmd.addr_space<*> → #gpu.address_space<*>
  │
  ▼ --gpu-kernel-outlining             (MLIR 内置)
  │   gpu.launch body → gpu.func in gpu.module
  │
  ▼ [优先] --gpu-lower-to-nvvm-pipeline{chip=sm_80}
  │   官方默认 NVVM 编译路径；处理 arith/memref/scf/vector/gpu/nvgpu
  │
  ▼ [fallback，若 gpu-lower-to-nvvm-pipeline 不可用]
  │   --nvvm-attach-target{chip=sm_80}
  │   --pass-pipeline='builtin.module(gpu.module(convert-gpu-to-nvvm),gpu-to-llvm)'
  │   --convert-arith-to-llvm --finalize-memref-to-llvm
  │   --convert-func-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts
  │
  ▼ mlir-translate --mlir-to-llvmir
  │
  ▼ （路径 A）llc --march=nvptx64 --mcpu=sm_80 -filetype=obj -o /dev/null
  │   → backend codegen legality smoke check
  │
  ▼ （路径 B）llc --march=nvptx64 --mcpu=sm_80 -filetype=asm | FileCheck
  │   → PTX 文本特征检查（.shared / .visible .entry，promoted path 专用）
```

---

## Pass 实现结构

### `lib/Conversion/SPMDToGPU/SPMDToGPU.cpp`

```cpp
struct ConvertSPMDToGPUPass
    : PassWrapper<ConvertSPMDToGPUPass, OperationPass<ModuleOp>> {
  void runOnOperation() override;
};

void ConvertSPMDToGPUPass::runOnOperation() {
  ModuleOp module = getOperation();
  // 直接 IR surgery：walk group-level forall，生成 gpu.launch
  // 不走 ConversionPattern，避免递归 legalization 复杂度
  module.walk([&](spmd::ForallOp forall) {
    if (isGroupLevel(forall))
      lowerGroupForallToGPULaunch(forall);
  });
}
```

**主要函数**：

```cpp
// group-level forall → gpu.launch（含 workgroup attribution for group buffers）
void lowerGroupForallToGPULaunch(spmd::ForallOp groupForall);

// 在 gpu.launch body 内，lane-level forall → 使用 launch region block args（%tx/%ty）
void lowerLaneForallToThreadArgs(spmd::ForallOp laneForall,
                                 gpu::LaunchOp launch);

// 计算 gridDim（基于 group forall lb/ub/step）
SmallVector<Value, 3> computeGridDim(spmd::ForallOp groupForall, OpBuilder &b);

// 计算 blockDim（correctness-first heuristic，硬限 ≤ 1024）
SmallVector<Value, 3> computeBlockDim(spmd::ForallOp groupForall, OpBuilder &b);

// 收集 group-address-space alloc → 加入 workgroup attribution
// + replaceAllUsesWith(attribution block arg) + 删除原 alloc
void collectAndRebindWorkgroupBuffers(spmd::ForallOp groupForall,
                                     gpu::LaunchOp launch);

// spmd.barrier → gpu.barrier memfence [#gpu.address_space<workgroup>]
// structural check: barrier 直接父 region 必须是 gpu.launch body
void lowerBarrier(spmd::BarrierOp barrier, bool hasWorkgroupMemory);

// #spmd.addr_space<*> → #gpu.address_space<*>（不写裸整数）
void remapAddressSpaces(gpu::LaunchOp launch);
```

---

## 文件变更清单

### 新建 / 替换文件

| 文件 | 内容 |
|------|------|
| `lib/Conversion/SPMDToGPU/SPMDToGPU.cpp` | Pass 主体（替换现有 stub）|
| `test/SPMD/lower-to-gpu.mlir` | Lit 测试（4 条 RUN lines）|

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `lib/Conversion/SPMDToGPU/CMakeLists.txt` | 添加 MLIRGPUDialect、MLIRNVVMDialect 依赖 |
| `tools/spmd-opt/spmd-opt.cpp` | 注册 `registerConvertSPMDToGPUPass()` |
| `lib/Transforms/SPMDPassRegistration.cpp` | 注册 GPU pass |

---

## Lit 测试设计（`test/SPMD/lower-to-gpu.mlir`）

### 验证模式说明

RUN 1/2 是**纯 IR 结构检查**，在 `spmd-opt` 内完成，不依赖 llc。

RUN 3 是**设备侧 codegen legality smoke check**：只针对 outlined 的 `gpu.module`/`gpu.func` 验证 NVPTX backend 能走通，不涉及 host module 的完整 offloading translation。

RUN 4 是 **PTX 文本特征检查**（promoted path），同样只检查 device-side output。

> **完整 host+device offloading 路径**（若需要）：在 `gpu-kernel-outlining` + NVVM lowering 后，还需补 `--gpu-module-to-binary` 将 device code 序列化，再由 `mlir-translate` 处理 `gpu.binary` 和 `gpu.launch_func`。本方案 MVP 阶段仅验证 device-side legality，不测试完整 offloading 路径。

```
// RUN 1: elementwise kernel GPU IR 检查
// RUN: spmd-opt %s -func-to-select=@ewise \
// RUN:   --normalize-spmd --plan-spmd-schedule --materialize-spmd-tiling \
// RUN:   --convert-spmd-to-gpu \
// RUN:   | FileCheck %s --check-prefix=GPU
// GPU: gpu.launch
// GPU: gpu.terminator
// GPU-NOT: spmd.forall

// RUN 2: promoted stencil workgroup memory + barrier 检查
// RUN: spmd-opt %s -func-to-select=@stencil \
// RUN:   --normalize-spmd --plan-spmd-schedule --materialize-spmd-tiling \
// RUN:   --promote-group-memory --convert-spmd-to-gpu \
// RUN:   | FileCheck %s --check-prefix=WG
// WG: gpu.launch
// WG-SAME: workgroup({{.*}}#gpu.address_space<workgroup>
// WG: gpu.barrier{{.*}}#gpu.address_space<workgroup>
// WG-NOT: memref.alloc

// RUN 3: 设备侧 codegen legality smoke check（优先官方 pipeline）
// RUN: spmd-opt %s -func-to-select=@ewise \
// RUN:   --normalize-spmd --plan-spmd-schedule --materialize-spmd-tiling \
// RUN:   --convert-spmd-to-gpu --gpu-kernel-outlining \
// RUN:   --gpu-lower-to-nvvm-pipeline='chip=sm_80' \
// RUN:   | mlir-translate --mlir-to-llvmir \
// RUN:   | llc --march=nvptx64 --mcpu=sm_80 -filetype=obj -o /dev/null
// (fallback: 若 gpu-lower-to-nvvm-pipeline 不可用，展开为手工 pass 序列)
// 注：此处 mlir-translate 针对 gpu.module 内的 device IR，非完整 host module

// RUN 4: PTX 文本特征检查（promoted path，设备侧 -filetype=asm）
// RUN: spmd-opt %s -func-to-select=@stencil \
// RUN:   --normalize-spmd --plan-spmd-schedule --materialize-spmd-tiling \
// RUN:   --promote-group-memory --convert-spmd-to-gpu --gpu-kernel-outlining \
// RUN:   --gpu-lower-to-nvvm-pipeline='chip=sm_80' \
// RUN:   | mlir-translate --mlir-to-llvmir \
// RUN:   | llc --march=nvptx64 --mcpu=sm_80 -filetype=asm \
// RUN:   | FileCheck %s --check-prefix=PTX
// PTX: .shared
// PTX: .visible .entry
```

---

## Phase 1 执行 Checklist（non-promoted path）

- [ ] 实现 `lowerGroupForallToGPULaunch`（不处理 group buffer）
- [ ] 实现 `lowerLaneForallToThreadArgs`（1D/2D → linear，使用 launch block args）
- [ ] 实现 `lowerBarrier`（plain `gpu.barrier`，含 structural check）
- [ ] 实现 `computeGridDim` / `computeBlockDim`
- [ ] 注册 pass，确认 pipeline 中 `SPMDToSCF` 先于 `SPMDToGPU`
- [ ] 检查 `gpu-kernel-outlining` 后 enclosing module 是否带 `gpu.container_module` 属性；若未自动添加，在 outlining 前手动设置或在 pass 末尾补上
- [ ] RUN 1 通过（GPU IR 检查）
- [ ] RUN 3 通过（device-side codegen smoke check）
- [ ] 原有 14 个 lit 测试不受影响

## Phase 2 执行 Checklist（promoted path）

- [ ] 实现 `collectAndRebindWorkgroupBuffers`：识别 group alloc → attribution + replaceAllUsesWith + erase
- [ ] 实现 `remapAddressSpaces`：`#spmd.addr_space<group>` → `#gpu.address_space<workgroup>`
- [ ] 更新 `lowerBarrier`：promoted path 发 `gpu.barrier memfence [#gpu.address_space<workgroup>]`
- [ ] cooperative copy lane forall 的 2D→1D 线性 thread 展开
- [ ] RUN 2 通过（workgroup attribution + memfence barrier 检查）
- [ ] RUN 4 通过（PTX `.shared` / `.visible .entry` 检查）
- [ ] 验证 IR 中无残留 `memref.alloc(..., #spmd.addr_space<group>)`

---

## 已知风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| `gpu-kernel-outlining` 要求合法 ModuleOp 结构 | 每次 `lowerGroupForallToGPULaunch` 后立即 `module.verify()` |
| `gpu.container_module` 属性缺失（影响 `gpu.launch_func` / translation）| outlining 前检查 enclosing module 是否已带 `gpu.container_module` 属性；若无，在 `SPMDToGPU` pass 末尾或 `gpu-kernel-outlining` 前手动设置；此属性缺失会导致后续 `gpu.launch_func` 和 LLVM translation 步骤出现难以定位的错误 |
| blockDim > 1024（CUDA 硬件限制）| `computeBlockDim` 中插入 `emitError` 断言 |
| `gpu.barrier` 放入 divergent 条件（workgroup deadlock）| structural check：barrier 直接父 region 必须是 `gpu.launch` body |
| workgroup buffer 重绑定漏掉某个 use（旧/新并存）| `replaceAllUsesWith` 之后断言原 value 无 uses，再 erase |
| `gpu-lower-to-nvvm-pipeline` 在本地 build 不可用 | Phase 1 先验证手工 fallback 路径；pipeline pass 验证成功后再切换为优先路径 |
| 误用完整 host module translation 路径（`mlir-translate` 遇到 `gpu.launch` 报错）| RUN 3/4 针对 device-side IR，不走完整 host offloading；如需完整路径，补 `--gpu-module-to-binary` |
| PTX FileCheck 脆弱（符号名变动）| 只检查 `.shared`、`.visible .entry` 等结构性特征，不检查具体符号名 |

---

## 验收标准

1. RUN 1：`gpu.launch` 出现，`spmd.forall` 消除，`gpu.terminator` 存在
2. RUN 2：`workgroup(... #gpu.address_space<workgroup>)` 出现；`gpu.barrier` 含 `memfence [workgroup]`；无残留 `memref.alloc`
3. RUN 3：`llc -filetype=obj -o /dev/null` 成功
4. RUN 4：PTX asm 含 `.shared` 和 `.visible .entry`
5. 原有 14 个 lit 测试全部通过
6. `SPMDToGPU` pass 代码中无 `spmd.if`/`spmd.reduce` lowering 逻辑
