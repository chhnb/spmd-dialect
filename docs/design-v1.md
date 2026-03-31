# SPMD Dialect Design v1

**Version:** 0.1
**Status:** Pre-implementation spec
**Scope:** MVP — `spmd` dialect + CPU 最小闭环 + group memory promotion demo

---

## 1. 核心主张

> 设计一个面向规则 SPMD kernel 的结构化中层 IR，使前端 DSL 的并行语义可以统一表示，mapping 与 memory hierarchy 优化在该 IR 层完成，而不是分散在各 DSL/backend 中。

三个核心贡献：

1. **Iteration-space as semantic center** — `spmd.forall` 是统一的逻辑并行域表示，不绑定 thread/block id。
2. **Delayed mapping via progressive refinement** — S0（语义态）→ S1（调度态）→ S2（物化态），并行语义先于执行语义。
3. **IR-level memory hierarchy optimization** — group memory promotion 作为中层 pass，不依赖 backend。

---

## 2. 程序子集（边界）

**面向的子集：**

- elementwise / map
- stencil
- structured reduction
- tiled BLAS-style kernels
- affine / quasi-affine 访问为主
- bounded structured control flow

**明确排除（第一版）：**

- pointer chasing / alias-heavy 程序
- 非结构化 CFG
- 动态并行
- 广义异步 runtime 编排

**与 Polyhedral IR 的关系：**

本 IR 借鉴 polyhedral 的迭代域表示，但不要求完整仿射分析，并增加 SPMD 执行语义（mapping/memory-space/barrier）作为一等概念；目标不是做 polyhedral scheduler，而是为 SPMD kernel 提供结构化的中层 IR。

---

## 3. 抽象执行模型

### 3.1 执行层级

| Level    | 语义                              | GPU 映射                    | CPU 映射               |
|----------|-----------------------------------|-----------------------------|------------------------|
| `grid`   | 全局问题空间                      | grid                        | 外层 chunk             |
| `group`  | 可协作执行单元，共享 group memory | block / workgroup           | parallel worker tile   |
| `lane`   | group 内单个 SPMD instance        | thread / invocation         | worker 内部迭代        |
| `vector` | lane 内 lockstep vector 执行      | per-thread vector fragment  | SIMD lane              |

### 3.2 内存空间

| Space     | 语义              | GPU 映射                     | CPU 映射             |
|-----------|-------------------|------------------------------|----------------------|
| `global`  | kernel 全局可见   | global memory                | main memory          |
| `group`   | 同一 group 内共享 | shared / workgroup memory    | tile scratchpad      |
| `private` | 单 lane 私有      | registers / local mem        | register / stack-local |

### 3.3 同步

MVP 只定义一个同步层级：

> `spmd.barrier {spmd.scope = #spmd.scope<group>}` — 同步当前 group 所有活跃 lane，建立 group memory 可见性边界。

---

## 4. 三阶段 IR 模型

### S0：Semantic SPMD IR

前端 lowering 的输出，纯语义态。

- 有 `spmd.forall / if / reduce`
- 所有 memref 在 `global` 地址空间
- **无** barrier，**无** group/private alloc
- mapping / tile attrs 为空

### S1：Scheduled SPMD IR

加入 schedule hints 后。

- `spmd.forall` 上携带 `spmd.mapping`, `spmd.tile_sizes`, `spmd.order`, `spmd.memory_policy`
- 未物化 tile/barrier/promoted buffer
- 仍然 backend-agnostic

### S2：Materialized SPMD IR

实现导向但仍 backend-agnostic 的物化态。

- tiling 已展开为 nested `spmd.forall`，各层有显式 `spmd.mapping`
- group/private memref 已物化（带 addr space attr）
- `spmd.barrier` 已插入
- 无 `gpu.thread_id` / `omp.parallel`（那是 backend lowering 的事）

---

## 5. Dialect 定义

**Dialect 名：** `spmd`

**依赖（复用，不重造）：** `func`, `arith`, `math`, `memref`, `affine`, `vector`

**自定义内容：** 5 个 op + 5 个 attr class

---

### 5.1 Attributes

#### `#spmd.level<...>`

```
LevelAttr ::= 'seq' | 'grid' | 'group' | 'lane' | 'vector'
```

用于标记 `spmd.forall` 的 mapping 层级。

#### `#spmd.scope<...>`

```
ScopeAttr ::= 'group'
```

MVP 只支持 `group`。

#### `#spmd.reduction_kind<...>`

```
ReductionKindAttr ::= 'add' | 'mul' | 'max' | 'min' | 'and' | 'or' | 'xor'
```

#### `#spmd.addr_space<...>`

```
AddressSpaceAttr ::= 'global' | 'group' | 'private'
```

用作 `memref` 的 memory space parameter：

```mlir
memref<32x32xf32, #spmd.addr_space<group>>
```

#### `#spmd.memory_policy<...>`

```
MemoryPolicyAttr ::= 'none' | 'prefer_group' | 'prefer_private' | 'no_promotion'
```

---

### 5.2 `spmd.forall`

**作用：** 矩形 iteration domain 上的逻辑并行执行集合。

**语法：**

```mlir
// 完整形式
spmd.forall (%i, %j) = (%lb0, %lb1) to (%ub0, %ub1) step (%s0, %s1)
    [attributes {...}] {
  ...
  spmd.yield
}

// 简写（lb=0, step=1）
spmd.forall (%i, %j) in (%N, %M) [attributes {...}] {
  ...
  spmd.yield
}
```

**Operands：** variadic `(lbs, ubs, steps)`, 类型均为 `index`

**Results：** 无

**Region：** 单 region，单 block，block args = induction vars（`index`），terminator = `spmd.yield`

**可选 Attrs：**

```
spmd.mapping       : #spmd.level<...>
spmd.tile_sizes    : DenseI64ArrayAttr   // 长度 = rank，值全正
spmd.order         : DenseI64ArrayAttr   // [0..rank-1] 的排列
spmd.memory_policy : #spmd.memory_policy<...>
```

**Verifier：**

1. `lbs`, `ubs`, `steps` 长度一致，rank ≥ 1
2. block args 个数 = rank，类型全为 `index`
3. 若 `spmd.order` 存在：必须是合法排列
4. 若 `spmd.tile_sizes` 存在：长度 = rank，值全正
5. 若 `spmd.mapping` 存在：必须是合法 `LevelAttr`
6. steps 在运行时必须 > 0（静态常量时 verifier 检查；动态值由调用方保证）

**语义：**

所有 `(i0, i1, ...)` 点构成的并行实例集合，不承诺执行顺序。
跨 iteration 的非结构化冲突访问属于非法 IR（未定义行为）。

**Canonicalization：**

- 规范到 0-based + unit-step 形式
- 单元素维度折叠
- 常量 trip count 折叠
- 非矩形域 → 矩形 + `spmd.if`

---

### 5.3 `spmd.if`

**作用：** per-instance 条件分支（不同 lane 走不同分支合法）。

**语法：**

```mlir
// 无结果
spmd.if %cond {
  spmd.yield
}

// 有结果
%r = spmd.if %cond -> (f32) {
  spmd.yield %x : f32
} else {
  spmd.yield %y : f32
}
```

**Verifier：**

1. `%cond` 类型为 `i1`
2. 有结果 → then/else 都必须存在，yield 类型和数量匹配 op 结果
3. 无结果 → else 可省略

---

### 5.4 `spmd.reduce`

**作用：** 显式结构化 reduction。MVP **只支持单维 reduction**（多维用 nested `spmd.reduce` 表达）。

**语法：**

```mlir
%result = spmd.reduce (%k) = (%lb) to (%ub) step (%step)
              init(%init_val)
              attributes {spmd.kind = #spmd.reduction_kind<add>} {
  %contrib = ...
  spmd.yield %contrib : f32
}
```

**Operands：** `lb, ub, step`（`index`）+ `init`（reduction type）

**Results：** 1 个，类型与 `init` 相同

**Region：** 单 block，block arg = `%k : index`，terminator = `spmd.yield`（恰好 1 个值）

**Verifier：**

1. `spmd.kind` 必须存在且合法
2. `init` 类型 = result 类型
3. body yield 恰好 1 个值，类型匹配 result
4. MVP：body 只允许纯算术 + non-volatile load，不允许未知 side effects

**语义：**

对 `[lb, ub)` 内所有 `k` 的 contribution 做 `kind` 组合，顺序不固定。

**浮点注意：** `add/mul` 的 reassociation 仅在 fast-math 语义下合法；否则 backend 应保守保序。

---

### 5.5 `spmd.barrier`

**作用：** group-scope 同步点。

**语法：**

```mlir
spmd.barrier {spmd.scope = #spmd.scope<group>}
```

**无 operands / results / regions。**

**Verifier：**

1. `spmd.scope` 必须存在，MVP 只允许 `group`
2. 其祖先中必须存在 `spmd.mapping = #spmd.level<group>` 的 `spmd.forall`

**使用阶段：** 不由前端生成；由 `PromoteGroupMemory` pass 插入，仅出现在 S2。

---

### 5.6 `spmd.yield`

**作用：** region terminator。

**Verifier（上下文相关）：**

- 在 `spmd.forall` 中：必须为空 yield
- 在 `spmd.if` 中：类型匹配 op results
- 在`spmd.reduce` 中：恰好 1 个值，类型匹配 result

---

### 5.7 Kernel 入口约定

复用 `func.func`，添加 `spmd.kernel` attr：

```mlir
func.func @kernel(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %N: index, %M: index)
    attributes {spmd.kernel} {
  ...
  func.return
}
```

**MVP kernel 约束：**

- 无返回值，输出通过 memref 写回
- 不允许递归
- 不允许未知 side-effecting `func.call`
- 参数类型：`index / integer / float` + `memref`

---

## 6. Pass Pipeline

```
Frontend lowering  (或 hand-written IR)
       │
       ▼
 [S0] NormalizeSPMD
       │  规范 bounds/steps；非矩形域 → 矩形 + mask
       ▼
 [S0] VerifySPMDKernelSubset
       │  op 白名单；barrier 作用域；S0 纯洁性检查
       ▼
 [S0→S1] PlanSPMDSchedule
       │  写入 tile_sizes / mapping / memory_policy attrs
       ▼
 [S1→S2] MaterializeTilingAndMapping
       │  展开 nested forall；标记各层 mapping
       ▼
 [S2] PromoteGroupMemory              ← 核心创新 pass
       │  分析 tile footprint；生成 group memref；插 barrier
       ▼
 [S2] VectorizeOrPrivatizeSPMD        (MVP 轻实现)
       │
       ▼
 LowerSPMDToBackend
       ├─ CPU: → scf + OpenMP + LLVM
       └─ GPU: → gpu dialect + NVVM/ROCDL
```

---

## 7. `PromoteGroupMemory` 算法框架

```
输入:  S2 中带 spmd.mapping=group 的 spmd.forall F
输出:  F.body 内插入 group memref alloc + cooperative copy
       + barrier + rewrite

步骤:
1. 收集 F.body 内所有 memref.load/store
2. 对每个候选 memref M:
   a. 计算 tile footprint（affine 访问区域）
   b. 若含 stencil halo，扩展 footprint
   c. Legality 检查:
      - footprint 可界定
      - 多个 lane 复用同一区域（reuse count > 1）
      - 无跨 group 写后读/写后写冲突
      - sizeof(footprint) ≤ target.maxGroupMemBytes
      - address 可重写为 tile-local index
   d. Profitability 检查:
      - copy-in amortized cost < 节省的 global 访问
      - 不显著恶化 occupancy
3. 对通过检查的 M，执行 promotion:
   a. memref.alloc → group addr space
   b. 在 compute forall 前插入 cooperative copy loop (lane-level forall)
   c. 插入 spmd.barrier（copy 完成后）
   d. 重写原 load → 访问 tile-local buffer
   e. MVP: 只做 read-only promotion，不做 write-back

注意: legality / profitability 判断由分析 pass 完成；
      不可分析的访问 ≠ 非法，只是不做激进优化。
```

---

## 8. Target Descriptor

```cpp
struct TargetDescriptor {
  enum BackendKind { CPU, CUDA, ROCM, SPIRV };
  BackendKind backend;
  int simdWidth;           // CPU SIMD / GPU vector width
  int subgroupWidth;       // GPU warp/wavefront size
  int maxGroupSize;        // max threads per block / work-items per workgroup
  int maxGroupMemBytes;    // shared/workgroup memory limit (bytes)
  int cacheLineBytes;
  int l1Bytes;
  int registerBudget;      // per-lane register budget (words)
  bool supportsGroupBarrier;
};
```

`PlanSPMDSchedule` 和 `PromoteGroupMemory` 以此为输入驱动决策。

---

## 9. 完整 IR 示例

### S0：elementwise

```mlir
func.func @ewise_square(%A: memref<?x?xf32>, %B: memref<?x?xf32>,
                         %N: index, %M: index)
    attributes {spmd.kernel} {
  spmd.forall (%i, %j) in (%N, %M) {
    %x = memref.load %A[%i, %j] : memref<?x?xf32>
    %y = arith.mulf %x, %x : f32
    memref.store %y, %B[%i, %j] : memref<?x?xf32>
    spmd.yield
  }
  func.return
}
```

### S0：reduction

```mlir
func.func @sum(%A: memref<?xf32>, %out: memref<1xf32>, %N: index)
    attributes {spmd.kernel} {
  %c0   = arith.constant 0 : index
  %c1   = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %sum = spmd.reduce (%k) = (%c0) to (%N) step (%c1)
             init(%zero)
             attributes {spmd.kind = #spmd.reduction_kind<add>} {
    %x = memref.load %A[%k] : memref<?xf32>
    spmd.yield %x : f32
  }
  memref.store %sum, %out[%c0] : memref<1xf32>
  func.return
}
```

### S2：stencil with group memory promotion

```mlir
func.func @stencil2d(%A: memref<?x?xf32>, %B: memref<?x?xf32>,
                      %N: index, %M: index)
    attributes {spmd.kernel} {
  %c0  = arith.constant 0  : index
  %c1  = arith.constant 1  : index
  %c8  = arith.constant 8  : index
  %c32 = arith.constant 32 : index
  %c34 = arith.constant 34 : index
  %c10 = arith.constant 10 : index

  spmd.forall (%ii, %jj) = (%c1, %c1) to (%N, %M) step (%c32, %c8)
      attributes {spmd.mapping = #spmd.level<group>} {

    %tile = memref.alloc() : memref<34x10xf32, #spmd.addr_space<group>>

    // cooperative copy: all lanes load halo tile from global
    spmd.forall (%li, %lj) = (%c0, %c0) to (%c34, %c10) step (%c1, %c1)
        attributes {spmd.mapping = #spmd.level<lane>} {
      %gi0 = arith.subi %ii, %c1 : index
      %gj0 = arith.subi %jj, %c1 : index
      %gi  = arith.addi %gi0, %li : index
      %gj  = arith.addi %gj0, %lj : index
      %vi  = arith.cmpi ult, %gi, %N : index
      %vj  = arith.cmpi ult, %gj, %M : index
      %ok  = arith.andi %vi, %vj : i1
      spmd.if %ok {
        %v = memref.load %A[%gi, %gj] : memref<?x?xf32>
        memref.store %v, %tile[%li, %lj]
            : memref<34x10xf32, #spmd.addr_space<group>>
        spmd.yield
      } else { spmd.yield }
      spmd.yield
    }

    spmd.barrier {spmd.scope = #spmd.scope<group>}

    // compute: read from tile (group memory)
    spmd.forall (%ti, %tj) = (%c0, %c0) to (%c32, %c8) step (%c1, %c1)
        attributes {spmd.mapping = #spmd.level<lane>} {
      %i   = arith.addi %ii, %ti : index
      %j   = arith.addi %jj, %tj : index
      %ti1 = arith.addi %ti, %c1 : index
      %tj1 = arith.addi %tj, %c1 : index
      %center = memref.load %tile[%ti,  %tj ]
          : memref<34x10xf32, #spmd.addr_space<group>>
      %right  = memref.load %tile[%ti,  %tj1]
          : memref<34x10xf32, #spmd.addr_space<group>>
      %down   = memref.load %tile[%ti1, %tj ]
          : memref<34x10xf32, #spmd.addr_space<group>>
      %t0 = arith.addf %center, %right : f32
      %t1 = arith.addf %t0, %down : f32
      memref.store %t1, %B[%i, %j] : memref<?x?xf32>
      spmd.yield
    }

    spmd.yield
  }
  func.return
}
```

---

## 10. Kernel Legality Pass：允许的 op 白名单

**允许：**

- `spmd.{forall, if, reduce, barrier, yield}`
- `arith.*`, `math.*`
- `memref.{load, store, subview, cast}`
- `affine.apply`
- `vector.*`（可选）
- `func.return`

**不允许（S0/S1）：**

- `cf.*`, `scf.while`
- `gpu.*`, `omp.*`
- 未知 side-effecting `func.call`
- `group` / `private` addr space memref（S0 中不允许）
- `spmd.barrier`（S0/S1 中不允许）

---

## 11. MVP 验收标准

### Phase 1：IR 立起来

- [x] 5 个 attr class：ODS 定义 + C++ 实现
- [x] 5 个 op：ODS 定义 + verifier + printer/parser
- [x] `spmd.kernel` legality pass
- [x] lit tests：`test/SPMD/ops.mlir` + `test/SPMD/invalid.mlir`

### Phase 2：CPU 最小闭环

- [x] `NormalizeSPMD`
- [x] `MaterializeTilingAndMapping`
- [x] `LowerSPMDToSCF`
- [x] `SCF → OpenMP → LLVM`
- [x] 跑通：elementwise / reduction / stencil（无 promotion）

### Phase 3：Group memory promotion demo

- [x] `PromoteGroupMemory`（只做 read-only stencil pattern）
- [x] 2D stencil 全流程：S0 → S2 → CPU

### Phase 4：GPU mapping

- [x] `group → block`，`lane → thread`
- [x] group addr space → shared memory
- [x] barrier → workgroup barrier
- [x] 跑通 stencil on CUDA

### Phase 5：Hierarchical GPU reduction（post-MVP）

- [x] `ReduceToHierarchicalGPU`：intra-block shared-memory tree + per-block global atomic
- [x] GPU speedup >1× vs CPU serial for N ≥ 64K（实测：1.1×@65K，10.2×@1M，24.0×@16M on B200）
- [x] Legality guards（non-pure body → fallback，non-Add kind → fallback）
- [x] 全套 lit tests + correctness harness（所有尺寸 PASS，rel_err < 1e-3）

---

## 12. 文件组织

```
spmd-dialect/
├── docs/
│   └── design-v1.md                  ← 本文件
├── include/spmd/IR/
│   ├── SPMDDialect.h
│   ├── SPMDOps.h
│   ├── SPMDOps.td
│   ├── SPMDAttrs.h
│   └── SPMDAttrs.td
├── lib/
│   ├── IR/
│   │   ├── SPMDDialect.cpp
│   │   ├── SPMDOps.cpp
│   │   └── SPMDAttrs.cpp
│   ├── Analysis/
│   │   ├── AccessSummaryAnalysis.cpp
│   │   └── PromotionPlanAnalysis.cpp
│   ├── Transforms/
│   │   ├── VerifySPMDKernelSubset.cpp
│   │   ├── NormalizeSPMD.cpp
│   │   ├── PlanSPMDSchedule.cpp
│   │   ├── MaterializeTilingAndMapping.cpp
│   │   └── PromoteGroupMemory.cpp
│   └── Conversion/
│       ├── SPMDToSCF/SPMDToSCF.cpp
│       ├── SPMDToOpenMP/SPMDToOpenMP.cpp
│       └── SPMDToGPU/SPMDToGPU.cpp
└── test/SPMD/
    ├── ops.mlir
    ├── invalid.mlir
    ├── normalize.mlir
    ├── promotion.mlir
    ├── lower-to-openmp.mlir
    └── lower-to-gpu.mlir
```

---

## 13. 论文 Contribution 表述

```
C1: A structured SPMD IR centered on spmd.forall, decoupling
    iteration-space semantics from backend execution mapping.

C2: A three-phase progressive refinement framework (Semantic →
    Scheduled → Materialized) that preserves analyzability while
    enabling target-specific optimization.

C3: An IR-level group-memory promotion framework that automatically
    identifies tile-reusable footprints and materializes cooperative
    loads, barriers, and address remapping across CPU and GPU targets.
```

---

## 附：与原始方案的主要修订

1. **`spmd.reduce` MVP 限制为单维**——避免多维 reduction 语义歧义；多维用 nested `spmd.reduce`。
2. **stencil 示例修正**——index 算术须先 `arith.addi` 再作 subscript，MLIR 不支持内联表达式。
3. **steps 约束放宽**——从"必须是编译期常量"改为"运行时必须 > 0"，允许动态 tile size。
4. **`PromoteGroupMemory` 补充算法草稿**——明确 legality/profitability 的判断流程。
5. **补充与 Polyhedral IR 的关系说明**——应对 reviewer 对比质疑。
