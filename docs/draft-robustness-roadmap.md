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
