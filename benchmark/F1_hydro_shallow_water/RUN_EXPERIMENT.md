# Gather Promotion Experiment — Run Plan

## 目的

验证核心假设：**对非结构化网格 kernel 的间接访问 (gather)，自动 shared memory promotion 能否带来有意义的加速？**

## 环境要求

- GPU 计算节点（login 节点没有 nvcc 和 GPU）
- CUDA toolkit (nvcc)
- GPU: 任意 NVIDIA GPU (sm_70+)

## Step 1: 获取 GPU 节点

```bash
# 根据你的集群调度系统，申请一个 GPU 节点
# 例如 SLURM:
srun --gres=gpu:1 --time=1:00:00 --pty bash

# 或者 PBS:
qsub -I -l nodes=1:ppn=1:gpus=1 -l walltime=01:00:00
```

## Step 2: 编译

```bash
cd /home/scratch.huanhuanc_gpu/spmd/spmd-dialect/benchmark/F1_hydro_shallow_water

# 查看 GPU 架构（选择对应的 -arch）
nvidia-smi | head -5

# 编译（根据 GPU 架构调整 sm_XX）
# A100: sm_80,  H100/B200: sm_90,  V100: sm_70,  RTX 3090: sm_86
nvcc -O3 -arch=sm_80 gather_experiment.cu -o gather_experiment
```

## Step 3: 运行实验

```bash
# 小网格 (热身，确认能跑)
./gather_experiment 64 100

# 中网格
./gather_experiment 128 100

# 大网格 (关键数据点)
./gather_experiment 256 100

# 超大网格 (如果 shared mem 够)
./gather_experiment 512 50

# 如果 512 报 shared memory 超限，用更小的 step 或改 BLOCK_SIZE
```

## Step 4: 记录结果

输出格式：
```
=== Results ===
Naive:    XX.XXX ms (median, 100 steps)
Promoted: XX.XXX ms (median, 100 steps)
Speedup:  X.XXx
Max |H_naive - H_promoted|: X.XXe-XX (should be ~0)
```

**把所有输出保存到文件：**
```bash
for N in 64 128 256 512; do
    echo "======= N=$N ======="
    ./gather_experiment $N 100 2>&1
done | tee gather_results.txt
```

## Step 5: 解读结果

| Speedup | 结论 | 下一步 |
|---------|------|--------|
| **> 1.3x** | Gather promotion 有价值 | 实现编译器 Pass (PromoteGatherMemory) |
| **1.1-1.3x** | 有一定收益但 L2 cache 已部分缓解 | 考虑在更大/更不规则的网格上测试 |
| **~1.0x** | L2 cache 完全覆盖 gather 需求 | 换方向：divergence reduction 或 layout 优化 |
| **< 1.0x** | Cooperative load 开销 > 收益 | 说明这个 kernel 不适合 gather promotion |

## Step 6: 补充实验（如果 Step 5 结果积极）

### 6a: 用真实 hydro-cal 网格测试

真实的非结构化网格比合成网格的 gather 更随机，L2 cache 效果更差，promotion 收益应该更大。需要额外写一个加载器（TODO）。

### 6b: 用 NCU profiler 分析瓶颈

```bash
ncu --set full ./gather_experiment 256 10 2>&1 | tee ncu_report.txt

# 关注指标：
# - l2_hit_rate: L2 cache 命中率（naive vs promoted 的差异）
# - sm__throughput: SM 利用率
# - dram__throughput: 显存带宽利用率
# - smsp__shared_ld_throughput: shared memory 读吞吐
```

## 文件说明

```
gather_experiment.cu     — 实验代码 (两个 kernel + benchmark harness)
RUN_EXPERIMENT.md        — 本文件
gather_results.txt       — 运行结果 (运行后生成)
```
