# Device-Side Graph Launch: Strategy S5 for GPU Simulation Benchmarks

## Background

We have an N×M GPU simulation benchmark matrix: 12 strategies × 21 cases, 270 rows CSV on A100.

Current Persistent strategy uses `cudaLaunchCooperativeKernel` + `grid_sync()`, which requires all blocks to be simultaneously resident. This causes 13/21 cases at large mesh sizes to be N/A (grid exceeds cooperative launch limit).

Blink (arxiv 2604.07609) demonstrates that device-side CUDA graph fire-and-forget launch achieves ~2μs GPU-side launch overhead with no grid size constraint, by using a single-block persistent scheduler kernel that launches compute graphs from the GPU.

Our environment: CUDA 12.6 + NVIDIA A100 80GB PCIe (sm_80).

## Goal

Implement a fifth CUDA execution strategy — **DeviceGraph** — that uses a GPU-resident scheduler kernel to launch step graphs via device-side `cudaGraphLaunch` (fire-and-forget mode). This fills the performance gap between host-side Graph replay (~5μs/step) and full Persistent fusion (~0μs/step) without cooperative launch grid size constraints.

## Core Architecture

1. **Step graph capture**: `cudaStreamBeginCapture` → launch all kernels for one timestep → `cudaStreamEndCapture` → `cudaGraphInstantiate`
2. **Persistent scheduler kernel** (1 block, 256 threads): loops N steps, each step does device-side `cudaGraphLaunch(step_graph_exec, cudaStreamGraphFireAndForget)`, then polls a completion flag
3. **Completion signaling**: last compute kernel in each step writes `atomicAdd(completion_flag, 1)` + `__threadfence_system()` to notify the scheduler
4. **Tail-launch recovery**: every 120 fire-and-forget launches, issue a tail launch that atomically replaces the current graph execution (Blink's mechanism to handle the 120 outstanding launch limit)

## Implementation Phases

### Phase 0: Technical Validation
- Verify device-side `cudaGraphLaunch` fire-and-forget works on A100 sm_80 with CUDA 12.6
- Minimal PoC: persistent scheduler + device graph launch for Heat2D (1 kernel/step)
- Measure actual device-side launch overhead (expect ~2μs, compare with host ~5μs)
- If sm_80 doesn't support fire-and-forget, evaluate alternatives (device-side stream launch, conditional nodes)

### Phase 1: Core Implementation
- Implement reusable device-side graph scheduler framework:
  - `persistent_scheduler.cuh` — scheduler kernel (1 block, 256 threads)
  - `step_graph_capture()` — captures single-step compute graph (supports 1-N kernels/step)
  - `poll_completion()` — GPU-side completion detection (atomic flag + threadfence)
  - `tail_launch_recovery()` — reset mechanism for 120-launch limit
- Implement on 3 representative cases:
  - Heat2D (1 kernel/step, small mesh — validate small-kernel advantage)
  - HydroF2 (2 kernels/step, 20w mesh — end-to-end application scenario)
  - StableFluids (102 kernels/step — extreme multi-kernel case)
- 5-strategy comparison: Sync, Async, Graph, Persistent, DeviceGraph

### Phase 2: Full Case Extension + Integration
- Implement all 21 cases in `device_graph_benchmark.cu`
- Focus on the 13 Persistent-N/A cases: Heat2D 512/1024/2048, GrayScott 512/1024, Jacobi2D 4096, StableFluids 1024, LU 512/1024, ADI 256/512, GramSchmidt 128/256
- Integrate into `run_matrix.py` (new strategy column "DeviceGraph")
- Update `plot_matrix.py` (add DeviceGraph to all analysis outputs)
- Update `test_correctness.py` (DeviceGraph correctness validation)

### Phase 3: Paper Data and Analysis
- Generate complete N×M matrix (13 strategies × 21 cases)
- Key figures:
  - 5-strategy launch overhead spectrum: host-sync(15μs) → host-async(6μs) → host-graph(5μs) → device-graph(2μs) → persistent(0μs)
  - DeviceGraph vs Graph speedup for the 13 Persistent-N/A cases
  - Updated strategy selection decision tree with DeviceGraph branch
  - Overhead% vs compute time scatter plot (5 CUDA strategies)
- Paper text: Related Work citing Blink, Contribution describing cross-domain migration, Results presenting data

## Expected Results

| Case Category | Current State | With DeviceGraph |
|---|---|---|
| Small mesh + Persistent OK | Persistent optimal (0μs) | DeviceGraph slightly worse (~2μs) |
| Large mesh + Persistent N/A (13 cases) | Only Graph available (5μs) | DeviceGraph new best (~2μs) |
| Multi-kernel/step | Graph optimal | DeviceGraph comparable |

## Contribution

"We introduce device-side graph scheduling as a fifth execution strategy for GPU simulation kernels, inspired by Blink's CPU-free LLM serving. Unlike cooperative persistent kernels (limited by max resident block count, 13/21 cases N/A), device-side graph launch uses a single-block GPU-resident scheduler with ~2μs overhead — filling the gap between host-side graph replay (~5μs) and full persistent fusion (~0μs) without grid size constraints."

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| sm_80 doesn't support fire-and-forget | Medium (may need sm_90) | Phase 0 validation; fallback to device-side stream launch |
| Actual overhead > 2μs | Low | Blink verified on A100 |
| 120 launch limit complexity | Low | Tail-launch recovery (Blink's proven mechanism) |
| Polling overhead significant | Medium | Can use cuda events as alternative |
