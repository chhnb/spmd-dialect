# Device-Side Graph Launch (Strategy S5) for GPU Simulation Benchmarks

## Goal Description

Implement a fifth CUDA execution strategy — **DeviceGraph** — that uses a GPU-resident single-block persistent scheduler kernel to launch per-step compute graphs via CUDA's device-side `cudaGraphLaunch` (fire-and-forget mode). This fills the performance gap between host-side Graph replay (~5μs/step overhead) and full Persistent kernel fusion (~0μs/step) without the cooperative launch grid size constraint that causes 13/21 cases to be N/A in the current Persistent strategy.

Inspired by Blink (arxiv 2604.07609), which demonstrated this pattern for CPU-free LLM inference serving.

## Acceptance Criteria

- AC-1: Technical validation of device-side graph launch on A100 (sm_80) + CUDA 12.6
  - Positive Tests (expected to PASS):
    - A minimal PoC (`device_graph_poc.cu`) compiles with `nvcc -arch=sm_80` and runs without runtime error on A100
    - Device-side `cudaGraphLaunch` with `cudaStreamGraphFireAndForget` completes successfully
    - Measured per-step device-side launch overhead is ≤ 5μs (target: ~2μs)
  - Negative Tests (expected to FAIL):
    - If sm_80 does not support fire-and-forget, the PoC returns a clear `cudaErrorNotSupported` (triggers fallback evaluation)
    - A PoC that uses host-side `cudaGraphLaunch` must show measurably higher overhead than device-side

- AC-2: Reusable device-side graph scheduler framework
  - Positive Tests (expected to PASS):
    - `persistent_scheduler.cuh` header is self-contained and can be included by any benchmark `.cu` file
    - `step_graph_capture()` successfully captures single-step graphs for cases with 1, 2, and 102 kernels/step
    - Completion polling correctly detects step completion via atomic flag + `__threadfence_system()`
    - Tail-launch recovery correctly resets after 120 fire-and-forget launches without deadlock or lost steps
  - Negative Tests (expected to FAIL):
    - A scheduler that does not implement tail-launch recovery must deadlock or error after 120 steps
    - A completion signal without `__threadfence_system()` should fail intermittently on multi-SM configurations

- AC-3: DeviceGraph timing data for 3 representative cases
  - Positive Tests (expected to PASS):
    - Heat2D (1 kernel/step) at sizes 128, 256, 512, 1024, 2048 — all produce numeric timing rows including sizes where Persistent is N/A
    - HydroF2 (2 kernels/step) at default (24020 cells) and 20w (207234 cells) — both produce numeric timing
    - StableFluids (102 kernels/step) at sizes 256 and 1024 — both produce numeric timing
    - Per-step overhead of DeviceGraph is measurably lower than host-side Graph for all sizes
  - Negative Tests (expected to FAIL):
    - A DeviceGraph implementation that falls back to host-side launch should produce overhead ≥ Graph baseline

- AC-4: Full 21-case integration into benchmark matrix
  - Positive Tests (expected to PASS):
    - All 21 cases have DeviceGraph rows in `matrix_results.csv` (numeric timing or documented N/A with reason)
    - All 13 previously Persistent-N/A cases have numeric DeviceGraph timing (no grid limit)
    - `run_matrix.py --strategies device_graph` generates reproducible CSV output
    - `plot_matrix.py` produces updated plots with DeviceGraph columns/series in all 4 output types
  - Negative Tests (expected to FAIL):
    - A DeviceGraph row with blank `overhead_pct` when a CUDA baseline exists
    - A regeneration of `matrix_results.csv` that drops or changes existing non-DeviceGraph rows

- AC-5: Correctness validation for DeviceGraph strategy
  - Positive Tests (expected to PASS):
    - For each case with DeviceGraph timing, running 100 steps produces output matching CUDA Sync within 5% relative error
    - `test_correctness.py` includes DeviceGraph in DSL backend consistency checks
  - Negative Tests (expected to FAIL):
    - An implementation that produces NaN, Inf, or >10% divergence from CUDA Sync reference

- AC-6: Paper-ready analysis artifacts
  - Positive Tests (expected to PASS):
    - 5-strategy launch overhead spectrum figure showing: Sync → Async → Graph → DeviceGraph → Persistent
    - Speedup table for DeviceGraph vs Graph on the 13 Persistent-N/A cases
    - Updated strategy selection decision tree with DeviceGraph branch
    - All timing data collected with ≥5 warmup + ≥10 timed runs per the benchmark contract
  - Negative Tests (expected to FAIL):
    - Analysis based on pre-warmup or insufficient timed runs

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
Full 21-case DeviceGraph implementation integrated into all benchmark infrastructure (runner, correctness, plots), with 5-strategy comparison data and paper-ready figures. Includes tail-launch recovery for >120-step runs and edge case handling for all kernel/step patterns.

### Lower Bound (Minimum Acceptable Scope)
Phase 0 PoC demonstrating device-side graph launch works on sm_80, plus timing data for 3 representative cases (Heat2D, HydroF2, StableFluids) showing measurable overhead reduction vs host-side Graph. If sm_80 is unsupported, a documented feasibility report with fallback recommendation.

### Allowed Choices
- Can use: CUDA 12.6 device graph API, fire-and-forget mode, cooperative groups for the scheduler, atomic-based completion signaling, CUDA events as alternative to polling
- Cannot use: host-side CPU involvement in the per-step scheduling loop (that would defeat the purpose), modifications to existing strategy implementations (S1-S4 must remain unchanged)

## Feasibility Hints and Suggestions

### Conceptual Approach

```
// Step 1: Capture one-step compute graph (host-side, once)
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
flux_kernel<<<grid, block, 0, stream>>>(args..., completion_flag, step_id);
update_kernel<<<grid, block, 0, stream>>>(args..., completion_flag, step_id);
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graph_exec, graph, ...);

// Step 2: Persistent scheduler kernel (runs on GPU, 1 block × 256 threads)
__global__ void device_scheduler(cudaGraphExec_t ge, int N, volatile int* done) {
    if (threadIdx.x == 0) {
        for (int s = 0; s < N; s++) {
            cudaGraphLaunch(ge, cudaStreamGraphFireAndForget);
            while (atomicAdd((int*)done, 0) <= s) { /* poll */ }
            if (s % 119 == 0 && s > 0) { /* tail-launch recovery */ }
        }
    }
}

// Step 3: Last kernel in step signals completion
__global__ void update_kernel(..., volatile int* done, int step) {
    // ... compute ...
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __threadfence_system();
        atomicAdd((int*)done, 1);
    }
}
```

### Relevant References
- `benchmark/overhead_solutions.cu` — current 4-strategy benchmark pattern (Graph capture at lines 78-87)
- `benchmark/hydro_osher_benchmark.cu` — persistent kernel with cooperative launch (lines 690-724)
- `benchmark/run_matrix.py` — strategy dispatch and timing integration
- `benchmark/plot_matrix.py` — analysis pipeline (needs DeviceGraph column)
- `benchmark/test_correctness.py` — correctness validation framework
- Blink paper (arxiv 2604.07609) — device-side graph scheduling architecture, tail-launch recovery

## Dependencies and Sequence

### Milestone 1: Technical Validation (Phase 0)
- Phase A: Implement `device_graph_poc.cu` — minimal Heat2D with device-side graph launch
- Phase B: Measure overhead, confirm sm_80 support
- Phase C: If sm_80 unsupported, evaluate fallback (device-side stream launch, conditional graph nodes)
- **Gate**: AC-1 must pass before proceeding

### Milestone 2: Core Framework + 3 Cases (Phase 1)
- Phase A: Extract scheduler into `persistent_scheduler.cuh`
- Phase B: Implement Heat2D, HydroF2, StableFluids device-graph benchmarks
- Phase C: 5-strategy comparison for these 3 cases
- Depends on: Milestone 1 (confirmed API support)

### Milestone 3: Full Integration (Phase 2)
- Phase A: Extend to all 21 cases in `device_graph_benchmark.cu`
- Phase B: Integrate into `run_matrix.py`, `plot_matrix.py`, `test_correctness.py`
- Phase C: Regenerate all CSV/plot artifacts
- Depends on: Milestone 2 (working framework)

### Milestone 4: Paper Artifacts (Phase 3)
- Phase A: Generate complete 13-strategy × 21-case matrix
- Phase B: Create analysis figures (overhead spectrum, speedup table, decision tree)
- Depends on: Milestone 3 (full data)

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- Use descriptive, domain-appropriate naming (e.g., `device_graph_scheduler`, `step_graph_capture`, not `phase1_impl`)
- Follow existing benchmark `.cu` file conventions for output format: `[StrategyName] median=X.XX ms, Y.YY us/step`

### Key Technical Constraints
- CUDA 12.6, sm_80 (A100), nvcc with `-rdc=true -lcudadevrt`
- Fire-and-forget outstanding launch limit: 120 (must implement tail-launch recovery for runs >120 steps)
- Scheduler kernel: exactly 1 block, ≤256 threads (minimal GPU resource usage)
- Completion signaling: `__threadfence_system()` required before atomic write for cross-kernel visibility
- Existing strategies S1-S4 must not be modified

---

## Original Design Draft

### Background
- N×M GPU simulation benchmark matrix: 12 strategies × 21 cases, 270 rows CSV on A100
- Persistent strategy uses `cudaLaunchCooperativeKernel` + `grid_sync()`, 13/21 cases N/A at large mesh
- Blink (arxiv 2604.07609): device-side CUDA graph fire-and-forget, ~2μs launch, no grid limit
- Environment: CUDA 12.6 + A100 sm_80

### Core Architecture
1. Step graph capture: `cudaStreamBeginCapture` → kernels → `cudaStreamEndCapture`
2. Persistent scheduler (1 block, 256 threads): device-side `cudaGraphLaunch` per step + poll completion
3. Completion: `atomicAdd(flag)` + `__threadfence_system()`
4. Tail-launch recovery every 120 launches

### Expected Contribution
Fifth execution strategy filling the gap between Graph (5μs) and Persistent (0μs), without grid size constraints. Cross-domain method migration from LLM serving to simulation kernel fusion.

### Risks
| Risk | Likelihood | Mitigation |
|---|---|---|
| sm_80 doesn't support fire-and-forget | Medium | Phase 0 validation; fallback to device-side stream launch |
| Actual overhead > 2μs | Low | Blink verified on A100 |
| 120 launch limit complexity | Low | Tail-launch recovery |
| Polling overhead significant | Medium | CUDA events as alternative |
