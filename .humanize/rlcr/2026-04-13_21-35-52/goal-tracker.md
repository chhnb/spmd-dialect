# Goal Tracker

<!--
This file tracks the ultimate goal, acceptance criteria, and plan evolution.
It prevents goal drift by maintaining a persistent anchor across all rounds.

RULES:
- IMMUTABLE SECTION: Do not modify after initialization
- MUTABLE SECTION: Update each round, but document all changes
- Every task must be in one of: Active, Completed, or Deferred
- Deferred items require explicit justification
-->

## IMMUTABLE SECTION
<!-- Do not modify after initialization -->

### Ultimate Goal

Implement a fifth CUDA execution strategy — **DeviceGraph** — that uses a GPU-resident single-block persistent scheduler kernel to launch per-step compute graphs via CUDA's device-side `cudaGraphLaunch` (fire-and-forget mode). This fills the performance gap between host-side Graph replay (~5μs/step overhead) and full Persistent kernel fusion (~0μs/step) without the cooperative launch grid size constraint that causes 13/21 cases to be N/A in the current Persistent strategy.

Inspired by Blink (arxiv 2604.07609), which demonstrated this pattern for CPU-free LLM inference serving.

## Acceptance Criteria

### Acceptance Criteria
<!-- Each criterion must be independently verifiable -->
<!-- Claude must extract or define these in Round 0 -->


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

---

## MUTABLE SECTION
<!-- Update each round with justification for changes -->

### Plan Version: 1 (Updated: Round 0)

#### Plan Evolution Log
<!-- Document any changes to the plan with justification -->
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initial plan | - | - |

#### Active Tasks
<!-- Map each task to its target Acceptance Criterion -->
| Task | Target AC | Status | Notes |
|------|-----------|--------|-------|
| Implement device_graph_poc.cu PoC | AC-1 | pending | Minimal Heat2D with device-side graph launch |
| Measure device-side launch overhead | AC-1 | pending | Compare with host-side Graph |
| Implement persistent_scheduler.cuh | AC-2 | pending | Reusable scheduler framework |
| Implement tail-launch recovery | AC-2 | pending | Handle 120-launch limit |
| Benchmark Heat2D with DeviceGraph | AC-3 | pending | 1 kernel/step, sizes 128-2048 |
| Benchmark HydroF2 with DeviceGraph | AC-3 | pending | 2 kernels/step, default+20w |
| Benchmark StableFluids with DeviceGraph | AC-3 | pending | 102 kernels/step |
| Extend to all 21 cases | AC-4 | pending | device_graph_benchmark.cu |
| Integrate into run_matrix.py | AC-4 | pending | New strategy column |
| Update plot_matrix.py | AC-4 | pending | DeviceGraph in all plots |
| DeviceGraph correctness validation | AC-5 | pending | 100-step comparison vs CUDA Sync |
| Generate paper-ready figures | AC-6 | pending | 5-strategy spectrum, speedup table |

### Completed and Verified
<!-- Only move tasks here after Codex verification -->
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|

### Explicitly Deferred
<!-- Items here require strong justification -->
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|

### Open Issues
<!-- Issues discovered during implementation -->
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|
