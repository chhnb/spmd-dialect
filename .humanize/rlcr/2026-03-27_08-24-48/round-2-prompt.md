Your work is not finished. Read and execute the below with ultrathink.

## Original Implementation Plan

**IMPORTANT**: Before proceeding, review the original plan you are implementing:
@docs/plan.md

This plan contains the full scope of work and requirements. Ensure your work aligns with this plan.

---

For all tasks that need to be completed, please use the Task system (TaskCreate, TaskUpdate, TaskList) to track each item in order of importance.
You are strictly prohibited from only addressing the most important issues - you MUST create Tasks for ALL discovered issues and attempt to resolve each one.

---
Below is Codex's review result:
<!-- CODEX's REVIEW RESULT START -->
**Findings**
1. Build credibility is broken by the analysis sources themselves. [`AccessSummary`](#/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/include/spmd/Analysis/AccessSummaryAnalysis.h#L13) is redefined in [`AccessSummaryAnalysis.cpp`](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/lib/Analysis/AccessSummaryAnalysis.cpp#L32), and [`PromotionRecord`](#/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/include/spmd/Analysis/PromotionPlanAnalysis.h#L14) is redefined again in [`PromotionPlanAnalysis.cpp`](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/lib/Analysis/PromotionPlanAnalysis.cpp#L29). Until that is fixed, Claude’s AC-1/AC-6 completion claim is not supportable.

2. `computeAccessSummaries()` is functionally broken even ignoring the redefinition. It inserts `memrefToIdx[mr]` before checking whether the memref was seen at [`AccessSummaryAnalysis.cpp:116`](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/lib/Analysis/AccessSummaryAnalysis.cpp#L116), then immediately indexes `result[idx]` at [`AccessSummaryAnalysis.cpp:127`](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/lib/Analysis/AccessSummaryAnalysis.cpp#L127) without ever creating a summary record for the first load. That blocks AC-8.

3. The required `materialize -> promote` pipeline cannot work as implemented. [`MaterializeTilingAndMapping.cpp:79`](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/lib/Transforms/MaterializeTilingAndMapping.cpp#L79) consumes `spmd.tile_sizes` and only preserves `spmd.mapping`/`spmd.memory_policy`, while [`PromoteGroupMemory.cpp:145`](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/lib/Transforms/PromoteGroupMemory.cpp#L145) still requires `spmd.tile_sizes` on the group forall and silently skips otherwise. That contradicts the Milestone 4 pipeline in [`docs/plan.md:219`](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/docs/plan.md#L219).

4. AC-7 and AC-8 end-to-end CPU lowering are still incomplete. [`SPMDToOpenMP.cpp:67`](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/lib/Conversion/SPMDToOpenMP/SPMDToOpenMP.cpp#L67) does not build `omp.parallel`/`omp.wsloop`; it only lowers barriers and removes the mapping attr, despite [`docs/plan.md:216`](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/docs/plan.md#L216). The supposed regression test [`lower-to-openmp.mlir:7`](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/test/SPMD/lower-to-openmp.mlir#L7) never invokes OpenMP lowering or LLVM translation, and [`lit.cfg.py:31`](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/test/lit.cfg.py#L31) does not register `mlir-translate` or `llc`.

5. `PromoteGroupMemory` emits unconditional halo/tail copies, which is semantically wrong for boundaries and partial tiles. The copy loop does direct global loads at [`PromoteGroupMemory.cpp:196`](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/lib/Transforms/PromoteGroupMemory.cpp#L196) with no bounds guard, while the design’s required S2 form explicitly masks the copy before the barrier at [`design-v1.md:474`](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/docs/design-v1.md#L474).

**Implementation Plan**
1. Make the tree buildable first: remove the duplicate struct definitions from the analysis `.cpp` files, keep the public headers as the single source of truth, and fix `computeAccessSummaries()` so it creates one summary record per memref before use.
2. Restore the planned S1->S2->promotion contract: either preserve `spmd.tile_sizes` on the materialized group forall or derive promotion footprints from the materialized loop nest, then rerun promotion after materialization exactly as the plan requires.
3. Finish the CPU path as specified: implement real `SPMDToOpenMP` lowering to `omp.parallel + omp.wsloop`, keep `NormalizeSPMD` aligned with [`docs/plan.md:213`](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/docs/plan.md#L213), and introduce the missing `TargetDescriptor` required by [`docs/plan.md:223`](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/docs/plan.md#L223) and [`design-v1.md:398`](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/docs/design-v1.md#L398).
4. Fix promotion correctness: guard cooperative copy loads/stores for halo and tail tiles, then add the missing oversized-footprint negative and the required end-to-end `mlir-translate`/`llc` tests.

**Goal Alignment**
`ACs: 8/8 addressed | Forgotten items: 1 | Unjustified deferrals: 4`

The forgotten tracker item is `PlanSPMDSchedule`, which is required by the original plan but is not listed in Active/Completed/Deferred in [`goal-tracker.md:132`](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/.humanize/rlcr/2026-03-27_08-24-48/goal-tracker.md#L132). Claude’s Goal Tracker Update Request should be rejected; the claimed AC completions are not justified by the current source state.

I could not write [`round-1-review-result.md`](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/.humanize/rlcr/2026-03-27_08-24-48/round-1-review-result.md) or update [`goal-tracker.md`](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/.humanize/rlcr/2026-03-27_08-24-48/goal-tracker.md) because this harness rejected every edit path I attempted (`apply_patch` and direct shell writes).
<!-- CODEX's REVIEW RESULT  END  -->
---

## Goal Tracker Reference (READ-ONLY after Round 0)

Before starting work, **read** @/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/.humanize/rlcr/2026-03-27_08-24-48/goal-tracker.md to understand:
- The Ultimate Goal and Acceptance Criteria you're working toward
- Which tasks are Active, Completed, or Deferred
- Any Plan Evolution that has occurred
- Open Issues that need attention

**IMPORTANT**: You CANNOT directly modify goal-tracker.md after Round 0.
If you need to update the Goal Tracker, include a "Goal Tracker Update Request" section in your summary (see below).

---

Note: You MUST NOT try to exit by lying, editing loop state files, or executing `cancel-rlcr-loop`.

After completing the work, please:
0. If the `code-simplifier` plugin is installed, use it to review and optimize your code. Invoke via: `/code-simplifier`, `@agent-code-simplifier`, or `@code-simplifier:code-simplifier (agent)`
1. Commit your changes with a descriptive commit message
2. Write your work summary into @/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/.humanize/rlcr/2026-03-27_08-24-48/round-2-summary.md

**If Goal Tracker needs updates**, include this section in your summary:
```markdown
## Goal Tracker Update Request

### Requested Changes:
- [E.g., "Mark Task X as completed with evidence: tests pass"]
- [E.g., "Add to Open Issues: discovered Y needs addressing"]
- [E.g., "Plan Evolution: changed approach from A to B because..."]
- [E.g., "Defer Task Z because... (impact on AC: none/minimal)"]

### Justification:
[Explain why these changes are needed and how they serve the Ultimate Goal]
```

Codex will review your request and update the Goal Tracker if justified.
