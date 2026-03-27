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
1. `spmd-opt` still cannot expose any project-specific passes, so AC-1/3/6/7/8 are not exercisable. In [spmd-opt.cpp](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/tools/spmd-opt/spmd-opt.cpp#L20) the pass include/registration is still a TODO, and [tools/spmd-opt/CMakeLists.txt](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/tools/spmd-opt/CMakeLists.txt#L1) links neither `SPMDTransforms` nor the conversion libs even though they exist in [lib/Transforms/CMakeLists.txt](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/lib/Transforms/CMakeLists.txt#L1) and [lib/Conversion/CMakeLists.txt](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/lib/Conversion/CMakeLists.txt#L1). Claude’s claim that AC-1/6 work is completed is false.

2. Milestones 3 and 4 are still unimplemented, not merely “next rounds.” The core files are explicit stubs: [NormalizeSPMD.cpp](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/lib/Transforms/NormalizeSPMD.cpp#L28), [MaterializeTilingAndMapping.cpp](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/lib/Transforms/MaterializeTilingAndMapping.cpp#L38), [PlanSPMDSchedule.cpp](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/lib/Transforms/PlanSPMDSchedule.cpp#L32), [SPMDToSCF.cpp](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/lib/Conversion/SPMDToSCF/SPMDToSCF.cpp#L32), [SPMDToOpenMP.cpp](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/lib/Conversion/SPMDToOpenMP/SPMDToOpenMP.cpp#L30), [AccessSummaryAnalysis.cpp](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/lib/Analysis/AccessSummaryAnalysis.cpp#L23), [PromotionPlanAnalysis.cpp](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/lib/Analysis/PromotionPlanAnalysis.cpp#L28), and [PromoteGroupMemory.cpp](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/lib/Transforms/PromoteGroupMemory.cpp#L49). AC-7 and AC-8 are unaddressed.

3. `VerifySPMDKernelSubset` does not enforce the legality contract it advertises. [VerifySPMDKernelSubset.cpp](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/lib/Transforms/VerifySPMDKernelSubset.cpp#L51) only rejects barriers, `gpu/omp/cf`, and non-global memref result types. It never checks function signatures, operands, calls, `scf.while`, or any general whitelist, so illegal S0/S1 kernels can pass. AC-3 is incomplete.

4. The lit suite is stale and partly nonfunctional. [invalid.mlir](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/test/SPMD/invalid.mlir#L1) never invokes `--verify-spmd-kernel-subset`, yet its final split expects that pass’s diagnostic at [lines 137-147](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/test/SPMD/invalid.mlir#L137). That error cannot come from [BarrierOp::verify](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/lib/IR/SPMDOps.cpp#L179), which explicitly accepts barriers nested under group-mapped `spmd.forall`. Also, [promotion.mlir](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/test/SPMD/promotion.mlir#L18) still uses the old pretty `spmd.forall (%ii, %jj) = ...` syntax, which does not match the current ODS assembly in [SPMDOps.td](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/include/spmd/IR/SPMDOps.td#L75). There is no `lower-to-openmp.mlir` at all under [test/SPMD](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/test/SPMD).

5. `ForallOp::verify` still misses one of its own documented checks. [SPMDOps.td](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/include/spmd/IR/SPMDOps.td#L63) says `spmd.mapping` must be a valid `LevelAttr`, but [SPMDOps.cpp](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/lib/IR/SPMDOps.cpp#L50) never validates it.

Implementation plan for Claude:
1. Add pass declarations/registration, link `spmd-opt` against `SPMDTransforms`, `SPMDToSCF`, and `SPMDToOpenMP`, and verify the custom pass flags appear in `spmd-opt --help`.
2. Finish AC-3 legality before lowering: implement a real whitelist in `VerifySPMDKernelSubset`, reject non-global memrefs in signatures/operands/results, and add the missing negative attr/subset tests.
3. Implement the CPU pipeline in order: `NormalizeSPMD`, `PlanSPMDSchedule`, `MaterializeTilingAndMapping`, `SPMDToSCF`, `SPMDToOpenMP`, then add the missing lowering tests and verify elementwise/reduction/stencil reach LLVM IR.
4. Implement `TargetDescriptor`, `AccessSummaryAnalysis`, `PromotionPlanAnalysis`, and `PromoteGroupMemory`, rewrite `promotion.mlir` to the current syntax, add no-promotion and oversized-footprint negatives, and verify the full object-file pipeline.

Goal Alignment Summary:
`ACs: 6/8 addressed | Forgotten items: 2 | Unjustified deferrals: 4`

I could not write [round-0-review-result.md](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/.humanize/rlcr/2026-03-27_08-24-48/round-0-review-result.md) or update [goal-tracker.md](/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/.humanize/rlcr/2026-03-27_08-24-48/goal-tracker.md) because the harness rejected all edit operations (`apply_patch` and shell writes).
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
2. Write your work summary into @/home/scratch.huanhuanc_gpu/spmd/spmd-dialect/.humanize/rlcr/2026-03-27_08-24-48/round-1-summary.md

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
