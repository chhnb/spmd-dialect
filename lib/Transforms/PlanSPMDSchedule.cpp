// PlanSPMDSchedule.cpp
//
// Attaches schedule hints (tile_sizes, mapping, memory_policy) to
// spmd.forall ops based on heuristics and/or user-provided hints.
// Transitions S0 -> S1.
//
// Strategy:
//   1. If forall already has all schedule attrs, skip.
//   2. Otherwise, query TargetDescriptor for defaults:
//      - tile_sizes: e.g., {32, 8} for 2D, scaled to group size
//      - mapping: outermost -> group, innermost -> lane
//      - memory_policy: prefer_group if access summary shows reuse > 1
//   3. Write attrs onto op.

#include "spmd/IR/SPMDOps.h"

#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::spmd;

namespace {
struct PlanSPMDSchedulePass
    : public PassWrapper<PlanSPMDSchedulePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanSPMDSchedulePass)

  StringRef getArgument() const override { return "plan-spmd-schedule"; }
  StringRef getDescription() const override {
    return "Attach tile/mapping/memory_policy hints to spmd.forall (S0 -> S1)";
  }

  void runOnOperation() override {
    // TODO: implement heuristic schedule planning
    // Stub for Phase 2.
  }
};
} // namespace

std::unique_ptr<Pass> mlir::spmd::createPlanSPMDSchedulePass() {
  return std::make_unique<PlanSPMDSchedulePass>();
}
