// PlanSPMDSchedule.cpp
//
// Attaches schedule hints (tile_sizes, mapping, memory_policy) to
// spmd.forall ops based on heuristics. Transitions S0 -> S1.
//
// Heuristics:
//   - A forall with no parent forall is "outermost" → mapping=group,
//     tile_sizes=[32, 32, ...], memory_policy=prefer_group.
//   - A forall whose parent is another forall → mapping=lane.
//   - Forall ops that already have all three hints are left unchanged.

#include "spmd/Analysis/TargetDescriptor.h"
#include "spmd/IR/SPMDOps.h"
#include "spmd/Transforms/SPMDPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::spmd;

static const TargetDescriptor kTarget = TargetDescriptor::cpuDefault();

namespace {
struct PlanSPMDSchedulePass
    : public PassWrapper<PlanSPMDSchedulePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanSPMDSchedulePass)

  StringRef getArgument() const override { return "plan-spmd-schedule"; }
  StringRef getDescription() const override {
    return "Attach tile/mapping/memory_policy hints to spmd.forall (S0 -> S1)";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    func.walk([&](ForallOp forall) {
      // Already fully annotated — skip.
      bool hasMapping = forall->hasAttr("spmd.mapping");
      bool hasTileSizes = forall->hasAttr("spmd.tile_sizes");
      bool hasMemPolicy = forall->hasAttr("spmd.memory_policy");
      if (hasMapping && hasTileSizes && hasMemPolicy)
        return;

      unsigned rank = forall.getRank();
      MLIRContext *ctx = &getContext();

      // Determine nesting: if the parent op (or its parent) is a ForallOp,
      // this is an inner (lane-level) forall.
      bool isNested = false;
      Operation *parent = forall->getParentOp();
      while (parent && !isa<func::FuncOp>(parent)) {
        if (isa<ForallOp>(parent)) {
          isNested = true;
          break;
        }
        parent = parent->getParentOp();
      }

      OpBuilder b(forall);
      if (isNested) {
        // Lane-level: add mapping=lane if missing.
        if (!hasMapping)
          forall->setAttr("spmd.mapping",
                          LevelAttr::get(ctx, LevelKind::Lane));
        // No tile_sizes or memory_policy for lane-level.
      } else {
        // Outermost (group-level) forall.
        if (!hasMapping)
          forall->setAttr("spmd.mapping",
                          LevelAttr::get(ctx, LevelKind::Group));

        if (!hasTileSizes) {
          // Default: 4*simdWidth for the first dim, simdWidth for subsequent.
          SmallVector<int64_t> tileSizes;
          for (unsigned d = 0; d < rank; ++d)
            tileSizes.push_back(d == 0 ? 4 * kTarget.simdWidth
                                       : kTarget.simdWidth);
          forall->setAttr("spmd.tile_sizes",
                          DenseI64ArrayAttr::get(ctx, tileSizes));
        }

        if (!hasMemPolicy)
          forall->setAttr(
              "spmd.memory_policy",
              MemoryPolicyAttr::get(ctx, MemoryPolicyKind::PreferGroup));
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::spmd::createPlanSPMDSchedulePass() {
  return std::make_unique<PlanSPMDSchedulePass>();
}

void mlir::spmd::registerPlanSPMDSchedulePass() {
  PassRegistration<PlanSPMDSchedulePass>();
}
