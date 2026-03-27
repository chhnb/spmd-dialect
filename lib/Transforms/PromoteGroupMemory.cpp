// PromoteGroupMemory.cpp
//
// Core innovation pass. Promotes tile-reusable memref slices to
// group address space, inserts cooperative copy loops and barriers.
//
// Input:  S2 IR with group-level spmd.forall containing lane-level
//         forall bodies that load from global memrefs
// Output: S2 IR with:
//           - memref.alloc in group addr space
//           - cooperative copy lane-level forall (load global -> store tile)
//           - spmd.barrier
//           - compute forall reading from tile instead of global
//
// MVP scope: read-only promotion only (no write-back).
// MVP pattern: stencil / repeated-load tiles with fixed halo.
//
// Algorithm (see design-v1.md §7):
//   For each group-level forall F:
//     1. Collect candidate memrefs M from AccessSummaryAnalysis
//     2. For each M: check legality (bounded footprint, reuse>1,
//        no cross-group conflict, fits in group mem, addr rewritable)
//     3. Check profitability (copy amortized, occupancy ok)
//     4. For profitable M:
//        a. Alloc tile buffer in group addr space
//        b. Insert cooperative copy loop before compute
//        c. Insert spmd.barrier after copy
//        d. Rewrite compute accesses to tile-local indices

#include "spmd/IR/SPMDOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::spmd;

namespace {
struct PromoteGroupMemoryPass
    : public PassWrapper<PromoteGroupMemoryPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PromoteGroupMemoryPass)

  StringRef getArgument() const override { return "promote-group-memory"; }
  StringRef getDescription() const override {
    return "Promote reused tile footprints to group memory with cooperative copy";
  }

  void runOnOperation() override {
    // TODO: implement PromoteGroupMemory
    // Phase 3 core implementation.
    //
    // Skeleton:
    //   getOperation().walk([&](ForallOp op) {
    //     if (!isGroupLevel(op)) return;
    //     auto plan = PromotionPlanAnalysis::compute(op, target);
    //     for (auto &record : plan.promotions)
    //       materializePromotion(op, record, rewriter);
    //   });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::spmd::createPromoteGroupMemoryPass() {
  return std::make_unique<PromoteGroupMemoryPass>();
}
