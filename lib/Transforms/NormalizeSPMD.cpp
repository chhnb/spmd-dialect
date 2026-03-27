// NormalizeSPMD.cpp
//
// Normalizes spmd.forall to a canonical form:
//   - 0-based lower bounds (lb = 0) where possible
//   - unit step where possible
//   - folds single-iteration dimensions
//   - converts non-rectangular domains to rectangular + spmd.if guard
//
// This pass runs on S0 and prepares IR for analysis and scheduling.

#include "spmd/IR/SPMDOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::spmd;

namespace {

// Pattern: normalize lb to 0 and step to 1 by shifting/scaling the
// induction variable. Replaces %i in [lb, ub) step s with
// %i' in [0, (ub-lb+s-1)/s) step 1, and adds %i = lb + %i' * s.
struct NormalizeForallBounds : public OpRewritePattern<ForallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ForallOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: implement normalization
    // Stub for Phase 2.
    return failure();
  }
};

struct NormalizeSPMDPass
    : public PassWrapper<NormalizeSPMDPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeSPMDPass)

  StringRef getArgument() const override { return "normalize-spmd"; }
  StringRef getDescription() const override {
    return "Normalize spmd.forall to canonical 0-based unit-step form";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<NormalizeForallBounds>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::spmd::createNormalizeSPMDPass() {
  return std::make_unique<NormalizeSPMDPass>();
}
