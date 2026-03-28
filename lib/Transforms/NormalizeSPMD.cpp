// NormalizeSPMD.cpp
//
// Normalizes spmd.forall to a canonical form:
//   - For each dimension where lb == 0 and step == 1: already canonical.
//   - For constant lb and step: shift iv → (lb + iv_new * step) so new forall
//     runs over [0, ceildiv(ub-lb, step)) with step 1.
//   - Folds single-iteration dimensions (ub - lb <= step → replaces iv with lb).
//
// This pass runs on S0 and prepares IR for analysis and scheduling.

#include "spmd/IR/SPMDOps.h"
#include "spmd/Transforms/SPMDPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::spmd;

namespace {

// Pattern: for each dimension with a non-zero constant lb or non-unit constant
// step, rewrite the body to use iv' = lb + iv_new * step and run the new
// forall over [0, ceildiv(ub-lb, step)) step 1.
//
// Only fires when ALL lbs and steps are arith.constant.index ops, so the
// transformation is always profitable (no runtime overhead).
struct NormalizeForallBounds : public OpRewritePattern<ForallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ForallOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned rank = op.getRank();

    // Check that all lbs and steps are constant — required to normalize.
    SmallVector<int64_t> lbVals, stepVals;
    for (unsigned d = 0; d < rank; ++d) {
      auto lbCst = op.getLowerBounds()[d].getDefiningOp<arith::ConstantIndexOp>();
      auto stCst = op.getSteps()[d].getDefiningOp<arith::ConstantIndexOp>();
      if (!lbCst || !stCst)
        return failure(); // dynamic — cannot normalize statically
      lbVals.push_back(lbCst.value());
      stepVals.push_back(stCst.value());
    }

    // Check if already fully canonical (all lb=0, all step=1).
    bool alreadyCanonical = true;
    for (unsigned d = 0; d < rank; ++d) {
      if (lbVals[d] != 0 || stepVals[d] != 1) {
        alreadyCanonical = false;
        break;
      }
    }
    if (alreadyCanonical)
      return failure();

    // Compute new upper bounds: ceildiv(ub - lb, step) for each dimension.
    SmallVector<Value> newLbs, newUbs, newSteps;
    for (unsigned d = 0; d < rank; ++d) {
      Value lb   = op.getLowerBounds()[d];
      Value ub   = op.getUpperBounds()[d];
      Value step = op.getSteps()[d];

      // new_ub = ceildiv(ub - lb, step)
      Value diff = rewriter.create<arith::SubIOp>(loc, ub, lb);
      Value newUb = rewriter.create<arith::CeilDivSIOp>(loc, diff, step);
      newLbs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      newUbs.push_back(newUb);
      newSteps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
    }

    // Create the new normalized forall op (without body — we'll fill it).
    auto newForall = rewriter.create<ForallOp>(loc, newLbs, newUbs, newSteps);

    // Copy all attributes from old op onto new op (preserves schedule hints).
    for (auto attr : op->getAttrs()) {
      // operandSegmentSizes must be kept as-is (same shape).
      newForall->setAttr(attr.getName(), attr.getValue());
    }

    // Inside the new body, add the iv remapping:
    //   original_iv[d] = lb[d] + new_iv[d] * step[d]
    // Then inline the old body.
    Block &newBody = newForall.getBody().front();
    rewriter.setInsertionPointToStart(&newBody);

    SmallVector<Value> remappedIvs;
    for (unsigned d = 0; d < rank; ++d) {
      Value newIv = newForall.getInductionVar(d);
      if (lbVals[d] == 0 && stepVals[d] == 1) {
        remappedIvs.push_back(newIv); // already canonical for this dim
      } else {
        // original_iv = lb + new_iv * step
        Value lb   = op.getLowerBounds()[d];
        Value step = op.getSteps()[d];
        Value scaled = rewriter.create<arith::MulIOp>(loc, newIv, step);
        Value shifted = rewriter.create<arith::AddIOp>(loc, lb, scaled);
        remappedIvs.push_back(shifted);
      }
    }

    // Inline old body block into new body, replacing old IVs with remapped vals.
    Block &oldBody = op.getBody().front();
    rewriter.inlineBlockBefore(&oldBody, newBody.getTerminator(), remappedIvs);

    // The inlined block contains the old spmd.yield at the end — erase it
    // (newForall's body already has its own spmd.yield terminator).
    Operation *inlinedYield = newBody.getTerminator()->getPrevNode();
    if (inlinedYield && isa<YieldOp>(inlinedYield))
      rewriter.eraseOp(inlinedYield);

    rewriter.eraseOp(op);
    return success();
  }
};

// Pattern: fold or eliminate dimensions with known iteration counts.
//
// Two sub-cases, both requiring compile-time constant bounds/step:
//   Zero-trip  (ub <= lb)             : body never executes → erase the forall.
//   Single-trip (lb < ub <= lb+step)  : body executes once with IV = lb →
//                                       replace IV with constant, drop dim.
//
// Zero-trip check is done first: if ANY dimension has ub <= lb, the entire
// N-D forall executes zero times and can be erased outright (body not inlined).
struct FoldSingleIterationDims : public OpRewritePattern<ForallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ForallOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned rank = op.getRank();

    // Pass 1: detect zero-trip dimensions.
    // If any constant-bound dimension has ub <= lb the forall body never runs.
    for (unsigned d = 0; d < rank; ++d) {
      auto lbCst = op.getLowerBounds()[d].getDefiningOp<arith::ConstantIndexOp>();
      auto ubCst = op.getUpperBounds()[d].getDefiningOp<arith::ConstantIndexOp>();
      if (!lbCst || !ubCst)
        continue;
      if (ubCst.value() <= lbCst.value()) {
        // Zero-trip: erase forall without executing the body.
        rewriter.eraseOp(op);
        return success();
      }
    }

    // Pass 2: detect single-trip dimensions (lb < ub && ub <= lb + step).
    SmallVector<bool>    isSingle(rank, false);
    SmallVector<int64_t> singleVals(rank, 0);
    bool hasSingle = false;

    for (unsigned d = 0; d < rank; ++d) {
      auto lbCst = op.getLowerBounds()[d].getDefiningOp<arith::ConstantIndexOp>();
      auto ubCst = op.getUpperBounds()[d].getDefiningOp<arith::ConstantIndexOp>();
      auto stCst = op.getSteps()[d].getDefiningOp<arith::ConstantIndexOp>();
      if (!lbCst || !ubCst || !stCst)
        continue;
      int64_t lb = lbCst.value(), ub = ubCst.value(), step = stCst.value();
      // lb < ub already guaranteed by pass 1 for constant dims.
      if (ub <= lb + step) { // exactly one iteration: IV == lb
        isSingle[d] = true;
        singleVals[d] = lb;
        hasSingle = true;
      }
    }
    if (!hasSingle)
      return failure();

    SmallVector<Value> newLbs, newUbs, newSteps;
    for (unsigned d = 0; d < rank; ++d) {
      if (!isSingle[d]) {
        newLbs.push_back(op.getLowerBounds()[d]);
        newUbs.push_back(op.getUpperBounds()[d]);
        newSteps.push_back(op.getSteps()[d]);
      }
    }

    Block &oldBody = op.getBody().front();

    if (newLbs.empty()) {
      // All dims fold: inline body directly before op, with const IVs.
      SmallVector<Value> constIvs;
      rewriter.setInsertionPoint(op);
      for (unsigned d = 0; d < rank; ++d)
        constIvs.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, singleVals[d]));
      rewriter.inlineBlockBefore(&oldBody, op, constIvs);
      // Erase the inlined spmd.yield (now just before op).
      Operation *prev = op->getPrevNode();
      if (prev && isa<YieldOp>(prev))
        rewriter.eraseOp(prev);
      rewriter.eraseOp(op);
      return success();
    }

    // Create a lower-rank forall and remap IVs.
    auto newForall = rewriter.create<ForallOp>(loc, newLbs, newUbs, newSteps);
    // Copy user-defined attrs (mapping, tile_sizes, memory_policy, etc.) but
    // skip operandSegmentSizes — the new op already has the correct value set
    // by ForallOp::create based on the actual (lower) rank.
    StringRef segSizesKey = "operandSegmentSizes";
    for (auto attr : op->getAttrs())
      if (attr.getName() != segSizesKey)
        newForall->setAttr(attr.getName(), attr.getValue());

    Block &newBody = newForall.getBody().front();
    rewriter.setInsertionPointToStart(&newBody);

    unsigned newDimIdx = 0;
    SmallVector<Value> remappedIvs;
    for (unsigned d = 0; d < rank; ++d) {
      if (isSingle[d])
        remappedIvs.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, singleVals[d]));
      else
        remappedIvs.push_back(newForall.getInductionVar(newDimIdx++));
    }

    rewriter.inlineBlockBefore(&oldBody, newBody.getTerminator(), remappedIvs);
    Operation *inlinedYield = newBody.getTerminator()->getPrevNode();
    if (inlinedYield && isa<YieldOp>(inlinedYield))
      rewriter.eraseOp(inlinedYield);

    rewriter.eraseOp(op);
    return success();
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
    patterns.add<NormalizeForallBounds, FoldSingleIterationDims>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::spmd::createNormalizeSPMDPass() {
  return std::make_unique<NormalizeSPMDPass>();
}

void mlir::spmd::registerNormalizeSPMDPass() {
  PassRegistration<NormalizeSPMDPass>();
}
