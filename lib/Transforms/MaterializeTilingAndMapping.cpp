// MaterializeTilingAndMapping.cpp
//
// Transforms S1 IR (with schedule hints as attrs) into S2 IR
// (explicit nested spmd.forall with mapping labels). S1→S2.
//
// Input:  spmd.forall with spmd.tile_sizes=[t0,t1,...] and spmd.mapping=group
// Output: outer group forall (step=tile_size) wrapping inner lane forall
//         (step=1, bounds=[0,tile_size)) with boundary guard.
//
// Example (1D):
//   spmd.forall (lb,) to (ub,) step (s,) {tile_sizes=[T], mapping=group} {
//     body(%i)
//   }
//   =>
//   spmd.forall (lb,) to (ub,) step (T,) {mapping=group} {
//   ^bb0(%ii: index):
//     spmd.forall (0,) to (T,) step (1,) {mapping=lane} {
//     ^bb0(%ti: index):
//       %i = ii + ti
//       spmd.if (%i < ub) : () { body(%i)  spmd.yield } else {}
//       spmd.yield
//     }
//     spmd.yield
//   }

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
struct MaterializeTiledForall : public OpRewritePattern<ForallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ForallOp op,
                                PatternRewriter &rewriter) const override {
    // Only materialize if tile_sizes hint is present.
    auto tileSizesAttr =
        op->getAttrOfType<DenseI64ArrayAttr>("spmd.tile_sizes");
    if (!tileSizesAttr)
      return failure();

    // Only handle group-level (outermost) forall.
    auto mappingAttr = op->getAttrOfType<LevelAttr>("spmd.mapping");
    if (!mappingAttr || mappingAttr.getValue() != LevelKind::Group)
      return failure();

    Location loc = op.getLoc();
    unsigned rank = op.getRank();
    MLIRContext *ctx = &getContext();
    ArrayRef<int64_t> tileSizes = tileSizesAttr.asArrayRef();

    auto lbs   = op.getLowerBounds();
    auto ubs   = op.getUpperBounds();
    auto steps = op.getSteps();

    // Compute outer step values: max(tile_size * step, 1).
    SmallVector<Value> outerSteps;
    for (unsigned d = 0; d < rank; ++d) {
      Value tileVal =
          rewriter.create<arith::ConstantIndexOp>(loc, tileSizes[d]);
      // outer_step = tile_size * original_step
      Value outerStep =
          rewriter.create<arith::MulIOp>(loc, tileVal, steps[d]);
      outerSteps.push_back(outerStep);
    }

    // Create the outer group forall: [lb, ub) step tile_size*orig_step
    auto outerForall =
        rewriter.create<ForallOp>(loc, lbs, SmallVector<Value>(ubs),
                                  outerSteps);
    // Set outer attrs: mapping=group, remove tile_sizes/memory_policy
    // (they've been consumed by this transformation).
    outerForall->setAttr("spmd.mapping",
                         LevelAttr::get(ctx, LevelKind::Group));
    // Propagate memory_policy for later passes.
    if (auto mp = op->getAttr("spmd.memory_policy"))
      outerForall->setAttr("spmd.memory_policy", mp);

    // Inside outer body, build inner lane forall: [0, tile_size) step 1.
    rewriter.setInsertionPoint(outerForall.getBody().front().getTerminator());

    SmallVector<Value> innerLbs, innerUbs, innerSteps;
    for (unsigned d = 0; d < rank; ++d) {
      innerLbs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      innerUbs.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, tileSizes[d]));
      innerSteps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
    }

    auto innerForall = rewriter.create<ForallOp>(loc, innerLbs, innerUbs,
                                                  innerSteps);
    innerForall->setAttr("spmd.mapping",
                         LevelAttr::get(ctx, LevelKind::Lane));

    // Inside inner body: compute original IVs as outer_iv + inner_iv,
    // then guard with spmd.if (in-bounds check) and inline original body.
    rewriter.setInsertionPoint(
        innerForall.getBody().front().getTerminator());

    SmallVector<Value> origIvs;
    for (unsigned d = 0; d < rank; ++d) {
      Value outerIv = outerForall.getInductionVar(d);
      Value innerIv = innerForall.getInductionVar(d);
      Value origIv  = rewriter.create<arith::AddIOp>(loc, outerIv, innerIv);
      origIvs.push_back(origIv);
    }

    // Build in-bounds condition: AND over all dims: (orig_iv < ub)
    Value inBounds = nullptr;
    for (unsigned d = 0; d < rank; ++d) {
      Value cmp = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, origIvs[d], ubs[d]);
      inBounds = inBounds ? rewriter.create<arith::AndIOp>(loc, inBounds, cmp)
                          : cmp;
    }

    // Create spmd.if to guard the body.
    auto guardIf = rewriter.create<IfOp>(loc, TypeRange{}, inBounds,
                                          /*hasElse=*/false);

    // Fill the then region: inline original body, remapping old IVs.
    rewriter.setInsertionPoint(
        guardIf.getThenRegion().front().getTerminator());

    Block &oldBody = op.getBody().front();
    rewriter.inlineBlockBefore(&oldBody,
                                guardIf.getThenRegion().front().getTerminator(),
                                origIvs);

    // Erase the old spmd.yield that was inlined.
    Operation *inlinedYield =
        guardIf.getThenRegion().front().getTerminator()->getPrevNode();
    if (inlinedYield && isa<YieldOp>(inlinedYield))
      rewriter.eraseOp(inlinedYield);

    rewriter.eraseOp(op);
    return success();
  }
};

struct MaterializeTilingAndMappingPass
    : public PassWrapper<MaterializeTilingAndMappingPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeTilingAndMappingPass)

  StringRef getArgument() const override { return "materialize-spmd-tiling"; }
  StringRef getDescription() const override {
    return "Materialize spmd.forall tile hints into nested forall (S1 -> S2)";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<MaterializeTiledForall>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::spmd::createMaterializeTilingAndMappingPass() {
  return std::make_unique<MaterializeTilingAndMappingPass>();
}

void mlir::spmd::registerMaterializeTilingAndMappingPass() {
  PassRegistration<MaterializeTilingAndMappingPass>();
}
