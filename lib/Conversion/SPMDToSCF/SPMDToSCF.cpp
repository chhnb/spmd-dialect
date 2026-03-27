// SPMDToSCF.cpp
//
// Lowers spmd dialect S2 IR to scf + arith for sequential CPU execution.
// This is the "correctness" baseline; parallelism is re-introduced later by
// the standard --convert-scf-to-openmp or --convert-spmd-to-openmp passes.
//
// Mapping:
//   spmd.forall (any mapping)  -> N nested scf.for
//   spmd.if                    -> scf.if
//   spmd.reduce                -> scf.for with iter_args accumulator
//   spmd.barrier               -> no-op (erase)
//   spmd.yield (in forall)     -> handled by parent pattern (erase)
//   spmd.yield (in reduce)     -> scf.yield with combined value
//   spmd.yield (in if)         -> scf.yield with same operands

#include "spmd/IR/SPMDOps.h"
#include "spmd/Transforms/SPMDPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::spmd;

namespace {

//===----------------------------------------------------------------------===//
// spmd.forall → N nested scf.for
//===----------------------------------------------------------------------===//

struct ForallToSCFFor : public OpRewritePattern<ForallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ForallOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned rank = op.getRank();
    auto lbs   = op.getLowerBounds();
    auto ubs   = op.getUpperBounds();
    auto steps = op.getSteps();

    // Create outermost scf.for at the current insertion point.
    rewriter.setInsertionPoint(op);
    auto outerFor = rewriter.create<scf::ForOp>(loc, lbs[0], ubs[0], steps[0]);
    SmallVector<Value> ivs{outerFor.getInductionVar()};
    scf::ForOp innermostFor = outerFor;

    // Create progressively nested inner loops.
    for (unsigned d = 1; d < rank; ++d) {
      rewriter.setInsertionPoint(innermostFor.getBody()->getTerminator());
      auto innerFor =
          rewriter.create<scf::ForOp>(loc, lbs[d], ubs[d], steps[d]);
      ivs.push_back(innerFor.getInductionVar());
      innermostFor = innerFor;
    }

    // Inline the forall body before the innermost scf.for's yield.
    Operation *innermostYield = innermostFor.getBody()->getTerminator();
    Block &forallBody = op.getBody().front();
    rewriter.inlineBlockBefore(&forallBody, innermostYield, ivs);

    // After inlining, the spmd.yield (forall body terminator) now sits just
    // before the scf.yield. Erase it.
    Operation *spmdYield = innermostYield->getPrevNode();
    if (spmdYield && isa<YieldOp>(spmdYield))
      rewriter.eraseOp(spmdYield);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// spmd.if → scf.if
//===----------------------------------------------------------------------===//

struct IfToSCFIf : public OpRewritePattern<IfOp> {
  using OpRewritePattern::OpRewritePattern;

  // Inline a region's single block into dstBlock, converting spmd.yield to
  // scf.yield.  If the builder already placed an scf.yield in dstBlock
  // (the no-result case), the inlined ops go before it and the spmd.yield is
  // erased.  If the block is empty (the has-result case), the spmd.yield is
  // converted in place.
  void transferRegion(PatternRewriter &rewriter, Region &src,
                      Block *dstBlock) const {
    Block &srcBlock = src.front();

    // Find the builder-created scf.yield, if any.
    Operation *existingYield = nullptr;
    if (!dstBlock->empty() && isa<scf::YieldOp>(dstBlock->back()))
      existingYield = &dstBlock->back();

    if (existingYield) {
      // No-result case: inline before the existing scf.yield.
      rewriter.inlineBlockBefore(&srcBlock, existingYield, /*args=*/{});
      Operation *spmdYield = existingYield->getPrevNode();
      if (spmdYield && isa<YieldOp>(spmdYield))
        rewriter.eraseOp(spmdYield);
    } else {
      // Has-result case: block is empty; inline at end, then convert yield.
      rewriter.inlineBlockBefore(&srcBlock, dstBlock, dstBlock->end(),
                                  /*args=*/{});
      // spmd.yield is now the last op; replace with scf.yield.
      Operation *spmdYield = &dstBlock->back();
      assert(isa<YieldOp>(spmdYield) && "expected spmd.yield at end");
      auto yieldOp = cast<YieldOp>(spmdYield);
      rewriter.setInsertionPoint(spmdYield);
      rewriter.create<scf::YieldOp>(spmdYield->getLoc(),
                                     yieldOp.getValues());
      rewriter.eraseOp(spmdYield);
    }
  }

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TypeRange resultTypes = op.getResultTypes();
    bool hasElse = !op.getElseRegion().empty() &&
                   !op.getElseRegion().front().empty();

    auto newIf = rewriter.create<scf::IfOp>(loc, resultTypes,
                                              op.getCondition(), hasElse);

    transferRegion(rewriter, op.getThenRegion(),
                   &newIf.getThenRegion().front());
    if (hasElse)
      transferRegion(rewriter, op.getElseRegion(),
                     &newIf.getElseRegion().front());

    rewriter.replaceOp(op, newIf.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// spmd.reduce → scf.for with iter_args
//===----------------------------------------------------------------------===//

struct ReduceToSCFFor : public OpRewritePattern<ReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  Value buildCombine(PatternRewriter &rewriter, Location loc,
                     ReductionKind kind, Value acc, Value partial,
                     Type type) const {
    bool isFloat = isa<FloatType>(type);
    switch (kind) {
    case ReductionKind::Add:
      return isFloat ? rewriter.create<arith::AddFOp>(loc, acc, partial)
                     : (Value)rewriter.create<arith::AddIOp>(loc, acc, partial);
    case ReductionKind::Mul:
      return isFloat ? rewriter.create<arith::MulFOp>(loc, acc, partial)
                     : (Value)rewriter.create<arith::MulIOp>(loc, acc, partial);
    case ReductionKind::Max:
      return isFloat
                 ? rewriter.create<arith::MaximumFOp>(loc, acc, partial)
                 : (Value)rewriter.create<arith::MaxSIOp>(loc, acc, partial);
    case ReductionKind::Min:
      return isFloat
                 ? rewriter.create<arith::MinimumFOp>(loc, acc, partial)
                 : (Value)rewriter.create<arith::MinSIOp>(loc, acc, partial);
    case ReductionKind::And:
      return rewriter.create<arith::AndIOp>(loc, acc, partial);
    case ReductionKind::Or:
      return rewriter.create<arith::OrIOp>(loc, acc, partial);
    case ReductionKind::Xor:
      return rewriter.create<arith::XOrIOp>(loc, acc, partial);
    }
    llvm_unreachable("unhandled ReductionKind");
  }

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type elemType = op.getResult().getType();
    auto kindAttr = op->getAttrOfType<ReductionKindAttr>("spmd.kind");

    // Create scf.for with the init value as iter_arg.
    auto forOp = rewriter.create<scf::ForOp>(
        loc, op.getLowerBound(), op.getUpperBound(), op.getStep(),
        ValueRange{op.getInit()});

    Block *forBody    = forOp.getBody();
    Value  loopIv     = forOp.getInductionVar();
    Value  acc        = forOp.getRegionIterArg(0);
    // The builder creates: scf.yield(%acc) as the body terminator.
    Operation *forYield = forBody->getTerminator();

    // Inline the reduce body before the scf.yield, mapping the index arg.
    Block &reduceBody = op.getBody().front();
    rewriter.inlineBlockBefore(&reduceBody, forYield, {loopIv});

    // After inlining: spmd.yield(%partial) is just before forYield.
    Operation *spmdYield = forYield->getPrevNode();
    assert(spmdYield && isa<YieldOp>(spmdYield));
    auto yieldOp  = cast<YieldOp>(spmdYield);
    Value partial = yieldOp.getValues()[0];

    // Build the combine op before spmdYield.
    rewriter.setInsertionPoint(spmdYield);
    Value combined =
        buildCombine(rewriter, loc, kindAttr.getValue(), acc, partial, elemType);

    // Replace the old scf.yield (with acc as operand) with one that
    // yields the combined value.
    rewriter.setInsertionPoint(forYield);
    rewriter.create<scf::YieldOp>(loc, ValueRange{combined});
    rewriter.eraseOp(forYield);
    rewriter.eraseOp(spmdYield);

    rewriter.replaceOp(op, forOp.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// spmd.barrier → no-op (CPU sequential execution)
//===----------------------------------------------------------------------===//

struct BarrierToNoop : public OpRewritePattern<BarrierOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BarrierOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct SPMDToSCFPass
    : public PassWrapper<SPMDToSCFPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SPMDToSCFPass)

  StringRef getArgument() const override { return "convert-spmd-to-scf"; }
  StringRef getDescription() const override {
    return "Lower spmd dialect to scf for sequential CPU execution";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<ForallToSCFFor, IfToSCFIf, ReduceToSCFFor, BarrierToNoop>(
        &getContext());

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();

    // Verify no SPMD ops remain.
    bool hasSPMD = false;
    func.walk([&](Operation *op) {
      if (isa<ForallOp, IfOp, ReduceOp, BarrierOp, YieldOp>(op)) {
        op->emitError("SPMD op not lowered by convert-spmd-to-scf");
        hasSPMD = true;
      }
    });
    if (hasSPMD)
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::spmd::createSPMDToSCFPass() {
  return std::make_unique<SPMDToSCFPass>();
}

void mlir::spmd::registerSPMDToSCFPass() {
  PassRegistration<SPMDToSCFPass>();
}
