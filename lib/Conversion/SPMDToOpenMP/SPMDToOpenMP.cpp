// SPMDToOpenMP.cpp
//
// Lowers group-level spmd.forall directly to omp.parallel + omp.wsloop +
// omp.loop_nest, leaving lane-level spmd.forall for --convert-spmd-to-scf.
//
// Pipeline:
//   --convert-spmd-to-openmp   (this pass)
//   --convert-spmd-to-scf      (lowers remaining lane foralls and spmd ops)
//
// Mapping:
//   group-level spmd.forall → omp.parallel { omp.wsloop { omp.loop_nest } }
//   spmd.barrier            → omp.barrier
//   lane-level spmd.forall  → unchanged (--convert-spmd-to-scf handles it)

#include "spmd/IR/SPMDOps.h"
#include "spmd/Transforms/SPMDPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::spmd;

namespace {

//===----------------------------------------------------------------------===//
// spmd.barrier → omp.barrier
//===----------------------------------------------------------------------===//

struct BarrierToOMPBarrier : public OpRewritePattern<BarrierOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BarrierOp op,
                                PatternRewriter &rewriter) const override {
    omp::BarrierOp::create(rewriter, op.getLoc());
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// group-level spmd.forall → omp.parallel { omp.wsloop { omp.loop_nest } }
//===----------------------------------------------------------------------===//

struct GroupForallToOmpParallel : public OpRewritePattern<ForallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ForallOp op,
                                PatternRewriter &rewriter) const override {
    auto mappingAttr = op->getAttrOfType<LevelAttr>("spmd.mapping");
    if (!mappingAttr || mappingAttr.getValue() != LevelKind::Group)
      return failure();

    Location loc = op.getLoc();
    int64_t rank = static_cast<int64_t>(op.getRank());
    SmallVector<Value> lbs(op.getLowerBounds());
    SmallVector<Value> ubs(op.getUpperBounds());
    SmallVector<Value> steps(op.getSteps());

    // 1. Create omp.parallel (no clauses).
    auto parallelOp = omp::ParallelOp::create(
        rewriter, loc,
        /*allocate_vars=*/ValueRange{},
        /*allocator_vars=*/ValueRange{},
        /*if_expr=*/Value{},
        /*num_threads_vars=*/ValueRange{},
        /*private_vars=*/ValueRange{},
        /*private_syms=*/nullptr,
        /*private_needs_barrier=*/nullptr,
        /*proc_bind_kind=*/omp::ClauseProcBindKindAttr{},
        /*reduction_mod=*/nullptr,
        /*reduction_vars=*/ValueRange{},
        /*reduction_byref=*/DenseBoolArrayAttr{},
        /*reduction_syms=*/ArrayAttr{});

    // 2. Populate the parallel region.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.createBlock(&parallelOp.getRegion());

    // 3. Create omp.wsloop (no clauses).
    auto wsloopOp = omp::WsloopOp::create(rewriter, loc);

    // 4. Terminate the parallel block.
    omp::TerminatorOp::create(rewriter, loc);

    // 5. Create the single block inside the wsloop.
    rewriter.createBlock(&wsloopOp.getRegion());

    // 6. Create omp.loop_nest with the forall's loop bounds.
    //    The collapse_num_loops = rank collapses all dims into one schedule.
    auto loopOp = omp::LoopNestOp::create(
        rewriter, loc, rank, lbs, ubs, steps,
        /*loop_inclusive=*/false, /*tile_sizes=*/nullptr);

    // 7. Move the forall body region into the loop_nest region.
    //    The forall entry block (with IVs as block args) becomes the
    //    loop_nest's entry block; IVs are the omp.loop_nest induction vars.
    rewriter.inlineRegionBefore(op.getBody(), loopOp.getRegion(),
                                loopOp.getRegion().begin());

    // 8. Replace spmd.yield (forall body terminator) with omp.yield.
    Block &nestBlock = loopOp.getRegion().front();
    if (!nestBlock.empty()) {
      Operation *termOp = &nestBlock.back();
      if (isa<YieldOp>(termOp)) {
        rewriter.setInsertionPoint(termOp);
        omp::YieldOp::create(rewriter, termOp->getLoc(), ValueRange{});
        rewriter.eraseOp(termOp);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct SPMDToOpenMPPass
    : public PassWrapper<SPMDToOpenMPPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SPMDToOpenMPPass)

  StringRef getArgument() const override { return "convert-spmd-to-openmp"; }
  StringRef getDescription() const override {
    return "Lower group-level spmd.forall to omp.parallel+wsloop+loop_nest; "
           "spmd.barrier to omp.barrier. Lane-level foralls remain for "
           "--convert-spmd-to-scf.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<omp::OpenMPDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<BarrierToOMPBarrier, GroupForallToOmpParallel>(&getContext());
    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::spmd::createSPMDToOpenMPPass() {
  return std::make_unique<SPMDToOpenMPPass>();
}

void mlir::spmd::registerSPMDToOpenMPPass() {
  PassRegistration<SPMDToOpenMPPass>();
}
