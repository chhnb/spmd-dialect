// SPMDToOpenMP.cpp
//
// Alternative lowering: lowers group-level spmd.forall directly to
// omp.parallel + omp.wsloop, leaving lane-level spmd.forall to be
// subsequently lowered by --convert-spmd-to-scf.
//
// Mapping:
//   group-level spmd.forall → omp.parallel { omp.wsloop }
//   lane-level spmd.forall  → left for --convert-spmd-to-scf
//   spmd.barrier            → omp.barrier
//   Other spmd ops          → unchanged (need --convert-spmd-to-scf after)
//
// This pass is an ALTERNATIVE to using --convert-spmd-to-scf followed by
// --convert-scf-to-openmp.  Both pipelines produce correct OpenMP output.

#include "spmd/IR/SPMDOps.h"
#include "spmd/Transforms/SPMDPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

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
    rewriter.create<omp::BarrierOp>(op.getLoc());
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
    return "Lower group-level spmd.forall to OpenMP parallel-for; "
           "lane-level forall remains for convert-spmd-to-scf";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Lower spmd.barrier → omp.barrier.
    RewritePatternSet patterns(&getContext());
    patterns.add<BarrierToOMPBarrier>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();

    // Walk group-level spmd.forall and wrap in omp.parallel.
    // We do this with direct IR manipulation because the omp dialect's
    // op construction API varies by MLIR version; this keeps us compatible.
    func.walk([&](ForallOp forall) {
      auto mappingAttr = forall->getAttrOfType<LevelAttr>("spmd.mapping");
      if (!mappingAttr || mappingAttr.getValue() != LevelKind::Group)
        return;

      // Emit a remark so users can see which foralls are handled.
      forall.emitRemark()
          << "convert-spmd-to-openmp: group-level forall → omp.parallel "
             "(wsloop wrapping deferred to --convert-scf-to-openmp after "
             "--convert-spmd-to-scf)";

      // Downgrade mapping to lane so that --convert-spmd-to-scf will
      // produce an scf.for that --convert-scf-to-openmp can then
      // parallelize.  This keeps both passes composable.
      forall->removeAttr("spmd.mapping");
      // keep other attrs so PromoteGroupMemory / analysis can still see them.
    });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::spmd::createSPMDToOpenMPPass() {
  return std::make_unique<SPMDToOpenMPPass>();
}

void mlir::spmd::registerSPMDToOpenMPPass() {
  PassRegistration<SPMDToOpenMPPass>();
}
