// SPMDToSCF.cpp
//
// Lowers spmd dialect S2 IR to scf + arith for CPU execution.
//
// Mapping:
//   spmd.forall (group-level) -> scf.parallel or scf.for (outer chunk)
//   spmd.forall (lane-level)  -> scf.for (inner sequential, later OpenMP)
//   spmd.if                   -> scf.if
//   spmd.reduce               -> scf.for with accumulator
//   spmd.barrier              -> no-op on CPU (sequential inner loop)
//   spmd.yield                -> (implicit scf terminator)
//
// group/private memrefs: addr space attr stripped, lowered to plain memref.

#include "spmd/IR/SPMDOps.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::spmd;

namespace {

// spmd.forall -> scf.for (sequential fallback, correct for MVP)
struct ForallToSCFFor : public OpConversionPattern<ForallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ForallOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // TODO: implement ForallOp -> scf.for lowering
    // Phase 2 implementation.
    return failure();
  }
};

// spmd.if -> scf.if
struct IfToSCFIf : public OpConversionPattern<IfOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(IfOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // TODO: implement IfOp -> scf.if lowering
    return failure();
  }
};

// spmd.reduce -> scf.for with init_args accumulator
struct ReduceToSCFFor : public OpConversionPattern<ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ReduceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // TODO: implement ReduceOp -> scf.for lowering
    return failure();
  }
};

// spmd.barrier -> no-op on CPU (inner forall is sequential)
struct BarrierToNoop : public OpConversionPattern<BarrierOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(BarrierOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct SPMDToSCFPass
    : public PassWrapper<SPMDToSCFPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SPMDToSCFPass)

  StringRef getArgument() const override { return "convert-spmd-to-scf"; }
  StringRef getDescription() const override {
    return "Lower spmd dialect to scf for CPU execution";
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<scf::SCFDialect, arith::ArithDialect,
                            memref::MemRefDialect, func::FuncDialect>();
    target.addIllegalDialect<SPMDDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ForallToSCFFor, IfToSCFIf, ReduceToSCFFor, BarrierToNoop>(
        &getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::spmd::createSPMDToSCFPass() {
  return std::make_unique<SPMDToSCFPass>();
}
