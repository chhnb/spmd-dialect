#include "spmd/IR/SPMDOps.h"
#include "spmd/IR/SPMDAttrs.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::spmd;

//===----------------------------------------------------------------------===//
// Tablegen-generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "spmd/IR/SPMDOps.cpp.inc"

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

LogicalResult YieldOp::verify() {
  // Context-dependent verification is done in the parent op's verifier.
  // Here we only check that types are valid.
  return success();
}

//===----------------------------------------------------------------------===//
// ForallOp
//===----------------------------------------------------------------------===//

LogicalResult ForallOp::verify() {
  unsigned rank = getLowerBounds().size();

  if (rank == 0)
    return emitOpError("rank must be >= 1");

  if (getUpperBounds().size() != rank || getSteps().size() != rank)
    return emitOpError("lowerBounds, upperBounds, and steps must have equal length");

  Block &body = getBody().front();
  if (body.getNumArguments() != rank)
    return emitOpError("body block arg count must equal rank");

  for (auto arg : body.getArguments())
    if (!arg.getType().isIndex())
      return emitOpError("all body block args must be of index type");

  if (auto tileSizes = getOperation()->getAttrOfType<DenseI64ArrayAttr>("spmd.tile_sizes")) {
    if ((unsigned)tileSizes.size() != rank)
      return emitOpError("spmd.tile_sizes length must equal rank");
    for (int64_t sz : tileSizes.asArrayRef())
      if (sz <= 0)
        return emitOpError("spmd.tile_sizes values must be positive");
  }

  if (auto order = getOperation()->getAttrOfType<DenseI64ArrayAttr>("spmd.order")) {
    if ((unsigned)order.size() != rank)
      return emitOpError("spmd.order length must equal rank");
    llvm::SmallVector<bool> seen(rank, false);
    for (int64_t d : order.asArrayRef()) {
      if (d < 0 || (unsigned)d >= rank)
        return emitOpError("spmd.order values must be in [0, rank)");
      if (seen[d])
        return emitOpError("spmd.order must be a permutation (no duplicates)");
      seen[d] = true;
    }
  }

  // Verify static steps are positive
  for (auto step : getSteps()) {
    if (auto constOp = step.getDefiningOp<arith::ConstantIndexOp>()) {
      if (constOp.value() <= 0)
        return emitOpError("step values must be positive");
    }
  }

  return success();
}

void ForallOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  // TODO: normalize to 0-based unit-step, fold unit-trip-count dims, etc.
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

LogicalResult IfOp::verify() {
  if (!getCondition().getType().isInteger(1))
    return emitOpError("condition must be i1");

  if (!getResults().empty()) {
    if (getElseRegion().empty())
      return emitOpError("else region required when op has results");

    auto thenYield = cast<YieldOp>(getThenRegion().front().getTerminator());
    auto elseYield = cast<YieldOp>(getElseRegion().front().getTerminator());

    if (thenYield.getValues().size() != getResults().size())
      return emitOpError("then yield count must match result count");
    if (elseYield.getValues().size() != getResults().size())
      return emitOpError("else yield count must match result count");

    for (auto [res, thenVal, elseVal] :
         llvm::zip(getResults(), thenYield.getValues(), elseYield.getValues())) {
      if (thenVal.getType() != res.getType())
        return emitOpError("then yield type mismatch");
      if (elseVal.getType() != res.getType())
        return emitOpError("else yield type mismatch");
    }
  }

  return success();
}

void IfOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  // TODO: constant condition folding, empty else elimination
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceOp::verify() {
  if (getInit().getType() != getResult().getType())
    return emitOpError("init type must match result type");

  if (!getOperation()->hasAttr("spmd.kind"))
    return emitOpError("spmd.kind attribute is required");

  Block &body = getBody().front();
  if (body.getNumArguments() != 1 || !body.getArgument(0).getType().isIndex())
    return emitOpError("body must have exactly one index block arg");

  auto yieldOp = cast<YieldOp>(body.getTerminator());
  if (yieldOp.getValues().size() != 1)
    return emitOpError("body must yield exactly one value");
  if (yieldOp.getValues()[0].getType() != getResult().getType())
    return emitOpError("yielded type must match result type");

  return success();
}

//===----------------------------------------------------------------------===//
// BarrierOp
//===----------------------------------------------------------------------===//

LogicalResult BarrierOp::verify() {
  if (!getOperation()->hasAttr("spmd.scope"))
    return emitOpError("spmd.scope attribute is required");

  // Check that we are nested inside a group-level spmd.forall
  bool foundGroupForall = false;
  Operation *parent = getOperation()->getParentOp();
  while (parent) {
    if (auto forall = dyn_cast<ForallOp>(parent)) {
      if (auto mapping = forall->getAttr("spmd.mapping")) {
        // TODO: check that mapping is group-level LevelAttr
        foundGroupForall = true;
        break;
      }
    }
    parent = parent->getParentOp();
  }

  if (!foundGroupForall)
    return emitOpError(
        "spmd.barrier must be nested inside a spmd.forall with "
        "spmd.mapping = #spmd.level<group>");

  return success();
}
