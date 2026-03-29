#include "spmd/IR/SPMDOps.h"
#include "spmd/IR/SPMDAttrs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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
  Operation *parent = getOperation()->getParentOp();

  if (auto forall = dyn_cast_or_null<ForallOp>(parent)) {
    if (!getValues().empty())
      return emitOpError("spmd.yield inside spmd.forall must have no operands");
    return success();
  }

  if (auto reduce = dyn_cast_or_null<ReduceOp>(parent)) {
    if (getValues().size() != 1)
      return emitOpError(
          "spmd.yield inside spmd.reduce must yield exactly one value");
    if (getValues()[0].getType() != reduce.getResult().getType())
      return emitOpError(
          "spmd.yield type must match spmd.reduce result type");
    return success();
  }

  // For spmd.if, type matching is checked in IfOp::verify; accept here.
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
    return emitOpError(
        "lowerBounds, upperBounds, and steps must have equal length");

  Block &body = getBody().front();
  if (body.getNumArguments() != rank)
    return emitOpError("body block arg count must equal rank");

  for (auto arg : body.getArguments())
    if (!arg.getType().isIndex())
      return emitOpError("all body block args must be of index type");

  if (auto tileSizes =
          getOperation()->getAttrOfType<DenseI64ArrayAttr>("spmd.tile_sizes")) {
    if ((unsigned)tileSizes.size() != rank)
      return emitOpError("spmd.tile_sizes length must equal rank");
    for (int64_t sz : tileSizes.asArrayRef())
      if (sz <= 0)
        return emitOpError("spmd.tile_sizes values must be positive");
  }

  if (auto order =
          getOperation()->getAttrOfType<DenseI64ArrayAttr>("spmd.order")) {
    if ((unsigned)order.size() != rank)
      return emitOpError("spmd.order length must equal rank");
    llvm::SmallVector<bool> seen(rank, false);
    for (int64_t d : order.asArrayRef()) {
      if (d < 0 || (unsigned)d >= rank)
        return emitOpError("spmd.order values must be in [0, rank)");
      if (seen[d])
        return emitOpError(
            "spmd.order must be a permutation (duplicate entry)");
      seen[d] = true;
    }
  }

  // Verify static steps are positive
  for (auto step : getSteps()) {
    if (auto cst = step.getDefiningOp<arith::ConstantIndexOp>())
      if (cst.value() <= 0)
        return emitOpError("step values must be positive");
  }

  // Verify spmd.mapping, if present, is a non-seq LevelAttr
  if (auto mappingAttr =
          getOperation()->getAttrOfType<LevelAttr>("spmd.mapping")) {
    if (mappingAttr.getValue() == LevelKind::Seq)
      return emitOpError(
          "spmd.mapping = seq is not valid for spmd.forall; "
          "use grid/group/lane/vector to denote parallel execution level");
  }

  return success();
}

void ForallOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  // TODO: normalize to 0-based unit-step, fold unit-trip-count dims
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

LogicalResult IfOp::verify() {
  // Defense-in-depth: ODS type constraint on $condition catches non-i1 at
  // parse time with "must be 1-bit signless integer", so this branch is only
  // reachable via programmatic op construction that bypasses the parser.
  if (!getCondition().getType().isInteger(1))
    return emitOpError("condition must be i1");

  if (!getResults().empty()) {
    // Defense-in-depth: the MLIR parser requires the declared region count
    // (2 for spmd.if) to be present, so "else region required" is only
    // reachable via programmatic op construction.
    if (getElseRegion().empty())
      return emitOpError("else region required when op has results");

    auto thenYield =
        cast<YieldOp>(getThenRegion().front().getTerminator());
    auto elseYield =
        cast<YieldOp>(getElseRegion().front().getTerminator());

    if (thenYield.getValues().size() != getResults().size())
      return emitOpError("then yield count must match result count");
    if (elseYield.getValues().size() != getResults().size())
      return emitOpError("else yield count must match result count");

    for (auto [res, tv, ev] : llvm::zip(getResults(),
                                         thenYield.getValues(),
                                         elseYield.getValues())) {
      if (tv.getType() != res.getType())
        return emitOpError("then yield type does not match result type");
      if (ev.getType() != res.getType())
        return emitOpError("else yield type does not match result type");
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

  // Check spmd.kind is present and is a typed ReductionKindAttr
  auto kindAttr = getOperation()->getAttrOfType<ReductionKindAttr>("spmd.kind");
  if (!kindAttr)
    return emitOpError(
        "spmd.kind attribute is required and must be a #spmd.reduction_kind");

  Block &body = getBody().front();
  if (body.getNumArguments() != 1 ||
      !body.getArgument(0).getType().isIndex())
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
  // Check spmd.scope is present and is a typed ScopeAttr
  auto scopeAttr = getOperation()->getAttrOfType<ScopeAttr>("spmd.scope");
  if (!scopeAttr)
    return emitOpError(
        "spmd.scope attribute is required and must be a #spmd.scope");

  // MVP: only group scope
  if (scopeAttr.getValue() != ScopeKind::Group)
    return emitOpError("only group scope is supported in MVP");

  // Check that we are nested inside a group-level spmd.forall
  bool foundGroupForall = false;
  Operation *parent = getOperation()->getParentOp();
  while (parent) {
    if (auto forall = dyn_cast<ForallOp>(parent)) {
      auto mappingAttr = forall->getAttrOfType<LevelAttr>("spmd.mapping");
      if (mappingAttr && mappingAttr.getValue() == LevelKind::Group) {
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
