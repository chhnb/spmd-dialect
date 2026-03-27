// AccessSummaryAnalysis.cpp
//
// Collects memref access information for spmd.forall bodies to feed into
// PromotionPlanAnalysis.
//
// For each group-level spmd.forall F containing a lane-level spmd.forall L:
//   For each memref.load inside L:
//     - Record the loaded memref.
//     - For each index dimension, determine the offset from the tile origin:
//         index = outer_iv[d] + inner_iv[d] + const_offset
//       Store (min_offset, max_offset) per dimension.
//
// Limitation (MVP): only handles affine patterns of the form
//   outer_iv + inner_iv + arith.constant
// More complex index expressions are conservatively recorded as unbounded.

#include "spmd/Analysis/AccessSummaryAnalysis.h"
#include "spmd/IR/SPMDOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::spmd;

namespace mlir {
namespace spmd {

/// Try to decompose `index` as (outer_iv + inner_iv + const).
/// Returns the constant if successful, std::nullopt otherwise.
static std::optional<int64_t>
getConstOffset(Value index, ValueRange outerIvs, ValueRange innerIvs) {
  // Case 1: index == inner_iv → offset 0
  for (auto iv : innerIvs)
    if (index == iv)
      return 0;
  // Case 2: index == outer_iv + inner_iv → from an addi chain
  // We look for: addi(outer+inner, const) or addi(const, outer+inner)
  // Try to peel constants from arith.addi chains.
  int64_t accumulated = 0;
  Value cur = index;
  while (auto add = cur.getDefiningOp<arith::AddIOp>()) {
    // Check if one operand is a constant.
    auto lhsCst = add.getLhs().getDefiningOp<arith::ConstantIndexOp>();
    auto rhsCst = add.getRhs().getDefiningOp<arith::ConstantIndexOp>();
    if (lhsCst) {
      accumulated += lhsCst.value();
      cur = add.getRhs();
    } else if (rhsCst) {
      accumulated += rhsCst.value();
      cur = add.getLhs();
    } else {
      // Neither operand is constant — might be (outer+inner).
      // Check if this op is outer_iv + inner_iv.
      bool lhsIsOuter = false, rhsIsOuter = false;
      bool lhsIsInner = false, rhsIsInner = false;
      for (auto iv : outerIvs) {
        if (add.getLhs() == iv) lhsIsOuter = true;
        if (add.getRhs() == iv) rhsIsOuter = true;
      }
      for (auto iv : innerIvs) {
        if (add.getLhs() == iv) lhsIsInner = true;
        if (add.getRhs() == iv) rhsIsInner = true;
      }
      if ((lhsIsOuter && rhsIsInner) || (lhsIsInner && rhsIsOuter))
        return accumulated; // offset from tile origin
      // Not decomposable.
      return std::nullopt;
    }
    // After peeling the constant, check if `cur` is outer+inner sum.
    for (auto iv : innerIvs)
      if (cur == iv)
        return accumulated;
    for (auto iv : outerIvs)
      if (cur == iv)
        return accumulated;
  }
  return std::nullopt;
}

/// Compute access summaries for all global memrefs loaded inside the inner
/// (lane-level) forall of a group-level forall.
SmallVector<AccessSummary>
computeAccessSummaries(ForallOp groupForall, ForallOp laneForall) {
  SmallVector<AccessSummary> result;
  DenseMap<Value, unsigned> memrefToIdx;

  ValueRange outerIvs = groupForall.getInductionVars();
  ValueRange innerIvs = laneForall.getInductionVars();

  laneForall.getBody().front().walk([&](memref::LoadOp load) {
    Value mr = load.getMemRef();
    auto mrType = cast<MemRefType>(mr.getType());

    // Skip non-global memrefs (already promoted or private).
    if (auto addrSpace = dyn_cast_or_null<AddressSpaceAttr>(
            mrType.getMemorySpace())) {
      if (addrSpace.getValue() != AddressSpaceKind::Global)
        return;
    }

    unsigned ndim = mrType.getRank();
    unsigned idx;
    auto mapIt = memrefToIdx.find(mr);
    if (mapIt == memrefToIdx.end()) {
      idx = result.size();
      memrefToIdx[mr] = idx;
      AccessSummary s;
      s.memref = mr;
      s.rank   = ndim;
      s.minOffset.assign(ndim, 0);
      s.maxOffset.assign(ndim, 0);
      result.push_back(std::move(s));
    } else {
      idx = mapIt->second;
    }
    AccessSummary &summary = result[idx];

    // Analyze each index.
    auto indices = load.getIndices();
    for (unsigned d = 0; d < ndim && d < indices.size(); ++d) {
      auto offset = getConstOffset(indices[d], outerIvs, innerIvs);
      if (offset) {
        summary.minOffset[d] = std::min(summary.minOffset[d], *offset);
        summary.maxOffset[d] = std::max(summary.maxOffset[d], *offset);
      } else {
        // Conservative: mark as unbounded (skip promotion for this memref).
        const int64_t kUnbounded = INT64_MAX / 2;
        summary.minOffset[d] = -kUnbounded;
        summary.maxOffset[d] =  kUnbounded;
      }
    }
  });

  return result;
}

} // namespace spmd
} // namespace mlir
