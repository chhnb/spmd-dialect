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
//   outer_iv + inner_iv * step + arith.constant
// (step == 1 is the common case; step > 1 arises when the original S0 forall
//  had a non-unit step and was materialized without prior NormalizeSPMD.)
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

/// Return true if `v` is one of the values in `range`.
static bool isOneOf(Value v, ValueRange range) {
  for (auto r : range)
    if (v == r)
      return true;
  return false;
}

/// Try to recognize `v` as `inner_iv * const_step` and return the step.
/// Returns 1 if `v` itself is an inner_iv (implicit step=1).
/// Returns std::nullopt if not recognizable.
static std::optional<int64_t> getInnerIvStep(Value v, ValueRange innerIvs) {
  if (isOneOf(v, innerIvs))
    return 1;
  if (auto mul = v.getDefiningOp<arith::MulIOp>()) {
    auto lhsCst = mul.getLhs().getDefiningOp<arith::ConstantIndexOp>();
    auto rhsCst = mul.getRhs().getDefiningOp<arith::ConstantIndexOp>();
    if (lhsCst && isOneOf(mul.getRhs(), innerIvs))
      return lhsCst.value();
    if (rhsCst && isOneOf(mul.getLhs(), innerIvs))
      return rhsCst.value();
  }
  return std::nullopt;
}

/// Try to decompose `index` as (outer_iv + inner_iv * step + const).
/// Returns the constant offset from the tile origin if successful,
/// std::nullopt otherwise.
///
/// Handles:
///   - step=1 (post-NormalizeSPMD) and step>1 (non-normalized S1 input)
///   - both arith.addi and arith.subi of constants (negative halos)
static std::optional<int64_t>
getConstOffset(Value index, ValueRange outerIvs, ValueRange innerIvs) {
  // Case 1: index == inner_iv (or inner_iv * step) → offset 0
  if (getInnerIvStep(index, innerIvs))
    return 0;

  // Case 2: peel arith.addi / arith.subi constants and look for
  //   outer_iv + inner_iv * step.
  // arith.subi(lhs, const) is treated as addi(lhs, -const).
  int64_t accumulated = 0;
  Value cur = index;
  while (true) {
    if (auto add = cur.getDefiningOp<arith::AddIOp>()) {
      auto lhsCst = add.getLhs().getDefiningOp<arith::ConstantIndexOp>();
      auto rhsCst = add.getRhs().getDefiningOp<arith::ConstantIndexOp>();
      if (lhsCst) {
        accumulated += lhsCst.value();
        cur = add.getRhs();
      } else if (rhsCst) {
        accumulated += rhsCst.value();
        cur = add.getLhs();
      } else {
        // Neither side is constant — check for outer + scaled_inner.
        bool lhsIsOuter = isOneOf(add.getLhs(), outerIvs);
        bool rhsIsOuter = isOneOf(add.getRhs(), outerIvs);
        bool lhsIsScaledInner =
            getInnerIvStep(add.getLhs(), innerIvs).has_value();
        bool rhsIsScaledInner =
            getInnerIvStep(add.getRhs(), innerIvs).has_value();
        if ((lhsIsOuter && rhsIsScaledInner) || (lhsIsScaledInner && rhsIsOuter))
          return accumulated;
        return std::nullopt;
      }
    } else if (auto sub = cur.getDefiningOp<arith::SubIOp>()) {
      // sub(lhs, rhsConst) == addi(lhs, -rhsConst)
      auto rhsCst = sub.getRhs().getDefiningOp<arith::ConstantIndexOp>();
      if (!rhsCst)
        return std::nullopt;
      accumulated -= rhsCst.value();
      cur = sub.getLhs();
    } else {
      return std::nullopt;
    }
    // After each step: check if cur is already a terminal value.
    if (isOneOf(cur, innerIvs) || isOneOf(cur, outerIvs))
      return accumulated;
    if (getInnerIvStep(cur, innerIvs))
      return accumulated;
  }
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
      // Use sentinel values so the first decomposable offset correctly seeds
      // the bounding box.  Initializing to 0 would zero-clamp asymmetric
      // halos: e.g. a stencil reading only A[i+1],A[i+2] would get minOffset=0
      // instead of 1, and a negative-only halo would get maxOffset=0 instead
      // of the true (negative) max.
      //
      // kUnbounded is also used by the else-branch below, so the PromotionPlan
      // bounded-check (>= INT64_MAX/4) correctly rejects dims that have no
      // decomposable access at all.
      static const int64_t kSentinel = INT64_MAX / 2;
      AccessSummary s;
      s.memref = mr;
      s.rank   = ndim;
      s.minOffset.assign(ndim,  kSentinel);  // will be min()'d to first offset
      s.maxOffset.assign(ndim, -kSentinel);  // will be max()'d to first offset
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
