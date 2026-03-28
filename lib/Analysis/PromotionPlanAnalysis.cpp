// PromotionPlanAnalysis.cpp
//
// Determines which memrefs inside a group-level spmd.forall should be
// promoted to group memory, and computes the tile buffer dimensions.
//
// MVP legality requirements:
//   1. Access pattern is decomposable (AccessSummary has bounded offsets).
//   2. Access is read-only (only memref.load, no memref.store to this memref).
//   3. Tile footprint fits within target.maxGroupMemBytes (48 KB default).
//
// MVP profitability:
//   Always promote when legality passes and reuse count > 1 (any stencil).
//   (Full profitability model deferred to a future round.)

#include "spmd/Analysis/AccessSummaryAnalysis.h"
#include "spmd/Analysis/PromotionPlanAnalysis.h"
#include "spmd/Analysis/TargetDescriptor.h"
#include "spmd/IR/SPMDOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::spmd;

namespace mlir {
namespace spmd {

/// Default target for the MVP pass (CPU).
static const TargetDescriptor kDefaultTarget = TargetDescriptor::cpuDefault();

/// Check that there is no memref.store to `mr` inside `laneForall`.
static bool isReadOnly(Value mr, ForallOp laneForall) {
  bool hasStore = false;
  laneForall.getBody().front().walk([&](memref::StoreOp store) {
    if (store.getMemRef() == mr)
      hasStore = true;
  });
  return !hasStore;
}

/// Compute the element size of the memref's element type in bytes.
static int64_t elemBytes(MemRefType mrType) {
  Type elemTy = mrType.getElementType();
  if (elemTy.isF32() || elemTy.isInteger(32)) return 4;
  if (elemTy.isF64() || elemTy.isInteger(64)) return 8;
  if (elemTy.isF16() || elemTy.isInteger(16)) return 2;
  if (elemTy.isInteger(8))  return 1;
  return 4; // conservative default
}

/// Compute promotion plan for a (groupForall, laneForall) pair.
SmallVector<PromotionRecord>
computePromotionPlan(ForallOp groupForall, ForallOp laneForall,
                     ArrayRef<int64_t> tileSizes,
                     ArrayRef<int64_t> origSteps) {
  SmallVector<PromotionRecord> plan;

  // Only promote when policy = prefer_group.
  auto policyAttr =
      groupForall->getAttrOfType<MemoryPolicyAttr>("spmd.memory_policy");
  if (!policyAttr || policyAttr.getValue() != MemoryPolicyKind::PreferGroup)
    return plan;

  SmallVector<AccessSummary> summaries =
      computeAccessSummaries(groupForall, laneForall);

  for (auto &summary : summaries) {
    Value mr = summary.memref;
    auto mrType = cast<MemRefType>(mr.getType());

    // 1. Bounded access pattern check.
    bool bounded = true;
    for (unsigned d = 0; d < summary.rank; ++d) {
      if (summary.maxOffset[d] >= INT64_MAX / 4 ||
          summary.minOffset[d] <= -INT64_MAX / 4) {
        bounded = false;
        break;
      }
    }
    if (!bounded)
      continue;

    // 2. Read-only check.
    if (!isReadOnly(mr, laneForall))
      continue;

    // 3. Compute tile footprint dimensions.
    //
    // Dense-tile layout: tile[k] = A[outer + k + minOffset[d]] for k in
    //   [0, extent[d]).  For a stepped kernel (original loop step s > 1),
    //   lane `i` accesses A[outer + i*s + off].  The last lane accesses up
    //   to outer + (tileSize-1)*s + maxOffset[d], so:
    //
    //     extent[d] = (tileSize - 1) * step + maxOffset[d] - minOffset[d] + 1
    //
    //   For step == 1 this reduces to the previous formula
    //   (tileSize + maxOffset - minOffset), since (t-1)*1 + off + 1 = t + off.
    SmallVector<int64_t> tileDims;
    int64_t footprintBytes = elemBytes(mrType);
    for (unsigned d = 0; d < summary.rank; ++d) {
      int64_t tileSize = (d < tileSizes.size()) ? tileSizes[d] : 1;
      int64_t step     = (d < origSteps.size() && origSteps[d] > 0)
                             ? origSteps[d]
                             : 1;
      int64_t dim = (tileSize - 1) * step +
                    summary.maxOffset[d] - summary.minOffset[d] + 1;
      tileDims.push_back(dim);
      footprintBytes *= dim;
    }

    // 4. Footprint size check.
    if (footprintBytes > kDefaultTarget.maxGroupMemBytes) {
      groupForall.emitRemark()
          << "promote-group-memory: skipping " << mr
          << " — tile footprint " << footprintBytes
          << " B exceeds maxGroupMemBytes (" << kDefaultTarget.maxGroupMemBytes
          << " B)";
      continue;
    }

    // 5. Reuse / profitability check.
    //    Only promote when at least one dimension has a non-trivial access
    //    footprint (maxOffset > minOffset).  This is the "stencil pattern"
    //    indicator: different lanes touch the same tile element, so there is
    //    genuine data reuse.  A simple copy kernel (every lane reads exactly
    //    one distinct element, all offsets == 0) has no reuse and must not
    //    be promoted — promotion would only add overhead.
    bool hasReuse = false;
    for (unsigned d = 0; d < summary.rank; ++d) {
      if (summary.maxOffset[d] > summary.minOffset[d]) {
        hasReuse = true;
        break;
      }
    }
    if (!hasReuse)
      continue;

    PromotionRecord rec;
    rec.globalMemref = mr;
    rec.rank      = summary.rank;
    rec.tileDims  = tileDims;
    rec.minOffset = summary.minOffset;
    rec.maxOffset = summary.maxOffset;
    plan.push_back(rec);
  }

  return plan;
}

} // namespace spmd
} // namespace mlir
