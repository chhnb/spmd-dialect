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
#include "spmd/IR/SPMDOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::spmd;

namespace mlir {
namespace spmd {

/// Maximum allowed group memory footprint in bytes (48 KB).
static constexpr int64_t kMaxGroupMemBytes = 48 * 1024;

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
                     ArrayRef<int64_t> tileSizes) {
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
    SmallVector<int64_t> tileDims;
    int64_t footprintBytes = elemBytes(mrType);
    for (unsigned d = 0; d < summary.rank; ++d) {
      int64_t tileSize = (d < tileSizes.size()) ? tileSizes[d] : 1;
      // footprint in dimension d:
      // from minOffset[d] to tileSize-1 + maxOffset[d]
      int64_t dim = tileSize + summary.maxOffset[d] - summary.minOffset[d];
      tileDims.push_back(dim);
      footprintBytes *= dim;
    }

    // 4. Footprint size check.
    if (footprintBytes > kMaxGroupMemBytes) {
      groupForall.emitRemark()
          << "promote-group-memory: skipping " << mr
          << " — tile footprint " << footprintBytes
          << " B exceeds maxGroupMemBytes (" << kMaxGroupMemBytes << " B)";
      continue;
    }

    // 5. Reuse check: always satisfied when maxOffset > 0 in any dim
    //    (stencil pattern) or when the tile is accessed by multiple lanes.
    //    For MVP: always promote if bounds are satisfied.

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
