#pragma once
//===- PromotionPlanAnalysis.h - Group memory promotion plan --------------===//

#include "spmd/Analysis/AccessSummaryAnalysis.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace spmd {

class ForallOp;

/// Describes a single memref promotion to group memory.
struct PromotionRecord {
  mlir::Value globalMemref;
  unsigned rank;
  llvm::SmallVector<int64_t> tileDims;  ///< tile buffer dimensions
  llvm::SmallVector<int64_t> minOffset; ///< min const offset (usually 0)
  llvm::SmallVector<int64_t> maxOffset;
};

/// Compute the promotion plan for a (groupForall, laneForall) pair.
/// `tileSizes` comes from the spmd.tile_sizes attr of groupForall.
llvm::SmallVector<PromotionRecord>
computePromotionPlan(ForallOp groupForall, ForallOp laneForall,
                     llvm::ArrayRef<int64_t> tileSizes);

} // namespace spmd
} // namespace mlir
