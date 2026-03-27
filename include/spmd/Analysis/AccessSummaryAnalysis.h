#pragma once
//===- AccessSummaryAnalysis.h - Memref access summary --------------------===//

#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace spmd {

class ForallOp;

/// Per-memref access summary for a (group, lane) forall pair.
struct AccessSummary {
  mlir::Value memref;
  llvm::SmallVector<int64_t> minOffset; ///< min const offset per dim
  llvm::SmallVector<int64_t> maxOffset; ///< max const offset per dim
  unsigned rank;
};

/// Compute access summaries for all global memrefs loaded inside `laneForall`
/// (which is expected to be nested inside `groupForall`).
llvm::SmallVector<AccessSummary>
computeAccessSummaries(ForallOp groupForall, ForallOp laneForall);

} // namespace spmd
} // namespace mlir
