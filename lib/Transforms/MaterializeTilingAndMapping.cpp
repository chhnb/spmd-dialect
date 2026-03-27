// MaterializeTilingAndMapping.cpp
//
// Transforms S1 IR (with schedule hints as attrs) into S2 IR
// (explicit nested spmd.forall with mapping labels).
//
// Input: spmd.forall with spmd.tile_sizes and spmd.mapping attrs
// Output: nested spmd.forall:
//   outer forall with step = tile_size, mapping = group
//   inner forall with step = 1,         mapping = lane
//
// Example:
//   spmd.forall (%i, %j) in (%N, %M) {spmd.tile_sizes=[32,8], spmd.mapping=group}
//   =>
//   spmd.forall (%ii, %jj) = (0,0) to (N,M) step (32,8) {spmd.mapping=group} {
//     spmd.forall (%ti, %tj) = (0,0) to (32,8) step (1,1) {spmd.mapping=lane} {
//       %i = ii + ti
//       %j = jj + tj
//       [original body with %i,%j replaced]
//     }
//   }
//
// Boundary handling: wraps body in spmd.if with in-bounds check.

#include "spmd/IR/SPMDOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::spmd;

namespace {
struct MaterializeTiledForall : public OpRewritePattern<ForallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ForallOp op,
                                PatternRewriter &rewriter) const override {
    // Only materialize if tile_sizes hint is present
    auto tileSizes = op->getAttrOfType<DenseI64ArrayAttr>("spmd.tile_sizes");
    if (!tileSizes)
      return failure();

    // TODO: implement tiling expansion
    // Stub for Phase 2.
    return failure();
  }
};

struct MaterializeTilingAndMappingPass
    : public PassWrapper<MaterializeTilingAndMappingPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeTilingAndMappingPass)

  StringRef getArgument() const override { return "materialize-spmd-tiling"; }
  StringRef getDescription() const override {
    return "Materialize spmd.forall tile hints into nested forall (S1 -> S2)";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<MaterializeTiledForall>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::spmd::createMaterializeTilingAndMappingPass() {
  return std::make_unique<MaterializeTilingAndMappingPass>();
}
