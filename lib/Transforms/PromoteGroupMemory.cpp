// PromoteGroupMemory.cpp
//
// Core innovation pass. Promotes tile-reusable global memref slices to
// group address space, inserts cooperative copy loops and barriers.
//
// Algorithm (MVP — read-only stencil pattern):
//   For each group-level forall G with memory_policy = prefer_group:
//     1. Find the first lane-level forall L nested directly inside G.
//     2. Use PromotionPlanAnalysis to find promotable memrefs.
//     3. For each promotable memref M with tile dimensions D:
//        a. Alloc tile buffer T : memref<D... x ElemType, #spmd.addr_space<group>>
//        b. Before L: insert a cooperative copy lane forall that loads M → T.
//        c. After the copy forall: insert spmd.barrier.
//        d. Clone the compute body (L) with loads from M replaced by loads from T.
//        e. Erase original L.
//
// Negative cases:
//   - memory_policy = no_promotion: skip entirely (no transformation).
//   - Footprint > kMaxGroupMemBytes: skip with remark (handled in analysis).
//   - No inner lane forall found: skip with remark.

#include "spmd/Analysis/PromotionPlanAnalysis.h"
#include "spmd/IR/SPMDAttrs.h"
#include "spmd/IR/SPMDOps.h"
#include "spmd/Transforms/SPMDPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::spmd;

namespace {

/// Find the first direct child of `groupForall` that is a lane-level
/// spmd.forall. Returns null if not found.
static ForallOp findDirectLaneForall(ForallOp groupForall) {
  for (auto &op : groupForall.getBody().front()) {
    if (auto laneForall = dyn_cast<ForallOp>(&op)) {
      auto mapping = laneForall->getAttrOfType<LevelAttr>("spmd.mapping");
      if (mapping && mapping.getValue() == LevelKind::Lane)
        return laneForall;
      return laneForall; // accept any inner forall if no mapping attr
    }
  }
  return ForallOp{};
}

/// Build the tile buffer memref type: memref<d0 x d1 x ... x ElemTy, group>.
static MemRefType buildTileType(MLIRContext *ctx, Type elemTy,
                                ArrayRef<int64_t> dims) {
  auto addrSpace = AddressSpaceAttr::get(ctx, AddressSpaceKind::Group);
  return MemRefType::get(dims, elemTy, MemRefLayoutAttrInterface{}, addrSpace);
}

/// Replace all uses of `globalMr` (loads inside `block`) with loads from
/// `tileMr`, adjusting indices by subtracting `minOffset`.
static void rewriteLoadsToTile(Block &block, Value globalMr, Value tileMr,
                                ArrayRef<int64_t> minOffset,
                                ValueRange outerIvs, ValueRange innerIvs,
                                OpBuilder &rewriter) {
  block.walk([&](memref::LoadOp load) {
    if (load.getMemRef() != globalMr)
      return;
    Location loc = load.getLoc();
    // Set IP before load first — tileIndex ops must be inserted here too,
    // so the IP is valid for all creates in this iteration.
    rewriter.setInsertionPoint(load);
    SmallVector<Value> tileIndices;
    auto srcIndices = load.getIndices();
    for (unsigned d = 0; d < srcIndices.size(); ++d) {
      Value srcIdx = srcIndices[d];
      // Tile index = inner_iv + const_offset - minOffset[d].
      // For simplicity: peel the outer IV from srcIdx to get the tile-local idx.
      // srcIdx = outer_iv[d] + inner_iv_or_offset → tile_idx = inner_iv_or_offset
      // We subtract the outer IV if it appears.
      Value tileIdx = srcIdx;
      if (d < outerIvs.size()) {
        // Try to subtract outer IV using arith.
        tileIdx = rewriter.create<arith::SubIOp>(loc, srcIdx, outerIvs[d]);
      }
      // Subtract minOffset if non-zero.
      if (d < minOffset.size() && minOffset[d] != 0) {
        Value minOff = rewriter.create<arith::ConstantIndexOp>(loc, minOffset[d]);
        tileIdx = rewriter.create<arith::SubIOp>(loc, tileIdx, minOff);
      }
      tileIndices.push_back(tileIdx);
    }
    auto newLoad = rewriter.create<memref::LoadOp>(loc, tileMr, tileIndices);
    load.getResult().replaceAllUsesWith(newLoad.getResult());
    load->erase();
  });
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct PromoteGroupMemoryPass
    : public PassWrapper<PromoteGroupMemoryPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PromoteGroupMemoryPass)

  StringRef getArgument() const override { return "promote-group-memory"; }
  StringRef getDescription() const override {
    return "Promote reused tile footprints to group memory with cooperative copy";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    // Collect group-level foralls to process (avoid modifying while walking).
    SmallVector<ForallOp> groupForalls;
    func.walk([&](ForallOp forall) {
      auto mapping = forall->getAttrOfType<LevelAttr>("spmd.mapping");
      if (mapping && mapping.getValue() == LevelKind::Group)
        groupForalls.push_back(forall);
    });

    for (ForallOp groupForall : groupForalls) {
      // Skip if no_promotion policy.
      auto policyAttr =
          groupForall->getAttrOfType<MemoryPolicyAttr>("spmd.memory_policy");
      if (policyAttr &&
          policyAttr.getValue() == MemoryPolicyKind::NoPromotion) {
        groupForall.emitRemark()
            << "promote-group-memory: skipping, memory_policy=no_promotion";
        continue;
      }

      // Only promote if prefer_group is set (or none → let analysis decide).
      if (policyAttr &&
          policyAttr.getValue() != MemoryPolicyKind::PreferGroup)
        continue;

      // Find inner lane forall.
      ForallOp laneForall = findDirectLaneForall(groupForall);
      if (!laneForall) {
        groupForall.emitRemark()
            << "promote-group-memory: no lane-level forall found; skipping";
        continue;
      }

      // Get tile_sizes for footprint computation.
      auto tileSizesAttr =
          groupForall->getAttrOfType<DenseI64ArrayAttr>("spmd.tile_sizes");
      if (!tileSizesAttr)
        continue;
      ArrayRef<int64_t> tileSizes = tileSizesAttr.asArrayRef();

      // Derive the per-dimension original loop step from the outer forall's
      // runtime step: origStep[d] = outerStep[d] / tileSizes[d].
      // The outer forall step = tileSize * origStep after materialization.
      // Falls back to 1 for non-constant or non-divisible steps.
      SmallVector<int64_t> origSteps;
      auto outerStepVals = groupForall.getSteps();
      for (unsigned d = 0; d < tileSizes.size(); ++d) {
        int64_t origStep = 1;
        if (d < outerStepVals.size()) {
          if (auto cst =
                  outerStepVals[d].getDefiningOp<arith::ConstantIndexOp>()) {
            int64_t outerStepVal = cst.value();
            if (tileSizes[d] > 0 && outerStepVal % tileSizes[d] == 0)
              origStep = outerStepVal / tileSizes[d];
          }
        }
        origSteps.push_back(origStep);
      }

      // Run promotion plan analysis.
      SmallVector<PromotionRecord> plan =
          computePromotionPlan(groupForall, laneForall, tileSizes, origSteps);

      if (plan.empty())
        continue;

      // Emit remark describing the promotion decision.
      {
        int64_t totalFootprint = 0;
        // reuseCount = sum over records of (max halo span across dims + 1).
        // This indicates how many distinct global-memory elements are
        // accessed per output element (reuse via shared memory).
        int64_t reuseCount = 0;
        for (auto &rec : plan) {
          int64_t fp = 4; // default 4 bytes per element (f32/i32)
          for (int64_t d : rec.tileDims)
            fp *= d;
          totalFootprint += fp;
          int64_t maxSpan = 0;
          for (unsigned d = 0; d < rec.rank; ++d) {
            int64_t span = rec.maxOffset[d] - rec.minOffset[d];
            if (span > maxSpan) maxSpan = span;
          }
          reuseCount += (maxSpan + 1);
        }
        groupForall.emitRemark()
            << "promote-group-memory: promoting " << plan.size()
            << " memref(s), reuseCount=" << reuseCount
            << ", footprint ~" << totalFootprint
            << " B, memory_policy=prefer_group";
      }

      OpBuilder builder(ctx);

      // Process each promotion record.
      for (auto &rec : plan) {
        Value globalMr  = rec.globalMemref;
        auto globalType = cast<MemRefType>(globalMr.getType());
        Type  elemTy    = globalType.getElementType();

        // a. Allocate tile buffer in group address space before laneForall.
        builder.setInsertionPoint(laneForall);
        MemRefType tileType = buildTileType(ctx, elemTy, rec.tileDims);
        auto tileAlloc = builder.create<memref::AllocOp>(
            laneForall.getLoc(), tileType);
        Value tileMr = tileAlloc.getResult();

        // b. Insert cooperative copy lane forall.
        //    Iterates over the tile dimensions [0, tileDims[d]).
        SmallVector<Value> copyLbs, copyUbs, copySteps;
        for (int64_t dim : rec.tileDims) {
          copyLbs.push_back(
              builder.create<arith::ConstantIndexOp>(laneForall.getLoc(), 0));
          copyUbs.push_back(
              builder.create<arith::ConstantIndexOp>(laneForall.getLoc(), dim));
          copySteps.push_back(
              builder.create<arith::ConstantIndexOp>(laneForall.getLoc(), 1));
        }
        auto copyForall = builder.create<ForallOp>(
            laneForall.getLoc(), copyLbs, copyUbs, copySteps);
        copyForall->setAttr("spmd.mapping",
                             LevelAttr::get(ctx, LevelKind::Lane));

        // Initialize copy forall body: generated builder leaves region empty.
        Block *copyBodyPtr = builder.createBlock(&copyForall.getBody());
        for (unsigned d = 0; d < rec.rank; ++d)
          copyBodyPtr->addArgument(builder.getIndexType(), laneForall.getLoc());
        {
          OpBuilder::InsertionGuard g(builder);
          builder.setInsertionPointToEnd(copyBodyPtr);
          builder.create<YieldOp>(laneForall.getLoc());
        }

        // Fill the copy forall body: load from global → store to tile.
        Block &copyBody = *copyBodyPtr;
        builder.setInsertionPoint(copyBody.getTerminator());

        ValueRange copyIvs = copyForall.getInductionVars();
        ValueRange outerIvs = groupForall.getInductionVars();

        // Global index = outer_iv[d] + copy_iv[d] + minOffset[d]
        Location copyLoc = laneForall.getLoc();
        SmallVector<Value> globalIndices;
        for (unsigned d = 0; d < rec.rank; ++d) {
          Value idx = builder.create<arith::AddIOp>(copyLoc, outerIvs[d], copyIvs[d]);
          if (rec.minOffset[d] != 0) {
            Value off = builder.create<arith::ConstantIndexOp>(copyLoc, rec.minOffset[d]);
            idx = builder.create<arith::AddIOp>(copyLoc, idx, off);
          }
          globalIndices.push_back(idx);
        }

        // Build in-bounds condition: globalIndex[d] < globalDim[d] for all d.
        // This guards boundary tiles that are smaller than the tile size.
        Value inBounds = nullptr;
        for (unsigned d = 0; d < rec.rank; ++d) {
          Value globalUb;
          if (!globalType.isDynamicDim(d)) {
            globalUb = builder.create<arith::ConstantIndexOp>(
                copyLoc, globalType.getDimSize(d));
          } else {
            globalUb = builder.create<memref::DimOp>(copyLoc, globalMr, d);
          }
          Value cmp = builder.create<arith::CmpIOp>(
              copyLoc, arith::CmpIPredicate::ult, globalIndices[d], globalUb);
          inBounds = inBounds
                         ? builder.create<arith::AndIOp>(copyLoc, inBounds, cmp)
                         : cmp;
        }

        // Wrap load-store in spmd.if to prevent out-of-bounds access on
        // boundary tiles.
        auto guardIf =
            builder.create<IfOp>(copyLoc, TypeRange{}, inBounds,
                                  /*hasElse=*/false);
        Block &guardThen = guardIf.getThenRegion().front();
        builder.setInsertionPoint(guardThen.getTerminator());

        // Tile index = copy_iv[d] (minOffset already accounted for in globalIndex).
        SmallVector<Value> tileIndices(copyIvs.begin(), copyIvs.end());
        Value loadedVal =
            builder.create<memref::LoadOp>(copyLoc, globalMr, globalIndices);
        builder.create<memref::StoreOp>(copyLoc, loadedVal, tileMr, tileIndices);

        // c. Insert spmd.barrier after the copy forall.
        builder.setInsertionPointAfter(copyForall);
        auto barrierOp =
            builder.create<BarrierOp>(laneForall.getLoc());
        barrierOp->setAttr("spmd.scope",
                            ScopeAttr::get(ctx, ScopeKind::Group));

        // d. Rewrite compute loads in laneForall to use tile buffer.
        builder.setInsertionPointToStart(&laneForall.getBody().front());
        rewriteLoadsToTile(laneForall.getBody().front(), globalMr, tileMr,
                            rec.minOffset, outerIvs,
                            laneForall.getInductionVars(), builder);
      } // for each promotion record
    } // for each group forall
  }
};
} // namespace

std::unique_ptr<Pass> mlir::spmd::createPromoteGroupMemoryPass() {
  return std::make_unique<PromoteGroupMemoryPass>();
}

void mlir::spmd::registerPromoteGroupMemoryPass() {
  PassRegistration<PromoteGroupMemoryPass>();
}
