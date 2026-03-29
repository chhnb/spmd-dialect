// SPMDToGPU.cpp
//
// Lowers S2 spmd dialect IR to gpu dialect for CUDA/ROCm targets.
//
// Mapping (direct IR surgery — no ConversionPattern):
//   group-level spmd.forall → gpu.launch (gridDim/blockDim from tile_sizes)
//   lane-level  spmd.forall → thread index (linear, uses launch block args)
//   spmd.if                 → scf.if  (via greedy pattern rewrite)
//   spmd.reduce             → scf.for (via greedy pattern rewrite)
//   spmd.barrier (group)    → gpu.barrier [memfence workgroup]
//   group addr space alloc  → gpu.launch workgroup attribution +
//                             replaceAllUsesWith + erase
//
// Pass level: OperationPass<ModuleOp>
// GPU thread layout: 1D linear block (blockDim.x = product of tile_sizes).
// For 2D+ lane foralls the lane IVs are delinearized from threadIdx.x.

#include "spmd/IR/SPMDOps.h"
#include "spmd/Transforms/SPMDPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::spmd;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static bool isGroupLevel(ForallOp op) {
  auto attr = op->getAttrOfType<LevelAttr>("spmd.mapping");
  return attr && attr.getValue() == LevelKind::Group;
}

static bool isLaneLevel(ForallOp op) {
  auto attr = op->getAttrOfType<LevelAttr>("spmd.mapping");
  return attr && attr.getValue() == LevelKind::Lane;
}

// Collect group-address-space memref.alloc ops in the top-level of the body.
static SmallVector<memref::AllocOp>
collectGroupAllocs(ForallOp groupForall) {
  SmallVector<memref::AllocOp> result;
  for (Operation &op : groupForall.getBody().front()) {
    auto allocOp = dyn_cast<memref::AllocOp>(&op);
    if (!allocOp)
      continue;
    auto memTy = cast<MemRefType>(allocOp.getType());
    auto addrSpace = dyn_cast_or_null<AddressSpaceAttr>(memTy.getMemorySpace());
    if (addrSpace && addrSpace.getValue() == AddressSpaceKind::Group)
      result.push_back(allocOp);
  }
  return result;
}

// Compute {gx, gy, gz} = ceildivui((ub[d] - lb[d]), step[d]) for each dim.
// Missing dims are padded with constant 1.
static SmallVector<Value>
computeGridDim(ForallOp groupForall, OpBuilder &b) {
  Location loc = groupForall.getLoc();
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value> dims;
  auto lbs   = groupForall.getLowerBounds();
  auto ubs   = groupForall.getUpperBounds();
  auto steps = groupForall.getSteps();
  for (unsigned d = 0; d < groupForall.getRank(); ++d) {
    Value range = b.create<arith::SubIOp>(loc, ubs[d], lbs[d]);
    dims.push_back(b.create<arith::CeilDivUIOp>(loc, range, steps[d]));
  }
  while (dims.size() < 3)
    dims.push_back(c1);
  return dims;
}

// Compute the maximum linearized blockDim across all lane foralls in the body.
// Returns 1 if no lane forall is found.
static int64_t computeMaxLinearBlockDim(ForallOp groupForall) {
  int64_t maxBlock = 1;
  groupForall.getBody().front().walk([&](ForallOp laneForall) {
    if (!isLaneLevel(laneForall))
      return;
    int64_t prod = 1;
    for (Value ub : laneForall.getUpperBounds()) {
      auto cstOp = ub.getDefiningOp<arith::ConstantIndexOp>();
      if (!cstOp) { prod = -1; return; } // non-constant: can't compute
      prod *= cstOp.value();
    }
    if (prod > maxBlock)
      maxBlock = prod;
  });
  // Fallback: use tile_sizes if lane UBs are non-constant.
  if (maxBlock == 1) {
    if (auto ts = groupForall->getAttrOfType<DenseI64ArrayAttr>("spmd.tile_sizes")) {
      int64_t prod = 1;
      for (int64_t v : ts.asArrayRef())
        prod *= v;
      maxBlock = prod;
    }
  }
  return maxBlock;
}

// Delinearize a 1D thread index `tx` into per-dim IVs using constant lane UBs.
// For rank 1: {tx}.
// For rank 2 with ubs=[R, C]: {tx/C, tx%C} (row-major).
// For rank N (general): row-major delinearization.
// Returns empty on non-constant UBs.
static SmallVector<Value>
delinearizeTx(Value tx, ForallOp laneForall, OpBuilder &b) {
  Location loc = laneForall.getLoc();
  unsigned rank = laneForall.getRank();
  if (rank == 1)
    return {tx};

  // Collect constant upper bounds.
  SmallVector<int64_t> ubConsts(rank);
  for (unsigned d = 0; d < rank; ++d) {
    Value ub = laneForall.getUpperBounds()[d];
    auto cstOp = ub.getDefiningOp<arith::ConstantIndexOp>();
    if (!cstOp)
      return {}; // non-constant: fallback needed
    ubConsts[d] = cstOp.value();
  }

  // Row-major delinearization: stride[d] = product of ubConsts[d+1..]
  SmallVector<Value> ivs;
  Value remainder = tx;
  for (unsigned d = 0; d < rank; ++d) {
    int64_t stride = 1;
    for (unsigned j = d + 1; j < rank; ++j)
      stride *= ubConsts[j];
    if (d + 1 < rank) {
      Value strideVal = b.create<arith::ConstantIndexOp>(loc, stride);
      ivs.push_back(b.create<arith::DivUIOp>(loc, remainder, strideVal));
      remainder = b.create<arith::RemUIOp>(loc, remainder, strideVal);
    } else {
      ivs.push_back(remainder);
    }
  }
  return ivs;
}

//===----------------------------------------------------------------------===//
// spmd.if → scf.if  (ported from SPMDToSCF)
//===----------------------------------------------------------------------===//

struct IfToSCFIfGPU : public OpRewritePattern<IfOp> {
  using OpRewritePattern::OpRewritePattern;

  void transferRegion(PatternRewriter &rewriter, Region &src,
                      Block *dstBlock) const {
    Block &srcBlock = src.front();
    Operation *existingYield = nullptr;
    if (!dstBlock->empty() && isa<scf::YieldOp>(dstBlock->back()))
      existingYield = &dstBlock->back();
    if (existingYield) {
      rewriter.inlineBlockBefore(&srcBlock, existingYield, {});
      if (Operation *prev = existingYield->getPrevNode())
        if (isa<YieldOp>(prev))
          rewriter.eraseOp(prev);
    } else {
      rewriter.inlineBlockBefore(&srcBlock, dstBlock, dstBlock->end(), {});
      Operation *spmdYield = &dstBlock->back();
      assert(isa<YieldOp>(spmdYield) && "expected spmd.yield at end");
      auto yieldOp = cast<YieldOp>(spmdYield);
      rewriter.setInsertionPoint(spmdYield);
      rewriter.create<scf::YieldOp>(spmdYield->getLoc(), yieldOp.getValues());
      rewriter.eraseOp(spmdYield);
    }
  }

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TypeRange resultTypes = op.getResultTypes();
    bool hasElse = !op.getElseRegion().empty() &&
                   !op.getElseRegion().front().empty();
    auto newIf = rewriter.create<scf::IfOp>(loc, resultTypes,
                                             op.getCondition(), hasElse);
    transferRegion(rewriter, op.getThenRegion(), &newIf.getThenRegion().front());
    if (hasElse)
      transferRegion(rewriter, op.getElseRegion(),
                     &newIf.getElseRegion().front());
    rewriter.replaceOp(op, newIf.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// spmd.reduce → scf.for  (ported from SPMDToSCF)
//===----------------------------------------------------------------------===//

struct ReduceToSCFForGPU : public OpRewritePattern<ReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  Value buildCombine(OpBuilder &b, Location loc, ReductionKind kind,
                     Value acc, Value partial, Type type) const {
    bool isFloat = isa<FloatType>(type);
    switch (kind) {
    case ReductionKind::Add:
      return isFloat ? b.create<arith::AddFOp>(loc, acc, partial)
                     : (Value)b.create<arith::AddIOp>(loc, acc, partial);
    case ReductionKind::Mul:
      return isFloat ? b.create<arith::MulFOp>(loc, acc, partial)
                     : (Value)b.create<arith::MulIOp>(loc, acc, partial);
    case ReductionKind::Max:
      return isFloat ? b.create<arith::MaximumFOp>(loc, acc, partial)
                     : (Value)b.create<arith::MaxSIOp>(loc, acc, partial);
    case ReductionKind::Min:
      return isFloat ? b.create<arith::MinimumFOp>(loc, acc, partial)
                     : (Value)b.create<arith::MinSIOp>(loc, acc, partial);
    case ReductionKind::And:
      return b.create<arith::AndIOp>(loc, acc, partial);
    case ReductionKind::Or:
      return b.create<arith::OrIOp>(loc, acc, partial);
    case ReductionKind::Xor:
      return b.create<arith::XOrIOp>(loc, acc, partial);
    }
    llvm_unreachable("unhandled ReductionKind");
  }

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type elemType = op.getResult().getType();
    auto kindAttr = op->getAttrOfType<ReductionKindAttr>("spmd.kind");
    Block &reduceBody   = op.getBody().front();
    Value  reduceBodyIv = reduceBody.getArgument(0);
    Value  spmdPartial  = cast<YieldOp>(reduceBody.getTerminator()).getValues()[0];
    rewriter.setInsertionPoint(op);
    auto forOp = rewriter.create<scf::ForOp>(
        loc, op.getLowerBound(), op.getUpperBound(), op.getStep(),
        ValueRange{op.getInit()},
        [&](OpBuilder &b, Location bodyLoc, Value iv, ValueRange iterArgs) {
          IRMapping mapping;
          mapping.map(reduceBodyIv, iv);
          for (Operation &bodyOp : reduceBody)
            if (!isa<YieldOp>(bodyOp))
              b.clone(bodyOp, mapping);
          Value partial  = mapping.lookupOrDefault(spmdPartial);
          Value combined = buildCombine(b, bodyLoc, kindAttr.getValue(),
                                        iterArgs[0], partial, elemType);
          b.create<scf::YieldOp>(bodyLoc, ValueRange{combined});
        });
    rewriter.replaceOp(op, forOp.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lane forall lowering (direct IR surgery)
//===----------------------------------------------------------------------===//

// Lower a lane-level forall in the launch body:
//   1. Compute per-dim thread IVs from threadIdx.x (with delinearization).
//   2. Add a bounds check: scf.if (tx < lane_ub_product) { body }.
//   3. Replace lane IVs, move body ops into the scf.if, erase the forall.
static void lowerLaneForallInLaunch(ForallOp laneForall,
                                     gpu::LaunchOp launch) {
  Location loc = laneForall.getLoc();
  OpBuilder b(laneForall);

  Value tx = launch.getThreadIds().x;
  unsigned rank = laneForall.getRank();

  // Compute per-dim IVs.
  SmallVector<Value> threadIVs = delinearizeTx(tx, laneForall, b);
  if (threadIVs.empty()) {
    // Fallback for non-constant UBs: use tx for every dim.
    for (unsigned i = 0; i < rank; ++i)
      threadIVs.push_back(tx);
  }

  // Compute the product of lane UBs (for the bounds check).
  int64_t laneUBProduct = 1;
  bool allConst = true;
  for (Value ub : laneForall.getUpperBounds()) {
    auto cstOp = ub.getDefiningOp<arith::ConstantIndexOp>();
    if (!cstOp) { allConst = false; break; }
    laneUBProduct *= cstOp.value();
  }

  // Wrap body in scf.if(tx < lane_ub_product) for correctness when
  // blockDim > lane_ub_product (e.g., promoted path with copy vs compute tiles).
  // This check is trivially true when blockDim == lane_ub_product.
  Value laneUBVal = b.create<arith::ConstantIndexOp>(loc, laneUBProduct);
  Value inBounds   = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ult, tx, laneUBVal);
  auto ifOp = b.create<scf::IfOp>(loc, TypeRange{}, inBounds,
                                    /*withElse=*/false);
  // Clear the auto-generated scf.yield from the then block so we can fill it.
  ifOp.getThenRegion().front().clear();

  // Replace lane IVs with computed thread IVs.
  for (unsigned i = 0; i < rank; ++i)
    laneForall.getInductionVar(i).replaceAllUsesWith(threadIVs[i]);

  // Move ops from lane body into the scf.if then block.
  Block &laneBody = laneForall.getBody().front();
  Block &thenBlock = ifOp.getThenRegion().front();
  while (!laneBody.empty()) {
    Operation *op = &laneBody.front();
    if (isa<YieldOp>(op)) {
      op->erase();
      break;
    }
    op->moveBefore(&thenBlock, thenBlock.end());
  }
  // Add scf.yield to terminate the then block.
  OpBuilder thenBuilder(&thenBlock, thenBlock.end());
  thenBuilder.create<scf::YieldOp>(loc);

  laneForall.erase();
}

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

struct ConvertSPMDToGPUPass
    : public PassWrapper<ConvertSPMDToGPUPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertSPMDToGPUPass)

  StringRef getArgument() const override { return "convert-spmd-to-gpu"; }
  StringRef getDescription() const override {
    return "Lower spmd dialect to gpu dialect for CUDA/ROCm";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    OpBuilder b(ctx);
    bool hadError = false;

    // ── Phase 1: lower group foralls → gpu.launch ──────────────────────────
    //
    // Collect first to avoid iterator invalidation during surgery.
    SmallVector<ForallOp> groupForalls;
    module.walk([&](ForallOp op) {
      if (isGroupLevel(op))
        groupForalls.push_back(op);
    });

    for (ForallOp groupForall : groupForalls) {
      Location loc = groupForall.getLoc();
      b.setInsertionPoint(groupForall);

      // Validate: no nested group foralls allowed.
      bool hasNestedGroup = false;
      groupForall.getBody().front().walk([&](ForallOp nested) {
        if (nested != groupForall && isGroupLevel(nested)) {
          nested->emitError("nested group-level spmd.forall is not supported");
          hasNestedGroup = true;
        }
      });
      if (hasNestedGroup) { hadError = true; continue; }

      // Collect group-addr-space alloc ops.
      SmallVector<memref::AllocOp> groupAllocs = collectGroupAllocs(groupForall);

      // Compute gridDim.
      SmallVector<Value> gridDims = computeGridDim(groupForall, b);

      // Compute blockDim (max linearized size across all lane foralls).
      int64_t blockDimTotal = computeMaxLinearBlockDim(groupForall);
      if (blockDimTotal > 1024) {
        groupForall->emitError("computed blockDim ")
            << blockDimTotal << " exceeds CUDA maximum of 1024";
        hadError = true;
        continue;
      }

      Value c1       = b.create<arith::ConstantIndexOp>(loc, 1);
      Value blockX   = b.create<arith::ConstantIndexOp>(loc, blockDimTotal);
      Value gx = gridDims[0], gy = gridDims[1], gz = gridDims[2];

      // Build workgroup attribution types for group allocs.
      auto wgSpace = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
      SmallVector<Type> wgTypes;
      for (auto allocOp : groupAllocs) {
        auto memTy = cast<MemRefType>(allocOp.getType());
        wgTypes.push_back(MemRefType::get(memTy.getShape(),
                                           memTy.getElementType(),
                                           memTy.getLayout(), wgSpace));
      }

      // Create gpu.launch.
      auto launchOp = gpu::LaunchOp::create(
          b, loc, gx, gy, gz, blockX, c1, c1,
          /*dynamicSharedMemorySize=*/Value{},
          /*asyncTokenType=*/nullptr,
          /*asyncDependencies=*/ValueRange{},
          TypeRange{wgTypes},
          /*privateAttributions=*/TypeRange{});

      // Add gpu.terminator (not auto-created by build).
      Block *launchBody = &launchOp.getBody().front();
      {
        OpBuilder bodyBuilder(launchBody, launchBody->end());
        bodyBuilder.create<gpu::TerminatorOp>(loc);
      }
      Operation *gpuTerminator = launchBody->getTerminator();

      // Insert group IV computations before the terminator.
      b.setInsertionPoint(gpuTerminator);
      auto blockIds = launchOp.getBlockIds();
      SmallVector<Value> blockIdVals = {blockIds.x, blockIds.y, blockIds.z};
      auto lbs   = groupForall.getLowerBounds();
      auto steps = groupForall.getSteps();
      SmallVector<Value> groupIVs;
      for (unsigned d = 0; d < groupForall.getRank(); ++d) {
        // gi[d] = blockIdx[d] * step[d] + lb[d]
        Value gi = b.create<arith::MulIOp>(loc, blockIdVals[d], steps[d]);
        // Check for non-zero lb
        Value lb = lbs[d];
        if (auto cst = lb.getDefiningOp<arith::ConstantIndexOp>())
          if (cst.value() != 0)
            gi = b.create<arith::AddIOp>(loc, gi, lb);
        groupIVs.push_back(gi);
      }

      // Replace group forall IV uses with computed groupIVs.
      for (unsigned d = 0; d < groupForall.getRank(); ++d)
        groupForall.getInductionVar(d).replaceAllUsesWith(groupIVs[d]);

      // Replace workgroup alloc uses with launch workgroup attribution args.
      auto wgAttrs = launchOp.getWorkgroupAttributions();
      for (unsigned i = 0; i < groupAllocs.size(); ++i)
        groupAllocs[i].getResult().replaceAllUsesWith(wgAttrs[i]);

      // Erase group alloc ops (no uses remain after replaceAllUsesWith).
      for (auto allocOp : groupAllocs)
        allocOp.erase();

      // Move remaining ops from group body to before the terminator.
      Block &groupBody = groupForall.getBody().front();
      while (!groupBody.empty()) {
        Operation *op = &groupBody.front();
        if (isa<YieldOp>(op)) {
          op->erase();
          break;
        }
        op->moveBefore(gpuTerminator);
      }

      // ── Phase 2: lower lane foralls inside the launch ───────────────────
      //
      // Collect lane foralls (in program order; walk visits post-order for
      // nested ops, so innermost are handled first — correct for surgery).
      SmallVector<ForallOp> laneForalls;
      launchOp.walk([&](ForallOp lf) {
        if (isLaneLevel(lf))
          laneForalls.push_back(lf);
      });
      for (ForallOp laneForall : laneForalls)
        lowerLaneForallInLaunch(laneForall, launchOp);

      // ── Phase 3: lower spmd.barrier → gpu.barrier ──────────────────────
      bool hasWorkgroupMem = !groupAllocs.empty();
      SmallVector<BarrierOp> barriers;
      launchOp.walk([&](BarrierOp barrier) {
        barriers.push_back(barrier);
      });
      for (BarrierOp barrier : barriers) {
        // Structural check: barrier must not be inside a scf.if.
        // (It was moved from the group forall body and should already be
        // at the launch body level, which is the convergent point.)
        Operation *parent = barrier->getParentOp();
        if (!isa<gpu::LaunchOp>(parent)) {
          barrier->emitError("gpu.barrier must be at gpu.launch body level; "
                             "found nested inside ")
              << parent->getName();
          hadError = true;
        }
        OpBuilder barrierBuilder(barrier);
        if (hasWorkgroupMem) {
          // gpu.barrier with workgroup memfence.
          auto wgAddrSpaceAttr = gpu::AddressSpaceAttr::get(
              ctx, gpu::AddressSpace::Workgroup);
          auto addrSpacesAttr = ArrayAttr::get(ctx, {wgAddrSpaceAttr});
          gpu::BarrierOp::create(barrierBuilder, barrier.getLoc(),
                                  addrSpacesAttr);
        } else {
          gpu::BarrierOp::create(barrierBuilder, barrier.getLoc());
        }
        barrier.erase();
      }

      // Set gpu.container_module on the enclosing module.
      if (!module->hasAttr(gpu::GPUDialect::getContainerModuleAttrName()))
        module->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                        UnitAttr::get(ctx));

      // Erase the original group forall (body is now in the launch).
      groupForall.erase();
    }

    // ── Phase 4: lower spmd.if / spmd.reduce via greedy patterns ──────────
    RewritePatternSet patterns(ctx);
    patterns.add<IfToSCFIfGPU, ReduceToSCFForGPU>(ctx);
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();

    if (hadError)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::spmd::createSPMDToGPUPass() {
  return std::make_unique<ConvertSPMDToGPUPass>();
}

void mlir::spmd::registerConvertSPMDToGPUPass() {
  PassRegistration<ConvertSPMDToGPUPass>();
}
