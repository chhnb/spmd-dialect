// SPMDToGPU.cpp
//
// Lowers S2 spmd dialect IR to gpu dialect for CUDA/ROCm targets.
//
// Mapping (direct IR surgery — no ConversionPattern):
//   group-level spmd.forall → gpu.launch (gridDim/blockDim from tile_sizes)
//   lane-level  spmd.forall → thread index (linear, uses launch block args)
//   spmd.if                 → scf.if  (via greedy pattern rewrite)
//   spmd.reduce (f32 Add, inside gpu.launch, constant blockDim, feeds one
//               rank-0 atomic_rmw addf, pure body)
//               → workgroup-memory tree reduction + single atomic per block
//                 (ReduceToHierarchicalGPU, benefit=2)
//   spmd.reduce (all other cases) → scf.for (ReduceToSCFForGPU, benefit=1)
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
#include "llvm/Support/MathExtras.h"

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

// Collect all group-address-space memref.alloc ops in the group forall body.
// Walks the full body (not just top-level ops) to match any allocs that the
// promote-group-memory pass may have inserted at any nesting depth.
// In practice these are always at the top level, but a full walk is more robust.
static SmallVector<memref::AllocOp>
collectGroupAllocs(ForallOp groupForall) {
  SmallVector<memref::AllocOp> result;
  groupForall.getBody().front().walk([&](memref::AllocOp allocOp) {
    auto memTy = cast<MemRefType>(allocOp.getType());
    auto addrSpace = dyn_cast_or_null<AddressSpaceAttr>(memTy.getMemorySpace());
    if (addrSpace && addrSpace.getValue() == AddressSpaceKind::Group)
      result.push_back(allocOp);
  });
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
// Uses trip counts ceildiv(ub[d]-lb[d], step[d]) per dim, not raw UBs.
// Returns 1 if no lane forall is found.
static int64_t computeMaxLinearBlockDim(ForallOp groupForall) {
  int64_t maxBlock = 1;
  groupForall.getBody().front().walk([&](ForallOp laneForall) {
    if (!isLaneLevel(laneForall))
      return;
    int64_t prod = 1;
    auto lbs   = laneForall.getLowerBounds();
    auto ubs   = laneForall.getUpperBounds();
    auto steps = laneForall.getSteps();
    for (unsigned d = 0; d < laneForall.getRank(); ++d) {
      auto ubCst   = ubs[d].getDefiningOp<arith::ConstantIndexOp>();
      auto lbCst   = lbs[d].getDefiningOp<arith::ConstantIndexOp>();
      auto stepCst = steps[d].getDefiningOp<arith::ConstantIndexOp>();
      if (!ubCst || !lbCst || !stepCst) { prod = -1; return; }
      prod *= llvm::divideCeil(ubCst.value() - lbCst.value(), stepCst.value());
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

// Delinearize a 1D thread index `tx` into per-dim 0-based indices using
// constant trip counts ceildiv(ub[d]-lb[d], step[d]) as row-major strides.
// Returns empty on non-constant bounds (caller must handle).
// Note: returns 0-based indices — caller reconstructs iv = lb + idx * step.
static SmallVector<Value>
delinearizeTx(Value tx, ForallOp laneForall, OpBuilder &b) {
  Location loc = laneForall.getLoc();
  unsigned rank = laneForall.getRank();

  // Collect constant trip counts per dim.
  auto lbs   = laneForall.getLowerBounds();
  auto ubs   = laneForall.getUpperBounds();
  auto steps = laneForall.getSteps();
  SmallVector<int64_t> tripConsts(rank);
  for (unsigned d = 0; d < rank; ++d) {
    auto ubCst   = ubs[d].getDefiningOp<arith::ConstantIndexOp>();
    auto lbCst   = lbs[d].getDefiningOp<arith::ConstantIndexOp>();
    auto stepCst = steps[d].getDefiningOp<arith::ConstantIndexOp>();
    if (!ubCst || !lbCst || !stepCst)
      return {}; // non-constant: caller handles
    tripConsts[d] = llvm::divideCeil(ubCst.value() - lbCst.value(), stepCst.value());
  }

  // Row-major delinearization: stride[d] = product of tripConsts[d+1..]
  SmallVector<Value> idxs;
  Value remainder = tx;
  for (unsigned d = 0; d < rank; ++d) {
    int64_t stride = 1;
    for (unsigned j = d + 1; j < rank; ++j)
      stride *= tripConsts[j];
    if (d + 1 < rank) {
      Value strideVal = b.create<arith::ConstantIndexOp>(loc, stride);
      idxs.push_back(b.create<arith::DivUIOp>(loc, remainder, strideVal));
      remainder = b.create<arith::RemUIOp>(loc, remainder, strideVal);
    } else {
      idxs.push_back(remainder);
    }
  }
  return idxs; // 0-based indices
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
// spmd.reduce → hierarchical shared-memory tree + single atomic per block
//
// Fires when ALL legality conditions hold:
//   L1: result type is f32
//   L2: spmd.kind = Add
//   L3: reduce body contains only arith.*, math.*, memref.load ops (no calls,
//       no stores, no other side-effecting ops)
//   L4: reduce op is inside a gpu.launch body
//   L5: the launch's blockDim.x is a compile-time constant
//   L6: reduce result feeds exactly one memref.atomic_rmw addf on a rank-0 memref
//
// When any condition fails, the op is left for ReduceToSCFForGPU (fallback).
//===----------------------------------------------------------------------===//

struct ReduceToHierarchicalGPU : public OpRewritePattern<ReduceOp> {
  ReduceToHierarchicalGPU(MLIRContext *ctx)
      : OpRewritePattern<ReduceOp>(ctx, /*benefit=*/2) {}

  // Returns true if the reduce body has any side-effecting op other than loads.
  static bool bodyHasSideEffects(ReduceOp op) {
    bool found = false;
    op.getBody().front().walk([&](Operation *inner) {
      if (found) return;
      if (isa<YieldOp>(inner)) return;
      // Allow arith and math dialect ops and memref.load.
      if (inner->getDialect() ==
          inner->getContext()->getLoadedDialect("arith"))
        return;
      if (inner->getDialect() ==
          inner->getContext()->getLoadedDialect("math"))
        return;
      if (isa<memref::LoadOp>(inner)) return;
      found = true;
    });
    return found;
  }

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const override {
    // L1: f32 result
    if (!isa<Float32Type>(op.getResult().getType()))
      return failure();

    // L2: Add kind
    auto kindAttr = op->getAttrOfType<ReductionKindAttr>("spmd.kind");
    if (!kindAttr || kindAttr.getValue() != ReductionKind::Add)
      return failure();

    // L3: pure body
    if (bodyHasSideEffects(op)) {
      op.emitRemark("hierarchical reduction lowering skipped: "
                    "non-pure reduce body");
      return failure();
    }

    // L4: inside a gpu.launch
    auto launchOp = op->getParentOfType<gpu::LaunchOp>();
    if (!launchOp) return failure();

    // L5: compile-time constant blockDim.x
    auto blockDimCst =
        launchOp.getBlockSizeX().getDefiningOp<arith::ConstantIndexOp>();
    if (!blockDimCst) return failure();
    int64_t blockDim = blockDimCst.value();
    if (blockDim < 2 || (blockDim & (blockDim - 1)) != 0)
      return failure(); // must be a power of two ≥ 2 for tree reduction

    // L6: result used only by one memref.atomic_rmw addf on a rank-0 memref
    Value reduceResult = op.getResult();
    if (!reduceResult.hasOneUse()) return failure();
    auto atomicOp =
        dyn_cast<memref::AtomicRMWOp>(*reduceResult.getUsers().begin());
    if (!atomicOp) return failure();
    if (atomicOp.getKind() != arith::AtomicRMWKind::addf) return failure();
    if (cast<MemRefType>(atomicOp.getMemref().getType()).getRank() != 0)
      return failure();

    // ── All checks passed. Build hierarchical reduction. ──────────────────

    Location loc = op.getLoc();
    MLIRContext *ctx = getContext();
    auto f32Ty = rewriter.getF32Type();

    Value tx = launchOp.getThreadIds().x;
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // Workgroup address space and memory fence (used by gpu.barrier).
    auto wgAddrSpace =
        gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
    auto wgMemFence = ArrayAttr::get(ctx, {wgAddrSpace});

    // Scratch buffer: one f32 slot per thread for the scatter + tree path.
    auto smemType = MemRefType::get(ArrayRef<int64_t>{blockDim}, f32Ty,
                                      /*map=*/AffineMap{}, wgAddrSpace);
    Value smem = launchOp.addWorkgroupAttribution(smemType, loc);

    rewriter.setInsertionPoint(op);

    // Re-materialize any constant operands that the normalize-spmd pass hoisted
    // to the host function scope.  Values defined outside the gpu.launch would
    // be captured by gpu-kernel-outlining and turned into kernel parameters
    // (~1 µs overhead each on B200), breaking the AC-7 speedup target.
    // By recreating them as arith.constant ops at the current insertion point
    // (inside the launch body), they become PTX immediates instead.
    auto rematerializeExternal = [&](Value v) -> Value {
      if (!v.getDefiningOp() || launchOp->isAncestor(v.getDefiningOp()))
        return v; // block arg or already inside the launch
      if (auto cst = v.getDefiningOp<arith::ConstantOp>())
        return rewriter.create<arith::ConstantOp>(loc, cst.getValue());
      return v;
    };

    // Step 1: thread-strided local accumulation.
    // Thread tx processes elements at indices: lb + tx*step, lb + (tx+blockDim)*step, …
    Block &reduceBody    = op.getBody().front();
    Value  reduceBodyIv  = reduceBody.getArgument(0);
    Value  yieldedValue  =
        cast<YieldOp>(reduceBody.getTerminator()).getValues()[0];

    Value blockDimVal = rewriter.create<arith::ConstantIndexOp>(loc, blockDim);
    Value lb   = rematerializeExternal(op.getLowerBound());
    Value ub   = rematerializeExternal(op.getUpperBound());
    Value step = rematerializeExternal(op.getStep());
    Value init = rematerializeExternal(op.getInit());
    Value txScaled    = rewriter.create<arith::MulIOp>(loc, tx, step);
    Value startIdx    = rewriter.create<arith::AddIOp>(loc, lb, txScaled);
    Value stridedStep = rewriter.create<arith::MulIOp>(loc, blockDimVal, step);

    auto forOp = rewriter.create<scf::ForOp>(
        loc, startIdx, ub, stridedStep,
        ValueRange{init},
        [&](OpBuilder &fb, Location fl, Value iv, ValueRange iterArgs) {
          IRMapping mapping;
          mapping.map(reduceBodyIv, iv);
          // Remap external constants referenced in the reduce body so that
          // cloned ops use fresh arith.constant ops inside the kernel rather
          // than the host-scope originals (which would become kernel params).
          for (Operation &inner : reduceBody) {
            for (Value operand : inner.getOperands()) {
              if (mapping.contains(operand)) continue;
              if (!operand.getDefiningOp() ||
                  launchOp->isAncestor(operand.getDefiningOp()))
                continue;
              if (auto cst = operand.getDefiningOp<arith::ConstantOp>())
                mapping.map(operand,
                            fb.create<arith::ConstantOp>(fl, cst.getValue()));
            }
          }
          for (Operation &inner : reduceBody)
            if (!isa<YieldOp>(inner))
              fb.clone(inner, mapping);
          Value partial  = mapping.lookupOrDefault(yieldedValue);
          Value combined = fb.create<arith::AddFOp>(fl, iterArgs[0], partial);
          fb.create<scf::YieldOp>(fl, ValueRange{combined});
        });
    Value threadPartial = forOp.getResult(0);

    // Step 2: scatter thread partial into workgroup scratch; sync.
    rewriter.create<memref::StoreOp>(loc, threadPartial, smem, ValueRange{tx});
    rewriter.create<gpu::BarrierOp>(loc, wgMemFence);

    // Step 3: statically-unrolled tree — log2(blockDim) scf.if + gpu.barrier
    // pairs, stride halving from blockDim/2 down to 1.  Static unrolling is
    // required because a halving stride cannot be expressed as a scf.for with
    // a fixed step; blockDim is a compile-time constant so the loop bound is
    // fixed and small (8 iterations for blockDim=256).
    for (int64_t stride = blockDim / 2; stride >= 1; stride /= 2) {
      Value strideVal = rewriter.create<arith::ConstantIndexOp>(loc, stride);
      Value cond      = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, tx, strideVal);
      auto ifOp = rewriter.create<scf::IfOp>(
          loc, TypeRange{}, cond, /*withElseRegion=*/false);
      {
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
        Value a        = rewriter.create<memref::LoadOp>(loc, smem, ValueRange{tx});
        Value peerIdx  = rewriter.create<arith::AddIOp>(loc, tx, strideVal);
        Value b_val    = rewriter.create<memref::LoadOp>(loc, smem, ValueRange{peerIdx});
        Value s        = rewriter.create<arith::AddFOp>(loc, a, b_val);
        rewriter.create<memref::StoreOp>(loc, s, smem, ValueRange{tx});
      }
      rewriter.create<gpu::BarrierOp>(loc, wgMemFence);
    }

    // Step 4: thread 0 flushes block sum to global accumulator.
    Value txIsZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, tx, c0);
    auto guardOp = rewriter.create<scf::IfOp>(
        loc, TypeRange{}, txIsZero, /*withElseRegion=*/false);
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(&guardOp.getThenRegion().front());
      Value blockSum = rewriter.create<memref::LoadOp>(loc, smem, ValueRange{c0});
      rewriter.create<memref::AtomicRMWOp>(
          loc, arith::AtomicRMWKind::addf, blockSum,
          atomicOp.getMemref(), atomicOp.getIndices());
    }

    // Erase the original atomic_rmw (its operand %sum is still live here),
    // then erase the reduce op (now has no uses).
    rewriter.eraseOp(atomicOp);
    rewriter.eraseOp(op);
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
//   1. Compute trip counts ceildiv(ub[d]-lb[d], step[d]) per dim.
//   2. For multi-dim with non-constant bounds: emit error (unsupported).
//   3. For rank-1 with dynamic bounds: use tx as 0-based index; guard via
//      runtime ceildivui(ub-lb, step).
//   4. For constant bounds: delinearize tx using trip-count strides.
//   5. Reconstruct IVs as iv[d] = lb[d] + idx[d] * step[d].
//   6. Wrap body in scf.if(tx < trip_product) and move ops in.
static LogicalResult lowerLaneForallInLaunch(ForallOp laneForall,
                                              gpu::LaunchOp launch) {
  Location loc = laneForall.getLoc();
  OpBuilder b(laneForall);

  Value tx = launch.getThreadIds().x;
  unsigned rank = laneForall.getRank();
  auto lbs   = laneForall.getLowerBounds();
  auto ubs   = laneForall.getUpperBounds();
  auto steps = laneForall.getSteps();

  // Compute per-dim trip counts. Detect non-constant dims.
  SmallVector<int64_t> tripConsts(rank, -1);
  bool allConst = true;
  for (unsigned d = 0; d < rank; ++d) {
    auto ubCst   = ubs[d].getDefiningOp<arith::ConstantIndexOp>();
    auto lbCst   = lbs[d].getDefiningOp<arith::ConstantIndexOp>();
    auto stepCst = steps[d].getDefiningOp<arith::ConstantIndexOp>();
    if (!ubCst || !lbCst || !stepCst) { allConst = false; break; }
    tripConsts[d] = llvm::divideCeil(ubCst.value() - lbCst.value(), stepCst.value());
  }

  // Multi-dim with non-constant bounds is unsupported.
  if (!allConst && rank > 1) {
    laneForall->emitError(
        "dynamic multi-dim lane forall is not supported for GPU lowering");
    return failure();
  }

  // Compute the guard value (trip product) and 0-based indices.
  Value guardVal;
  SmallVector<Value> idxs;

  if (allConst) {
    // Delinearize tx using trip-count strides.
    if (rank == 1) {
      idxs.push_back(tx);
    } else {
      idxs = delinearizeTx(tx, laneForall, b);
      assert(!idxs.empty() && "delinearizeTx failed with all-const bounds");
    }
    int64_t tripProd = 1;
    for (int64_t t : tripConsts) tripProd *= t;
    guardVal = b.create<arith::ConstantIndexOp>(loc, tripProd);
  } else {
    // rank == 1, dynamic: 0-based index is tx; guard = ceildivui(ub-lb, step).
    idxs.push_back(tx);
    Value range = b.create<arith::SubIOp>(loc, ubs[0], lbs[0]);
    guardVal = b.create<arith::CeilDivUIOp>(loc, range, steps[0]);
  }

  // Reconstruct per-dim IVs: iv[d] = lb[d] + idx[d] * step[d].
  SmallVector<Value> threadIVs;
  for (unsigned d = 0; d < rank; ++d) {
    Value iv = b.create<arith::MulIOp>(loc, idxs[d], steps[d]);
    iv = b.create<arith::AddIOp>(loc, iv, lbs[d]);
    threadIVs.push_back(iv);
  }

  // Wrap body in scf.if(tx < trip_product) for correctness when
  // blockDim > trip_product (e.g., promoted path with copy vs compute tiles).
  Value inBounds = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ult, tx, guardVal);
  auto ifOp = b.create<scf::IfOp>(loc, TypeRange{}, inBounds,
                                    /*withElse=*/false);
  // Clear the auto-generated scf.yield so we can fill the then block.
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
  // Terminate the then block.
  OpBuilder thenBuilder(&thenBlock, thenBlock.end());
  thenBuilder.create<scf::YieldOp>(loc);

  laneForall.erase();
  return success();
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

    // Lower group foralls → gpu.launch.
    //
    // Collect top-level group foralls only (not nested inside another forall).
    // Walking post-order: avoid adding inner group foralls that the outer will
    // catch and error on — otherwise the inner one gets processed first.
    SmallVector<ForallOp> groupForalls;
    module.walk([&](ForallOp op) {
      if (!isGroupLevel(op)) return;
      // Skip if this group forall is nested inside another forall.
      if (op->getParentOfType<ForallOp>()) return;
      groupForalls.push_back(op);
    });

    for (ForallOp groupForall : groupForalls) {
      Location loc = groupForall.getLoc();
      b.setInsertionPoint(groupForall);

      // Validate: rank must be ≤ 3 (GPU has 3D grid).
      if (groupForall.getRank() > 3) {
        groupForall->emitError("group forall rank > 3 is not supported for "
                               "GPU lowering (max 3D grid)");
        hadError = true;
        continue;
      }

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

      // Emit remark with launch configuration for diagnostics.
      {
        bool has2DFlatten = false;
        groupForall.getBody().front().walk([&](ForallOp lf) {
          if (isLaneLevel(lf) && lf.getRank() > 1)
            has2DFlatten = true;
        });
        auto remark = groupForall.emitRemark();
        // Report gridDim: try to fold each component to a constant; emit ? for
        // dynamic dimensions (e.g., when the problem size N is a function arg).
        remark << "convert-spmd-to-gpu: gridDim=";
        for (unsigned i = 0; i < gridDims.size(); ++i) {
          if (i > 0) remark << "x";
          if (auto cst = gridDims[i].getDefiningOp<arith::ConstantIndexOp>())
            remark << cst.value();
          else
            remark << "?";
        }
        remark << " blockDim=" << blockDimTotal
               << " workgroupBuffers=" << groupAllocs.size();
        if (has2DFlatten)
          remark << " 2DLaneFlattening=true";
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
        // gi[d] = blockIdx[d] * step[d] + lb[d]  (always add lb, even if 0)
        Value gi = b.create<arith::MulIOp>(loc, blockIdVals[d], steps[d]);
        gi = b.create<arith::AddIOp>(loc, gi, lbs[d]);
        groupIVs.push_back(gi);
      }

      // Replace group forall IV uses with computed groupIVs.
      for (unsigned d = 0; d < groupForall.getRank(); ++d)
        groupForall.getInductionVar(d).replaceAllUsesWith(groupIVs[d]);

      // Replace workgroup alloc uses with launch workgroup attribution args.
      auto wgAttrs = launchOp.getWorkgroupAttributions();
      for (unsigned i = 0; i < groupAllocs.size(); ++i)
        groupAllocs[i].getResult().replaceAllUsesWith(wgAttrs[i]);

      // Erase group alloc ops — assert zero uses after replaceAllUsesWith.
      for (auto allocOp : groupAllocs) {
        assert(allocOp->use_empty() &&
               "group alloc still has uses after workgroup rebinding");
        allocOp.erase();
      }

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

      // Lower lane foralls inside the launch.
      //
      // Collect lane foralls (in program order; walk visits post-order for
      // nested ops, so innermost are handled first — correct for surgery).
      SmallVector<ForallOp> laneForalls;
      launchOp.walk([&](ForallOp lf) {
        if (isLaneLevel(lf))
          laneForalls.push_back(lf);
      });
      for (ForallOp laneForall : laneForalls)
        if (failed(lowerLaneForallInLaunch(laneForall, launchOp)))
          hadError = true;

      // Lower spmd.barrier → gpu.barrier.
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

    // Lower spmd.if / spmd.reduce via greedy patterns.
    // ReduceToHierarchicalGPU has benefit=2 (higher priority than
    // ReduceToSCFForGPU's default benefit=1); it falls back automatically
    // for any op that fails the legality checks.
    //
    // Region-simplification is disabled so that arith.constant ops created
    // inside the gpu.launch body by ReduceToHierarchicalGPU stay there.
    // gpu.launch is NOT isolated-from-above, so the default "Aggressive" CSE
    // would move the tree-stride constants (blockDim, each halving stride, and
    // the loop trip-count) to the enclosing host function.  gpu-kernel-outlining
    // would then treat them as captured values and add extra kernel parameters
    // (~1 µs each on B200), increasing launch overhead.
    RewritePatternSet patterns(ctx);
    patterns.add<IfToSCFIfGPU>(ctx);
    patterns.add<ReduceToHierarchicalGPU>(ctx);
    patterns.add<ReduceToSCFForGPU>(ctx);
    // Disable region simplification AND constant CSE so that arith.constant
    // ops stay inside the gpu.launch body where ReduceToHierarchicalGPU places
    // them.  OperationFolder (used by the greedy rewriter) would otherwise move
    // constants into parent regions for "more aggressive CSE'ing", hoisting the
    // tree-stride constants to the host function and turning them into kernel params.
    GreedyRewriteConfig cfg;
    cfg.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    cfg.enableConstantCSE(false);
    if (failed(applyPatternsGreedily(module, std::move(patterns), cfg)))
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
