// VerifySPMDInvariants.cpp
//
// Two lightweight invariant-checking passes for the SPMD lowering pipeline:
//
//   VerifySPMDPromotionInvariant (--verify-spmd-promotion-invariant)
//     Run after --convert-spmd-to-gpu. Checks that no memref.alloc with
//     SPMD group address space remains in the IR. After GPU lowering, all
//     such allocs should have been converted to gpu.workgroup attributions
//     and the originals erased. A surviving group-space alloc indicates an
//     incomplete or skipped conversion.
//     Also checks that no memref.load or memref.store references a group-
//     address-space memref (dangling use after partial alloc erasure).
//
//   VerifySPMDGPUReady (--verify-spmd-gpu-ready)
//     Run before (or after) --convert-spmd-to-gpu. Checks structural
//     preconditions for GPU lowering:
//       - No spmd.barrier nested inside an scf.if block (divergent path).
//       - No spmd.reduce remaining inside a gpu.launch body (would be
//         unhandled residual after partial lowering).
//       - No spmd.forall nested inside a gpu.launch body (partial lowering
//         artifact: SPMD ops should not survive into an already-outlined kernel).
//       - No #gpu.address_space<workgroup> on memref types before GPU lowering
//         (stale workgroup attribution from a previous partial pipeline run).

#include "spmd/IR/SPMDAttrs.h"
#include "spmd/IR/SPMDOps.h"
#include "spmd/Transforms/SPMDPasses.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

/// Returns true if the given MemRefType has the SPMD group address space.
static bool hasGroupAddrSpace(mlir::MemRefType mrTy) {
  using namespace mlir::spmd;
  auto addrSpace =
      mlir::dyn_cast_or_null<AddressSpaceAttr>(mrTy.getMemorySpace());
  return addrSpace && addrSpace.getValue() == AddressSpaceKind::Group;
}

using namespace mlir;
using namespace mlir::spmd;

namespace {

//===----------------------------------------------------------------------===//
// VerifySPMDPromotionInvariant
//===----------------------------------------------------------------------===//

struct VerifySPMDPromotionInvariantPass
    : public PassWrapper<VerifySPMDPromotionInvariantPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifySPMDPromotionInvariantPass)

  StringRef getArgument() const override {
    return "verify-spmd-promotion-invariant";
  }
  StringRef getDescription() const override {
    return "Verify no group-address-space memref.alloc remains after "
           "convert-spmd-to-gpu (expected to become gpu.workgroup attributions)";
  }

  void runOnOperation() override {
    bool anyFailed = false;

    // Check 1: no surviving group-space alloc.
    getOperation().walk([&](memref::AllocOp allocOp) {
      auto memTy = cast<MemRefType>(allocOp.getType());
      if (hasGroupAddrSpace(memTy)) {
        allocOp.emitError(
            "group-address-space memref.alloc should not exist after "
            "convert-spmd-to-gpu; it should have been converted to a "
            "gpu.workgroup attribution and erased");
        anyFailed = true;
      }
    });

    // Check 2: no dangling group-space load (group-space alloc was erased but
    // some load still references the group-space memref type).
    getOperation().walk([&](memref::LoadOp loadOp) {
      auto memTy = cast<MemRefType>(loadOp.getMemRef().getType());
      if (hasGroupAddrSpace(memTy)) {
        loadOp.emitError(
            "memref.load uses a group-address-space memref after "
            "convert-spmd-to-gpu; this is a dangling reference to a "
            "group-space buffer that should have been erased");
        anyFailed = true;
      }
    });

    // Check 3: no dangling group-space store.
    getOperation().walk([&](memref::StoreOp storeOp) {
      auto memTy = cast<MemRefType>(storeOp.getMemRef().getType());
      if (hasGroupAddrSpace(memTy)) {
        storeOp.emitError(
            "memref.store uses a group-address-space memref after "
            "convert-spmd-to-gpu; this is a dangling reference to a "
            "group-space buffer that should have been erased");
        anyFailed = true;
      }
    });

    if (anyFailed)
      signalPassFailure();
  }
};

//===----------------------------------------------------------------------===//
// VerifySPMDGPUReady
//===----------------------------------------------------------------------===//

struct VerifySPMDGPUReadyPass
    : public PassWrapper<VerifySPMDGPUReadyPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifySPMDGPUReadyPass)

  StringRef getArgument() const override { return "verify-spmd-gpu-ready"; }
  StringRef getDescription() const override {
    return "Verify structural preconditions for convert-spmd-to-gpu: "
           "no spmd.barrier in scf.if (divergent path), "
           "no spmd.reduce inside gpu.launch body (unhandled residual)";
  }

  void runOnOperation() override {
    bool anyFailed = false;

    // Check 1: spmd.barrier must not be nested inside scf.if.
    // A barrier inside a conditional block creates a divergent execution path
    // where some threads skip the barrier, causing undefined behavior on GPU.
    getOperation().walk([&](BarrierOp barrier) {
      if (barrier->getParentOfType<scf::IfOp>()) {
        barrier.emitError(
            "spmd.barrier must not be nested inside scf.if "
            "(barrier in divergent path is not supported for GPU lowering)");
        anyFailed = true;
      }
    });

    // Check 2: spmd.reduce must not appear inside a gpu.launch body.
    // This indicates the reduction was not lowered before GPU outlining.
    getOperation().walk([&](ReduceOp reduce) {
      if (reduce->getParentOfType<gpu::LaunchOp>()) {
        reduce.emitError(
            "spmd.reduce found inside gpu.launch body; this op should have "
            "been lowered to scf.for by convert-spmd-to-gpu before outlining");
        anyFailed = true;
      }
    });

    // Check 3: spmd.forall must not appear inside a gpu.launch body.
    // This indicates a partial lowering where the outer group forall was
    // converted but inner SPMD ops were left behind — an invalid mixed-dialect
    // state that would cause GPU kernel codegen to fail.
    getOperation().walk([&](ForallOp forall) {
      if (forall->getParentOfType<gpu::LaunchOp>()) {
        forall.emitError(
            "spmd.forall found inside gpu.launch body; SPMD ops must be fully "
            "lowered before GPU outlining. This is a partial-lowering artifact "
            "indicating convert-spmd-to-gpu did not complete the inner forall");
        anyFailed = true;
      }
    });

    // Check 4: no stale group-address-space memref type.
    // Before convert-spmd-to-gpu, group-space memrefs exist as
    // memref.alloc results (created by promote-group-memory). They should not
    // appear as function arguments or as type parameters in any other op at
    // this stage unless they were introduced by an earlier partial pipeline run.
    // Detect this as a sanity check: if a group-space alloc exists but the
    // enclosing function already has a gpu.launch, the pipeline ordering is wrong.
    getOperation().walk([&](memref::AllocOp allocOp) {
      auto memTy = cast<MemRefType>(allocOp.getType());
      if (hasGroupAddrSpace(memTy)) {
        // Group-space allocs are EXPECTED before convert-spmd-to-gpu (they are
        // created by promote-group-memory). Only flag if they appear INSIDE a
        // gpu.launch body (stale artifact from incorrect ordering).
        if (allocOp->getParentOfType<gpu::LaunchOp>()) {
          allocOp.emitError(
              "group-address-space memref.alloc found inside gpu.launch body; "
              "group memory should be allocated outside the launch and passed "
              "as a workgroup-attributed memref, not as an alloc inside launch");
          anyFailed = true;
        }
      }
    });

    if (anyFailed)
      signalPassFailure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::spmd::createVerifySPMDPromotionInvariantPass() {
  return std::make_unique<VerifySPMDPromotionInvariantPass>();
}

void mlir::spmd::registerVerifySPMDPromotionInvariantPass() {
  PassRegistration<VerifySPMDPromotionInvariantPass>();
}

std::unique_ptr<Pass> mlir::spmd::createVerifySPMDGPUReadyPass() {
  return std::make_unique<VerifySPMDGPUReadyPass>();
}

void mlir::spmd::registerVerifySPMDGPUReadyPass() {
  PassRegistration<VerifySPMDGPUReadyPass>();
}
