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
//
//   VerifySPMDGPUReady (--verify-spmd-gpu-ready)
//     Run before (or after) --convert-spmd-to-gpu. Checks structural
//     preconditions for GPU lowering:
//       - No spmd.barrier nested inside an scf.if block (divergent path).
//       - No spmd.reduce remaining inside a gpu.launch body (would be
//         unhandled residual after partial lowering).

#include "spmd/IR/SPMDAttrs.h"
#include "spmd/IR/SPMDOps.h"
#include "spmd/Transforms/SPMDPasses.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

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

    getOperation().walk([&](memref::AllocOp allocOp) {
      auto memTy = cast<MemRefType>(allocOp.getType());
      auto addrSpace =
          dyn_cast_or_null<AddressSpaceAttr>(memTy.getMemorySpace());
      if (addrSpace && addrSpace.getValue() == AddressSpaceKind::Group) {
        allocOp.emitError(
            "group-address-space memref.alloc should not exist after "
            "convert-spmd-to-gpu; it should have been converted to a "
            "gpu.workgroup attribution and erased");
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
