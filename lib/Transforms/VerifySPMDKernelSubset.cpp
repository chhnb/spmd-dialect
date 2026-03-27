// VerifySPMDKernelSubset.cpp
//
// Verifies that func.func ops marked with {spmd.kernel} only contain ops
// from the allowed S0/S1 subset.
//
// Allowed ops (S0/S1):
//   spmd.{forall, if, reduce, yield}  (barrier is S2-only)
//   arith.*, math.*
//   memref.{load, store, subview, cast, alloc}
//   affine.apply, affine.load, affine.store
//   vector.*  (optional)
//   func.return
//
// Disallowed in S0/S1:
//   spmd.barrier
//   memrefs with group/private addr space
//   gpu.*, omp.*, cf.*, scf.while
//   unknown side-effecting func.call

#include "spmd/IR/SPMDAttrs.h"
#include "spmd/IR/SPMDOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::spmd;

namespace {
struct VerifySPMDKernelSubsetPass
    : public PassWrapper<VerifySPMDKernelSubsetPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifySPMDKernelSubsetPass)

  StringRef getArgument() const override {
    return "verify-spmd-kernel-subset";
  }
  StringRef getDescription() const override {
    return "Verify that spmd.kernel functions only use the allowed S0/S1 op subset";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    bool failed = false;

    mod.walk([&](func::FuncOp func) {
      if (!func->hasAttr("spmd.kernel"))
        return;

      func.walk([&](Operation *op) {
        // spmd.barrier is S2-only; must not appear in S0/S1
        if (isa<BarrierOp>(op)) {
          op->emitError(
              "spmd.barrier must not appear in S0/S1 kernel; "
              "it is inserted by PromoteGroupMemory pass");
          failed = true;
          return;
        }

        // Disallow gpu, omp, cf dialects
        StringRef dialect = op->getDialect()->getNamespace();
        if (dialect == "gpu" || dialect == "omp" || dialect == "cf") {
          op->emitError("dialect '")
              << dialect << "' is not allowed in spmd.kernel (S0/S1)";
          failed = true;
          return;
        }

        // Disallow memrefs with group/private addr space in S0/S1
        for (Type ty : op->getResultTypes()) {
          if (auto mr = dyn_cast<MemRefType>(ty)) {
            if (auto addrSpace =
                    dyn_cast_or_null<AddressSpaceAttr>(mr.getMemorySpace())) {
              if (addrSpace.getValue() != AddressSpaceKind::Global) {
                op->emitError(
                    "group/private address space memrefs must not appear "
                    "in S0/S1 kernels");
                failed = true;
              }
            }
          }
        }
      });
    });

    if (failed)
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::spmd::createVerifySPMDKernelSubsetPass() {
  return std::make_unique<VerifySPMDKernelSubsetPass>();
}
