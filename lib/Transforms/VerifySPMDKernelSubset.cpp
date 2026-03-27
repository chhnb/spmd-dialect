// VerifySPMDKernelSubset.cpp
//
// Checks that functions marked with {spmd.kernel} only contain ops
// from the allowed subset. This is a separate pass from the op verifier
// because legality depends on context (e.g., which phase we are in).
//
// Allowed ops (S0/S1):
//   spmd.{forall, if, reduce, yield}
//   arith.*, math.*
//   memref.{load, store, subview, cast}
//   affine.apply
//   vector.* (optional)
//   func.return
//
// Disallowed in S0/S1:
//   spmd.barrier
//   memref with group/private addr space
//   gpu.*, omp.*
//   cf.*, scf.while
//   unknown side-effecting func.call

#include "spmd/IR/SPMDOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::spmd;

namespace {
struct VerifySPMDKernelSubsetPass
    : public PassWrapper<VerifySPMDKernelSubsetPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifySPMDKernelSubsetPass)

  StringRef getArgument() const override { return "verify-spmd-kernel-subset"; }
  StringRef getDescription() const override {
    return "Verify that spmd.kernel functions only use the allowed op subset";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    bool failed = false;

    mod.walk([&](func::FuncOp func) {
      if (!func->hasAttr("spmd.kernel"))
        return;

      func.walk([&](Operation *op) {
        // spmd.barrier not allowed in S0/S1 — will be added later by passes
        if (isa<BarrierOp>(op)) {
          op->emitError("spmd.barrier must not appear in S0/S1 kernel; "
                        "it is inserted by PromoteGroupMemory pass");
          failed = true;
        }

        // Disallow gpu/omp/cf ops
        StringRef dialect = op->getDialect()->getNamespace();
        if (dialect == "gpu" || dialect == "omp" || dialect == "cf") {
          op->emitError("dialect '") << dialect
            << "' is not allowed in spmd.kernel (S0/S1)";
          failed = true;
        }

        // TODO: check for group/private addr space in S0/S1 memrefs
        // TODO: check for unknown side-effecting func.call
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
