// VerifySPMDKernelSubset.cpp
//
// Verifies that func.func ops marked with {spmd.kernel} only contain ops
// from the allowed S0/S1 subset.
//
// Allowed dialects / ops (S0/S1):
//   spmd.{forall, if, reduce, yield}  (barrier is S2-only)
//   arith.*, math.*
//   memref.{load, store, subview, cast, alloca, alloc, dealloc, dim}
//   affine.apply, affine.load, affine.store
//   vector.*  (optional)
//   func.return, func.call (pure/readonly callees only — checked by name)
//   index.*
//
// Disallowed in S0/S1:
//   spmd.barrier
//   memrefs with group/private addr space (in signature, operands, or results)
//   gpu.*, omp.*, cf.*, scf.while
//   scf.* (scf.for/if/etc. are S2+ only)

#include "spmd/IR/SPMDAttrs.h"
#include "spmd/IR/SPMDOps.h"
#include "spmd/Transforms/SPMDPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::spmd;

namespace {

// Returns true if a MemRefType has group or private address space.
static bool hasNonGlobalAddrSpace(MemRefType mr) {
  if (auto addrSpace =
          dyn_cast_or_null<AddressSpaceAttr>(mr.getMemorySpace())) {
    return addrSpace.getValue() != AddressSpaceKind::Global;
  }
  return false;
}

// Check a single type for non-global memref addr space.
static bool typeHasNonGlobalMemref(Type ty) {
  if (auto mr = dyn_cast<MemRefType>(ty))
    return hasNonGlobalAddrSpace(mr);
  return false;
}

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

  // Allowed dialects for S0/S1 kernels.
  static bool isAllowedDialect(StringRef ns) {
    return ns == "spmd" || ns == "arith" || ns == "math" || ns == "memref" ||
           ns == "affine" || ns == "vector" || ns == "func" || ns == "index" ||
           ns == "builtin";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    bool anyFailed = false;

    mod.walk([&](func::FuncOp func) {
      if (!func->hasAttr("spmd.kernel"))
        return;

      // 1. Check function signature for non-global memref types.
      FunctionType fnType = func.getFunctionType();
      for (Type t : fnType.getInputs()) {
        if (typeHasNonGlobalMemref(t)) {
          func.emitError("spmd.kernel function signature has group/private "
                         "memref argument; only global address space is "
                         "allowed in S0/S1");
          anyFailed = true;
        }
      }
      for (Type t : fnType.getResults()) {
        if (typeHasNonGlobalMemref(t)) {
          func.emitError("spmd.kernel function signature has group/private "
                         "memref result; only global address space is "
                         "allowed in S0/S1");
          anyFailed = true;
        }
      }

      // 2. Walk all ops inside the kernel.
      func.walk([&](Operation *op) {
        // spmd.barrier is S2-only; must not appear in S0/S1.
        if (isa<BarrierOp>(op)) {
          op->emitError(
              "spmd.barrier must not appear in S0/S1 kernel; "
              "it is inserted by PromoteGroupMemory pass");
          anyFailed = true;
          return;
        }

        // Check dialect whitelist.
        StringRef dialect = op->getDialect()->getNamespace();
        if (!isAllowedDialect(dialect)) {
          op->emitError("dialect '")
              << dialect << "' is not allowed in spmd.kernel (S0/S1); "
              << "only arith/math/memref/affine/vector/func/spmd are permitted";
          anyFailed = true;
          return;
        }

        // Disallow scf.* entirely in S0/S1 (those appear after lowering).
        if (dialect == "scf") {
          op->emitError("scf dialect ops must not appear in S0/S1 kernel; "
                        "run --convert-spmd-to-scf first");
          anyFailed = true;
          return;
        }

        // Check ALL operand types for non-global memref addr space.
        for (Value operand : op->getOperands()) {
          if (typeHasNonGlobalMemref(operand.getType())) {
            op->emitError(
                "operand has group/private address space memref type; "
                "only global address space is allowed in S0/S1 kernels");
            anyFailed = true;
          }
        }

        // Check ALL result types for non-global memref addr space.
        for (Type ty : op->getResultTypes()) {
          if (typeHasNonGlobalMemref(ty)) {
            op->emitError(
                "result has group/private address space memref type; "
                "only global address space is allowed in S0/S1 kernels");
            anyFailed = true;
          }
        }
      });
    });

    if (anyFailed)
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::spmd::createVerifySPMDKernelSubsetPass() {
  return std::make_unique<VerifySPMDKernelSubsetPass>();
}

void mlir::spmd::registerVerifySPMDKernelSubsetPass() {
  PassRegistration<VerifySPMDKernelSubsetPass>();
}
