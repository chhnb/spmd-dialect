// EraseSpmdMemorySpaces.cpp
//
// Pre-LLVM-lowering cleanup: strip #spmd.addr_space<X> from all memref types.
// After all SPMD semantics have been lowered (promote, convert-to-scf, etc.),
// the address-space decoration is no longer meaningful on CPU and
// --finalize-memref-to-llvm cannot convert non-integer address spaces.
//
// This pass walks every SSA value (block arguments and op results) whose type
// is a MemRefType with a SPMD AddressSpaceAttr and replaces it with the
// equivalent memref in the default (0) address space via setType().
// Because all SPMD-level verification has already run, the mutation is safe.

#include "spmd/IR/SPMDDialect.h"
#include "spmd/IR/SPMDOps.h"
#include "spmd/Transforms/SPMDPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::spmd;

namespace {

/// Returns the default-addr-space equivalent of `t` if it has a SPMD addr
/// space, otherwise returns `t` unchanged.
static MemRefType stripSpmdAddrSpace(MemRefType t) {
  if (!t.getMemorySpace())
    return t;
  if (!isa<AddressSpaceAttr>(t.getMemorySpace()))
    return t;
  return MemRefType::get(t.getShape(), t.getElementType(), t.getLayout(),
                         Attribute{});
}

struct EraseSpmdMemorySpacesPass
    : public PassWrapper<EraseSpmdMemorySpacesPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EraseSpmdMemorySpacesPass)

  StringRef getArgument() const override {
    return "spmd-erase-memory-spaces";
  }
  StringRef getDescription() const override {
    return "Strip SPMD addr-space attrs from memrefs before LLVM lowering";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Update block-argument types throughout all regions.
    func.walk([](Block *block) {
      for (BlockArgument arg : block->getArguments()) {
        auto m = dyn_cast<MemRefType>(arg.getType());
        if (!m) continue;
        MemRefType stripped = stripSpmdAddrSpace(m);
        if (stripped != m)
          arg.setType(stripped);
      }
    });

    // Update op-result types.
    func.walk([](Operation *op) {
      for (OpResult result : op->getResults()) {
        auto m = dyn_cast<MemRefType>(result.getType());
        if (!m) continue;
        MemRefType stripped = stripSpmdAddrSpace(m);
        if (stripped != m)
          result.setType(stripped);
      }
    });

    // Update the function's own type signature.
    SmallVector<Type> newInputs, newResults;
    bool changed = false;
    for (Type t : func.getFunctionType().getInputs()) {
      auto m = dyn_cast<MemRefType>(t);
      Type updated = (m ? MemRefType(stripSpmdAddrSpace(m)) : t);
      if (updated != t) changed = true;
      newInputs.push_back(updated);
    }
    for (Type t : func.getFunctionType().getResults()) {
      auto m = dyn_cast<MemRefType>(t);
      Type updated = (m ? MemRefType(stripSpmdAddrSpace(m)) : t);
      if (updated != t) changed = true;
      newResults.push_back(updated);
    }
    if (changed)
      func.setFunctionType(
          FunctionType::get(func.getContext(), newInputs, newResults));
  }
};

} // namespace

std::unique_ptr<Pass> mlir::spmd::createEraseSpmdMemorySpacesPass() {
  return std::make_unique<EraseSpmdMemorySpacesPass>();
}

void mlir::spmd::registerEraseSpmdMemorySpacesPass() {
  PassRegistration<EraseSpmdMemorySpacesPass>();
}
