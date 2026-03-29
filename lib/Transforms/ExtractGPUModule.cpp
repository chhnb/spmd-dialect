// ExtractGPUModule.cpp
//
// Testing utility pass: lifts the body of the first gpu.module in a
// module into the enclosing module so that mlir-translate --mlir-to-llvmir
// can translate the device-side NVVM IR.
//
// Usage (in lit tests only):
//   spmd-opt ... --convert-gpu-to-nvvm --spmd-extract-gpu-module
//   | mlir-translate --mlir-to-llvmir
//   | llc --march=nvptx64 ...
//
// The pass:
//   1. Finds the first gpu.module op at module body level.
//   2. Moves all non-terminator ops from its body to the enclosing module.
//   3. Erases all func.func ops from the enclosing module (host-side IR that
//      mlir-translate cannot lower).
//   4. Erases the now-empty gpu.module wrapper.

#include "spmd/Transforms/SPMDPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct ExtractGPUModulePass
    : public PassWrapper<ExtractGPUModulePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExtractGPUModulePass)

  StringRef getArgument() const override {
    return "spmd-extract-gpu-module";
  }
  StringRef getDescription() const override {
    return "Lift the first gpu.module body into the enclosing module for "
           "device-side mlir-translate (lit testing utility)";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Find the first gpu.module at the top level.
    gpu::GPUModuleOp gpuMod;
    for (auto &op : *module.getBody()) {
      if (auto gm = dyn_cast<gpu::GPUModuleOp>(op)) {
        gpuMod = gm;
        break;
      }
    }
    if (!gpuMod)
      return;

    // Move device ops (llvm.func, llvm.mlir.global, …) from gpu.module body
    // to the enclosing module, placing them just before the gpu.module op.
    SmallVector<Operation *> deviceOps;
    for (auto &op : gpuMod.getBody().front())
      if (!op.hasTrait<OpTrait::IsTerminator>())
        deviceOps.push_back(&op);
    for (Operation *op : deviceOps)
      op->moveBefore(gpuMod);

    // Erase all func.func ops (host-side MLIR; not LLVM-translatable).
    SmallVector<Operation *> toErase;
    for (auto &op : *module.getBody())
      if (isa<func::FuncOp>(op))
        toErase.push_back(&op);
    for (Operation *op : toErase)
      op->erase();

    // Erase the now-empty gpu.module wrapper.
    gpuMod.erase();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::spmd::createExtractGPUModulePass() {
  return std::make_unique<ExtractGPUModulePass>();
}

void mlir::spmd::registerExtractGPUModulePass() {
  PassRegistration<ExtractGPUModulePass>();
}
