// ConvertGpuMemorySpacesToNumeric.cpp
//
// Pre-LLVM-lowering cleanup: convert #gpu.address_space<*> attributes on
// memref types to plain integer address spaces so that --finalize-memref-to-llvm
// can lower them without needing the LLVMTypeConverter configured with GPU
// address space mappings.
//
// NVPTX mapping (matches populateCommonGPUTypeAndAttributeConversions):
//   workgroup → 3  (NVVM shared memory)
//   private   → 0  (default / stack)
//   global    → 1  (global / generic)

#include "spmd/Transforms/SPMDPasses.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

/// Convert a GPU address space attribute to the corresponding NVPTX integer
/// address space number, or return -1 if the attribute is not a GPU address
/// space.
static int getIntAddrSpace(Attribute memSpace) {
  auto gpuAS = dyn_cast_or_null<gpu::AddressSpaceAttr>(memSpace);
  if (!gpuAS)
    return -1;
  switch (gpuAS.getValue()) {
  case gpu::AddressSpace::Workgroup:
    return 3;
  case gpu::AddressSpace::Private:
    return 0;
  case gpu::AddressSpace::Global:
    return 1;
  }
  return -1;
}

/// Replace a MemRefType's #gpu.address_space<X> with an integer attr if
/// applicable, otherwise return the type unchanged.
static MemRefType convertMemRefType(MemRefType t) {
  int as = getIntAddrSpace(t.getMemorySpace());
  if (as < 0)
    return t;
  Attribute intAS = IntegerAttr::get(IntegerType::get(t.getContext(), 64),
                                     static_cast<int64_t>(as));
  return MemRefType::get(t.getShape(), t.getElementType(), t.getLayout(),
                         as == 0 ? Attribute{} : intAS);
}

struct ConvertGpuMemorySpacesToNumericPass
    : public PassWrapper<ConvertGpuMemorySpacesToNumericPass,
                         OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ConvertGpuMemorySpacesToNumericPass)

  StringRef getArgument() const override {
    return "spmd-convert-gpu-memref-spaces";
  }
  StringRef getDescription() const override {
    return "Convert #gpu.address_space<X> on memrefs to integer address "
           "spaces for LLVM lowering";
  }

  void runOnOperation() override {
    Operation *root = getOperation();

    // Update block arguments.
    root->walk([](Block *block) {
      for (BlockArgument arg : block->getArguments()) {
        auto m = dyn_cast<MemRefType>(arg.getType());
        if (!m)
          continue;
        MemRefType converted = convertMemRefType(m);
        if (converted != m)
          arg.setType(converted);
      }
    });

    // Update op result types.
    root->walk([](Operation *op) {
      for (OpResult result : op->getResults()) {
        auto m = dyn_cast<MemRefType>(result.getType());
        if (!m)
          continue;
        MemRefType converted = convertMemRefType(m);
        if (converted != m)
          result.setType(converted);
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::spmd::createConvertGpuMemorySpacesToNumericPass() {
  return std::make_unique<ConvertGpuMemorySpacesToNumericPass>();
}

void mlir::spmd::registerConvertGpuMemorySpacesToNumericPass() {
  PassRegistration<ConvertGpuMemorySpacesToNumericPass>();
}
