#pragma once
//===- SPMDPasses.h - SPMD dialect pass declarations ----------------------===//
//
// Declarations for all passes in the spmd dialect: transforms, analyses, and
// conversions. Include this header in spmd-opt and any tool that needs to
// register the full pass pipeline.
//
//===----------------------------------------------------------------------===//

#include <memory>

namespace mlir {
class Pass;

namespace spmd {

//===----------------------------------------------------------------------===//
// Verification / IR legality
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createVerifySPMDKernelSubsetPass();
void registerVerifySPMDKernelSubsetPass();

//===----------------------------------------------------------------------===//
// Transform passes (S0 → S1 → S2)
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNormalizeSPMDPass();
void registerNormalizeSPMDPass();

std::unique_ptr<Pass> createPlanSPMDSchedulePass();
void registerPlanSPMDSchedulePass();

std::unique_ptr<Pass> createMaterializeTilingAndMappingPass();
void registerMaterializeTilingAndMappingPass();

std::unique_ptr<Pass> createPromoteGroupMemoryPass();
void registerPromoteGroupMemoryPass();

std::unique_ptr<Pass> createEraseSpmdMemorySpacesPass();
void registerEraseSpmdMemorySpacesPass();

std::unique_ptr<Pass> createConvertGpuMemorySpacesToNumericPass();
void registerConvertGpuMemorySpacesToNumericPass();

//===----------------------------------------------------------------------===//
// Conversion passes (S2 → target dialect)
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createSPMDToSCFPass();
void registerSPMDToSCFPass();

std::unique_ptr<Pass> createSPMDToOpenMPPass();
void registerSPMDToOpenMPPass();

std::unique_ptr<Pass> createSPMDToGPUPass();
void registerConvertSPMDToGPUPass();

//===----------------------------------------------------------------------===//
// Combined registration
//===----------------------------------------------------------------------===//

/// Register all SPMD dialect passes with the global pass registry.
void registerSPMDPasses();

} // namespace spmd
} // namespace mlir
