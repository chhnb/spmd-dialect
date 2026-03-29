// SPMDPassRegistration.cpp
// Provides the combined registerSPMDPasses() entry point.

#include "spmd/Transforms/SPMDPasses.h"

void mlir::spmd::registerSPMDPasses() {
  registerVerifySPMDKernelSubsetPass();
  registerVerifySPMDPromotionInvariantPass();
  registerVerifySPMDGPUReadyPass();
  registerNormalizeSPMDPass();
  registerPlanSPMDSchedulePass();
  registerMaterializeTilingAndMappingPass();
  registerPromoteGroupMemoryPass();
  registerEraseSpmdMemorySpacesPass();
  registerExtractGPUModulePass();
  registerSPMDToSCFPass();
  registerSPMDToOpenMPPass();
  registerConvertSPMDToGPUPass();
}
