// SPMDPassRegistration.cpp
// Provides the combined registerSPMDPasses() entry point.

#include "spmd/Transforms/SPMDPasses.h"

void mlir::spmd::registerSPMDPasses() {
  registerVerifySPMDKernelSubsetPass();
  registerNormalizeSPMDPass();
  registerPlanSPMDSchedulePass();
  registerMaterializeTilingAndMappingPass();
  registerPromoteGroupMemoryPass();
  registerEraseSpmdMemorySpacesPass();
  registerSPMDToSCFPass();
  registerSPMDToOpenMPPass();
}
