add_mlir_library(SPMDTransforms
  VerifySPMDKernelSubset.cpp
  NormalizeSPMD.cpp
  PlanSPMDSchedule.cpp
  MaterializeTilingAndMapping.cpp
  PromoteGroupMemory.cpp
  EraseSpmdMemorySpaces.cpp
  ConvertGpuMemorySpacesToNumeric.cpp
  SPMDPassRegistration.cpp

  LINK_LIBS PUBLIC
    SPMDDialect
    SPMDAnalysis
    SPMDToSCF
    SPMDToOpenMP
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRArithDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRAffineDialect
    MLIRSCFDialect
    MLIRGPUDialect
)
