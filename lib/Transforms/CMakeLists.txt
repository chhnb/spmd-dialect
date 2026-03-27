add_mlir_library(SPMDTransforms
  VerifySPMDKernelSubset.cpp
  NormalizeSPMD.cpp
  PlanSPMDSchedule.cpp
  MaterializeTilingAndMapping.cpp
  PromoteGroupMemory.cpp

  LINK_LIBS PUBLIC
    SPMDDialect
    SPMDAnalysis
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRAffineDialect
)
