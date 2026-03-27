add_mlir_conversion_library(SPMDToGPU
  SPMDToGPU.cpp

  LINK_LIBS PUBLIC
    SPMDDialect
    MLIRGPUDialect
    MLIRArithDialect
    MLIRIR
    MLIRPass
)
