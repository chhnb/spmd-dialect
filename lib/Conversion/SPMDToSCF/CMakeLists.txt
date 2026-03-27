add_mlir_conversion_library(SPMDToSCF
  SPMDToSCF.cpp

  LINK_LIBS PUBLIC
    SPMDDialect
    MLIRSCFDialect
    MLIRArithDialect
    MLIRIR
    MLIRPass
)
