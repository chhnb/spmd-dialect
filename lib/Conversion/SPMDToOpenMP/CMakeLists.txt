add_mlir_conversion_library(SPMDToOpenMP
  SPMDToOpenMP.cpp

  LINK_LIBS PUBLIC
    SPMDToSCF
    MLIROpenMPDialect
    MLIRSCFDialect
    MLIRIR
    MLIRPass
)
