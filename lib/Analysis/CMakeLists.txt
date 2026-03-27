add_mlir_library(SPMDAnalysis
  AccessSummaryAnalysis.cpp
  PromotionPlanAnalysis.cpp

  LINK_LIBS PUBLIC
    SPMDDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRAffineAnalysis
)
