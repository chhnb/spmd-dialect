add_mlir_dialect_library(SPMDDialect
  SPMDDialect.cpp
  SPMDOps.cpp
  SPMDAttrs.cpp

  DEPENDS
    SPMDOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRAffineDialect
    MLIRMathDialect
)
