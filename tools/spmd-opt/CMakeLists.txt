set(LIBS
  SPMDDialect
  SPMDTransforms
  SPMDAnalysis
  SPMDToSCF
  SPMDToOpenMP
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIROpenMPDialect
  MLIRSCFDialect
  MLIRVectorDialect
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
)

add_llvm_executable(spmd-opt spmd-opt.cpp)
llvm_update_compile_flags(spmd-opt)
target_link_libraries(spmd-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(spmd-opt)
