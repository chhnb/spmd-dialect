llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(SPMD_TEST_DEPENDS
  FileCheck
  spmd-opt
)

add_lit_testsuite(check-spmd "Running the SPMD dialect regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${SPMD_TEST_DEPENDS}
)
set_target_properties(check-spmd PROPERTIES FOLDER "Tests")
