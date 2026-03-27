//===- spmd-opt.cpp - SPMD dialect optimizer driver -----------------------===//
//
// Tool driver for the spmd MLIR dialect. Registers all spmd ops, attrs, and
// passes, then hands off to MlirOptMain.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "spmd/IR/SPMDDialect.h"
#include "spmd/Transforms/SPMDPasses.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::spmd::registerSPMDPasses();

  mlir::DialectRegistry registry;
  // Register all upstream MLIR dialects (includes LLVM, CF, SCF, Arith, etc.)
  // so that full lowering pipelines (--convert-scf-to-cf,
  // --convert-arith-to-llvm, --finalize-memref-to-llvm, --convert-func-to-llvm,
  // --reconcile-unrealized-casts) work end-to-end within spmd-opt.
  mlir::registerAllDialects(registry);
  // Add the out-of-tree SPMD dialect that is not covered by registerAllDialects.
  registry.insert<mlir::spmd::SPMDDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "SPMD dialect optimizer driver\n",
                        registry));
}
