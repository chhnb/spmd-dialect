//===- spmd-opt.cpp - SPMD dialect optimizer driver -----------------------===//
//
// Tool driver for the spmd MLIR dialect. Registers all spmd ops, attrs, and
// passes, then hands off to MlirOptMain.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "spmd/IR/SPMDDialect.h"
// TODO: uncomment as passes are registered:
// #include "spmd/Transforms/SPMDPasses.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  // TODO: mlir::spmd::registerPasses();

  mlir::DialectRegistry registry;
  // Register spmd dialect and all dialects it depends on
  registry.insert<mlir::spmd::SPMDDialect,
                  mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect,
                  mlir::func::FuncDialect,
                  mlir::math::MathDialect,
                  mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect,
                  mlir::vector::VectorDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "SPMD dialect optimizer driver\n",
                        registry));
}
