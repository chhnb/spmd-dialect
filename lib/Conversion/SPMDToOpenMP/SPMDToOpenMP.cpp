// SPMDToOpenMP.cpp
//
// Lowers group-level spmd.forall to OpenMP parallel for.
// Runs after SPMDToSCF lowered lane-level forall to scf.for.
//
// Mapping:
//   group-level spmd.forall -> omp.parallel + omp.wsloop (or omp.parallel_for)
//   lane-level (already scf.for) -> remains scf.for inside parallel region
//   group memory alloc -> threadprivate or stack-local in parallel region

#include "spmd/IR/SPMDOps.h"

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::spmd;

namespace {
struct SPMDToOpenMPPass
    : public PassWrapper<SPMDToOpenMPPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SPMDToOpenMPPass)

  StringRef getArgument() const override { return "convert-spmd-to-openmp"; }
  StringRef getDescription() const override {
    return "Lower group-level spmd.forall to OpenMP parallel for";
  }

  void runOnOperation() override {
    // TODO: implement Phase 2
  }
};
} // namespace

std::unique_ptr<Pass> mlir::spmd::createSPMDToOpenMPPass() {
  return std::make_unique<SPMDToOpenMPPass>();
}
