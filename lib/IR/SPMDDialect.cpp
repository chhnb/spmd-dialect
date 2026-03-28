#include "spmd/IR/SPMDDialect.h"
#include "spmd/IR/SPMDAttrs.h"
#include "spmd/IR/SPMDOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::spmd;

// Tablegen-generated dialect definition
#include "spmd/IR/SPMDDialect.cpp.inc"

// Tablegen-generated enum definitions (must precede attr defs)
#include "spmd/IR/SPMDEnums.cpp.inc"

// Tablegen-generated attr storage structs + parse/print implementations.
// Must be defined before initialize() which instantiates registerParametricStorageType.
#define GET_ATTRDEF_CLASSES
#include "spmd/IR/SPMDAttrs.cpp.inc"

void SPMDDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "spmd/IR/SPMDOps.cpp.inc"
  >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "spmd/IR/SPMDAttrs.cpp.inc"
  >();
}
