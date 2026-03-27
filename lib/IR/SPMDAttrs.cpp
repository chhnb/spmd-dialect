#include "spmd/IR/SPMDAttrs.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::spmd;

// Include tablegen-generated attr definitions
#define GET_ATTRDEF_CLASSES
#include "spmd/IR/SPMDAttrs.cpp.inc"
