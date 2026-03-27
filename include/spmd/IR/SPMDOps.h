#pragma once

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "spmd/IR/SPMDAttrs.h"
#include "spmd/IR/SPMDDialect.h"

// Tablegen-generated op declarations
#define GET_OP_CLASSES
#include "spmd/IR/SPMDOps.h.inc"
