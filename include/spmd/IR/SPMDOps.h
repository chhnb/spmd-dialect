#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "spmd/IR/SPMDDialect.h"
#include "spmd/IR/SPMDAttrs.h"

// Include tablegen-generated op declarations
#define GET_OP_CLASSES
#include "spmd/IR/SPMDOps.h.inc"
