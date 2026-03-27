#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"

#include "spmd/IR/SPMDDialect.h"

// Include tablegen-generated attr declarations
#define GET_ATTRDEF_CLASSES
#include "spmd/IR/SPMDAttrs.h.inc"
