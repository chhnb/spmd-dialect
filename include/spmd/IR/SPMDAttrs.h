#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"

#include "spmd/IR/SPMDDialect.h"

// Tablegen-generated enum declarations (LevelKind, ScopeKind, etc.)
#include "spmd/IR/SPMDEnums.h.inc"

// Tablegen-generated attr declarations
#define GET_ATTRDEF_CLASSES
#include "spmd/IR/SPMDAttrs.h.inc"
