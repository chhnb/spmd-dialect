// SPMDDialect.td — Dialect top-level definition

#ifndef SPMD_DIALECT
#define SPMD_DIALECT

include "mlir/IR/BuiltinDialect.td"

def SPMD_Dialect : Dialect {
  let name = "spmd";
  let summary = "Structured SPMD IR for regular kernels";
  let description = [{
    The `spmd` dialect is a structured intermediate representation for
    regular SPMD kernels. It preserves iteration-space semantics, supports
    delayed mapping to abstract execution hierarchies (grid/group/lane/vector),
    and exposes analyzable memory behavior for locality and memory-hierarchy
    optimization.

    Design principles:
    - spmd.forall is the semantic center: logical parallel iteration domain,
      not thread/block ids.
    - Delayed mapping: parallel semantics separate from execution semantics.
    - Backend-agnostic memory spaces: global / group / private.
    - Minimal dialect surface: reuse func, arith, math, memref, affine, vector.

    Three refinement phases:
      S0 (Semantic)     — pure iteration-space semantics, no mapping attrs
      S1 (Scheduled)    — schedule hints attached as attrs on spmd.forall
      S2 (Materialized) — explicit tiling, mapping, group/private memrefs,
                          barriers; still backend-agnostic

    Program subset:
      elementwise, stencil, structured reduction, tiled kernels,
      affine/quasi-affine access patterns, bounded structured control flow.
  }];

  let cppNamespace = "::mlir::spmd";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // SPMD_DIALECT
