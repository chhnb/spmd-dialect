// SPMDAttrs.td — Attribute definitions for the spmd dialect

#ifndef SPMD_ATTRS
#define SPMD_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "spmd/IR/SPMDDialect.td"

class SPMD_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<SPMD_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// LevelAttr — abstract execution level
//===----------------------------------------------------------------------===//

def SPMD_LevelAttr : SPMD_Attr<"Level", "level"> {
  let summary = "Abstract execution level for SPMD mapping";
  let description = [{
    Represents one of the abstract execution levels:
      seq    — sequential (no parallelism)
      grid   — outermost parallel domain
      group  — cooperative group sharing group memory
      lane   — individual SPMD instance within a group
      vector — lockstep vector execution within a lane
  }];

  let parameters = (ins
    "LevelKind":$kind
  );

  let extraClassDeclaration = [{
    enum class LevelKind { Seq, Grid, Group, Lane, Vector };
  }];
}

//===----------------------------------------------------------------------===//
// ScopeAttr — synchronization scope
//===----------------------------------------------------------------------===//

def SPMD_ScopeAttr : SPMD_Attr<"Scope", "scope"> {
  let summary = "Synchronization scope for spmd.barrier";
  let description = [{
    MVP only supports 'group' scope:
    all active lanes in the same group synchronize at the barrier.
  }];

  let parameters = (ins
    "ScopeKind":$kind
  );

  let extraClassDeclaration = [{
    enum class ScopeKind { Group };
  }];
}

//===----------------------------------------------------------------------===//
// ReductionKindAttr — reduction combiner
//===----------------------------------------------------------------------===//

def SPMD_ReductionKindAttr : SPMD_Attr<"ReductionKind", "reduction_kind"> {
  let summary = "Reduction combiner kind for spmd.reduce";
  let description = [{
    Supported combiners: add, mul, max, min, and, or, xor.
    Note: floating-point add/mul imply reassociation; only valid under
    fast-math semantics. Backends should fall back to sequential order
    when fast-math is not enabled.
  }];

  let parameters = (ins
    "ReductionKindEnum":$kind
  );

  let extraClassDeclaration = [{
    enum class ReductionKindEnum { Add, Mul, Max, Min, And, Or, Xor };
  }];
}

//===----------------------------------------------------------------------===//
// AddressSpaceAttr — abstract memory space for memref
//===----------------------------------------------------------------------===//

def SPMD_AddressSpaceAttr : SPMD_Attr<"AddressSpace", "addr_space"> {
  let summary = "Abstract memory space for SPMD IR";
  let description = [{
    Used as the memory-space parameter of memref types:
      global  — kernel-wide visible storage (input/output buffers)
      group   — shared within a cooperative group
      private — per-lane private storage (registers / stack-local)

    Only 'global' should appear in S0/S1.
    'group' and 'private' are materialized by PromoteGroupMemory and
    VectorizeOrPrivatizeSPMD passes in S2.
  }];

  let parameters = (ins
    "AddressSpaceKind":$kind
  );

  let extraClassDeclaration = [{
    enum class AddressSpaceKind { Global, Group, Private };
  }];
}

//===----------------------------------------------------------------------===//
// MemoryPolicyAttr — schedule hint for memory placement
//===----------------------------------------------------------------------===//

def SPMD_MemoryPolicyAttr : SPMD_Attr<"MemoryPolicy", "memory_policy"> {
  let summary = "Memory placement policy hint on spmd.forall";
  let description = [{
    Advises the PromoteGroupMemory pass on promotion strategy:
      none           — no preference, let analysis decide
      prefer_group   — try to promote reused tiles to group memory
      prefer_private — try to scalarize / register-tile
      no_promotion   — disable promotion for this forall
  }];

  let parameters = (ins
    "MemoryPolicyKind":$kind
  );

  let extraClassDeclaration = [{
    enum class MemoryPolicyKind { None, PreferGroup, PreferPrivate, NoPromotion };
  }];
}

#endif // SPMD_ATTRS
