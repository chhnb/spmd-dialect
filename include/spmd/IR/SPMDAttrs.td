// SPMDAttrs.td — Attribute definitions for the spmd dialect

#ifndef SPMD_ATTRS
#define SPMD_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "spmd/IR/SPMDDialect.td"

class SPMD_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<SPMD_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// LevelAttr — abstract execution level
//===----------------------------------------------------------------------===//

def SPMD_LevelSeq    : I32EnumAttrCase<"Seq",    0, "seq">;
def SPMD_LevelGrid   : I32EnumAttrCase<"Grid",   1, "grid">;
def SPMD_LevelGroup  : I32EnumAttrCase<"Group",  2, "group">;
def SPMD_LevelLane   : I32EnumAttrCase<"Lane",   3, "lane">;
def SPMD_LevelVector : I32EnumAttrCase<"Vector", 4, "vector">;

def SPMD_LevelKind : I32EnumAttr<"LevelKind",
    "Abstract execution level for SPMD mapping",
    [SPMD_LevelSeq, SPMD_LevelGrid, SPMD_LevelGroup,
     SPMD_LevelLane, SPMD_LevelVector]> {
  let cppNamespace = "::mlir::spmd";
}

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
  let parameters = (ins "LevelKind":$value);
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// ScopeAttr — synchronization scope
//===----------------------------------------------------------------------===//

def SPMD_ScopeGroup : I32EnumAttrCase<"Group", 0, "group">;

def SPMD_ScopeKind : I32EnumAttr<"ScopeKind",
    "Synchronization scope",
    [SPMD_ScopeGroup]> {
  let cppNamespace = "::mlir::spmd";
}

def SPMD_ScopeAttr : SPMD_Attr<"Scope", "scope"> {
  let summary = "Synchronization scope for spmd.barrier";
  let description = [{
    MVP only supports 'group' scope:
    all active lanes in the same group synchronize at the barrier.
  }];
  let parameters = (ins "ScopeKind":$value);
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// ReductionKindAttr — reduction combiner
//===----------------------------------------------------------------------===//

def SPMD_ReductionAdd : I32EnumAttrCase<"Add", 0, "add">;
def SPMD_ReductionMul : I32EnumAttrCase<"Mul", 1, "mul">;
def SPMD_ReductionMax : I32EnumAttrCase<"Max", 2, "max">;
def SPMD_ReductionMin : I32EnumAttrCase<"Min", 3, "min">;
def SPMD_ReductionAnd : I32EnumAttrCase<"And", 4, "and">;
def SPMD_ReductionOr  : I32EnumAttrCase<"Or",  5, "or">;
def SPMD_ReductionXor : I32EnumAttrCase<"Xor", 6, "xor">;

def SPMD_ReductionKind : I32EnumAttr<"ReductionKind",
    "Reduction combiner kind",
    [SPMD_ReductionAdd, SPMD_ReductionMul, SPMD_ReductionMax,
     SPMD_ReductionMin, SPMD_ReductionAnd, SPMD_ReductionOr,
     SPMD_ReductionXor]> {
  let cppNamespace = "::mlir::spmd";
}

def SPMD_ReductionKindAttr : SPMD_Attr<"ReductionKind", "reduction_kind"> {
  let summary = "Reduction combiner kind for spmd.reduce";
  let description = [{
    Supported combiners: add, mul, max, min, and, or, xor.
    Note: floating-point add/mul imply reassociation; only valid under
    fast-math semantics. Backends should fall back to sequential order
    when fast-math is not enabled.
  }];
  let parameters = (ins "ReductionKind":$value);
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// AddressSpaceAttr — abstract memory space for memref
//===----------------------------------------------------------------------===//

def SPMD_AddrSpaceGlobal  : I32EnumAttrCase<"Global",  0, "global">;
def SPMD_AddrSpaceGroup   : I32EnumAttrCase<"Group",   1, "group">;
def SPMD_AddrSpacePrivate : I32EnumAttrCase<"Private", 2, "private">;

def SPMD_AddressSpaceKind : I32EnumAttr<"AddressSpaceKind",
    "Abstract memory space",
    [SPMD_AddrSpaceGlobal, SPMD_AddrSpaceGroup, SPMD_AddrSpacePrivate]> {
  let cppNamespace = "::mlir::spmd";
}

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
  let parameters = (ins "AddressSpaceKind":$value);
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// MemoryPolicyAttr — schedule hint for memory placement
//===----------------------------------------------------------------------===//

def SPMD_PolicyNone         : I32EnumAttrCase<"None",         0, "none">;
def SPMD_PolicyPreferGroup  : I32EnumAttrCase<"PreferGroup",  1, "prefer_group">;
def SPMD_PolicyPreferPrivate: I32EnumAttrCase<"PreferPrivate",2, "prefer_private">;
def SPMD_PolicyNoPromotion  : I32EnumAttrCase<"NoPromotion",  3, "no_promotion">;

def SPMD_MemoryPolicyKind : I32EnumAttr<"MemoryPolicyKind",
    "Memory placement policy",
    [SPMD_PolicyNone, SPMD_PolicyPreferGroup,
     SPMD_PolicyPreferPrivate, SPMD_PolicyNoPromotion]> {
  let cppNamespace = "::mlir::spmd";
}

def SPMD_MemoryPolicyAttr : SPMD_Attr<"MemoryPolicy", "memory_policy"> {
  let summary = "Memory placement policy hint on spmd.forall";
  let description = [{
    Advises the PromoteGroupMemory pass on promotion strategy:
      none           — no preference, let analysis decide
      prefer_group   — try to promote reused tiles to group memory
      prefer_private — try to scalarize / register-tile
      no_promotion   — disable promotion for this forall
  }];
  let parameters = (ins "MemoryPolicyKind":$value);
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// KernelAttr — marks a func.func as an SPMD kernel entry point
//===----------------------------------------------------------------------===//

def SPMD_KernelAttr : SPMD_Attr<"Kernel", "kernel"> {
  let summary = "Marks a func.func as an SPMD kernel entry point";
  let description = [{
    Unit attribute: presence means the function is an SPMD kernel.
    Used by VerifySPMDKernelSubset to enforce S0/S1 legality constraints.
  }];
  // No parameters — pure unit attribute semantics
}

#endif // SPMD_ATTRS
